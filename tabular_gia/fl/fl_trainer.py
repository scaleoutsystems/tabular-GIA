import logging
import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from fl.metrics.fl_metrics import eval, progress_write, round_bar
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep
from leakpro.utils.seed import capture_rng_state


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FLTrainResult:
    executed_rounds: int
    exp_min: float
    exp_avg: float
    exp_max: float
    fl_rows: list[dict]


@dataclass(frozen=True)
class FLCallbacks:
    attack_init_fn: Callable[[int], None] | None
    attack_fn: Callable[..., None] | None


class FLTrainer:
    def __init__(
        self,
        fl_cfg: FedAvgConfig | FedSGDConfig,
        batch_size: int,
        client_dataloaders: list[DataLoader],
        val_loader: DataLoader,
        test_loader: DataLoader,
        callbacks: FLCallbacks,
    ) -> None:
        self.fl_cfg = fl_cfg
        self.batch_size = batch_size
        self.client_dataloaders = client_dataloaders
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.callbacks = callbacks
        self.model_wrapper = None
        self.model = None
        self.criterion = None
        self.task = None
        self.best_val_loss = float("inf")
        self.best_round = 0
        self.best_state_dict: dict | None = None
        self.fl_rows: list[dict] = []

    def fit(self, model_wrapper) -> FLTrainResult:
        raise NotImplementedError

    def _set_model(self, model_wrapper) -> None:
        self.model_wrapper = model_wrapper
        self.model = model_wrapper
        self.criterion = model_wrapper.criterion
        self.task = model_wrapper.task

    def _eval(self, loaders: list[DataLoader]) -> dict:
        return eval(loaders, self.model, self.criterion, self.task)

    def _save_metrics(
        self,
        phase: str,
        round_idx: int,
        exp_min: float,
        exp_avg: float,
        exp_max: float,
        metrics_by_stage: dict[str, dict | None],
    ) -> None:
        row = {
            "phase": phase,
            "round": int(round_idx),
            "exp_min": float(exp_min),
            "exp_avg": float(exp_avg),
            "exp_max": float(exp_max),
        }
        for split_name, stats in metrics_by_stage.items():
            if stats is None:
                continue
            for metric_key, metric_value in stats.items():
                row[f"{split_name}_{metric_key}"] = metric_value
        self.fl_rows.append(row)

    def _log(self, stage: str, executed_rounds: int, train_stats: dict | None, stage_stats: dict) -> None:
        if stage not in {"val", "test"}:
            raise ValueError(f"Unknown stage '{stage}'")

        if self.task == "binary":
            if stage == "val":
                progress_write(
                    "Rounds %d - train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f"
                    % (
                        executed_rounds,
                        train_stats["loss"],
                        train_stats["acc"],
                        stage_stats["loss"],
                        stage_stats["acc"],
                    )
                )
                return
            progress_write("Test - loss=%.4f acc=%.4f" % (stage_stats["loss"], stage_stats["acc"]))
            return

        if self.task == "multiclass":
            if stage == "val":
                progress_write(
                    (
                        "Rounds %d - train_loss=%.4f train_acc=%.4f train_f1_macro=%.4f "
                        "val_loss=%.4f val_acc=%.4f val_f1_macro=%.4f"
                    )
                    % (
                        executed_rounds,
                        train_stats["loss"],
                        train_stats["acc"],
                        train_stats["f1_macro"],
                        stage_stats["loss"],
                        stage_stats["acc"],
                        stage_stats["f1_macro"],
                    )
                )
                return
            progress_write(
                "Test - loss=%.4f acc=%.4f f1_macro=%.4f"
                % (
                    stage_stats["loss"],
                    stage_stats["acc"],
                    stage_stats["f1_macro"],
                )
            )
            return

        if stage == "val":
            progress_write(
                (
                    "Rounds %d - train_loss=%.4f train_mse=%.4f train_mae=%.4f train_r2=%.4f "
                    "val_loss=%.4f val_mse=%.4f val_mae=%.4f val_r2=%.4f"
                )
                % (
                    executed_rounds,
                    train_stats["loss"],
                    train_stats["mse"],
                    train_stats["mae"],
                    train_stats["r2"],
                    stage_stats["loss"],
                    stage_stats["mse"],
                    stage_stats["mae"],
                    stage_stats["r2"],
                )
            )
            return
        progress_write(
            "Test - loss=%.4f mse=%.4f mae=%.4f r2=%.4f"
            % (
                stage_stats["loss"],
                stage_stats["mse"],
                stage_stats["mae"],
                stage_stats["r2"],
            )
        )

    def _client_effective_sizes(self, batch_size: int) -> list[int]:
        sizes: list[int] = []
        for loader in self.client_dataloaders:
            local_batch_size = int(loader.batch_size) if loader.batch_size is not None else int(batch_size)
            sizes.append(int(len(loader) * local_batch_size))
        return sizes

    def _exposure_stats(self, examples_seen: list[int], client_n_eff: list[int]) -> tuple[list[float], float, float, float]:
        if not client_n_eff:
            return [], 0.0, 0.0, 0.0
        current_exposures = [seen / max(1, n_eff) for seen, n_eff in zip(examples_seen, client_n_eff)]
        exp_min = min(current_exposures) if current_exposures else 0.0
        exp_avg = (sum(current_exposures) / len(current_exposures)) if current_exposures else 0.0
        exp_max = max(current_exposures) if current_exposures else 0.0
        return current_exposures, float(exp_min), float(exp_avg), float(exp_max)

    def _next_val_checkpoint(
        self,
        exp_min_prev: float,
        next_val_exposure: float,
        exp_min_curr: float,
    ) -> tuple[bool, float]:
        crossed = False
        while exp_min_prev < next_val_exposure <= exp_min_curr:
            crossed = True
            next_val_exposure += 1.0
        return crossed, float(next_val_exposure)

    def _finalize(
        self,
        executed_rounds: int,
        examples_seen: list[int],
        client_n_eff: list[int],
        protocol_label: str,
    ) -> FLTrainResult:
        _, exp_min, exp_avg, exp_max = self._exposure_stats(examples_seen, client_n_eff)
        if client_n_eff:
            logger.info("Final exposure: min=%.6f mean=%.6f max=%.6f", exp_min, exp_avg, exp_max)
        else:
            exp_min = float("nan")
            exp_avg = float("nan")
            exp_max = float("nan")
        if self.best_state_dict is not None:
            self.model_wrapper.restore_state(self.best_state_dict)
            logger.info(
                "Loaded best validation checkpoint: round=%d val_loss=%.4f",
                self.best_round,
                self.best_val_loss,
            )
        test_stats = self._eval([self.test_loader])
        tested_round = int(self.best_round) if self.best_state_dict is not None else int(executed_rounds)
        self._save_metrics(
            phase="final_test",
            round_idx=tested_round,
            exp_min=exp_min,
            exp_avg=exp_avg,
            exp_max=exp_max,
            metrics_by_stage={"test": test_stats},
        )
        self._log("test", executed_rounds, None, test_stats)
        logger.info("%s training completed.", protocol_label)
        return FLTrainResult(
            executed_rounds=int(executed_rounds),
            exp_min=float(exp_min),
            exp_avg=float(exp_avg),
            exp_max=float(exp_max),
            fl_rows=list(self.fl_rows),
        )


class FedAvgTrainer(FLTrainer):

    def _fedavg_train(self, global_model, client_updates):
        if not client_updates:
            return

        params = list(global_model.parameters())
        num_params = len(params)
        for deltas, _ in client_updates:
            if len(deltas) != num_params:
                raise ValueError("Client update does not match model parameters.")

        total_weight = sum(weight for _, weight in client_updates) or 1
        with torch.no_grad():
            for idx, param in enumerate(params):
                if not param.requires_grad:
                    continue
                agg = torch.zeros_like(param)
                for deltas, weight in client_updates:
                    delta = deltas[idx]
                    if delta is None:
                        continue
                    agg.add_(delta.detach().to(param.device, dtype=param.dtype), alpha=(weight / total_weight))
                param.add_(agg)

    def fit(self, model_wrapper) -> FLTrainResult:
        self._set_model(model_wrapper)
        cfg = self.fl_cfg
        local_steps_raw = cfg.local_steps
        local_steps_all = isinstance(local_steps_raw, str) and local_steps_raw.strip().lower() == "all"
        local_steps = None if local_steps_all else max(1, local_steps_raw)
        local_epochs = max(1, cfg.local_epochs)
        batch_size = self.batch_size
        lr = cfg.lr
        optimizer_name = cfg.optimizer
        num_clients = len(self.client_dataloaders)

        client_n_eff = self._client_effective_sizes(batch_size)
        n_max_eff = max(client_n_eff) if client_n_eff else 1
        min_exposure = cfg.min_exposure
        if min_exposure <= 0:
            raise ValueError(f"min_exposure must be > 0, got {min_exposure}")
        if local_steps_all:
            num_rounds = max(1, math.ceil(min_exposure / local_epochs))
            samples_per_round = "all_local_batches"
        else:
            samples_per_round = max(1, local_steps * batch_size * local_epochs)
            num_rounds = max(1, math.ceil((min_exposure * n_max_eff) / samples_per_round))

        if self.callbacks.attack_init_fn is not None:
            self.callbacks.attack_init_fn(num_rounds)

        logger.info(
            "Round budget: rounds=%d min_exposure=%.6f n_max_eff=%d samples_per_round=%s",
            num_rounds,
            min_exposure,
            n_max_eff,
            samples_per_round,
        )

        model = self.model
        criterion = self.criterion
        model.train()
        client_iters = [iter(dl) for dl in self.client_dataloaders] if not local_steps_all else []
        examples_seen = [0 for _ in range(num_clients)]
        next_val_exposure = 1.0 if self.val_loader is not None else float("inf")
        executed_rounds = 0
        exp_min_prev = 0.0

        def next_batch(client_idx: int):
            try:
                return next(client_iters[client_idx])
            except StopIteration:
                client_iters[client_idx] = iter(self.client_dataloaders[client_idx])
                return next(client_iters[client_idx])

        with round_bar(num_rounds, "Communication Rounds") as bar:
            for round_idx in range(1, num_rounds + 1):
                executed_rounds = round_idx
                model.train()
                bar.update(1)
                active_list = list(range(num_clients))
                random.shuffle(active_list)

                client_updates = []
                attack_payloads = [] if self.callbacks.attack_fn is not None else None
                for client_idx in active_list:
                    if local_steps_all:
                        batches = list(iter(self.client_dataloaders[client_idx]))
                    else:
                        batches = []
                        for _ in range(local_steps):
                            batches.append(next_batch(client_idx))

                    inputs_cat = torch.cat([b[0] for b in batches], dim=0)
                    labels_cat = torch.cat([b[1] for b in batches], dim=0)
                    batch_ds = TensorDataset(inputs_cat, labels_cat)
                    batch_loader = DataLoader(batch_ds, batch_size=batch_size, shuffle=False)
                    samples_seen = len(labels_cat) * local_epochs
                    examples_seen[client_idx] += samples_seen

                    optimizer = MetaAdam(lr=lr) if optimizer_name == "MetaAdam" else MetaSGD(lr=lr)
                    rng_pre = capture_rng_state()
                    deltas = train(model, batch_loader, optimizer, criterion, epochs=local_epochs)
                    rng_post = capture_rng_state()
                    client_updates.append((deltas, samples_seen))
                    if attack_payloads is not None:
                        attack_payloads.append((batch_loader, client_idx, deltas, rng_pre, rng_post))

                current_exposures, exp_min_curr, exp_avg_curr, exp_max_curr = self._exposure_stats(examples_seen, client_n_eff)
                crossed_val_checkpoint, next_val_exposure = self._next_val_checkpoint(
                    float(exp_min_prev),
                    float(next_val_exposure),
                    float(exp_min_curr),
                )
                if self.callbacks.attack_fn is not None and attack_payloads is not None:
                    self.callbacks.attack_fn(
                        model=self.model,
                        round_idx=int(round_idx),
                        attack_payloads=attack_payloads,
                        exp_min_prev=float(exp_min_prev),
                        exp_min_curr=float(exp_min_curr),
                        current_exposures=current_exposures,
                        exp_min=float(exp_min_curr),
                        exp_avg=float(exp_avg_curr),
                        exp_max=float(exp_max_curr),
                    )
                exp_min_prev = exp_min_curr

                self._fedavg_train(model, client_updates)

                if crossed_val_checkpoint:
                    train_stats = self._eval(self.client_dataloaders)
                    val_stats = self._eval([self.val_loader]) if self.val_loader is not None else None
                    if val_stats is not None:
                        val_loss = float(val_stats["loss"])
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_round = int(round_idx)
                            self.best_state_dict = self.model_wrapper.snapshot_state()
                    self._save_metrics(
                        phase="checkpoint",
                        round_idx=int(round_idx),
                        exp_min=float(exp_min_curr),
                        exp_avg=float(exp_avg_curr),
                        exp_max=float(exp_max_curr),
                        metrics_by_stage={"train": train_stats, "val": val_stats},
                    )
                    self._log("val", executed_rounds, train_stats, val_stats)

        return self._finalize(
            executed_rounds=executed_rounds,
            examples_seen=examples_seen,
            client_n_eff=client_n_eff,
            protocol_label="FedAvg",
        )


class FedSGDTrainer(FLTrainer):

    def _fedsgd_train(self, global_model, client_gradients, lr):
        if not client_gradients:
            return

        params = [p for p in global_model.parameters()]
        num_params = len(params)
        for grads in client_gradients:
            if len(grads) != num_params:
                raise ValueError("Client gradients do not match model parameters.")

        with torch.no_grad():
            for idx, param in enumerate(params):
                if not param.requires_grad:
                    continue
                grads = [g for g in (client[idx] for client in client_gradients) if g is not None]
                if not grads:
                    continue
                agg = torch.zeros_like(param)
                for grad in grads:
                    agg.add_(grad.detach().to(param.device, dtype=param.dtype))
                agg.div_(len(grads))
                param.add_(agg, alpha=-lr)

    def fit(self, model_wrapper) -> FLTrainResult:
        self._set_model(model_wrapper)
        cfg = self.fl_cfg
        local_steps = max(1, cfg.local_steps)
        local_epochs = max(1, cfg.local_epochs)
        batch_size = self.batch_size
        lr = cfg.lr
        num_clients = len(self.client_dataloaders)

        client_n_eff = self._client_effective_sizes(batch_size)
        n_max_eff = max(client_n_eff) if client_n_eff else 1
        samples_per_round = max(1, local_steps * batch_size * local_epochs)
        min_exposure = cfg.min_exposure
        if min_exposure <= 0:
            raise ValueError(f"min_exposure must be > 0, got {min_exposure}")
        num_rounds = max(1, math.ceil((min_exposure * n_max_eff) / samples_per_round))

        if self.callbacks.attack_init_fn is not None:
            self.callbacks.attack_init_fn(num_rounds)

        logger.info(
            "Round budget: rounds=%d min_exposure=%.6f n_max_eff=%d samples_per_round=%d",
            num_rounds,
            min_exposure,
            n_max_eff,
            samples_per_round,
        )

        model = self.model
        criterion = self.criterion
        model.train()
        client_iters = [iter(dl) for dl in self.client_dataloaders]
        examples_seen = [0 for _ in range(num_clients)]
        next_val_exposure = 1.0 if self.val_loader is not None else float("inf")
        executed_rounds = 0
        exp_min_prev = 0.0

        def next_batch(client_idx: int):
            try:
                return next(client_iters[client_idx])
            except StopIteration:
                client_iters[client_idx] = iter(self.client_dataloaders[client_idx])
                return next(client_iters[client_idx])

        with round_bar(num_rounds, "Communication Rounds") as bar:
            for round_idx in range(1, num_rounds + 1):
                executed_rounds = round_idx
                model.train()
                bar.update(1)
                active_list = list(range(num_clients))
                random.shuffle(active_list)

                client_gradients = []
                attack_payloads = [] if self.callbacks.attack_fn is not None else None
                for client_idx in active_list:
                    batches = []
                    for _ in range(local_steps):
                        batches.append(next_batch(client_idx))

                    inputs = torch.cat([b[0] for b in batches], dim=0)
                    labels = torch.cat([b[1] for b in batches], dim=0)
                    samples_seen = len(labels) * local_epochs
                    examples_seen[client_idx] += samples_seen

                    device = next(model.parameters()).device
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    batch_ds = TensorDataset(inputs, labels)
                    batch_loader = DataLoader(batch_ds, batch_size=len(batch_ds), shuffle=False)

                    rng_pre = capture_rng_state()
                    grads = train_nostep(model, batch_loader, MetaSGD(lr), criterion, epochs=local_epochs)
                    rng_post = capture_rng_state()
                    client_gradients.append([g.detach() if g is not None else None for g in grads])
                    if attack_payloads is not None:
                        attack_payloads.append((batch_loader, client_idx, grads, rng_pre, rng_post))

                current_exposures, exp_min_curr, exp_avg_curr, exp_max_curr = self._exposure_stats(examples_seen, client_n_eff)
                crossed_val_checkpoint, next_val_exposure = self._next_val_checkpoint(
                    float(exp_min_prev),
                    float(next_val_exposure),
                    float(exp_min_curr),
                )
                if self.callbacks.attack_fn is not None and attack_payloads is not None:
                    self.callbacks.attack_fn(
                        model=self.model,
                        round_idx=int(round_idx),
                        attack_payloads=attack_payloads,
                        exp_min_prev=float(exp_min_prev),
                        exp_min_curr=float(exp_min_curr),
                        current_exposures=current_exposures,
                        exp_min=float(exp_min_curr),
                        exp_avg=float(exp_avg_curr),
                        exp_max=float(exp_max_curr),
                    )
                exp_min_prev = exp_min_curr

                self._fedsgd_train(model, client_gradients, lr)

                if crossed_val_checkpoint:
                    train_stats = self._eval(self.client_dataloaders)
                    val_stats = self._eval([self.val_loader]) if self.val_loader is not None else None
                    if val_stats is not None:
                        val_loss = float(val_stats["loss"])
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_round = int(round_idx)
                            self.best_state_dict = self.model_wrapper.snapshot_state()
                    self._save_metrics(
                        phase="checkpoint",
                        round_idx=int(round_idx),
                        exp_min=float(exp_min_curr),
                        exp_avg=float(exp_avg_curr),
                        exp_max=float(exp_max_curr),
                        metrics_by_stage={"train": train_stats, "val": val_stats},
                    )
                    self._log("val", executed_rounds, train_stats, val_stats)

        return self._finalize(
            executed_rounds=executed_rounds,
            examples_seen=examples_seen,
            client_n_eff=client_n_eff,
            protocol_label="FedSGD",
        )


def build_fl_trainer(
    protocol: str,
    fl_cfg: FedAvgConfig | FedSGDConfig,
    batch_size: int,
    client_dataloaders: list[DataLoader],
    val_loader: DataLoader,
    test_loader: DataLoader,
    callbacks: FLCallbacks,
):
    if protocol == "fedsgd":
        return FedSGDTrainer(
            fl_cfg=fl_cfg,
            batch_size=batch_size,
            client_dataloaders=client_dataloaders,
            val_loader=val_loader,
            test_loader=test_loader,
            callbacks=callbacks,
        )
    if protocol == "fedavg":
        return FedAvgTrainer(
            fl_cfg=fl_cfg,
            batch_size=batch_size,
            client_dataloaders=client_dataloaders,
            val_loader=val_loader,
            test_loader=test_loader,
            callbacks=callbacks,
        )
    raise ValueError(f"Unknown protocol '{protocol}'")
