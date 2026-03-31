import logging
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch.func import functional_call, grad as func_grad, vmap
from torch.utils.data import DataLoader, TensorDataset

from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from fl.metrics.fl_metrics import eval, progress_write, round_bar
from fl.fl_optimizers import FLOptimizer, FLSGD
from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.model_mode_utils import bn_eval_mode


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
    attack_mode: str = "round_checkpoint"
    fixed_batch_k: int = 1
    attack_seed: int = 0


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

        def _metric_str(stats: dict, key: str) -> str:
            value = stats.get(key, float("nan"))
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return "nan"

        if self.task == "binary":
            if stage == "val":
                progress_write(
                    "Rounds %d - train_loss=%s train_acc=%s train_f1=%s train_roc_auc=%s train_pr_auc=%s "
                    "val_loss=%s val_acc=%s val_f1=%s val_roc_auc=%s val_pr_auc=%s"
                    % (
                        executed_rounds,
                        _metric_str(train_stats, "loss"),
                        _metric_str(train_stats, "acc"),
                        _metric_str(train_stats, "f1"),
                        _metric_str(train_stats, "roc_auc"),
                        _metric_str(train_stats, "pr_auc"),
                        _metric_str(stage_stats, "loss"),
                        _metric_str(stage_stats, "acc"),
                        _metric_str(stage_stats, "f1"),
                        _metric_str(stage_stats, "roc_auc"),
                        _metric_str(stage_stats, "pr_auc"),
                    )
                )
                return
            progress_write(
                "Test - loss=%s acc=%s f1=%s roc_auc=%s pr_auc=%s"
                % (
                    _metric_str(stage_stats, "loss"),
                    _metric_str(stage_stats, "acc"),
                    _metric_str(stage_stats, "f1"),
                    _metric_str(stage_stats, "roc_auc"),
                    _metric_str(stage_stats, "pr_auc"),
                )
            )
            return

        if self.task == "multiclass":
            if stage == "val":
                progress_write(
                    (
                        "Rounds %d - train_loss=%s train_acc=%s train_precision_macro=%s train_recall_macro=%s "
                        "train_f1_macro=%s train_f1_weighted=%s val_loss=%s val_acc=%s "
                        "val_precision_macro=%s val_recall_macro=%s val_f1_macro=%s val_f1_weighted=%s"
                    )
                    % (
                        executed_rounds,
                        _metric_str(train_stats, "loss"),
                        _metric_str(train_stats, "acc"),
                        _metric_str(train_stats, "precision_macro"),
                        _metric_str(train_stats, "recall_macro"),
                        _metric_str(train_stats, "f1_macro"),
                        _metric_str(train_stats, "f1_weighted"),
                        _metric_str(stage_stats, "loss"),
                        _metric_str(stage_stats, "acc"),
                        _metric_str(stage_stats, "precision_macro"),
                        _metric_str(stage_stats, "recall_macro"),
                        _metric_str(stage_stats, "f1_macro"),
                        _metric_str(stage_stats, "f1_weighted"),
                    )
                )
                return
            progress_write(
                "Test - loss=%s acc=%s precision_macro=%s recall_macro=%s f1_macro=%s f1_weighted=%s"
                % (
                    _metric_str(stage_stats, "loss"),
                    _metric_str(stage_stats, "acc"),
                    _metric_str(stage_stats, "precision_macro"),
                    _metric_str(stage_stats, "recall_macro"),
                    _metric_str(stage_stats, "f1_macro"),
                    _metric_str(stage_stats, "f1_weighted"),
                )
            )
            return

        if stage == "val":
            progress_write(
                (
                    "Rounds %d - train_loss=%s train_mse=%s train_mae=%s train_r2=%s "
                    "val_loss=%s val_mse=%s val_mae=%s val_r2=%s"
                )
                % (
                    executed_rounds,
                    _metric_str(train_stats, "loss"),
                    _metric_str(train_stats, "mse"),
                    _metric_str(train_stats, "mae"),
                    _metric_str(train_stats, "r2"),
                    _metric_str(stage_stats, "loss"),
                    _metric_str(stage_stats, "mse"),
                    _metric_str(stage_stats, "mae"),
                    _metric_str(stage_stats, "r2"),
                )
            )
            return
        progress_write(
            "Test - loss=%s mse=%s mae=%s r2=%s"
            % (
                _metric_str(stage_stats, "loss"),
                _metric_str(stage_stats, "mse"),
                _metric_str(stage_stats, "mae"),
                _metric_str(stage_stats, "r2"),
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

        tested_round = int(executed_rounds)
        tested_exp_min = float(exp_min)
        tested_exp_avg = float(exp_avg)
        tested_exp_max = float(exp_max)
        if self.best_state_dict is not None:
            self.model_wrapper.restore_state(self.best_state_dict)
            logger.info(
                "Loaded best validation checkpoint: round=%d val_loss=%.4f",
                self.best_round,
                self.best_val_loss,
            )
            tested_round = int(self.best_round)
            best_checkpoint = next((row for row in reversed(self.fl_rows) if row["phase"].strip() == "checkpoint" and row["round"] == tested_round), None)
            if best_checkpoint is not None:
                tested_exp_min = float(best_checkpoint["exp_min"])
                tested_exp_avg = float(best_checkpoint["exp_avg"])
                tested_exp_max = float(best_checkpoint["exp_max"])
        test_stats = self._eval([self.test_loader])
        self._save_metrics(
            phase="final_test",
            round_idx=tested_round,
            exp_min=tested_exp_min,
            exp_avg=tested_exp_avg,
            exp_max=tested_exp_max,
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

    def _build_fl_optimizer(self, optimizer_name: str, lr: float) -> FLOptimizer:
        if optimizer_name == "MetaSGD":
            return FLSGD(lr=lr)
        raise ValueError(f"Unsupported optimizer '{optimizer_name}' for FedAvgTrainer.train_fast")

    def train_fast_vectorized(
        self,
        model,
        inputs_by_client: torch.Tensor,
        labels_by_client: torch.Tensor,
        criterion,
        epochs: int,
        client_batch_size: int,
        lr: float,
    ) -> list[torch.Tensor | None]:
        """Vectorized FedAvg local-SGD updates across clients."""
        gpu_or_cpu = next(model.parameters()).device
        model.to(gpu_or_cpu)
        named_params = list(model.named_parameters())
        trainable_pos = [idx for idx, (_, p) in enumerate(named_params) if p.requires_grad]
        num_params = len(named_params)
        if not trainable_pos:
            return [None for _ in range(num_params)]

        num_clients = int(inputs_by_client.shape[0])
        trainable_names = [named_params[idx][0] for idx in trainable_pos]
        base_trainable = tuple(named_params[idx][1].detach().to(gpu_or_cpu) for idx in trainable_pos)
        client_params: tuple[torch.Tensor, ...] = tuple(
            p.unsqueeze(0).expand(num_clients, *p.shape).clone().detach().requires_grad_(p.requires_grad)
            for p in base_trainable
        )
        frozen_params = OrderedDict((name, p) for idx, (name, p) in enumerate(named_params) if idx not in trainable_pos)
        buffers = OrderedDict(model.named_buffers())

        def loss_for_client(
            client_trainable_values: tuple[torch.Tensor, ...],
            x_c: torch.Tensor,
            y_c: torch.Tensor,
        ) -> torch.Tensor:
            state = OrderedDict(zip(trainable_names, client_trainable_values))
            state.update(frozen_params)
            state.update(buffers)
            out = functional_call(model, state, (x_c,))
            return criterion(out, y_c).sum()

        grad_fn = func_grad(loss_for_client)
        rows_per_client = int(inputs_by_client.shape[1])
        step_size = max(1, int(client_batch_size))
        with bn_eval_mode(model):
            for _ in range(epochs):
                for start in range(0, rows_per_client, step_size):
                    end = min(rows_per_client, start + step_size)
                    x_mb = inputs_by_client[:, start:end]
                    y_mb = labels_by_client[:, start:end]
                    grads_by_client = vmap(
                        grad_fn,
                        in_dims=(0, 0, 0),
                        randomness="different",
                    )(client_params, x_mb, y_mb)
                    client_params = tuple(
                        (param_stack - (lr * grad_stack)).detach().requires_grad_(param_stack.requires_grad)
                        for param_stack, grad_stack in zip(client_params, grads_by_client)
                    )

        deltas: list[torch.Tensor | None] = [None for _ in range(num_params)]
        for pos, updated_stack, base in zip(trainable_pos, client_params, base_trainable):
            deltas[pos] = (updated_stack - base.unsqueeze(0)).detach()
        return deltas

    def train_fast(
        self,
        model,
        data: DataLoader,
        optimizer: FLOptimizer,
        criterion,
        epochs: int,
    ) -> list[torch.Tensor]:
        """Fast FL-only variant of gia_train.train with first-order gradients only."""
        gpu_or_cpu = next(model.parameters()).device
        patched_model = MetaModule(model)
        patched_model.parameters = OrderedDict(
            (name, param.detach().clone().requires_grad_(param.requires_grad))
            for name, param in patched_model.parameters.items()
        )
        outputs = None

        for _ in range(epochs):
            for inputs, labels in data:
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), (
                    labels.to(gpu_or_cpu, non_blocking=True) if isinstance(labels, torch.Tensor) else labels
                )
                with bn_eval_mode(patched_model):
                    outputs = patched_model(inputs, patched_model.parameters)
                loss = criterion(outputs, labels).sum()
                patched_model.parameters = optimizer.step(loss, patched_model.parameters)
                # Keep numerical updates identical while dropping step-to-step autograd history.
                patched_model.parameters = OrderedDict(
                    (name, p.detach().requires_grad_(p.requires_grad))
                    for name, p in patched_model.parameters.items()
                )

        model_delta = OrderedDict(
            (name, param - param_origin)
            for ((name, param), (name_origin, param_origin))
            in zip(patched_model.parameters.items(), OrderedDict(model.named_parameters()).items())
        )
        return list(model_delta.values())

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

    def _fedavg_train_vectorized(self, global_model, client_deltas, client_weights):
        if not client_deltas:
            return
        if not client_weights:
            return

        params = list(global_model.parameters())
        num_params = len(params)
        if len(client_deltas) != num_params:
            raise ValueError("Vectorized client deltas do not match model parameters.")
        n_clients = len(client_weights)
        total_weight = float(sum(client_weights)) or 1.0
        with torch.no_grad():
            for idx, param in enumerate(params):
                if not param.requires_grad:
                    continue
                delta_stack = client_deltas[idx]
                if delta_stack is None:
                    continue
                if int(delta_stack.shape[0]) != int(n_clients):
                    raise ValueError("Vectorized client delta stack does not match client weight count.")
                weights = torch.as_tensor(client_weights, device=delta_stack.device, dtype=delta_stack.dtype)
                shape = (n_clients,) + (1,) * (delta_stack.dim() - 1)
                weighted = (delta_stack * weights.view(shape)).sum(dim=0) / total_weight
                param.add_(weighted.detach().to(param.device, dtype=param.dtype))

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
        use_vectorized_clients = cfg.vectorized_clients
        if optimizer_name != "MetaSGD":
            raise ValueError(f"Only MetaSGD/FLSGD is supported for FedAvg, got optimizer='{optimizer_name}'.")
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
        logger.info(
            "FL runtime settings (FedAvg): local_steps=%s local_epochs=%d optimizer=%s lr=%.6g batch_size=%d clients=%d vectorized_clients=%s",
            "all" if local_steps_all else str(local_steps),
            local_epochs,
            optimizer_name,
            lr,
            batch_size,
            num_clients,
            use_vectorized_clients,
        )

        model = self.model
        criterion = self.criterion
        model.train()
        client_iters = [iter(dl) for dl in self.client_dataloaders] if not local_steps_all else []
        examples_seen = [0 for _ in range(num_clients)]
        next_val_exposure = 1.0 if self.val_loader is not None else float("inf")
        executed_rounds = 0
        exp_min_prev = 0.0
        exp_avg_prev = 0.0
        exp_max_prev = 0.0

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
                client_deltas_stacked = None
                vectorized_this_round = False
                vectorized_client_weights: list[int] = []
                prefetched_client_data: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
                if use_vectorized_clients:
                    per_client_inputs: list[torch.Tensor] = []
                    per_client_labels: list[torch.Tensor] = []
                    per_client_loaders: list[DataLoader] | None = [] if attack_payloads is not None else None
                    per_client_ids: list[int] = []
                    same_batch_size = True
                    expected_rows = None
                    for client_idx in active_list:
                        if local_steps_all:
                            batches = list(iter(self.client_dataloaders[client_idx]))
                        else:
                            batches = []
                            for _ in range(local_steps):
                                batches.append(next_batch(client_idx))

                        inputs_cat = torch.cat([b[0] for b in batches], dim=0)
                        labels_cat = torch.cat([b[1] for b in batches], dim=0)
                        rows = int(inputs_cat.shape[0])
                        if expected_rows is None:
                            expected_rows = rows
                        elif rows != expected_rows:
                            same_batch_size = False
                        samples_seen = len(labels_cat) * local_epochs
                        examples_seen[client_idx] += samples_seen
                        vectorized_client_weights.append(int(samples_seen))
                        prefetched_client_data.append((int(client_idx), inputs_cat, labels_cat, int(samples_seen)))
                        per_client_inputs.append(inputs_cat)
                        per_client_labels.append(labels_cat)
                        if per_client_loaders is not None:
                            batch_ds = TensorDataset(inputs_cat, labels_cat)
                            per_client_loaders.append(DataLoader(batch_ds, batch_size=batch_size, shuffle=False))
                        per_client_ids.append(int(client_idx))

                    if same_batch_size and per_client_inputs:
                        gpu_or_cpu = next(model.parameters()).device
                        inputs_by_client = torch.stack(per_client_inputs, dim=0).to(gpu_or_cpu, non_blocking=True)
                        labels_by_client = torch.stack(per_client_labels, dim=0).to(gpu_or_cpu, non_blocking=True)
                        client_deltas_stacked = self.train_fast_vectorized(
                            model=model,
                            inputs_by_client=inputs_by_client,
                            labels_by_client=labels_by_client,
                            criterion=criterion,
                            epochs=local_epochs,
                            client_batch_size=batch_size,
                            lr=lr,
                        )

                        first_delta = next((d for d in client_deltas_stacked if d is not None), None)
                        n_client = int(first_delta.shape[0]) if first_delta is not None else len(active_list)
                        num_params = len(client_deltas_stacked)
                        for c_idx in range(n_client):
                            deltas_full: list[torch.Tensor | None] = [None for _ in range(num_params)]
                            for p_idx, delta in enumerate(client_deltas_stacked):
                                if delta is not None:
                                    deltas_full[p_idx] = delta[c_idx].detach()
                            client_updates.append((deltas_full, vectorized_client_weights[c_idx]))
                            if attack_payloads is not None:
                                attack_payloads.append(
                                    (
                                        per_client_loaders[c_idx],  # type: ignore[index]
                                        per_client_ids[c_idx],
                                        deltas_full,
                                    )
                                )
                        vectorized_this_round = True

                if not vectorized_this_round:
                    if prefetched_client_data:
                        client_batches = prefetched_client_data
                    else:
                        client_batches = []
                        for client_idx in active_list:
                            if local_steps_all:
                                batches = list(iter(self.client_dataloaders[client_idx]))
                            else:
                                batches = []
                                for _ in range(local_steps):
                                    batches.append(next_batch(client_idx))
                            inputs_cat = torch.cat([b[0] for b in batches], dim=0)
                            labels_cat = torch.cat([b[1] for b in batches], dim=0)
                            samples_seen = len(labels_cat) * local_epochs
                            examples_seen[client_idx] += samples_seen
                            client_batches.append((int(client_idx), inputs_cat, labels_cat, int(samples_seen)))

                    for client_idx, inputs_cat, labels_cat, samples_seen in client_batches:
                        batch_ds = TensorDataset(inputs_cat, labels_cat)
                        batch_loader = DataLoader(batch_ds, batch_size=batch_size, shuffle=False)
                        fl_optimizer = self._build_fl_optimizer(optimizer_name=optimizer_name, lr=lr)
                        deltas = self.train_fast(
                            model=model,
                            data=batch_loader,
                            optimizer=fl_optimizer,
                            criterion=criterion,
                            epochs=local_epochs,
                        )
                        client_updates.append((deltas, samples_seen))
                        if attack_payloads is not None:
                            attack_payloads.append((batch_loader, client_idx, deltas))

                current_exposures, exp_min_curr, exp_avg_curr, exp_max_curr = self._exposure_stats(examples_seen, client_n_eff)
                crossed_val_checkpoint, next_val_exposure = self._next_val_checkpoint( float(exp_min_prev), float(next_val_exposure), float(exp_min_curr))

                if self.callbacks.attack_fn is not None and attack_payloads is not None:
                    self.callbacks.attack_fn(
                        model=self.model,
                        round_idx=int(round_idx),
                        attack_payloads=attack_payloads,
                        exp_min_prev=float(exp_min_prev),
                        exp_min_curr=float(exp_min_curr),
                        current_exposures=current_exposures,
                        exp_min=float(exp_min_prev),
                        exp_avg=float(exp_avg_prev),
                        exp_max=float(exp_max_prev),
                    )

                if vectorized_this_round and client_deltas_stacked is not None:
                    self._fedavg_train_vectorized(model, client_deltas_stacked, vectorized_client_weights)
                else:
                    self._fedavg_train(model, client_updates)

                exp_min_prev = exp_min_curr
                exp_avg_prev = exp_avg_curr
                exp_max_prev = exp_max_curr

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

    def train_nostep_fast(
        self,
        model,
        data: DataLoader,
        criterion,
        epochs: int,
    ) -> list[torch.Tensor | None]:
        """Fast FL-only variant of gia_train.train_nostep with first-order gradients only."""
        gpu_or_cpu = next(model.parameters()).device
        model.to(gpu_or_cpu)
        outputs = None
        params = list(model.parameters())
        grads = None
        for _ in range(epochs):
            for inputs, labels in data:
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), (
                    labels.to(gpu_or_cpu, non_blocking=True) if isinstance(labels, torch.Tensor) else labels
                )
                with bn_eval_mode(model):
                    outputs = model(inputs)
                loss = criterion(outputs, labels).sum()
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True,
                    allow_unused=True,
                )
        return list(grads) if grads is not None else [None for _ in params]

    def train_nostep_fast_vectorized(
        self,
        model,
        data: DataLoader,
        criterion,
        epochs: int,
    ) -> list[torch.Tensor | None]:
        """Vectorized FedSGD gradient extraction across clients using torch.func.vmap."""
        gpu_or_cpu = next(model.parameters()).device
        model.to(gpu_or_cpu)
        named_params = list(model.named_parameters())
        trainable_pos = [idx for idx, (_, p) in enumerate(named_params) if p.requires_grad]
        num_params = len(named_params)
        if not trainable_pos:
            return [None for _ in range(num_params)]

        trainable_names = [named_params[idx][0] for idx in trainable_pos]
        trainable_params = tuple(named_params[idx][1] for idx in trainable_pos)
        frozen_params = OrderedDict((name, p) for idx, (name, p) in enumerate(named_params) if idx not in trainable_pos)
        buffers = OrderedDict(model.named_buffers())

        def loss_for_client(trainable_values: tuple[torch.Tensor, ...], x_c: torch.Tensor, y_c: torch.Tensor) -> torch.Tensor:
            state = OrderedDict(zip(trainable_names, trainable_values))
            state.update(frozen_params)
            state.update(buffers)
            out = functional_call(model, state, (x_c,))
            return criterion(out, y_c).sum()

        grad_fn = func_grad(loss_for_client)
        client_gradients_trainable: tuple[torch.Tensor, ...] = tuple()
        with bn_eval_mode(model):
            for _ in range(epochs):
                for inputs_by_client, labels_by_client in data:
                    inputs_by_client = inputs_by_client.to(gpu_or_cpu, non_blocking=True)
                    labels_by_client = (
                        labels_by_client.to(gpu_or_cpu, non_blocking=True)
                        if isinstance(labels_by_client, torch.Tensor)
                        else torch.as_tensor(labels_by_client, device=gpu_or_cpu)
                    )
                    client_gradients_trainable = vmap(
                        grad_fn,
                        in_dims=(None, 0, 0),
                        randomness="different",
                    )(trainable_params, inputs_by_client, labels_by_client)
        client_gradients = [None for _ in range(num_params)]
        for pos, grad in zip(trainable_pos, client_gradients_trainable):
            client_gradients[pos] = grad
        return client_gradients

    def _fedsgd_train(self, global_model, client_gradients, lr):
        if not client_gradients:
            return

        params = [p for p in global_model.parameters()]
        num_params = len(params)
        for gradients in client_gradients:
            if len(gradients) != num_params:
                raise ValueError("Client gradients do not match model parameters.")

        with torch.no_grad():
            for idx, param in enumerate(params):
                if not param.requires_grad:
                    continue
                gradients = [g for g in (client[idx] for client in client_gradients) if g is not None]
                if not gradients:
                    continue
                agg = torch.zeros_like(param)
                for grad in gradients:
                    agg.add_(grad.detach().to(param.device, dtype=param.dtype))
                agg.div_(len(gradients))
                param.add_(agg, alpha=-lr)

    def _fedsgd_train_vectorized(self, global_model, client_gradients, lr):
        if not client_gradients:
            return
        params = [p for p in global_model.parameters()]
        num_params = len(params)
        if len(client_gradients) != num_params:
            raise ValueError("Vectorized client gradients do not match model parameters.")
        with torch.no_grad():
            for idx, param in enumerate(params):
                if not param.requires_grad:
                    continue
                gradient = client_gradients[idx]
                if gradient is None:
                    continue
                grad_mean = gradient.mean(dim=0)
                param.add_(grad_mean.detach().to(param.device, dtype=param.dtype), alpha=-lr)

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
        logger.info(
            "FL runtime settings (FedSGD): local_steps=%d local_epochs=%d optimizer=%s lr=%.6g batch_size=%d clients=%d vectorized_clients=%s",
            local_steps,
            local_epochs,
            cfg.optimizer,
            lr,
            batch_size,
            num_clients,
            cfg.vectorized_clients,
        )

        model = self.model
        criterion = self.criterion
        model.train()
        client_iters = [iter(dl) for dl in self.client_dataloaders]
        examples_seen = [0 for _ in range(num_clients)]
        next_val_exposure = 1.0 if self.val_loader is not None else float("inf")
        executed_rounds = 0
        exp_min_prev = 0.0
        exp_avg_prev = 0.0
        exp_max_prev = 0.0
        use_vectorized_clients = cfg.vectorized_clients
        attack_mode = self.callbacks.attack_mode.strip().lower()
        fixed_batch_k = max(1, int(self.callbacks.fixed_batch_k))
        fixed_batch_loaders: dict[int, list[DataLoader]] = {}
        if self.callbacks.attack_fn is not None and attack_mode == "fixed_batch":
            base_seed = int(self.callbacks.attack_seed)
            for client_idx, client_loader in enumerate(self.client_dataloaders):
                loader_batch_size = int(client_loader.batch_size)
                num_effective = int(len(client_loader) * loader_batch_size)
                generator = torch.Generator()
                generator.manual_seed(base_seed + (client_idx * 1_000_003))
                perm = torch.randperm(num_effective, generator=generator).tolist()
                selected = perm[: fixed_batch_k * loader_batch_size]
                x_all, y_all = client_loader.dataset.tensors
                client_fixed_loaders: list[DataLoader] = []
                for fixed_batch_id in range(fixed_batch_k):
                    sample_ids = selected[fixed_batch_id * loader_batch_size : (fixed_batch_id + 1) * loader_batch_size]
                    idx = torch.as_tensor(sample_ids, dtype=torch.long)
                    fixed_ds = TensorDataset(x_all.index_select(0, idx), y_all.index_select(0, idx))
                    client_fixed_loaders.append(DataLoader(fixed_ds, batch_size=len(fixed_ds), shuffle=False))
                fixed_batch_loaders[client_idx] = client_fixed_loaders

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

                client_gradients_list = []
                attack_payloads = [] if self.callbacks.attack_fn is not None else None
                client_gradients = None
                vectorized_this_round = False
                prefetched_client_data: list[tuple[int, torch.Tensor, torch.Tensor]] = []
                if use_vectorized_clients:
                    per_client_inputs: list[torch.Tensor] = []
                    per_client_labels: list[torch.Tensor] = []
                    per_client_loaders: list[DataLoader] | None = [] if attack_payloads is not None else None
                    per_client_ids: list[int] = []
                    same_batch_size = True
                    expected_rows = None
                    for client_idx in active_list:
                        batches = []
                        for _ in range(local_steps):
                            batches.append(next_batch(client_idx))

                        inputs = torch.cat([b[0] for b in batches], dim=0)
                        labels = torch.cat([b[1] for b in batches], dim=0)

                        rows = int(inputs.shape[0])
                        if expected_rows is None:
                            expected_rows = rows
                        elif rows != expected_rows:
                            same_batch_size = False
                        samples_seen = len(labels) * local_epochs
                        examples_seen[client_idx] += samples_seen
                        per_client_inputs.append(inputs)
                        per_client_labels.append(labels)
                        prefetched_client_data.append((int(client_idx), inputs, labels))
                        if per_client_loaders is not None:
                            batch_ds = TensorDataset(inputs, labels)
                            per_client_loaders.append(DataLoader(batch_ds, batch_size=len(batch_ds), shuffle=False))
                        per_client_ids.append(int(client_idx))

                    if same_batch_size and per_client_inputs:
                        gpu_or_cpu = next(model.parameters()).device
                        inputs_by_client = torch.stack(per_client_inputs, dim=0).to(gpu_or_cpu, non_blocking=True)
                        labels_by_client = torch.stack(per_client_labels, dim=0).to(gpu_or_cpu, non_blocking=True)
                        vectorized_loader = DataLoader(
                            TensorDataset(inputs_by_client, labels_by_client),
                            batch_size=int(inputs_by_client.shape[0]),
                            shuffle=False,
                        )
                        client_gradients = self.train_nostep_fast_vectorized(
                            model=model,
                            data=vectorized_loader,
                            criterion=criterion,
                            epochs=local_epochs,
                        )

                        first_grad = next((g for g in client_gradients if g is not None), None)
                        n_client = int(first_grad.shape[0]) if first_grad is not None else len(active_list)
                        num_params = len(client_gradients)
                        for c_idx in range(n_client):
                            grads_full = [None for _ in range(num_params)]
                            for p_idx, gradient in enumerate(client_gradients):
                                if gradient is not None:
                                    grads_full[p_idx] = gradient[c_idx].detach()
                            client_gradients_list.append(grads_full)
                            if attack_payloads is not None:
                                attack_payloads.append(
                                    (
                                        per_client_loaders[c_idx],  # type: ignore[index]
                                        per_client_ids[c_idx],
                                        grads_full,
                                    )
                                )
                        vectorized_this_round = True

                if not vectorized_this_round:
                    if prefetched_client_data:
                        client_batches = prefetched_client_data
                    else:
                        client_batches = []
                        for client_idx in active_list:
                            batches = []
                            for _ in range(local_steps):
                                batches.append(next_batch(client_idx))
                            inputs = torch.cat([b[0] for b in batches], dim=0)
                            labels = torch.cat([b[1] for b in batches], dim=0)
                            samples_seen = len(labels) * local_epochs
                            examples_seen[client_idx] += samples_seen
                            client_batches.append((int(client_idx), inputs, labels))

                    for client_idx, inputs, labels in client_batches:
                        batch_ds = TensorDataset(inputs, labels)
                        batch_loader = DataLoader(batch_ds, batch_size=len(batch_ds), shuffle=False)

                        grads = self.train_nostep_fast(
                            model=model,
                            data=batch_loader,
                            criterion=criterion,
                            epochs=local_epochs,
                        )

                        client_gradients_list.append([g.detach() if g is not None else None for g in grads])
                        if attack_payloads is not None:
                            attack_payloads.append((batch_loader, client_idx, grads))

                current_exposures, exp_min_curr, exp_avg_curr, exp_max_curr = self._exposure_stats(examples_seen, client_n_eff)
                crossed_val_checkpoint, next_val_exposure = self._next_val_checkpoint(
                    float(exp_min_prev),
                    float(next_val_exposure),
                    float(exp_min_curr),
                )
                if self.callbacks.attack_fn is not None:
                    if attack_mode == "fixed_batch":
                        for fixed_batch_id in range(fixed_batch_k):
                            fixed_payloads: list[tuple[DataLoader, int]] = []
                            per_client_inputs: list[torch.Tensor] = []
                            per_client_labels: list[torch.Tensor] = []
                            expected_rows: int | None = None
                            for client_idx in active_list:
                                fixed_batch_loader = fixed_batch_loaders[client_idx][fixed_batch_id]
                                xb = torch.cat([batch[0] for batch in fixed_batch_loader], dim=0)
                                yb = torch.cat([batch[1] for batch in fixed_batch_loader], dim=0)
                                rows = int(xb.shape[0])
                                if expected_rows is None:
                                    expected_rows = rows
                                elif rows != expected_rows:
                                    raise ValueError(
                                        "Vectorized fixed-batch attack requires same number of rows per client batch."
                                    )
                                fixed_payloads.append((fixed_batch_loader, int(client_idx)))
                                per_client_inputs.append(xb)
                                per_client_labels.append(yb)

                            gpu_or_cpu = next(model.parameters()).device
                            inputs_by_client = torch.stack(per_client_inputs, dim=0).to(gpu_or_cpu, non_blocking=True)
                            labels_by_client = torch.stack(per_client_labels, dim=0).to(gpu_or_cpu, non_blocking=True)
                            vectorized_loader = DataLoader(
                                TensorDataset(inputs_by_client, labels_by_client),
                                batch_size=int(inputs_by_client.shape[0]),
                                shuffle=False,
                            )
                            fixed_gradients = self.train_nostep_fast_vectorized(
                                model=model,
                                data=vectorized_loader,
                                criterion=criterion,
                                epochs=local_epochs,
                            )
                            num_params = len(fixed_gradients)
                            fixed_attack_payloads: list[tuple[DataLoader, int, list[torch.Tensor | None]]] = []
                            for c_idx, (fixed_batch_loader, client_idx) in enumerate(fixed_payloads):
                                grads_full: list[torch.Tensor | None] = [None for _ in range(num_params)]
                                for p_idx, gradient in enumerate(fixed_gradients):
                                    if gradient is not None:
                                        grads_full[p_idx] = gradient[c_idx].detach()
                                fixed_attack_payloads.append((fixed_batch_loader, client_idx, grads_full))
                            self.callbacks.attack_fn(
                                model=self.model,
                                round_idx=int(round_idx),
                                attack_payloads=fixed_attack_payloads,
                                exp_min_prev=float(exp_min_prev),
                                exp_min_curr=float(exp_min_curr),
                                current_exposures=current_exposures,
                                exp_min=float(exp_min_prev),
                                exp_avg=float(exp_avg_prev),
                                exp_max=float(exp_max_prev),
                                fixed_batch_id=int(fixed_batch_id),
                            )
                    elif attack_payloads is not None:
                        self.callbacks.attack_fn(
                            model=self.model,
                            round_idx=int(round_idx),
                            attack_payloads=attack_payloads,
                            exp_min_prev=float(exp_min_prev),
                            exp_min_curr=float(exp_min_curr),
                            current_exposures=current_exposures,
                            exp_min=float(exp_min_prev),
                            exp_avg=float(exp_avg_prev),
                            exp_max=float(exp_max_prev),
                        )

                if vectorized_this_round and client_gradients is not None:
                    self._fedsgd_train_vectorized(model, client_gradients, lr)
                else:
                    self._fedsgd_train(model, client_gradients_list, lr)

                exp_min_prev = exp_min_curr
                exp_avg_prev = exp_avg_curr
                exp_max_prev = exp_max_curr

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
