from __future__ import annotations

from dataclasses import asdict
import math
from copy import deepcopy
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep
from tabular_gia.metrics.tabular_metrics import (
    compute_reconstruction_metrics,
    prepare_tensors_for_metrics,
    write_debug_reconstruction_txt,
)

logger = logging.getLogger(__name__)


class AttackScheduler:
    def __init__(
        self,
        *,
        protocol: str,
        gia_cfg: GiaConfig,
        fl_cfg: FedAvgConfig | FedSGDConfig,
        seed: int,
        client_dataloaders: list[DataLoader],
    ) -> None:
        self.protocol = protocol
        self.gia_cfg = gia_cfg
        self.fl_cfg = fl_cfg
        self.seed = seed
        self.client_dataloaders = client_dataloaders

        self.attack_mode = gia_cfg.attack_mode.strip().lower()
        self.attack_schedule = gia_cfg.attack_schedule.strip().lower()
        self.fixed_batch_k = gia_cfg.fixed_batch_k

        self.exposure_include_round1_preagg = False
        self.exposure_targets: list[float] = []
        if self.attack_schedule == "exposure":
            raw = gia_cfg.attack_exposure_milestones
            self.exposure_include_round1_preagg = any(float(m) == 0.0 for m in raw)
            self.exposure_targets = sorted({float(m) for m in raw if float(m) > 0.0})

        self.checkpoint_rounds: set[int] | None = None
        self.exposure_round_labels: dict[int, list[float]] = {}
        self.next_exposure_idx = 0
        self.fixed_batch_loaders: dict[int, list[DataLoader]] = {}

        if self.attack_mode == "fixed_batch":
            self._build_fixed_batch_loaders()

    def _build_checkpoint_rounds(self, total_rounds: int) -> set[int] | None:
        if self.attack_schedule == "all":
            return None
        if self.attack_schedule == "pow2":
            rounds: set[int] = set()
            r = 1
            while r <= total_rounds:
                rounds.add(r)
                r <<= 1
            return rounds
        if self.attack_schedule == "fixed":
            raw = self.gia_cfg.attack_rounds
            return {int(r) for r in raw if 1 <= int(r) <= total_rounds}
        if self.attack_schedule == "logspace":
            count = self.gia_cfg.attack_num_checkpoints
            if count == 1:
                return {total_rounds}
            rounds = {int(round(math.exp((i / (count - 1)) * math.log(total_rounds)))) for i in range(count)}
            return {min(total_rounds, max(1, r)) for r in rounds}
        if self.attack_schedule == "auto":
            count = self.gia_cfg.auto_checkpoints
            if count <= 0:
                raise ValueError(f"auto_checkpoints must be > 0, got {count}")
            if count >= total_rounds:
                return set(range(1, total_rounds + 1))
            if count == 1:
                return {total_rounds}
            rounds = {
                1 + ((i * (total_rounds - 1)) // (count - 1))
                for i in range(count)
            }
            return {min(total_rounds, max(1, r)) for r in rounds}
        if self.attack_schedule == "exposure":
            return set()
        raise ValueError(f"Unknown attack_schedule '{self.attack_schedule}'")

    def on_num_rounds(self, total_rounds: int) -> None:
        self.checkpoint_rounds = self._build_checkpoint_rounds(int(total_rounds))
        if self.attack_schedule == "exposure" and self.exposure_include_round1_preagg and int(total_rounds) >= 1:
            self.checkpoint_rounds.add(1)
            self.exposure_round_labels.setdefault(1, []).append(0.0)

    def on_exposure_progress(self, round_idx: int, exp_min_prev: float, exp_min_curr: float) -> None:
        if self.attack_schedule != "exposure" or self.checkpoint_rounds is None:
            return
        crossed: list[float] = []
        while self.next_exposure_idx < len(self.exposure_targets) and self.exposure_targets[self.next_exposure_idx] <= exp_min_curr:
            milestone = self.exposure_targets[self.next_exposure_idx]
            if exp_min_prev < milestone:
                crossed.append(float(milestone))
            self.next_exposure_idx += 1
        if crossed:
            self.checkpoint_rounds.add(int(round_idx))
            existing = self.exposure_round_labels.get(int(round_idx), [])
            self.exposure_round_labels[int(round_idx)] = existing + [m for m in crossed if m not in existing]

    def checkpoint(self, round_idx: int) -> tuple[str, str] | None:
        if self.attack_schedule != "all":
            if self.checkpoint_rounds is None or int(round_idx) not in self.checkpoint_rounds:
                return None

        if self.attack_schedule == "exposure":
            crossed = self.exposure_round_labels.get(int(round_idx), [])
            label = ";".join(f"{m:.6g}" for m in crossed) if crossed else "exposure"
            return "exposure", label
        if self.attack_schedule == "all":
            return "round", "all"
        return "round", str(int(round_idx))

    def _sample_fixed_units(self, num_effective: int, unit_size: int, k_units: int, seed: int) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        perm = torch.randperm(num_effective, generator=generator).tolist()
        selected = perm[: k_units * unit_size]
        return [selected[i * unit_size : (i + 1) * unit_size] for i in range(k_units)]

    def _build_fixed_batch_loader(self, client_loader: DataLoader, sample_ids: list[int]) -> DataLoader:
        x_all, y_all = client_loader.dataset.tensors
        idx = torch.as_tensor(sample_ids, dtype=torch.long)
        fixed_batch_ds = TensorDataset(x_all.index_select(0, idx), y_all.index_select(0, idx))
        return DataLoader(fixed_batch_ds, batch_size=int(client_loader.batch_size), shuffle=False)

    def _build_fixed_batch_loaders(self) -> None:
        local_steps_raw = self.fl_cfg.local_steps
        local_steps_all = isinstance(local_steps_raw, str) and local_steps_raw.strip().lower() == "all"
        local_steps = None if local_steps_all else max(1, int(local_steps_raw))
        base_seed = self.seed

        for client_idx, client_loader in enumerate(self.client_dataloaders):
            batch_size = int(client_loader.batch_size)
            num_effective = int(len(client_loader) * batch_size)
            if self.protocol == "fedsgd":
                sequence_batches = 1
            else:
                sequence_batches = int(len(client_loader)) if local_steps_all else int(local_steps)
            unit_size = int(sequence_batches * batch_size)
            seed = base_seed + (client_idx * 1_000_003)
            unit_sample_ids = self._sample_fixed_units(
                num_effective=num_effective,
                unit_size=unit_size,
                k_units=self.fixed_batch_k,
                seed=seed,
            )
            self.fixed_batch_loaders[client_idx] = [self._build_fixed_batch_loader(client_loader, ids) for ids in unit_sample_ids]


class AttackRunner:
    def __init__(
        self,
        *,
        train_fn: object,
        attack_cfg_base: InvertingConfig,
        feature_schema: dict,
        seed: int,
        results_dir: Path,
        attack_mode: str,
    ) -> None:
        self.train_fn = train_fn
        self.attack_cfg_base = attack_cfg_base
        self.feature_schema = feature_schema
        self.seed = seed
        self.debug_dir = results_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.attack_mode = attack_mode
        self.attack_id_counter = 0
        self.attack_rows: list[dict] = []

    def run_attack(
        self,
        *,
        att_model: torch.nn.Module,
        batch_loader: DataLoader,
        round_idx: int,
        client_idx: int,
        client_updates,
        fixed_batch_id: int,
        checkpoint_type: str,
        checkpoint_label: str,
        attack_context: dict,
    ) -> dict:
        self.attack_id_counter += 1
        attack_id = self.attack_id_counter

        attack_cfg = deepcopy(self.attack_cfg_base)
        attacker = InvertingGradients(
            att_model,
            batch_loader,
            None,
            None,
            train_fn=self.train_fn,
            configs=attack_cfg,
            observed_client_gradient=client_updates,
        )
        attacker.prepare_attack()

        total_iters = int(attack_cfg.at_iterations or 0)
        last_i = -1
        with tqdm(total=total_iters, desc=f"GIA Round {round_idx} Client {client_idx}", leave=False) as bar:
            for i, score, _ in attacker.run_attack():
                step = max(0, int(i) - last_i)
                if step:
                    bar.update(step)
                last_i = int(i)
                loss = -float(score) if score is not None else float("nan")
                bar.set_postfix(loss=f"{loss:.6e}", best=f"{float(attacker.best_loss):.6e}")
            if total_iters > 0 and last_i + 1 < total_iters:
                bar.update(total_iters - (last_i + 1))

        artifact_name = f"attack_{attack_id:06d}_round_{int(round_idx):06d}_client_{int(client_idx):03d}.txt"
        results_path = self.debug_dir / artifact_name

        orig_tensor, recon_tensor, label_tensor = prepare_tensors_for_metrics(attacker, self.feature_schema, client_idx)
        reconstruction_metrics = compute_reconstruction_metrics(
            orig_tensor,
            recon_tensor,
            self.feature_schema,
            client_idx,
            random_baseline_seed=self.seed,
        )
        write_debug_reconstruction_txt(
            results_path,
            orig_tensor,
            recon_tensor,
            reconstruction_metrics["per_row_metrics"],
            self.feature_schema,
            client_idx,
            label_tensor,
        )

        metrics = dict(reconstruction_metrics["aggregate_metrics"])
        metrics.pop("nn_median", None)
        metrics["attack_id"] = int(attack_id)
        metrics["round"] = int(round_idx)
        metrics["client_idx"] = int(client_idx)
        metrics["attack_mode"] = self.attack_mode
        metrics["fixed_batch_id"] = int(fixed_batch_id)
        metrics["exp_min"] = float(attack_context["exp_min"])
        metrics["exp_avg"] = float(attack_context["exp_avg"])
        metrics["exp_max"] = float(attack_context["exp_max"])
        metrics["client_exp"] = float(attack_context["client_exp"])
        metrics["checkpoint_type"] = checkpoint_type
        metrics["checkpoint_label"] = checkpoint_label

        self.attack_rows.append(metrics)
        if int(fixed_batch_id) >= 0:
            tqdm.write(
                f"GIA done: round={int(round_idx)} client={int(client_idx)} "
                f"fixed_batch_id={int(fixed_batch_id)} best={float(attacker.best_loss):.6e}"
            )
        else:
            tqdm.write(
                f"GIA done: round={int(round_idx)} client={int(client_idx)} "
                f"best={float(attacker.best_loss):.6e}"
            )
        return metrics


class AttackEngine:
    def __init__(
        self,
        *,
        protocol: str,
        gia_cfg: GiaConfig,
        fl_cfg: FedAvgConfig | FedSGDConfig,
        seed: int,
        feature_schema: dict,
        client_dataloaders: list[DataLoader],
        criterion: torch.nn.Module,
        results_dir: Path,
    ) -> None:
        self.gia_cfg = gia_cfg
        self.scheduler = AttackScheduler(
            protocol=protocol,
            gia_cfg=gia_cfg,
            fl_cfg=fl_cfg,
            seed=seed,
            client_dataloaders=client_dataloaders,
        )
        lr = fl_cfg.lr
        optimizer_name = fl_cfg.optimizer
        optimizer = MetaAdam(lr=lr) if optimizer_name == "MetaAdam" else MetaSGD(lr=lr)
        invertingconfig = asdict(gia_cfg.invertingconfig)
        invertingconfig.pop("data_extension", None)
        attack_cfg_base = InvertingConfig(
            optimizer=optimizer,
            criterion=criterion,
            data_extension=GiaTabularExtension(),
            epochs=fl_cfg.local_epochs,
            **invertingconfig,
        )
        if protocol == "fedsgd":
            train_fn = train_nostep
        else:
            train_fn = train
        self.runner = AttackRunner(
            train_fn=train_fn,
            attack_cfg_base=attack_cfg_base,
            feature_schema=feature_schema,
            seed=seed,
            results_dir=results_dir,
            attack_mode=self.scheduler.attack_mode,
        )
        self.attack_cfg_base = attack_cfg_base
        logger.info(
            "Attack engine setup: mode=%s schedule=%s fixed_batch_k=%d auto_checkpoints=%d",
            self.scheduler.attack_mode,
            self.scheduler.attack_schedule,
            self.scheduler.fixed_batch_k,
            self.gia_cfg.auto_checkpoints,
        )

    def on_attack_init(self, num_rounds: int) -> None:
        self.scheduler.on_num_rounds(int(num_rounds))
        checkpoints = self.scheduler.checkpoint_rounds
        checkpoint_count = "all" if checkpoints is None else str(len(checkpoints))
        logger.info(
            "Attack schedule initialized: rounds=%d checkpoints=%s at_iterations=%d attack_lr=%.6g label_known=%s",
            int(num_rounds),
            checkpoint_count,
            int(self.attack_cfg_base.at_iterations),
            float(self.attack_cfg_base.attack_lr),
            bool(self.attack_cfg_base.label_known),
        )

    def on_attack(
        self,
        *,
        model: torch.nn.Module,
        round_idx: int,
        attack_payloads: list[tuple[DataLoader, int, list | None]],
        exp_min_prev: float,
        exp_min_curr: float,
        current_exposures: list[float],
        exp_min: float,
        exp_avg: float,
        exp_max: float,
    ) -> None:
        self.scheduler.on_exposure_progress(
            int(round_idx),
            float(exp_min_prev),
            float(exp_min_curr),
        )
        checkpoint = self.scheduler.checkpoint(int(round_idx))
        if checkpoint is None:
            return
        checkpoint_type, checkpoint_label = checkpoint

        for batch_loader, client_idx, client_updates in attack_payloads:
            client_idx = int(client_idx)
            attack_context = {
                "exp_min": float(exp_min),
                "exp_avg": float(exp_avg),
                "exp_max": float(exp_max),
                "client_exp": float(current_exposures[client_idx]) if current_exposures else 0.0,
            }
            if self.scheduler.attack_mode == "fixed_batch":
                fixed_batch_loaders = self.scheduler.fixed_batch_loaders[client_idx]
                for fixed_batch_id, fixed_batch_loader in enumerate(fixed_batch_loaders):
                    self.runner.run_attack(
                        att_model=model,
                        batch_loader=fixed_batch_loader,
                        round_idx=int(round_idx),
                        client_idx=client_idx,
                        client_updates=None,
                        fixed_batch_id=int(fixed_batch_id),
                        checkpoint_type=checkpoint_type,
                        checkpoint_label=checkpoint_label,
                        attack_context=attack_context,
                    )
                continue
            self.runner.run_attack(
                att_model=model,
                batch_loader=batch_loader,
                round_idx=int(round_idx),
                client_idx=client_idx,
                client_updates=client_updates,
                fixed_batch_id=-1,
                checkpoint_type=checkpoint_type,
                checkpoint_label=checkpoint_label,
                attack_context=attack_context,
            )

    def get_attack_rows(self) -> list[dict]:
        return list(self.runner.attack_rows)
