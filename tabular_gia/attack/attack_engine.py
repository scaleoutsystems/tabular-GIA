from __future__ import annotations

from dataclasses import asdict
import math
from copy import deepcopy
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep
from tabular_gia.metrics.tabular_metrics import (
    compute_reconstruction_metrics,
    prepare_tensors_for_metrics,
    write_debug_reconstruction_txt,
)

logger = logging.getLogger(__name__)


def _tqdm_kwargs(*, offset: int = 0) -> dict:
    raw_slot = os.environ.get("TABULAR_GIA_TQDM_SLOT")
    if raw_slot is None:
        return {}
    try:
        slot = int(raw_slot)
    except ValueError:
        return {}
    try:
        stride = int(os.environ.get("TABULAR_GIA_TQDM_STRIDE", "3"))
    except ValueError:
        stride = 3
    try:
        base = int(os.environ.get("TABULAR_GIA_TQDM_BASE", "0"))
    except ValueError:
        base = 0
    return {"position": int(base) + (slot * max(1, stride)) + int(offset)}


class AttackScheduler:
    def __init__(
        self,
        *,
        gia_cfg: GiaConfig,
    ) -> None:
        self.gia_cfg = gia_cfg

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
        self.total_rounds = 0

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
        self.total_rounds = int(total_rounds)
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
        tag = os.environ.get("TABULAR_GIA_TQDM_TAG")
        base_desc = f"GIA Round {round_idx} Client {client_idx}"
        render_desc = f"{tag} | {base_desc}" if tag else base_desc
        with tqdm(
            total=total_iters,
            desc=render_desc,
            leave=False,
            **_tqdm_kwargs(offset=1),
        ) as bar:
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

        orig_tensor, recon_tensor, label_tensor = prepare_tensors_for_metrics(
            original=attacker.original,
            best_reconstruction=attacker.best_reconstruction,
            reconstruction_labels=attacker.reconstruction_labels,
            model=attacker.model,
            feature_schema=self.feature_schema,
            client_idx=client_idx,
        )
        true_labels = torch.cat([batch[1] for batch in batch_loader], dim=0).detach().cpu().view(-1)
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
            true_label_tensor=true_labels,
            recovered_label_tensor=label_tensor,
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
        if os.environ.get("TABULAR_GIA_TQDM_SLOT") is None:
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

    def run_attack_vectorized(
        self,
        *,
        att_model: torch.nn.Module,
        attack_payloads: list[tuple[DataLoader, int, list, dict]],
        round_idx: int,
        checkpoint_type: str,
        checkpoint_label: str,
        fixed_batch_id: int = -1,
    ) -> list[dict]:
        if not attack_payloads:
            return []

        device = next(att_model.parameters()).device
        x_per_client: list[torch.Tensor] = []
        y_per_client: list[torch.Tensor] = []
        client_ids: list[int] = []
        attack_contexts: list[dict] = []
        observed_updates: list[list[torch.Tensor | None]] = []
        expected_rows: int | None = None
        expected_client_batch_size: int | None = None
        for batch_loader, client_idx, client_updates, attack_context in attack_payloads:
            if client_updates is None:
                raise ValueError("Vectorized attack requires observed gradients for every payload.")
            xb = torch.cat([batch[0] for batch in batch_loader], dim=0)
            yb = torch.cat([batch[1] for batch in batch_loader], dim=0)
            rows = int(xb.shape[0])
            if expected_rows is None:
                expected_rows = rows
            elif rows != expected_rows:
                raise ValueError("Vectorized attack requires same number of rows per client batch.")
            client_batch_size = int(batch_loader.batch_size) if batch_loader.batch_size is not None else rows
            if expected_client_batch_size is None:
                expected_client_batch_size = client_batch_size
            elif client_batch_size != expected_client_batch_size:
                raise ValueError("Vectorized attack requires same client local batch size across payloads.")
            x_per_client.append(xb.detach().cpu())
            y_per_client.append(yb.detach().cpu())
            client_ids.append(int(client_idx))
            attack_contexts.append(attack_context)
            observed_updates.append(client_updates)

        x_stack = torch.stack(x_per_client, dim=0)
        y_stack = torch.stack(y_per_client, dim=0)
        batch_loader = DataLoader(
            TensorDataset(x_stack, y_stack),
            batch_size=int(x_stack.shape[0]),
            shuffle=False,
        )

        first_updates = observed_updates[0]
        num_params = len(first_updates)
        obs_grads_stacked: list[torch.Tensor | None] = []
        for p_idx in range(num_params):
            grads = []
            for updates in observed_updates:
                if len(updates) != num_params:
                    raise ValueError("Observed gradient list length mismatch across vectorized payloads.")
                grads.append(updates[p_idx])
            none_mask = [grad is None for grad in grads]
            if all(none_mask):
                obs_grads_stacked.append(None)
                continue
            if any(none_mask):
                raise ValueError(
                    f"Inconsistent observed gradients at param index {p_idx}: mixed None/non-None across clients."
                )
            obs_grads_stacked.append(
                torch.stack([grad.detach().to(device) for grad in grads], dim=0)
            )

        attack_cfg = deepcopy(self.attack_cfg_base)
        attack_cfg.vectorized_client_batch_size = expected_client_batch_size
        attacker = InvertingGradients(
            att_model,
            batch_loader,
            None,
            None,
            train_fn=self.train_fn,
            configs=attack_cfg,
            observed_client_gradient=obs_grads_stacked,
        )
        attacker.prepare_attack()

        total_iters = int(attacker.configs.at_iterations or 0)
        last_i = -1
        best_loss = float("inf")
        tag = os.environ.get("TABULAR_GIA_TQDM_TAG")
        base_desc = f"GIA Round {round_idx} All Clients"
        render_desc = f"{tag} | {base_desc}" if tag else base_desc
        with tqdm(
            total=total_iters,
            desc=render_desc,
            leave=False,
            **_tqdm_kwargs(offset=1),
        ) as bar:
            for i, score, _ in attacker.run_attack():
                step = max(0, int(i) - last_i)
                if step:
                    bar.update(step)
                last_i = int(i)
                loss = -float(score) if score is not None else float("nan")
                best_loss = float(attacker.best_loss)
                if i % 250 == 0:
                    bar.set_postfix(loss=f"{loss:.6e}", best=f"{best_loss:.6e}")
            if total_iters > 0 and last_i + 1 < total_iters:
                bar.update(total_iters - (last_i + 1))
        best_x_rec = attacker.best_reconstruction.dataset.reconstruction.to(device)
        model_for_attack = attacker.model
        best_client_losses = attacker.vectorized_best_client_losses

        rows: list[dict] = []
        for c_idx, client_idx in enumerate(client_ids):
            attack_context = attack_contexts[c_idx]
            self.attack_id_counter += 1
            attack_id = self.attack_id_counter

            client_labels = y_stack[c_idx].detach().cpu()
            recon_loader = DataLoader(
                TensorDataset(best_x_rec[c_idx].detach().cpu(), client_labels),
                batch_size=best_x_rec.shape[1],
                shuffle=False,
            )
            artifact_name = f"attack_{attack_id:06d}_round_{int(round_idx):06d}_client_{client_idx:03d}.txt"
            results_path = self.debug_dir / artifact_name

            orig_tensor, recon_tensor, label_tensor = prepare_tensors_for_metrics(
                original=attacker.original[c_idx],
                best_reconstruction=recon_loader,
                reconstruction_labels=attacker.reconstruction_labels[c_idx],
                model=model_for_attack,
                feature_schema=self.feature_schema,
                client_idx=client_idx,
            )
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
                true_label_tensor=y_stack[c_idx].detach().cpu().view(-1),
                recovered_label_tensor=label_tensor,
            )
            metrics = dict(reconstruction_metrics["aggregate_metrics"])
            metrics.pop("nn_median", None)
            metrics["attack_id"] = int(attack_id)
            metrics["round"] = int(round_idx)
            metrics["client_idx"] = client_idx
            metrics["attack_mode"] = self.attack_mode
            metrics["fixed_batch_id"] = int(fixed_batch_id)
            metrics["exp_min"] = float(attack_context["exp_min"])
            metrics["exp_avg"] = float(attack_context["exp_avg"])
            metrics["exp_max"] = float(attack_context["exp_max"])
            metrics["client_exp"] = float(attack_context["client_exp"])
            metrics["checkpoint_type"] = checkpoint_type
            metrics["checkpoint_label"] = checkpoint_label
            self.attack_rows.append(metrics)
            rows.append(metrics)
            if best_client_losses is not None and c_idx < int(best_client_losses.shape[0]):
                client_best = float(best_client_losses[c_idx].detach().cpu())
            else:
                client_best = best_loss
            if os.environ.get("TABULAR_GIA_TQDM_SLOT") is None:
                tqdm.write(
                    f"GIA done: round={int(round_idx)} client={client_idx} best={client_best:.6e}"
                )
        return rows


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
        self.protocol = protocol
        self.gia_cfg = gia_cfg
        self.scheduler = AttackScheduler(
            gia_cfg=gia_cfg,
        )
        lr = fl_cfg.lr
        optimizer_name = fl_cfg.optimizer
        if optimizer_name != "MetaSGD":
            raise ValueError(f"Only MetaSGD is supported, got optimizer='{optimizer_name}'.")
        optimizer = MetaSGD(lr=lr)
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
            "Attack engine setup: mode=%s schedule=%s fixed_batch_k=%d auto_checkpoints=%d vectorized_attacks=%s",
            self.scheduler.attack_mode,
            self.scheduler.attack_schedule,
            self.scheduler.fixed_batch_k,
            self.gia_cfg.auto_checkpoints,
            self.gia_cfg.vectorized_attacks,
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
        fixed_batch_id: int = -1,
    ) -> None:
        scheduled_round = int(round_idx) + 1
        if self.scheduler.total_rounds > 0:
            scheduled_round = min(scheduled_round, self.scheduler.total_rounds)
        self.scheduler.on_exposure_progress(
            scheduled_round,
            float(exp_min_prev),
            float(exp_min_curr),
        )
        checkpoint = self.scheduler.checkpoint(int(round_idx))
        if checkpoint is None:
            return
        checkpoint_type, checkpoint_label = checkpoint

        payload_with_ctx: list[tuple[DataLoader, int, list | None, dict]] = []
        for batch_loader, client_idx, client_updates in attack_payloads:
            client_idx = int(client_idx)
            attack_context = {
                "exp_min": float(exp_min),
                "exp_avg": float(exp_avg),
                "exp_max": float(exp_max),
                "client_exp": float(current_exposures[client_idx]) if current_exposures else 0.0,
            }
            payload_with_ctx.append((batch_loader, client_idx, client_updates, attack_context))
        effective_fixed_batch_id = -1
        if self.scheduler.attack_mode == "fixed_batch":
            effective_fixed_batch_id = int(fixed_batch_id)
            if effective_fixed_batch_id < 0:
                raise ValueError("fixed_batch_id must be provided when attack_mode is 'fixed_batch'.")

        can_vectorize = (
            self.protocol in {"fedsgd", "fedavg"}
            and self.gia_cfg.vectorized_attacks
            and len(payload_with_ctx) > 1
            and all(client_updates is not None for _bl, _cid, client_updates, _ctx in payload_with_ctx)
        )
        if can_vectorize:
            try:
                self.runner.run_attack_vectorized(
                    att_model=model,
                    attack_payloads=[
                        (batch_loader, int(client_idx), client_updates, attack_context)  # type: ignore[arg-type]
                        for batch_loader, client_idx, client_updates, attack_context in payload_with_ctx
                    ],
                    round_idx=int(round_idx),
                    checkpoint_type=checkpoint_type,
                    checkpoint_label=checkpoint_label,
                    fixed_batch_id=effective_fixed_batch_id,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Batched vectorized attack fallback to per-client mode at round=%d due to: %s",
                    int(round_idx),
                    exc,
                )

        for batch_loader, client_idx, client_updates, attack_context in payload_with_ctx:
            self.runner.run_attack(
                att_model=model,
                batch_loader=batch_loader,
                round_idx=int(round_idx),
                client_idx=client_idx,
                client_updates=client_updates,
                fixed_batch_id=effective_fixed_batch_id,
                checkpoint_type=checkpoint_type,
                checkpoint_label=checkpoint_label,
                attack_context=attack_context,
            )

    def get_attack_rows(self) -> list[dict]:
        return list(self.runner.attack_rows)
