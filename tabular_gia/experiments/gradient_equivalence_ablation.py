from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABULAR_GIA_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, TABULAR_GIA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.base import BaseConfig
from configs.dataset.dataset import DatasetConfig
from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig
from configs.model.model import ModelConfig
from leakpro.fl_utils.data_utils import CustomTensorDataset
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep, train_nostep_vectorized, train_vectorized
from leakpro.fl_utils.similarity_measurements import cosine_similarity_weights
from leakpro.utils.seed import seed_everything
from tabular_gia.runner.run import RunConfig, build_runtime

Protocol = Literal["fedsgd", "fedavg"]


@dataclass(frozen=True)
class SeriesDiff:
    mean_abs_delta: float
    max_abs_delta: float
    final_step_abs_delta: float


@dataclass(frozen=True)
class PairwiseModeDiff:
    left_mode: str
    right_mode: str
    total_loss: SeriesDiff
    per_client_loss: SeriesDiff
    final_reconstruction_l2_mean: float
    final_reconstruction_l2_max: float
    final_reconstruction_abs_max: float


@dataclass(frozen=True)
class ModeTrace:
    mode: str
    total_losses: list[float]
    per_client_losses: list[list[float]]  # [step][client]
    final_reconstruction: Tensor


@dataclass(frozen=True)
class ObservedGradientDiff:
    global_max_abs_delta: float
    global_mean_abs_delta: float


def _configure_repro(seed: int) -> None:
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _build_runtime_for_ablation(seed: int, protocol: Protocol, num_clients: int) -> tuple:
    base_cfg = BaseConfig(seed=seed, protocol=protocol)
    dataset_cfg = DatasetConfig()
    model_cfg = ModelConfig(preset="small")
    fl_cfg = (
        FedSGDConfig(num_clients=int(num_clients))
        if protocol == "fedsgd"
        else FedAvgConfig(local_steps=1, local_epochs=1, num_clients=int(num_clients))
    )
    gia_cfg = GiaConfig()
    run_cfg = RunConfig(
        base_cfg=base_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        fl_cfg=fl_cfg,
        gia_cfg=gia_cfg,
        results_dir=Path("/tmp/tabular_gia_equivalence_ablation"),
        fl_only=False,
    )
    runtime = build_runtime(run_cfg)
    return run_cfg, runtime


def _to_batch_loader(x: Tensor, y: Tensor) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=int(x.shape[0]), shuffle=False)


def _collect_client_batches(client_dataloaders: list[DataLoader], num_clients: int) -> tuple[list[Tensor], list[Tensor], int]:
    x_list: list[Tensor] = []
    y_list: list[Tensor] = []
    expected_batch_size: int | None = None
    for loader in client_dataloaders[:num_clients]:
        xb, yb = next(iter(loader))
        local_batch_size = int(loader.batch_size) if loader.batch_size is not None else int(xb.shape[0])
        if expected_batch_size is None:
            expected_batch_size = local_batch_size
        elif local_batch_size != expected_batch_size:
            raise ValueError(
                f"Expected same client batch size across selected clients, got {expected_batch_size} and {local_batch_size}."
            )
        x_list.append(xb.detach().clone())
        y_list.append(yb.detach().clone())
    return x_list, y_list, int(expected_batch_size or 1)


def _observed_updates_std(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    x_list: list[Tensor],
    y_list: list[Tensor],
    *,
    protocol: Protocol,
    local_epochs: int,
    fl_lr: float,
    client_batch_size: int,
) -> list[list[Tensor | None]]:
    out: list[list[Tensor | None]] = []
    for x, y in zip(x_list, y_list):
        if protocol == "fedavg":
            loader = DataLoader(TensorDataset(x, y), batch_size=int(client_batch_size), shuffle=False)
            grads = train(model, loader, MetaSGD(lr=fl_lr), criterion, epochs=int(local_epochs))
        else:
            loader = _to_batch_loader(x, y)
            grads = train_nostep(model, loader, MetaSGD(lr=fl_lr), criterion, epochs=int(local_epochs))
        out.append([g.detach().clone() if g is not None else None for g in grads])
    return out


def _observed_updates_vec(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    x_list: list[Tensor],
    y_list: list[Tensor],
    *,
    protocol: Protocol,
    local_epochs: int,
    fl_lr: float,
    client_batch_size: int,
) -> list[list[Tensor | None]]:
    x_stack = torch.stack(x_list, dim=0)
    y_stack = torch.stack(y_list, dim=0)
    loader = DataLoader(TensorDataset(x_stack, y_stack), batch_size=int(x_stack.shape[0]), shuffle=False)
    if protocol == "fedavg":
        grads_full = train_vectorized(
            model,
            loader,
            MetaSGD(lr=fl_lr),
            criterion,
            epochs=int(local_epochs),
            client_batch_size=int(client_batch_size),
        )
    else:
        grads_full = train_nostep_vectorized(
            model,
            loader,
            MetaSGD(lr=fl_lr),
            criterion,
            epochs=int(local_epochs),
        )
    num_clients = int(x_stack.shape[0])
    out: list[list[Tensor | None]] = []
    for c_idx in range(num_clients):
        out.append(
            [g[c_idx].detach().clone() if g is not None else None for g in grads_full]
        )
    return out


def _gradient_diff(
    grads_a: list[list[Tensor | None]],
    grads_b: list[list[Tensor | None]],
) -> ObservedGradientDiff:
    abs_sum = 0.0
    elem_count = 0
    max_abs = 0.0
    for client_a, client_b in zip(grads_a, grads_b):
        for ga, gb in zip(client_a, client_b):
            if ga is None and gb is None:
                continue
            if (ga is None) != (gb is None):
                raise ValueError("Observed gradient mismatch: one path has None where the other has a tensor.")
            assert ga is not None and gb is not None
            d = (ga - gb).abs()
            max_abs = max(max_abs, float(d.max().item()))
            abs_sum += float(d.sum().item())
            elem_count += int(d.numel())
    mean_abs = (abs_sum / elem_count) if elem_count > 0 else 0.0
    return ObservedGradientDiff(
        global_max_abs_delta=float(max_abs),
        global_mean_abs_delta=float(mean_abs),
    )


def _attack_trace_std(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    observed_grads: list[list[Tensor | None]],
    x0: Tensor,
    y_stack: Tensor,
    *,
    protocol: Protocol,
    local_epochs: int,
    fl_lr: float,
    client_batch_size: int,
    steps: int,
    attack_lr: float,
    top10norms: bool,
    mode: str,
) -> ModeTrace:
    num_clients = int(x0.shape[0])
    recon_params = [torch.nn.Parameter(x0[c].detach().clone()) for c in range(num_clients)]
    optimizers = [torch.optim.Adam([p], lr=attack_lr) for p in recon_params]
    total_losses: list[float] = []
    per_client_losses: list[list[float]] = []

    for _ in range(steps):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        losses: list[Tensor] = []
        for c_idx in range(num_clients):
            if protocol == "fedavg":
                loader = DataLoader(
                    TensorDataset(recon_params[c_idx], y_stack[c_idx]),
                    batch_size=int(client_batch_size),
                    shuffle=False,
                )
                rec_grads = train(model, loader, MetaSGD(lr=fl_lr), criterion, epochs=int(local_epochs))
            else:
                loader = DataLoader(
                    CustomTensorDataset(recon_params[c_idx], y_stack[c_idx]),
                    batch_size=int(recon_params[c_idx].shape[0]),
                    shuffle=False,
                )
                rec_grads = train_nostep(model, loader, MetaSGD(lr=fl_lr), criterion, epochs=int(local_epochs))
            loss_c = cosine_similarity_weights(rec_grads, observed_grads[c_idx], top10norms).reshape(())
            losses.append(loss_c)

        stacked = torch.stack(losses, dim=0)
        total = stacked.sum()
        total.backward()
        for opt in optimizers:
            opt.step()

        total_losses.append(float(total.detach().cpu()))
        per_client_losses.append([float(v.detach().cpu()) for v in stacked])

    final_recon = torch.stack([p.detach().clone() for p in recon_params], dim=0)
    return ModeTrace(
        mode=mode,
        total_losses=total_losses,
        per_client_losses=per_client_losses,
        final_reconstruction=final_recon,
    )


def _attack_trace_vec(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    observed_grads: list[list[Tensor | None]],
    x0: Tensor,
    y_stack: Tensor,
    *,
    protocol: Protocol,
    local_epochs: int,
    fl_lr: float,
    client_batch_size: int,
    steps: int,
    attack_lr: float,
    top10norms: bool,
    mode: str,
) -> ModeTrace:
    recon = torch.nn.Parameter(x0.detach().clone())
    optimizer = torch.optim.Adam([recon], lr=attack_lr)
    total_losses: list[float] = []
    per_client_losses: list[list[float]] = []
    num_clients = int(x0.shape[0])

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        loader = DataLoader(
            CustomTensorDataset(recon, y_stack),
            batch_size=int(recon.shape[0]),
            shuffle=False,
        )
        if protocol == "fedavg":
            rec_full = train_vectorized(
                model,
                loader,
                MetaSGD(lr=fl_lr),
                criterion,
                epochs=int(local_epochs),
                client_batch_size=int(client_batch_size),
            )
        else:
            rec_full = train_nostep_vectorized(model, loader, MetaSGD(lr=fl_lr), criterion, epochs=int(local_epochs))
        losses: list[Tensor] = []
        for c_idx in range(num_clients):
            rec_client = [g[c_idx] if g is not None else None for g in rec_full]
            loss_c = cosine_similarity_weights(rec_client, observed_grads[c_idx], top10norms).reshape(())
            losses.append(loss_c)

        stacked = torch.stack(losses, dim=0)
        total = stacked.sum()
        total.backward()
        optimizer.step()

        total_losses.append(float(total.detach().cpu()))
        per_client_losses.append([float(v.detach().cpu()) for v in stacked])

    return ModeTrace(
        mode=mode,
        total_losses=total_losses,
        per_client_losses=per_client_losses,
        final_reconstruction=recon.detach().clone(),
    )


def _series_diff(values_a: list[float], values_b: list[float]) -> SeriesDiff:
    diffs = [abs(a - b) for a, b in zip(values_a, values_b)]
    return SeriesDiff(
        mean_abs_delta=float(sum(diffs) / len(diffs)) if diffs else 0.0,
        max_abs_delta=float(max(diffs)) if diffs else 0.0,
        final_step_abs_delta=float(diffs[-1]) if diffs else 0.0,
    )


def _flatten_per_client_losses(per_client_losses: list[list[float]]) -> list[float]:
    flat: list[float] = []
    for row in per_client_losses:
        flat.extend(row)
    return flat


def _mode_diff(left: ModeTrace, right: ModeTrace) -> PairwiseModeDiff:
    recon_delta = (left.final_reconstruction - right.final_reconstruction).detach()
    recon_l2 = torch.norm(recon_delta.view(recon_delta.shape[0], -1), dim=1)
    return PairwiseModeDiff(
        left_mode=left.mode,
        right_mode=right.mode,
        total_loss=_series_diff(left.total_losses, right.total_losses),
        per_client_loss=_series_diff(
            _flatten_per_client_losses(left.per_client_losses),
            _flatten_per_client_losses(right.per_client_losses),
        ),
        final_reconstruction_l2_mean=float(recon_l2.mean().item()),
        final_reconstruction_l2_max=float(recon_l2.max().item()),
        final_reconstruction_abs_max=float(recon_delta.abs().max().item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: per-step gradient equivalence across 4 FL/attack vectorization modes.")
    parser.add_argument("--protocol", type=str, choices=["fedsgd", "fedavg"], default="fedavg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients to include in the ablation.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Client local epochs for observed update generation.")
    parser.add_argument("--fl-lr", type=float, default=1e-2, help="MetaSGD LR for observed update generation.")
    parser.add_argument("--steps", type=int, default=1, help="Number of attack optimization steps to compare.")
    parser.add_argument("--attack-lr", type=float, default=0.03)
    parser.add_argument("--top10norms", action="store_true", help="Use top10norms in cosine similarity.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write JSON summary.")
    args = parser.parse_args()

    _configure_repro(args.seed)
    protocol: Protocol = "fedavg" if args.protocol == "fedavg" else "fedsgd"
    run_cfg, runtime = _build_runtime_for_ablation(args.seed, protocol, int(args.num_clients))
    model = runtime.model_wrapper
    model.eval()
    criterion = model.criterion

    if run_cfg.model_cfg.preset != "small":
        raise ValueError(f"This ablation is restricted to model preset='small', got {run_cfg.model_cfg.preset}.")

    x_list, y_list, client_batch_size = _collect_client_batches(runtime.client_dataloaders, args.num_clients)
    x0 = torch.randn_like(torch.stack(x_list, dim=0))
    y_stack = torch.stack(y_list, dim=0)

    obs_std = _observed_updates_std(
        model,
        criterion,
        x_list,
        y_list,
        protocol=protocol,
        local_epochs=int(args.local_epochs),
        fl_lr=float(args.fl_lr),
        client_batch_size=int(client_batch_size),
    )
    obs_vec = _observed_updates_vec(
        model,
        criterion,
        x_list,
        y_list,
        protocol=protocol,
        local_epochs=int(args.local_epochs),
        fl_lr=float(args.fl_lr),
        client_batch_size=int(client_batch_size),
    )
    obs_diff = _gradient_diff(obs_std, obs_vec)

    modes: dict[str, ModeTrace] = {}
    for fl_mode, observed in (("std_fl", obs_std), ("vec_fl", obs_vec)):
        modes[f"{fl_mode}__std_atk"] = _attack_trace_std(
            model=model,
            criterion=criterion,
            observed_grads=observed,
            x0=x0,
            y_stack=y_stack,
            protocol=protocol,
            local_epochs=int(args.local_epochs),
            fl_lr=float(args.fl_lr),
            client_batch_size=int(client_batch_size),
            steps=args.steps,
            attack_lr=args.attack_lr,
            top10norms=args.top10norms,
            mode=f"{fl_mode}__std_atk",
        )
        modes[f"{fl_mode}__vec_atk"] = _attack_trace_vec(
            model=model,
            criterion=criterion,
            observed_grads=observed,
            x0=x0,
            y_stack=y_stack,
            protocol=protocol,
            local_epochs=int(args.local_epochs),
            fl_lr=float(args.fl_lr),
            client_batch_size=int(client_batch_size),
            steps=args.steps,
            attack_lr=args.attack_lr,
            top10norms=args.top10norms,
            mode=f"{fl_mode}__vec_atk",
        )

    comparisons = {
        "attack_path_with_std_fl": _mode_diff(modes["std_fl__std_atk"], modes["std_fl__vec_atk"]),
        "attack_path_with_vec_fl": _mode_diff(modes["vec_fl__std_atk"], modes["vec_fl__vec_atk"]),
        "fl_path_with_std_atk": _mode_diff(modes["std_fl__std_atk"], modes["vec_fl__std_atk"]),
        "fl_path_with_vec_atk": _mode_diff(modes["std_fl__vec_atk"], modes["vec_fl__vec_atk"]),
    }

    summary = {
        "config": {
            "seed": int(args.seed),
            "protocol": str(protocol),
            "num_clients": int(args.num_clients),
            "local_epochs": int(args.local_epochs),
            "fl_lr": float(args.fl_lr),
            "steps": int(args.steps),
            "attack_lr": float(args.attack_lr),
            "top10norms": bool(args.top10norms),
            "dataset_path": run_cfg.dataset_cfg.dataset_path,
            "dataset_meta_path": run_cfg.dataset_cfg.dataset_meta_path,
            "batch_size": int(run_cfg.dataset_cfg.batch_size),
            "model_preset": "small",
        },
        "observed_gradient_equivalence_std_vs_vec_fl": asdict(obs_diff),
        "pairwise_mode_diffs": {name: asdict(diff) for name, diff in comparisons.items()},
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
