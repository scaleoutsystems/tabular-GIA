from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABULAR_GIA_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, TABULAR_GIA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.base import BaseConfig
from configs.dataset.dataset import DatasetConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig
from configs.model.model import ModelConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train_nostep
from leakpro.utils.seed import seed_everything
from tabular_gia.runner.run import RunConfig, build_runtime


@dataclass(frozen=True)
class GradStats:
    total_l2: float
    num_l2: float
    cat_l2: float
    total_abs_mean: float
    num_abs_mean: float
    cat_abs_mean: float
    num_nonzero_frac: float
    cat_nonzero_frac: float
    has_nan: bool


@dataclass(frozen=True)
class ClientResult:
    client_idx: int
    rows: int
    gia_dim: int
    num_dim: int
    cat_dim: int
    loss_before: float
    loss_after_one_step: float
    loss_after_k_steps: float
    one_step_delta: float
    k_step_delta: float
    grad_stats: GradStats


@dataclass(frozen=True)
class BatchSizeResult:
    batch_size: int
    parameter_settings: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation for FTTransformer GIA differentiability across Adult batch sizes."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="Adult client batch sizes to test.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument(
        "--gia-forward-modes",
        type=str,
        nargs="+",
        default=["logits"],
        help="FTTransformer GIA forward modes to test. Supported: logits, probabilities.",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0, 2.0],
        help="FTTransformer GIA softmax temperatures to test.",
    )
    parser.add_argument(
        "--logit-scales",
        type=float,
        nargs="+",
        default=[1.0, 2.5, 5.0, 10.0, 20.0],
        help="FTTransformer GIA initialization logit scales to test.",
    )
    parser.add_argument(
        "--client-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Client indices whose first local batch should be tested.",
    )
    parser.add_argument(
        "--attack-lr",
        type=float,
        default=0.06,
        help="Attack optimizer LR for the differentiability check.",
    )
    parser.add_argument(
        "--attack-steps",
        type=int,
        default=5,
        help="Number of attack steps for the loss decrease check.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def _configure_repro(seed: int) -> None:
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _adult_run_config(*, seed: int, batch_size: int, num_clients: int) -> RunConfig:
    base_cfg = BaseConfig(seed=int(seed), protocol="fedsgd")
    dataset_cfg = DatasetConfig(
        dataset_path="data/binary/adult/adult.csv",
        dataset_meta_path="data/binary/adult/adult.yaml",
        batch_size=int(batch_size),
    )
    model_cfg = ModelConfig(preset="fttransformer")
    fl_cfg = FedSGDConfig(num_clients=int(num_clients), local_steps=1, local_epochs=1)
    gia_cfg = GiaConfig()
    return RunConfig(
        base_cfg=base_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        fl_cfg=fl_cfg,
        gia_cfg=gia_cfg,
        results_dir=Path("/tmp/tabular_gia_fttransformer_gia_ablation"),
        fl_only=False,
    )


def _make_single_batch_loader(loader: DataLoader) -> DataLoader:
    xb, yb = next(iter(loader))
    return DataLoader(TensorDataset(xb, yb), batch_size=int(xb.shape[0]), shuffle=False)


def _grad_stats(grad: torch.Tensor, num_dim: int) -> GradStats:
    if grad is None:
        raise ValueError("Expected reconstruction gradient to be populated.")
    grad_detached = grad.detach()
    total_flat = grad_detached.reshape(-1)
    num_grad = grad_detached[:, :num_dim] if num_dim > 0 else grad_detached[:, :0]
    cat_grad = grad_detached[:, num_dim:]
    threshold = 1e-12
    return GradStats(
        total_l2=float(total_flat.norm().item()),
        num_l2=float(num_grad.reshape(-1).norm().item()) if num_grad.numel() else 0.0,
        cat_l2=float(cat_grad.reshape(-1).norm().item()) if cat_grad.numel() else 0.0,
        total_abs_mean=float(total_flat.abs().mean().item()),
        num_abs_mean=float(num_grad.abs().mean().item()) if num_grad.numel() else 0.0,
        cat_abs_mean=float(cat_grad.abs().mean().item()) if cat_grad.numel() else 0.0,
        num_nonzero_frac=float((num_grad.abs() > threshold).float().mean().item()) if num_grad.numel() else 0.0,
        cat_nonzero_frac=float((cat_grad.abs() > threshold).float().mean().item()) if cat_grad.numel() else 0.0,
        has_nan=bool(torch.isnan(grad_detached).any().item()),
    )


def _evaluate_client(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    client_loader: DataLoader,
    fl_lr: float,
    attack_lr: float,
    attack_steps: int,
) -> ClientResult:
    batch_loader = _make_single_batch_loader(client_loader)
    observed_client_gradient = train_nostep(
        model,
        batch_loader,
        MetaSGD(lr=float(fl_lr)),
        criterion,
        epochs=1,
    )
    observed_client_gradient = [g.detach().clone() if g is not None else None for g in observed_client_gradient]

    attack_cfg = InvertingConfig(
        attack_lr=float(attack_lr),
        at_iterations=max(1, int(attack_steps)),
        optimizer=MetaSGD(lr=float(fl_lr)),
        criterion=criterion,
        data_extension=GiaTabularExtension(),
        epochs=1,
        label_known=True,
    )
    attacker = InvertingGradients(
        model,
        batch_loader,
        None,
        None,
        train_fn=train_nostep,
        configs=attack_cfg,
        observed_client_gradient=observed_client_gradient,
    )
    attacker.prepare_attack()
    if not hasattr(model, "n_num_features"):
        raise ValueError("This ablation expects FTTransformerWrapper with n_num_features.")
    num_dim = int(model.n_num_features)
    gia_dim = int(attacker.reconstruction.shape[1])
    cat_dim = max(0, gia_dim - num_dim)

    reconstruction = attacker.reconstruction
    reconstruction.grad = None
    attacker.model.zero_grad(set_to_none=True)
    loss_before = attacker._tabular_reconstruction_loss()
    loss_before.backward()
    grad = reconstruction.grad
    stats = _grad_stats(grad, num_dim)
    loss_before_value = float(loss_before.detach().item())

    optimizer = torch.optim.Adam([reconstruction], lr=float(attack_lr))
    optimizer.step()
    reconstruction.grad = None
    attacker.model.zero_grad(set_to_none=True)
    loss_after_one = attacker._tabular_reconstruction_loss()
    loss_after_one_value = float(loss_after_one.detach().item())

    for _ in range(max(0, int(attack_steps) - 1)):
        optimizer.zero_grad(set_to_none=True)
        attacker.model.zero_grad(set_to_none=True)
        loss = attacker._tabular_reconstruction_loss()
        loss.backward()
        optimizer.step()

    reconstruction.grad = None
    attacker.model.zero_grad(set_to_none=True)
    loss_after_k = attacker._tabular_reconstruction_loss()
    loss_after_k_value = float(loss_after_k.detach().item())

    return ClientResult(
        client_idx=-1,
        rows=int(reconstruction.shape[0]),
        gia_dim=gia_dim,
        num_dim=num_dim,
        cat_dim=cat_dim,
        loss_before=loss_before_value,
        loss_after_one_step=loss_after_one_value,
        loss_after_k_steps=loss_after_k_value,
        one_step_delta=loss_after_one_value - loss_before_value,
        k_step_delta=loss_after_k_value - loss_before_value,
        grad_stats=stats,
    )


def main() -> None:
    args = parse_args()
    _configure_repro(int(args.seed))

    results: list[BatchSizeResult] = []
    for batch_size in [int(bs) for bs in args.batch_sizes]:
        run_cfg = _adult_run_config(seed=int(args.seed), batch_size=batch_size, num_clients=int(args.num_clients))
        runtime = build_runtime(run_cfg)
        model = runtime.model_wrapper
        model.eval()
        criterion = model.criterion
        parameter_settings: list[dict] = []
        for gia_forward_mode in [str(v).strip().lower() for v in args.gia_forward_modes]:
            if gia_forward_mode not in {"logits", "probabilities"}:
                raise ValueError(f"Unsupported gia_forward_mode '{gia_forward_mode}'.")
            mode_settings: list[tuple[float | None, float | None]]
            if gia_forward_mode == "probabilities":
                mode_settings = [(None, None)]
            else:
                mode_settings = [
                    (float(temperature), float(logit_scale))
                    for temperature in args.temperatures
                    for logit_scale in args.logit_scales
                ]
            for temperature, logit_scale in mode_settings:
                model.gia_forward_mode = gia_forward_mode
                if temperature is not None:
                    model.gia_soft_temperature = float(temperature)
                if logit_scale is not None:
                    model.gia_init_logit_scale = float(logit_scale)
                client_results: list[ClientResult] = []
                for client_idx in args.client_indices:
                    _configure_repro(int(args.seed))
                    client_idx = int(client_idx)
                    if client_idx >= len(runtime.client_dataloaders):
                        raise IndexError(f"Client index {client_idx} out of range for {len(runtime.client_dataloaders)} clients.")
                    result = _evaluate_client(
                        model=model,
                        criterion=criterion,
                        client_loader=runtime.client_dataloaders[client_idx],
                        fl_lr=float(run_cfg.fl_cfg.lr),
                        attack_lr=float(args.attack_lr),
                        attack_steps=int(args.attack_steps),
                    )
                    client_results.append(
                        ClientResult(
                            client_idx=client_idx,
                            rows=result.rows,
                            gia_dim=result.gia_dim,
                            num_dim=result.num_dim,
                            cat_dim=result.cat_dim,
                            loss_before=result.loss_before,
                            loss_after_one_step=result.loss_after_one_step,
                            loss_after_k_steps=result.loss_after_k_steps,
                            one_step_delta=result.one_step_delta,
                            k_step_delta=result.k_step_delta,
                            grad_stats=result.grad_stats,
                        )
                    )
                parameter_settings.append(
                    {
                        "gia_forward_mode": gia_forward_mode,
                        "gia_soft_temperature": None if temperature is None else float(temperature),
                        "gia_init_logit_scale": None if logit_scale is None else float(logit_scale),
                        "clients": [asdict(item) for item in client_results],
                    }
                )
        results.append(BatchSizeResult(batch_size=batch_size, parameter_settings=parameter_settings))

    summary = {
        "dataset": "adult",
        "model": "fttransformer",
        "seed": int(args.seed),
        "attack_lr": float(args.attack_lr),
        "attack_steps": int(args.attack_steps),
        "gia_forward_modes": [str(v).strip().lower() for v in args.gia_forward_modes],
        "temperatures": [float(v) for v in args.temperatures],
        "logit_scales": [float(v) for v in args.logit_scales],
        "results": [asdict(item) for item in results],
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
