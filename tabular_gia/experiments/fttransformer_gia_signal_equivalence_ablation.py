from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Literal

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep
from leakpro.utils.seed import seed_everything
from tabular_gia.runner.run import RunConfig, build_runtime


Protocol = Literal["fedsgd", "fedavg"]


@dataclass(frozen=True)
class SignalStats:
    n_clients: int
    n_tensors: int
    n_none_tensors: int
    n_values: int
    max_abs_error: float
    mean_abs_error: float
    relative_l2_error: float
    cosine_similarity: float
    native_l2: float
    surrogate_l2: float


@dataclass(frozen=True)
class EquivalenceRow:
    protocol: str
    batch_size: int
    gia_forward_mode: str
    gia_soft_temperature: float | None
    gia_init_logit_scale: float | None
    stats: SignalStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native FTTransformer training gradients/model deltas with the "
            "differentiable GIA forward surrogate on the same observed batch."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--client-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--protocols", type=str, nargs="+", default=["fedsgd", "fedavg"], choices=["fedsgd", "fedavg"])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    parser.add_argument("--logit-scales", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--include-probabilities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fl-lr", type=float, default=0.01)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def _configure_repro(seed: int) -> None:
    seed_everything(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _adult_run_config(*, seed: int, protocol: Protocol, batch_size: int, num_clients: int, fl_lr: float) -> RunConfig:
    base_cfg = BaseConfig(seed=int(seed), protocol=protocol)
    dataset_cfg = DatasetConfig(
        dataset_path="data/binary/adult/adult.csv",
        dataset_meta_path="data/binary/adult/adult.yaml",
        batch_size=int(batch_size),
    )
    model_cfg = ModelConfig(preset="fttransformer")
    if protocol == "fedavg":
        fl_cfg = FedAvgConfig(
            local_steps=1,
            local_epochs=1,
            max_client_dataset_examples=None,
            num_clients=int(num_clients),
            lr=float(fl_lr),
        )
    else:
        fl_cfg = FedSGDConfig(
            local_steps=1,
            local_epochs=1,
            num_clients=int(num_clients),
            lr=float(fl_lr),
        )
    gia_cfg = GiaConfig()
    return RunConfig(
        base_cfg=base_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        fl_cfg=fl_cfg,
        gia_cfg=gia_cfg,
        results_dir=Path("/tmp/tabular_gia_fttransformer_signal_equivalence"),
        fl_only=False,
    )


def _single_batch(loader: DataLoader) -> tuple[Tensor, Tensor]:
    x, y = next(iter(loader))
    return x.detach().clone(), y.detach().clone()


def _single_batch_loader(x: Tensor, y: Tensor) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=int(x.shape[0]), shuffle=False)


def _training_signal(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    protocol: Protocol,
    fl_lr: float,
) -> list[Tensor | None]:
    loader = _single_batch_loader(x, y)
    if protocol == "fedavg":
        signal = train(model, loader, MetaSGD(lr=float(fl_lr)), criterion, epochs=1)
    else:
        signal = train_nostep(model, loader, MetaSGD(lr=float(fl_lr)), criterion, epochs=1)
    return [item.detach().cpu().clone() if item is not None else None for item in signal]


def _compare_signals(native_signals: list[list[Tensor | None]], surrogate_signals: list[list[Tensor | None]]) -> SignalStats:
    flat_native: list[Tensor] = []
    flat_surrogate: list[Tensor] = []
    n_tensors = 0
    n_none_tensors = 0
    for native_client, surrogate_client in zip(native_signals, surrogate_signals):
        for native_tensor, surrogate_tensor in zip(native_client, surrogate_client):
            if native_tensor is None and surrogate_tensor is None:
                n_none_tensors += 1
                continue
            if (native_tensor is None) != (surrogate_tensor is None):
                raise ValueError("Native/surrogate signal mismatch: only one side returned None.")
            assert native_tensor is not None
            assert surrogate_tensor is not None
            n_tensors += 1
            flat_native.append(native_tensor.reshape(-1).double())
            flat_surrogate.append(surrogate_tensor.reshape(-1).double())

    if not flat_native:
        return SignalStats(
            n_clients=len(native_signals),
            n_tensors=0,
            n_none_tensors=n_none_tensors,
            n_values=0,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            relative_l2_error=0.0,
            cosine_similarity=1.0,
            native_l2=0.0,
            surrogate_l2=0.0,
        )

    native_vec = torch.cat(flat_native)
    surrogate_vec = torch.cat(flat_surrogate)
    diff = native_vec - surrogate_vec
    native_l2 = torch.linalg.vector_norm(native_vec)
    surrogate_l2 = torch.linalg.vector_norm(surrogate_vec)
    diff_l2 = torch.linalg.vector_norm(diff)
    denom = torch.clamp(native_l2, min=torch.finfo(native_l2.dtype).tiny)
    cosine_denom = torch.clamp(native_l2 * surrogate_l2, min=torch.finfo(native_l2.dtype).tiny)
    cosine = torch.dot(native_vec, surrogate_vec) / cosine_denom
    return SignalStats(
        n_clients=len(native_signals),
        n_tensors=int(n_tensors),
        n_none_tensors=int(n_none_tensors),
        n_values=int(native_vec.numel()),
        max_abs_error=float(diff.abs().max().item()),
        mean_abs_error=float(diff.abs().mean().item()),
        relative_l2_error=float((diff_l2 / denom).item()),
        cosine_similarity=float(cosine.item()),
        native_l2=float(native_l2.item()),
        surrogate_l2=float(surrogate_l2.item()),
    )


def _evaluate_setting(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    client_dataloaders: list[DataLoader],
    client_indices: list[int],
    protocol: Protocol,
    batch_size: int,
    gia_forward_mode: str,
    temperature: float | None,
    logit_scale: float | None,
    fl_lr: float,
) -> EquivalenceRow:
    model.gia_forward_mode = gia_forward_mode
    if temperature is not None:
        model.gia_soft_temperature = float(temperature)
    if logit_scale is not None:
        model.gia_init_logit_scale = float(logit_scale)

    native_signals: list[list[Tensor | None]] = []
    surrogate_signals: list[list[Tensor | None]] = []
    for client_idx in client_indices:
        x_native, y = _single_batch(client_dataloaders[int(client_idx)])
        x_surrogate = model.to_gia_space(x_native)
        native_signals.append(
            _training_signal(
                model=model,
                criterion=criterion,
                x=x_native,
                y=y,
                protocol=protocol,
                fl_lr=fl_lr,
            )
        )
        surrogate_signals.append(
            _training_signal(
                model=model,
                criterion=criterion,
                x=x_surrogate,
                y=y,
                protocol=protocol,
                fl_lr=fl_lr,
            )
        )

    return EquivalenceRow(
        protocol=protocol,
        batch_size=int(batch_size),
        gia_forward_mode=gia_forward_mode,
        gia_soft_temperature=None if temperature is None else float(temperature),
        gia_init_logit_scale=None if logit_scale is None else float(logit_scale),
        stats=_compare_signals(native_signals, surrogate_signals),
    )


def main() -> None:
    args = parse_args()
    _configure_repro(int(args.seed))

    rows: list[EquivalenceRow] = []
    protocols: list[Protocol] = ["fedavg" if item == "fedavg" else "fedsgd" for item in args.protocols]
    client_indices = [int(item) for item in args.client_indices]
    for protocol in protocols:
        for batch_size in [int(item) for item in args.batch_sizes]:
            run_cfg = _adult_run_config(
                seed=int(args.seed),
                protocol=protocol,
                batch_size=batch_size,
                num_clients=int(args.num_clients),
                fl_lr=float(args.fl_lr),
            )
            runtime = build_runtime(run_cfg)
            model = runtime.model_wrapper
            model.eval()
            criterion = model.criterion

            for temperature in [float(item) for item in args.temperatures]:
                for logit_scale in [float(item) for item in args.logit_scales]:
                    rows.append(
                        _evaluate_setting(
                            model=model,
                            criterion=criterion,
                            client_dataloaders=runtime.client_dataloaders,
                            client_indices=client_indices,
                            protocol=protocol,
                            batch_size=batch_size,
                            gia_forward_mode="logits",
                            temperature=temperature,
                            logit_scale=logit_scale,
                            fl_lr=float(args.fl_lr),
                        )
                    )
            if bool(args.include_probabilities):
                rows.append(
                    _evaluate_setting(
                        model=model,
                        criterion=criterion,
                        client_dataloaders=runtime.client_dataloaders,
                        client_indices=client_indices,
                        protocol=protocol,
                        batch_size=batch_size,
                        gia_forward_mode="probabilities",
                        temperature=None,
                        logit_scale=None,
                        fl_lr=float(args.fl_lr),
                    )
                )

    summary = {
        "dataset": "adult",
        "model": "fttransformer",
        "seed": int(args.seed),
        "num_clients": int(args.num_clients),
        "client_indices": client_indices,
        "batch_sizes": [int(item) for item in args.batch_sizes],
        "protocols": list(args.protocols),
        "fl_lr": float(args.fl_lr),
        "rows": [asdict(row) for row in rows],
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
