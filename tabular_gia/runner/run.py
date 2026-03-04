import csv
import logging
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl.dataloader.tabular_dataloader import load_dataset
from fl.metrics.fl_metrics import FL_METRIC_FIELDS
from fl.fedavg import run_fedavg
from fl.fedsgd import run_fedsgd
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep
from leakpro.utils.seed import restore_rng_state
from metrics.tabular_metrics import (
    ATTACK_METRIC_FIELDS,
    ROUNDS_SUMMARY_CSV_FIELDS,
    RUN_SUMMARY_CSV_FIELDS,
    compute_reconstruction_metrics,
    prepare_tensors_for_metrics,
    summarize_round,
    summarize_run,
    write_debug_reconstruction_txt,
    write_rounds_summary,
    write_run_summary,
)
from model.model import TabularMLP


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    dataset_cfg: dict
    model_cfg: dict
    fl_cfg: dict
    gia_cfg: dict
    results_dir: Path
    fl_only: bool


@dataclass(frozen=True)
class Runtime:
    model: torch.nn.Module
    criterion: torch.nn.Module
    feature_schema: dict
    client_dataloaders: list[DataLoader]
    val_loader: DataLoader
    test_loader: DataLoader


@dataclass(frozen=True)
class Callbacks:
    attack_fn: object
    round_summary_fn: object
    num_rounds_fn: object
    exposure_progress_fn: object
    fl_metrics_fn: object


ATTACKS_CSV_FIELDS = (
    # identity/context fields
    "attack_id",
    "round",
    "client_idx",
    "attack_mode",
    "fixed_batch_id",
    "checkpoint_type",
    "checkpoint_label",
    "exp_min",
    "exp_avg",
    "exp_max",
    "client_exp",
    "row_count",
    # shared metric fields
    *ATTACK_METRIC_FIELDS,
    # "artifact_path",
)

FL_CSV_FIELDS = (
    "phase",
    "round",
    "exp_min",
    "exp_avg",
    "exp_max",
    *(f"{split}_{metric}" for split in ("train", "val", "test") for metric in FL_METRIC_FIELDS),
)

class MetricsCsvWriter:
    def __init__(self, results_dir: Path) -> None:
        self.fl_path = results_dir / "fl.csv"
        self.attacks_path = results_dir / "attacks.csv"

    def _write_row(self, path: Path, payload: dict, fieldnames: tuple[str, ...] | None = None) -> None:
        write_header = not path.exists() or path.stat().st_size == 0
        active_fieldnames = list(fieldnames) if fieldnames is not None else list(payload.keys())
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=active_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(payload)

    def write_fl(self, payload: dict) -> None:
        unknown = sorted(set(payload.keys()) - set(FL_CSV_FIELDS))
        if unknown:
            raise ValueError(f"Unexpected fl.csv fields: {unknown}")
        ordered_payload = {field: payload.get(field, "") for field in FL_CSV_FIELDS}
        self._write_row(self.fl_path, ordered_payload, FL_CSV_FIELDS)

    def write_attack(self, payload: dict) -> None:
        unknown = sorted(set(payload.keys()) - set(ATTACKS_CSV_FIELDS))
        if unknown:
            raise ValueError(f"Unexpected attacks.csv fields: {unknown}")
        ordered_payload = {field: payload.get(field, "") for field in ATTACKS_CSV_FIELDS}
        self._write_row(self.attacks_path, ordered_payload, ATTACKS_CSV_FIELDS)


class SummaryCollector:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.round_summaries: list[dict] = []

    def on_round(self, round_idx: int, metrics_list: list[dict]) -> None:
        if not metrics_list:
            return
        metric_rows = [
            {"row_count": row["row_count"], **{k: row[k] for k in ROUNDS_SUMMARY_CSV_FIELDS if k in row}}
            for row in metrics_list
        ]
        round_summary = summarize_round(metric_rows, round_idx)
        if round_summary is not None:
            self.round_summaries.append(round_summary)

    def finalize(self) -> dict | None:
        write_rounds_summary(self.results_dir, self.round_summaries)
        run_summary = summarize_run(self.round_summaries)
        if run_summary is not None:
            run_summary = {field: run_summary[field] for field in RUN_SUMMARY_CSV_FIELDS if field in run_summary}
            write_run_summary(self.results_dir, run_summary)
        return run_summary


class AttackScheduler:
    def __init__(
        self,
        protocol: str,
        gia_cfg: dict,
        fl_cfg: dict,
        dataset_cfg: dict,
        client_dataloaders: list[DataLoader],
    ) -> None:
        self.protocol = protocol
        self.gia_cfg = gia_cfg
        self.fl_cfg = fl_cfg
        self.dataset_cfg = dataset_cfg
        self.client_dataloaders = client_dataloaders

        self.attack_mode = str(gia_cfg["attack_mode"]).strip().lower()
        self.attack_schedule = str(gia_cfg["attack_schedule"]).strip().lower()
        self.fixed_batch_k = int(gia_cfg["fixed_batch_k"])

        self.exposure_include_round1_preagg = False
        self.exposure_targets: list[float] = []
        if self.attack_schedule == "exposure":
            raw = gia_cfg["attack_exposure_milestones"]
            self.exposure_include_round1_preagg = any(float(m) == 0.0 for m in raw)
            self.exposure_targets = sorted(
                {
                    float(m)
                    for m in raw
                    if float(m) > 0.0
                }
            )

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
            raw = self.gia_cfg["attack_rounds"]
            return {int(r) for r in raw if 1 <= int(r) <= total_rounds}
        if self.attack_schedule in {"log", "logspace"}:
            count = int(self.gia_cfg["attack_num_checkpoints"])
            if count == 1:
                return {total_rounds}
            rounds = {
                int(round(math.exp((i / (count - 1)) * math.log(total_rounds))))
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
        if self.attack_schedule != "exposure":
            return
        if self.checkpoint_rounds is None:
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
            merged = existing + [m for m in crossed if m not in existing]
            self.exposure_round_labels[int(round_idx)] = merged

    def checkpoint(self, round_idx: int) -> tuple[str, str] | None:
        if self.attack_schedule != "all":
            if self.checkpoint_rounds is None:
                return None
            if int(round_idx) not in self.checkpoint_rounds:
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
        local_steps_raw = self.fl_cfg["local_steps"]
        local_steps_all = isinstance(local_steps_raw, str) and local_steps_raw.strip().lower() == "all"
        local_steps = None if local_steps_all else max(1, int(local_steps_raw))
        base_seed = int(self.dataset_cfg["seed"])

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
                k_units=int(self.fixed_batch_k),
                seed=seed,
            )
            self.fixed_batch_loaders[client_idx] = [self._build_fixed_batch_loader(client_loader, ids) for ids in unit_sample_ids]


class AttackRunner:
    def __init__(
        self,
        protocol: str,
        train_fn: object,
        mismatch_label: str,
        attack_cfg_base: InvertingConfig,
        feature_schema: dict,
        results_dir: Path,
        csv_sink: MetricsCsvWriter,
        attack_mode: str,
    ) -> None:
        self.protocol = protocol
        self.train_fn = train_fn
        self.mismatch_label = mismatch_label
        self.attack_cfg_base = attack_cfg_base
        self.feature_schema = feature_schema
        self.results_dir = results_dir
        self.debug_dir = results_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.csv_sink = csv_sink
        self.attack_mode = attack_mode
        self.attack_id_counter = 0

    def _assert_update_match(self, expected_updates: list, attack_updates: list) -> None:
        if len(expected_updates) != len(attack_updates):
            raise AssertionError(
                f"{self.mismatch_label} length mismatch: client={len(expected_updates)} attacker={len(attack_updates)}"
            )
        for idx, (a, b) in enumerate(zip(expected_updates, attack_updates)):
            if a is None or b is None:
                continue
            if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                raise AssertionError(f"{self.mismatch_label} mismatch at param {idx}")

    def run_attack(
        self,
        att_model,
        batch_loader,
        round_idx: int,
        client_idx: int,
        client_updates,
        rng_pre,
        rng_post,
        fixed_batch_id: int,
        checkpoint_type: str,
        checkpoint_label: str,
        attack_context: dict | None,
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
        )
        if rng_pre is not None:
            attacker.replay_rng_state = rng_pre
        attacker.prepare_attack()
        if rng_post is not None:
            restore_rng_state(rng_post)

        if client_updates is not None:
            self._assert_update_match(client_updates, attacker.client_gradient)

        total_iters = int(attack_cfg.at_iterations or 0)
        last_i = -1
        with tqdm(total=total_iters, desc=f"GIA Round {round_idx} Client {client_idx}", leave=False) as bar:
            for i, score, _ in attacker.run_attack():
                step = max(0, int(i) - last_i)
                if step:
                    bar.update(step)
                last_i = int(i)
                loss = -float(score) if score is not None else float("nan")
                best_loss = float(attacker.best_loss)
                bar.set_postfix(loss=f"{loss:.6e}", best=f"{best_loss:.6e}")
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
        if attack_context is not None:
            metrics["exp_min"] = float(attack_context["exp_min"])
            metrics["exp_avg"] = float(attack_context["exp_avg"])
            metrics["exp_max"] = float(attack_context["exp_max"])
            metrics["client_exp"] = float(attack_context["client_exp"])
        metrics["checkpoint_type"] = checkpoint_type
        metrics["checkpoint_label"] = checkpoint_label
        self.csv_sink.write_attack(metrics)

        if fixed_batch_id >= 0:
            tqdm.write(
                f"GIA done: round={round_idx} client={client_idx} fixed_batch_id={fixed_batch_id} "
                f"best={float(attacker.best_loss):.6e}"
            )
        else:
            tqdm.write(f"GIA done: round={round_idx} client={client_idx} best={float(attacker.best_loss):.6e}")
        return metrics


class RunEngine:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.results_dir = spec.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_runtime(self) -> Runtime:
        fl_cfg = deepcopy(self.spec.fl_cfg)
        num_clients = int(fl_cfg["num_clients"])
        dataset_cfg = deepcopy(self.spec.dataset_cfg)

        client_dataloaders, val_loader, test_loader, feature_schema = load_dataset(
            **dataset_cfg,
            num_clients=num_clients,
        )

        task = feature_schema["task"]
        if task == "binary":
            criterion = torch.nn.BCEWithLogitsLoss()
            out_dim = 1
        elif task == "multiclass":
            criterion = torch.nn.CrossEntropyLoss()
            out_dim = feature_schema["num_classes"]
        else:
            criterion = torch.nn.MSELoss()
            out_dim = 1

        model_cfg = deepcopy(self.spec.model_cfg)
        if "use_preset" in model_cfg:
            preset_name = model_cfg["use_preset"]
            if preset_name is not None:
                presets = model_cfg["presets"]
                if preset_name not in presets:
                    raise ValueError(f"Model preset '{preset_name}' not found in model config.")
                model_cfg = presets[preset_name]

        model = TabularMLP(
            d_in=feature_schema["num_features"],
            d_out=out_dim,
            **model_cfg,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if device.type == "cuda":
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(device))
        else:
            logger.info("Using CPU")

        return Runtime(
            model=model,
            criterion=criterion,
            feature_schema=feature_schema,
            client_dataloaders=client_dataloaders,
            val_loader=val_loader,
            test_loader=test_loader,
        )

    def _build_attack_cfg(self, criterion: torch.nn.Module, fl_cfg: dict) -> InvertingConfig:
        lr = float(fl_cfg["lr"])
        optimizer_name = fl_cfg["optimizer"]
        optimizer = MetaAdam(lr=lr) if optimizer_name == "MetaAdam" else MetaSGD(lr=lr)

        invertingconfig = dict(self.spec.gia_cfg["invertingconfig"])
        invertingconfig.pop("data_extension", None)
        return InvertingConfig(
            optimizer=optimizer,
            criterion=criterion,
            data_extension=GiaTabularExtension(),
            epochs=fl_cfg["local_epochs"],
            **invertingconfig,
        )

    def _build_callbacks(self, runtime: Runtime, fl_cfg: dict) -> tuple[Callbacks, SummaryCollector]:
        csv_sink = MetricsCsvWriter(self.results_dir)
        summary_collector = SummaryCollector(self.results_dir)
        if self.spec.fl_only:
            callbacks = Callbacks(
                attack_fn=None,
                round_summary_fn=summary_collector.on_round,
                num_rounds_fn=None,
                exposure_progress_fn=None,
                fl_metrics_fn=csv_sink.write_fl,
            )
            return callbacks, summary_collector

        attack_cfg_base = self._build_attack_cfg(runtime.criterion, fl_cfg)

        if self.spec.protocol == "fedsgd":
            attack_train_fn = train_nostep
            mismatch_label = "Gradient"
        else:
            attack_train_fn = train
            mismatch_label = "Delta"

        scheduler = AttackScheduler(
            protocol=self.spec.protocol,
            gia_cfg=self.spec.gia_cfg,
            fl_cfg=fl_cfg,
            dataset_cfg=self.spec.dataset_cfg,
            client_dataloaders=runtime.client_dataloaders,
        )

        attack_runner = AttackRunner(
            protocol=self.spec.protocol,
            train_fn=attack_train_fn,
            mismatch_label=mismatch_label,
            attack_cfg_base=attack_cfg_base,
            feature_schema=runtime.feature_schema,
            results_dir=self.results_dir,
            csv_sink=csv_sink,
            attack_mode=scheduler.attack_mode,
        )

        def attack_fn(att_model, batch_loader, round_idx, client_idx, client_updates, rng_pre, rng_post, attack_context):
            checkpoint = scheduler.checkpoint(int(round_idx))
            if checkpoint is None:
                return None
            checkpoint_type, checkpoint_label = checkpoint

            if scheduler.attack_mode == "fixed_batch":
                metrics_list = []
                fixed_batch_loaders = scheduler.fixed_batch_loaders[int(client_idx)]
                for fixed_batch_id, fixed_batch_loader in enumerate(fixed_batch_loaders):
                    metrics = attack_runner.run_attack(
                        att_model=att_model,
                        batch_loader=fixed_batch_loader,
                        round_idx=int(round_idx),
                        client_idx=int(client_idx),
                        client_updates=None,
                        rng_pre=None,
                        rng_post=None,
                        fixed_batch_id=int(fixed_batch_id),
                        checkpoint_type=checkpoint_type,
                        checkpoint_label=checkpoint_label,
                        attack_context=attack_context,
                    )
                    metrics_list.append(metrics)
                return metrics_list

            return attack_runner.run_attack(
                att_model=att_model,
                batch_loader=batch_loader,
                round_idx=int(round_idx),
                client_idx=int(client_idx),
                client_updates=client_updates,
                rng_pre=rng_pre,
                rng_post=rng_post,
                fixed_batch_id=-1,
                checkpoint_type=checkpoint_type,
                checkpoint_label=checkpoint_label,
                attack_context=attack_context,
            )

        callbacks = Callbacks(
            attack_fn=attack_fn,
            round_summary_fn=summary_collector.on_round,
            num_rounds_fn=scheduler.on_num_rounds,
            exposure_progress_fn=scheduler.on_exposure_progress,
            fl_metrics_fn=csv_sink.write_fl,
        )
        return callbacks, summary_collector

    def _run_fl(self, runtime: Runtime, fl_cfg: dict, callbacks: Callbacks) -> None:
        fl_kwargs = {
            "cfg": fl_cfg,
            "global_model": runtime.model,
            "criterion": runtime.criterion,
            "attack_fn": callbacks.attack_fn,
            "client_dataloaders": runtime.client_dataloaders,
            "val": runtime.val_loader,
            "test": runtime.test_loader,
            "round_summary_fn": callbacks.round_summary_fn,
            "num_rounds_fn": callbacks.num_rounds_fn,
            "exposure_progress_fn": callbacks.exposure_progress_fn,
            "fl_metrics_fn": callbacks.fl_metrics_fn,
        }

        if self.spec.protocol == "fedsgd":
            run_fedsgd(**fl_kwargs)
            return

        if self.spec.protocol == "fedavg":
            def optimizer_fn(name: str | None, lr: float):
                if name == "MetaAdam":
                    return MetaAdam(lr=lr)
                return MetaSGD(lr=lr)

            run_fedavg(optimizer_fn=optimizer_fn, **fl_kwargs)
            return

        raise ValueError(f"Unknown protocol '{self.spec.protocol}'")

    def run(self) -> dict | None:
        fl_cfg = deepcopy(self.spec.fl_cfg)
        fl_cfg["batch_size"] = int(self.spec.dataset_cfg["batch_size"])

        runtime = self._build_runtime()
        callbacks, summary_collector = self._build_callbacks(runtime, fl_cfg)
        self._run_fl(runtime, fl_cfg, callbacks)
        return summary_collector.finalize()
