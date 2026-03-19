import logging
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TABULAR_GIA_ROOT = Path(__file__).resolve().parents[1]

from configs.dataset.dataset import DatasetConfig
from configs.base import BaseConfig
from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig
from configs.model.model import ModelConfig
from fl.dataloader.tabular_dataloader import load_dataset
from helper.summary_aggregation import (
    RunSummaryBuilder,
)
from tabular_gia.attack.attack_engine import AttackEngine
from tabular_gia.fl.fl_trainer import FLCallbacks, build_fl_trainer
from tabular_gia.model.model_wrapper import ModelWrapper
from tabular_gia.helper.results_writer import RunCsvWriter


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


@dataclass(frozen=True)
class RunConfig:
    base_cfg: BaseConfig
    dataset_cfg: DatasetConfig
    model_cfg: ModelConfig
    fl_cfg: FedAvgConfig | FedSGDConfig
    gia_cfg: GiaConfig
    results_dir: Path
    fl_only: bool


@dataclass(frozen=True)
class Runtime:
    model_wrapper: ModelWrapper
    feature_schema: dict
    client_dataloaders: list[DataLoader]
    val_loader: DataLoader
    test_loader: DataLoader


def _count_model_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def build_runtime(config: RunConfig) -> Runtime:
    num_clients = config.fl_cfg.num_clients
    max_client_dataset_examples = None
    if config.base_cfg.protocol == "fedavg" and isinstance(config.fl_cfg, FedAvgConfig):
        max_client_dataset_examples = config.fl_cfg.max_client_dataset_examples

    logger.debug("Model config: %s", config.model_cfg)
    encoding_mode = ModelWrapper.infer_encoding_mode(config.model_cfg)
    dataset_cfg = deepcopy(config.dataset_cfg)
    dataset_path = Path(dataset_cfg.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = TABULAR_GIA_ROOT / dataset_path
    dataset_cfg.dataset_path = str(dataset_path)

    dataset_meta_path = Path(dataset_cfg.dataset_meta_path)
    if not dataset_meta_path.is_absolute():
        dataset_meta_path = TABULAR_GIA_ROOT / dataset_meta_path
    dataset_cfg.dataset_meta_path = str(dataset_meta_path)
    logger.info(
        "Runtime setup: protocol=%s seed=%d fl_only=%s dataset=%s model_preset=%s encoding=%s num_clients=%d",
        config.base_cfg.protocol,
        config.base_cfg.seed,
        config.fl_only,
        dataset_path,
        config.model_cfg.preset,
        encoding_mode,
        num_clients,
    )

    client_dataloaders, val_loader, test_loader, feature_schema = load_dataset(
        dataset_cfg=dataset_cfg,
        num_clients=num_clients,
        seed=config.base_cfg.seed,
        encoding_mode=encoding_mode,
        max_client_dataset_examples=max_client_dataset_examples,
    )

    model_wrapper = ModelWrapper.from_config(
        model_cfg=config.model_cfg,
        feature_schema=feature_schema,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device)
    if device.type == "cuda":
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(device))
    else:
        logger.info("Using CPU")
    total_params, trainable_params = _count_model_params(model_wrapper)
    logger.info(
        "Model footprint: total_params=%d trainable_params=%d device=%s",
        total_params,
        trainable_params,
        device.type,
    )

    return Runtime(
        model_wrapper=model_wrapper,
        feature_schema=feature_schema,
        client_dataloaders=client_dataloaders,
        val_loader=val_loader,
        test_loader=test_loader,
    )


@dataclass(frozen=True)
class RunResult:
    fl_rows: list[dict]
    attack_rows: list[dict]
    round_summaries: list[dict]
    run_summary: dict | None


class RunEngine:
    def __init__(self, config: RunConfig, runtime: Runtime) -> None:
        self.config = config
        self.runtime = runtime
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_callbacks(
        self,
        runtime: Runtime,
        fl_cfg: FedAvgConfig | FedSGDConfig,
    ) -> tuple[FLCallbacks, RunSummaryBuilder, RunCsvWriter, AttackEngine | None]:
        csv_sink = RunCsvWriter(self.results_dir)
        summary_builder = RunSummaryBuilder()
        if self.config.fl_only:
            return FLCallbacks(attack_init_fn=None, attack_fn=None), summary_builder, csv_sink, None

        attack_engine = AttackEngine(
            protocol=self.config.base_cfg.protocol,
            gia_cfg=self.config.gia_cfg,
            fl_cfg=fl_cfg,
            seed=self.config.base_cfg.seed,
            feature_schema=runtime.feature_schema,
            client_dataloaders=runtime.client_dataloaders,
            criterion=runtime.model_wrapper.criterion,
            results_dir=self.results_dir,
        )
        callbacks = FLCallbacks(
            attack_init_fn=attack_engine.on_attack_init,
            attack_fn=attack_engine.on_attack,
        )
        return callbacks, summary_builder, csv_sink, attack_engine

    def _run_fl(self, runtime: Runtime, fl_cfg: FedAvgConfig | FedSGDConfig, callbacks: FLCallbacks):
        trainer = build_fl_trainer(
            protocol=self.config.base_cfg.protocol,
            fl_cfg=fl_cfg,
            batch_size=self.config.dataset_cfg.batch_size,
            client_dataloaders=runtime.client_dataloaders,
            val_loader=runtime.val_loader,
            test_loader=runtime.test_loader,
            callbacks=callbacks,
        )
        return trainer.fit(runtime.model_wrapper)

    def run(self) -> RunResult:
        logger.info(
            "Run start: protocol=%s seed=%d fl_only=%s results_dir=%s",
            self.config.base_cfg.protocol,
            self.config.base_cfg.seed,
            self.config.fl_only,
            self.results_dir,
        )
        logger.debug("RunConfig: %s", self.config)
        fl_cfg = self.config.fl_cfg
        logger.info(
            "FL config: num_clients=%d local_steps=%s local_epochs=%d optimizer=%s lr=%.6g",
            fl_cfg.num_clients,
            fl_cfg.local_steps,
            fl_cfg.local_epochs,
            fl_cfg.optimizer,
            fl_cfg.lr,
        )
        if isinstance(fl_cfg, FedAvgConfig):
            logger.info(
                "FedAvg extras: max_client_dataset_examples=%d",
                fl_cfg.max_client_dataset_examples,
            )
        if isinstance(fl_cfg, FedSGDConfig):
            logger.info(
                "FedSGD extras: vectorized_clients=%s",
                fl_cfg.vectorized_clients,
            )
        if not self.config.fl_only:
            gia_inv = self.config.gia_cfg.invertingconfig
            logger.info(
                "Attack config: mode=%s schedule=%s at_iterations=%d attack_lr=%.6g label_known=%s",
                self.config.gia_cfg.attack_mode,
                self.config.gia_cfg.attack_schedule,
                gia_inv.at_iterations,
                gia_inv.attack_lr,
                gia_inv.label_known,
            )

        callbacks, summary_builder, csv_sink, attack_engine = self._build_callbacks(self.runtime, fl_cfg)
        fl_result = self._run_fl(self.runtime, fl_cfg, callbacks)

        for fl_row in fl_result.fl_rows:
            csv_sink.write_fl(fl_row)

        attack_rows = [] if attack_engine is None else attack_engine.get_attack_rows()
        for attack_row in attack_rows:
            csv_sink.write_attack(attack_row)

        round_summaries = summary_builder.build_round_summaries(attack_rows)
        run_summary = summary_builder.build_run_summary(round_summaries)
        csv_sink.write_round_summaries(round_summaries)
        csv_sink.write_run_summary(run_summary)
        return RunResult(
            fl_rows=list(fl_result.fl_rows),
            attack_rows=list(attack_rows),
            round_summaries=list(round_summaries),
            run_summary=run_summary,
        )
