import argparse
import logging
import yaml
from pathlib import Path
from copy import deepcopy
import torch
from tqdm import tqdm
from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD
from leakpro.fl_utils.gia_train import train_nostep, train
from leakpro.utils.seed import seed_everything, restore_rng_state

from fl.dataloader.tabular_dataloader import load_dataset
from fl.fedsgd import run_fedsgd
from fl.fedavg import run_fedavg

from tabular_metrics import evaluate_batch_rows, compute_reconstruction_metrics, write_round_summary

from model.model import TabularMLP

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


def main(base_cfg_path: str, dataset_cfg_path: str, model_cfg_path: str, gia_cfg_path: str, results_base_path: str | None = None):
    # load base.yaml
    base_cfg_path = Path(base_cfg_path)
    logger.info("Loading base config: base=%s", base_cfg_path)
    with open(base_cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f) or {}
        protocol = base_cfg.get("protocol")
        seed = base_cfg.get("seed")
    
    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(device))
    else:
        logger.info("Using CPU")

    # load dataset cfg
    logger.info("Loading dataset config: base=%s", dataset_cfg_path)
    with open(dataset_cfg_path, "r") as f:
        ds_cfg = yaml.safe_load(f) or {}
    
    # load model cfg
    logger.info("Loading model config: base=%s", model_cfg_path)
    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f) or {}

    # load fl cfg
    fl_cfg_path = base_cfg_path.parent / "fl" / f"{protocol}.yaml"
    logger.info("Loading fl config: base=%s", fl_cfg_path)
    with open(fl_cfg_path, "r") as f:
        fl_cfg = yaml.safe_load(f) or {}
        fl_cfg["batch_size"] = ds_cfg.get("batch_size")

    # load gia cfg
    logger.info("Loading gia config: base=%s", gia_cfg_path)
    with open(gia_cfg_path, "r") as f:
        gia_cfg = yaml.safe_load(f) or {}
        gia_cfg = gia_cfg.get(protocol, {}).get("invertingconfig", {})
    if results_base_path is None:
        results_base_path = gia_cfg.get("results_dir", "leakpro_output/tabular_gia")
    results_dir = Path(results_base_path) / protocol
    results_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    num_clients = fl_cfg.get("num_clients")
    ds_cfg["seed"] = seed
    client_dataloaders, val_loader, test_loader, feature_schema = load_dataset(
        **ds_cfg,
        num_clients=num_clients,
    )
    task = feature_schema["task"]
    if task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()


    # load model TabularMLP
    out_dim = 1 if task == "binary" else feature_schema["num_classes"]
    preset_name = "large"
    presets = model_cfg.get("presets", {}) or {}
    if preset_name not in presets:
        raise ValueError(f"Preset '{preset_name}' not found in model config.")
    preset_cfg = presets[preset_name] or {}
    model = TabularMLP(
        d_in=feature_schema["num_features"],
        d_hidden=preset_cfg.get("d_hidden", 64),
        d_out=out_dim,
        num_layers=preset_cfg.get("num_layers", 2),
        norm=preset_cfg.get("norm", "batchnorm"),
        dropout=preset_cfg.get("dropout", 0.0),
        activation=preset_cfg.get("activation", "relu"),
    )

    model.to(device)

    # build attack config
    lr = fl_cfg.get("lr")
    optimizer = fl_cfg.get("optimizer")
    logging.info(f"Optimizer: {optimizer}")
    def optimizer_fn(optimizer, lr):
        return MetaAdam(lr=lr) if optimizer == "MetaAdam" else MetaSGD(lr=lr)

    attack_cfg_base = InvertingConfig(
        attack_lr=gia_cfg.get("attack_lr"),
        at_iterations=gia_cfg.get("at_iterations"),
        optimizer=optimizer_fn(optimizer, lr),
        criterion=criterion,
        data_extension=GiaTabularExtension(),
        epochs=fl_cfg.get("local_epochs"),
    )

    # run fl training and inversion
    def round_summary_fn(epoch_idx: int, round_idx: int, metrics_list: list[dict]) -> None:
        write_round_summary(results_dir, epoch_idx, round_idx, metrics_list)

    if protocol == "fedsgd":
        def attack_fn(att_model, batch_loader, epoch_idx, round_idx, client_idx, client_grads, rng_pre, rng_post):
            # fresh config per attack instance to avoid cross-attack state sharing
            attack_cfg = deepcopy(attack_cfg_base)
            attacker = InvertingGradients(
                att_model,
                batch_loader,
                None,
                None,
                train_fn=train_nostep,
                configs=attack_cfg,
            )
            attacker.replay_rng_state = rng_pre
            attacker.prepare_attack()
            restore_rng_state(rng_post)

            # assert that the gradient computed in run_fedsgd is the same as in the GIA pipeline
            if len(client_grads) != len(attacker.client_gradient):
                raise AssertionError(
                    f"Gradient length mismatch: client={len(client_grads)} attacker={len(attacker.client_gradient)}"
                )
            for idx, (a, b) in enumerate(zip(client_grads, attacker.client_gradient)):
                if a is None or b is None:
                    continue
                if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                    raise AssertionError(f"Gradient mismatch at param {idx}")

            # perform GIA 
            total_iters = int(attack_cfg.at_iterations or 0)
            last_i = -1
            with tqdm(
                total=total_iters, desc=f"GIA Epoch {epoch_idx} Round {round_idx} Client {client_idx}", leave=False,
            ) as bar:
                for i, score, _ in attacker.run_attack():
                    step = max(0, int(i) - last_i)
                    if step:
                        bar.update(step)
                    last_i = int(i)
                    loss = -float(score) if score is not None else float("nan")
                    best_loss = float(attacker.best_loss)
                    bar.set_postfix(loss=f"{loss:.6f}", best=f"{best_loss:.6f}")
                if total_iters > 0 and last_i + 1 < total_iters:
                    bar.update(total_iters - (last_i + 1))

            out_dir = results_dir / f"epoch_{epoch_idx}" / f"round_{round_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            results_path = out_dir / f"client_{client_idx}.txt"
            evaluate_batch_rows(attacker, feature_schema, str(results_path), client_idx)
            metrics = compute_reconstruction_metrics(attacker, feature_schema, client_idx)
            tqdm.write(
                f"GIA done: epoch={epoch_idx} round={round_idx} client={client_idx} "
                f"best={float(attacker.best_loss):.6f}"
            )
            return metrics

        run_fedsgd(fl_cfg, model, criterion, attack_fn, client_dataloaders, val_loader, test_loader, round_summary_fn)

    elif protocol == "fedavg":
        def attack_fn(att_model, batch_loader, epoch_idx, round_idx, client_idx, client_deltas, rng_pre, rng_post):
            # fresh config per attack instance to avoid cross-attack state sharing
            attack_cfg = deepcopy(attack_cfg_base)
            attacker = InvertingGradients(
                att_model,
                batch_loader,
                None,
                None,
                train_fn=train,
                configs=attack_cfg,
            )
            attacker.replay_rng_state = rng_pre
            attacker.prepare_attack()
            restore_rng_state(rng_post)

            # assert that the deltas / model update computed in run_fedavg is the same as in the GIA pipeline
            if len(client_deltas) != len(attacker.client_gradient):
                raise AssertionError(
                    f"Delta length mismatch: client={len(client_deltas)} attacker={len(attacker.client_gradient)}"
                )
            for idx, (a, b) in enumerate(zip(client_deltas, attacker.client_gradient)):
                if a is None or b is None:
                    continue
                if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                    raise AssertionError(f"Delta mismatch at param {idx}")

            total_iters = int(attack_cfg.at_iterations or 0)
            last_i = -1
            with tqdm(
                total=total_iters,
                desc=f"GIA Epoch {epoch_idx} Round {round_idx} Client {client_idx}",
                leave=False,
            ) as bar:
                for i, score, _ in attacker.run_attack():
                    step = max(0, int(i) - last_i)
                    if step:
                        bar.update(step)
                    last_i = int(i)
                    loss = -float(score) if score is not None else float("nan")
                    best_loss = float(attacker.best_loss)
                    bar.set_postfix(loss=f"{loss:.6f}", best=f"{best_loss:.6f}")
                if total_iters > 0 and last_i + 1 < total_iters:
                    bar.update(total_iters - (last_i + 1))

            out_dir = results_dir / f"epoch_{epoch_idx}" / f"round_{round_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            results_path = out_dir / f"client_{client_idx}.txt"
            evaluate_batch_rows(attacker, feature_schema, str(results_path), client_idx)
            metrics = compute_reconstruction_metrics(attacker, feature_schema, client_idx)
            tqdm.write(
                f"GIA done: epoch={epoch_idx} round={round_idx} client={client_idx} "
                f"best={float(attacker.best_loss):.6f}"
            )
            return metrics

        run_fedavg(fl_cfg, model, criterion, optimizer_fn, attack_fn, client_dataloaders, val_loader, test_loader, round_summary_fn)


if __name__ == "__main__":
    base_cfg_path = "configs/base.yaml"
    dataset_cfg_path = "configs/dataset/dataset.yaml"
    model_cfg_path = "configs/model/model.yaml"
    gia_cfg_path = "configs/gia/gia.yaml"
    main(base_cfg_path, dataset_cfg_path, model_cfg_path, gia_cfg_path)
