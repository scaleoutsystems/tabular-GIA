"""FL training FedAvg implementation"""
import logging
import math
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from leakpro.fl_utils.gia_train import train
from leakpro.fl_utils.gia_optimizers import MetaAdam
from leakpro.utils.seed import capture_rng_state
from fl.metrics.fl_metrics import eval_epoch, infer_task_from_criterion, progress_write, round_bar


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def run_fedavg(cfg, global_model, criterion, optimizer_fn, attack_fn, client_dataloaders, val, test, round_summary_fn=None):
    # 1. parse fedavg cfg
    full_dataset_passes = cfg.get("full_dataset_passes")
    local_steps = cfg.get("local_steps", 1)
    local_epochs = cfg.get("local_epochs", 1)
    batch_size = cfg.get("batch_size")
    num_clients = len(client_dataloaders)
    client_participation = cfg.get("client_participation")
    lr = cfg.get("lr")
    optimizer = cfg.get("optimizer")

    # 1.1 derive expected rounds per epoch from batch size, local_steps, participation, num_clients, and dataset size(s)
    # batch_size * local_steps * num_clients * participation * rounds = len(data)
    # rounds = len(data from combined client_dataloaders) / (batch_size*local_steps*num_clients*participation)
    ds_size = sum(len(dl.dataset) for dl in client_dataloaders)
    expected_rounds_per_full_dataset_pass = math.ceil(ds_size / (batch_size * local_steps * num_clients * client_participation))
    logging.info("Expected rounds per full dataset pass: %d", expected_rounds_per_full_dataset_pass)
    clients_per_round = max(1, math.ceil(client_participation * num_clients))
    # NOTE: we divide full_dataset_passes by local_epochs to achieve true full dataset passes and compute
    full_dataset_passes = max(1, int(full_dataset_passes / local_epochs))

    for i in range(full_dataset_passes):
        # 2. initialize global model
        global_model.train()
        # persistent iterators per client for strict one-pass
        client_iters = [iter(dl) for dl in client_dataloaders]
        active_clients = set(range(num_clients))
        round_idx = 0
        with round_bar(expected_rounds_per_full_dataset_pass, f"Epoch {i + 1}/{full_dataset_passes}") as bar:
            while active_clients:
                global_model.train()
                round_idx += 1
                if round_idx > bar.total:
                    bar.total = round_idx
                    bar.refresh()
                bar.update(1)
                # 3. local client training and update collection
                k = min(clients_per_round, len(active_clients))
                active_list = list(active_clients)
                random.shuffle(active_list)

                client_updates = []
                round_metrics = []
                selected = 0
                for client_idx in active_list:
                    if selected >= k:
                        break
                    batches = []
                    for _ in range(local_steps):
                        try:
                            batches.append(next(client_iters[client_idx]))
                        except StopIteration:
                            break
                    if not batches:
                        active_clients.remove(client_idx)
                        continue

                    inputs_cat = torch.cat([b[0] for b in batches], dim=0)
                    labels_cat = torch.cat([b[1] for b in batches], dim=0)
                    batch_ds = TensorDataset(inputs_cat, labels_cat)
                    batch_loader = DataLoader(batch_ds, batch_size=batch_size, shuffle=False)
                    samples_seen = len(labels_cat) * local_epochs

                    rng_pre = capture_rng_state()
                    deltas = train(global_model, batch_loader, optimizer_fn(optimizer, lr), criterion, epochs=local_epochs)
                    rng_post = capture_rng_state()
                    client_updates.append((deltas, samples_seen))
                    selected += 1

                    # 4. gradient inversion attack using attack_fn
                    if attack_fn is not None:
                        metrics = attack_fn(global_model, batch_loader, i + 1, round_idx, client_idx, deltas, rng_pre, rng_post)
                        if metrics is not None:
                            round_metrics.append(metrics)

                # 5. aggregate with fedavg_train
                fedavg_train(global_model, client_updates)
                if round_summary_fn is not None and round_metrics:
                    round_summary_fn(i + 1, round_idx, round_metrics)

        # 6. epoch evaluation with validation loader
        task = infer_task_from_criterion(criterion)
        train_stats = eval_epoch(client_dataloaders, global_model, criterion, task)
        val_stats = eval_epoch([val], global_model, criterion, task) if val is not None else None

        if task in ("binary", "multiclass"):
            progress_write(
                "Epoch %d/%d - train_loss=%.4f train_acc=%.4f%s"
                % (
                    i + 1,
                    full_dataset_passes,
                    train_stats["loss"],
                    train_stats.get("acc", float("nan")),
                    "" if val_stats is None else f" val_loss={val_stats['loss']:.4f} val_acc={val_stats.get('acc', float('nan')):.4f}",
                )
            )
        else:
            progress_write(
                "Epoch %d/%d - train_loss=%.4f train_mse=%.4f train_r2=%.4f%s"
                % (
                    i + 1,
                    full_dataset_passes,
                    train_stats["loss"],
                    train_stats.get("mse", float("nan")),
                    train_stats.get("r2", float("nan")),
                    "" if val_stats is None else f" val_loss={val_stats['loss']:.4f} val_mse={val_stats.get('mse', float('nan')):.4f} val_r2={val_stats.get('r2', float('nan')):.4f}",
                )
            )

    # 7. perform final evaluation on test dataloader
    test_stats = eval_epoch([test], global_model, criterion, task)
    if task in ("binary", "multiclass"):
        logger.info(
            "Test - loss=%.4f acc=%.4f",
            test_stats["loss"],
            test_stats.get("acc", float("nan")),
        )
    else:
        logger.info(
            "Test - loss=%.4f mse=%.4f r2=%.4f",
            test_stats["loss"],
            test_stats.get("mse", float("nan")),
            test_stats.get("r2", float("nan")),
        )
    logging.info("FedAvg training completed.")


def fedavg_train(global_model, client_updates):
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
