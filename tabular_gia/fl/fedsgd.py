"""FL training FedSGD implementation"""
import logging
import math
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from leakpro.fl_utils.gia_train import train_nostep
from leakpro.fl_utils.gia_optimizers import MetaSGD # currently unused but wired
from tabular_gia.fl.metrics.fl_metrics import (
    eval_epoch,
    infer_task_from_criterion,
    progress_write,
    round_bar,
)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def run_fedsgd(cfg, global_model, criterion, attack_fn, client_dataloaders, val, test):
    # 1. parse cfg and take out essentials like effective epochs, num_clients, participation, lr
    epochs = cfg.get("full_dataset_passes")
    local_steps = cfg.get("local_steps", 1)
    local_epochs = cfg.get("local_epochs", 1)
    batch_size = cfg.get("batch_size")
    num_clients = len(client_dataloaders)
    client_participation = cfg.get("client_participation")
    lr = cfg.get("lr")

    # 1.1 derive expected rounds per epoch from batch size, participation, num_clients, and dataset size(s)
    # batch_size * num_clients * participation * rounds = len(data)
    # rounds = len(data from combined client_dataloaders) / (batch_size*steps=1*num_clients*participation)
    ds_size = sum(len(dl.dataset) for dl in client_dataloaders)
    expected_rounds_per_epoch = math.ceil(ds_size / (batch_size * 1 * num_clients * client_participation))
    logging.info("Expected rounds per epoch: %d", expected_rounds_per_epoch)
    clients_per_round = max(1, math.ceil(client_participation * num_clients))

    for i in range(epochs):
        # 2. initialize global model
        global_model.train()
        # persistent iterators per client for strict one-pass
        client_iters = [iter(dl) for dl in client_dataloaders]
        active_clients = set(range(num_clients))
        round_idx = 0
        with round_bar(expected_rounds_per_epoch, f"Epoch {i + 1}/{epochs}") as bar:
            while active_clients:
                global_model.train()
                round_idx += 1
                if round_idx > bar.total:
                    bar.total = round_idx
                    bar.refresh()
                bar.update(1)
                # 3. compute gradients using train_nostep
                k = min(clients_per_round, len(active_clients))
                active_list = list(active_clients)
                random.shuffle(active_list)

                client_gradients = []
                selected = 0
                for client_idx in active_list:
                    if selected >= k:
                        break
                    try:
                        batch = next(client_iters[client_idx])
                    except StopIteration:
                        active_clients.remove(client_idx)
                        continue

                    inputs, labels = batch
                    device = next(global_model.parameters()).device
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    batch_ds = TensorDataset(inputs, labels)
                    batch_loader = DataLoader(batch_ds, batch_size=len(batch_ds), shuffle=False)

                    grads = train_nostep(global_model, batch_loader, MetaSGD(lr), criterion, epochs=local_epochs)
                    client_gradients.append([g.detach() if g is not None else None for g in grads])
                    selected += 1

                    # 4. gradient inversion attack using attack_fn
                    if attack_fn is not None:
                        attack_fn(global_model, batch_loader, i + 1, round_idx, client_idx, grads)

                # 5. aggregate with fedsgd_train
                fedsgd_train(global_model, client_gradients, lr)
                #adamw_state = fedsgd_train_adamw(global_model, client_gradients, adamw_state, lr)

        # 6. epoch evaluation with validation loader
        task = infer_task_from_criterion(criterion)
        train_stats = eval_epoch(client_dataloaders, global_model, criterion, task)
        val_stats = eval_epoch([val], global_model, criterion, task) if val is not None else None

        if task in ("binary", "multiclass"):
            progress_write(
                "Epoch %d/%d - train_loss=%.4f train_acc=%.4f%s"
                % (
                    i + 1,
                    epochs,
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
                    epochs,
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
    logging.info("FedSGD training completed.")


def fedsgd_train(global_model, client_gradients, lr):
    """Gradient aggregation SGD"""
    if not client_gradients:
        return

    params = [p for p in global_model.parameters()]
    num_params = len(params)
    for grads in client_gradients:
        if len(grads) != num_params:
            raise ValueError("Client gradients do not match model parameters.")

    with torch.no_grad():
        for idx, param in enumerate(params):
            if not param.requires_grad:
                continue
            grads = [g for g in (client[idx] for client in client_gradients) if g is not None]
            if not grads:
                continue
            agg = torch.zeros_like(param)
            for grad in grads:
                agg.add_(grad.detach().to(param.device, dtype=param.dtype))
            agg.div_(len(grads))
            param.add_(agg, alpha=-lr)
