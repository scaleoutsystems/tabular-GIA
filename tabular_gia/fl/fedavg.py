"""FL training FedAvg implementation"""
import logging
import math
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from leakpro.fl_utils.gia_train import train
from leakpro.utils.seed import capture_rng_state
from fl.metrics.fl_metrics import (
    eval_epoch,
    infer_task_from_criterion,
    progress_write,
    round_bar,
)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def run_fedavg(
    cfg,
    global_model,
    criterion,
    optimizer_fn,
    attack_fn,
    client_dataloaders,
    val,
    test,
    round_summary_fn=None,
    num_rounds_fn=None,
    exposure_progress_fn=None,
    fl_metrics_fn=None,
):
    # 1. parse fedavg cfg
    local_steps_raw = cfg.get("local_steps", 1)
    local_steps_all = isinstance(local_steps_raw, str) and local_steps_raw.strip().lower() == "all"
    local_steps = None if local_steps_all else max(1, int(local_steps_raw))
    local_epochs = max(1, int(cfg.get("local_epochs", 1)))
    batch_size = cfg.get("batch_size")
    num_clients = len(client_dataloaders)
    lr = cfg.get("lr")
    optimizer = cfg.get("optimizer")
    task = infer_task_from_criterion(criterion)

    # 1.1 derive round budget directly from min exposure target.
    client_n_eff = [int(len(dl) * int(getattr(dl, "batch_size", batch_size) or batch_size)) for dl in client_dataloaders]
    n_max_eff = max(client_n_eff) if client_n_eff else 1
    min_exposure = cfg.get("min_exposure")
    if min_exposure is None:
        raise ValueError("Missing required FL config field: min_exposure")
    min_exposure = float(min_exposure)
    if min_exposure <= 0:
        raise ValueError(f"min_exposure must be > 0, got {min_exposure}")
    if local_steps_all:
        num_rounds = max(1, math.ceil(min_exposure / local_epochs))
        samples_per_client_per_round = "all_local_batches"
    else:
        samples_per_client_per_round = max(1, local_steps * batch_size * local_epochs)
        num_rounds = max(1, math.ceil((min_exposure * n_max_eff) / samples_per_client_per_round))

    if num_rounds_fn is not None:
        num_rounds_fn(int(num_rounds))

    logging.info(
        "Round budget: rounds=%d min_exposure=%.6f n_max_eff=%d samples_per_round=%s",
        num_rounds,
        min_exposure,
        n_max_eff,
        samples_per_client_per_round,
    )

    # 2. initialize global model
    global_model.train()
    # Persistent iterators per client; wrap around when exhausted.
    client_iters = [iter(dl) for dl in client_dataloaders] if not local_steps_all else []
    examples_seen = [0 for _ in range(num_clients)]
    next_val_exposure = 1.0 if val is not None else float("inf")
    best_val_loss = float("inf")
    best_round = 0
    best_state_dict = None

    def _next_batch(client_idx: int):
        try:
            return next(client_iters[client_idx])
        except StopIteration:
            client_iters[client_idx] = iter(client_dataloaders[client_idx])
            return next(client_iters[client_idx])

    executed_rounds = 0
    exp_min_prev = 0.0
    with round_bar(num_rounds, "Communication Rounds") as bar:
        for round_idx in range(1, num_rounds + 1):
            executed_rounds = round_idx
            global_model.train()
            bar.update(1)
            # 3. local client training and update collection
            active_list = list(range(num_clients))
            random.shuffle(active_list)

            client_updates = []
            attack_payloads = [] if attack_fn is not None else None
            for client_idx in active_list:
                if local_steps_all:
                    # Classic FedAvg local step: one full local dataloader pass per round.
                    batches = list(iter(client_dataloaders[client_idx]))
                else:
                    batches = []
                    for _ in range(local_steps):
                        batches.append(_next_batch(client_idx))

                inputs_cat = torch.cat([b[0] for b in batches], dim=0)
                labels_cat = torch.cat([b[1] for b in batches], dim=0)
                batch_ds = TensorDataset(inputs_cat, labels_cat)
                batch_loader = DataLoader(batch_ds, batch_size=batch_size, shuffle=False)
                samples_seen = len(labels_cat) * local_epochs
                examples_seen[client_idx] += samples_seen

                rng_pre = capture_rng_state()
                deltas = train(global_model, batch_loader, optimizer_fn(optimizer, lr), criterion, epochs=local_epochs)
                rng_post = capture_rng_state()
                client_updates.append((deltas, samples_seen))
                if attack_payloads is not None:
                    attack_payloads.append((batch_loader, client_idx, deltas, rng_pre, rng_post))

            current_exposures = [seen / max(1, n_eff) for seen, n_eff in zip(examples_seen, client_n_eff)] if client_n_eff else []
            exp_min_curr = min(current_exposures) if current_exposures else 0.0
            exp_avg_curr = (sum(current_exposures) / len(current_exposures)) if current_exposures else 0.0
            exp_max_curr = max(current_exposures) if current_exposures else 0.0
            crossed_val_checkpoint = False
            while exp_min_prev < next_val_exposure <= exp_min_curr:
                crossed_val_checkpoint = True
                next_val_exposure += 1.0
            if exposure_progress_fn is not None:
                exposure_progress_fn(int(round_idx), float(exp_min_prev), float(exp_min_curr))
            exp_min_prev = exp_min_curr

            # 4. gradient inversion attack using attack_fn
            round_metrics = []
            if attack_payloads is not None:
                for batch_loader, client_idx, deltas, rng_pre, rng_post in attack_payloads:
                    attack_context = {
                        "exp_min": float(exp_min_curr),
                        "exp_avg": float(exp_avg_curr),
                        "exp_max": float(exp_max_curr),
                        "client_exp": float(current_exposures[client_idx]) if current_exposures else 0.0,
                    }
                    metrics = attack_fn(global_model, batch_loader, round_idx, client_idx, deltas, rng_pre, rng_post, attack_context)
                    if isinstance(metrics, list):
                        round_metrics.extend(metrics)
                    elif metrics is not None:
                        round_metrics.append(metrics)

            # 5. aggregate with fedavg_train
            fedavg_train(global_model, client_updates)
            if round_summary_fn is not None and round_metrics:
                round_summary_fn(round_idx, round_metrics)

            # 6. round-level evaluation with validation loader
            if crossed_val_checkpoint:
                train_stats = eval_epoch(client_dataloaders, global_model, criterion, task)
                val_stats = eval_epoch([val], global_model, criterion, task) if val is not None else None
                if val_stats is not None:
                    val_loss = float(val_stats["loss"])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_round = int(round_idx)
                        best_state_dict = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
                if fl_metrics_fn is not None:
                    metric_keys = list(train_stats.keys()) if train_stats is not None else (list(val_stats.keys()) if val_stats is not None else [])
                    row = {
                        "phase": "checkpoint",
                        "round": int(round_idx),
                        "exp_min": float(exp_min_curr),
                        "exp_avg": float(exp_avg_curr),
                        "exp_max": float(exp_max_curr),
                    }
                    for split_name, stats in (("train", train_stats), ("val", val_stats), ("test", None)):
                        stats = stats or {}
                        for metric_key in metric_keys:
                            row[f"{split_name}_{metric_key}"] = stats.get(metric_key)
                    fl_metrics_fn(row)
                if task == "binary":
                    progress_write(
                        "Rounds %d - train_loss=%.4f train_acc=%.4f%s"
                        % (
                            executed_rounds,
                            train_stats["loss"],
                            train_stats.get("acc", float("nan")),
                            "" if val_stats is None else f" val_loss={val_stats['loss']:.4f} val_acc={val_stats.get('acc', float('nan')):.4f}",
                        )
                    )
                elif task == "multiclass":
                    progress_write(
                        "Rounds %d - train_loss=%.4f train_acc=%.4f train_f1_macro=%.4f%s"
                        % (
                            executed_rounds,
                            train_stats["loss"],
                            train_stats.get("acc", float("nan")),
                            train_stats.get("f1_macro", float("nan")),
                            "" if val_stats is None else f" val_loss={val_stats['loss']:.4f} val_acc={val_stats.get('acc', float('nan')):.4f} val_f1_macro={val_stats.get('f1_macro', float('nan')):.4f}",
                        )
                    )
                else:
                    progress_write(
                        "Rounds %d - train_loss=%.4f train_mse=%.4f train_mae=%.4f train_r2=%.4f%s"
                        % (
                            executed_rounds,
                            train_stats["loss"],
                            train_stats.get("mse", float("nan")),
                            train_stats.get("mae", float("nan")),
                            train_stats.get("r2", float("nan")),
                            "" if val_stats is None else f" val_loss={val_stats['loss']:.4f} val_mse={val_stats.get('mse', float('nan')):.4f} val_mae={val_stats.get('mae', float('nan')):.4f} val_r2={val_stats.get('r2', float('nan')):.4f}",
                        )
                    )

    if client_n_eff:
        final_exposures = [seen / max(1, n_eff) for seen, n_eff in zip(examples_seen, client_n_eff)]
        logging.info(
            "Final exposure: min=%.6f mean=%.6f max=%.6f",
            min(final_exposures),
            sum(final_exposures) / len(final_exposures),
            max(final_exposures),
        )

    if best_state_dict is not None:
        global_model.load_state_dict(best_state_dict)
        logger.info("Loaded best validation checkpoint: round=%d val_loss=%.4f", best_round, best_val_loss)

    # 7. perform final evaluation on test dataloader
    test_stats = eval_epoch([test], global_model, criterion, task)
    if fl_metrics_fn is not None:
        metric_keys = list(test_stats.keys()) if test_stats is not None else []
        tested_round = int(best_round) if best_state_dict is not None else int(executed_rounds)
        row = {
            "phase": "final_test",
            "round": tested_round,
            "exp_min": float(min(final_exposures)) if client_n_eff else float("nan"),
            "exp_avg": float(sum(final_exposures) / len(final_exposures)) if client_n_eff else float("nan"),
            "exp_max": float(max(final_exposures)) if client_n_eff else float("nan"),
        }
        for split_name, stats in (("train", None), ("val", None), ("test", test_stats)):
            stats = stats or {}
            for metric_key in metric_keys:
                row[f"{split_name}_{metric_key}"] = stats.get(metric_key)
        fl_metrics_fn(row)
    if task == "binary":
        logger.info(
            "Test - loss=%.4f acc=%.4f",
            test_stats["loss"],
            test_stats.get("acc", float("nan")),
        )
    elif task == "multiclass":
        logger.info(
            "Test - loss=%.4f acc=%.4f f1_macro=%.4f",
            test_stats["loss"],
            test_stats.get("acc", float("nan")),
            test_stats.get("f1_macro", float("nan")),
        )
    else:
        logger.info(
            "Test - loss=%.4f mse=%.4f mae=%.4f r2=%.4f",
            test_stats["loss"],
            test_stats.get("mse", float("nan")),
            test_stats.get("mae", float("nan")),
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
