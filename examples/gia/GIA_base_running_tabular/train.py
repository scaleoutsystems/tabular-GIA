"""Train a tabular global model and save a checkpoint for GIA demos."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from leakpro.utils.seed import seed_everything

from model import TabularMLP
from tabular import get_tabular_loaders, load_tabular_config


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
	name = name.lower()
	if name == "adamw":
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	if name == "adam":
		return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	logger.warning("Unknown optimizer '%s'; falling back to AdamW", name)
	return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def scheduled_lr(epoch: int, total: int, base_lr: float, warmup_pct: float, min_lr_scale: float) -> float:
	warmup = min(max(0, int(round(total * warmup_pct))), max(0, total - 1))
	decay = max(1, total - warmup)
	start_lr = base_lr * min_lr_scale
	final_lr = base_lr * min_lr_scale
	if warmup > 0 and epoch < warmup:
		progress = (epoch + 1) / warmup
		return max(final_lr, start_lr + (base_lr - start_lr) * progress)
	decay_progress = min(1.0, max(0.0, (epoch - warmup + 1) / decay))
	return max(final_lr, base_lr - (base_lr - final_lr) * decay_progress)


def run_epoch(
	loader: DataLoader,
	model: nn.Module,
	criterion: nn.Module,
	is_binary: bool,
	is_multiclass: bool,
	metric_classes: int,
	optimizer: torch.optim.Optimizer | None = None,
	device: torch.device | None = None,
) -> Dict[str, float | torch.Tensor | int]:
	train_mode = optimizer is not None
	model.train() if train_mode else model.eval()

	loss_sum = correct = total = 0.0
	cm = torch.zeros((metric_classes, metric_classes), dtype=torch.long) if (is_binary or is_multiclass) else None
	abs_err = ss_res = sum_y = sum_y2 = 0.0

	with torch.enable_grad() if train_mode else torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device) if device is not None else xb
			yb = yb.to(device) if device is not None else yb
			logits = model(xb)
			if is_binary:
				yb_t = yb.float().view(-1, 1)
				loss = criterion(logits, yb_t)
				preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
			elif is_multiclass:
				loss = criterion(logits, yb)
				preds = logits.argmax(dim=1)
			else:
				preds = logits.view(-1)
				yb_t = yb.view(-1)
				loss = criterion(preds, yb_t)
				abs_err += (preds - yb_t).abs().sum().item()
				ss_res += ((preds - yb_t) ** 2).sum().item()
				sum_y += yb_t.sum().item()
				sum_y2 += (yb_t ** 2).sum().item()

			if train_mode:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			loss_sum += loss.item() * yb.size(0)
			total += yb.numel()
			if is_binary or is_multiclass:
				correct += (preds == yb.view(-1)).sum().item()
				idx = preds.cpu() * metric_classes + yb.view(-1).long().cpu()
				cm += torch.bincount(idx, minlength=metric_classes * metric_classes).view(metric_classes, metric_classes)

	return {
		"loss": loss_sum / max(1, total),
		"acc": (correct / max(1, total)) if (is_binary or is_multiclass) else None,
		"cm": cm,
		"abs_err": abs_err,
		"ss_res": ss_res,
		"sum_y": sum_y,
		"sum_y2": sum_y2,
		"total": total,
	}


def _macro_f1(cm: torch.Tensor) -> float:
	tp = cm.diag()
	fp = cm.sum(dim=1) - tp
	fn = cm.sum(dim=0) - tp
	den = (2 * tp + fp + fn).clamp(min=1e-12)
	return float(((2 * tp) / den).mean().item())


def log_classification(epoch: int, epochs: int, lr: float, train_stats: Dict, val_stats: Dict, is_multiclass: bool) -> None:
	if is_multiclass:
		train_f1 = _macro_f1(train_stats["cm"].to(torch.float32))
		val_f1 = _macro_f1(val_stats["cm"].to(torch.float32))
		logger.info(
			"Epoch %d/%d - lr=%.5f train_acc=%.4f val_acc=%.4f train_f1=%.4f val_f1=%.4f train_loss=%.4f val_loss=%.4f",
			epoch + 1,
			epochs,
			lr,
			train_stats["acc"],
			val_stats["acc"],
			train_f1,
			val_f1,
			train_stats["loss"],
			val_stats["loss"],
		)
	else:
		logger.info(
			"Epoch %d/%d - lr=%.5f train_acc=%.4f val_acc=%.4f train_loss=%.4f val_loss=%.4f",
			epoch + 1,
			epochs,
			lr,
			train_stats["acc"],
			val_stats["acc"],
			train_stats["loss"],
			val_stats["loss"],
		)


def log_regression(epoch: int, epochs: int, lr: float, train_stats: Dict, val_stats: Dict) -> None:
	val_mae = val_stats["abs_err"] / max(1, val_stats["total"])
	y_mean = val_stats["sum_y"] / max(1, val_stats["total"])
	ss_tot = val_stats["sum_y2"] - val_stats["total"] * (y_mean ** 2)
	val_r2 = 1.0 - (val_stats["ss_res"] / ss_tot) if ss_tot != 0 else float("nan")
	logger.info(
		"Epoch %d/%d - lr=%.5f train_loss=%.4f val_loss=%.4f val_mae=%.4f val_r2=%.4f",
		epoch + 1,
		epochs,
		lr,
		train_stats["loss"],
		val_stats["loss"],
		val_mae,
		val_r2,
	)


def test(model: nn.Module, loader: DataLoader, criterion: nn.Module, is_binary: bool, is_multiclass: bool, metric_classes: int) -> None:
	device = next(model.parameters()).device
	stats = run_epoch(loader, model, criterion, is_binary, is_multiclass, metric_classes, device=device)
	if is_binary or is_multiclass:
		acc = stats["acc"]
		f1 = _macro_f1(stats["cm"].to(torch.float32)) if is_multiclass else None
		logger.info("Test accuracy: %.4f%s", acc, f" | macro_f1={f1:.4f}" if f1 is not None else "")
	else:
		mse = stats["loss"]
		mae = stats["abs_err"] / max(1, stats["total"])
		y_mean = stats["sum_y"] / max(1, stats["total"])
		ss_tot = stats["sum_y2"] - stats["total"] * (y_mean ** 2)
		r2 = 1.0 - (stats["ss_res"] / ss_tot) if ss_tot != 0 else float("nan")
		logger.info("Test MSE: %.4f | MAE: %.4f | R2: %.4f", mse, mae, r2)


def train(
	model: nn.Module,
	loaders: Dict,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	cfg: dict,
	is_binary: bool,
	is_multiclass: bool,
	metric_classes: int,
	head_out: int,
	data_mean: torch.Tensor,
	data_std: torch.Tensor,
	encoder_meta: dict,
	checkpoint_path: Path,
) -> float:
	device = next(model.parameters()).device
	trainer_cfg = cfg.get("trainer", {}) or {}
	epochs = trainer_cfg.get("epochs", 20)
	lr = trainer_cfg.get("lr", 1e-3)
	warmup_pct = float(trainer_cfg.get("warmup_pct", 0.10))
	min_lr_scale = float(trainer_cfg.get("min_lr_scale", 0.01))
	best_val = float("-inf") if (is_binary or is_multiclass) else float("inf")

	for epoch in range(epochs):
		current_lr = scheduled_lr(epoch, epochs, lr, warmup_pct, min_lr_scale)
		for pg in optimizer.param_groups:
			pg["lr"] = current_lr

		train_stats = run_epoch(loaders["train_loader"], model, criterion, is_binary, is_multiclass, metric_classes, optimizer, device)
		val_stats = run_epoch(loaders["val_loader"], model, criterion, is_binary, is_multiclass, metric_classes, device=device)
		val_metric = val_stats["acc"] if (is_binary or is_multiclass) else -val_stats["loss"]

		if is_binary or is_multiclass:
			log_classification(epoch, epochs, current_lr, train_stats, val_stats, is_multiclass)
		else:
			log_regression(epoch, epochs, current_lr, train_stats, val_stats)

		improved = (val_metric >= best_val) if (is_binary or is_multiclass) else (val_stats["loss"] <= best_val)
		if improved:
			best_val = val_metric if (is_binary or is_multiclass) else val_stats["loss"]
			torch.save(
				{
					"model_state": model.state_dict(),
					"data_mean": data_mean,
					"data_std": data_std,
					"config": cfg,
					"encoder_meta": encoder_meta,
						"meta": {
							"epochs": epoch + 1,
							"train_acc": train_stats.get("acc"),
							"train_loss": train_stats["loss"],
							"val_metric": val_metric if (is_binary or is_multiclass) else None,
							"val_loss": val_stats["loss"] if not (is_binary or is_multiclass) else None,
							"feature_dim": loaders["n_features"],
							"num_classes": loaders["num_classes"],  # number of target classes
							"head_out": head_out,  # model output dim; 1 for binary BCE head
							"task": "binary" if is_binary else ("multiclass" if is_multiclass else "regression"),
						},
				},
				checkpoint_path,
			)
			logger.info("Saved checkpoint to %s", checkpoint_path)

	return best_val


def train_global_model(cfg: dict) -> Tuple[Path, float]:
	seed_everything(cfg["seed"])
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	trainer_cfg = cfg.get("trainer", {}) or {}

	data_path = Path(cfg["data_path"])
	dataset_dir = data_path.parent
	ckpt_dir_cfg = trainer_cfg.get("checkpoint_dir")
	checkpoint_dir = Path(ckpt_dir_cfg) if ckpt_dir_cfg else dataset_dir / "checkpoints"
	if not checkpoint_dir.is_absolute():
		checkpoint_dir = dataset_dir / checkpoint_dir
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = checkpoint_dir / f"{data_path.stem}.ckpt"

	loaders, encoder_meta = get_tabular_loaders(cfg)
	train_loader = loaders["train_loader"]
	val_loader = loaders["val_loader"]
	logger.info(
		"Split sizes: train=%d val=%d test=%d client=%d | batch_size=%d",
		len(train_loader.dataset),
		len(val_loader.dataset) if val_loader is not None else 0,
		len(loaders["test_loader"].dataset) if loaders.get("test_loader") is not None else 0,
		len(loaders["client_loader"].dataset),
		cfg.get("batch_size", 64),
	)

	n_features = loaders["n_features"]
	num_classes = loaders["num_classes"]
	is_binary = num_classes == 2
	is_multiclass = num_classes > 2
	is_regression = num_classes == 1 and next(iter(train_loader))[1].dtype != torch.long
	out_classes = 1 if is_binary else num_classes
	metric_classes = 2 if is_binary else num_classes

	logger.info("Detected task=%s, num_classes=%s", "binary" if is_binary else ("multiclass" if is_multiclass else "regression"), num_classes)
	model = TabularMLP(d_in=n_features, num_classes=out_classes).to(device)
	criterion = (nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss() if is_multiclass else nn.MSELoss()).to(device)

	optimizer = make_optimizer(
		model,
		str(trainer_cfg.get("optimizer", "adamw")),
		lr=trainer_cfg.get("lr", 1e-3),
		weight_decay=trainer_cfg.get("weight_decay", 1e-4),
	)

	best_val = train(
		model,
		loaders,
		criterion,
		optimizer,
		cfg,
		is_binary,
		is_multiclass,
		metric_classes,
		out_classes,
		loaders["data_mean"],
		loaders["data_std"],
		encoder_meta,
		checkpoint_path,
	)

	if loaders.get("test_loader") is not None:
		state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
		model.load_state_dict(state["model_state"])
		model = model.to(device).eval()
		test(model, loaders["test_loader"], criterion, is_binary, is_multiclass, metric_classes)

	return checkpoint_path, best_val


if __name__ == "__main__":
	config_path = Path(__file__).resolve().parent / "config.yaml"
	cfg = load_tabular_config(config_path)
	ckpt, best = train_global_model(cfg)
	logger.info("Best validation metric: %.4f (checkpoint: %s)", best, ckpt)
