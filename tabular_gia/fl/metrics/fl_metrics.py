from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

FL_METRIC_FIELDS = ("loss", "acc", "f1_macro", "mse", "mae", "r2")


def _f1_macro_multiclass(conf_mat: torch.Tensor) -> float:
    tp = conf_mat.diag().float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp
    denom = (2 * tp) + fp + fn
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))
    return float(f1.mean().item())


def infer_task_from_criterion(criterion: torch.nn.Module) -> str:
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        return "binary"
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        return "multiclass"
    return "regression"


def eval_epoch(loaders: list[DataLoader], model: torch.nn.Module, criterion: torch.nn.Module, task: str | None = None) -> dict:
    if task is None:
        task = infer_task_from_criterion(criterion)

    device = next(model.parameters()).device
    total = 0
    loss_sum = 0.0
    correct = 0
    ss_res = 0.0
    abs_err = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    conf_mat = None

    model.eval()
    with torch.no_grad():
        for loader in loaders:
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)

                if task == "binary":
                    yb_t = yb.float().view(-1, 1)
                    loss = criterion(logits, yb_t)
                    preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
                    correct += (preds == yb.view(-1)).sum().item()
                elif task == "multiclass":
                    loss = criterion(logits, yb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb.view(-1)).sum().item()
                    num_classes = logits.shape[1]
                    if conf_mat is None:
                        conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)
                    y_true = yb.view(-1).long().cpu()
                    y_pred = preds.view(-1).long().cpu()
                    idx = y_true * num_classes + y_pred
                    conf_mat += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
                else:
                    preds = logits.view(-1)
                    yb_t = yb.view(-1).float()
                    loss = criterion(preds, yb_t)
                    ss_res += ((preds - yb_t) ** 2).sum().item()
                    abs_err += torch.abs(preds - yb_t).sum().item()
                    sum_y += yb_t.sum().item()
                    sum_y2 += (yb_t ** 2).sum().item()

                loss_sum += loss.item() * yb.size(0)
                total += yb.size(0)

    stats = {"loss": loss_sum / max(1, total)}
    if task in ("binary", "multiclass"):
        stats["acc"] = correct / max(1, total)
        if task == "multiclass":
            stats["f1_macro"] = _f1_macro_multiclass(conf_mat) if conf_mat is not None else float("nan")
    else:
        y_mean = sum_y / max(1, total)
        ss_tot = sum_y2 - total * (y_mean ** 2)
        stats["mse"] = stats["loss"]
        stats["mae"] = abs_err / max(1, total)
        stats["r2"] = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")
    return stats


@contextmanager
def round_bar(total: int, desc: str):
    bar = tqdm(total=total, desc=desc, unit="round", leave=False)
    try:
        yield bar
    finally:
        bar.close()


def progress_write(message: str) -> None:
    tqdm.write(message)
