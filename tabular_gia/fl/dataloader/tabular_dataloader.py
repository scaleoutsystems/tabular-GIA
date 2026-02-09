import numpy as np
import random
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
from functools import partial

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _handle_missing_values(df: pd.DataFrame, missing_values) -> pd.DataFrame:
    if missing_values is None:
        raise ValueError("missing_values must be set in the dataset meta yaml")
    return df.replace(missing_values, pd.NA)


def _infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = []
    cat_cols = []
    for col in df.columns:
        series = df[col]
        non_na = series.dropna()
        if non_na.empty:
            num_cols.append(col)
            continue
        converted = pd.to_numeric(non_na, errors="coerce")
        if not converted.notna().all():
            cat_cols.append(col)
            continue
        values = converted.to_numpy()
        is_int_like = np.all(np.isclose(values, np.round(values)))
        if is_int_like:
            unique_count = pd.Series(values).nunique(dropna=True)
            unique_ratio = unique_count / max(1, len(values))
            if unique_count <= 20 and unique_ratio <= 0.05:
                cat_cols.append(col)
                continue
        num_cols.append(col)
    return num_cols, cat_cols


def _fill_missing_values(
    df: pd.DataFrame,
    missing_values,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    if num_cols:
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if cat_cols:
        filled = df[cat_cols].astype(object)
        df[cat_cols] = filled.where(filled.notna(), missing_values)
    return df


def preprocess(
    X: pd.DataFrame,
    missing_values,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
)-> tuple[pd.DataFrame, list[str], list[str]]:
    X = _handle_missing_values(X, missing_values)

    if num_cols is None or cat_cols is None:
        num_cols, cat_cols = _infer_column_types(X)

    X = _fill_missing_values(X, missing_values, num_cols, cat_cols)
    return X, num_cols, cat_cols


def _fit_encoders(
    X_train: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    missing_values,
) -> dict:
    if cat_cols:
        X_cat = X_train[cat_cols].astype(object)
        cat_dummy_cols = pd.get_dummies(X_cat, drop_first=False).columns.tolist()
    else:
        cat_dummy_cols = []
    cat_categories = {
        c: pd.Categorical(X_train[c].astype(object), ordered=False).categories.tolist()
        for c in cat_cols
    }
    return {
        "cat_dummy_cols": cat_dummy_cols,
        "cat_categories": cat_categories,
    }


def _apply_encoders(
    X: pd.DataFrame,
    meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    X_num = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0) if num_cols else pd.DataFrame(index=X.index)
    if cat_cols:
        X_cat = X[cat_cols].astype(object)
        X_cat = pd.get_dummies(X_cat, drop_first=False).reindex(columns=meta["cat_dummy_cols"], fill_value=0)
    else:
        X_cat = pd.DataFrame(index=X.index)
    if len(X_num.columns) and len(X_cat.columns):
        return pd.concat([X_num, X_cat], axis=1)
    return X_num if len(X_num.columns) else X_cat


def _normalize_numeric(
    X: pd.DataFrame,
    num_cols: list[str],
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame:
    if not num_cols:
        return X
    X = X.copy()
    X[num_cols] = (X[num_cols] - mean) / std
    return X


def denormalize_numeric(
    X: pd.DataFrame,
    num_cols: list[str],
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame:
    if not num_cols:
        return X
    X = X.copy()
    X[num_cols] = (X[num_cols] * std) + mean
    return X


def _split_target(df: pd.DataFrame, target, no_header) -> tuple[pd.DataFrame, pd.Series]:
    if target is None:
        raise ValueError("target must be set in the dataset meta yaml")

    if no_header:
        if not isinstance(target, int):
            raise ValueError("target must be an integer column index when no_header is true")
        if target < 0 or target >= df.shape[1]:
            raise ValueError(f"target index {target} is out of bounds for {df.shape[1]} columns")
        y = df.iloc[:, target]
        X = df.drop(columns=[target])
    else:
        if target not in df.columns:
            raise ValueError(f"target column '{target}' not found in dataset")
        y = df[target]
        X = df.drop(columns=[target])
    return X, y


def _infer_task_and_clean(
    y: pd.Series,
    missing_values,
) -> tuple[pd.Series, str, pd.Series]:
    if missing_values is not None:
        y = y.replace(missing_values, pd.NA)

    dtype_str = str(y.dtype)
    if dtype_str == "object" or dtype_str.startswith("category"):
        keep_mask = y.notna()
        y = y.loc[keep_mask].reset_index(drop=True)
        classes = pd.Categorical(y, ordered=False).categories.tolist()
        task = "binary" if len(classes) == 2 else "multiclass"
        return y, task, keep_mask

    y_num = pd.to_numeric(y, errors="coerce")
    keep_mask = y_num.notna()
    y_num = y_num.loc[keep_mask].reset_index(drop=True)
    unique = y_num.nunique(dropna=False)
    if unique <= 20:
        classes = pd.Categorical(y_num, ordered=False).categories.tolist()
        task = "binary" if len(classes) == 2 else "multiclass"
        return y_num, task, keep_mask

    return y_num, "regression", keep_mask


def _encode_target(y: pd.Series, task: str, target_classes: list | None) -> torch.Tensor:
    if task in ("binary", "multiclass"):
        if target_classes is None:
            raise ValueError("target_classes must be provided for classification tasks")
        codes = pd.Categorical(y, categories=target_classes).codes
        if (codes == -1).any():
            raise ValueError("Unseen target class encountered in split")
        if task == "binary":
            return torch.tensor(codes, dtype=torch.float32).view(-1, 1)
        return torch.tensor(codes, dtype=torch.long)

    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        raise ValueError("Regression target contains NaNs after preprocessing")
    return torch.tensor(y_num.to_numpy(), dtype=torch.float32).view(-1, 1)


def load_dataset(dataset_path: str, dataset_meta_path: str, num_clients: int, **kwargs):
    """Load and return dataloaders for a dataset. Kwargs are optional parameters to set dataloader speedups:
        num_workers,
        pin_memory,
        persistent_workers,
    """
    # 1. load meta yaml and parse
    logger.info("Loading dataset config: meta=%s data=%s", dataset_meta_path, dataset_path)
    with open(dataset_meta_path, "r") as f:
        meta = yaml.safe_load(f) or {}

    # 1.1 handle no_header if present
    no_header = bool(meta.get("no_header", False))
    if no_header:
        # 2. load csv dataset to pandas dataframe
        df = pd.read_csv(dataset_path, header=None)
    else:
        df = pd.read_csv(dataset_path)
    logger.info("Loaded data: rows=%d cols=%d no_header=%s", df.shape[0], df.shape[1], no_header)

    # 3 perform target split
    target = meta.get("target")
    missing_values = meta.get("missing_values")
    X_full, y_full = _split_target(df, target, no_header)
    logger.info("Target split: target=%s features=%d", target, X_full.shape[1])

    # 3.1 infer task objective and classes (binary, multiclass, regression)
    y, task, keep_mask = _infer_task_and_clean(y_full, missing_values)
    # drop rows with missing targets
    X_full = X_full.loc[keep_mask].reset_index(drop=True)
    if keep_mask.size:
        dropped = int((~keep_mask).sum())
    else:
        dropped = 0
    logger.info("Task inferred: %s | dropped_missing_targets=%d", task, dropped)

    # 4. split into train, val, test at ratios 70 / 15 / 15, stratify on target
    seed = kwargs.get("seed", 42)
    train_frac = kwargs.get("train_frac", 0.70)
    val_frac = kwargs.get("val_frac", 0.15)
    test_frac = kwargs.get("test_frac", 0.15)
    stratify_labels = y if task in ("binary", "multiclass") else None

    test_path = meta.get("has_test_split")
    if test_path:
        logger.info("External test split provided: %s", test_path)
        val_ratio = val_frac / max(1e-8, (train_frac + val_frac))
        X_train_split, X_val_split, y_train, y_val = train_test_split(
            X_full,
            y,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify_labels,
        )
        test_path = Path(test_path)
        #if not test_path.is_absolute():
        #    test_path = Path(dataset_path).parent / test_path
        if no_header:
            test_df = pd.read_csv(test_path, header=None)
        else:
            test_df = pd.read_csv(test_path)
        X_test_split, y_test_split = _split_target(test_df, target, no_header)
        y_test, task_test, keep_mask = _infer_task_and_clean(y_test_split, missing_values)
        X_test_split = X_test_split.loc[keep_mask].reset_index(drop=True)
        if task_test != task:
            raise ValueError(f"Test split task '{task_test}' does not match train task '{task}'")
    else:
        logger.info("No external test split; using internal train/val/test fractions.")
        X_train_split, X_holdout, y_train, y_temp = train_test_split(
            X_full,
            y,
            test_size=(1 - train_frac),
            random_state=seed,
            stratify=stratify_labels,
        )
        remaining = val_frac + test_frac
        test_size = (test_frac / remaining) if remaining > 0 else 0.5
        X_val_split, X_test_split, y_val, y_test = train_test_split(
            X_holdout,
            y_temp,
            test_size=test_size,
            random_state=seed,
            stratify=y_temp if task in ("binary", "multiclass") else None,
        )

    # 4.1 preprocess splits without one-hot (infer types on train only)
    X_train_split, num_cols, cat_cols = preprocess(X_train_split, missing_values)
    X_val_split, _, _ = preprocess(
        X_val_split,
        missing_values,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    X_test_split, _, _ = preprocess(
        X_test_split,
        missing_values,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    logger.info(
        "Preprocessed splits: train=%d val=%d test=%d features=%d",
        len(X_train_split),
        len(X_val_split),
        len(X_test_split),
        X_train_split.shape[1],
    )

    # 4.2 derive target classes from train (classification)
    if task in ("binary", "multiclass"):
        target_classes = pd.Categorical(y_train, ordered=False).categories.tolist()
        task = "binary" if len(target_classes) == 2 else "multiclass"
    else:
        target_classes = None
    logger.info("Target classes (train-derived): %s", target_classes if target_classes is not None else "regression")

    # 4.3 fit/apply encoders (one-hot) using train only
    encoder_meta = _fit_encoders(X_train_split, num_cols, cat_cols, missing_values)
    X_train      = _apply_encoders(X_train_split, encoder_meta, num_cols, cat_cols)
    X_val        = _apply_encoders(X_val_split, encoder_meta, num_cols, cat_cols)
    X_test       = _apply_encoders(X_test_split, encoder_meta, num_cols, cat_cols)
    feature_columns = X_train.columns.tolist()

    # global train stats for val/test normalization
    if num_cols:
        global_mean = X_train[num_cols].mean()
        global_std = X_train[num_cols].std().replace(0, 1e-6)
        X_val = _normalize_numeric(X_val, num_cols, global_mean, global_std)
        X_test = _normalize_numeric(X_test, num_cols, global_mean, global_std)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # 4.3 split train amongst num_clients, stratify on target
    if num_clients == 1:
        client_splits = [(X_train, y_train)]
    else:
        if task in ("binary", "multiclass"):
            splitter = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
            split_iter = splitter.split(X_train, y_train)
        else:
            splitter = KFold(n_splits=num_clients, shuffle=True, random_state=seed)
            split_iter = splitter.split(X_train)

        client_splits = []
        for _, client_idx in split_iter:
            X_client = X_train.iloc[client_idx].reset_index(drop=True)
            y_client = y_train.iloc[client_idx].reset_index(drop=True)
            client_splits.append((X_client, y_client))
        logger.info("Client splits: clients=%d", len(client_splits))

    # 5. create dataloaders
    batch_size = kwargs.get("batch_size", 64)
    num_workers = int(kwargs.get("num_workers", 0))
    pin_memory = bool(kwargs.get("pin_memory", False))
    persistent_workers = bool(kwargs.get("persistent_workers", num_workers > 0))
    generator = torch.Generator()
    generator.manual_seed(seed)

    worker_init_fn = partial(_seed_worker, seed=seed) if num_workers > 0 else None

    y_train_t = _encode_target(y_train, task, target_classes)
    y_val_t = _encode_target(y_val, task, target_classes)
    y_test_t = _encode_target(y_test, task, target_classes)

    _train_ds = TensorDataset(
        torch.tensor(X_train.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32),
        y_train_t,
    )
    val_ds = TensorDataset(
        torch.tensor(X_val.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32),
        y_val_t,
    )
    test_ds = TensorDataset(
        torch.tensor(X_test.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32),
        y_test_t,
    )

    client_dataloaders = []
    client_num_means = []
    client_num_stds = []
    for X_client, y_client in client_splits:
        if num_cols:
            client_mean = X_client[num_cols].mean()
            client_std = X_client[num_cols].std().replace(0, 1e-6)
            X_client = _normalize_numeric(X_client, num_cols, client_mean, client_std)
            client_num_means.append(client_mean.to_numpy(dtype=np.float32, copy=True))
            client_num_stds.append(client_std.to_numpy(dtype=np.float32, copy=True))
        else:
            client_num_means.append(np.array([], dtype=np.float32))
            client_num_stds.append(np.array([], dtype=np.float32))
        y_client_t = _encode_target(y_client, task, target_classes)
        client_ds = TensorDataset(
            torch.tensor(X_client.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32),
            y_client_t,
        )
        client_loader = DataLoader(
            client_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        client_dataloaders.append(client_loader)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    feature_schema = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_dummy_cols": encoder_meta["cat_dummy_cols"],
        "cat_categories": encoder_meta["cat_categories"],
        "feature_columns": feature_columns,
        "target_classes": target_classes,
        "task": task,
        "num_features": X_train.shape[1],
        "num_classes": len(target_classes) if task in ("binary", "multiclass") else 1,
        "client_num_mean": [m.tolist() for m in client_num_means],
        "client_num_std": [s.tolist() for s in client_num_stds],
    }
    logger.info("Feature schema: %s", feature_schema)

    return client_dataloaders, val_loader, test_loader, feature_schema

if __name__ == "__main__":
    client_dataloaders, val_loader, test_loader, feature_schema = load_dataset("/home/edgelab/ivo/tabular-GIA/tabular_gia/data/binary/adult/adult.csv", "/home/edgelab/ivo/tabular-GIA/tabular_gia/data/binary/adult/adult.yaml", 10)
