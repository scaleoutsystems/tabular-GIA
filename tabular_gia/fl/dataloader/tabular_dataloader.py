import numpy as np
import random
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

from configs.dataset.dataset import DatasetConfig

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


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
        "cat_cardinalities": [len(cat_categories[c]) for c in cat_cols],
    }


def _fit_ordinal_encoders(
    X_train: pd.DataFrame,
    cat_cols: list[str],
) -> dict:
    cat_categories = {
        c: pd.Categorical(X_train[c].astype(object), ordered=False).categories.tolist() + ["__UNK__"]
        for c in cat_cols
    }
    cat_cardinalities = [len(cat_categories[c]) for c in cat_cols]
    return {
        "cat_categories": cat_categories,
        "cat_cardinalities": cat_cardinalities,
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


def _apply_ordinal_encoders(
    X: pd.DataFrame,
    meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    X_num = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0) if num_cols else pd.DataFrame(index=X.index)
    if cat_cols:
        X_cat_cols: list[pd.Series] = []
        for col in cat_cols:
            categories = meta["cat_categories"][col]
            known_categories = categories[:-1]
            x_col = X[col].astype(object).where(lambda s: s.isin(known_categories), "__UNK__")
            codes = pd.Categorical(x_col, categories=categories, ordered=False).codes
            X_cat_cols.append(pd.Series(codes.astype(np.int64), name=col, index=X.index))
        X_cat = pd.concat(X_cat_cols, axis=1) if X_cat_cols else pd.DataFrame(index=X.index)
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
    X: torch.Tensor,
    num_count: int,
    mean: torch.Tensor | np.ndarray | list[float],
    std: torch.Tensor | np.ndarray | list[float],
) -> torch.Tensor:
    """Denormalize the first num_count columns in a tabular tensor."""
    if num_count <= 0:
        return X
    out = X.clone()
    mean_t = torch.as_tensor(mean, dtype=out.dtype, device=out.device)
    std_t = torch.as_tensor(std, dtype=out.dtype, device=out.device)
    out[:, :num_count] = (out[:, :num_count] * std_t[:num_count]) + mean_t[:num_count]
    return out


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


def _clean_target_for_task(
    y: pd.Series,
    missing_values,
    task: str,
) -> tuple[pd.Series, pd.Series]:
    if missing_values is not None:
        y = y.replace(missing_values, pd.NA)

    task_norm = str(task).strip().lower()
    if task_norm in {"binary", "multiclass"}:
        keep_mask = y.notna()
        return y.loc[keep_mask].reset_index(drop=True), keep_mask

    if task_norm != "regression":
        raise ValueError(f"Unknown task '{task}'. Expected one of: binary, multiclass, regression.")

    y_num = pd.to_numeric(y, errors="coerce")
    keep_mask = y_num.notna()
    return y_num.loc[keep_mask].reset_index(drop=True), keep_mask


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


def _split_clients_iid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_clients: int,
    task: str,
    seed: int,
) -> list[tuple[pd.DataFrame, pd.Series]]:
    if num_clients == 1:
        return [(X_train, y_train)]

    if task in ("binary", "multiclass"):
        splitter = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_train, y_train)
    else:
        splitter = KFold(n_splits=num_clients, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_train)

    client_splits: list[tuple[pd.DataFrame, pd.Series]] = []
    for _, client_idx in split_iter:
        X_client = X_train.iloc[client_idx].reset_index(drop=True)
        y_client = y_train.iloc[client_idx].reset_index(drop=True)
        client_splits.append((X_client, y_client))
    return client_splits


def _split_clients_dirichlet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_clients: int,
    seed: int,
    alpha: float,
    min_client_samples: int = 1,
    max_attempts: int = 50,
) -> list[tuple[pd.DataFrame, pd.Series]]:
    if num_clients == 1:
        return [(X_train, y_train)]
    if alpha <= 0:
        raise ValueError(f"dirichlet_alpha must be > 0, got {alpha}")

    rng = np.random.default_rng(seed)
    y_codes = pd.Categorical(y_train, ordered=False).codes
    num_classes = int(np.max(y_codes)) + 1 if len(y_codes) > 0 else 0

    for _ in range(max_attempts):
        idx_per_client: list[list[int]] = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = np.where(y_codes == class_id)[0]
            rng.shuffle(class_indices)
            probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(class_indices), probs)
            start = 0
            for client_id, count in enumerate(counts):
                if count > 0:
                    idx_per_client[client_id].extend(class_indices[start : start + count].tolist())
                start += count

        sizes = [len(v) for v in idx_per_client]
        if sizes and min(sizes) >= min_client_samples:
            client_splits: list[tuple[pd.DataFrame, pd.Series]] = []
            for idxs in idx_per_client:
                rng.shuffle(idxs)
                X_client = X_train.iloc[idxs].reset_index(drop=True)
                y_client = y_train.iloc[idxs].reset_index(drop=True)
                client_splits.append((X_client, y_client))
            return client_splits

    raise RuntimeError(
        f"Failed to build Dirichlet client splits with alpha={alpha}, "
        f"num_clients={num_clients}, min_client_samples={min_client_samples} "
        f"after {max_attempts} attempts."
    )


def _cap_client_splits(
    client_splits: list[tuple[pd.DataFrame, pd.Series]],
    *,
    max_client_dataset_examples: int,
    seed: int,
) -> list[tuple[pd.DataFrame, pd.Series]]:
    capped_splits: list[tuple[pd.DataFrame, pd.Series]] = []
    for client_idx, (X_client, y_client) in enumerate(client_splits):
        if len(X_client) <= max_client_dataset_examples:
            capped_splits.append((X_client, y_client))
            continue
        rng = np.random.default_rng(seed + (client_idx * 1_000_003))
        selected = rng.permutation(len(X_client))[:max_client_dataset_examples]
        selected = np.asarray(selected, dtype=np.int64)
        X_capped = X_client.iloc[selected].reset_index(drop=True)
        y_capped = y_client.iloc[selected].reset_index(drop=True)
        capped_splits.append((X_capped, y_capped))
    return capped_splits


def load_dataset(
    dataset_cfg: DatasetConfig,
    num_clients: int,
    seed: int,
    encoding_mode: str,
    max_client_dataset_examples: int | None = None,
):
    """Load and return dataloaders for a dataset. Kwargs are optional parameters to set dataloader speedups:
        num_workers,
        pin_memory,
        persistent_workers,
    """
    dataset_path = Path(dataset_cfg.dataset_path)
    dataset_meta_path = Path(dataset_cfg.dataset_meta_path)

    # 1. load meta yaml and parse
    logger.info("Loading dataset config: meta=%s data=%s", dataset_meta_path, dataset_path)
    with open(dataset_meta_path, "r") as f:
        meta = yaml.safe_load(f) or {}

    # 1.1 handle no_header if present
    no_header = bool(meta["no_header"])
    if no_header:
        # 2. load csv dataset to pandas dataframe
        df = pd.read_csv(dataset_path, header=None)
    else:
        df = pd.read_csv(dataset_path)
    logger.info("Loaded data: rows=%d cols=%d no_header=%s", df.shape[0], df.shape[1], no_header)

    exclude_cols = meta["exclude_cols"] if "exclude_cols" in meta else []
    if exclude_cols is None:
        exclude_cols = []
    if len(exclude_cols) > 0:
        drop_cols = [str(col) for col in exclude_cols]
        missing_exclude_cols = [col for col in drop_cols if col not in df.columns]
        if len(missing_exclude_cols) > 0:
            raise ValueError(
                f"exclude_cols contains columns not found in dataset: {missing_exclude_cols}"
            )
        df = df.drop(columns=drop_cols)
        logger.info("Dropped excluded columns: count=%d cols=%s", len(drop_cols), drop_cols)

    # 3 perform target split
    target = meta["target"]
    task = str(meta["task"]).strip().lower()
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError(f"meta.task must be one of binary|multiclass|regression, got '{meta['task']}'")
    missing_values = meta["missing_values"]
    X_full, y_full = _split_target(df, target, no_header)
    logger.info("Target split: target=%s features=%d", target, X_full.shape[1])

    # 3.1 clean target according to declared task
    y, keep_mask = _clean_target_for_task(y_full, missing_values, task)
    # drop rows with missing targets
    X_full = X_full.loc[keep_mask].reset_index(drop=True)
    if keep_mask.size:
        dropped = int((~keep_mask).sum())
    else:
        dropped = 0
    logger.info("Task declared: %s | dropped_missing_targets=%d", task, dropped)

    # 4. split into train, val, test at ratios 70 / 15 / 15, stratify on target
    train_frac = dataset_cfg.train_frac
    val_frac = dataset_cfg.val_frac
    test_frac = dataset_cfg.test_frac
    stratify_labels = y if task in ("binary", "multiclass") else None

    test_path = meta["has_test_split"]
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
        y_test, keep_mask = _clean_target_for_task(y_test_split, missing_values, task)
        X_test_split = X_test_split.loc[keep_mask].reset_index(drop=True)
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
    encoding_mode = encoding_mode.strip().lower()
    if encoding_mode == "onehot":
        encoder_meta = _fit_encoders(X_train_split, num_cols, cat_cols, missing_values)
        X_train = _apply_encoders(X_train_split, encoder_meta, num_cols, cat_cols)
        X_val = _apply_encoders(X_val_split, encoder_meta, num_cols, cat_cols)
        X_test = _apply_encoders(X_test_split, encoder_meta, num_cols, cat_cols)
    elif encoding_mode == "ordinal":
        encoder_meta = _fit_ordinal_encoders(X_train_split, cat_cols)
        encoder_meta["cat_dummy_cols"] = []
        X_train = _apply_ordinal_encoders(X_train_split, encoder_meta, num_cols, cat_cols)
        X_val = _apply_ordinal_encoders(X_val_split, encoder_meta, num_cols, cat_cols)
        X_test = _apply_ordinal_encoders(X_test_split, encoder_meta, num_cols, cat_cols)
    else:
        raise ValueError(f"Unknown encoding_mode '{encoding_mode}'. Use 'onehot' or 'ordinal'.")
    feature_columns = X_train.columns.tolist()

    # global train stats for val/test normalization
    global_min = None
    global_max = None
    if num_cols:
        global_mean = X_train[num_cols].mean()
        global_std = X_train[num_cols].std(ddof=0).replace(0, 1e-6)
        global_min = X_train[num_cols].min()
        global_max = X_train[num_cols].max()
        X_val = _normalize_numeric(X_val, num_cols, global_mean, global_std)
        X_test = _normalize_numeric(X_test, num_cols, global_mean, global_std)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # 4.3 split train amongst num_clients
    partition_strategy = dataset_cfg.partition_strategy.strip().lower()
    if partition_strategy == "iid":
        client_splits = _split_clients_iid(X_train, y_train, num_clients, task, seed)
    elif partition_strategy == "dirichlet":
        if task not in ("binary", "multiclass"):
            logger.warning(
                "Dirichlet partition requested for task=%s. Falling back to IID split.",
                task,
            )
            client_splits = _split_clients_iid(X_train, y_train, num_clients, task, seed)
        else:
            dirichlet_alpha = dataset_cfg.dirichlet_alpha
            min_client_samples = dataset_cfg.min_client_samples
            max_attempts = dataset_cfg.dirichlet_max_attempts
            client_splits = _split_clients_dirichlet(
                X_train,
                y_train,
                num_clients,
                seed=seed,
                alpha=dirichlet_alpha,
                min_client_samples=min_client_samples,
                max_attempts=max_attempts,
            )
    else:
        raise ValueError(f"Unknown partition_strategy '{partition_strategy}'. Use 'iid' or 'dirichlet'.")

    if max_client_dataset_examples is not None and max_client_dataset_examples <= 0:
        raise ValueError(
            "max_client_dataset_examples must be > 0 when set, "
            f"got {max_client_dataset_examples}"
        )

    if max_client_dataset_examples is not None:
        client_splits = _cap_client_splits(
            client_splits,
            max_client_dataset_examples=max_client_dataset_examples,
            seed=seed,
        )
        logger.info(
            "Applied fixed client dataset cap: max_client_dataset_examples=%d",
            max_client_dataset_examples,
        )
    logger.info(
        "Client splits: strategy=%s clients=%d",
        partition_strategy,
        len(client_splits),
    )

    # 5. create dataloaders
    batch_size = dataset_cfg.batch_size
    num_workers = dataset_cfg.num_workers
    pin_memory = dataset_cfg.pin_memory
    persistent_workers = dataset_cfg.persistent_workers

    def _make_generator(loader_seed: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(loader_seed))
        return generator

    def _seed_worker_fn(loader_seed: int):
        def _seed_worker(worker_id: int) -> None:
            worker_seed = int(loader_seed) + int(worker_id)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return _seed_worker

    y_train_t = _encode_target(y_train, task, target_classes)
    y_val_t = _encode_target(y_val, task, target_classes)
    y_test_t = _encode_target(y_test, task, target_classes)

    _train_ds = TensorDataset(torch.tensor(X_train.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32), y_train_t)
    val_ds = TensorDataset(torch.tensor(X_val.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32), y_val_t)
    test_ds = TensorDataset(torch.tensor(X_test.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32), y_test_t)

    client_dataloaders = []
    client_num_means = []
    client_num_stds = []
    client_cat_probs: list[dict[str, list[float]]] = []
    for client_idx, (X_client, y_client) in enumerate(client_splits):
        if num_cols:
            client_mean = X_client[num_cols].mean()
            client_std = X_client[num_cols].std(ddof=0).replace(0, 1e-6)
            X_client = _normalize_numeric(X_client, num_cols, client_mean, client_std)
            client_num_means.append(client_mean.to_numpy(dtype=np.float32, copy=True))
            client_num_stds.append(client_std.to_numpy(dtype=np.float32, copy=True))
        else:
            client_num_means.append(np.array([], dtype=np.float32))
            client_num_stds.append(np.array([], dtype=np.float32))

        # Client-side categorical marginals are useful for prior baseline metrics.
        cat_probs_for_client: dict[str, list[float]] = {}
        if encoding_mode == "onehot":
            current_idx = len(num_cols)
            for col in cat_cols:
                cats = encoder_meta["cat_categories"][col]
                if isinstance(cats, dict):
                    cats = cats["categories"]
                n_cats = len(cats)
                if n_cats <= 0 or current_idx + n_cats > X_client.shape[1]:
                    continue
                probs = X_client.iloc[:, current_idx : current_idx + n_cats].mean(axis=0).to_numpy(dtype=np.float32)
                total = float(np.sum(probs))
                if total > 0:
                    probs = probs / total
                else:
                    probs = np.full(n_cats, 1.0 / n_cats, dtype=np.float32)
                cat_probs_for_client[col] = probs.tolist()
                current_idx += n_cats
        else:
            for col in cat_cols:
                cats = encoder_meta["cat_categories"][col]
                n_cats = len(cats)
                if n_cats <= 0:
                    continue
                codes = pd.to_numeric(X_client[col], errors="coerce").to_numpy()
                codes = np.rint(codes).astype(np.int64, copy=False)
                codes = np.clip(codes, 0, n_cats - 1)
                counts = np.bincount(codes, minlength=n_cats).astype(np.float32, copy=False)
                total = float(max(1, codes.shape[0]))
                probs = counts / total
                cat_probs_for_client[col] = probs.tolist()
        client_cat_probs.append(cat_probs_for_client)

        y_client_t = _encode_target(y_client, task, target_classes)
        client_ds = TensorDataset(torch.tensor(X_client.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32), y_client_t)
        if len(client_ds) < batch_size:
            raise ValueError(
                f"Client {client_idx} has {len(client_ds)} samples, smaller than batch_size={batch_size}. "
                "With drop_last=True this client would produce zero batches. "
                "Reduce batch_size, reduce num_clients, or increase client data."
            )
        client_loader = DataLoader(
            client_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=_seed_worker_fn(seed + (client_idx * 1_000_003)) if num_workers > 0 else None,
            generator=_make_generator(seed + (client_idx * 1_000_003)),
        )
        client_dataloaders.append(client_loader)

    val_seed = seed + 2_000_003
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_seed_worker_fn(val_seed) if num_workers > 0 else None,
        generator=_make_generator(val_seed),
    )
    test_seed = seed + 3_000_003
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_seed_worker_fn(test_seed) if num_workers > 0 else None,
        generator=_make_generator(test_seed),
    )

    feature_schema = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_dummy_cols": encoder_meta["cat_dummy_cols"],
        "cat_categories": encoder_meta["cat_categories"],
        "cat_cardinalities": encoder_meta["cat_cardinalities"],
        "feature_columns": feature_columns,
        "target_classes": target_classes,
        "task": task,
        "num_features": X_train.shape[1],
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "encoding_mode": encoding_mode,
        "num_classes": len(target_classes) if task in ("binary", "multiclass") else 1,
        "client_num_mean": [m.tolist() for m in client_num_means],
        "client_num_std": [s.tolist() for s in client_num_stds],
        "train_num_min": global_min.tolist() if num_cols else [],
        "train_num_max": global_max.tolist() if num_cols else [],
        "client_cat_probs": client_cat_probs,
    }
    logger.info("Feature schema: %s", feature_schema)

    return client_dataloaders, val_loader, test_loader, feature_schema

if __name__ == "__main__":
    dataset_cfg = DatasetConfig(dataset_path="data/binary/adult/adult.csv", dataset_meta_path="data/binary/adult/adult.yaml")
    client_dataloaders, val_loader, test_loader, feature_schema = load_dataset(
        dataset_cfg,
        num_clients=10,
        seed=42,
        encoding_mode="onehot",
    )
