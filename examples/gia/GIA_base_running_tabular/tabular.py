"""Generic tabular dataloader for CSVs with a header row."""

from pathlib import Path
from typing import Optional, Any

import logging
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger(__name__)
if not logger.handlers:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def load_tabular_config(config_path: str | Path) -> dict[str, Any]:
	"""Load config and resolve data/meta paths and trainer/model settings."""
	config_path = Path(config_path)
	with open(config_path, "r") as f:
		cfg_all = yaml.safe_load(f) or {}

	ds_cfg = cfg_all.get("dataset", {})
	trainer_cfg = cfg_all.get("trainer", {})
	model_cfg = cfg_all.get("model", {})

	if not ds_cfg:
		raise ValueError("`dataset` section missing in config file")

	root = config_path.parent
	data_path = ds_cfg.get("data_path")
	meta_path = ds_cfg.get("meta_path")
	if data_path is not None:
		data_path = Path(data_path)
		if not data_path.is_absolute():
			data_path = root / data_path
	if meta_path is not None:
		meta_path = Path(meta_path)
		if not meta_path.is_absolute():
			meta_path = root / meta_path

	logger.info(
		"Loaded tabular config: data_path=%s batch_size=%s client_frac=%.3f train_frac=%.3f val_frac=%.3f",
		data_path,
		ds_cfg.get("batch_size", 64),
		ds_cfg.get("client_frac", 0.10),
		ds_cfg.get("train_frac", 0.70),
		ds_cfg.get("val_frac", 0.15),
	)

	return {
		"data_path": data_path,
		"meta_path": meta_path,
		"batch_size": ds_cfg.get("batch_size", 64),
		"seed": ds_cfg.get("seed", 42),
		"normalize_numeric": ds_cfg.get("normalize_numeric", True),
		"one_hot_categoricals": ds_cfg.get("one_hot_categoricals", True),
		"client_frac": ds_cfg.get("client_frac", 0.10),
		"train_frac": ds_cfg.get("train_frac", 0.70),
		"val_frac": ds_cfg.get("val_frac", 0.15),
		"trainer": trainer_cfg,
		"model": model_cfg,
	}


def _load_dataset_meta(data_path: Path, meta_path: Optional[Path]) -> dict[str, Any]:
	"""Load dataset meta."""
	if meta_path is None:
		return {}
	meta_path = Path(meta_path)
	if not meta_path.exists():
		raise FileNotFoundError(f"Meta path not found: {meta_path}")
	with open(meta_path, "r") as f:
		return yaml.safe_load(f) or {}


def _infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
	"""Find which feature columns are categorical vs numeric."""
	cat_cols = [c for c in X.columns if str(X[c].dtype) == "object" or str(X[c].dtype).startswith("category")]
	num_cols = [c for c in X.columns if c not in cat_cols]
	return num_cols, cat_cols


def _fit_encoders(X_train: pd.DataFrame, num_cols: list[str], cat_cols: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
	"""Fit numeric mean/std and categorical dummy/categories on the train split."""
	missing_token = cfg.get("missing_token") or ""
	cat_dummy_cols = pd.get_dummies(X_train[cat_cols].fillna(missing_token), drop_first=False).columns.tolist() if cat_cols else []
	cat_categories = {
		c: pd.Categorical(X_train[c].fillna(missing_token), ordered=False).categories.tolist() for c in cat_cols
	}
	return {
		"num_cols": num_cols,
		"cat_cols": cat_cols,
		"num_mean": X_train[num_cols].mean() if num_cols else pd.Series(dtype=float),
		"num_std": (X_train[num_cols].std().replace(0, 1e-6) if num_cols else pd.Series(dtype=float)),
		"cat_dummy_cols": cat_dummy_cols,
		"cat_categories": cat_categories,
	}


def _apply_encoders(X: pd.DataFrame, meta: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
	"""Normalize numerics and encode categoricals to match train."""
	num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
	missing_token = cfg.get("missing_token") or ""
	
	# normalization (fill missing with 0 before centering)
	X_num = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0) if num_cols else pd.DataFrame(index=X.index)
	if num_cols and cfg.get("normalize_numeric", True):
		X_num = (X_num - meta["num_mean"]) / meta["num_std"]

	if cat_cols:
		X_cat = X[cat_cols].fillna(missing_token)
		if cfg.get("one_hot_categoricals", True):
			X_cat = pd.get_dummies(X_cat, drop_first=False).reindex(columns=meta["cat_dummy_cols"], fill_value=0)
		else:
			mapped = {c: pd.Categorical(X_cat[c], categories=meta.get("cat_categories", {}).get(c)).codes for c in cat_cols}
			X_cat = pd.DataFrame(mapped, index=X.index)
	else:
		X_cat = pd.DataFrame(index=X.index)

	return pd.concat([X_num, X_cat], axis=1) if len(X_num.columns) and len(X_cat.columns) else (X_num if len(X_num.columns) else X_cat)


def _encode_target_train(y: pd.Series, mode: str, provided_classes: list[Any] | None = None) -> tuple[torch.Tensor, int, list[Any] | None, str]:
	"""Encode training targets using the pre-decided mode and classes."""
	if mode == "classification":
		cats = provided_classes or pd.Categorical(y, ordered=False).categories.tolist()
		codes = pd.Categorical(y, categories=cats).codes
		return torch.tensor(codes, dtype=torch.long), len(cats), cats, mode

	y_num = pd.to_numeric(y, errors="coerce").fillna(0)
	return torch.tensor(y_num.values, dtype=torch.float32), 1, None, mode


def _encode_target_apply(y: pd.Series, target_classes: list[str] | None, mode: str) -> torch.Tensor:
	"""Encode eval targets with the training mapping"""
	if mode == "regression":
		y_num = pd.to_numeric(y, errors="coerce").fillna(0)
		return torch.tensor(y_num.values, dtype=torch.float32)
	encoded = pd.Categorical(y, categories=target_classes).codes
	if (encoded == -1).any():
		raise ValueError("Unseen label encountered in evaluation split; extend target_classes or fix data")
	return torch.tensor(encoded, dtype=torch.long)


def _log_target_stats(y: pd.Series, target_col: str, mode: str) -> None:
	"""Log target distribution before splitting."""
	try:
		if mode == "classification":
			counts = y.value_counts(dropna=False)
			total = len(y)
			logger.info("Target stats [%s] (classification): classes=%d", target_col, len(counts))
			for cls, cnt in counts.items():
				pct = cnt / max(1, total)
				logger.info("  class=%s count=%d pct=%.3f", cls, cnt, pct)
		else:
			y_num = pd.to_numeric(y, errors="coerce")
			desc = y_num.describe(percentiles=[0.25, 0.5, 0.75])
			logger.info(
				"Target stats [%s] (regression): count=%d mean=%.4f std=%.4f min=%.4f p25=%.4f median=%.4f p75=%.4f max=%.4f",
				target_col,
				desc["count"],
				desc.get("mean", float("nan")),
				desc.get("std", float("nan")),
				desc.get("min", float("nan")),
				desc.get("25%", float("nan")),
				desc.get("50%", float("nan")),
				desc.get("75%", float("nan")),
				desc.get("max", float("nan")),
			)
	except Exception:
		logger.warning("Could not compute target stats for %s", target_col)


def _read_frame(path: Path, target_col: Any, no_header: bool, read_kwargs: dict[str, Any]) -> pd.DataFrame:
	"""Read CSV and verify the target column exists."""
	df = pd.read_csv(path, **read_kwargs)
	if no_header:
		if not isinstance(target_col, int):
			raise ValueError("For headerless CSVs (`no_header: true`), `target` must be an integer column index (0-based).")
		if target_col < 0 or target_col >= df.shape[1]:
			raise ValueError(f"Target column index {target_col} is out of bounds for CSV with {df.shape[1]} columns.")
	else:
		if target_col not in df.columns:
			raise ValueError(f"Target column '{target_col}' not found in CSV header.")
	return df


def _resolve_test_path(data_path: Path, test_path: Any) -> Path:
	"""Resolve test path relative to the data file when needed."""
	test_path_raw = Path(test_path)
	if test_path_raw.is_absolute() or test_path_raw.exists():
		return test_path_raw
	return data_path.parent / test_path_raw


def _impute_features(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], missing_values: Any) -> pd.DataFrame:
	"""Impute features: numerics -> 0, categoricals -> missing token."""
	df = df.copy()
	if num_cols:
		df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
	if cat_cols:
		df[cat_cols] = df[cat_cols].fillna(missing_values)
	return df


def _prepare_target(y: pd.Series, missing_values: Any) -> tuple[pd.Series, str, list[Any] | None]:
	"""Impute target and decide classification vs regression."""
	if str(y.dtype) == "object" or str(y.dtype).startswith("category"):
		y = y.fillna(missing_values)
		return y, "classification", pd.Categorical(y, ordered=False).categories.tolist()

	y_num = pd.to_numeric(y, errors="coerce").fillna(0)
	if y_num.nunique(dropna=False) <= 20:
		cats = pd.Categorical(y_num, ordered=False).categories.tolist()
		return y_num, "classification", cats
	return y_num, "regression", None


def load_tabular_dataset(
	cfg: dict[str, Any],
	encoder_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
	"""Load data, split, encode, and return datasets plus metadata."""
	# read dataset meta + CSVs
	data_path = cfg.get("data_path")
	if data_path is None:
		raise ValueError("`data_path` is required in the config")

	data_path = Path(data_path)
	if not data_path.exists():
		raise FileNotFoundError(f"Data path not found: {data_path}")

	meta = _load_dataset_meta(data_path, cfg.get("meta_path"))
	no_header = bool(meta.get("no_header", False))
	target_col = meta.get("target")
	test_path = meta.get("has_test_split")
	if target_col is None:
		raise ValueError("No target column provided; set cfg.target or provide a YAML with `target:`.")

	missing_values = meta.get("missing_values")
	if missing_values is None:
		raise ValueError("`missing_values` must be specified in the dataset meta YAML to mark missing entries.")
	# stash token for downstream helpers
	cfg = {**cfg, "missing_token": missing_values}
	read_kwargs = {"na_values": [missing_values]} if missing_values is not None else {}
	if no_header:
		read_kwargs["header"] = None

	df = _read_frame(data_path, target_col, no_header, read_kwargs)
	test_df = None
	if test_path is not None:
		test_path_resolved = _resolve_test_path(data_path, test_path)
		if not test_path_resolved.exists():
			raise FileNotFoundError(f"Provided test split not found: {test_path_resolved}")
		test_df = _read_frame(test_path_resolved, target_col, no_header, read_kwargs)

	# split target vs features
	target_label = str(target_col)
	y = df[target_col]
	X = df.drop(columns=[target_col])
	y_test = test_df[target_col] if test_df is not None else None
	X_test = test_df.drop(columns=[target_col]) if test_df is not None else None

	# infer dtypes and impute features
	num_cols, cat_cols = _infer_column_types(X)
	X = _impute_features(X, num_cols, cat_cols, missing_values)
	X_test = _impute_features(X_test, num_cols, cat_cols, missing_values) if X_test is not None else None

	# decide task mode and clean targets
	y, mode, target_classes_full = _prepare_target(y, missing_values)
	if y_test is not None:
		y_test = y_test.fillna(missing_values) if mode == "classification" else pd.to_numeric(y_test, errors="coerce").fillna(0)

	_log_target_stats(y, target_label, mode)

	logger.info("Loaded data: %d rows, %d cols (after minimal imputation)", df.shape[0], df.shape[1])

	# split client/global then train/val/test
	is_regression = mode == "regression"
	seed = cfg.get("seed", 42)
	client_frac = cfg.get("client_frac", 0.1)
	global_idx, client_idx = train_test_split(y.index, test_size=client_frac, random_state=seed, stratify=None if is_regression else y)
	X_global, y_global = X.loc[global_idx], y.loc[global_idx]
	X_client, y_client = X.loc[client_idx], y.loc[client_idx]
	logger.info("Client/global split: client=%d global=%d (client_frac=%.3f)", len(client_idx), len(global_idx), client_frac)

	use_external_test = test_df is not None

	if use_external_test:
		train_idx, val_idx = train_test_split(y_global.index, test_size=0.15, random_state=seed, stratify=None if is_regression else y_global)
		X_train, y_train = X_global.loc[train_idx], y_global.loc[train_idx]
		X_val, y_val = X_global.loc[val_idx], y_global.loc[val_idx]
		logger.info("Global split with external test: train=%d val=%d", len(train_idx), len(val_idx))
	else:
		train_frac = cfg.get("train_frac", 0.70)
		val_frac = cfg.get("val_frac", 0.15)
		if train_frac + val_frac > 1.0 + 1e-6:
			raise ValueError("train_frac + val_frac must be <= 1.0 over the global pool")
		train_idx, temp_idx = train_test_split(y_global.index, test_size=1 - train_frac, random_state=seed, stratify=None if is_regression else y_global)
		X_train, y_train = X_global.loc[train_idx], y_global.loc[train_idx]
		X_temp, y_temp = X_global.loc[temp_idx], y_global.loc[temp_idx]
		remaining_frac = max(1e-8, 1 - train_frac)
		val_ratio = min(1.0, val_frac / remaining_frac)
		val_idx, test_idx = train_test_split(y_temp.index, test_size=1 - val_ratio, random_state=seed, stratify=None if is_regression else y_temp)
		X_val, y_val = X_temp.loc[val_idx], y_temp.loc[val_idx]
		X_test, y_test = X_temp.loc[test_idx], y_temp.loc[test_idx]
		logger.info("Val/test split: val=%d test=%d", len(val_idx), len(test_idx))

	if encoder_meta is None:
		encoder_meta = _fit_encoders(X_train, num_cols, cat_cols, cfg)
		logger.info(
			"Fitted encoders: num_cols=%d cat_cols=%d one_hot=%s",
			len(num_cols),
			len(cat_cols),
			cfg.get("one_hot_categoricals", True),
		)
	else:
		# ensure we use provided meta; override inferred columns
		num_cols = encoder_meta.get("num_cols", num_cols)
		cat_cols = encoder_meta.get("cat_cols", cat_cols)
		logger.info(
			"Using provided encoder_meta: num_cols=%d cat_cols=%d one_hot=%s",
			len(num_cols),
			len(cat_cols),
			cfg.get("one_hot_categoricals", True),
		)

	# encode features
	train_proc = _apply_encoders(X_train, encoder_meta, cfg)
	val_proc = _apply_encoders(X_val, encoder_meta, cfg)
	test_proc = _apply_encoders(X_test, encoder_meta, cfg) if X_test is not None else pd.DataFrame(index=[])
	client_proc = _apply_encoders(X_client, encoder_meta, cfg)

	feature_columns = train_proc.columns.tolist()
	encoder_meta["feature_columns"] = feature_columns

	# pandas with mixed dtypes (float + uint8/bool) yields object arrays; coerce to float32 explicitly
	train_features = torch.tensor(train_proc.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
	val_features = torch.tensor(val_proc.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
	test_features = torch.tensor(test_proc.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
	client_features = torch.tensor(client_proc.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)

	data_mean = train_features.mean(dim=0)
	data_std = train_features.std(dim=0).clamp(min=1e-6)

	# encode targets
	y_train_t, num_classes, target_classes, target_mode = _encode_target_train(y_train, mode, provided_classes=target_classes_full)
	y_val_t = _encode_target_apply(y_val, target_classes, target_mode)
	y_test_t = _encode_target_apply(y_test, target_classes, target_mode) if y_test is not None else torch.tensor([])
	y_client_t = _encode_target_apply(y_client, target_classes, target_mode)

	# For binary classification, use float column vectors so single-logit BCE heads match label shape
	if target_mode == "classification" and num_classes == 2:
		y_train_t = y_train_t.float().view(-1, 1)
		y_val_t = y_val_t.float().view(-1, 1)
		y_test_t = y_test_t.float().view(-1, 1) if y_test_t.numel() else y_test_t
		y_client_t = y_client_t.float().view(-1, 1)

	encoder_meta["target_classes"] = target_classes
	encoder_meta["target_mode"] = target_mode

	return {
		"train_ds": TensorDataset(train_features, y_train_t),
		"val_ds": TensorDataset(val_features, y_val_t),
		"test_ds": TensorDataset(test_features, y_test_t),
		"client_ds": TensorDataset(client_features, y_client_t),
		"data_mean": data_mean,
		"data_std": data_std,
		"n_features": train_features.shape[1],
		"num_classes": num_classes,
		"encoder_meta": encoder_meta,
	}


def get_tabular_loaders(
	cfg: Optional[dict[str, Any]] = None,
	encoder_meta: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Create DataLoaders and metadata for tabular data.

	Returns (splits_dict, encoder_meta)
	"""

	cfg = cfg or {}
	result = load_tabular_dataset(cfg, encoder_meta=encoder_meta)

	loaders: dict[str, Any] = {
		"train_loader": DataLoader(result["train_ds"], batch_size=cfg.get("batch_size", 64), shuffle=True),
		"val_loader": DataLoader(result["val_ds"], batch_size=cfg.get("batch_size", 64), shuffle=False) if result["val_ds"] is not None else None,
		"test_loader": DataLoader(result["test_ds"], batch_size=cfg.get("batch_size", 64), shuffle=False) if result["test_ds"] is not None else None,
		"client_loader": DataLoader(result["client_ds"], batch_size=cfg.get("batch_size", 64), shuffle=True),
		"data_mean": result["data_mean"],
		"data_std": result["data_std"],
		"n_features": result["n_features"],
		"num_classes": result["num_classes"],
	}

	return loaders, result["encoder_meta"]