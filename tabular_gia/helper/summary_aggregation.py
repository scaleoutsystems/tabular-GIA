from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Protocol

import numpy as np

from tabular_gia.fl.metrics.fl_metrics import FL_METRIC_FIELDS
from tabular_gia.metrics.tabular_metrics import ATTACK_METRIC_FIELDS

ATTACKS_CSV_FIELDS = (
    "attack_id",
    "round",
    "client_idx",
    "attack_mode",
    "fixed_batch_id",
    "checkpoint_type",
    "checkpoint_label",
    "exp_min",
    "exp_avg",
    "exp_max",
    "client_exp",
    "row_count",
    *ATTACK_METRIC_FIELDS,
)


ROUNDS_SUMMARY_CSV_FIELDS = (
    "round",
    "num_clients",
    "total_rows",
    "exp_min",
    "exp_avg",
    "exp_max",
    *ATTACK_METRIC_FIELDS,
)

RUN_SUMMARY_CSV_FIELDS = (
    "num_rounds",
    "total_rows",
    *ATTACK_METRIC_FIELDS,
)

FL_CSV_FIELDS = (
    "phase",
    "round",
    "exp_min",
    "exp_avg",
    "exp_max",
    *(f"{split}_{metric}" for split in ("train", "val", "test") for metric in FL_METRIC_FIELDS),
)

SWEEP_RESULTS_PER_SEED_CSV_FIELDS = (
    "run_id",
    "seed",
    *RUN_SUMMARY_CSV_FIELDS,
)

SWEEP_RESULTS_CSV_FIELDS = (
    "run_id",
    "num_seeds",
    *RUN_SUMMARY_CSV_FIELDS,
)


def _weighted_metric_means(
    records: List[Dict],
    weights: np.ndarray,
    excluded_keys: set[str],
) -> Dict[str, float]:
    metric_keys = [k for k in records[0].keys() if k not in excluded_keys]
    means: Dict[str, float] = {}
    for key in metric_keys:
        values = np.array([record[key] for record in records], dtype=float)
        valid = ~np.isnan(values)
        if not valid.any():
            means[key] = float("nan")
            continue
        if key == "nn_min":
            means[key] = float(np.min(values[valid]))
            continue
        w_valid = weights[valid]
        denom = float(np.sum(w_valid))
        if denom <= 0:
            means[key] = float("nan")
            continue
        means[key] = float(np.sum(values[valid] * w_valid) / denom)
    return means


def summarize_round(
    metrics_list: List[Dict],
    round_idx: int,
) -> Dict | None:
    if not metrics_list:
        return None

    rows = np.array([metric["row_count"] for metric in metrics_list], dtype=float)
    total_rows = float(rows.sum()) if rows.size else 0.0
    weights = rows / total_rows if total_rows > 0 else np.zeros_like(rows)
    metric_means = _weighted_metric_means(metrics_list, weights, {"client_idx", "row_count"})

    return {
        "round": int(round_idx),
        "num_clients": int(len(metrics_list)),
        "total_rows": int(total_rows),
        **metric_means,
    }


def summarize_run(round_summaries: List[Dict]) -> Dict | None:
    if not round_summaries:
        return None
    rows = np.array([summary["total_rows"] for summary in round_summaries], dtype=float)
    total_rows = float(rows.sum()) if rows.size else 0.0
    weights = rows / total_rows if total_rows > 0 else np.zeros_like(rows)
    metric_means = _weighted_metric_means(
        round_summaries,
        weights,
        {"round", "num_clients", "total_rows"},
    )
    return {
        "num_rounds": int(len(round_summaries)),
        "total_rows": int(total_rows),
        **metric_means,
    }


class RunSummaryBuilder:
    def build_round_summaries(self, attack_rows: list[dict]) -> list[dict]:
        round_attack_rows: dict[int, list[dict]] = {}
        for attack_row in attack_rows:
            round_idx = int(attack_row["round"])
            if round_idx not in round_attack_rows:
                round_attack_rows[round_idx] = []
            round_attack_rows[round_idx].append(attack_row)

        round_summaries: list[dict] = []
        for round_idx in sorted(round_attack_rows):
            metrics_list = round_attack_rows[round_idx]
            metric_rows = [
                {"row_count": row["row_count"], **{k: row[k] for k in ROUNDS_SUMMARY_CSV_FIELDS if k in row}}
                for row in metrics_list
            ]
            round_summary = summarize_round(metric_rows, round_idx)
            if round_summary is not None:
                round_summaries.append(round_summary)
        return round_summaries

    def build_run_summary(self, round_summaries: list[dict]) -> dict | None:
        run_summary = summarize_run(round_summaries)
        if run_summary is not None:
            run_summary = {field: run_summary[field] for field in RUN_SUMMARY_CSV_FIELDS if field in run_summary}
        return run_summary


@dataclass(frozen=True)
class SeedAggregationResult:
    fl_rows: list[dict]
    attack_rows: list[dict]
    round_summaries: list[dict]
    run_summary: dict | None


class SeedRunResult(Protocol):
    fl_rows: list[dict]
    attack_rows: list[dict]
    round_summaries: list[dict]


class SeedSummaryBuilder:
    def _to_float(self, value: object) -> float | None:
        if value is None:
            return None
        raw = str(value).strip()
        if raw == "":
            return None
        try:
            parsed = float(raw)
        except ValueError:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    def _aggregate_rows(
        self,
        *,
        rows: list[dict[str, str]],
        fieldnames: tuple[str, ...],
        group_keys: tuple[str, ...],
        int_fields: set[str],
        weight_field: str | None = None,
        sum_fields: set[str] | None = None,
    ) -> list[dict]:
        if not rows:
            return []
        sum_fields = set() if sum_fields is None else set(sum_fields)
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
        for row in rows:
            key = tuple(str(row.get(k, "")) for k in group_keys)
            grouped.setdefault(key, []).append(row)

        aggregated: list[dict] = []
        for key, bucket in grouped.items():
            out: dict[str, object] = {k: key[i] for i, k in enumerate(group_keys)}
            weights_raw = [self._to_float(r.get(weight_field)) for r in bucket] if weight_field is not None else None
            for field in fieldnames:
                if field in group_keys:
                    continue
                vals: list[float] = []
                wts: list[float] = []
                for idx, row in enumerate(bucket):
                    parsed = self._to_float(row.get(field))
                    if parsed is None:
                        continue
                    vals.append(parsed)
                    if weights_raw is not None:
                        weight = weights_raw[idx]
                        wts.append(1.0 if weight is None or weight <= 0 else float(weight))

                if vals:
                    if field in sum_fields:
                        agg_value = float(sum(vals))
                    elif weight_field is not None and field != weight_field:
                        denom = float(sum(wts))
                        agg_value = (
                            float(sum(v * w for v, w in zip(vals, wts)) / denom)
                            if denom > 0
                            else float(sum(vals) / len(vals))
                        )
                    else:
                        agg_value = float(sum(vals) / len(vals))
                    out[field] = int(round(agg_value)) if field in int_fields else agg_value
                    continue

                first_non_empty = next((str(row.get(field, "")).strip() for row in bucket if str(row.get(field, "")).strip()), "")
                out[field] = first_non_empty
            aggregated.append(out)

        def _sort_value(v: object):
            parsed = self._to_float(v)
            if parsed is not None:
                return (0, parsed)
            return (1, str(v))

        aggregated.sort(key=lambda row: tuple(_sort_value(row.get(k, "")) for k in group_keys))
        return aggregated

    def build_seed_aggregate(self, run_results: list[SeedRunResult]) -> SeedAggregationResult:
        fl_rows_raw: list[dict] = []
        attack_rows_raw: list[dict] = []
        round_rows_raw: list[dict] = []
        for result in run_results:
            fl_rows_raw.extend(result.fl_rows)
            attack_rows_raw.extend(result.attack_rows)
            round_rows_raw.extend(result.round_summaries)

        fl_rows_non_final = self._aggregate_rows(
            rows=[row for row in fl_rows_raw if str(row.get("phase", "")).strip() != "final_test"],
            fieldnames=FL_CSV_FIELDS,
            group_keys=("phase", "round"),
            int_fields={"round"},
            weight_field=None,
        )
        fl_rows_final = self._aggregate_rows(
            rows=[row for row in fl_rows_raw if str(row.get("phase", "")).strip() == "final_test"],
            fieldnames=FL_CSV_FIELDS,
            group_keys=("phase",),
            int_fields=set(),
            weight_field=None,
        )
        for row in fl_rows_final:
            row["round"] = -1
        fl_rows = fl_rows_non_final + fl_rows_final
        fl_rows.sort(key=lambda row: (str(row.get("phase", "")), float(self._to_float(row.get("round")) or 0.0)))

        attack_rows = self._aggregate_rows(
            rows=attack_rows_raw,
            fieldnames=ATTACKS_CSV_FIELDS,
            group_keys=("round", "client_idx", "attack_mode", "fixed_batch_id", "checkpoint_type", "checkpoint_label"),
            int_fields={"attack_id", "round", "client_idx", "fixed_batch_id", "row_count"},
            weight_field="row_count",
            sum_fields={"row_count"},
        )
        for idx, row in enumerate(attack_rows, start=1):
            row["attack_id"] = int(idx)

        round_summaries = self._aggregate_rows(
            rows=round_rows_raw,
            fieldnames=ROUNDS_SUMMARY_CSV_FIELDS,
            group_keys=("round",),
            int_fields={"round", "num_clients", "total_rows"},
            weight_field="total_rows",
            sum_fields={"total_rows"},
        )

        run_summary = summarize_run(round_summaries)
        if run_summary is not None:
            run_summary = {field: run_summary[field] for field in RUN_SUMMARY_CSV_FIELDS if field in run_summary}

        return SeedAggregationResult(
            fl_rows=fl_rows,
            attack_rows=attack_rows,
            round_summaries=round_summaries,
            run_summary=run_summary,
        )
