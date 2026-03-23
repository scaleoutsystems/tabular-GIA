from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from helper.summary_aggregation import (
    ATTACKS_CSV_FIELDS,
    FL_CSV_FIELDS,
    FL_STATS_CSV_FIELDS,
    ROUNDS_SUMMARY_CSV_FIELDS,
    ROUNDS_SUMMARY_STATS_CSV_FIELDS,
    RUN_SUMMARY_CSV_FIELDS,
    RUN_SUMMARY_STATS_CSV_FIELDS,
    SWEEP_RESULTS_CSV_FIELDS,
    SWEEP_RESULTS_PER_SEED_CSV_FIELDS,
    SeedAggregationResult,
)


@dataclass(frozen=True)
class TableSpec:
    filename: str
    fieldnames: tuple[str, ...]
    label: str
    mode: str
    string_fields: tuple[str, ...] = ()


class ResultsWriter:
    def _validate_unknown_fields(self, row: dict, fieldnames: tuple[str, ...], label: str) -> None:
        unknown = sorted(set(row.keys()) - set(fieldnames))
        if unknown:
            raise ValueError(f"Unexpected {label} fields: {unknown}")

    def _project_row(self, row: dict, fieldnames: tuple[str, ...], *, string_fields: set[str]) -> dict:
        def _normalize_value(field: str, value: object) -> object:
            if field in string_fields:
                return "" if value is None else value
            if value is None:
                return float("nan")
            if isinstance(value, str) and value.strip() == "":
                return float("nan")
            return value

        return {field: _normalize_value(field, row.get(field)) for field in fieldnames}

    def write_rows(
        self,
        out_path: Path,
        rows: list[dict],
        fieldnames: tuple[str, ...],
        *,
        label: str,
        mode: str = "w",
        string_fields: tuple[str, ...] = (),
    ) -> None:
        if not rows:
            return
        if mode not in {"w", "a"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected 'w' or 'a'.")

        projected_rows: list[dict] = []
        string_field_set = set(string_fields)
        for row in rows:
            self._validate_unknown_fields(row, fieldnames, label)
            projected_rows.append(
                self._project_row(
                    row,
                    fieldnames,
                    string_fields=string_field_set,
                )
            )

        write_header = mode == "w" or (mode == "a" and (not out_path.exists() or out_path.stat().st_size == 0))
        with open(out_path, mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            if write_header:
                writer.writeheader()
            for row in projected_rows:
                writer.writerow(row)

    def write_table(self, base_dir: Path, spec: TableSpec, rows: list[dict]) -> None:
        self.write_rows(
            base_dir / spec.filename,
            rows,
            spec.fieldnames,
            label=spec.label,
            mode=spec.mode,
            string_fields=spec.string_fields,
        )


class RunCsvWriter:
    _TABLES: dict[str, TableSpec] = {
        "fl": TableSpec("fl.csv", FL_CSV_FIELDS, "fl.csv", "a", string_fields=("phase",)),
        "attacks": TableSpec(
            "attacks.csv",
            ATTACKS_CSV_FIELDS,
            "attacks.csv",
            "a",
            string_fields=("attack_id", "attack_mode", "checkpoint_type", "checkpoint_label"),
        ),
        "rounds_summary": TableSpec("rounds_summary.csv", ROUNDS_SUMMARY_CSV_FIELDS, "rounds_summary.csv", "a"),
        "run_summary": TableSpec("run_summary.csv", RUN_SUMMARY_CSV_FIELDS, "run_summary.csv", "a"),
    }

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.writer = ResultsWriter()

    def _write(self, key: str, rows: list[dict]) -> None:
        self.writer.write_table(self.results_dir, self._TABLES[key], rows)

    def write_fl(self, payload: dict) -> None:
        self._write("fl", [payload])

    def write_attack(self, payload: dict) -> None:
        self._write("attacks", [payload])

    def write_round_summaries(self, rows: list[dict]) -> None:
        self._write("rounds_summary", rows)

    def write_run_summary(self, row: dict | None) -> None:
        if row is None:
            return
        self._write("run_summary", [row])


class SweepResultsWriter:
    _AGG_TABLES: dict[str, TableSpec] = {
        "fl": TableSpec("fl.csv", FL_CSV_FIELDS, "aggregated/fl.csv", "w", string_fields=("phase",)),
        "fl_stats": TableSpec(
            "fl_stats.csv",
            FL_STATS_CSV_FIELDS,
            "aggregated/fl_stats.csv",
            "w",
            string_fields=("phase",),
        ),
        "attacks": TableSpec(
            "attacks.csv",
            ATTACKS_CSV_FIELDS,
            "aggregated/attacks.csv",
            "w",
            string_fields=("attack_id", "attack_mode", "checkpoint_type", "checkpoint_label"),
        ),
        "rounds_summary": TableSpec("rounds_summary.csv", ROUNDS_SUMMARY_CSV_FIELDS, "aggregated/rounds_summary.csv", "w"),
        "rounds_summary_stats": TableSpec(
            "rounds_summary_stats.csv",
            ROUNDS_SUMMARY_STATS_CSV_FIELDS,
            "aggregated/rounds_summary_stats.csv",
            "w",
        ),
        "run_summary": TableSpec("run_summary.csv", RUN_SUMMARY_CSV_FIELDS, "aggregated/run_summary.csv", "w"),
        "run_summary_stats": TableSpec(
            "run_summary_stats.csv",
            RUN_SUMMARY_STATS_CSV_FIELDS,
            "aggregated/run_summary_stats.csv",
            "w",
        ),
    }
    _SWEEP_TABLES: dict[str, TableSpec] = {
        "sweep_results": TableSpec("sweep_results.csv", SWEEP_RESULTS_CSV_FIELDS, "sweep_results.csv", "w"),
        "sweep_results_per_seed": TableSpec(
            "sweep_results_per_seed.csv",
            SWEEP_RESULTS_PER_SEED_CSV_FIELDS,
            "sweep_results_per_seed.csv",
            "w",
        ),
    }

    def __init__(self) -> None:
        self.writer = ResultsWriter()

    def _write_agg(self, aggregated_dir: Path, key: str, rows: list[dict]) -> None:
        self.writer.write_table(aggregated_dir, self._AGG_TABLES[key], rows)

    def _write_sweep(self, experiment_dir: Path, key: str, rows: list[dict]) -> None:
        self.writer.write_table(experiment_dir, self._SWEEP_TABLES[key], rows)

    def write_aggregated_fl(self, aggregated_dir: Path, rows: list[dict]) -> None:
        self._write_agg(aggregated_dir, "fl", rows)

    def write_aggregated_attacks(self, aggregated_dir: Path, rows: list[dict]) -> None:
        self._write_agg(aggregated_dir, "attacks", rows)

    def write_aggregated_fl_stats(self, aggregated_dir: Path, rows: list[dict]) -> None:
        self._write_agg(aggregated_dir, "fl_stats", rows)

    def write_aggregated_rounds(self, aggregated_dir: Path, rows: list[dict]) -> None:
        self._write_agg(aggregated_dir, "rounds_summary", rows)

    def write_aggregated_rounds_stats(self, aggregated_dir: Path, rows: list[dict]) -> None:
        self._write_agg(aggregated_dir, "rounds_summary_stats", rows)

    def write_aggregated_run_summary(self, aggregated_dir: Path, row: dict) -> None:
        self._write_agg(aggregated_dir, "run_summary", [row])

    def write_aggregated_run_summary_stats(self, aggregated_dir: Path, row: dict) -> None:
        self._write_agg(aggregated_dir, "run_summary_stats", [row])

    def write_sweep_results(self, experiment_dir: Path, rows: list[dict]) -> None:
        self._write_sweep(experiment_dir, "sweep_results", rows)

    def write_sweep_results_per_seed(self, experiment_dir: Path, rows: list[dict]) -> None:
        self._write_sweep(experiment_dir, "sweep_results_per_seed", rows)

    def write_seed_aggregate(self, run_dir: Path, aggregate: SeedAggregationResult) -> None:
        aggregated_dir = run_dir / "aggregated"
        aggregated_dir.mkdir(parents=True, exist_ok=True)
        self.write_aggregated_fl(aggregated_dir, aggregate.fl_rows)
        self.write_aggregated_fl_stats(aggregated_dir, aggregate.fl_rows_stats)
        self.write_aggregated_attacks(aggregated_dir, aggregate.attack_rows)
        self.write_aggregated_rounds(aggregated_dir, aggregate.round_summaries)
        self.write_aggregated_rounds_stats(aggregated_dir, aggregate.round_summaries_stats)
        if aggregate.run_summary is not None:
            self.write_aggregated_run_summary(aggregated_dir, aggregate.run_summary)
        if aggregate.run_summary_stats is not None:
            self.write_aggregated_run_summary_stats(aggregated_dir, aggregate.run_summary_stats)
