"""Batch runner for statistical evaluation of the QFL malicious-client detector.

Run from the project root, for example:

    python -m tests.run_qfl_statistics --config tests/qfl_statistics_quick.json --overwrite

The runner executes every scenario for every seed, randomly selects malicious
client IDs in each run, stores per-run/client/round CSV data, and creates
aggregate summary plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tests.qfl_statistical_experiment import (
    ExperimentConfig,
    make_run_id,
    run_experiment,
)


RUN_METRICS_FILENAME = "run_metrics.csv"
CLIENT_METRICS_FILENAME = "client_metrics.csv"
ROUND_METRICS_FILENAME = "round_metrics.csv"
AGGREGATE_FILENAME = "aggregate_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated QFL malicious-client detection experiments and "
            "produce statistical CSV files and plots."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON experiment configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output-directory override. Useful for benchmarking the "
            "fast implementation without deleting existing results."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory and start from the beginning.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs and continue an interrupted batch.",
    )
    parser.add_argument(
        "--verbose-clients",
        action="store_true",
        help="Print one detailed line per client per communication round.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Run only a named scenario. This option may be repeated. "
            "Without it, all scenarios in the JSON file are run."
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit for testing the batch runner.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned runs without training any models.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit("The configuration root must be a JSON object")
    return data


def resolve_output_dir(config_data: Mapping[str, Any]) -> Path:
    raw = str(config_data.get("output_dir", "results/qfl_statistics"))
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_experiment_configs(
    config_data: Mapping[str, Any],
    selected_scenarios: Sequence[str],
) -> List[ExperimentConfig]:
    common = config_data.get("common", {})
    scenarios = config_data.get("scenarios")
    seeds = config_data.get("seeds")

    if not isinstance(common, dict):
        raise SystemExit("'common' must be a JSON object")
    if not isinstance(scenarios, list) or not scenarios:
        raise SystemExit("'scenarios' must be a non-empty JSON list")
    if not isinstance(seeds, list) or not seeds:
        raise SystemExit("'seeds' must be a non-empty JSON list")

    selected = set(selected_scenarios)
    configs: List[ExperimentConfig] = []
    seen_names = set()

    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise SystemExit("Every scenario must be a JSON object")

        name = str(scenario.get("scenario_name", "")).strip()
        if not name:
            raise SystemExit("Every scenario requires a non-empty 'scenario_name'")
        if name in seen_names:
            raise SystemExit(f"Duplicate scenario_name: {name}")
        seen_names.add(name)

        if selected and name not in selected:
            continue

        merged = dict(common)
        merged.update(scenario)

        for seed_value in seeds:
            run_values = dict(merged)
            run_values["seed"] = int(seed_value)
            try:
                config = ExperimentConfig(**run_values)
                config.validate()
            except TypeError as exc:
                raise SystemExit(
                    f"Unsupported or missing field in scenario '{name}': {exc}"
                ) from exc
            except ValueError as exc:
                raise SystemExit(f"Invalid scenario '{name}': {exc}") from exc
            configs.append(config)

    if selected:
        missing = selected - seen_names
        if missing:
            raise SystemExit(
                "Unknown scenario name(s): " + ", ".join(sorted(missing))
            )

    if not configs:
        raise SystemExit("No experiment runs were selected")
    return configs


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=True)
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return

    fieldnames = list(rows[0].keys())
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_completed_run_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return completed

    for result_file in runs_dir.glob("*/run_metrics.json"):
        try:
            with result_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            run_id = data.get("run_id")
            if run_id:
                completed.add(str(run_id))
        except (OSError, json.JSONDecodeError):
            # An incomplete/corrupt run is not treated as completed and will be rerun.
            continue
    return completed


def save_one_run(output_dir: Path, output: Any) -> None:
    run_id = str(output.run_metrics["run_id"])
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "run_config.json", output.config)
    write_json(run_dir / "run_metrics.json", output.run_metrics)
    write_csv(run_dir / "client_metrics.csv", output.client_rows)
    write_csv(run_dir / "round_metrics.csv", output.round_rows)


def collect_completed_results(
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, str]]]:
    run_rows: List[Dict[str, Any]] = []
    client_rows: List[Dict[str, str]] = []
    round_rows: List[Dict[str, str]] = []

    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return run_rows, client_rows, round_rows

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "run_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                run_rows.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue

        client_rows.extend(read_csv(run_dir / "client_metrics.csv"))
        round_rows.extend(read_csv(run_dir / "round_metrics.csv"))

    run_rows.sort(
        key=lambda row: (
            str(row.get("scenario_name", "")),
            int(row.get("seed", 0)),
        )
    )
    client_rows.sort(
        key=lambda row: (
            row.get("scenario_name", ""),
            int(float(row.get("seed", 0))),
            int(float(row.get("client_id", 0))),
        )
    )
    round_rows.sort(
        key=lambda row: (
            row.get("scenario_name", ""),
            int(float(row.get("seed", 0))),
            int(float(row.get("round", 0))),
        )
    )
    return run_rows, client_rows, round_rows


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def describe(values: Iterable[Any]) -> Tuple[int, float, float, float]:
    clean = [number for value in values if (number := finite_float(value)) is not None]
    if not clean:
        return 0, math.nan, math.nan, math.nan

    count = len(clean)
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if count > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
    return count, mean, std, ci95


def build_aggregate_summary(run_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        key = (
            str(row["scenario_name"]),
            int(row["num_clients"]),
            int(row["num_malicious"]),
        )
        grouped[key].append(row)

    metrics = [
        "exact_precision",
        "exact_recall",
        "exact_f1",
        "exact_accuracy",
        "exact_false_positive_rate",
        "watchlist_precision",
        "watchlist_recall",
        "watchlist_f1",
        "watchlist_accuracy",
        "watchlist_false_positive_rate",
        "adaptive_precision",
        "adaptive_recall",
        "adaptive_f1",
        "adaptive_accuracy",
        "adaptive_false_positive_rate",
        "score_gap",
        "mean_malicious_score",
        "mean_benign_score",
        "final_global_weight_norm",
        "final_global_round_fidelity",
        "mean_global_round_fidelity",
        "run_seconds",
        "mean_round_seconds",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for (scenario_name, num_clients, num_malicious), rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "scenario_name": scenario_name,
            "num_clients": num_clients,
            "num_malicious": num_malicious,
            "num_benign": num_clients - num_malicious,
            "malicious_ratio": num_malicious / num_clients,
            "number_of_runs": len(rows),
        }

        for metric in metrics:
            count, mean, std, ci95 = describe(row.get(metric) for row in rows)
            summary[f"{metric}_n"] = count
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_ci95"] = ci95

        summary_rows.append(summary)
    return summary_rows


def _plot_metric_by_ratio(
    summary_rows: Sequence[Mapping[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    by_client_count: MutableMapping[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        mean = finite_float(row.get(f"{metric}_mean"))
        if mean is None:
            continue
        by_client_count[int(row["num_clients"])].append(row)

    if not by_client_count:
        return

    plt.figure(figsize=(8, 5))
    for num_clients, rows in sorted(by_client_count.items()):
        ordered = sorted(rows, key=lambda row: float(row["malicious_ratio"]))
        x_values = [100.0 * float(row["malicious_ratio"]) for row in ordered]
        y_values = [float(row[f"{metric}_mean"]) for row in ordered]
        errors = [float(row[f"{metric}_ci95"]) for row in ordered]
        plt.errorbar(
            x_values,
            y_values,
            yerr=errors,
            marker="o",
            capsize=4,
            label=f"{num_clients} total clients",
        )

    plt.xlabel("Malicious Client Ratio (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-0.05, 1.05) if metric.endswith(("precision", "recall", "f1", "accuracy")) else None
    plt.grid()
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_score_gap(
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    by_client_count: MutableMapping[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if finite_float(row.get("score_gap_mean")) is not None:
            by_client_count[int(row["num_clients"])].append(row)

    if not by_client_count:
        return

    plt.figure(figsize=(8, 5))
    for num_clients, rows in sorted(by_client_count.items()):
        ordered = sorted(rows, key=lambda row: float(row["malicious_ratio"]))
        x_values = [100.0 * float(row["malicious_ratio"]) for row in ordered]
        y_values = [float(row["score_gap_mean"]) for row in ordered]
        errors = [float(row["score_gap_ci95"]) for row in ordered]
        plt.errorbar(
            x_values,
            y_values,
            yerr=errors,
            marker="o",
            capsize=4,
            label=f"{num_clients} total clients",
        )

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Malicious Client Ratio (%)")
    plt.ylabel("Score Gap: Minimum Malicious - Maximum Benign")
    plt.title("Cumulative Detection-Score Separation")
    plt.grid()
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_client_score_boxplot(
    client_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    benign: List[float] = []
    malicious: List[float] = []

    for row in client_rows:
        value = finite_float(row.get("cumulative_detection_score"))
        if value is None:
            continue
        if int(float(row.get("is_malicious", 0))) == 1:
            malicious.append(value)
        else:
            benign.append(value)

    if not benign or not malicious:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot([benign, malicious], tick_labels=["Benign", "Malicious"])
    plt.ylabel("Cumulative Detection Score")
    plt.title("Cumulative Scores Across All Completed Runs")
    plt.grid(axis="y")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_aggregate_plots(
    output_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    client_rows: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = output_dir / "aggregate_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    definitions = [
        ("exact_precision", "Exact Top-k Precision", "Precision"),
        ("exact_recall", "Exact Top-k Recall", "Recall"),
        ("exact_f1", "Exact Top-k F1-score", "F1-score"),
        ("exact_accuracy", "Exact Top-k Accuracy", "Accuracy"),
        ("adaptive_precision", "Adaptive MAD Precision", "Precision"),
        ("adaptive_recall", "Adaptive MAD Recall", "Recall"),
        ("adaptive_f1", "Adaptive MAD F1-score", "F1-score"),
        ("adaptive_accuracy", "Adaptive MAD Accuracy", "Accuracy"),
    ]

    for metric, title, ylabel in definitions:
        _plot_metric_by_ratio(
            summary_rows,
            metric,
            title,
            ylabel,
            plot_dir / f"{metric}_vs_malicious_ratio.png",
        )

    _plot_score_gap(summary_rows, plot_dir / "score_gap_vs_malicious_ratio.png")
    _plot_client_score_boxplot(
        client_rows,
        plot_dir / "benign_vs_malicious_cumulative_score_boxplot.png",
    )


def consolidate_and_summarise(
    output_dir: Path,
    create_plots: bool = True,
) -> None:
    run_rows, client_rows, round_rows = collect_completed_results(output_dir)
    write_csv(output_dir / RUN_METRICS_FILENAME, run_rows)
    write_csv(output_dir / CLIENT_METRICS_FILENAME, client_rows)
    write_csv(output_dir / ROUND_METRICS_FILENAME, round_rows)

    summary_rows = build_aggregate_summary(run_rows)
    write_csv(output_dir / AGGREGATE_FILENAME, summary_rows)
    if create_plots:
        create_aggregate_plots(output_dir, summary_rows, client_rows)

    print("\nStatistical outputs updated:")
    print(" ", output_dir / RUN_METRICS_FILENAME)
    print(" ", output_dir / CLIENT_METRICS_FILENAME)
    print(" ", output_dir / ROUND_METRICS_FILENAME)
    print(" ", output_dir / AGGREGATE_FILENAME)
    if create_plots:
        print(" ", output_dir / "aggregate_plots")


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise SystemExit("Use either --overwrite or --resume, not both")
    if args.max_runs is not None and args.max_runs <= 0:
        raise SystemExit("--max-runs must be greater than zero")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_data = load_json(config_path)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = resolve_output_dir(config_data)
    configs = build_experiment_configs(config_data, args.scenario)
    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    print("QFL statistical evaluation")
    print("Project root:", PROJECT_ROOT)
    print("Configuration:", config_path)
    print("Output directory:", output_dir)
    print("Planned runs:", len(configs))

    for index, config in enumerate(configs, start=1):
        print(
            f"  {index:03d}. {make_run_id(config)} | "
            f"benign={config.num_clients - config.num_malicious}, "
            f"malicious={config.num_malicious}"
        )

    if args.dry_run:
        print("\nDry run only; no models were trained.")
        return

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    if output_dir.exists() and not args.resume and not args.overwrite:
        if any(output_dir.iterdir()):
            raise SystemExit(
                f"Output directory already exists and is not empty: {output_dir}\n"
                "Use --resume to continue or --overwrite to start again."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "batch_configuration.json", config_data)

    completed = load_completed_run_ids(output_dir) if args.resume else set()

    total = len(configs)
    for run_number, config in enumerate(configs, start=1):
        run_id = make_run_id(config)
        if run_id in completed:
            print(f"\n[{run_number}/{total}] Skipping completed run: {run_id}")
            continue

        print(f"\n[{run_number}/{total}] Starting: {run_id}")
        print(
            f"Random malicious IDs will be selected for seed {config.seed}; "
            f"benign={config.num_clients - config.num_malicious}, "
            f"malicious={config.num_malicious}."
        )

        run_dir = output_dir / "runs" / run_id
        if run_dir.exists() and not (run_dir / "run_metrics.json").exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            output = run_experiment(
                config,
                run_dir,
                verbose_clients=args.verbose_clients,
            )
            save_one_run(output_dir, output)
        except KeyboardInterrupt:
            print("\nInterrupted. Completed runs are preserved.")
            print("Use the same command with --resume to continue.")
            consolidate_and_summarise(output_dir)
            raise SystemExit(130)
        except Exception as exc:
            print(f"\nRun failed: {run_id}")
            print(f"Error: {type(exc).__name__}: {exc}")
            print("Completed runs are preserved. Correct the issue and use --resume.")
            consolidate_and_summarise(output_dir)
            raise

        # Rebuild master CSVs after every completed run. This keeps useful
        # partial statistics even when a long experiment is interrupted.
        consolidate_and_summarise(output_dir, create_plots=False)

    # Aggregate figures are created once after all selected runs finish.
    # This avoids regenerating every PNG after every individual seed.
    consolidate_and_summarise(output_dir, create_plots=True)
    print("\nAll selected QFL statistical experiments are complete.")


if __name__ == "__main__":
    main()
