#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from analyze_stage1c_temporal import (
    SGD_SAM,
    SGD_SGD,
    bootstrap,
    load_runs,
    matched_reductions,
    scalar_metrics,
    write_rows,
)


def gauge_label(run: dict) -> str:
    return f"gauge_{float(run.get('factor_gauge_scale') or 1.0):g}"


def gauge_metrics(run: dict) -> dict[str, float]:
    output = scalar_metrics(run)
    transition = run["transition"]
    for key in (
        "start_factor_a_norm",
        "start_factor_b_norm",
        "start_factor_balance_gap",
        "end_factor_a_norm",
        "end_factor_b_norm",
        "end_factor_norm_ratio",
        "end_factor_balance_gap",
        "reachable_coverage",
        "start_tangent_dimension",
    ):
        if transition.get(key) is not None:
            output[key] = float(transition[key])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired analysis of Factor-LoRA gauge sensitivity"
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/focus_stage2d_gauge_effects.csv")
    )
    parser.add_argument(
        "--cells-output",
        type=Path,
        default=Path("analysis/focus_stage2d_gauge_cells.csv"),
    )
    parser.add_argument("--reference-gauge", type=float, default=1.0)
    args = parser.parse_args()

    runs = load_runs(args.input)
    index = {}
    for run in runs:
        key = (run["order"], gauge_label(run), run["schedule"], int(run["seed"]))
        if key in index:
            raise ValueError(f"duplicate gauge run: {key}")
        index[key] = run

    cell_samples: dict[tuple, list[float]] = defaultdict(list)
    for run in runs:
        for metric, value in gauge_metrics(run).items():
            cell_samples[(run["order"], gauge_label(run), run["schedule"], metric)].append(value)
    cell_rows = []
    for row_index, ((order, gauge, schedule, metric), values) in enumerate(
        sorted(cell_samples.items(), key=str)
    ):
        mean, low, high = bootstrap(values, 80_000 + row_index)
        cell_rows.append({
            "task_order": str(order), "gauge": gauge,
            "old_task_optimizer": schedule[0], "new_task_optimizer": schedule[1],
            "metric": metric, "n_seeds": len(values), "mean": mean,
            "ci95_low": low, "ci95_high": high,
        })
    write_rows(cell_rows, args.cells_output)

    samples: dict[tuple, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    orders = sorted({run["order"] for run in runs})
    gauges = sorted({gauge_label(run) for run in runs})
    seeds = sorted({int(run["seed"]) for run in runs})
    for order in orders:
        for gauge in gauges:
            for seed in seeds:
                control = index.get((order, gauge, SGD_SGD, seed))
                treatment = index.get((order, gauge, SGD_SAM, seed))
                if control is None or treatment is None:
                    continue
                left, right = gauge_metrics(control), gauge_metrics(treatment)
                for metric in sorted(left.keys() & right.keys()):
                    samples[(order, metric)][gauge][seed] = left[metric] - right[metric]
                for metric, value in matched_reductions(control, treatment).items():
                    samples[(order, metric)][gauge][seed] = value

    rows = []
    reference = f"gauge_{float(args.reference_gauge):g}"
    for row_index, ((order, metric), by_gauge) in enumerate(sorted(samples.items(), key=str)):
        for gauge, by_seed in sorted(by_gauge.items()):
            values = list(by_seed.values())
            mean, low, high = bootstrap(values, 90_000 + row_index)
            rows.append({
                "task_order": str(order), "metric": metric, "contrast": gauge,
                "n_paired_seeds": len(values), "mean": mean,
                "ci95_low": low, "ci95_high": high,
            })
        reference_values = by_gauge.get(reference, {})
        for gauge, by_seed in sorted(by_gauge.items()):
            if gauge == reference:
                continue
            common = sorted(reference_values.keys() & by_seed.keys())
            if not common:
                continue
            values = [by_seed[seed] - reference_values[seed] for seed in common]
            mean, low, high = bootstrap(values, 100_000 + row_index)
            rows.append({
                "task_order": str(order), "metric": metric,
                "contrast": f"{gauge}_minus_{reference}",
                "n_paired_seeds": len(values), "mean": mean,
                "ci95_low": low, "ci95_high": high,
            })
    write_rows(rows, args.output)
    print(
        f"wrote {len(rows)} gauge effects to {args.output} and "
        f"{len(cell_rows)} cells to {args.cells_output}"
    )


if __name__ == "__main__":
    main()
