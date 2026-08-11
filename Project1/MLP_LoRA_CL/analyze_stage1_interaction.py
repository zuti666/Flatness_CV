#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap_interval(values: list[float], repeats: int = 10000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(2026)
    samples = generator.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def load_summaries(root: Path) -> list[dict]:
    records = []
    for path in root.rglob("metrics.json"):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        record["run_dir"] = str(path.parent)
        record["task_order"] = tuple(record.get("angles", []))
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired optimizer-by-parameterization interaction test")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/stage1_interaction.csv"))
    args = parser.parse_args()
    summaries = load_summaries(args.input)
    index = {
        (
            item["task_order"], item["seed"], item["parameterization"], item["optimizer"]
        ): item
        for item in summaries
    }
    rows = []
    for order in sorted({item["task_order"] for item in summaries}):
        for optimizer in ("sam", "gam_fd"):
            paired = defaultdict(list)
            for seed in sorted({item["seed"] for item in summaries if item["task_order"] == order}):
                for parameterization in ("dense", "random_subspace", "factor_lora"):
                    baseline = index.get((order, seed, parameterization, "sgd"))
                    flat = index.get((order, seed, parameterization, optimizer))
                    if baseline is None or flat is None:
                        continue
                    paired[(parameterization, "accuracy_gain")].append(
                        flat["final_average_accuracy"] - baseline["final_average_accuracy"]
                    )
                    paired[(parameterization, "forgetting_reduction")].append(
                        baseline["average_forgetting"] - flat["average_forgetting"]
                    )
            for metric in ("accuracy_gain", "forgetting_reduction"):
                for parameterization in ("dense", "random_subspace", "factor_lora"):
                    values = paired.get((parameterization, metric), [])
                    if values:
                        mean, low, high = bootstrap_interval(values)
                        rows.append(
                            {
                                "task_order": str(order),
                                "optimizer": optimizer,
                                "contrast": parameterization,
                                "metric": metric,
                                "n_paired_seeds": len(values),
                                "mean": mean,
                                "ci95_low": low,
                                "ci95_high": high,
                            }
                        )
                dense = paired.get(("dense", metric), [])
                lora = paired.get(("factor_lora", metric), [])
                if dense and len(dense) == len(lora):
                    interaction = (np.asarray(lora) - np.asarray(dense)).tolist()
                    mean, low, high = bootstrap_interval(interaction)
                    rows.append(
                        {
                            "task_order": str(order),
                            "optimizer": optimizer,
                            "contrast": "factor_lora_minus_dense_interaction",
                            "metric": metric,
                            "n_paired_seeds": len(interaction),
                            "mean": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )
    if not rows:
        raise SystemExit("No complete SGD/Flat paired runs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} paired interaction rows to {args.output}")


if __name__ == "__main__":
    main()
