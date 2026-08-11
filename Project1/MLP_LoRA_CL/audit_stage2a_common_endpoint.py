#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage-2A common Task-A endpoints")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/focus_stage2a_endpoint_audit.csv")
    )
    args = parser.parse_args()

    grouped: dict[tuple[tuple[float, ...], int], list[dict]] = defaultdict(list)
    for root in args.input:
        for metrics_path in root.rglob("metrics.json"):
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("task_a_parameterization") != "dense":
                continue
            transition_path = metrics_path.parent / "immediate_transitions.json"
            transitions = json.loads(transition_path.read_text())
            if len(transitions) != 1:
                raise ValueError(f"expected one immediate transition in {metrics_path.parent}")
            transition = transitions[0]
            grouped[(tuple(metrics["angles"]), int(metrics["seed"]))].append(
                {
                    "path": str(metrics_path.parent),
                    "parameterization": metrics["parameterization"],
                    "schedule": tuple(metrics["optimizer_schedule"]),
                    "hash": transition.get("common_start_weight_sha256"),
                    "weight_error": float(
                        transition.get("common_reparameterization_weight_error", float("inf"))
                    ),
                    "logit_error": float(
                        transition.get("common_reparameterization_logit_error", float("inf"))
                    ),
                    "old_start_loss": float(metrics["loss_matrix"][0][0]),
                    "old_start_accuracy": float(metrics["accuracy_matrix"][0][0]),
                }
            )

    rows = []
    failures = []
    for (order, seed), records in sorted(grouped.items()):
        hashes = {record["hash"] for record in records}
        loss_values = [record["old_start_loss"] for record in records]
        accuracy_values = [record["old_start_accuracy"] for record in records]
        row = {
            "task_order": str(order),
            "seed": seed,
            "n_runs": len(records),
            "n_unique_hashes": len(hashes),
            "max_weight_error": max(record["weight_error"] for record in records),
            "max_logit_error": max(record["logit_error"] for record in records),
            "old_start_loss_range": max(loss_values) - min(loss_values),
            "old_start_accuracy_range": max(accuracy_values) - min(accuracy_values),
            "passed": (
                len(hashes) == 1
                and None not in hashes
                and max(record["weight_error"] for record in records) <= 1e-7
                and max(record["logit_error"] for record in records) <= 1e-6
                and max(loss_values) - min(loss_values) <= 1e-10
                and max(accuracy_values) - min(accuracy_values) <= 1e-10
            ),
        }
        rows.append(row)
        if not row["passed"]:
            failures.append((order, seed, records))

    if not rows:
        raise ValueError("no completed Stage-2A runs found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"audited {len(rows)} order/seed groups; "
        f"{len(rows) - len(failures)} passed; wrote {args.output}"
    )
    if failures:
        labels = ", ".join(f"{order}/seed={seed}" for order, seed, _ in failures)
        raise RuntimeError(f"common-endpoint audit failed: {labels}")


if __name__ == "__main__":
    main()
