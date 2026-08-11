from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_profiles(input_roots: list[Path]) -> dict:
    grouped: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    for input_root in input_roots:
        for path in sorted(
            input_root.glob("angles-*/seed_*/transform_identity/sgd/step_geometry.json")
        ):
            angle_tag = path.parents[3].name.removeprefix("angles-")
            seed = int(path.parents[2].name.removeprefix("seed_"))
            with path.open(encoding="utf-8") as handle:
                rows = json.load(handle)
            grouped[angle_tag].append(
                (seed, [float(row["effective_step_norm"]) for row in rows])
            )
    if not grouped:
        raise RuntimeError(
            f"No Identity-SGD step geometry found under {input_roots}"
        )

    profiles = {}
    for angle_tag, values in sorted(grouped.items()):
        lengths = {len(profile) for _, profile in values}
        if len(lengths) != 1:
            raise RuntimeError(f"Inconsistent profile lengths for {angle_tag}: {lengths}")
        array = np.asarray([profile for _, profile in values], dtype=float)
        median = np.median(array, axis=0)
        profiles[angle_tag] = {
            "source_seeds": [seed for seed, _ in values],
            "num_steps": int(median.size),
            "median_step_norms": median.tolist(),
            "summary": {
                "minimum": float(median.min()),
                "median": float(np.median(median)),
                "maximum": float(median.max()),
                "path_sum": float(median.sum()),
                "quadratic_step_budget": float(np.square(median).sum()),
                "effective_step_count": float(
                    median.sum() ** 2 / np.square(median).sum()
                ),
            },
        }
    return {
        "sources": [str(path) for path in input_roots],
        "estimator": "per-step median across independent Identity-SGD P2b seeds",
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze P3 effective-step target profiles")
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_profiles(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value["summary"] for key, value in result["profiles"].items()}, indent=2))


if __name__ == "__main__":
    main()
