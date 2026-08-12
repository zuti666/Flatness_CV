from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import build_quadratic_problem, build_traces, resolved_config
from src.sensitivity import (
    build_sweep_cases,
    fixed_effective_radius_rows,
    initialization_ensemble_rows,
    loglog_spectral_fit,
    match_primary_rho,
    quality_decomposition_rows,
    strength_matched_rows,
)


class SensitivityUnitTests(unittest.TestCase):
    def test_loglog_fit_recovers_known_power(self) -> None:
        eigenvalues = np.geomspace(0.1, 10.0, 20)
        fit = loglog_spectral_fit(
            {"eigenvalue": value, "gain": 0.7 * value**2}
            for value in eigenvalues
        )
        self.assertAlmostEqual(float(fit["spectral_log_slope"]), 2.0, places=12)
        self.assertAlmostEqual(float(fit["spectral_log_r2"]), 1.0, places=12)

    def test_sweep_cases_are_ofat_and_unique(self) -> None:
        cases = build_sweep_cases(
            rho_values=(1e-3, 1e-2),
            condition_numbers=(4.0, 100.0),
            dimensions=(10, 20),
            inner_steps=(1, 5),
        )
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case.case_id for case in cases}), 8)
        for case in cases:
            self.assertIn(case.factor, {"rho", "condition", "dimension", "inner_steps"})
            if case.factor == "condition":
                self.assertAlmostEqual(
                    float(case.overrides["lambda_min"])
                    * float(case.overrides["lambda_max"]),
                    1.0,
                    places=14,
                )

        with self.assertRaises(ValueError):
            build_sweep_cases(
                rho_values=(1e-2,),
                condition_numbers=(100.0,),
                dimensions=(20.5,),
                inner_steps=(5,),
            )

    def test_native_radius_matching_hits_target(self) -> None:
        config = resolved_config({"make_plots": False})
        hessian, weights, gradient = build_quadratic_problem(config)
        traces, _ = build_traces(hessian, weights, config)
        chosen = [
            next(trace for trace in traces if trace.key == key)
            for key in ("sam", "gam", "lookbehind_k5_fixed_budget")
        ]
        for template in chosen:
            with self.subTest(key=template.key):
                rho, matched = match_primary_rho(
                    template,
                    hessian,
                    weights,
                    0.25,
                    initial_rho=float(config["primary_rho_scale"]),
                )
                achieved = float(np.linalg.norm(matched.correction)) / float(
                    np.linalg.norm(gradient)
                )
                self.assertGreater(rho, 0.0)
                self.assertAlmostEqual(achieved, 0.25, places=9)

    def test_strength_table_is_strict_and_semantically_complete(self) -> None:
        rows = strength_matched_rows(
            resolved_config({"inner_steps": [1, 2], "make_plots": False}),
            targets=(0.1,),
        )
        self.assertTrue(rows)
        self.assertTrue(any(row["method"] == "gam" for row in rows))
        self.assertTrue(any(row["method"] == "lookbehind" for row in rows))
        for row in rows:
            self.assertLess(float(row["matching_relative_error"]), 2e-9)
            self.assertAlmostEqual(
                float(row["achieved_correction_gradient_ratio"]), 0.1, places=8
            )
            if row["method"] == "gam":
                self.assertIsNone(row["q0"])
                self.assertEqual(row["object_kind"], "final_regularizer")

    def test_fixed_effective_radius_keeps_matched_sam_constant(self) -> None:
        rows = fixed_effective_radius_rows(
            resolved_config({"make_plots": False}), (1, 2, 5)
        )
        matched = [row for row in rows if row["method"] == "matched_sam"]
        self.assertEqual(len(matched), 3)
        ratios = [float(row["correction_gradient_ratio"]) for row in matched]
        self.assertLess(max(ratios) - min(ratios), 1e-13)
        for row in matched:
            self.assertAlmostEqual(float(row["spectral_log_slope"]), 1.0, places=12)

    def test_quality_decomposition_exposes_radius_utilization(self) -> None:
        rows = quality_decomposition_rows(resolved_config({"make_plots": False}))
        matched = next(
            row for row in rows if row["key"] == "matched_sam_k5_fixed_budget"
        )
        self.assertAlmostEqual(float(matched["radius_utilization"]), 0.6, places=12)
        self.assertGreater(float(matched["boundary_q0"]), float(matched["raw_q0"]))
        sam = next(row for row in rows if row["key"] == "sam")
        self.assertAlmostEqual(float(sam["radius_utilization"]), 1.0, places=12)
        self.assertAlmostEqual(float(sam["raw_q0"]), float(sam["boundary_q0"]), places=12)

    def test_initialization_ensemble_is_deterministic_and_distinct(self) -> None:
        config = resolved_config({"make_plots": False})
        first, first_summary = initialization_ensemble_rows(
            config, sample_count=5, seed=17
        )
        second, second_summary = initialization_ensemble_rows(
            config, sample_count=5, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        counts = {
            kind: sum(row["initialization"] == kind for row in first)
            for kind in ("equal_gradient", "random_w", "random_g")
        }
        self.assertEqual(counts, {"equal_gradient": 1, "random_w": 5, "random_g": 5})


class SensitivityCliTests(unittest.TestCase):
    def test_quick_cli_products(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
            }
        )
        with tempfile.TemporaryDirectory(prefix="e001-sensitivity-") as temporary:
            destination = Path(temporary) / "first"
            second_destination = Path(temporary) / "second"
            for current in (destination, second_destination):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "run_e001_sensitivity.py",
                        "--quick",
                        "--no-plots",
                        "--output-dir",
                        str(current),
                    ],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=90,
                    check=False,
                )
                if completed.returncode != 0:
                    self.fail(
                        f"sensitivity CLI failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                    )
            required = {
                "manifest.json",
                "sensitivity_summary.csv",
                "strength_matched.csv",
                "fixed_effective_radius.csv",
                "quality_decomposition.csv",
                "initialization_samples.csv",
                "initialization_summary.csv",
            }
            self.assertTrue(all((destination / name).is_file() for name in required))
            for name in required - {"manifest.json"}:
                self.assertEqual(
                    (destination / name).read_bytes(),
                    (second_destination / name).read_bytes(),
                    f"sensitivity output is not deterministic: {name}",
                )
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_count"], 8)
            self.assertEqual(manifest["grids"]["strength_targets"], [0.1])
            self.assertEqual(
                manifest["grids"]["initialization_samples_per_random_family"], 8
            )
            with (destination / "strength_matched.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertLess(max(float(row["matching_relative_error"]) for row in rows), 2e-9)


if __name__ == "__main__":
    unittest.main()
