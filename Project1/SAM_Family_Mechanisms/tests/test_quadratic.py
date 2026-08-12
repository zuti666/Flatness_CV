from __future__ import annotations

import csv
import json
import math
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

from src.experiment import resolved_config
from src.diagnostics import (
    inner_problem_quality,
    maximize_quadratic_on_ball,
    nested_hessian_fit,
    path_metrics,
)
from src.operators import (
    gam_trace,
    lookbehind_trace,
    multistep_sam_trace,
    sam_trace,
    unit,
)

class QuadraticOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hessian = np.diag([0.4, 1.7, 6.0]).astype(np.float64)
        self.weights = np.array([1.2, -0.7, 0.35], dtype=np.float64)

    def test_sam_correction_is_exact_quadratic_hvp(self) -> None:
        rho = 0.13
        trace = sam_trace(self.hessian, self.weights, rho)
        gradient = self.hessian @ self.weights
        expected = rho * (self.hessian @ unit(gradient))

        np.testing.assert_allclose(trace.gradient, gradient, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(trace.correction, expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(trace.direction, gradient + expected, rtol=1e-13, atol=1e-14)

    def test_gam_keeps_probe_increment_and_final_regularizer_distinct(self) -> None:
        rho = 0.09
        trace = gam_trace(self.hessian, self.weights, rho)
        gradient = self.hessian @ self.weights
        normalized_gradient = unit(gradient)
        hessian_gradient = self.hessian @ normalized_gradient
        expected_probe = rho * (self.hessian @ unit(hessian_gradient))
        expected_final = rho * self.hessian @ unit(gradient + expected_probe)

        self.assertIsNotNone(trace.probe_increment)
        self.assertIsNotNone(trace.final_regularizer)
        np.testing.assert_allclose(
            trace.probe_increment, expected_probe, rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(
            trace.final_regularizer, expected_final, rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(
            trace.correction, trace.final_regularizer, rtol=0.0, atol=0.0
        )
        self.assertGreater(
            float(np.linalg.norm(trace.final_regularizer - trace.probe_increment)),
            1e-4,
        )

    def test_fixed_step_and_fixed_budget_path_radii(self) -> None:
        inner_steps = 4
        rho_step = 0.035
        fixed_step = multistep_sam_trace(
            self.hessian,
            self.weights,
            rho_step,
            inner_steps,
            "fixed_step",
        )
        fixed_step_metrics = path_metrics(fixed_step)
        self.assertAlmostEqual(
            fixed_step_metrics["path_radius"], inner_steps * rho_step, places=13
        )

        rho_budget = 0.11
        fixed_budget = multistep_sam_trace(
            self.hessian,
            self.weights,
            rho_budget / inner_steps,
            inner_steps,
            "fixed_budget",
        )
        fixed_budget_metrics = path_metrics(fixed_budget)
        self.assertAlmostEqual(fixed_budget_metrics["path_radius"], rho_budget, places=13)
        self.assertLessEqual(
            fixed_budget_metrics["endpoint_radius"],
            fixed_budget_metrics["path_radius"] + 1e-14,
        )

    def test_lookbehind_k1_is_sam(self) -> None:
        rho = 0.08
        sam = sam_trace(self.hessian, self.weights, rho)
        lookbehind = lookbehind_trace(
            self.hessian, self.weights, rho, inner_steps=1, protocol="fixed_step"
        )

        np.testing.assert_allclose(lookbehind.direction, sam.direction, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            lookbehind.correction, sam.correction, rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            lookbehind.perturbation, sam.perturbation, rtol=0.0, atol=1e-14
        )

    def test_trust_region_solution_has_boundary_norm_and_satisfies_kkt(self) -> None:
        matrix = np.diag([0.3, 1.4, 4.5])
        linear = np.array([0.7, -1.1, 0.45])
        radius = 0.27
        solution = maximize_quadratic_on_ball(matrix, linear, radius)

        self.assertAlmostEqual(float(np.linalg.norm(solution)), radius, places=12)
        stationarity_vector = linear + matrix @ solution
        multiplier = float(np.dot(solution, stationarity_vector) / radius**2)
        residual = stationarity_vector - multiplier * solution
        self.assertGreater(multiplier, float(np.linalg.eigvalsh(matrix)[-1]))
        self.assertLess(float(np.linalg.norm(residual)), 2e-11)
        with self.assertRaises(ValueError):
            maximize_quadratic_on_ball(-np.eye(3), np.zeros(3), radius)

    def test_inner_problem_oracles_reach_q0_and_q1_boundaries(self) -> None:
        radius = 0.21
        gradient = self.hessian @ self.weights
        delta_zero = maximize_quadratic_on_ball(self.hessian, gradient, radius)
        delta_first = maximize_quadratic_on_ball(
            self.hessian @ self.hessian,
            self.hessian @ gradient,
            radius,
        )
        quality_zero = inner_problem_quality(
            self.hessian, self.weights, delta_zero, radius
        )
        quality_first = inner_problem_quality(
            self.hessian, self.weights, delta_first, radius
        )

        self.assertAlmostEqual(quality_zero["q0"], 1.0, places=12)
        self.assertAlmostEqual(quality_first["q1"], 1.0, places=12)
        for quality in (quality_zero, quality_first):
            self.assertAlmostEqual(quality["delta0_star_norm"], radius, places=12)
            self.assertAlmostEqual(quality["delta1_star_norm"], radius, places=12)
            self.assertGreaterEqual(quality["q0"], -1e-12)
            self.assertLessEqual(quality["q0"], 1.0 + 1e-11)
            self.assertGreaterEqual(quality["q1"], -1e-12)
            self.assertLessEqual(quality["q1"], 1.0 + 1e-11)

    def test_nested_fit_r2_is_monotone(self) -> None:
        gradient = self.hessian @ self.weights
        normalized_gradient = unit(gradient)
        correction = (
            0.8 * self.hessian @ normalized_gradient
            - 0.15 * np.linalg.matrix_power(self.hessian, 2) @ normalized_gradient
            + 0.04 * np.linalg.matrix_power(self.hessian, 3) @ normalized_gradient
        )
        fit = nested_hessian_fit(correction, self.hessian, gradient, max_order=3)
        values = [fit[f"r2_{order}"] for order in range(1, 4)]

        self.assertTrue(all(value is not None for value in values))
        numeric_values = [float(value) for value in values if value is not None]
        for previous, current in zip(numeric_values, numeric_values[1:]):
            self.assertGreaterEqual(current + 1e-14, previous)
        for order, value in enumerate(numeric_values, start=1):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            self.assertGreaterEqual(float(fit[f"delta_r2_{order}"]), -1e-14)
        self.assertAlmostEqual(numeric_values[-1], 1.0, places=12)

    def test_config_rejects_lossy_or_nonfinite_values(self) -> None:
        invalid_overrides = (
            {"dimension": 3.5},
            {"inner_steps": [2.7]},
            {"rho_scales": [float("nan")]},
            {"lambda_max": float("inf")},
            {"lambda_min": 1e-300},
            {"lambda_max": 1e300},
            {"primary_rho_scale": 1e-20},
            {"rho_scales": [1e308]},
            {"make_plots": "false"},
            {"path_protocols": ["fixed_step", "fixed_step"]},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolved_config(overrides)


class QuadraticCliTests(unittest.TestCase):
    def _run_experiment(self, output_dir: Path) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
            }
        )
        completed = subprocess.run(
            [sys.executable, "run_quadratic.py", "--output-dir", str(output_dir)],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "quadratic CLI failed with exit code "
                f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

    @staticmethod
    def _assert_finite_json(test_case: unittest.TestCase, path: Path) -> None:
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant {value}")

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant)

        def visit(value: object) -> None:
            if isinstance(value, float):
                test_case.assertTrue(math.isfinite(value), f"non-finite value in {path}")
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payload)

    def test_cli_products_are_complete_finite_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quadratic-cli-") as temporary:
            temporary_root = Path(temporary)
            first = temporary_root / "first"
            second = temporary_root / "second"
            self._run_experiment(first)
            self._run_experiment(second)

            # Keep this list explicit: these files are the documented E001 contract.
            required_files = {
                "manifest.json",
                "arrays.npz",
                "hvp_scan.csv",
                "spectral_gain.csv",
                "method_summary.csv",
                "metrics.json",
                "spectral_gain.png",
                "top_subspace_curvature.png",
                "hp_fit.png",
                "inner_quality.png",
            }
            produced = {
                path.relative_to(first).as_posix()
                for path in first.rglob("*")
                if path.is_file()
            }
            self.assertTrue(
                required_files.issubset(produced),
                f"missing CLI products: {sorted(required_files - produced)}",
            )
            for relative_path in required_files:
                self.assertGreater((first / relative_path).stat().st_size, 0)

            json_files = sorted(first.rglob("*.json"))
            self.assertGreaterEqual(len(json_files), 2)
            for json_file in json_files:
                self._assert_finite_json(self, json_file)

            first_csv = {
                path.relative_to(first).as_posix(): path
                for path in first.rglob("*.csv")
            }
            second_csv = {
                path.relative_to(second).as_posix(): path
                for path in second.rglob("*.csv")
            }
            self.assertEqual(set(first_csv), set(second_csv))
            core_csv = {"hvp_scan.csv", "spectral_gain.csv", "method_summary.csv"}
            self.assertTrue(core_csv.issubset(first_csv))
            for name in sorted(core_csv):
                self.assertEqual(
                    first_csv[name].read_bytes(),
                    second_csv[name].read_bytes(),
                    f"CSV output is not deterministic: {name}",
                )
                with first_csv[name].open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows, f"CSV contains no data rows: {name}")

            with (first / "spectral_gain.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                spectral_reader = csv.DictReader(handle)
                self.assertTrue(
                    {"correction_projection", "ghat_projection", "rho_scale", "rho"}
                    .issubset(spectral_reader.fieldnames or [])
                )

            with (first / "method_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                summary_rows = {row["key"]: row for row in csv.DictReader(handle)}
            self.assertEqual(summary_rows["gam"]["object_kind"], "final_regularizer")
            self.assertEqual(summary_rows["gam"]["q0"], "")
            self.assertEqual(
                summary_rows["gam_probe_direction"]["quality_perturbation_kind"],
                "gam_probe_perturbation",
            )
            self.assertNotEqual(summary_rows["gam_probe_direction"]["q0"], "")
            self.assertEqual(summary_rows["gam_probe_increment"]["q0"], "")
            self.assertEqual(summary_rows["matched_sam_k2_fixed_step"]["inner_steps"], "2")
            self.assertEqual(
                summary_rows["matched_sam_k2_fixed_step"]["method"], "matched_sam"
            )
            self.assertEqual(summary_rows["gam"]["path_novelty"], "")
            self.assertNotEqual(summary_rows["gam"]["h1_residual"], "")
            lookbehind = summary_rows["lookbehind_k5_fixed_budget"]
            self.assertNotEqual(lookbehind["rho_fit_over_rho_eff"], "")
            self.assertNotEqual(lookbehind["matched_correction_norm_ratio"], "")
            self.assertNotEqual(lookbehind["matched_correction_relative_error"], "")

            manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runtime"]["dtype"], "float64")
            self.assertTrue(manifest["compute_budget"])
            self.assertTrue(manifest["objects"])
            self.assertEqual(len(manifest["code_fingerprint"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
