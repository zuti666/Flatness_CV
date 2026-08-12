from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.e002_core import (
    FlatTanhMLP,
    full_hessian,
    gradient,
    hessian_vector_product,
    idealized_gam_direction,
    lookbehind_plain_sgd_delta,
    make_two_moons,
    orthogonal_component,
    recompose_looksam_direction,
    sam_direction,
    sam_k_direction,
    sgd_direction,
)
from src.diagnostics import eigensystem_descending, signed_curvature_metrics


class E002CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.model = FlatTanhMLP(hidden_width=3)
        cls.data = make_two_moons(
            train_samples=48,
            validation_samples=24,
            test_samples=32,
            noise=0.12,
            label_flip=0.125,
            seed=73,
        )
        cls.inputs = torch.from_numpy(cls.data.x_train[:16]).to(torch.float64)
        cls.targets = torch.from_numpy(cls.data.y_train[:16]).to(torch.int64)
        cls.theta = cls.model.init_parameters(91, dtype=torch.float64)

    def test_parameter_count_and_data_are_deterministic(self) -> None:
        model = FlatTanhMLP(hidden_width=16)
        first_theta = model.init_parameters(17, dtype=torch.float64)
        second_theta = model.init_parameters(17, dtype=torch.float64)
        other_theta = model.init_parameters(18, dtype=torch.float64)

        self.assertEqual(model.parameter_count, 82)
        self.assertEqual(first_theta.shape, (82,))
        self.assertEqual(first_theta.dtype, torch.float64)
        torch.testing.assert_close(first_theta, second_theta, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(first_theta, other_theta))

        arguments = {
            "train_samples": 40,
            "validation_samples": 20,
            "test_samples": 30,
            "noise": 0.15,
            "label_flip": 0.10,
            "seed": 3407,
        }
        first = make_two_moons(**arguments)
        second = make_two_moons(**arguments)
        changed = make_two_moons(**{**arguments, "seed": 3408})
        for field in (
            "x_train",
            "y_train",
            "x_validation",
            "y_validation",
            "x_test",
            "y_test",
            "flipped_indices",
        ):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        self.assertFalse(np.array_equal(first.x_train, changed.x_train))
        self.assertEqual(first.flipped_indices.size, 4)
        np.testing.assert_array_equal(first.flipped_indices, np.sort(first.flipped_indices))
        np.testing.assert_allclose(first.x_train.mean(axis=0), 0.0, rtol=0.0, atol=2e-15)
        np.testing.assert_allclose(first.x_train.std(axis=0), 1.0, rtol=0.0, atol=2e-15)

    def test_hessian_is_symmetric_and_hvp_matches_matrix_and_finite_difference(self) -> None:
        hessian = full_hessian(
            self.model, self.theta, self.inputs, self.targets
        )
        self.assertEqual(
            hessian.shape, (self.model.parameter_count, self.model.parameter_count)
        )
        self.assertTrue(torch.isfinite(hessian).all().item())
        torch.testing.assert_close(hessian, hessian.T, rtol=0.0, atol=2e-15)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(131)
        vector = torch.randn(
            self.model.parameter_count, generator=generator, dtype=torch.float64
        )
        vector /= torch.linalg.vector_norm(vector)
        exact = hessian_vector_product(
            self.model, self.theta, self.inputs, self.targets, vector
        )
        torch.testing.assert_close(exact, hessian @ vector, rtol=2e-10, atol=2e-11)

        finite_difference_step = 1e-5
        positive = gradient(
            self.model,
            self.theta + finite_difference_step * vector,
            self.inputs,
            self.targets,
        )
        negative = gradient(
            self.model,
            self.theta - finite_difference_step * vector,
            self.inputs,
            self.targets,
        )
        finite_difference = (positive - negative) / (2.0 * finite_difference_step)
        torch.testing.assert_close(exact, finite_difference, rtol=2e-7, atol=2e-9)

    def test_indefinite_quadratic_keeps_positive_negative_and_signed_projections(self) -> None:
        hessian = np.diag([-4.0, -0.5, 0.0, 2.0, 7.0]).astype(np.float64)
        vector = np.array([0.5, -2.0, 3.0, -1.5, 0.25], dtype=np.float64)
        metrics = signed_curvature_metrics(vector, hessian)

        positive = 2.0 * 1.5**2 + 7.0 * 0.25**2
        negative = 4.0 * 0.5**2 + 0.5 * 2.0**2
        self.assertAlmostEqual(
            float(metrics["positive_curvature_quadratic"]), positive, places=14
        )
        self.assertAlmostEqual(
            float(metrics["negative_curvature_quadratic"]), negative, places=14
        )
        self.assertGreater(float(metrics["positive_rayleigh"]), 0.0)
        self.assertGreater(float(metrics["negative_rayleigh"]), 0.0)
        self.assertAlmostEqual(
            float(vector @ hessian @ vector), positive - negative, places=14
        )

        eigenvalues, eigenvectors = eigensystem_descending(hessian)
        correction = np.array([-0.6, 1.25, -2.0, 0.4, -0.3], dtype=np.float64)
        signed_projection = eigenvectors.T @ correction
        reconstructed = eigenvectors @ signed_projection
        np.testing.assert_allclose(reconstructed, correction, rtol=0.0, atol=1e-15)
        self.assertTrue(np.any(signed_projection > 0.0))
        self.assertTrue(np.any(signed_projection < 0.0))
        self.assertFalse(np.array_equal(signed_projection, np.abs(signed_projection)))
        self.assertEqual(eigenvalues.tolist(), [7.0, 2.0, 0.0, -0.5, -4.0])

    def test_sam_with_zero_radius_is_exactly_sgd(self) -> None:
        sgd = sgd_direction(self.model, self.theta, self.inputs, self.targets)
        sam = sam_direction(
            self.model, self.theta, self.inputs, self.targets, rho=0.0
        )

        torch.testing.assert_close(sam.clean_gradient, sgd.direction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(sam.direction, sgd.direction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(sam.correction, torch.zeros_like(sam.correction))
        torch.testing.assert_close(sam.perturbation, torch.zeros_like(sam.perturbation))

    def test_gam_degeneracies_and_three_distinct_objects_are_finite(self) -> None:
        sgd = sgd_direction(self.model, self.theta, self.inputs, self.targets)
        alpha_zero = idealized_gam_direction(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho=0.04,
            alpha=0.0,
        )
        rho_zero = idealized_gam_direction(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho=0.0,
            alpha=1.0,
        )
        torch.testing.assert_close(alpha_zero.direction, sgd.direction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(rho_zero.direction, sgd.direction, rtol=0.0, atol=0.0)

        gam = idealized_gam_direction(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho=0.04,
            alpha=1.0,
        )
        for name in ("probe_direction", "probe_increment", "final_regularizer"):
            value = getattr(gam, name)
            self.assertIsNotNone(value, name)
            assert value is not None
            self.assertEqual(value.shape, self.theta.shape)
            self.assertTrue(torch.isfinite(value).all().item(), name)
        assert gam.final_regularizer is not None
        torch.testing.assert_close(
            gam.correction, gam.final_regularizer, rtol=1e-13, atol=1e-14
        )

    def test_lookbehind_plain_sgd_identities_and_closed_form_delta(self) -> None:
        rho = 0.035
        learning_rate = 0.07
        sam = sam_direction(
            self.model, self.theta, self.inputs, self.targets, rho=rho
        )
        k1 = lookbehind_plain_sgd_delta(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho_step=rho,
            inner_steps=1,
            learning_rate=learning_rate,
            alpha=1.0,
        )
        torch.testing.assert_close(k1.direction, sam.direction, rtol=0.0, atol=0.0)
        assert k1.slow_delta is not None
        torch.testing.assert_close(
            k1.slow_delta, -learning_rate * sam.direction, rtol=1e-14, atol=1e-15
        )

        inner_steps = 4
        zero_radius = lookbehind_plain_sgd_delta(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho_step=0.0,
            inner_steps=inner_steps,
            learning_rate=learning_rate,
            alpha=1.0 / inner_steps,
        )
        sgd = sgd_direction(self.model, self.theta, self.inputs, self.targets)
        torch.testing.assert_close(
            zero_radius.direction, sgd.direction, rtol=2e-15, atol=2e-16
        )

        alpha = 0.35
        trace = lookbehind_plain_sgd_delta(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho_step=0.02,
            inner_steps=3,
            learning_rate=learning_rate,
            alpha=alpha,
        )
        assert trace.path_gradients is not None
        assert trace.path_mean_surrogate is not None
        assert trace.effective_direction is not None
        assert trace.slow_delta is not None
        expected_direction = alpha * trace.path_gradients.sum(dim=0)
        expected_delta = -learning_rate * expected_direction
        torch.testing.assert_close(trace.direction, expected_direction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(trace.effective_direction, expected_direction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(trace.slow_delta, expected_delta, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            trace.path_mean_surrogate,
            trace.path_gradients.mean(dim=0),
            rtol=0.0,
            atol=0.0,
        )

    def test_looksam_orthogonal_cache_and_sam_k_refresh_contract(self) -> None:
        rho = 0.04
        alpha = 0.7
        sgd = sgd_direction(self.model, self.theta, self.inputs, self.targets)
        sam = sam_direction(
            self.model, self.theta, self.inputs, self.targets, rho=rho
        )
        cached = orthogonal_component(sam.direction, sgd.direction)
        self.assertGreater(float(torch.linalg.vector_norm(cached)), 0.0)
        orthogonality_scale = max(
            1.0,
            float(torch.linalg.vector_norm(cached) * torch.linalg.vector_norm(sgd.direction)),
        )
        self.assertLessEqual(
            abs(float(torch.dot(cached, sgd.direction))),
            2e-14 * orthogonality_scale,
        )

        recomposed = recompose_looksam_direction(
            sgd.direction, cached, alpha=alpha
        )
        correction = recomposed - sgd.direction
        self.assertTrue(torch.isfinite(recomposed).all().item())
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(correction)),
            alpha * float(torch.linalg.vector_norm(sgd.direction)),
            places=13,
        )
        cosine = torch.dot(correction, cached) / (
            torch.linalg.vector_norm(correction) * torch.linalg.vector_norm(cached)
        )
        self.assertAlmostEqual(float(cosine), 1.0, places=13)

        refresh = sam_k_direction(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho,
            step=10,
            period=5,
        )
        nonrefresh = sam_k_direction(
            self.model,
            self.theta,
            self.inputs,
            self.targets,
            rho,
            step=11,
            period=5,
        )
        torch.testing.assert_close(refresh.direction, sam.direction, rtol=0.0, atol=0.0)
        torch.testing.assert_close(nonrefresh.direction, sgd.direction, rtol=0.0, atol=0.0)
        self.assertTrue(refresh.metadata["refresh"])
        self.assertFalse(nonrefresh.metadata["refresh"])


class E002CliTests(unittest.TestCase):
    def test_quick_cpu_cli_writes_auditable_products(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONHASHSEED": "3407",
            }
        )
        with tempfile.TemporaryDirectory(prefix="e002-pilot-") as temporary:
            destination = Path(temporary) / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_e002_pilot.py",
                    "--quick",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(destination),
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
                    f"E002 quick CLI failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            required = {
                "manifest.json",
                "integrity.json",
                "resolved_config.json",
                "data.npz",
                "batch_indices.npz",
                "batch_probes.npz",
                "initial_state.pt",
                "method_fidelity.csv",
                "covariance_summary.csv",
                "taylor_summary.csv",
                "signed_spectral_transfer.csv",
                "endpoint_summary.csv",
                "metrics.json",
                "shared_taylor.png",
                "looksam_temporal.png",
                "on_policy_trajectories.png",
            }
            missing = sorted(name for name in required if not (destination / name).is_file())
            self.assertEqual(missing, [])
            self.assertTrue(all((destination / name).stat().st_size > 0 for name in required))
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runtime"]["device"]["requested"], "cpu")
            self.assertFalse(metrics["gates"]["gpu_identity_verified"])
            self.assertTrue(metrics["gates"]["method_fidelity_passed"])


if __name__ == "__main__":
    unittest.main()
