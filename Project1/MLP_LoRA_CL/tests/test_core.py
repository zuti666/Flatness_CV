from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from small_cl.diagnostics import (
    _factor_jacobian_stats,
    collect_diagnostic_batches,
    effective_tangent_basis,
    exact_directional_diagnostics,
    ggn_safe_route_diagnostics,
    prospective_reachable_direction_diagnostics,
    reachable_coverage,
    subspace_overlap,
    trainable_coordinate_hvp_diagnostics,
)
from small_cl.data import prepare_dataset
from small_cl.metrics import continual_metrics
from small_cl.models import AdaptiveMLP, FullMLP
from small_cl.optimizers import make_sgd, train_batch
from small_cl.normalized_steps import (
    apply_factor_step_,
    factor_effective_delta,
    make_factor_step_candidate,
    resolve_factor_candidate,
    solve_effective_step_scale,
)
from small_cl.gauge import (
    actual_factor_step_metrics,
    apply_factor_gauge_,
    make_gauge_matrix,
    pullback_gradient_metrics,
    tangent_projector_distance,
)
from small_cl.experiment import (
    _reparameterize_at_effective_weight,
    _resolve_optimizer_schedule,
    _task_training_config,
)


class DataPreparationTests(unittest.TestCase):
    def test_nonoverlap_diagnostic_batches_are_deterministic(self) -> None:
        inputs = torch.arange(24, dtype=torch.float32).view(12, 2)
        targets = torch.arange(12)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        batches = collect_diagnostic_batches(dataset, 3, 4, torch.device("cpu"))
        self.assertEqual(len(batches), 4)
        self.assertTrue(torch.equal(torch.cat([batch[1] for batch in batches]), targets))

    def test_rotated_mnist_is_downloaded_and_manifested(self) -> None:
        config = {
            "data": {"name": "rotated_mnist", "root": "data", "download": True}
        }
        fake_train = torch.utils.data.TensorDataset(torch.zeros(7, 1), torch.zeros(7))
        fake_test = torch.utils.data.TensorDataset(torch.zeros(3, 1), torch.zeros(3))
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "small_cl.data.datasets.MNIST", side_effect=[fake_train, fake_test]
        ) as mnist:
            manifest = prepare_dataset(config, Path(directory))
            self.assertEqual(manifest["train_samples"], 7)
            self.assertEqual(manifest["test_samples"], 3)
            self.assertTrue(manifest["download_enabled"])
            self.assertEqual(mnist.call_count, 2)
            for call in mnist.call_args_list:
                self.assertTrue(call.kwargs["download"])
                self.assertEqual(Path(call.args[0]), Path(directory) / "data")


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        self.inputs = torch.randn(8, 4)

    def test_dense_and_lora_start_from_same_function(self) -> None:
        models = [
            AdaptiveMLP(self.base, parameterization=name, rank=2, lora_alpha=2)
            for name in (
                "dense", "random_subspace", "fixed_lora_tangent",
                "fixed_mature_tangent", "factor_lora", "balanced_lora",
                "projected_rank",
            )
        ]
        for model in models:
            self.assertTrue(torch.equal(model.effective_weight(), self.base.middle.weight))
            self.assertTrue(torch.equal(models[0](self.inputs), model(self.inputs)))

    def test_merge_reset_is_function_preserving(self) -> None:
        model = AdaptiveMLP(self.base, parameterization="lora", rank=2, lora_alpha=2)
        with torch.no_grad():
            model.lora_b.normal_()
        before = model(self.inputs).detach()
        model.merge_and_reset()
        after = model(self.inputs).detach()
        self.assertLess(float((before - after).abs().max()), 1e-6)
        self.assertEqual(float(model.effective_delta().abs().max()), 0.0)

    def test_initial_lora_reachable_projection(self) -> None:
        model = AdaptiveMLP(self.base, parameterization="lora", rank=2, lora_alpha=2)
        geometry = effective_tangent_basis(model)
        delta = torch.randn(3, 2) @ model.lora_a.detach()
        self.assertAlmostEqual(reachable_coverage(geometry, delta), 1.0, places=5)
        self.assertAlmostEqual(subspace_overlap(geometry, geometry, 9), 1.0, places=5)

    def test_linear_control_dimensions(self) -> None:
        expected = 3 * 2
        for name in ("random_subspace", "fixed_lora_tangent"):
            model = AdaptiveMLP(self.base, parameterization=name, rank=2, subspace_seed=4)
            self.assertEqual(model.effective_subspace_basis().shape, (9, expected))

        random_control = AdaptiveMLP(
            self.base, parameterization="random_subspace", rank=2,
            subspace_seed=4, subspace_dim=7,
        )
        self.assertEqual(random_control.effective_subspace_basis().shape, (9, 7))
        mature = AdaptiveMLP(
            self.base, parameterization="fixed_mature_tangent", rank=2,
            subspace_seed=4, subspace_dim=8,
        )
        self.assertEqual(mature.effective_subspace_basis().shape, (9, 8))

    def test_balancing_preserves_effective_weight(self) -> None:
        model = AdaptiveMLP(self.base, parameterization="balanced_lora", rank=2)
        with torch.no_grad():
            model.lora_b.normal_()
            model.lora_a.normal_()
        before = model.effective_weight().detach().clone()
        model.rebalance_factors()
        self.assertLess(float((before - model.effective_weight()).abs().max()), 1e-5)
        gap = model.factor_geometry()["factor_balance_gap"]
        self.assertLess(gap, 1e-4)

    def test_factor_gauge_scaling_preserves_initial_function(self) -> None:
        torch.manual_seed(13)
        small = AdaptiveMLP(
            self.base, parameterization="factor_lora", rank=2,
            factor_gauge_scale=0.1,
        )
        torch.manual_seed(13)
        large = AdaptiveMLP(
            self.base, parameterization="factor_lora", rank=2,
            factor_gauge_scale=10.0,
        )
        self.assertEqual(float(small.effective_delta().abs().max()), 0.0)
        self.assertEqual(float(large.effective_delta().abs().max()), 0.0)
        self.assertTrue(torch.equal(small(self.inputs), large(self.inputs)))
        self.assertGreater(float(large.lora_a.norm()), float(small.lora_a.norm()))

    def test_common_endpoint_reparameterization_preserves_function(self) -> None:
        dense = AdaptiveMLP(self.base, parameterization="dense", rank=2)
        with torch.no_grad():
            dense.dense_delta.normal_(std=0.05)
        before_weight = dense.effective_weight().detach().clone()
        before_logits = dense(self.inputs).detach().clone()
        config = {
            "model": {
                "input_dim": 4,
                "hidden_dim": 3,
                "num_classes": 2,
                "activation": "softplus",
                "rank": 2,
            }
        }
        factor = _reparameterize_at_effective_weight(
            dense, config, "factor_lora", lora_alpha=2.0, subspace_seed=17
        )
        self.assertEqual(factor.parameterization, "factor_lora")
        self.assertEqual(float(factor.effective_delta().abs().max()), 0.0)
        self.assertTrue(torch.equal(before_weight, factor.effective_weight()))
        self.assertTrue(torch.equal(before_logits, factor(self.inputs)))

    def test_mature_factor_gauges_preserve_function_and_tangent(self) -> None:
        model = AdaptiveMLP(self.base, parameterization="factor_lora", rank=2)
        with torch.no_grad():
            model.lora_a.normal_()
            model.lora_b.normal_()
        reference_weight = model.effective_weight().detach().clone()
        reference_logits = model(self.inputs).detach().clone()
        reference_basis = effective_tangent_basis(model)
        for name in ("identity", "scalar_0.5", "scalar_2", "orthogonal", "anisotropic_4"):
            with self.subTest(name=name):
                transformed = copy.deepcopy(model)
                matrix = make_gauge_matrix(
                    name, transformed.rank, 37,
                    device=transformed.lora_a.device, dtype=transformed.lora_a.dtype,
                )
                apply_factor_gauge_(transformed, matrix)
                self.assertLess(
                    float((transformed.effective_weight() - reference_weight).norm()), 1e-5
                )
                self.assertLess(float((transformed(self.inputs) - reference_logits).abs().max()), 1e-5)
                self.assertLess(
                    tangent_projector_distance(
                        reference_basis, effective_tangent_basis(transformed)
                    ),
                    1e-3,
                )

    def test_orthogonal_gauge_preserves_pullback_but_scalar_does_not(self) -> None:
        model = AdaptiveMLP(self.base, parameterization="factor_lora", rank=2)
        with torch.no_grad():
            model.lora_a.normal_()
            model.lora_b.normal_()
        old_gradient = torch.randn_like(model.base_weight)
        new_gradient = torch.randn_like(model.base_weight)
        reference = pullback_gradient_metrics(model, old_gradient, new_gradient)
        orthogonal = copy.deepcopy(model)
        apply_factor_gauge_(
            orthogonal,
            make_gauge_matrix(
                "orthogonal", 2, 17,
                device=model.lora_a.device, dtype=model.lora_a.dtype,
            ),
        )
        orthogonal_result = pullback_gradient_metrics(
            orthogonal, old_gradient, new_gradient
        )
        self.assertAlmostEqual(
            reference["pullback_old_new_inner"],
            orthogonal_result["pullback_old_new_inner"],
            places=5,
        )
        scalar = copy.deepcopy(model)
        apply_factor_gauge_(
            scalar,
            make_gauge_matrix(
                "scalar_2", 2, 17,
                device=model.lora_a.device, dtype=model.lora_a.dtype,
            ),
        )
        scalar_result = pullback_gradient_metrics(scalar, old_gradient, new_gradient)
        self.assertGreater(
            abs(
                reference["pullback_old_new_inner"]
                - scalar_result["pullback_old_new_inner"]
            ),
            1e-4,
        )


class OptimizerAndDiagnosticTests(unittest.TestCase):
    def test_factor_step_normalization_hits_exact_effective_norm(self) -> None:
        torch.manual_seed(43)
        base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        model = AdaptiveMLP(base, parameterization="factor_lora", rank=2, lora_alpha=2)
        with torch.no_grad():
            model.lora_a.normal_()
            model.lora_b.normal_()
        delta_a = 0.03 * torch.randn_like(model.lora_a)
        delta_b = 0.03 * torch.randn_like(model.lora_b)
        raw = factor_effective_delta(model, delta_a, delta_b)
        target = 0.37 * float(raw.norm())
        alpha, resolved, relative_error = solve_effective_step_scale(
            model, delta_a, delta_b, target
        )
        self.assertGreater(alpha, 0.0)
        self.assertLess(relative_error, 1e-7)
        self.assertAlmostEqual(float(resolved.norm()), target, places=6)
        before = model.effective_weight().detach().clone()
        apply_factor_step_(model, alpha * delta_a, alpha * delta_b)
        actual = model.effective_weight().detach() - before
        self.assertLess(float((actual - resolved).norm()), 2e-6)

    def test_normalized_candidate_records_true_sam_weight_correction(self) -> None:
        torch.manual_seed(47)
        base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        model = AdaptiveMLP(base, parameterization="factor_lora", rank=2, lora_alpha=2)
        with torch.no_grad():
            model.lora_b.normal_(std=0.2)
        inputs = torch.randn(12, 4)
        targets = torch.randint(0, 2, (12,))
        configuration = {
            "lr": 0.03,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "sam_rho": 0.01,
            "perturbation_metric": "effective_weight",
            "grad_clip": None,
        }
        candidate = make_factor_step_candidate(
            model, inputs, targets, "sam", configuration
        )
        raw = factor_effective_delta(model, candidate.delta_a, candidate.delta_b)
        target = 0.5 * float(raw.norm())
        _, _, resolved, metrics = resolve_factor_candidate(
            model, candidate, "normalized", target
        )
        self.assertAlmostEqual(float(resolved.norm()), target, places=6)
        self.assertGreaterEqual(metrics["sam_raw_correction_ratio"], 0.0)
        self.assertGreaterEqual(metrics["sam_resolved_correction_ratio"], 0.0)
        self.assertGreaterEqual(metrics["sam_resolved_update_cosine"], -1.0)
        self.assertLessEqual(metrics["sam_resolved_update_cosine"], 1.0)

    def test_analytic_factor_jacobian_spectrum_matches_explicit_svd(self) -> None:
        torch.manual_seed(31)
        a = torch.randn(2, 4)
        b = torch.randn(3, 2)
        scale = 0.7
        directions = []
        for component in range(2):
            for column in range(4):
                unit = torch.zeros(4)
                unit[column] = 1
                directions.append((scale * torch.outer(b[:, component], unit)).flatten())
        for row in range(3):
            for component in range(2):
                unit = torch.zeros(3)
                unit[row] = 1
                directions.append((scale * torch.outer(unit, a[component])).flatten())
        singular = torch.linalg.svdvals(torch.stack(directions, dim=1))
        tolerance = singular.max() * 14 * torch.finfo(singular.dtype).eps
        nonzero = singular[singular > tolerance]
        operator, condition, rank = _factor_jacobian_stats(a, b, scale)
        self.assertAlmostEqual(operator, float(nonzero.max()), places=5)
        self.assertAlmostEqual(condition, float(nonzero.max() / nonzero.min()), places=5)
        self.assertEqual(rank, len(nonzero))

    def test_task_specific_optimizer_controls(self) -> None:
        configuration = {
            "lr": 0.01,
            "momentum": 0.0,
            "task_a_lr": 0.02,
            "task_a_momentum": 0.9,
            "task_b_lr": 0.03,
            "parameterization_lrs": {"factor_lora": 0.04},
            "sam_rho": 0.02,
            "parameterization_sam_rhos": {"factor_lora": 0.01},
        }
        task_a = _task_training_config(configuration, 0, "factor_lora")
        factor_b = _task_training_config(configuration, 1, "factor_lora")
        dense_b = _task_training_config(configuration, 1, "dense")
        self.assertEqual((task_a["lr"], task_a["momentum"]), (0.02, 0.9))
        self.assertEqual(factor_b["lr"], 0.04)
        self.assertEqual(dense_b["lr"], 0.03)
        self.assertEqual(factor_b["sam_rho"], 0.01)
        self.assertEqual(dense_b["sam_rho"], 0.02)

    def test_optimizer_schedule_resolution(self) -> None:
        self.assertEqual(
            _resolve_optimizer_schedule(
                {"optimizer": "sgd", "optimizer_schedule": ["SAM", "sgd"]}, 2
            ),
            ["sam", "sgd"],
        )
        self.assertEqual(
            _resolve_optimizer_schedule({"optimizer": "sam", "optimizer_schedule": None}, 2),
            ["sam", "sam"],
        )
        with self.assertRaises(ValueError):
            _resolve_optimizer_schedule(
                {"optimizer": "sgd", "optimizer_schedule": ["sgd"]}, 2
            )

    def _run_method(self, method: str) -> None:
        torch.manual_seed(3)
        base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        model = AdaptiveMLP(base, parameterization="dense", rank=2)
        configuration = {
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "sam_rho": 0.03,
            "gam_radius": 0.01,
            "gam_weight": 0.02,
            "grad_clip": None,
        }
        optimizer = make_sgd(model.trainable_parameters(), configuration)
        result = train_batch(
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.randn(12, 4),
            torch.randint(0, 2, (12,)),
            method,
            configuration,
        )
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertTrue(torch.isfinite(model.effective_weight()).all())

    def test_all_training_methods(self) -> None:
        for method in ("sgd", "sam", "gam_fd", "gam_exact", "random_perturb"):
            with self.subTest(method=method):
                self._run_method(method)

    def test_all_parameterizations_take_an_sgd_step(self) -> None:
        configuration = {
            "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0,
            "sam_rho": 0.03, "gam_radius": 0.01, "gam_weight": 0.02,
            "grad_clip": None,
        }
        for name in (
            "dense", "random_subspace", "fixed_lora_tangent", "fixed_mature_tangent",
            "factor_lora", "balanced_lora", "projected_rank"
        ):
            with self.subTest(parameterization=name):
                torch.manual_seed(11)
                base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
                model = AdaptiveMLP(base, parameterization=name, rank=1)
                before = model.effective_weight().detach().clone()
                optimizer = make_sgd(model.trainable_parameters(), configuration)
                train_batch(
                    model, optimizer, nn.CrossEntropyLoss(), torch.randn(12, 4),
                    torch.randint(0, 2, (12,)), "sgd", configuration,
                )
                self.assertGreater(float((model.effective_weight() - before).norm()), 0.0)
                if name == "projected_rank":
                    self.assertLessEqual(int(torch.linalg.matrix_rank(model.dense_delta)), 1)

    def test_actual_factor_step_decomposition_closes(self) -> None:
        torch.manual_seed(19)
        base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        model = AdaptiveMLP(base, parameterization="factor_lora", rank=2)
        with torch.no_grad():
            model.lora_b.normal_()
        before_a = model.lora_a.detach().clone()
        before_b = model.lora_b.detach().clone()
        before_weight = model.effective_weight().detach().clone()
        configuration = {
            "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0,
            "sam_rho": 0.03, "gam_radius": 0.01, "gam_weight": 0.02,
            "grad_clip": None,
        }
        optimizer = make_sgd(model.trainable_parameters(), configuration)
        train_batch(
            model, optimizer, nn.CrossEntropyLoss(), torch.randn(12, 4),
            torch.randint(0, 2, (12,)), "sgd", configuration,
        )
        result = actual_factor_step_metrics(model, before_a, before_b, before_weight)
        self.assertLess(result["factor_step_closure_relative_error"], 1e-4)

    def test_effective_weight_perturbation_is_matched(self) -> None:
        configuration = {
            "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0,
            "sam_rho": 0.02, "gam_radius": 0.02, "gam_weight": 0.02,
            "perturbation_metric": "effective_weight", "grad_clip": None,
        }
        for name in ("dense", "factor_lora"):
            with self.subTest(parameterization=name):
                torch.manual_seed(21)
                base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
                model = AdaptiveMLP(base, parameterization=name, rank=1)
                optimizer = make_sgd(model.trainable_parameters(), configuration)
                result = train_batch(
                    model, optimizer, nn.CrossEntropyLoss(), torch.randn(10, 4),
                    torch.randint(0, 2, (10,)), "sam", configuration,
                )
                self.assertAlmostEqual(result["effective_perturbation_norm"], 0.02, places=5)

    def test_exact_directional_diagnostic_and_fd_error(self) -> None:
        torch.manual_seed(5)
        base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        model = AdaptiveMLP(base, parameterization="dense", rank=2)
        inputs = torch.randn(10, 4)
        targets = torch.randint(0, 2, (10,))
        start = model.effective_weight().detach().clone()
        end = start + 0.01 * torch.randn_like(start)
        result = exact_directional_diagnostics(
            model, inputs, targets, start, end, finite_difference_radii=[1e-2, 1e-3]
        )
        self.assertTrue(math.isfinite(result["directional_curvature"]))
        self.assertIn("0.001", result["hvp_fd_relative_error"])
        self.assertLess(result["hvp_fd_relative_error"]["0.001"], 0.05)

    def test_ggn_safe_route_outputs(self) -> None:
        torch.manual_seed(8)
        base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        model = AdaptiveMLP(base, parameterization="fixed_lora_tangent", rank=1)
        old_batch = (torch.randn(8, 4), torch.randint(0, 2, (8,)))
        new_batch = (torch.randn(8, 4), torch.randint(0, 2, (8,)))
        start = model.effective_weight().detach().clone()
        end = start + 0.01 * model.effective_subspace_basis()[:, 0].view_as(start)
        result = ggn_safe_route_diagnostics(
            model, old_batch, new_batch, start, end,
            model.effective_subspace_basis(), 0.01, 0.01, 2,
        )
        for key in ("P_r_lambda", "tau_r", "kappa_G", "old_new_ggn_top_overlap"):
            self.assertTrue(math.isfinite(result[key]))

    def test_prospective_reachable_direction_outputs(self) -> None:
        torch.manual_seed(18)
        base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
        model = AdaptiveMLP(base, parameterization="factor_lora", rank=1)
        old_batch = (torch.randn(8, 4), torch.randint(0, 2, (8,)))
        new_batch = (torch.randn(8, 4), torch.randint(0, 2, (8,)))
        result = prospective_reachable_direction_diagnostics(
            model,
            old_batch,
            new_batch,
            model.effective_weight().detach(),
            model.effective_subspace_basis(),
        )
        for key in (
            "prospective_reachable_gradient_fraction",
            "prospective_hessian_curvature",
            "prospective_ggn_curvature",
        ):
            self.assertTrue(math.isfinite(result[key]))

    def test_trainable_coordinate_hvp_diagnostic(self) -> None:
        for name in ("dense", "factor_lora"):
            with self.subTest(parameterization=name):
                torch.manual_seed(31)
                base = FullMLP(input_dim=4, hidden_dim=3, num_classes=2, activation="softplus")
                model = AdaptiveMLP(base, parameterization=name, rank=1)
                result = trainable_coordinate_hvp_diagnostics(
                    model, torch.randn(10, 4), torch.randint(0, 2, (10,)), 0.01
                )
                self.assertAlmostEqual(result["effective_weight_radius"], 0.01, places=5)
                self.assertTrue(math.isfinite(result["fd_relative_error"]))
                self.assertTrue(math.isfinite(result["pushed_forward_relative_error"]))


class MetricTests(unittest.TestCase):
    def test_known_accuracy_matrix(self) -> None:
        matrix = [
            [0.80, float("nan"), float("nan")],
            [0.70, 0.90, float("nan")],
            [0.60, 0.80, 0.85],
        ]
        result = continual_metrics(matrix)
        self.assertAlmostEqual(result["final_average_accuracy"], 0.75)
        self.assertAlmostEqual(result["backward_transfer"], -0.15)
        self.assertAlmostEqual(result["average_forgetting"], 0.15)


if __name__ == "__main__":
    unittest.main()
