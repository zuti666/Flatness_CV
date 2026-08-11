"""Focused regressions for PGSR subspaces, snapshots, and forked resume."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
for path in (str(PROJECT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models_LoRAbasedCL import inclora
from pgsr.bank import right_subspace_from_factors
from pgsr.selector import make_orthogonal_complement_candidate


def _load_pgsr_learner_class():
    spec = importlib.util.spec_from_file_location("pgsr_inclora_test_module", PROJECT_DIR / "pgsr_inclora.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Learner


def test_rank_deficient_update_uses_seeded_fresh_supplement():
    output = torch.arange(1, 8, dtype=torch.float32).unsqueeze(1)
    direction = torch.arange(1, 10, dtype=torch.float32)
    b = torch.zeros(7, 3)
    a = torch.zeros(3, 9)
    b[:, :1] = output
    a[0] = direction

    basis, spectrum = right_subspace_from_factors(b, a, rank=3, supplement_seed=17)
    expected = direction / direction.norm()

    assert int((spectrum > spectrum.max() * 1e-6).sum()) == 1
    assert torch.allclose(
        basis[:, :1] @ basis[:, :1].t(),
        expected[:, None] @ expected[None, :],
        atol=2e-6,
    )
    assert torch.allclose(basis.t() @ basis, torch.eye(3), atol=2e-6)


def test_zero_update_supplement_does_not_depend_on_unused_a():
    b = torch.zeros(6, 3)
    a_one = torch.randn(3, 8)
    a_two = torch.randn(3, 8)

    basis_one, spectrum_one = right_subspace_from_factors(
        b, a_one, rank=3, supplement_seed=23
    )
    basis_two, spectrum_two = right_subspace_from_factors(
        b, a_two, rank=3, supplement_seed=23
    )

    assert torch.count_nonzero(spectrum_one) == 0
    assert torch.count_nonzero(spectrum_two) == 0
    assert torch.equal(basis_one, basis_two)


def test_perpendicular_control_is_orthogonal_to_history_union():
    gradients = [torch.randn(7, 12), torch.randn(5, 12)]
    history = []
    for seed in (3, 7):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        candidate = []
        for _ in gradients:
            basis, _ = torch.linalg.qr(torch.randn(12, 3, generator=generator))
            candidate.append(basis)
        history.append(candidate)

    perpendicular = make_orthogonal_complement_candidate(
        history, gradients, 3, seed=11
    )

    for site_index, basis in enumerate(perpendicular):
        assert torch.allclose(basis.t() @ basis, torch.eye(3), atol=2e-6)
        for historical in history:
            assert float((basis.t() @ historical[site_index]).abs().max()) < 2e-6


def test_vit_snapshot_builder_honors_explicit_task_idx():
    captured = {}

    class FakeViT:
        def eval(self):
            return self

    class FakeLoRA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.task_id = kwargs["cur_task_index"]

    learner = inclora.Learner.__new__(inclora.Learner)
    learner.args = {
        "lora_rank": 4,
        "backbone_type": "vit_base_patch16_224",
        "filepath": "/tmp/pgsr_snapshot_test/",
        "increment": 5,
    }
    learner._cur_task = 7

    with patch.object(inclora.timm, "create_model", lambda *args, **kwargs: FakeViT()):
        with patch.object(inclora, "LoRA_ViT_timm", FakeLoRA):
            backbone = learner._build_incremental_lora(eval_mode=True, task_idx=2)

    assert captured["cur_task_index"] == 2
    assert backbone.task_id == 2


def test_fork_resume_copies_full_additive_history(tmp_path=None):
    if tmp_path is None:
        with tempfile.TemporaryDirectory(prefix="pgsr_resume_test_") as directory:
            test_fork_resume_copies_full_additive_history(Path(directory))
        return

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    for task_id in (0, 1):
        (source / f"lora_w_a_{task_id}.pt").write_bytes(b"a")
        (source / f"lora_w_b_{task_id}.pt").write_bytes(b"b")
        (source / f"lora_meta_{task_id}.json").write_text("[]", encoding="utf-8")
    (source / "fc_state_1.pt").write_bytes(b"fc")
    (source / "pgsr").mkdir()
    (source / "pgsr" / "task_01_selection.json").write_text("{}", encoding="utf-8")

    class DummyNetwork:
        def __init__(self):
            self.updated_to = None
            self.loaded = None

        def update_fc(self, total_classes):
            self.updated_to = total_classes

        def load_fc(self, directory, task_idx):
            self.loaded = (directory, task_idx)

    class DummyDataManager:
        @staticmethod
        def get_task_class_range(task_idx):
            return task_idx * 5, (task_idx + 1) * 5

    learner_class = _load_pgsr_learner_class()
    learner = learner_class.__new__(learner_class)
    dummy_network = DummyNetwork()
    learner.args = {"filepath": str(target)}
    learner._network = dummy_network
    learner._unwrap_network = lambda: dummy_network
    learner._refresh_distributed_context = lambda: None
    learner._prepare_network = lambda: None
    learner._log = lambda message: None

    learner.load_task_checkpoint(str(source), 1, DummyDataManager())

    for task_id in (0, 1):
        assert (target / f"lora_w_a_{task_id}.pt").read_bytes() == b"a"
        assert (target / f"lora_w_b_{task_id}.pt").read_bytes() == b"b"
    assert (target / "pgsr" / "task_01_selection.json").exists()
    assert learner._cur_task == 1
    assert learner._known_classes == learner._total_classes == 10
    assert dummy_network.updated_to == 10
    assert dummy_network.loaded == (str(target.resolve()) + "/", 1)


if __name__ == "__main__":
    checks = (
        test_rank_deficient_update_uses_seeded_fresh_supplement,
        test_zero_update_supplement_does_not_depend_on_unused_a,
        test_perpendicular_control_is_orthogonal_to_history_union,
        test_vit_snapshot_builder_honors_explicit_task_idx,
        test_fork_resume_copies_full_additive_history,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"{len(checks)} PGSR regression checks passed")
