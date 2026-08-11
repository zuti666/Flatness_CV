from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets
from torchvision.transforms import functional as TF


def task_angles(num_tasks: int, rotation_span: float, explicit: list[float] | None = None) -> list[float]:
    if explicit is not None:
        if len(explicit) != num_tasks:
            raise ValueError("data.angles must contain exactly data.num_tasks values")
        return [float(value) for value in explicit]
    if num_tasks == 1:
        return [0.0]
    return torch.linspace(-rotation_span / 2.0, rotation_span / 2.0, num_tasks).tolist()


class RotatedDataset(Dataset):
    def __init__(self, base: Dataset, angle: float) -> None:
        self.base = base
        self.angle = float(angle)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, target = self.base[index]
        image = TF.to_tensor(image) if not isinstance(image, torch.Tensor) else image.float()
        return TF.rotate(image, self.angle), int(target)


class SyntheticRotatedDigits(Dataset):
    """Download-free structured proxy used for CI and smoke tests, not final claims."""

    def __init__(self, size: int, angle: float, split_seed: int, noise: float = 0.35) -> None:
        prototype_generator = torch.Generator().manual_seed(1927)
        prototypes = torch.randn(10, 1, 28, 28, generator=prototype_generator)
        sample_generator = torch.Generator().manual_seed(split_seed)
        self.targets = torch.randint(0, 10, (size,), generator=sample_generator)
        images = prototypes[self.targets] + noise * torch.randn(
            size, 1, 28, 28, generator=sample_generator
        )
        self.images = TF.rotate(images, float(angle))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.targets[index])


def _subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


@dataclass
class TaskData:
    angles: list[float]
    train: list[Dataset]
    test: list[Dataset]
    metadata: dict | None = None


def _data_root(config: dict, project_root: Path) -> Path:
    root = Path(config["data"]["root"])
    root = root if root.is_absolute() else project_root / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_mnist(config: dict, project_root: Path) -> tuple[Dataset, Dataset]:
    """Load both MNIST splits, downloading them once when configured."""
    root = _data_root(config, project_root)
    download = bool(config["data"].get("download", True))
    try:
        train = datasets.MNIST(root, train=True, download=download)
        test = datasets.MNIST(root, train=False, download=download)
    except RuntimeError as error:
        if not download:
            raise RuntimeError(
                f"MNIST was not found under {root}. Set data.download: true or run "
                "run_grid.py --config <config> --prepare-data-only."
            ) from error
        raise RuntimeError(f"MNIST download or integrity check failed under {root}: {error}") from error
    return train, test


def prepare_dataset(config: dict, project_root: Path) -> dict[str, object]:
    """Materialize external data before a grid starts and return a short manifest."""
    name = config["data"]["name"].lower()
    if name == "rotated_mnist":
        train, test = _load_mnist(config, project_root)
        return {
            "name": name,
            "root": str(_data_root(config, project_root)),
            "train_samples": len(train),
            "test_samples": len(test),
            "download_enabled": bool(config["data"].get("download", True)),
        }
    if name in {"synthetic", "teacher_student"}:
        return {"name": name, "external_download": False}
    raise ValueError(f"Unsupported dataset: {config['data']['name']}")


def _teacher_subspaces(hidden_dim: int, rank: int, angle_degrees: float, seed: int):
    if 2 * rank > hidden_dim:
        raise ValueError("teacher target_rank must satisfy 2 * rank <= hidden_dim")
    generator = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(hidden_dim, 2 * rank, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(hidden_dim, 2 * rank, generator=generator))
    angle = torch.tensor(angle_degrees * torch.pi / 180.0)
    left_0, left_orthogonal = left[:, :rank], left[:, rank:]
    right_0, right_orthogonal = right[:, :rank], right[:, rank:]
    left_1 = angle.cos() * left_0 + angle.sin() * left_orthogonal
    right_1 = angle.cos() * right_0 + angle.sin() * right_orthogonal
    return (left_0, right_0), (left_1, right_1)


def _teacher_dataset(
    base_model,
    middle_weight: torch.Tensor,
    size: int,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(size, base_model.fc1.in_features, generator=generator)
    device = next(base_model.parameters()).device
    with torch.no_grad():
        logits = base_model.activation(base_model.fc1(inputs.to(device)))
        logits = base_model.activation(
            torch.nn.functional.linear(logits, middle_weight.to(device), base_model.middle.bias)
        )
        logits = base_model.fc3(logits)
        targets = logits.argmax(dim=1).cpu()
    return TensorDataset(inputs, targets)


def build_task_datasets(config: dict, project_root: Path, base_model=None) -> TaskData:
    data_config = config["data"]
    angles = task_angles(
        int(data_config["num_tasks"]),
        float(data_config["rotation_span"]),
        data_config.get("angles"),
    )
    name = data_config["name"].lower()
    train_tasks: list[Dataset] = []
    test_tasks: list[Dataset] = []

    if name == "rotated_mnist":
        train_base, test_base = _load_mnist(config, project_root)
        for task_id, angle in enumerate(angles):
            train_tasks.append(
                _subset(RotatedDataset(train_base, angle), data_config.get("train_subset"), config["seed"] + task_id)
            )
            test_tasks.append(
                _subset(RotatedDataset(test_base, angle), data_config.get("test_subset"), config["seed"] + 10_000 + task_id)
            )
    elif name == "synthetic":
        train_size = int(data_config.get("train_subset") or 1024)
        test_size = int(data_config.get("test_subset") or 256)
        for task_id, angle in enumerate(angles):
            train_tasks.append(SyntheticRotatedDigits(train_size, angle, 50_000 + config["seed"]))
            test_tasks.append(SyntheticRotatedDigits(test_size, angle, 60_000 + config["seed"]))
    elif name == "teacher_student":
        if base_model is None:
            raise ValueError("teacher_student data requires the shared base model")
        if int(data_config["num_tasks"]) != 2:
            raise ValueError("teacher_student is a two-task controlled experiment")
        teacher_config = data_config["teacher"]
        target_rank = int(teacher_config["target_rank"])
        principal_angle = float(teacher_config["principal_angle_degrees"])
        strength = float(teacher_config["update_strength"])
        pairs = _teacher_subspaces(
            base_model.middle.out_features,
            target_rank,
            principal_angle,
            int(teacher_config["seed"]),
        )
        teacher_weights = [
            base_model.middle.weight.detach().cpu() + strength * left @ right.T
            for left, right in pairs
        ]
        train_size = int(data_config.get("train_subset") or 2048)
        test_size = int(data_config.get("test_subset") or 1024)
        for task_id, weight in enumerate(teacher_weights):
            train_tasks.append(
                _teacher_dataset(base_model, weight, train_size, 70_000 + config["seed"] + task_id)
            )
            test_tasks.append(
                _teacher_dataset(base_model, weight, test_size, 80_000 + config["seed"] + task_id)
            )
        angles = [0.0, principal_angle]
        return TaskData(
            angles=angles,
            train=train_tasks,
            test=test_tasks,
            metadata={
                "target_rank": target_rank,
                "target_principal_angle_degrees": principal_angle,
                "teacher_update_strength": strength,
            },
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_config['name']}")
    return TaskData(angles=angles, train=train_tasks, test=test_tasks, metadata=None)


def build_unrotated_base_datasets(config: dict, project_root: Path) -> tuple[Dataset, Dataset]:
    data_config = config["data"]
    pretrain_angle = float(config["base"].get("pretrain_angle", 0.0))
    if data_config["name"].lower() == "synthetic":
        train_size = int(data_config.get("train_subset") or 1024)
        test_size = int(data_config.get("test_subset") or 256)
        return (
            SyntheticRotatedDigits(train_size, pretrain_angle, 50_000 + config["base"]["seed"]),
            SyntheticRotatedDigits(test_size, pretrain_angle, 60_000 + config["base"]["seed"]),
        )
    if data_config["name"].lower() == "teacher_student":
        raise ValueError("teacher_student must use base.mode=random")
    train_base, test_base = _load_mnist(config, project_root)
    return RotatedDataset(train_base, pretrain_angle), RotatedDataset(test_base, pretrain_angle)
