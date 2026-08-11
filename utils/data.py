import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from utils.toolkit import split_images_labels
import re
from typing import Optional, Callable, Any
import os
import pickle
from PIL import Image
import yaml
try:
    from scipy import io as scipy_io
except Exception:
    scipy_io = None

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    loader_mode = "pil"
    loader_options = None

    def __init__(self, args=None):
        self.args = args


class iFlowers102(iData):
    use_path = True

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.train_trsf, self.test_trsf, self.common_trsf = _build_cub_transforms()

    def download_data(self):
        root = _resolve_existing_dataset_dir(
            self.args,
            "flowers-102"
        )

        jpg_dir = os.path.join(root, "jpg")
        label_file = os.path.join(root, "imagelabels.mat")
        split_file = os.path.join(root, "setid.mat")

        assert scipy_io is not None, "Flowers102 requires scipy"

        labels = scipy_io.loadmat(label_file)["labels"][0] - 1
        split = scipy_io.loadmat(split_file)

        train_ids = np.concatenate([split["trnid"][0], split["valid"][0]]) - 1
        test_ids = split["tstid"][0] - 1

        all_images = sorted(os.listdir(jpg_dir))

        def build(ids):
            data, targets = [], []
            for i in ids:
                path = os.path.join(jpg_dir, all_images[i])
                data.append(path)
                targets.append(labels[i])
            return np.array(data), np.array(targets)

        self.train_data, self.train_targets = build(train_ids)
        self.test_data, self.test_targets = build(test_ids)

        self.class_order = np.arange(102).tolist()


class iOxfordPet(iData):
    use_path = True

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.train_trsf, self.test_trsf, self.common_trsf = _build_cub_transforms()

    def download_data(self):
        root = _resolve_existing_dataset_dir(
            self.args,
            "oxford-iiit-pet"
        )

        img_dir = os.path.join(root, "images")
        ann_dir = os.path.join(root, "annotations")

        def read_split(file):
            data, targets = [], []
            with open(file, "r") as f:
                for line in f:
                    name, cls, *_ = line.strip().split()
                    data.append(os.path.join(img_dir, name + ".jpg"))
                    targets.append(int(cls) - 1)
            return np.array(data), np.array(targets)

        self.train_data, self.train_targets = read_split(
            os.path.join(ann_dir, "trainval.txt")
        )
        self.test_data, self.test_targets = read_split(
            os.path.join(ann_dir, "test.txt")
        )

        self.class_order = np.arange(37).tolist()

class iAircraft(iData):
    use_path = True
    loader_mode = "torchvision_tensor"
    loader_options = {
        "pin_memory": True,
        "persistent_workers": True,
    }

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.train_trsf, self.test_trsf, self.common_trsf = _build_aircraft_transforms()

    def download_data(self):
        root = _resolve_existing_dataset_dir(
            self.args,
            "fgvc-aircraft-2013b/data"
        )

        img_dir = os.path.join(root, "images")

        # Step 1: Collect all unique labels from all splits
        all_labels = []
        split_files = [
            "images_variant_train.txt",
            "images_variant_val.txt",
            "images_variant_test.txt"
        ]
        
        for split_file in split_files:
            file_path = os.path.join(root, split_file)
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) < 2:
                        continue
                    label = " ".join(parts[1:])
                    all_labels.append(label)

        # Build global label_map from all unique labels
        label_map = {}
        cur = 0
        for label in all_labels:
            if label not in label_map:
                label_map[label] = cur
                cur += 1

        # Step 2: Read data using the global label_map
        def read_file_with_global_map(file):
            data, targets = [], []
            with open(file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) < 2:
                        continue
                    name = parts[0]
                    label_str = " ".join(parts[1:])
                    
                    # Use the pre-built global map
                    targets.append(label_map[label_str])
                    data.append(os.path.join(img_dir, name + ".jpg"))
            return np.array(data), np.array(targets)

        train_data, train_targets = read_file_with_global_map(
            os.path.join(root, "images_variant_train.txt")
        )
        val_data, val_targets = read_file_with_global_map(
            os.path.join(root, "images_variant_val.txt")
        )
        test_data, test_targets = read_file_with_global_map(
            os.path.join(root, "images_variant_test.txt")
        )

        self.train_data = np.concatenate([train_data, val_data])
        self.train_targets = np.concatenate([train_targets, val_targets])
        self.test_data = test_data
        self.test_targets = test_targets

        self.class_order = np.arange(100).tolist()

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        data_root = _preferred_data_root(self.args, create=True)
        train_dataset = datasets.cifar.CIFAR10(data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR10_224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False
        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(10).tolist()

    def download_data(self):
        data_root = _preferred_data_root(self.args, create=True)
        train_dataset = datasets.cifar.CIFAR10(data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        data_root = _preferred_data_root(self.args, create=True)
        train_dataset = datasets.cifar.CIFAR100(data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t


DEFAULT_SHARED_DATA_ROOT = "/data/140-0/datasets"
PROJECT_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _normalize_path(path):
    return os.path.abspath(os.path.expanduser(str(path)))


def _unique_paths(paths):
    out = []
    seen = set()
    for path in paths:
        if not path:
            continue
        norm = _normalize_path(path)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _explicit_data_roots(args=None):
    roots = []
    if isinstance(args, dict):
        for key in ("data_root", "dataset_root", "datasets_root", "data_dir", "dataset_dir"):
            value = args.get(key)
            if isinstance(value, (list, tuple)):
                roots.extend(value)
            elif value:
                roots.append(value)
    return _unique_paths(roots)


def _candidate_data_roots(args=None):
    roots = []
    roots.extend(_explicit_data_roots(args))
    for env_key in ("FLATNESS_CV_DATA_ROOT", "DATA_ROOT", "DATASET_ROOT"):
        value = os.environ.get(env_key)
        if value:
            roots.append(value)
    roots.extend([DEFAULT_SHARED_DATA_ROOT, PROJECT_DATA_ROOT])
    return _unique_paths(roots)


def _preferred_data_root(args=None, create=False):
    explicit_roots = _explicit_data_roots(args)
    if explicit_roots:
        root = explicit_roots[0]
        if create:
            os.makedirs(root, exist_ok=True)
        return root

    for root in _candidate_data_roots(args):
        if os.path.isdir(root):
            return root

    if create:
        os.makedirs(DEFAULT_SHARED_DATA_ROOT, exist_ok=True)
        return DEFAULT_SHARED_DATA_ROOT
    return DEFAULT_SHARED_DATA_ROOT


def _expand_root_relative(root, relative_path):
    root = _normalize_path(root)
    rel = str(relative_path).strip()
    if not rel:
        return []
    if os.path.isabs(rel):
        return [rel]
    rel = rel.lstrip("./")
    if rel.startswith("data/"):
        rel = rel[len("data/"):]

    candidates = [os.path.join(root, rel)]
    parts = rel.split("/", 1)
    if len(parts) == 2 and os.path.basename(root) == parts[0]:
        candidates.append(os.path.join(root, parts[1]))
    return _unique_paths(candidates)


def _candidate_paths_from_roots(args, *relative_paths):
    candidates = []
    for relative_path in relative_paths:
        if not relative_path:
            continue
        if os.path.isabs(str(relative_path)):
            candidates.append(str(relative_path))
            continue
        for root in _candidate_data_roots(args):
            candidates.extend(_expand_root_relative(root, relative_path))
    return _unique_paths(candidates)


def _find_existing_dir(*candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _resolve_existing_dir(*candidates):
    path = _find_existing_dir(*candidates)
    if path is not None:
        return path
    raise FileNotFoundError(
        "None of the dataset directories exist: {}".format(", ".join(candidates))
    )


def _find_existing_file(*candidates):
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _resolve_existing_file(*candidates):
    path = _find_existing_file(*candidates)
    if path is not None:
        return path
    raise FileNotFoundError(
        "None of the dataset files exist: {}".format(", ".join(candidates))
    )


def _find_existing_dataset_dir(args, *relative_paths):
    return _find_existing_dir(*_candidate_paths_from_roots(args, *relative_paths))


def _resolve_existing_dataset_dir(args, *relative_paths):
    candidates = _candidate_paths_from_roots(args, *relative_paths)
    return _resolve_existing_dir(*candidates)


def _find_existing_dataset_file(args, *relative_paths):
    return _find_existing_file(*_candidate_paths_from_roots(args, *relative_paths))


def _resolve_existing_dataset_file(args, *relative_paths):
    candidates = _candidate_paths_from_roots(args, *relative_paths)
    return _resolve_existing_file(*candidates)


def _find_cars196_raw_root(*candidates):
    for root in candidates:
        if not os.path.isdir(root):
            continue
        train_ok = os.path.isdir(os.path.join(root, "cars_train"))
        test_ok = os.path.isdir(os.path.join(root, "cars_test"))
        devkit_ok = os.path.isfile(os.path.join(root, "devkit", "cars_train_annos.mat"))
        test_ann_ok = os.path.isfile(os.path.join(root, "cars_test_annos_withlabels.mat"))
        if train_ok and test_ok and devkit_ok and test_ann_ok:
            return root
    return None


def _load_cars196_raw_split(root: str, train: bool):
    if scipy_io is None:
        raise ImportError(
            "scipy is required to read raw Stanford Cars annotations. "
            "Install scipy or prepare ImageFolder-style train/test directories."
        )

    if train:
        ann_path = os.path.join(root, "devkit", "cars_train_annos.mat")
        image_dir = os.path.join(root, "cars_train")
    else:
        ann_path = os.path.join(root, "cars_test_annos_withlabels.mat")
        image_dir = os.path.join(root, "cars_test")

    mat = scipy_io.loadmat(ann_path)["annotations"][0]

    images, labels = [], []
    for record in mat:
        class_id = int(record[4].item()) - 1
        image_name = str(record[5].item())
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Missing Cars196 image: {image_path}")
        images.append(image_path)
        labels.append(class_id)

    return np.array(images), np.array(labels, dtype=np.int64)


def _build_cub_transforms():
    train_trsf = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
    return train_trsf, test_trsf, common_trsf


def _build_aircraft_transforms():
    train_trsf = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
    return train_trsf, test_trsf, common_trsf


def _build_cars196_transforms():
    # Consistent with data/seq_cars196.py (Mammoth reference):
    # resize shorter side to 224 (preserving aspect ratio) then center-crop to 224x224.
    train_trsf = [
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]
    return train_trsf, test_trsf, common_trsf

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        data_root = _preferred_data_root(self.args, create=True)
        train_dataset = datasets.cifar.CIFAR100(data_root, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_root, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "ImageNet/train",
            "imagenet/train",
            "imagenet1000/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "ImageNet/val",
            "imagenet/val",
            "imagenet1000/val",
            "ImageNet/test",
            "imagenet/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "seed_1993_subset_100_imagenet/train",
            "imagenet100/train",
            "imagenet-100/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "seed_1993_subset_100_imagenet/test",
            "seed_1993_subset_100_imagenet/val",
            "imagenet100/test",
            "imagenet100/val",
            "imagenet-100/test",
            "imagenet-100/val",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "imagenet-r/train",
            "imagenetr/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "imagenet-r/test",
            "imagenetr/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "imagenet-a/train",
            "imageneta/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "imagenet-a/test",
            "imageneta/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetC(iData):
    """ImageNet-C reorganized to ImageFolder layout (ImageNet-R style).

    Expected directory (created by scripts/prepare_imagenet_c_split.py):
        data/imagenet-c-r/train/<wnid>/*.JPEG
        data/imagenet-c-r/test/<wnid>/*.JPEG

    Default assumes 200 classes to align with ImageNet-R subset.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args is not None and args.get("model_name", "") == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        # Default to 200 classes to match ImageNet-R style; will be reset
        # dynamically after reading data to the detected class count.
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "imagenet-c-r/train",
            "imagenet-c/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "imagenet-c-r/test",
            "imagenet-c/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # If class count differs from default 200 (e.g., using all classes),
        # update class_order accordingly to keep consistency downstream.
        num_classes = len(train_dset.classes)
        self.class_order = np.arange(num_classes).tolist()





class iTinyImageNetC(iData):
    """Tiny-ImageNet-C reorganized to ImageFolder layout.

    Expected directory (created by scripts/prepare_tiny_imagenet_c_split.py):
        data/tiny-imagenet-c-r/train/<wnid>/*.JPEG
        data/tiny-imagenet-c-r/test/<wnid>/*.JPEG
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args is not None and args.get("model_name", "") == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "tiny-imagenetc_1_noise_CL/train",
            "tiny-imagenetc-allcorruptions-CL/train",
            "tiny-imagenet-c-r/train",
            "tiny-imagenet-c/train",
            # Fallback for the current shared-dataset layout where only the
            # raw single-corruption tree is present.
            "tiny-imagenet-c/extracted/Tiny-ImageNet-C/gaussian_noise/2",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "tiny-imagenetc_1_noise_CL/test",
            "tiny-imagenetc-allcorruptions-CL/test",
            "tiny-imagenet-c-r/test",
            "tiny-imagenet-c/test",
            # Keep test loading working when only the raw corruption root is
            # available on the machine.
            "tiny-imagenet-c/extracted/Tiny-ImageNet-C/gaussian_noise/2",
        )

        train_dset = datasets.ImageFolder(train_dir,
                                        #   is_valid_file=keep_tiny_imagenet_c_cl_s123_id05,
                                        #   allow_empty=True,
                                          transform=self.train_trsf)
        test_dset = datasets.ImageFolder(test_dir,
                                        #  is_valid_file=keep_tiny_imagenet_c_cl_s123_id05,
                                        #  allow_empty=True,
                                          transform=self.train_trsf)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        num_classes = len(train_dset.classes)
        self.class_order = np.arange(num_classes).tolist()


class iTinyImageNetP(iData):
    """Tiny-ImageNet-P prepared to ImageFolder layout.

    Expected directory (created by data/prepare_tiny_imagenet_p_split.py):
        data/tiny_imagenerp_noise_CL/train/<wnid>/<corruption__test_xxxx>/fXXXX.jpg
        data/tiny_imagenerp_noise_CL/test/<wnid>/<corruption__test_xxxx>/fXXXX.jpg

    We flatten subfolders by ImageFolder which indexes leaf images; thus using
    nested folders under each class is fine.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args is not None and args.get("model_name", "") == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "data/tiny_imagenerp_noise_CL/train",
            "tiny_imagenerp_noise_CL/train",
            "tiny-imagenet-p-r/train",
            "tiny-imagenet-p/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "data/tiny_imagenerp_noise_CL/test",
            "tiny_imagenerp_noise_CL/test",
            "tiny-imagenet-p-r/test",
            "tiny-imagenet-p/test",
        )

        # train_dset = datasets.ImageFolder(train_dir)
        # test_dset = datasets.ImageFolder(test_dir)

        # 关键：在构造阶段用 is_valid_file 过滤，且允许空类
        train_dset = datasets.ImageFolder(
            train_dir,
            # is_valid_file=keep_every_k_file_TinyImageNetP,
            # allow_empty=True,
            transform=self.train_trsf,
        )
        test_dset = datasets.ImageFolder(
            test_dir,
            # is_valid_file=keep_every_k_file_TinyImageNetP,
            # allow_empty=True,
            transform=self.test_trsf,
        )

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        num_classes = len(train_dset.classes)
        self.class_order = np.arange(num_classes).tolist()





class CUB(iData):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf, self.test_trsf, self.common_trsf = _build_cub_transforms()
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "cub/train",
            "cub200/train",
            "cub-200/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "cub/test",
            "cub200/test",
            "cub-200/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        self.class_order = np.arange(len(train_dset.classes)).tolist()


class Cars196(iData):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf, self.test_trsf, self.common_trsf = _build_cars196_transforms()
        self.class_order = np.arange(196).tolist()

    def download_data(self):
        train_dir = _find_existing_dataset_dir(
            self.args,
            "cars196/train",
            "cars-196/train",
            "stanford-cars/train",
        )
        test_dir = _find_existing_dataset_dir(
            self.args,
            "cars196/test",
            "cars-196/test",
            "stanford-cars/test",
        )

        if train_dir is not None and test_dir is not None:
            train_dset = datasets.ImageFolder(train_dir)
            test_dset = datasets.ImageFolder(test_dir)

            self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
            self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
            self.class_order = np.arange(len(train_dset.classes)).tolist()
            return

        raw_root = _find_cars196_raw_root(
            *_candidate_paths_from_roots(
                self.args,
                "cars196",
                "cars-196",
                "stanford-cars",
            )
        )
        if raw_root is not None:
            self.train_data, self.train_targets = _load_cars196_raw_split(raw_root, train=True)
            self.test_data, self.test_targets = _load_cars196_raw_split(raw_root, train=False)
            self.class_order = np.arange(len(np.unique(self.train_targets))).tolist()
            return

        raise FileNotFoundError(
            "Cars196 expects either ImageFolder layout "
            "(cars196/train and cars196/test under a configured data root) "
            "or raw Stanford Cars layout "
            "(cars_train, cars_test, devkit/cars_train_annos.mat, cars_test_annos_withlabels.mat)."
        )


class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "objectnet/train",
            "objectnet-1.0/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "objectnet/test",
            "objectnet-1.0/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "omnibenchmark/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "omnibenchmark/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        train_dir = _resolve_existing_dataset_dir(
            self.args,
            "vtab-cil/vtab/train",
            "vtab/train",
        )
        test_dir = _resolve_existing_dataset_dir(
            self.args,
            "vtab-cil/vtab/test",
            "vtab/test",
        )

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class iDomainNet(iData):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

    def download_data(self):
        # Prefer the paper split files when present; keep the existing YAML
        # split fallback for the local full DomainNet layout.
        train_data_config = self._load_split_config("train")
        test_data_config = self._load_split_config("test")
        self.train_data = np.array(self._remap_paths(train_data_config['data']))
        self.train_targets = np.array(train_data_config['targets'])
        self.test_data = np.array(self._remap_paths(test_data_config['data']))
        self.test_targets = np.array(test_data_config['targets'])

    def _load_split_config(self, split):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        split_dir = self._configured_split_dir(project_root)
        pkl_path = _find_existing_file(
            os.path.join(split_dir, f"domainnet_{split}.pkl") if split_dir else "",
            os.path.join(project_root, "dataloaders", "splits", f"domainnet_{split}.pkl"),
            os.path.join("dataloaders", "splits", f"domainnet_{split}.pkl"),
        )
        if pkl_path is not None:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

        yaml_path = _find_existing_file(
            os.path.join(split_dir, f"domainnet_{split}.yaml") if split_dir else "",
            os.path.join(project_root, "dataloaders", "splits", f"domainnet_{split}.yaml"),
            os.path.join("dataloaders", "splits", f"domainnet_{split}.yaml"),
        )
        if yaml_path is None:
            yaml_path = _resolve_existing_dataset_file(
                self.args,
                f"DomainNet/splits/domainnet_{split}.yaml",
            )
        with open(yaml_path, "r") as f:
            return yaml.load(f, Loader=yaml.Loader)

    def _configured_split_dir(self, project_root):
        if not isinstance(self.args, dict):
            return None
        split_dir = self.args.get("domainnet_split_dir")
        if not split_dir:
            return None
        split_dir = os.path.expanduser(str(split_dir))
        if os.path.isabs(split_dir):
            return split_dir
        return os.path.join(project_root, split_dir)

    def _remap_paths(self, paths):
        data_path = self._domainnet_data_path()
        if data_path is None:
            return list(paths)
        data_path = data_path.rstrip("/")
        remapped = []
        for path in paths:
            path = str(path)
            if path.startswith("data/DomainNet"):
                path = path.replace("data/DomainNet", data_path, 1)
            remapped.append(path)
        return remapped

    def _domainnet_data_path(self):
        if hasattr(self, "_resolved_domainnet_data_path"):
            return self._resolved_domainnet_data_path

        resolved = None
        if not isinstance(self.args, dict):
            self._resolved_domainnet_data_path = resolved
            return resolved
        data_path = self.args.get("data_path") or self.args.get("domainnet_data_path")
        if data_path:
            self._resolved_domainnet_data_path = str(data_path)
            return self._resolved_domainnet_data_path
        for root in _candidate_data_roots(self.args):
            for rel in ("domainnet", "DomainNet"):
                candidate = os.path.join(root, rel)
                if all(os.path.isdir(os.path.join(candidate, d)) for d in self.domain_names):
                    self._resolved_domainnet_data_path = candidate
                    return self._resolved_domainnet_data_path

        # Existing local YAML splits may contain absolute image paths. Use them
        # to infer the root so old configs still work after pkl splits are added.
        yaml_path = _find_existing_dataset_file(
            self.args,
            "DomainNet/splits/domainnet_train.yaml",
        )
        if yaml_path is not None:
            with open(yaml_path, "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            for path in config.get("data", []):
                candidate = self._root_from_sample_path(path)
                if candidate is not None:
                    self._resolved_domainnet_data_path = candidate
                    return self._resolved_domainnet_data_path

        self._resolved_domainnet_data_path = resolved
        return self._resolved_domainnet_data_path

    def _root_from_sample_path(self, path):
        path = str(path)
        for domain in self.domain_names:
            marker = f"/{domain}/"
            idx = path.find(marker)
            if idx < 0:
                continue
            root = path[:idx]
            if all(os.path.isdir(os.path.join(root, d)) for d in self.domain_names):
                return root
        return None


def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr


# ---------------------------------------------------------------------------
# Heterogeneous 5-dataset CIL setting
# Task order: CUB(200) → Aircraft(100) → Cars196(196) → Flowers102(102) → OxfordPet(37)
# Global label space: 635 classes, each dataset gets a non-overlapping offset.
# Usage: set "dataset": "het5" and "shuffle": false in config.
# ---------------------------------------------------------------------------
_HET5_REGISTRY = [
    # (name,        iData_class,   offset,  num_classes)
    ("cub200",      None,            0,      200),
    ("aircraft",    None,          200,      100),
    ("cars196",     None,          300,      196),
    ("flowers102",  None,          496,      102),
    ("oxfordpet",   None,          598,       37),
]
_HET5_TOTAL = 635


class iHet5Datasets(iData):
    """Heterogeneous CIL: 5 fine-grained datasets chained sequentially.

    Each dataset is treated as one task. Labels are remapped to a global
    non-overlapping space so DataManager.get_dataset() works unchanged.

    Requires shuffle=False in config (class_order is identity [0..634]).
    Exposes ``task_splits`` so DataManager can set _increments directly.
    """

    use_path = True

    def __init__(self, args=None):
        super().__init__(args)
        self.train_trsf, self.test_trsf, self.common_trsf = _build_cub_transforms()
        self.task_splits = [ncls for _, _, _, ncls in _HET5_REGISTRY]  # [200,100,196,102,37]

    def download_data(self):
        # All classes are defined in this same module; reference them directly.
        _cls_map = {
            "cub200":     CUB,
            "aircraft":   iAircraft,
            "cars196":    Cars196,
            "flowers102": iFlowers102,
            "oxfordpet":  iOxfordPet,
        }

        tr_data, tr_tgt, te_data, te_tgt = [], [], [], []
        for name, _, offset, _ in _HET5_REGISTRY:
            idata = _cls_map[name](self.args)
            idata.download_data()
            tr_data.append(idata.train_data)
            tr_tgt.append(idata.train_targets.astype(np.int64) + offset)   # cast first: uint8 overflows at offset>255
            te_data.append(idata.test_data)
            te_tgt.append(idata.test_targets.astype(np.int64) + offset)

        self.train_data    = np.concatenate(tr_data)
        self.train_targets = np.concatenate(tr_tgt).astype(np.int64)
        self.test_data     = np.concatenate(te_data)
        self.test_targets  = np.concatenate(te_tgt).astype(np.int64)
        self.class_order   = np.arange(_HET5_TOTAL).tolist()  # identity, no shuffle
