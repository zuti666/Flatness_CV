import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,Subset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, iImageNetR,iImageNetA,CUB, objectnet, omnibenchmark, vtab, iImageNetC, iTinyImageNetC, iTinyImageNetP,iDomainNet
import torch, math, json, os
from typing import Optional, List, Dict, Tuple
from torch.utils.data import DataLoader
class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_task_class_range(self, task):
        assert 0 <= task < len(self._increments), "Task index out of range"
        start = sum(self._increments[:task])
        end = start + self._increments[task]
        return start, end

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label



def fractional_loader(
    loader,
    fraction: Optional[float]=0.1,
    seed: Optional[int] = None,
    balanced: bool = True,
    batch_size: Optional[int] = None
):
    """Return a DataLoader restricted to a random subset of the original dataset."""
    
    frac = float(fraction)
    

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader
    
    total = len(dataset)
    
    if total <= 0:
        return loader

    subset_size = max(1, int(math.ceil(total * frac)))
    print(f"Origin dataset size: {total}, subset_size:{subset_size}")

    # ==== NEW: 保证“类别均衡”时至少覆盖所有类别 ====
    labels = extract_labels(dataset) if balanced else None
    if balanced and labels is not None:
        num_classes = len({int(y) for y in labels})
        if subset_size < num_classes:
            # 保障每类至少 1 个样本
            subset_size = num_classes
    # ==== END NEW ====

    if subset_size >= total:
        return loader

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.manual_seed(torch.seed())

    indices: List[int]

    # 若上面没能拿到 labels（例如 balanced=False 或提取失败），此处再尝试一次以兼容旧逻辑
    if labels is None:
        labels = extract_labels(dataset) if balanced else None

    if labels is not None and len(labels) == total:
        per_class: Dict[int, List[int]] = {}
        for local_idx, lbl in enumerate(labels):
            per_class.setdefault(int(lbl), []).append(local_idx)
        class_items = list(per_class.items())
        num_classes = len(class_items)
        chosen: List[int] = []
        if subset_size >= num_classes and num_classes > 0:
            remaining_pool: List[int] = []
            for _, idxs in class_items:
                perm = torch.randperm(len(idxs), generator=generator)
                first_pick = idxs[perm[0].item()]
                chosen.append(first_pick)
                if len(idxs) > 1:
                    remaining_pool.extend([idxs[i] for i in perm[1:].tolist()])
            remaining_needed = subset_size - num_classes
            if remaining_needed > 0 and remaining_pool:
                perm = torch.randperm(len(remaining_pool), generator=generator)[:remaining_needed]
                chosen.extend([remaining_pool[i] for i in perm.tolist()])
            indices = chosen[:subset_size]
        elif num_classes > 0:
            # 注意：这一分支只有在 subset_size < num_classes 时出现；
            # 但我们在“NEW”补丁中已把 subset_size 提升到 num_classes，通常不会走到这里。
            class_perm = torch.randperm(num_classes, generator=generator)[:subset_size]
            for class_idx in class_perm.tolist():
                idxs = class_items[class_idx][1]
                perm = torch.randperm(len(idxs), generator=generator)
                chosen.append(idxs[perm[0].item()])
            indices = chosen
        else:
            indices = torch.randperm(total, generator=generator)[:subset_size].tolist()
    else:
        indices = torch.randperm(total, generator=generator)[:subset_size].tolist()

    subset = Subset(dataset, indices)

    if batch_size is None:
        batch_size = loader.batch_size
        


    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
        "worker_init_fn": getattr(loader, "worker_init_fn", None),
        "pin_memory_device": getattr(loader, "pin_memory_device", ""),
        "timeout": loader.timeout,
    }
    if loader.num_workers > 0:
        dl_kwargs["persistent_workers"] = getattr(loader, "persistent_workers", False)
        prefetch_factor = getattr(loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor
    multiprocessing_context = getattr(loader, "multiprocessing_context", None)
    if multiprocessing_context is not None:
        dl_kwargs["multiprocessing_context"] = multiprocessing_context

    return DataLoader(subset, **dl_kwargs)




def extract_labels(dataset):
    # 返回每个样本的整数标签列表；支持 Subset/张量/ndarray/list
    if isinstance(dataset, Subset):
        base = extract_labels(dataset.dataset)
        idxs = dataset.indices.tolist() if hasattr(dataset.indices, "tolist") else list(dataset.indices)
        return [int(base[i]) for i in idxs] if base is not None else None
    for attr in ("labels","targets","y"):
        lab = getattr(dataset, attr, None)
        if lab is None or callable(lab): 
            continue
        if isinstance(lab, torch.Tensor): return lab.detach().cpu().flatten().tolist()
        if isinstance(lab, np.ndarray):  return lab.astype(int).flatten().tolist()
        if isinstance(lab, list):        return [int(x) for x in lab]
    return None

# def balanced_indices(dataset, fraction: float, seed: Optional[int], ensure_per_class: bool=True) -> List[int]:
#     # 类别均衡随机子采样：至少保证每类抽到 1 个（若 ensure_per_class=True）
#     total = len(dataset)
#     k = max(1, int(math.ceil(total * float(fraction))))
#     g = torch.Generator()
#     g.manual_seed(int(seed) if seed is not None else torch.seed())
#     labels = extract_labels(dataset)
#     if labels is None or len(labels) != total:
#         # 无法拿到标签，退化为纯随机
#         return torch.randperm(total, generator=g)[:k].tolist()
#     per_class: Dict[int, List[int]] = {}
#     for i, y in enumerate(labels):
#         per_class.setdefault(int(y), []).append(i)
#     # 先每类至少 1 个（若需要），剩余配额从“其余池”随机抽
#     chosen: List[int] = []
#     rest: List[int] = []
#     for _, idxs in per_class.items():
#         perm = torch.randperm(len(idxs), generator=g)
#         if ensure_per_class and len(chosen) < k:
#             chosen.append(idxs[perm[0].item()])
#             if len(idxs) > 1:
#                 rest += [idxs[i] for i in perm[1:].tolist()]
#         else:
#             rest += [idxs[i] for i in perm.tolist()]
#     remain = k - len(chosen)
#     if remain > 0 and rest:
#         perm = torch.randperm(len(rest), generator=g)[:remain]
#         chosen += [rest[i] for i in perm.tolist()]
#     return chosen[:k]


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == 'domainnet':
        return iDomainNet(args)
    elif name == "tiny_imagenetc":
        return iTinyImageNetC(args)
    elif name == "tiny_imagenetp":
        return iTinyImageNetP(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
