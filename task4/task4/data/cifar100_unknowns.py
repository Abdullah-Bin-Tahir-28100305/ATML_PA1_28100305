

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

try:
    from data.cifar10 import build_transforms
except ImportError:
    from .cifar10 import build_transforms


def _select_class_subset(dataset, class_names, cap, seed):
   
    name_to_idx = {name: i for i, name in enumerate(dataset.classes)}
    missing = [n for n in class_names if n not in name_to_idx]
    if missing:
        raise ValueError(f"CIFAR-100 fine classes not found: {missing}\n"
                         f"available example names: {dataset.classes[:10]} ...")
    wanted = {name_to_idx[n] for n in class_names}
    targets = np.asarray(dataset.targets)
    idx = np.where(np.isin(targets, list(wanted)))[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    idx = np.sort(idx[:cap])
    return Subset(dataset, idx.tolist())


def load_unknown_groups(cfg):
    
    tfm = build_transforms(cfg, train=False)            # clean eval transform
    ds = CIFAR100(cfg["known"]["root"], train=False, download=True, transform=tfm)
    cap = cfg["unknowns"]["images_per_group"]           # 800
    seed = cfg["seed"]
    near = _select_class_subset(ds, cfg["unknowns"]["near"], cap, seed)
    far = _select_class_subset(ds, cfg["unknowns"]["far"], cap, seed)
    return near, far


def unknown_class_names(dataset_subset):
   
    base = dataset_subset.dataset                       # underlying CIFAR100
    names = base.classes
    return [names[base.targets[i]] for i in dataset_subset.indices]
