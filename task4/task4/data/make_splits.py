

import os
import json
import numpy as np
from torch.utils.data import Subset

try:
    from utils import ensure_dir
except ImportError:
    from ..utils import ensure_dir


def _stratified_indices(labels, train_frac, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n_train = int(round(train_frac * len(idx_c)))
        train_idx.extend(idx_c[:n_train].tolist())
        val_idx.extend(idx_c[n_train:].tolist())
    return sorted(train_idx), sorted(val_idx)


def build_or_load_split(cfg, base_train_dataset):
   
    path = cfg["paths"]["splits_file"]
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    ensure_dir(os.path.dirname(path))
    labels = base_train_dataset.targets                 # CIFAR-10 integer labels
    tr, va = _stratified_indices(labels, cfg["known"]["train_frac"], cfg["seed"])
    split = {"seed": cfg["seed"], "train_frac": cfg["known"]["train_frac"],
             "train": tr, "val": va,
             "n_train": len(tr), "n_val": len(va)}
    with open(path, "w") as f:
        json.dump(split, f)
    return split


def make_train_val_subsets(train_aug_dataset, train_clean_dataset, split):
   
    train_subset = Subset(train_aug_dataset, split["train"])
    val_subset = Subset(train_clean_dataset, split["val"])
    return train_subset, val_subset
