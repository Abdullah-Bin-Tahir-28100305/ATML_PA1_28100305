# =============================================================================
# task4/data/make_splits.py
# -----------------------------------------------------------------------------
# PURPOSE: build the stratified 90/10 train/validation split of the OFFICIAL
# CIFAR-10 TRAIN partition (seed 6304), persist the indices, and expose Subset
# views. Spec: "Create a stratified 90/10 split of the official training
# partition using seed 6304, use only the training portion for optimization, and
# select checkpoints by CIFAR-10 validation accuracy."
#
# ML CONCEPT — WHY A HELD-OUT VALIDATION SPLIT IN OSR:
#   The validation split does double duty: (1) closed-set checkpoint selection by
#   accuracy, and (2) CALIBRATING THE REJECTION THRESHOLD — the 95th percentile
#   of the unknownness score on this KNOWN validation set. Both use known data
#   only, so no unknown ever influences training or threshold choice.
#   Stratification preserves class proportions so the val accuracy and the
#   threshold percentile are stable, faithful estimates.
#
# LINKS: cifar10.py provides the base dataset; extract_outputs.py uses these
# splits for training (train), checkpoint selection (val), and threshold
# calibration (val features).
# =============================================================================

import os
import json
import numpy as np
from torch.utils.data import Subset

try:
    from utils import ensure_dir
except ImportError:
    from ..utils import ensure_dir


def _stratified_indices(labels, train_frac, seed):
    """Return (train_idx, val_idx) preserving class proportions (stratified)."""
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
    """Create the 90/10 split once and cache indices to JSON; return the dict.

    base_train_dataset: any CIFAR10 train dataset (we only read its .targets).
    Saving indices (not pixels) makes the split reproducible byte-for-byte.
    """
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
    """Return (train_subset[aug], val_subset[clean]) using the saved indices.

    The TRAIN subset uses the AUGMENTED dataset (crop/flip/RandAugment) for
    optimization; the VAL subset uses the CLEAN dataset for deterministic
    accuracy + threshold calibration.
    """
    train_subset = Subset(train_aug_dataset, split["train"])
    val_subset = Subset(train_clean_dataset, split["val"])
    return train_subset, val_subset
