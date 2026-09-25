# =============================================================================
# shared/pacs_protocol.py
# -----------------------------------------------------------------------------
# PURPOSE: the reusable PACS *protocol* shared by Tasks 2 and 3:
#   (1) build a stratified 80/20 train/val split PER SOURCE DOMAIN (seed 6304)
#       and persist it to JSON so Tasks 2 and 3 use identical splits;
#   (2) provide subset Datasets for those splits and for the target domain;
#   (3) provide the DOMAIN-BALANCED batch iterator used during adaptation
#       (8 images from each source domain + 24 target images per update),
#       cycling shorter loaders so every update is full and balanced.
#
# WHY THIS MATTERS (spec):
#   - "Reuse the same source splits across both tasks." -> we save/load one JSON.
#   - "Each adaptation update should contain eight examples from each source
#      domain and 24 target examples ... cycle a loader when necessary." -> the
#      MultiDomainBatchSampler below implements exactly this.
#
# LINKS:
#   - pacs.py: loads the raw per-domain ImageFolders.
#   - task2/train.py: consumes make_source_loaders / make_target_loader and the
#     domain-balanced iterator.
# =============================================================================

import os
import json
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

try:
    from shared.pacs import load_domain_dataset, PACS_CLASSES
except ImportError:
    from pacs import load_domain_dataset, PACS_CLASSES


# -----------------------------------------------------------------------------
# 1. STRATIFIED SPLIT GENERATION + PERSISTENCE
# -----------------------------------------------------------------------------
def _stratified_indices(labels, train_frac, seed):
    """Return (train_idx, val_idx) preserving class proportions (stratified).

    ML CONCEPT — STRATIFIED SPLIT: split within each class so both train and val
    keep the same class distribution. This makes the mean source-validation
    macro-F1 (our checkpoint-selection signal) a faithful, low-variance estimate.
    """
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


def build_or_load_splits(cfg):
    """Create the per-source-domain 80/20 splits once and cache them to JSON.

    The JSON stores, for each source domain, the train and val IMAGE INDICES
    (into that domain's ImageFolder). Saving indices — not pixels — makes the
    split perfectly reproducible and shareable between Task 2 and Task 3.
    Returns the loaded splits dict.
    """
    split_path = cfg["paths"]["shared_splits"]
    # If the split file already exists, reuse it verbatim (this is what lets
    # Task 3 reuse Task 2's splits unchanged).
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    splits = {"seed": cfg["seed"], "train_frac": cfg["split"]["train_frac"],
              "domains": {}}
    for domain in cfg["dataset"]["source_domains"]:
        # We load with train=False transform here only to READ labels cheaply;
        # the transform does not affect the label list or ordering.
        ds = load_domain_dataset(cfg, domain, train=False)
        labels = [y for _, y in ds.samples]           # class index per image
        tr, va = _stratified_indices(labels, cfg["split"]["train_frac"], cfg["seed"])
        splits["domains"][domain] = {"train": tr, "val": va,
                                     "num_images": len(labels)}
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


# -----------------------------------------------------------------------------
# 2. SUBSET DATASETS FOR EACH SPLIT
# -----------------------------------------------------------------------------
def make_source_datasets(cfg, splits, train: bool):
    """Return dicts of per-domain train and val Subset datasets.

    `train=True` applies training augmentation (used for the train subsets);
    `train=False` applies eval transforms (used for the val subsets and for any
    evaluation pass). We build both a train-augmented and eval-clean view.
    """
    train_sets, val_sets = {}, {}
    for domain in cfg["dataset"]["source_domains"]:
        idx = splits["domains"][domain]
        # Train subset: augmented transform (train=True).
        ds_train = load_domain_dataset(cfg, domain, train=True)
        train_sets[domain] = Subset(ds_train, idx["train"])
        # Val subset: clean/eval transform (train=False) for stable measurement.
        ds_eval = load_domain_dataset(cfg, domain, train=False)
        val_sets[domain] = Subset(ds_eval, idx["val"])
    return train_sets, val_sets


def make_target_dataset(cfg, train: bool):
    """Return the FULL target-domain dataset.

    In transductive UDA the *entire* target domain is available UNLABELED during
    adaptation (spec). We still keep its labels in the object, but the training
    code must never read target labels — only at the very end for evaluation.
    `train=True` gives the augmented view used as the unlabeled adaptation set;
    `train=False` gives the clean view used for final labeled evaluation.
    """
    return load_domain_dataset(cfg, cfg["dataset"]["target_domain"], train=train)


# -----------------------------------------------------------------------------
# 3. DOMAIN-BALANCED BATCH ITERATOR
# -----------------------------------------------------------------------------
class InfiniteLoader:
    """Wrap a DataLoader so it yields batches forever, restarting when exhausted.

    Spec: "cycle a loader when necessary." Different domains have different
    sizes, so to guarantee every adaptation update has 8 images from EACH source
    and 24 target images, we let each loader restart independently.
    """
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)               # restart this loader
            return next(self.it)


def make_domain_balanced_iterators(cfg, train_sets, target_set):
    """Build one InfiniteLoader per source domain + one for the target.

    Each source loader yields `per_source_batch` (=8) images; the target loader
    yields `target_batch` (=24). The training loop pulls one batch from each per
    update, giving 3x8=24 source images and 24 target images — equal source and
    target totals (spec).
    """
    per_source = cfg["train"]["per_source_batch"]     # 8
    target_bs = cfg["train"]["target_batch"]          # 24

    source_iters = {}
    for domain, ds in train_sets.items():
        loader = DataLoader(ds, batch_size=per_source, shuffle=True,
                            drop_last=True, num_workers=2)
        source_iters[domain] = InfiniteLoader(loader)

    target_loader = DataLoader(target_set, batch_size=target_bs, shuffle=True,
                               drop_last=True, num_workers=2)
    target_iter = InfiniteLoader(target_loader)
    return source_iters, target_iter


def steps_per_epoch(cfg, train_sets):
    """Define an 'epoch' as one pass over the LARGEST source train split.

    Because loaders cycle, 'epoch' is a bookkeeping unit for the schedule and
    early stopping. Using the largest source ensures every source is seen at
    least once per epoch on average.
    """
    per_source = cfg["train"]["per_source_batch"]
    max_len = max(len(ds) for ds in train_sets.values())
    return max(1, max_len // per_source)
