# =============================================================================
# task4/data/cifar100_unknowns.py
# -----------------------------------------------------------------------------
# PURPOSE: build the NEAR and FAR unknown evaluation sets from the CIFAR-100 TEST
# partition, restricted to the fixed fine-class lists in the config.
#
# CRITICAL RULES (spec — enforced by construction here):
#   * Unknowns come ONLY from CIFAR-100 TEST (never train): CIFAR-100 train images
#     may not be used for anything.
#   * The near/far grouping is FIXED and may not be revised after seeing results.
#   * These sets are EVALUATION-ONLY: they never touch training, checkpoint
#     selection, score design, or threshold selection.
#
# ML CONCEPT — NEAR vs FAR SEMANTIC DIFFICULTY:
#   NEAR unknowns (bus, wolf, leopard, ...) are visually/semantically close to
#   CIFAR-10 knowns (truck, dog, cat, ...), so a closed-set classifier is more
#   likely to confidently absorb them into a wrong known class -> HARD to reject.
#   FAR unknowns (bottle, chair, keyboard, ...) share little with the known label
#   space -> EASIER to reject. Reporting near and far separately reveals how
#   rejection degrades with semantic similarity.
#
# LINKS: uses cifar10.build_transforms for the SAME clean eval transform as the
# known test set (so every score sees identically-preprocessed inputs).
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

try:
    from data.cifar10 import build_transforms
except ImportError:
    from .cifar10 import build_transforms


def _select_class_subset(dataset, class_names, cap, seed):
    """Return a Subset of `dataset` limited to `class_names`, <= `cap` per group.

    We map each requested fine-class NAME to its CIFAR-100 label index via the
    dataset's own .classes list (authoritative ordering), then gather test images
    of those classes. If the total exceeds `cap`, we take a deterministic
    (seeded) subset so the evaluation set is fixed and reproducible.
    """
    name_to_idx = {name: i for i, name in enumerate(dataset.classes)}
    # Validate every requested name exists (guards against typos / naming drift).
    missing = [n for n in class_names if n not in name_to_idx]
    if missing:
        raise ValueError(f"CIFAR-100 fine classes not found: {missing}\n"
                         f"available example names: {dataset.classes[:10]} ...")
    wanted = {name_to_idx[n] for n in class_names}
    targets = np.asarray(dataset.targets)
    idx = np.where(np.isin(targets, list(wanted)))[0]

    # Deterministic cap to `cap` images for the group (spec: 800 per group).
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    idx = np.sort(idx[:cap])
    return Subset(dataset, idx.tolist())


def load_unknown_groups(cfg):
    """Return (near_subset, far_subset) CIFAR-100-test unknown evaluation sets.

    Both use the SAME clean eval transform as the CIFAR-10 test set so knowns and
    unknowns are preprocessed identically.
    """
    tfm = build_transforms(cfg, train=False)            # clean eval transform
    ds = CIFAR100(cfg["known"]["root"], train=False, download=True, transform=tfm)
    cap = cfg["unknowns"]["images_per_group"]           # 800
    seed = cfg["seed"]
    near = _select_class_subset(ds, cfg["unknowns"]["near"], cap, seed)
    far = _select_class_subset(ds, cfg["unknowns"]["far"], cap, seed)
    return near, far


def unknown_class_names(dataset_subset):
    """Return the CIFAR-100 fine-class name for each example in a Subset.

    Used by the failure analysis to report WHICH unknown class was wrongly
    accepted (e.g. 'wolf' accepted as 'dog').
    """
    base = dataset_subset.dataset                       # underlying CIFAR100
    names = base.classes
    return [names[base.targets[i]] for i in dataset_subset.indices]
