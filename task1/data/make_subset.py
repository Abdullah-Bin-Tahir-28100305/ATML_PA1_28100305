# =============================================================================
# data/make_subset.py
# -----------------------------------------------------------------------------
# PURPOSE: build the three fixed data collections every experiment reuses:
#   (1) a STRATIFIED 80/20 train/val split of the official TRAIN partition
#       -> used to fit and early-stop each linear classifier head;
#   (2) a CLASS-BALANCED 500-image subset of the official TEST partition
#       -> the "evaluation subset" on which all interventions are measured;
#   (3) the raw dataset objects and class names shared across the project.
#
# WHY THIS FILE EXISTS SEPARATELY (repo-structure rationale):
#   The assignment says "Keep transformation generation separate from
#   evaluation so the exact same images can be reused across models." This file
#   is the *selection* stage. It fixes WHICH images every model sees, and saves
#   their identifiers, so ResNet, ViT and CLIP are compared on identical data.
#
# LINKS TO OTHER FILES:
#   - utils.set_seed / load_config: reproducibility + configuration.
#   - Consumed by models/backbones.py (to train heads on the split) and by
#     scripts/run_task1.py (to build the eval subset that feeds every
#     intervention in data/transforms.py and data/make_cue_conflicts.py).
# =============================================================================

import os
import json                       # to save selected image identifiers
import numpy as np
import torch
from torchvision import datasets, transforms   # dataset loaders + PIL<->tensor
from torch.utils.data import Subset            # a view over a subset of indices

# Allow running this file directly OR importing it as data.make_subset.
try:
    from utils import load_config, set_seed, ensure_dir
except ImportError:  # when imported as a package from the project root
    from ..utils import load_config, set_seed, ensure_dir


# STL-10's ten class names, in the label-index order torchvision uses. We hard
# code them because CLIP zero-shot needs human-readable class names to build
# text prompts ("a photo of a {class}."). Order matters: index i must be the
# name of label i.
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def _to_224_tensor(image_size: int):
    """Return a torchvision transform: PIL image -> float tensor in [0,1], 224x224.

    ML CONCEPT — A COMMON CANVAS:
      The spec requires interventions on "a common 224x224 RGB image before
      applying each model's required normalization." So here we ONLY resize to
      224x224 and convert to a tensor in [0,1]. We deliberately DO NOT normalise
      (no mean/std subtraction) — that per-model step happens later inside each
      backbone wrapper. This keeps a single, model-agnostic pixel space in which
      all transformations are defined.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # force exact 224x224
        transforms.ToTensor(),                        # HWC uint8 -> CHW float[0,1]
    ])


def load_datasets(cfg: dict):
    """Download (if needed) and return the official TRAIN and TEST datasets.

    Returns (train_ds, test_ds, class_names). Each item yields (image_tensor,
    label) where image_tensor is a 3x224x224 float tensor in [0,1].
    """
    name = cfg["dataset"]["name"]
    root = cfg["dataset"]["root"]
    tfm = _to_224_tensor(cfg["image_size"])

    if name == "stl10":
        # split='train' / 'test' select the OFFICIAL partitions. We fit heads on
        # 'train' and evaluate interventions on a subset of 'test' — never mixing
        # them, which would leak test information into training (a cardinal sin).
        train_ds = datasets.STL10(root, split="train", download=True, transform=tfm)
        test_ds  = datasets.STL10(root, split="test",  download=True, transform=tfm)
        class_names = STL10_CLASSES
    elif name == "oxford_pets":
        # Oxford-IIIT Pets support (the spec's alternative dataset). Its 37
        # breed names are provided by torchvision as .classes.
        train_ds = datasets.OxfordIIITPet(root, split="trainval", download=True, transform=tfm)
        test_ds  = datasets.OxfordIIITPet(root, split="test",     download=True, transform=tfm)
        class_names = [c.lower() for c in train_ds.classes]
    else:
        raise ValueError(f"Unknown dataset '{name}' (use stl10 or oxford_pets)")

    return train_ds, test_ds, class_names


def _labels_of(ds) -> np.ndarray:
    """Extract the integer label array of a dataset WITHOUT decoding images.

    We need labels to stratify. Reading them via the dataset's internal arrays
    is far cheaper than iterating and decoding every image, which would be slow.
    Different torchvision datasets store labels under different attributes, so we
    try the common ones in turn.
    """
    for attr in ("labels", "_labels"):          # STL10 -> .labels, Pets -> ._labels
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    # Fallback: iterate the (label side of the) dataset. Slower but universal.
    return np.asarray([y for _, y in ds])


def stratified_train_val_split(train_ds, cfg: dict):
    """Split the official TRAIN set into 80% train / 20% val, PRESERVING class
    proportions (stratification), using seed 6304.

    ML CONCEPT — STRATIFIED SAMPLING:
      A plain random split can, by chance, put too few examples of some class
      into validation, making the early-stopping signal noisy or biased. By
      splitting *within each class* and taking 80/20 of each, both halves have
      the same class distribution as the whole — a faithful miniature. This is
      the standard practice when the split drives model selection.
    """
    set_seed(cfg["seed"])                        # determinism for the split
    labels = _labels_of(train_ds)                # per-sample class labels
    train_frac = cfg["split"]["train_frac"]      # 0.8
    rng = np.random.default_rng(cfg["seed"])     # a *local* generator, seeded

    train_idx, val_idx = [], []
    for c in np.unique(labels):                  # loop over each class id
        idx_c = np.where(labels == c)[0]         # indices of that class
        rng.shuffle(idx_c)                       # shuffle within the class
        n_train = int(round(train_frac * len(idx_c)))  # 80% of this class
        train_idx.extend(idx_c[:n_train].tolist())     # first 80% -> train
        val_idx.extend(idx_c[n_train:].tolist())       # last 20%  -> val

    # Subset is a lightweight view: it does not copy image data, it just remaps
    # indices. train_view[i] returns train_ds[train_idx[i]].
    return Subset(train_ds, train_idx), Subset(train_ds, val_idx)


def build_eval_subset(test_ds, cfg: dict):
    """Select a class-balanced subset of the official TEST set (default 500).

    Implements the spec: "Select a class-balanced subset of 500 official test
    images using seed 6304. If a class has insufficient examples, use all
    available examples and document the imbalance. Save the selected image
    identifiers."

    Returns (subset, selected_indices, imbalance_report).
    """
    set_seed(cfg["seed"])
    labels = _labels_of(test_ds)
    total = cfg["eval_subset"]["size"]           # 500
    classes = np.unique(labels)
    per_class = total // len(classes)            # e.g. 500 / 10 = 50 per class
    rng = np.random.default_rng(cfg["seed"])

    selected, report = [], {}
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        take = min(per_class, len(idx_c))        # cap at what's available
        selected.extend(idx_c[:take].tolist())
        # DOCUMENT THE IMBALANCE: if a class had fewer than per_class images we
        # record the shortfall so it can be reported honestly (spec requirement).
        report[int(c)] = {"available": int(len(idx_c)),
                          "requested": int(per_class),
                          "taken": int(take)}

    selected = sorted(selected)                  # stable, deterministic order
    return Subset(test_ds, selected), selected, report


def save_identifiers(selected_indices, report, class_names, cfg: dict):
    """Persist WHICH test images were chosen, so every model reuses the same set.

    Saving identifiers (not the pixels) is the reproducibility contract: the
    exact evaluation set can be reconstructed byte-for-byte from these indices.
    """
    out_dir = ensure_dir(cfg["paths"]["results_dir"])
    payload = {
        "seed": cfg["seed"],
        "dataset": cfg["dataset"]["name"],
        "num_selected": len(selected_indices),
        "selected_test_indices": selected_indices,   # the identifiers
        "class_names": class_names,
        "per_class_imbalance_report": report,        # documents any shortfall
    }
    path = os.path.join(out_dir, "eval_subset_identifiers.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# Running this file on its own is a quick sanity check: it downloads the data,
# builds every collection, prints the sizes, and writes the identifiers file.
if __name__ == "__main__":
    cfg = load_config()
    train_ds, test_ds, class_names = load_datasets(cfg)
    tr, va = stratified_train_val_split(train_ds, cfg)
    ev, sel, rep = build_eval_subset(test_ds, cfg)
    path = save_identifiers(sel, rep, class_names, cfg)
    print(f"train={len(tr)}  val={len(va)}  eval_subset={len(ev)}")
    print(f"classes={class_names}")
    print(f"identifiers saved -> {path}")
