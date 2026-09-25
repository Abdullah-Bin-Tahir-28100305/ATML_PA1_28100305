# =============================================================================
# shared/pacs.py
# -----------------------------------------------------------------------------
# PURPOSE: everything about LOADING the PACS dataset — locating the image files,
# building the train/eval image transforms, and exposing a simple per-domain
# Dataset. This lives in shared/ because BOTH Task 2 and Task 3 use the exact
# same PACS protocol (spec: "Use one shared PACS protocol for Tasks 2 and 3").
#
# PACS layout on disk (the standard public layout):
#   root/PACS/
#     photo/<class>/*.jpg
#     art_painting/<class>/*.jpg
#     cartoon/<class>/*.jpg
#     sketch/<class>/*.jpg
# Each domain folder contains 7 class subfolders (dog, elephant, giraffe,
# guitar, horse, house, person). torchvision's ImageFolder reads exactly this
# layout, giving us (image, class_index) pairs with a consistent class ordering.
#
# LINKS TO OTHER FILES:
#   - pacs_protocol.py builds splits and domain-balanced loaders on top of this.
#   - task2/train.py consumes those loaders.
# =============================================================================

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Canonical class order for PACS. We FIX it here (rather than trusting the
# filesystem's alphabetical order) so class index i means the same class in every
# domain and across both tasks. ImageFolder sorts alphabetically, which happens
# to match this list, but we assert it below to be safe.
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

# The four PACS domains. Folder names use 'art_painting' (underscore).
PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]


def build_transforms(cfg, train: bool):
    """Return the torchvision transform pipeline for train vs eval.

    ML CONCEPT — TRAIN vs EVAL AUGMENTATION:
      During TRAINING we add mild augmentation (random crop + horizontal flip)
      to reduce overfitting and expose the network to small appearance changes.
      During EVALUATION we use a deterministic center crop so the measurement is
      stable and reproducible. Spec: "random 224x224 crop with horizontal
      flipping during training; 224x224 center crop for validation and
      evaluation." Both then apply the SAME ImageNet normalization the pretrained
      ResNet-18 expects.
    """
    resize = cfg["image"]["resize"]         # 256
    crop = cfg["image"]["crop"]             # 224
    mean = cfg["image"]["norm_mean"]
    std = cfg["image"]["norm_std"]

    if train:
        return transforms.Compose([
            transforms.Resize((resize, resize)),      # 256x256
            transforms.RandomCrop(crop),              # random 224x224 window
            transforms.RandomHorizontalFlip(),        # 50% left-right flip
            transforms.ToTensor(),                    # -> CHW float in [0,1]
            transforms.Normalize(mean, std),          # ImageNet normalization
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize, resize)),      # 256x256
            transforms.CenterCrop(crop),              # deterministic 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def load_domain_dataset(cfg, domain: str, train: bool):
    """Load ONE PACS domain as an ImageFolder with the appropriate transform.

    Returns an ImageFolder whose samples are (path, class_index). We verify the
    class ordering matches PACS_CLASSES so labels are consistent everywhere.
    """
    root = cfg["dataset"]["root"]
    domain_dir = os.path.join(root, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(
            f"PACS domain folder not found: {domain_dir}\n"
            f"Expected layout: {root}/<domain>/<class>/*.jpg  (see README).")
    tfm = build_transforms(cfg, train=train)
    ds = ImageFolder(domain_dir, transform=tfm)
    # Safety check: ImageFolder's discovered classes must match our fixed order,
    # otherwise label i would mean different things in different domains.
    assert ds.classes == PACS_CLASSES, (
        f"Class order mismatch in {domain_dir}:\n"
        f"  found:    {ds.classes}\n  expected: {PACS_CLASSES}")
    return ds
