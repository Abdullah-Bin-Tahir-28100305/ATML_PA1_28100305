# =============================================================================
# task4/data/cifar10.py
# -----------------------------------------------------------------------------
# PURPOSE: everything about the KNOWN data (CIFAR-10) — loading, the train vs
# eval transforms, and the optional RandAugment insertion for GCSC.
#
# Spec transforms: "random crop to 32x32 with 4-pixel padding and random
# horizontal flipping" for training; deterministic (no aug) for eval. GCSC adds
# RandAugment(num_ops=2, magnitude=9) AFTER crop+flip and BEFORE normalization.
#
# ML CONCEPT — WHY A SEPARATE, UN-AUGMENTED VIEW:
#   The Mahalanobis score estimates class means/covariance from UNAUGMENTED
#   CIFAR-10 TRAIN features (spec), so we expose an eval-transform training view
#   for that. Keeping data construction separate from evaluation guarantees every
#   score sees identical examples.
#
# LINKS: make_splits.py builds the 90/10 split on top of this; extract_outputs.py
# runs models over these datasets; the transforms here are the single source of
# truth for augmentation.
# =============================================================================

from torchvision import transforms
from torchvision.datasets import CIFAR10


def build_transforms(cfg, train: bool, randaugment: bool = False):
    """Return the CIFAR-10 transform pipeline.

    train=True  -> random crop (4px pad) + horizontal flip (+ optional RandAugment)
    train=False -> no augmentation (deterministic), just ToTensor + Normalize
    randaugment -> only for GCSC's training transform (num_ops/magnitude from cfg)
    """
    size = cfg["image"]["size"]
    pad = cfg["image"]["crop_padding"]
    mean = cfg["image"]["norm_mean"]
    std = cfg["image"]["norm_std"]

    if train:
        ops = [
            transforms.RandomCrop(size, padding=pad),   # random 32x32 crop, 4px pad
            transforms.RandomHorizontalFlip(),          # 50% flip
        ]
        if randaugment:
            # GCSC: stronger positive augmentation, inserted AFTER crop+flip and
            # BEFORE ToTensor/Normalize (spec). RandAugment operates on PIL images.
            ra = cfg["method"]["randaugment"]
            ops.append(transforms.RandAugment(num_ops=ra["num_ops"],
                                              magnitude=ra["magnitude"]))
        ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(ops)
    else:
        # Deterministic eval transform (also used for Mahalanobis train features).
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def load_cifar10_train(cfg, train_transform: bool, randaugment: bool = False):
    """Official CIFAR-10 TRAIN partition with the chosen transform.

    train_transform=True  -> augmented view (for training)
    train_transform=False -> clean view (for Mahalanobis stats / feature caching)
    """
    tfm = build_transforms(cfg, train=train_transform, randaugment=randaugment)
    return CIFAR10(cfg["known"]["root"], train=True, download=True, transform=tfm)


def load_cifar10_test(cfg):
    """Official CIFAR-10 TEST set with the clean eval transform (known-class eval)."""
    tfm = build_transforms(cfg, train=False)
    return CIFAR10(cfg["known"]["root"], train=False, download=True, transform=tfm)


# CIFAR-10 class names, in torchvision's label order, for the failure analysis.
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
