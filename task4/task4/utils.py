# =============================================================================
# task4/utils.py
# -----------------------------------------------------------------------------
# SHARED HELPERS for Task 4 (Open-Set Recognition): config loading with a single
# `inherit:` level, global seeding, device selection, directory creation.
# Centralising these keeps the whole pipeline reproducible and consistent.
# =============================================================================

import os
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch (CPU+CUDA) so the pipeline is deterministic.

    ML CONCEPT — REPRODUCIBILITY: the assignment mandates seed 6304 for the
    90/10 split, all training, and checkpoint selection. Seeding every RNG (and
    forcing deterministic cuDNN) makes runs repeatable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    """Load a YAML config, resolving a one-level `inherit:` to a base file.

    Method configs start with `inherit: base.yaml`; we load base first then
    deep-merge the method file on top, so shared settings live once in base.yaml.
    """
    path = os.path.abspath(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "inherit" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("inherit"))
        with open(base_path, "r") as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_device() -> torch.device:
    """Best available device: CUDA (NVIDIA/Colab) -> MPS (Apple M-series) -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
