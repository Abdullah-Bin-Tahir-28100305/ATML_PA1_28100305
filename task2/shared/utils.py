# =============================================================================
# shared/utils.py
# -----------------------------------------------------------------------------
# SHARED HELPERS for Task 2 (and Task 3): config loading with inheritance,
# global seeding, device selection, directory creation. Centralised so every
# script behaves identically and reproducibly.
# =============================================================================

import os
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed every RNG so the whole pipeline is deterministic given one integer.

    ML CONCEPT — REPRODUCIBILITY: neural pipelines draw randomness from Python,
    NumPy, and PyTorch (CPU + CUDA). Seeding all of them — plus forcing
    deterministic cuDNN kernels — makes runs repeatable, which the assignment
    requires ("Use seed 6304 for every comparison").
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    """Load a YAML config, resolving a single-level `inherit:` to a base file.

    Our method configs start with `inherit: base.yaml`. We load the base first,
    then deep-merge the method file on top, so shared settings live once in
    base.yaml and each method overrides only what differs. This is the mechanism
    that keeps the comparison controlled.
    """
    path = os.path.abspath(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "inherit" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("inherit"))
        with open(base_path, "r") as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)                   # method overrides base
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins on conflicts)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)            # merge nested dicts
        else:
            out[k] = v                                 # override scalar/list
    return out


def get_device() -> torch.device:
    """Best available device: CUDA (NVIDIA / Colab) -> MPS (Apple M-series) -> CPU.

    CUDA = NVIDIA GPUs (e.g. Colab). MPS = Apple Silicon GPU backend (the Mac
    equivalent of CUDA; there is no CUDA on a Mac). CPU is the universal
    fallback. The rest of the code just calls .to(device) and never cares which.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if missing; return it. Idempotent."""
    os.makedirs(path, exist_ok=True)
    return path
