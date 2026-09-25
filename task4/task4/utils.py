

import os
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
   
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
