

import os              # filesystem paths, directory creation
import random          # Python's built-in RNG (used by some torchvision ops)
import numpy as np     # NumPy RNG (used by sklearn, our subset sampling, etc.)
import torch           # PyTorch RNG (weight init, dataloader shuffling, CUDA)
import yaml            # to parse configs/config.yaml


def set_seed(seed: int) -> None:
    
    os.environ["PYTHONHASHSEED"] = str(seed)  # stabilise dict/set hash order
    random.seed(seed)                         # seed Python's global RNG
    np.random.seed(seed)                      # seed NumPy's global RNG
    torch.manual_seed(seed)                   # seed PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)          # seed all GPU RNGs (if any)
    # The two flags below trade a little speed for exact reproducibility:
    # cuDNN can pick nondeterministic algorithms otherwise.
    torch.backends.cudnn.deterministic = True # force deterministic kernels
    torch.backends.cudnn.benchmark = False    # disable autotuner (nondeterm.)


def load_config(path: str = None) -> dict:
    
    if path is None:
        
        here = os.path.dirname(os.path.abspath(__file__))  # .../task1
        path = os.path.join(here, "configs", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)  # safe_load avoids executing arbitrary YAML


def get_device() -> torch.device:
   
    if torch.cuda.is_available():                      # NVIDIA GPU (e.g. Colab)
        return torch.device("cuda")
    # hasattr guard: older torch builds may not expose the `mps` submodule.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")                     # Apple Silicon GPU (M1..)
    return torch.device("cpu")                          # universal fallback


def ensure_dir(path: str) -> str:
   
    os.makedirs(path, exist_ok=True)
    return path
