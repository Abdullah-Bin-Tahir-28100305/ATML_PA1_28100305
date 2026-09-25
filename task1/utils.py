# =============================================================================
# utils.py
# -----------------------------------------------------------------------------
# SHARED HELPERS used by every other file in the project. Centralising these
# means the *reproducibility machinery* (seeding, config loading, device
# selection) lives in exactly one place, so all scripts behave identically.
#
# LINKS TO OTHER FILES:
#   - Imported by data/make_subset.py, data/transforms.py,
#     data/make_cue_conflicts.py, models/backbones.py, analysis/*.py,
#     scripts/run_task1.py.
#   - `set_seed` is the concrete implementation of the assignment's
#     "Fix random seeds where practical" instruction.
# =============================================================================

import os              # filesystem paths, directory creation
import random          # Python's built-in RNG (used by some torchvision ops)
import numpy as np     # NumPy RNG (used by sklearn, our subset sampling, etc.)
import torch           # PyTorch RNG (weight init, dataloader shuffling, CUDA)
import yaml            # to parse configs/config.yaml


def set_seed(seed: int) -> None:
    """Make the ENTIRE pipeline deterministic given a single integer seed.

    ML CONCEPT — CONTROLLING PSEUDO-RANDOMNESS:
      Neural-network pipelines pull "random" numbers from several independent
      generators. If we seed only one of them, the others still vary run to
      run and results become irreproducible. We therefore seed ALL of them:
        * Python's `random`  -> some torchvision transforms use it
        * NumPy's RNG        -> sklearn splits, our subset sampling
        * PyTorch CPU RNG    -> linear-head weight initialisation, shuffling
        * PyTorch CUDA RNG   -> the same, but on GPU
      We also set PYTHONHASHSEED so hash-based ordering is stable.
    """
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
    """Read configs/config.yaml into a plain Python dict.

    Every script calls this so they all read the SAME knobs. This is what makes
    the config file the single 'recorded configuration' the assignment asks for.
    """
    if path is None:
        # Default: configs/config.yaml relative to the project root. We compute
        # the project root from THIS file's location so the path works no
        # matter which directory you launch the script from.
        here = os.path.dirname(os.path.abspath(__file__))  # .../task1
        path = os.path.join(here, "configs", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)  # safe_load avoids executing arbitrary YAML


def get_device() -> torch.device:
    """Return the best available compute device: CUDA, then Apple MPS, then CPU.

    All heavy tensor work (feature extraction, style transfer) runs on this
    device. Using a helper means every file agrees on where computation happens.

    DEVICE BACKENDS EXPLAINED:
      * CUDA = NVIDIA GPUs (e.g. Google Colab's free GPU, lab workstations).
      * MPS  = 'Metal Performance Shaders', Apple's GPU backend for Apple
               Silicon Macs (M1/M2/M3/M4). This is the Mac equivalent of CUDA;
               there is no CUDA on a Mac because CUDA is NVIDIA-only.
      * CPU  = universal fallback; correct everywhere, just slower.
    We prefer CUDA when present (fastest, most mature), then MPS on an Apple
    Silicon Mac, then CPU. The rest of the code never needs to know which one it
    got — it just calls .to(device).
    """
    if torch.cuda.is_available():                      # NVIDIA GPU (e.g. Colab)
        return torch.device("cuda")
    # hasattr guard: older torch builds may not expose the `mps` submodule.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")                     # Apple Silicon GPU (M1..)
    return torch.device("cpu")                          # universal fallback


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not exist; return the path.

    Used before writing any results/plots so scripts never crash on a missing
    folder. `exist_ok=True` makes the call idempotent (safe to run repeatedly).
    """
    os.makedirs(path, exist_ok=True)
    return path
