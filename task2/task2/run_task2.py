# =============================================================================
# task2/run_task2.py
# -----------------------------------------------------------------------------
# TOP-LEVEL ORCHESTRATOR for Task 2. Runs the whole pipeline end to end, in the
# order the assignment prescribes:
#   1. Train Source-only ERM   (also the Task 3 ERM baseline)
#   2. Train DAN  (MMD)
#   3. Train DANN (adversarial)
#   4. Train CDAN (conditional adversarial)
#   5. Final common evaluation + alignment diagnostic (source/target/separability)
#   6. Controlled alignment-strength study
# It also plots the classification/alignment loss curves required as evidence.
#
# HOW TO RUN:
#   cd task2
#   python run_task2.py               # full pipeline (GPU strongly recommended)
#   python run_task2.py --skip_train  # only re-run evaluation on saved checkpoints
#
# WHY AN ORCHESTRATOR: it guarantees the exact required order (all training and
# checkpoint freezing BEFORE any target-label evaluation), which is a hard rule
# of the assignment ("Freeze the complete experimental decision before final
# target evaluation").
# =============================================================================

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import json
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from shared.utils import load_config, ensure_dir
from train import train_one_config
from evaluate_final import evaluate_all
from controlled_study import run_controlled_study

CONFIG_DIR = os.path.join(THIS_DIR, "configs")
METHOD_CONFIGS = {
    "source_only": os.path.join(CONFIG_DIR, "source_only.yaml"),
    "dan":         os.path.join(CONFIG_DIR, "dan.yaml"),
    "dann":        os.path.join(CONFIG_DIR, "dann.yaml"),
    "cdan":        os.path.join(CONFIG_DIR, "cdan.yaml"),
}


def plot_loss_curves(results_dir):
    """Plot classification vs alignment loss over steps for each method.

    Spec Required Evidence: "Classification and alignment or domain-loss curves
    sufficient to assess whether each adaptation method trained as intended."
    We read each method's history_<name>.json (written by train.py) and plot the
    running per-step loss components.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for name in METHOD_CONFIGS:
        hist_path = os.path.join(results_dir, f"history_{name}.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        logs = hist.get("step_logs", [])
        if not logs:
            continue
        steps = [l["step"] for l in logs]
        cls = [l.get("loss_cls", np.nan) for l in logs]
        align = [l.get("loss_align", np.nan) for l in logs]

        plt.figure(figsize=(7, 5))
        plt.plot(steps, cls, label="classification loss", alpha=0.8)
        plt.plot(steps, align, label="alignment / domain loss", alpha=0.8)
        plt.xlabel("training step"); plt.ylabel("loss")
        plt.title(f"{name}: loss curves")
        plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"loss_curves_{name}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true",
                        help="reuse existing checkpoints; only evaluate + study")
    parser.add_argument("--skip_study", action="store_true",
                        help="skip the controlled alignment-strength study")
    args = parser.parse_args()

    base_cfg = load_config(METHOD_CONFIGS["source_only"])
    results_dir = ensure_dir(base_cfg["paths"]["results_dir"])

    # ---- Steps 1-4: train each method through the SAME pipeline ----
    if not args.skip_train:
        for name, cfg_path in METHOD_CONFIGS.items():
            print(f"\n########## TRAIN: {name} ##########")
            cfg = load_config(cfg_path)
            train_one_config(cfg)

    # ---- loss curves (required evidence) ----
    plot_loss_curves(results_dir)

    # ---- Step 5: final common evaluation + diagnostic ----
    print("\n########## FINAL EVALUATION ##########")
    evaluate_all(base_cfg)

    # ---- Step 6: controlled alignment-strength study (DAN lambda sweep) ----
    if not args.skip_study:
        print("\n########## CONTROLLED STUDY ##########")
        run_controlled_study(METHOD_CONFIGS["dan"])

    print("\n[all done] see the results/ folder for tables, curves, and JSON.")


if __name__ == "__main__":
    main()
