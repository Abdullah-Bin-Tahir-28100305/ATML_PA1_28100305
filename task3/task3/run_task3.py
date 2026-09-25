# =============================================================================
# task3/run_task3.py
# -----------------------------------------------------------------------------
# TOP-LEVEL ORCHESTRATOR for Task 3, in the order the assignment prescribes:
#   1. ERM baseline      (reuse Task 2's Source-only checkpoint if provided,
#                         otherwise train an equivalent ERM under the same config)
#   2. DAN-DG            (pairwise source MMD)
#   3. SAM               (sharpness-aware minimization)
#   4. Final evaluation + diagnostics (Sketch loaded only here) + per-class study
#   5. Controlled study  (DAN-DG lambda sweep by default)
# Also plots the classification / MMD-penalty training curves (required evidence).
#
# HOW TO RUN (from inside task3/):
#   python run_task3.py                                  # train all + evaluate + study
#   python run_task3.py --erm_checkpoint /path/source_only.pt   # reuse Task 2 ERM
#   python run_task3.py --skip_train                     # only evaluate saved ckpts
#
# The strict ordering (all training + selection frozen BEFORE Sketch is touched)
# is enforced by evaluating Sketch only in step 4.
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
from evaluate_sketch import evaluate_all
from controlled_study import run_controlled_study

CONFIG_DIR = os.path.join(THIS_DIR, "configs")
METHOD_CONFIGS = {
    "erm":    os.path.join(CONFIG_DIR, "erm.yaml"),
    "dan_dg": os.path.join(CONFIG_DIR, "dan_dg.yaml"),
    "sam":    os.path.join(CONFIG_DIR, "sam.yaml"),
}


def plot_loss_curves(results_dir):
    """Plot classification vs MMD-penalty loss per method (required evidence).

    Spec: "Training curves that include classification loss and, where
    applicable, the MMD penalty." ERM/SAM have no alignment term (flat 0);
    DAN-DG's loss_align is the pairwise-source MMD penalty.
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
            continue                                    # e.g. ERM loaded from Task 2
        steps = list(range(len(logs)))
        cls = [l.get("loss_cls", np.nan) for l in logs]
        align = [l.get("loss_align", np.nan) for l in logs]
        plt.figure(figsize=(7, 5))
        plt.plot(steps, cls, label="classification loss", alpha=0.8)
        if name == "dan_dg":
            plt.plot(steps, align, label="MMD penalty (pairwise source)", alpha=0.8)
        plt.xlabel("training step"); plt.ylabel("loss")
        plt.title(f"{name}: training curves")
        plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"loss_curves_{name}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--erm_checkpoint", default=None,
                        help="Task 2 Source-only checkpoint to reuse as the ERM baseline")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_study", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(METHOD_CONFIGS["erm"])
    results_dir = ensure_dir(base_cfg["paths"]["results_dir"])

    if not args.skip_train:
        for name, cfg_path in METHOD_CONFIGS.items():
            print(f"\n########## TRAIN: {name} ##########")
            cfg = load_config(cfg_path)
            # let the CLI inject the Task 2 ERM checkpoint for the ERM run
            if name == "erm" and args.erm_checkpoint:
                cfg["paths"]["task2_erm_checkpoint"] = args.erm_checkpoint
            train_one_config(cfg)

    plot_loss_curves(results_dir)

    print("\n########## FINAL EVALUATION (Sketch loaded here) ##########")
    evaluate_all(base_cfg)

    if not args.skip_study:
        print("\n########## CONTROLLED STUDY ##########")
        run_controlled_study(METHOD_CONFIGS["dan_dg"])

    print("\n[all done] see results/ for tables, curves, and JSON.")


if __name__ == "__main__":
    main()
