

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
