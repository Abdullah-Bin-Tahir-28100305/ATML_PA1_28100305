# =============================================================================
# task2/controlled_study.py  — Controlled Design Study (Step 6)
# -----------------------------------------------------------------------------
# PURPOSE: run the ONE bounded alignment-strength sweep the assignment asks for,
# holding everything else fixed. We default to the DAN option:
#     vary lambda_MMD in {0.1, 1, 10}
# (The DANN option — vary max GRL strength in {0.25,0.5,1} — is supported too;
#  it is selected by enabling controlled_study in dann.yaml instead of dan.yaml.)
#
# ML CONCEPT — ALIGNMENT STRENGTH TRADE-OFF (spec):
#   "State what you expect the increasing alignment pressure to do to source
#    performance, domain separability, and target recognition before interpreting
#    the results." Too little alignment barely closes the domain gap; too much can
#    distort features and hurt source (and sometimes target) performance. The
#    sweep traces this trade-off. Crucially: "These target results are for
#    analysis only and may not be used to revise the tested settings" — so we
#    NEVER pick the sweep value using target numbers.
#
# For each sweep value we train a fresh model, then evaluate source-val + target
# + domain separability, and tabulate them so the trade-off is visible.
#
# LINKS: train.train_one_config (training) and the same evaluation utilities used
# by evaluate_final.py.
# =============================================================================

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import json
import copy
import argparse
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from shared.utils import load_config, set_seed, get_device, ensure_dir
from shared.pacs_protocol import (build_or_load_splits, make_source_datasets,
                                  make_target_dataset)
from models.backbone import ResNet18Backbone
from models.classifier_head import ClassifierHead
from evaluation.metrics import evaluate_loader
from evaluation.domain_separability import compute_domain_separability
from train import train_one_config
from torch.utils.data import DataLoader, ConcatDataset


def _set_nested(cfg, param, value):
    """Set cfg['method'][param] = value (the sweep params all live under method)."""
    cfg = copy.deepcopy(cfg)
    cfg["method"][param] = value
    return cfg


def run_controlled_study(config_path):
    """Run the alignment-strength sweep defined in the given method config."""
    cfg = load_config(config_path)
    cs = cfg.get("controlled_study", {})
    if not cs.get("enabled", False):
        print(f"[info] controlled_study.enabled is False in {config_path}; "
              f"nothing to sweep. Enable it in the method config to run this.")
        return None

    param = cs["sweep_param"]                          # e.g. 'lambda_mmd'
    values = cs["sweep_values"]                        # e.g. [0.1, 1.0, 10.0]
    device = get_device()
    num_classes = cfg["dataset"]["num_classes"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    # Prepare the evaluation loaders once (identical across sweep values).
    splits = build_or_load_splits(cfg)
    _, val_sets = make_source_datasets(cfg, splits, train=False)
    val_loaders = {d: DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
                   for d, ds in val_sets.items()}
    combined_src_val = DataLoader(ConcatDataset(list(val_sets.values())),
                                  batch_size=64, shuffle=True, num_workers=2)
    target_eval = make_target_dataset(cfg, train=False)
    target_loader = DataLoader(target_eval, batch_size=64, shuffle=False, num_workers=2)

    table = []
    for v in values:
        print(f"\n===== controlled study: {param} = {v} =====")
        cfg_v = _set_nested(cfg, param, v)
        run_name = f"{cfg['method']['name']}_{param}_{v}"
        # Train (do not overwrite the main checkpoints during the sweep).
        best_state, _, _ = train_one_config(cfg_v, run_name=run_name,
                                            save_checkpoint=False)

        # Rebuild the best model in memory for evaluation.
        backbone = ResNet18Backbone().to(device)
        head = ClassifierHead(backbone.feature_dim, num_classes).to(device)
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        backbone.eval(); head.eval()

        # Source metrics (selection-relevant), target metrics (analysis only),
        # and domain separability.
        per_domain = {d: evaluate_loader(backbone, head, ld, device, num_classes)
                      for d, ld in val_loaders.items()}
        mean_src_f1 = float(np.mean([r["macro_f1"] for r in per_domain.values()]))
        t = evaluate_loader(backbone, head, target_loader, device, num_classes)
        sep = compute_domain_separability(backbone, combined_src_val,
                                          target_loader, cfg_v, device)
        table.append({
            param: v,
            "mean_source_macro_f1": mean_src_f1,
            "domain_separability": sep["domain_separability"],
            "target_top1": t["top1"],
            "target_macro_f1": t["macro_f1"],
        })
        print(f"  {param}={v}: src_f1={mean_src_f1:.3f} "
              f"sep={sep['domain_separability']:.3f} tgt_top1={t['top1']:.3f}")

    # Save the sweep table + a simple plot.
    with open(os.path.join(results_dir, "controlled_study.json"), "w") as f:
        json.dump({"param": param, "values": values, "table": table}, f, indent=2)
    _plot_study(param, table, results_dir)
    print(f"[done] controlled study written to {results_dir}")
    return table


def _plot_study(param, table, results_dir):
    """Plot source-F1, domain separability, and target top-1 vs the swept param."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [row[param] for row in table]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, [r["mean_source_macro_f1"] for r in table], "o-", label="mean source macro-F1")
    plt.plot(xs, [r["domain_separability"] for r in table], "s-", label="domain separability")
    plt.plot(xs, [r["target_top1"] for r in table], "^-", label="target top-1")
    plt.xscale("log")                                  # sweep values span decades
    plt.xlabel(f"alignment strength ({param})")
    plt.ylabel("score")
    plt.title(f"Controlled alignment-strength study ({param})")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "controlled_study.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "configs", "dan.yaml"),
                        help="method config whose controlled_study block to run")
    args = parser.parse_args()
    run_controlled_study(args.config)
