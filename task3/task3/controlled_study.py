

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

from shared.utils import load_config, get_device, ensure_dir
from shared.pacs_protocol import (build_or_load_splits, make_source_datasets,
                                  make_target_dataset)
from models.backbone import ResNet18Backbone
from models.classifier_head import ClassifierHead
from selection.source_validation import evaluate_source_val
from evaluation.domain_metrics import evaluate_target
from evaluation.source_domain_separability import compute_source_domain_separability
from evaluation.sharpness import compute_sharpness_proxy
from train import train_one_config
from torch.utils.data import DataLoader


def run_controlled_study(config_path):
    cfg = load_config(config_path)
    cs = cfg.get("controlled_study", {})
    if not cs.get("enabled", False):
        print(f"[info] controlled_study.enabled is False in {config_path}; nothing to sweep.")
        return None

    param = cs["sweep_param"]                           # 'lambda_dg' or 'rho'
    values = cs["sweep_values"]
    device = get_device()
    num_classes = cfg["dataset"]["num_classes"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    splits = build_or_load_splits(cfg)
    _, val_sets = make_source_datasets(cfg, splits, train=False)
    val_loaders = {d: DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
                   for d, ds in val_sets.items()}
    target_loader = DataLoader(make_target_dataset(cfg, train=False),
                               batch_size=64, shuffle=False, num_workers=2)

    method_name = cfg["method"]["name"]
    table = []
    for v in values:
        print(f"\n===== controlled study: {param} = {v} =====")
        cfg_v = copy.deepcopy(cfg)
        cfg_v["method"][param] = v                      # override the swept knob
        run_name = f"{method_name}_{param}_{v}"
        best_state, _, _ = train_one_config(cfg_v, run_name=run_name, save_checkpoint=False)

        backbone = ResNet18Backbone().to(device)
        head = ClassifierHead(backbone.feature_dim, num_classes).to(device)
        backbone.load_state_dict(best_state["backbone"]); head.load_state_dict(best_state["head"])
        backbone.eval(); head.eval()

        val = evaluate_source_val(backbone, head, val_loaders, device, num_classes)
        t = evaluate_target(backbone, head, target_loader, device, num_classes)
        # method-specific diagnostic
        if method_name == "dan_dg":
            diag = compute_source_domain_separability(backbone, val_loaders, cfg_v, device)
            diag_val = diag["source_domain_separability"]; diag_name = "source_domain_separability"
        else:  # sam
            diag = compute_sharpness_proxy(backbone, head, val_sets, cfg_v, device)
            diag_val = diag["delta_sharp"]; diag_name = "delta_sharp"

        table.append({param: v, "mean_source_macro_f1": val["mean_macro_f1"],
                      diag_name: diag_val, "sketch_top1": t["top1"],
                      "sketch_macro_f1": t["macro_f1"]})
        print(f"  {param}={v}: src_f1={val['mean_macro_f1']:.3f} "
              f"{diag_name}={diag_val:.4f} sketch_top1={t['top1']:.3f}")

    with open(os.path.join(results_dir, "controlled_study.json"), "w") as f:
        json.dump({"method": method_name, "param": param, "values": values, "table": table}, f, indent=2)
    _plot_study(param, table, method_name, results_dir)
    print(f"[done] controlled study written to {results_dir}")
    return table


def _plot_study(param, table, method_name, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r[param] for r in table]
    diag_name = "source_domain_separability" if method_name == "dan_dg" else "delta_sharp"
    plt.figure(figsize=(7, 5))
    plt.plot(xs, [r["mean_source_macro_f1"] for r in table], "o-", label="mean source macro-F1")
    plt.plot(xs, [r[diag_name] for r in table], "s-", label=diag_name)
    plt.plot(xs, [r["sketch_top1"] for r in table], "^-", label="sketch top-1 (analysis only)")
    plt.xscale("log")
    plt.xlabel(f"strength ({param})"); plt.ylabel("score")
    plt.title(f"Controlled study ({method_name}, {param})")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "controlled_study.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "configs", "dan_dg.yaml"))
    args = parser.parse_args()
    run_controlled_study(args.config)
