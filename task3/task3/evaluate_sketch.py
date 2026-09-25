

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import json
import argparse
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from shared.utils import load_config, set_seed, get_device, ensure_dir
from shared.pacs import PACS_CLASSES
from shared.pacs_protocol import (build_or_load_splits, make_source_datasets,
                                  make_target_dataset)
from models.backbone import ResNet18Backbone
from models.classifier_head import ClassifierHead
from selection.source_validation import evaluate_source_val
from evaluation.domain_metrics import (evaluate_target, per_class_accuracy,
                                       compare_to_baseline, dominant_confusions)
from evaluation.source_domain_separability import compute_source_domain_separability
from evaluation.sharpness import compute_sharpness_proxy
from torch.utils.data import DataLoader


def _load_model(ckpt_path, num_classes, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(backbone.feature_dim, num_classes).to(device)
    backbone.load_state_dict(state["backbone"]); head.load_state_dict(state["head"])
    backbone.eval(); head.eval()
    return backbone, head


def evaluate_all(cfg, method_names=("erm", "dan_dg", "sam")):
    set_seed(cfg["seed"])
    device = get_device()
    num_classes = cfg["dataset"]["num_classes"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    ckpt_dir = cfg["paths"]["checkpoints_dir"]

    splits = build_or_load_splits(cfg)
    _, val_sets = make_source_datasets(cfg, splits, train=False)
    val_loaders = {d: DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
                   for d, ds in val_sets.items()}

    target_set = make_target_dataset(cfg, train=False)
    target_loader = DataLoader(target_set, batch_size=64, shuffle=False, num_workers=2)

    summary, target_preds = {}, {}
    for name in method_names:
        ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[warn] missing checkpoint for '{name}' ({ckpt_path}); skipping.")
            continue
        backbone, head = _load_model(ckpt_path, num_classes, device)

        val = evaluate_source_val(backbone, head, val_loaders, device, num_classes)
        t = evaluate_target(backbone, head, target_loader, device, num_classes)
        target_preds[name] = (t["preds"], t["labels"])
        sep = compute_source_domain_separability(backbone, val_loaders, cfg, device)
        sharp = compute_sharpness_proxy(backbone, head, val_sets, cfg, device)

        summary[name] = {
            "per_source_domain": val["per_domain"],
            "mean_source_top1": val["mean_top1"],
            "mean_source_macro_f1": val["mean_macro_f1"],
            "worst_source_top1": val["worst_top1"],
            "worst_source_macro_f1": val["worst_macro_f1"],
            "sketch_top1": t["top1"],
            "sketch_macro_f1": t["macro_f1"],
            "source_domain_separability": sep["source_domain_separability"],
            "delta_sharp": sharp["delta_sharp"],
        }
        print(f"[eval] {name:7s} mean_src_f1={val['mean_macro_f1']:.3f} "
              f"worst_src_f1={val['worst_macro_f1']:.3f} "
              f"sketch_top1={t['top1']:.3f} sep={sep['source_domain_separability']:.3f} "
              f"delta_sharp={sharp['delta_sharp']:.4f}")

    if "erm" in summary:
        base = summary["erm"]["sketch_top1"]
        for name in summary:
            summary[name]["sketch_top1_change_vs_erm"] = summary[name]["sketch_top1"] - base

    class_report = {}
    if "erm" in target_preds:
        base_preds, base_labels = target_preds["erm"]
        base_pca = per_class_accuracy(base_preds, base_labels, num_classes)
        for name, (preds, labels) in target_preds.items():
            pca = per_class_accuracy(preds, labels, num_classes)
            class_report[name] = {
                "per_class_accuracy": {PACS_CLASSES[c]: pca[c] for c in range(num_classes)},
                "vs_erm": compare_to_baseline(pca, base_pca, PACS_CLASSES),
                "dominant_confusions": dominant_confusions(preds, labels, num_classes, PACS_CLASSES),
            }

    out = {"summary_table": summary, "class_analysis": class_report,
           "class_names": PACS_CLASSES,
           "separability_chance": 1.0 / len(cfg["dataset"]["source_domains"])}
    with open(os.path.join(results_dir, "task3_final_results.json"), "w") as f:
        json.dump(_json_safe(out), f, indent=2)
    _write_summary_csv(summary, cfg, results_dir)
    print(f"[done] Task 3 final evaluation written to {results_dir}")
    return out


def _write_summary_csv(summary, cfg, results_dir):
    import csv
    src = cfg["dataset"]["source_domains"]
    header = ["method"] + [f"{d}_val_top1" for d in src] + [f"{d}_val_f1" for d in src] + \
             ["mean_src_top1", "mean_src_f1", "worst_src_top1", "worst_src_f1",
              "sketch_top1", "sketch_f1", "sketch_top1_change_vs_erm",
              "source_domain_separability", "delta_sharp"]
    rows = [header]
    for name, s in summary.items():
        row = [name]
        row += [f"{s['per_source_domain'][d]['top1']:.4f}" for d in src]
        row += [f"{s['per_source_domain'][d]['macro_f1']:.4f}" for d in src]
        row += [f"{s['mean_source_top1']:.4f}", f"{s['mean_source_macro_f1']:.4f}",
                f"{s['worst_source_top1']:.4f}", f"{s['worst_source_macro_f1']:.4f}",
                f"{s['sketch_top1']:.4f}", f"{s['sketch_macro_f1']:.4f}",
                f"{s.get('sketch_top1_change_vs_erm', float('nan')):.4f}",
                f"{s['source_domain_separability']:.4f}", f"{s['delta_sharp']:.4f}"]
        rows.append(row)
    with open(os.path.join(results_dir, "task3_summary_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(THIS_DIR, "configs", "base.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_all(cfg)
