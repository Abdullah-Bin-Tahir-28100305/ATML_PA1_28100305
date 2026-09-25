

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
from evaluation.metrics import (evaluate_loader, collect_predictions,
                                top1_accuracy, macro_f1)
from evaluation.domain_separability import compute_domain_separability
from evaluation.class_analysis import (per_class_accuracy, compare_to_baseline,
                                       dominant_confusions)
from torch.utils.data import DataLoader


def _load_model(ckpt_path, num_classes, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(backbone.feature_dim, num_classes).to(device)
    backbone.load_state_dict(state["backbone"])
    head.load_state_dict(state["head"])
    backbone.eval(); head.eval()                       # full eval for measurement
    return backbone, head


def evaluate_all(cfg, method_names=("source_only", "dan", "dann", "cdan")):
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

    summary = {}                                       # method -> metrics dict
    target_preds = {}                                  # method -> (preds, labels)

    for name in method_names:
        ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[warn] checkpoint missing for '{name}' ({ckpt_path}); skipping.")
            continue
        backbone, head = _load_model(ckpt_path, num_classes, device)

        per_domain = {}
        for d, ld in val_loaders.items():
            r = evaluate_loader(backbone, head, ld, device, num_classes)
            per_domain[d] = {"top1": r["top1"], "macro_f1": r["macro_f1"]}
        mean_src_top1 = float(np.mean([v["top1"] for v in per_domain.values()]))
        mean_src_f1 = float(np.mean([v["macro_f1"] for v in per_domain.values()]))

        # TARGET metrics (labels used only here, after freezing).
        t = evaluate_loader(backbone, head, target_loader, device, num_classes)
        target_preds[name] = (t["preds"], t["labels"])

        
        combined_src_val = DataLoader(
            torch.utils.data.ConcatDataset(list(val_sets.values())),
            batch_size=64, shuffle=True, num_workers=2)
        sep = compute_domain_separability(backbone, combined_src_val,
                                          target_loader, cfg, device)

        summary[name] = {
            "per_source_domain": per_domain,
            "mean_source_top1": mean_src_top1,
            "mean_source_macro_f1": mean_src_f1,
            "target_top1": t["top1"],
            "target_macro_f1": t["macro_f1"],
            "domain_separability": sep["domain_separability"],
        }
        print(f"[eval] {name:12s} src_f1={mean_src_f1:.3f} "
              f"tgt_top1={t['top1']:.3f} sep={sep['domain_separability']:.3f}")

    if "source_only" in summary:
        base_top1 = summary["source_only"]["target_top1"]
        for name in summary:
            summary[name]["target_top1_change_vs_source_only"] = \
                summary[name]["target_top1"] - base_top1

    class_report = {}
    if "source_only" in target_preds:
        base_preds, base_labels = target_preds["source_only"]
        base_pca = per_class_accuracy(base_preds, base_labels, num_classes)
        for name, (preds, labels) in target_preds.items():
            pca = per_class_accuracy(preds, labels, num_classes)
            class_report[name] = {
                "per_class_accuracy": {PACS_CLASSES[c]: pca[c] for c in range(num_classes)},
                "vs_source_only": compare_to_baseline(pca, base_pca, PACS_CLASSES),
                "dominant_confusions": dominant_confusions(
                    preds, labels, num_classes, PACS_CLASSES),
            }

    out = {"summary_table": summary, "class_analysis": class_report,
           "class_names": PACS_CLASSES}
    with open(os.path.join(results_dir, "task2_final_results.json"), "w") as f:
        json.dump(_json_safe(out), f, indent=2)
    _write_summary_csv(summary, cfg, results_dir)
    print(f"[done] final evaluation written to {results_dir}")
    return out


def _write_summary_csv(summary, cfg, results_dir):
    import csv
    src_domains = cfg["dataset"]["source_domains"]
    header = ["method"] + [f"{d}_val_top1" for d in src_domains] + \
             [f"{d}_val_f1" for d in src_domains] + \
             ["mean_src_top1", "mean_src_f1", "target_top1", "target_f1",
              "target_top1_change_vs_source_only", "domain_separability"]
    rows = [header]
    for name, s in summary.items():
        row = [name]
        row += [f"{s['per_source_domain'][d]['top1']:.4f}" for d in src_domains]
        row += [f"{s['per_source_domain'][d]['macro_f1']:.4f}" for d in src_domains]
        row += [f"{s['mean_source_top1']:.4f}", f"{s['mean_source_macro_f1']:.4f}",
                f"{s['target_top1']:.4f}", f"{s['target_macro_f1']:.4f}",
                f"{s.get('target_top1_change_vs_source_only', float('nan')):.4f}",
                f"{s['domain_separability']:.4f}"]
        rows.append(row)
    with open(os.path.join(results_dir, "task2_summary_table.csv"), "w", newline="") as f:
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
