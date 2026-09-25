

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple-Silicon safety net
import sys
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../task3
PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # repo root
sys.path.insert(0, THIS_DIR)                                   # models/, methods/, ...
sys.path.insert(0, PROJECT_ROOT)                              # shared/

from shared.utils import load_config, set_seed, get_device, ensure_dir
from shared.pacs_protocol import (build_or_load_splits, make_source_datasets,
                                  InfiniteLoader, steps_per_epoch)
from models.backbone import ResNet18Backbone
from models.classifier_head import ClassifierHead
from methods.erm import ERM
from methods.dan_dg import DANDG
from methods.sam import SAMMethod, SAM
from selection.source_validation import evaluate_source_val
from torch.utils.data import DataLoader


def build_method(cfg):
    name = cfg["method"]["name"]
    if name == "erm":
        return ERM(cfg)
    if name == "dan_dg":
        return DANDG(cfg)
    if name == "sam":
        return SAMMethod(cfg)
    raise ValueError(f"Unknown method '{name}'")


def make_source_iters(cfg, train_sets):
   
    per = cfg["train"]["per_source_batch"]             # 8
    iters = {}
    for domain, ds in train_sets.items():
        loader = DataLoader(ds, batch_size=per, shuffle=True,
                            drop_last=True, num_workers=2)
        iters[domain] = InfiniteLoader(loader)
    return iters


def make_val_loaders(cfg, val_sets):
    return {d: DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
            for d, ds in val_sets.items()}


def _pull_balanced_batch(source_iters, cfg, device):

    batch = {}
    for domain in cfg["dataset"]["source_domains"]:
        imgs, labels = source_iters[domain].next()
        batch[domain] = (imgs.to(device), labels.to(device))
    return batch


def _forward_batch(backbone, head, batch):
    
    feats_by_domain, all_logits, all_labels = {}, [], []
    for domain, (imgs, labels) in batch.items():
        feats = backbone(imgs)                          # (8, 512)
        feats_by_domain[domain] = feats
        all_logits.append(head(feats))                  # (8, 7)
        all_labels.append(labels)
    return feats_by_domain, torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def _load_erm_checkpoint(cfg, backbone, head, device):
   
    path = cfg["paths"].get("task2_erm_checkpoint")
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=False)
        backbone.load_state_dict(state["backbone"])
        head.load_state_dict(state["head"])
        print(f"[info] loaded Task 2 ERM checkpoint from {path}")
        return True
    return False


def train_one_config(cfg, run_name=None, save_checkpoint=True):
    set_seed(cfg["seed"])
    device = get_device()
    num_classes = cfg["dataset"]["num_classes"]
    run_name = run_name or cfg["method"]["name"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints_dir"])
    print(f"[info] method={run_name} device={device}")

    splits = build_or_load_splits(cfg)
    train_sets, val_sets = make_source_datasets(cfg, splits, train=True)
    _, val_sets_eval = make_source_datasets(cfg, splits, train=False)  # clean val
    source_iters = make_source_iters(cfg, train_sets)
    val_loaders = make_val_loaders(cfg, val_sets_eval)
    n_steps = steps_per_epoch(cfg, train_sets)

    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(backbone.feature_dim, num_classes).to(device)
    method = build_method(cfg).to(device)

    if cfg["method"]["name"] == "erm" and _load_erm_checkpoint(cfg, backbone, head, device):
        backbone.eval(); head.eval()
        val = evaluate_source_val(backbone, head, val_loaders, device, num_classes)
        best_state = {"backbone": copy.deepcopy(backbone.state_dict()),
                      "head": copy.deepcopy(head.state_dict()),
                      "epoch": -1, "mean_source_val_macro_f1": val["mean_macro_f1"],
                      "config": cfg, "run_name": run_name, "loaded_from_task2": True}
        history = {"epoch": [], "mean_source_val_macro_f1": [], "step_logs": [],
                   "loaded_from_task2": True}
        if save_checkpoint:
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
            torch.save(best_state, ckpt_path)
            with open(os.path.join(results_dir, f"history_{run_name}.json"), "w") as f:
                json.dump(history, f, indent=2)
            print(f"[done] ERM loaded from Task 2; mean source-val macro-F1="
                  f"{val['mean_macro_f1']:.4f} -> {ckpt_path}")
            return best_state, history, ckpt_path
        return best_state, history, None

    params = list(backbone.parameters()) + list(head.parameters())
    for m in method.extra_modules():
        params += list(m.parameters())
    if method.uses_sam:
        optimizer = SAM(params, torch.optim.AdamW, rho=cfg["method"]["rho"],
                        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                                      weight_decay=cfg["train"]["weight_decay"])

    best_score, best_state, epochs_no_improve = -1.0, None, 0
    history = {"epoch": [], "mean_source_val_macro_f1": [],
               "worst_source_val_macro_f1": [], "step_logs": []}

    
    global_step = 0
    warmup_steps = int(getattr(method, "warmup_steps", 0) or 0)
    has_warmup = hasattr(method, "set_align_scale") and warmup_steps > 0

    for epoch in range(cfg["train"]["max_epochs"]):
        backbone.train(); head.train(); method.train()
        backbone.set_bn_eval()                          # frozen BN running stats

        for _ in range(n_steps):
            if has_warmup:
                method.set_align_scale(min(1.0, global_step / max(1, warmup_steps)))
            global_step += 1
            
            batch = _pull_balanced_batch(source_iters, cfg, device)

            if method.uses_sam:
               
                feats, logits, labels = _forward_batch(backbone, head, batch)
                loss1, logs = method.compute_loss(feats, logits, labels)
                loss1.backward()
                optimizer.first_step(zero_grad=True)    # move weights to theta+eps
                backbone.set_bn_eval()                  # keep BN frozen on pass 2
                
                feats2, logits2, labels2 = _forward_batch(backbone, head, batch)
                loss2, _ = method.compute_loss(feats2, logits2, labels2)
                loss2.backward()
                optimizer.second_step(zero_grad=True)   # restore theta, AdamW step
            else:
                feats, logits, labels = _forward_batch(backbone, head, batch)
                loss, logs = method.compute_loss(feats, logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logs["epoch"] = epoch
            history["step_logs"].append(logs)

        backbone.eval(); head.eval()
        val = evaluate_source_val(backbone, head, val_loaders, device, num_classes)
        mean_f1 = val["mean_macro_f1"]
        history["epoch"].append(epoch)
        history["mean_source_val_macro_f1"].append(mean_f1)
        history["worst_source_val_macro_f1"].append(val["worst_macro_f1"])
        print(f"  epoch {epoch:2d} | mean src-val macro-F1={mean_f1:.4f} "
              f"| worst={val['worst_macro_f1']:.4f}")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_state = {"backbone": copy.deepcopy(backbone.state_dict()),
                          "head": copy.deepcopy(head.state_dict()),
                          "epoch": epoch, "mean_source_val_macro_f1": mean_f1,
                          "config": cfg, "run_name": run_name}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["train"]["early_stop_patience"]:
                print(f"  early stopping at epoch {epoch}")
                break

    ckpt_path = None
    if save_checkpoint and best_state is not None:
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
        torch.save(best_state, ckpt_path)
        with open(os.path.join(results_dir, f"history_{run_name}.json"), "w") as f:
            json.dump(_json_safe(history), f, indent=2)
        print(f"[done] best mean src-val macro-F1={best_score:.4f} -> {ckpt_path}")
    return best_state, history, ckpt_path


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
    parser.add_argument("--config", required=True)
    parser.add_argument("--erm_checkpoint", default=None,
                        help="path to Task 2 Source-only checkpoint to reuse as ERM")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.erm_checkpoint:
        cfg["paths"]["task2_erm_checkpoint"] = args.erm_checkpoint
    train_one_config(cfg)
