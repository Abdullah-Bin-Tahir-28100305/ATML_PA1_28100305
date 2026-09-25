# =============================================================================
# task2/train.py
# -----------------------------------------------------------------------------
# THE COMMON TRAINING LOOP shared by all four methods (spec: "all methods must
# run through the same training and evaluation pipeline"). It:
#   * builds ResNet-18 backbone + 7-class head + (for DANN/CDAN) a discriminator;
#   * assembles domain-balanced batches (8 per source x3 + 24 target);
#   * enforces the frozen-BatchNorm policy after every model.train();
#   * ramps the GRL alpha on the standard DANN schedule;
#   * selects the checkpoint by MEAN SOURCE-VALIDATION macro-F1 (no target labels);
#   * early-stops after 5 epochs without improvement;
#   * logs per-step losses (for the required loss curves) and saves the best
#     checkpoint + per-epoch history.
#
# This file is imported by the CLI at the bottom and by controlled_study.py.
#
# LINKS: shared/{utils,pacs,pacs_protocol}, models/*, methods/*, evaluation/metrics.
# =============================================================================

import os
# Apple-Silicon safety net: run any op lacking a Metal kernel on CPU instead of
# crashing (harmless on CUDA/CPU). Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn

# Make both the project root and the shared/ package importable.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../task2
PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # repo root
sys.path.insert(0, THIS_DIR)                                   # for models/, methods/
sys.path.insert(0, PROJECT_ROOT)                              # for shared/

from shared.utils import load_config, set_seed, get_device, ensure_dir
from shared.pacs_protocol import (build_or_load_splits, make_source_datasets,
                                  make_target_dataset,
                                  make_domain_balanced_iterators, steps_per_epoch)
from models.backbone import ResNet18Backbone
from models.classifier_head import ClassifierHead
from models.domain_discriminator import grl_alpha
from methods.source_only import SourceOnly
from methods.dan import DAN
from methods.dann import DANN
from methods.cdan import CDAN
from evaluation.metrics import evaluate_loader, mean_source_val_macro_f1
from torch.utils.data import DataLoader


def build_method(cfg, feature_dim, num_classes):
    """Instantiate the method object named in the config.

    Each method shares the same interface (methods/base_method.py), so the loop
    below is method-agnostic. DANN/CDAN also own a discriminator whose parameters
    we add to the optimizer via method.extra_modules().
    """
    name = cfg["method"]["name"]
    if name == "source_only":
        return SourceOnly(cfg)
    if name == "dan":
        return DAN(cfg)
    if name == "dann":
        return DANN(cfg, feature_dim=feature_dim)
    if name == "cdan":
        return CDAN(cfg, feature_dim=feature_dim, num_classes=num_classes)
    raise ValueError(f"Unknown method '{name}'")


def make_val_loaders(cfg, val_sets):
    """Plain (non-cycling) loaders over each source-VAL subset for evaluation."""
    return {d: DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
            for d, ds in val_sets.items()}


def train_one_config(cfg, run_name=None, save_checkpoint=True):
    """Train a single method end-to-end and return (best_state, history, paths).

    `run_name` labels the outputs (defaults to the method name). `save_checkpoint`
    can be turned off during the controlled study's sweep runs.
    """
    set_seed(cfg["seed"])                              # full determinism (6304)
    device = get_device()
    num_classes = cfg["dataset"]["num_classes"]
    run_name = run_name or cfg["method"]["name"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints_dir"])
    print(f"[info] method={run_name} device={device}")

    # ---- DATA ----------------------------------------------------------------
    splits = build_or_load_splits(cfg)                # stratified 80/20, seed 6304
    train_sets, val_sets = make_source_datasets(cfg, splits, train=True)
    # Target used UNLABELED during adaptation (augmented view).
    target_set = make_target_dataset(cfg, train=True)
    source_iters, target_iter = make_domain_balanced_iterators(cfg, train_sets, target_set)
    val_loaders = make_val_loaders(cfg, val_sets)
    n_steps = steps_per_epoch(cfg, train_sets)        # bookkeeping unit for schedule

    # ---- MODEL ---------------------------------------------------------------
    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(backbone.feature_dim, num_classes).to(device)
    method = build_method(cfg, backbone.feature_dim, num_classes).to(device)

    # Collect all trainable parameters: backbone + head + any method modules
    # (e.g. the domain discriminator for DANN/CDAN).
    # Backbone + head always train at the spec learning rate.
    model_params = list(backbone.parameters()) + list(head.parameters())
    # Any method modules (the domain discriminator for DANN/CDAN) are collected
    # separately so they CAN be given their own learning rate.
    extra_params = []
    for m in method.extra_modules():
        extra_params += list(m.parameters())
    # `params` is the flat list used for gradient clipping (all trainable params).
    params = model_params + extra_params

    base_lr = cfg["train"]["lr"]
    weight_decay = cfg["train"]["weight_decay"]
    # Two-timescale option: if the method config sets a separate `disc_lr`, the
    # discriminator (extra modules) trains at that rate while backbone+head keep
    # the spec lr. This is the standard DANN stabiliser and leaves the spec's
    # model learning rate untouched. Absent disc_lr, everything shares base_lr
    # exactly as before (so DAN/CDAN/source_only are unaffected).
    disc_lr = cfg["method"].get("disc_lr", None)
    if disc_lr is not None and extra_params:
        optimizer = torch.optim.AdamW(
            [{"params": model_params, "lr": base_lr},
             {"params": extra_params, "lr": disc_lr}],
            lr=base_lr, weight_decay=weight_decay)
        print(f"[info] two-timescale optimizer: model lr={base_lr}, disc lr={disc_lr}")
    else:
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    # Gradient-clipping ceiling (see base.yaml). 0/None disables it. This is the
    # numerical-stability guard that keeps the adversarial DANN/CDAN runs from
    # diverging; it is applied identically to every method so the comparison stays
    # controlled, and it rarely triggers on the already-stable non-adversarial runs.
    grad_clip_norm = cfg["train"].get("grad_clip_norm", 0.0)

    # ---- TRAIN LOOP ----------------------------------------------------------
    total_epochs = cfg["train"]["max_epochs"]
    total_steps = total_epochs * n_steps              # for GRL progress p in [0,1]
    global_step = 0
    best_score, best_state, epochs_no_improve = -1.0, None, 0
    history = {"epoch": [], "mean_source_val_macro_f1": [], "step_logs": []}

    grl_gamma = cfg["method"].get("grl_gamma", 10.0)
    grl_max = cfg["method"].get("grl_max_alpha", 1.0)

    for epoch in range(total_epochs):
        # Put trainable modules in TRAIN mode, then RE-FREEZE BatchNorm running
        # stats (spec: only BN modules go to eval, not the whole model).
        backbone.train(); head.train(); method.train()
        backbone.set_bn_eval()                        # <-- frozen BN running stats

        for _ in range(n_steps):
            # GRL alpha depends on overall training progress p in [0,1].
            p = global_step / max(1, total_steps)
            alpha = grl_alpha(p, grl_gamma, grl_max)  # 0 for non-adversarial too

            # --- assemble a domain-balanced batch ---
            src_imgs, src_labels = [], []
            for domain in cfg["dataset"]["source_domains"]:
                imgs, labels = source_iters[domain].next()   # 8 imgs from domain
                src_imgs.append(imgs); src_labels.append(labels)
            src_imgs = torch.cat(src_imgs, 0).to(device)      # 24 source imgs
            src_labels = torch.cat(src_labels, 0).to(device)  # 24 source labels
            tgt_imgs, _ = target_iter.next()                  # 24 target imgs (unlabeled)
            tgt_imgs = tgt_imgs.to(device)

            # --- forward ---
            src_feats = backbone(src_imgs)                    # (24, 512)
            tgt_feats = backbone(tgt_imgs)                    # (24, 512)
            src_logits = head(src_feats)                      # (24, 7)
            tgt_logits = head(tgt_feats)                      # (24, 7) (for CDAN/logging)

            # --- method-specific loss (class loss + alignment) ---
            loss, logs = method.compute_loss(
                src_feats, src_logits, src_labels,
                tgt_feats, tgt_logits, alpha)

            # --- optimize ---
            optimizer.zero_grad()
            loss.backward()                                  # GRL reverses grads internally
            # Clip the global gradient norm BEFORE the step. This bounds the size
            # of each update so an adversarial gradient spike (DANN/CDAN) cannot
            # blow the weights up. Returns the pre-clip total norm, which we log
            # so the loss curves can show whether/when clipping was active.
            if grad_clip_norm and grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                logs["grad_norm"] = float(grad_norm)
            optimizer.step()

            logs["epoch"] = epoch; logs["step"] = global_step
            history["step_logs"].append(logs)
            global_step += 1

        # ---- checkpoint selection: MEAN SOURCE-VAL MACRO-F1 (no target!) -----
        # Evaluate source-val splits. We keep BN frozen and use no-grad; the
        # backbone stays in train()+BN-eval configuration, which is fine for a
        # forward pass and matches how features behave during training.
        backbone.eval(); head.eval()                         # deterministic eval
        per_domain = {d: evaluate_loader(backbone, head, ld, device, num_classes)
                      for d, ld in val_loaders.items()}
        mean_f1 = mean_source_val_macro_f1(per_domain)
        history["epoch"].append(epoch)
        history["mean_source_val_macro_f1"].append(mean_f1)
        print(f"  epoch {epoch:2d} | mean source-val macro-F1 = {mean_f1:.4f}")

        # early stopping: keep the best-so-far weights (backbone+head+disc).
        if mean_f1 > best_score:
            best_score = mean_f1
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
                "method": copy.deepcopy(method.state_dict()),
                "epoch": epoch, "mean_source_val_macro_f1": mean_f1,
                "config": cfg, "run_name": run_name,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["train"]["early_stop_patience"]:
                print(f"  early stopping at epoch {epoch} "
                      f"(no improvement for {epochs_no_improve} epochs)")
                break

    # ---- SAVE ----------------------------------------------------------------
    ckpt_path = None
    if save_checkpoint and best_state is not None:
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
        torch.save(best_state, ckpt_path)
        with open(os.path.join(results_dir, f"history_{run_name}.json"), "w") as f:
            json.dump(_json_safe(history), f, indent=2)
        print(f"[done] best mean source-val macro-F1={best_score:.4f} "
              f"-> {ckpt_path}")
    return best_state, history, ckpt_path


def _json_safe(obj):
    """Convert numpy/torch scalars to plain Python for JSON serialization."""
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
    # drop the nested config dict from step logs to keep files small
    return obj


# -----------------------------------------------------------------------------
# CLI: train one method from its config file.
#   python task2/train.py --config task2/configs/source_only.yaml
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a method YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_one_config(cfg)
