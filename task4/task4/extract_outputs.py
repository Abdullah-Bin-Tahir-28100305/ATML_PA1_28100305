# =============================================================================
# task4/extract_outputs.py
# -----------------------------------------------------------------------------
# PURPOSE: the SCORE-EXTRACTION stage. For a fixed, selected checkpoint, run the
# model ONCE over every evaluation set and cache the raw outputs:
#   * CIFAR-10 TRAIN (clean/unaugmented) features + labels  -> for Mahalanobis fit
#   * CIFAR-10 VAL   logits + features                      -> threshold calibration
#   * CIFAR-10 TEST  logits + features + labels             -> known eval
#   * NEAR unknown   logits + features                      -> unknown eval
#   * FAR  unknown   logits + features                      -> unknown eval
# For PROSER we also cache the DUMMY logits; for RPL the distance-logits.
#
# WHY A SEPARATE EXTRACTION STAGE (spec):
#   "Keep dataset construction, model training, score extraction, and evaluation
#   separate so that every score receives identical examples." Caching the outputs
#   once and scoring from the cache guarantees MSP/MLS/Energy/Mahalanobis all read
#   EXACTLY the same logits/features (spec: "All four scores must use exactly the
#   same saved logits and features.").
#
# LINKS: models/resnet_cifar, methods/{proser,rpl}, data/*; feeds evaluate_osr.py.
# =============================================================================

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import argparse
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils import load_config, set_seed, get_device, ensure_dir
from data.cifar10 import load_cifar10_train, load_cifar10_test
from data.make_splits import build_or_load_split
from data.cifar100_unknowns import load_unknown_groups, unknown_class_names
from models.resnet_cifar import CifarResNet18
from methods.proser import ProserModel
from methods.rpl import RPLModel
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def _run(model, loader, device, method):
    """Return dict of stacked numpy outputs for a loader.

    Always returns 'logits', 'features', 'labels'. Adds 'dummy' for PROSER and
    'dist' (distance-logits) for RPL.
    """
    model.eval()
    out = {"logits": [], "features": [], "labels": [], "dummy": [], "dist": []}
    for imgs, labels in loader:
        imgs = imgs.to(device)
        if method == "proser":
            feats = model.backbone.forward_features(imgs)
            known = model.backbone.forward_from_features(feats)
            dummy = model.dummy(feats)
            out["dummy"].append(dummy.cpu().numpy())
            out["logits"].append(known.cpu().numpy())
            out["features"].append(feats.cpu().numpy())
        elif method == "rpl":
            feats, dist = model.distances(imgs)
            # For RPL the closed-set 'logits' are the distance-logits (argmax = pred).
            out["logits"].append(dist.cpu().numpy())
            out["dist"].append(dist.cpu().numpy())
            out["features"].append(feats.cpu().numpy())
        else:
            feats = model.forward_features(imgs)
            logits = model.forward_from_features(feats)
            out["logits"].append(logits.cpu().numpy())
            out["features"].append(feats.cpu().numpy())
        out["labels"].append(np.asarray(labels))
    res = {}
    for k, v in out.items():
        if len(v) > 0:
            res[k] = np.concatenate(v, axis=0)
    return res


def _load_model(cfg, ckpt_path, device):
    """Rebuild the correct model type from a checkpoint and load weights."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    method = state["method"]
    num_classes = cfg["known"]["num_classes"]
    backbone = CifarResNet18(num_classes=num_classes).to(device)
    if method == "proser":
        model = ProserModel(backbone, num_classes, cfg["method"]["num_dummy"]).to(device)
        model.load_state_dict(state["full_model"])
    elif method == "rpl":
        model = RPLModel(backbone, num_classes).to(device)
        model.load_state_dict(state["full_model"])
    else:
        backbone.load_state_dict(state["model"]); model = backbone
    model.eval()
    return model, method


def extract_for_checkpoint(cfg, run_name):
    """Extract + cache all outputs for one checkpoint into cache/<run_name>.npz."""
    set_seed(cfg["seed"])
    device = get_device()
    cache_dir = ensure_dir(cfg["paths"]["cache_dir"])
    ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], f"{run_name}.pt")
    model, method = _load_model(cfg, ckpt_path, device)
    bs = 256; nw = cfg["train"]["num_workers"]

    # CIFAR-10 train (CLEAN) restricted to the TRAIN split -> Mahalanobis stats.
    train_clean = load_cifar10_train(cfg, train_transform=False)
    split = build_or_load_split(cfg, train_clean)
    train_sub = Subset(train_clean, split["train"])
    val_sub = Subset(train_clean, split["val"])
    test_set = load_cifar10_test(cfg)
    near, far = load_unknown_groups(cfg)

    loaders = {
        "train": DataLoader(train_sub, batch_size=bs, shuffle=False, num_workers=nw),
        "val":   DataLoader(val_sub, batch_size=bs, shuffle=False, num_workers=nw),
        "test":  DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nw),
        "near":  DataLoader(near, batch_size=bs, shuffle=False, num_workers=nw),
        "far":   DataLoader(far, batch_size=bs, shuffle=False, num_workers=nw),
    }
    cache = {"method": method, "run_name": run_name}
    for split_name, loader in loaders.items():
        res = _run(model, loader, device, method)
        for k, v in res.items():
            cache[f"{split_name}_{k}"] = v
        print(f"  extracted {split_name}: {res['logits'].shape[0]} examples")

    # store unknown class names for the failure analysis (evaluation-only labels)
    cache["near_names"] = np.array(unknown_class_names(near))
    cache["far_names"] = np.array(unknown_class_names(far))

    out_path = os.path.join(cache_dir, f"{run_name}.npz")
    np.savez_compressed(out_path, **cache)
    print(f"[done] cached outputs -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    extract_for_checkpoint(cfg, args.run_name)
