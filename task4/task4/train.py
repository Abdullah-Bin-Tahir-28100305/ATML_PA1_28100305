

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils import load_config, set_seed, get_device, ensure_dir
from data.cifar10 import load_cifar10_train
from data.make_splits import build_or_load_split, make_train_val_subsets
from models.resnet_cifar import CifarResNet18
from methods.vanilla import Vanilla
from methods.gcsc import GCSC
from methods.proser import PROSER, ProserModel
from methods.rpl import RPL, RPLModel
from torch.utils.data import DataLoader


def build_method(cfg):
    name = cfg["method"]["name"]
    return {"vanilla": Vanilla, "gcsc": GCSC, "proser": PROSER, "rpl": RPL}[name](cfg)


@torch.no_grad()
def _val_accuracy(model, val_loader, device, method_name):
    
    model.eval()
    correct = total = 0
    for imgs, labels in val_loader:
        imgs = imgs.to(device); labels = labels.to(device)
        if method_name == "proser":
            logits, _ = model(imgs)                      
        elif method_name == "rpl":
            _, logits = model.distances(imgs)           
        else:
            logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


def _make_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def train_one(cfg, run_name=None, init_checkpoint=None):
    set_seed(cfg["seed"])
    device = get_device()
    num_classes = cfg["known"]["num_classes"]
    method = build_method(cfg)
    run_name = run_name or cfg["method"]["name"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints_dir"])
    print(f"[info] method={run_name} device={device}")

    use_ra = getattr(method, "uses_randaugment", False)
    train_aug = load_cifar10_train(cfg, train_transform=True, randaugment=use_ra)
    train_clean = load_cifar10_train(cfg, train_transform=False)
    split = build_or_load_split(cfg, train_clean)
    train_set, val_set = make_train_val_subsets(train_aug, train_clean, split)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, drop_last=True,
                              num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=cfg["train"]["num_workers"])

    backbone = CifarResNet18(num_classes=num_classes).to(device)
    mname = cfg["method"]["name"]
    if mname == "proser":
        model = ProserModel(backbone, num_classes, cfg["method"]["num_dummy"]).to(device)
        if init_checkpoint and os.path.exists(init_checkpoint):
            state = torch.load(init_checkpoint, map_location=device, weights_only=False)
            backbone.load_state_dict(state["model"])
            print(f"[info] PROSER initialized from {init_checkpoint}")
        else:
            print("[warn] PROSER init checkpoint not found; starting from scratch backbone.")
    elif mname == "rpl":
        model = RPLModel(backbone, num_classes).to(device)
    else:
        model = backbone

    if mname == "proser":
        ft = cfg["method"]["finetune"]
        epochs = ft["epochs"]
        optimizer = torch.optim.SGD(model.parameters(), lr=ft["lr"],
                                    momentum=ft["momentum"], weight_decay=ft["weight_decay"])
    else:
        epochs = cfg["train"]["epochs"]
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"],
                                    momentum=cfg["train"]["momentum"],
                                    weight_decay=cfg["train"]["weight_decay"])
    scheduler = _make_scheduler(optimizer, epochs)

    best_acc, best_state = -1.0, None
    history = {"epoch": [], "val_acc": [], "step_logs": []}
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            loss, logs = method.loss(model, imgs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            history["step_logs"].append({**logs, "epoch": epoch})
        scheduler.step()

        val_acc = _val_accuracy(model, val_loader, device, mname)
        history["epoch"].append(epoch); history["val_acc"].append(val_acc)
        print(f"  epoch {epoch:3d} | val_acc={val_acc:.4f}")
        if val_acc > best_acc:                           # select by val accuracy
            best_acc = val_acc
            best_state = {"model": copy.deepcopy(
                              (model.backbone if mname in ("proser", "rpl") else model).state_dict()),
                          "full_model": copy.deepcopy(model.state_dict()),
                          "epoch": epoch, "val_acc": val_acc,
                          "config": cfg, "run_name": run_name, "method": mname}

    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    torch.save(best_state, ckpt_path)
    with open(os.path.join(results_dir, f"history_{run_name}.json"), "w") as f:
        json.dump(_json_safe(history), f, indent=2)
    print(f"[done] best val_acc={best_acc:.4f} -> {ckpt_path}")
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
    parser.add_argument("--init_checkpoint", default=None,
                        help="for PROSER: the Vanilla checkpoint to fine-tune from")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_one(cfg, init_checkpoint=args.init_checkpoint)
