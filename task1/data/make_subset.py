

import os
import json                       # to save selected image identifiers
import numpy as np
import torch
from torchvision import datasets, transforms   # dataset loaders + PIL<->tensor
from torch.utils.data import Subset            # a view over a subset of indices

try:
    from utils import load_config, set_seed, ensure_dir
except ImportError:  # when imported as a package from the project root
    from ..utils import load_config, set_seed, ensure_dir



STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def _to_224_tensor(image_size: int):
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # force exact 224x224
        transforms.ToTensor(),                        # HWC uint8 -> CHW float[0,1]
    ])


def load_datasets(cfg: dict):
   
    name = cfg["dataset"]["name"]
    root = cfg["dataset"]["root"]
    tfm = _to_224_tensor(cfg["image_size"])

    if name == "stl10":
       
        train_ds = datasets.STL10(root, split="train", download=True, transform=tfm)
        test_ds  = datasets.STL10(root, split="test",  download=True, transform=tfm)
        class_names = STL10_CLASSES
    elif name == "oxford_pets":
        
        train_ds = datasets.OxfordIIITPet(root, split="trainval", download=True, transform=tfm)
        test_ds  = datasets.OxfordIIITPet(root, split="test",     download=True, transform=tfm)
        class_names = [c.lower() for c in train_ds.classes]
    else:
        raise ValueError(f"Unknown dataset '{name}' (use stl10 or oxford_pets)")

    return train_ds, test_ds, class_names


def _labels_of(ds) -> np.ndarray:
   
    for attr in ("labels", "_labels"):          # STL10 -> .labels, Pets -> ._labels
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    # Fallback: iterate the (label side of the) dataset. Slower but universal.
    return np.asarray([y for _, y in ds])


def stratified_train_val_split(train_ds, cfg: dict):
   
    set_seed(cfg["seed"])                        # determinism for the split
    labels = _labels_of(train_ds)                # per-sample class labels
    train_frac = cfg["split"]["train_frac"]      # 0.8
    rng = np.random.default_rng(cfg["seed"])     # a *local* generator, seeded

    train_idx, val_idx = [], []
    for c in np.unique(labels):                  # loop over each class id
        idx_c = np.where(labels == c)[0]         # indices of that class
        rng.shuffle(idx_c)                       # shuffle within the class
        n_train = int(round(train_frac * len(idx_c)))  # 80% of this class
        train_idx.extend(idx_c[:n_train].tolist())     # first 80% -> train
        val_idx.extend(idx_c[n_train:].tolist())       # last 20%  -> val

   
    return Subset(train_ds, train_idx), Subset(train_ds, val_idx)


def build_eval_subset(test_ds, cfg: dict):
   
    set_seed(cfg["seed"])
    labels = _labels_of(test_ds)
    total = cfg["eval_subset"]["size"]           # 500
    classes = np.unique(labels)
    per_class = total // len(classes)            # e.g. 500 / 10 = 50 per class
    rng = np.random.default_rng(cfg["seed"])

    selected, report = [], {}
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        take = min(per_class, len(idx_c))        # cap at what's available
        selected.extend(idx_c[:take].tolist())
        
        report[int(c)] = {"available": int(len(idx_c)),
                          "requested": int(per_class),
                          "taken": int(take)}

    selected = sorted(selected)                  # stable, deterministic order
    return Subset(test_ds, selected), selected, report


def save_identifiers(selected_indices, report, class_names, cfg: dict):
    
    out_dir = ensure_dir(cfg["paths"]["results_dir"])
    payload = {
        "seed": cfg["seed"],
        "dataset": cfg["dataset"]["name"],
        "num_selected": len(selected_indices),
        "selected_test_indices": selected_indices,   # the identifiers
        "class_names": class_names,
        "per_class_imbalance_report": report,        # documents any shortfall
    }
    path = os.path.join(out_dir, "eval_subset_identifiers.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path



if __name__ == "__main__":
    cfg = load_config()
    train_ds, test_ds, class_names = load_datasets(cfg)
    tr, va = stratified_train_val_split(train_ds, cfg)
    ev, sel, rep = build_eval_subset(test_ds, cfg)
    path = save_identifiers(sel, rep, class_names, cfg)
    print(f"train={len(tr)}  val={len(va)}  eval_subset={len(ev)}")
    print(f"classes={class_names}")
    print(f"identifiers saved -> {path}")
