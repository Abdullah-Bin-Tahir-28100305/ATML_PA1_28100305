# =============================================================================
# scripts/run_task1.py
# -----------------------------------------------------------------------------
# THE ORCHESTRATOR. This single entry point runs all six experiments IN ORDER,
# exactly as the spec's "Steps" section lists them:
#   1. Clean Baseline
#   2. Color Bias        (grayscale + our additional color intervention)
#   3. Shape vs Texture  (cue conflict; shape bias + coverage)
#   4. Translation       (accuracy & consistency vs displacement)
#   5. Patch Structure   (4x4 shuffle)
#   6. Representation Analysis (cosine stability I_T + t-SNE/UMAP)
# and writes every required piece of evidence to results/.
#
# DESIGN PRINCIPLE (spec): "Keep transformation generation separate from
# evaluation so the exact same images can be reused across models." We build
# each intervention's image tensors ONCE, then loop the models over the SAME
# tensors. That guarantees ResNet/ViT/CLIP are compared on byte-identical inputs.
#
# HOW TO RUN:
#   cd task1
#   python scripts/run_task1.py           # runs everything, writes results/
#   python scripts/run_task1.py --quick   # small smoke test (fewer images)
#
# LINKS: this file imports and coordinates EVERY other module:
#   utils, data/make_subset, data/transforms, data/make_cue_conflicts,
#   models/backbones, analysis/{evaluate_bias, feature_similarity, representation}.
# =============================================================================

import os
# APPLE-SILICON (MPS) SAFETY NET: a few PyTorch ops don't yet have a native
# Metal kernel. Setting this env var BEFORE importing torch tells PyTorch to
# transparently run any such op on the CPU instead of crashing, so the whole
# pipeline works on an M1/M2/M3 Mac. It has no effect on CUDA or CPU machines,
# so it is safe to leave on everywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import argparse
import numpy as np
import torch

# Make the project root importable no matter where we launch from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import load_config, set_seed, get_device, ensure_dir
from data.make_subset import (load_datasets, stratified_train_val_split,
                              build_eval_subset, save_identifiers)
from data.transforms import (to_grayscale, hue_rotate, translate,
                             make_patch_permutation, patch_shuffle)
from data.make_cue_conflicts import generate_cue_conflicts
from models.backbones import build_backbones, train_linear_head, LinearHead
from analysis.evaluate_bias import (standard_metrics, prediction_consistency,
                                    shape_bias_and_coverage,
                                    collect_conflict_examples,
                                    logits_to_pred_and_conf)
from analysis.feature_similarity import cosine_stability
from analysis.representation import visualize_backbone


# -----------------------------------------------------------------------------
# Helper: stack a Subset/Dataset into one (images, labels) tensor pair.
# -----------------------------------------------------------------------------
def stack_dataset(ds):
    """Materialise a dataset into (images (N,3,224,224), labels (N,)) tensors.

    We do this once for the 500-image eval subset so every intervention operates
    on the SAME in-memory tensor. The subset is small, so keeping it in RAM is
    both fine and the cleanest way to guarantee identical inputs across models.
    """
    imgs, ys = [], []
    for x, y in ds:
        imgs.append(x)
        ys.append(int(y))
    return torch.stack(imgs), torch.tensor(ys)


def apply_transform_to_batch(images, fn):
    """Apply a per-image transform `fn` to every image in a batch tensor."""
    return torch.stack([fn(images[i]) for i in range(images.shape[0])])


# -----------------------------------------------------------------------------
# Helper: get predictions for one model on a batch of images.
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict(backbone, head, images, device, batch_size=128):
    """Return logits for a trained-head model. Batches to bound memory."""
    outs = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start:start + batch_size].to(device)
        feats = backbone.extract_features(batch)          # frozen features
        outs.append(head(feats).cpu())                    # linear-head logits
    return torch.cat(outs)


@torch.no_grad()
def predict_clip_zeroshot(clip_backbone, images, class_names, device, batch_size=128):
    """Return CLIP zero-shot logits over class prompts. Batches to bound memory."""
    outs = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start:start + batch_size].to(device)
        outs.append(clip_backbone.zero_shot_logits(batch, class_names).cpu())
    return torch.cat(outs)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: tiny subset, few conflicts")
    parser.add_argument("--decoder_weights", default=None,
                        help="optional path to pretrained AdaIN decoder weights "
                             "(naoto0804/pytorch-AdaIN decoder.pth)")
    parser.add_argument("--vgg_weights", default=None,
                        help="path to the matching AdaIN VGG encoder weights "
                             "(naoto0804/pytorch-AdaIN vgg_normalised.pth). REQUIRED "
                             "together with --decoder_weights for authentic AdaIN.")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])                     # global determinism
    device = get_device()
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    print(f"[info] device={device}  results -> {results_dir}")

    # -------------------------------------------------------------------------
    # DATA: official train/test, stratified split, and the 500-image eval subset
    # -------------------------------------------------------------------------
    train_ds, test_ds, class_names = load_datasets(cfg)
    train_view, val_view = stratified_train_val_split(train_ds, cfg)
    eval_subset, sel_idx, imbalance = build_eval_subset(test_ds, cfg)
    save_identifiers(sel_idx, imbalance, class_names, cfg)   # reproducibility file

    # Materialise the eval subset once; every intervention reuses these tensors.
    clean_images, clean_labels = stack_dataset(eval_subset)
    if args.quick:                                  # smoke test: keep it tiny
        clean_images, clean_labels = clean_images[:40], clean_labels[:40]
    print(f"[info] eval subset: {clean_images.shape[0]} images, "
          f"{len(class_names)} classes")

    # -------------------------------------------------------------------------
    # MODELS: build backbones and train one linear head per (trained) backbone
    # -------------------------------------------------------------------------
    backbones = build_backbones(cfg)               # dict name -> Backbone
    # Move every backbone to the compute device ONCE. train_linear_head also does
    # this, but we do it up front so feature extraction in later steps (which may
    # run before/after training in any order) always has the backbone on-device.
    for bb in backbones.values():
        bb.to(device)
    heads, head_info = {}, {}
    for name, bb in backbones.items():
        print(f"[info] training linear head for {name} ...")
        head, info = train_linear_head(bb, train_view, val_view, cfg)
        heads[name] = head
        head_info[name] = info                      # best val acc + history

    # Master results container; serialized to JSON at the end.
    results = {"config": cfg, "class_names": class_names,
               "eval_subset_size": int(clean_images.shape[0]),
               "imbalance_report": imbalance,
               "head_info": {k: v["best_val_acc"] for k, v in head_info.items()},
               "experiments": {}}

    # Convenience: get logits for ALL model 'heads' (trained heads + CLIP z-shot)
    # on a given image tensor. Returns dict head_name -> logits.
    def all_model_logits(images):
        out = {}
        for name, bb in backbones.items():
            out[name] = predict(bb, heads[name], images, device)         # trained head
        if "clip_vitb32" in backbones:
            out["clip_zeroshot"] = predict_clip_zeroshot(                 # zero-shot
                backbones["clip_vitb32"], images, class_names, device)
        return out

    # =========================================================================
    # STEP 1 — CLEAN BASELINE
    # -------------------------------------------------------------------------
    # Establishes the common reference. Every later intervention is compared both
    # in ABSOLUTE terms and RELATIVE to this baseline (spec).
    # =========================================================================
    print("[step 1] clean baseline")
    clean_logits = all_model_logits(clean_images)
    clean_preds = {}
    baseline = {}
    for hname, logits in clean_logits.items():
        metrics, pred, conf = standard_metrics(logits, clean_labels)
        baseline[hname] = metrics
        clean_preds[hname] = pred                    # cached for consistency later
    results["experiments"]["clean_baseline"] = baseline

    # =========================================================================
    # STEP 2 — COLOR BIAS  (grayscale required + hue rotation additional)
    # -------------------------------------------------------------------------
    # Changes color while preserving geometry. We report accuracy change and
    # prediction consistency vs the clean images for each color transform.
    # =========================================================================
    print("[step 2] color bias (grayscale + hue rotation)")
    color_results = {}
    color_transforms = {
        "grayscale": lambda im: to_grayscale(im),                     # required
        cfg["color"]["additional"]:                                   # our choice
            lambda im: hue_rotate(im, cfg["color"]["hue_degrees"]),
    }
    for tname, fn in color_transforms.items():
        t_images = apply_transform_to_batch(clean_images, fn)
        t_logits = all_model_logits(t_images)
        block = {}
        for hname, logits in t_logits.items():
            metrics, pred, conf = standard_metrics(logits, clean_labels)
            block[hname] = {
                **metrics,                                    # absolute metrics
                "acc_change_vs_clean": metrics["top1"] - baseline[hname]["top1"],
                "consistency_vs_clean": prediction_consistency(clean_preds[hname], pred),
            }
        color_results[tname] = block
    results["experiments"]["color_bias"] = color_results

    # =========================================================================
    # STEP 3 — SHAPE vs TEXTURE  (cue conflict; shape bias + coverage)
    # -------------------------------------------------------------------------
    # Generate the conflict set ONCE (with accept/reject log), then classify
    # each prediction as shape/texture/other and compute the two metrics.
    # =========================================================================
    print("[step 3] shape vs texture (cue conflict)")
    if args.quick:
        cfg["cue_conflict"]["target_valid"] = 24     # keep smoke test fast
    conflict = generate_cue_conflicts(cfg, args.decoder_weights, args.vgg_weights)
    conflict_images = conflict["images"]
    conflict_meta = conflict["meta"]
    shape_results = {"generation_log": conflict["log"], "models": {}}
    if conflict_images.numel() > 0:
        cc_logits = all_model_logits(conflict_images)
        for hname, logits in cc_logits.items():
            pred, conf = logits_to_pred_and_conf(logits)
            sb = shape_bias_and_coverage(pred.numpy(), conflict_meta)
            examples = collect_conflict_examples(pred.numpy(), conflict_meta, class_names)
            shape_results["models"][hname] = {**sb, "examples": examples}
    results["experiments"]["shape_vs_texture"] = shape_results

    # =========================================================================
    # STEP 4 — TRANSLATION  (accuracy & consistency vs displacement)
    # -------------------------------------------------------------------------
    # For each shift in {0,8,16,32} we translate in all four cardinal directions,
    # evaluate each, and AVERAGE the metrics across directions (spec).
    # =========================================================================
    print("[step 4] translation")
    shifts = cfg["translation"]["shifts"]
    dirs = cfg["translation"]["directions"]
    translation_results = {hname: {"shift": [], "acc": [], "consistency": []}
                           for hname in clean_logits.keys()}
    for s in shifts:
        # Build the four directional versions for this shift.
        dir_images = {d: apply_transform_to_batch(clean_images, lambda im, d=d: translate(im, s, d))
                      for d in dirs}
        # Per model, average accuracy and consistency across the four directions.
        per_model_acc = {h: [] for h in clean_logits.keys()}
        per_model_cons = {h: [] for h in clean_logits.keys()}
        for d in dirs:
            d_logits = all_model_logits(dir_images[d])
            for hname, logits in d_logits.items():
                metrics, pred, conf = standard_metrics(logits, clean_labels)
                per_model_acc[hname].append(metrics["top1"])
                per_model_cons[hname].append(
                    prediction_consistency(clean_preds[hname], pred))
        for hname in clean_logits.keys():
            translation_results[hname]["shift"].append(s)
            translation_results[hname]["acc"].append(float(np.mean(per_model_acc[hname])))
            translation_results[hname]["consistency"].append(float(np.mean(per_model_cons[hname])))
    results["experiments"]["translation"] = translation_results
    _plot_translation(translation_results, results_dir)

    # =========================================================================
    # STEP 5 — PATCH STRUCTURE  (4x4 shuffle, fixed permutation, reused)
    # -------------------------------------------------------------------------
    # One non-identity permutation from seed 6304, reused across models. We
    # report accuracy drop and prediction consistency vs clean.
    # =========================================================================
    print("[step 5] patch shuffle")
    perm = make_patch_permutation(cfg["patch_shuffle"]["grid"], cfg["seed"])
    shuffled_images = apply_transform_to_batch(
        clean_images, lambda im: patch_shuffle(im, cfg["patch_shuffle"]["grid"], perm))
    patch_results = {"permutation": perm.tolist(), "models": {}}
    ps_logits = all_model_logits(shuffled_images)
    for hname, logits in ps_logits.items():
        metrics, pred, conf = standard_metrics(logits, clean_labels)
        patch_results["models"][hname] = {
            **metrics,
            "acc_drop_vs_clean": baseline[hname]["top1"] - metrics["top1"],
            "consistency_vs_clean": prediction_consistency(clean_preds[hname], pred),
        }
    results["experiments"]["patch_shuffle"] = patch_results

    # =========================================================================
    # STEP 6 — REPRESENTATION ANALYSIS  (cosine stability + projection plots)
    # -------------------------------------------------------------------------
    # For grayscale, cue conflict, translation (we use shift=16), and patch
    # shuffle, measure I_T (cosine stability) and make a t-SNE/UMAP plot.
    # NOTE: I_T is computed on the BACKBONE features, so CLIP appears ONCE here
    # (its single image encoder), not split into head/zero-shot.
    # =========================================================================
    print("[step 6] representation analysis")
    # Prepare paired (clean, transformed) image sets for each required transform.
    gray_images = apply_transform_to_batch(clean_images, to_grayscale)
    trans16_images = apply_transform_to_batch(
        clean_images, lambda im: translate(im, 16, "right"))
    # The spec lists FOUR transforms for representation analysis: grayscale,
    # cue conflict, translation, and patch shuffling. Each entry is a triple
    # (clean_set, transformed_set, labels) where the two image sets are aligned
    # 1-to-1 (image i in clean pairs with image i in transformed).
    stability_transforms = {
        "grayscale":     (clean_images, gray_images, clean_labels),
        "translation16": (clean_images, trans16_images, clean_labels),
        "patch_shuffle": (clean_images, shuffled_images, clean_labels),
    }
    # Cue conflict pairs each STYLIZED image with its CONTENT (shape-source)
    # image — same geometry, different texture — which is the natural clean
    # counterpart. The generator stored these aligned content images for us.
    if conflict_images.numel() > 0 and conflict["content_images"].numel() > 0:
        cc_labels = torch.tensor([m["shape_label"] for m in conflict_meta])
        stability_transforms["cue_conflict"] = (
            conflict["content_images"], conflict_images, cc_labels)

    rep_results = {}
    for tname, (clean_set, trans_set, labs) in stability_transforms.items():
        rep_results[tname] = {}
        for name, bb in backbones.items():
            it = cosine_stability(bb, clean_set, trans_set)
            rep_results[tname][name] = it
            # One projection plot per backbone per transform (spec: per-backbone).
            visualize_backbone(bb, clean_set, trans_set, labs, class_names,
                               cfg, tname, results_dir)
    results["experiments"]["representation_stability"] = rep_results

    # -------------------------------------------------------------------------
    # SAVE everything (numbers as JSON, plus a compact human-readable summary).
    # -------------------------------------------------------------------------
    with open(os.path.join(results_dir, "task1_results.json"), "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    _write_summary_tables(results, results_dir)
    print(f"[done] all results written to {results_dir}")


# -----------------------------------------------------------------------------
# Small plotting + serialization helpers
# -----------------------------------------------------------------------------
def _plot_translation(translation_results, out_dir):
    """Plot accuracy and consistency vs displacement (required evidence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ["acc", "consistency"]:
        plt.figure(figsize=(7, 5))
        for hname, data in translation_results.items():
            plt.plot(data["shift"], data[metric], marker="o", label=hname)
        plt.xlabel("displacement (pixels), averaged over 4 directions")
        plt.ylabel(metric)
        plt.title(f"Translation: {metric} vs displacement")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"translation_{metric}.png"), dpi=150)
        plt.close()


def _json_safe(obj):
    """Recursively convert numpy/torch types to plain Python for JSON dumping."""
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
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


def _write_summary_tables(results, out_dir):
    """Write a compact CSV comparing clean/grayscale/additional-color/patch-shuffle.

    This is the spec's first Required-Evidence item: "A compact comparison of
    clean, grayscale, additional-color, and patch-shuffle performance."
    """
    import csv
    exp = results["experiments"]
    add_color = results["config"]["color"]["additional"]
    rows = [["model", "clean_top1", "grayscale_top1",
             f"{add_color}_top1", "patch_shuffle_top1"]]
    for hname in exp["clean_baseline"].keys():
        clean = exp["clean_baseline"][hname]["top1"]
        gray = exp["color_bias"]["grayscale"][hname]["top1"]
        addc = exp["color_bias"][add_color][hname]["top1"]
        patch = exp["patch_shuffle"]["models"][hname]["top1"]
        rows.append([hname, f"{clean:.4f}", f"{gray:.4f}",
                     f"{addc:.4f}", f"{patch:.4f}"])
    with open(os.path.join(out_dir, "summary_compact_comparison.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
