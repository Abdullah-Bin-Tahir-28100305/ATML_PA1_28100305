# =============================================================================
# task4/evaluate_osr.py
# -----------------------------------------------------------------------------
# THE EVALUATION STAGE. From the cached outputs it produces every Required
# Evidence item:
#   (1) Table A: MSP / MLS / Energy / Mahalanobis on the FROZEN VANILLA model —
#       near/far/all-unknown AUROC + validation-calibrated rejection (FPR@95TPR).
#   (2) Table B: Vanilla / GCSC / PROSER using CSA + near/far OSR with MLS as the
#       common score, plus a second PROSER row using its placeholder-based score.
#   (3) A compact score-distribution / ROC figure for MSP, MLS, Mahalanobis.
#   (4) Failure analysis: >=3 near + >=3 far unknowns wrongly accepted under the
#       Vanilla MLS threshold, with class/prediction/score/threshold.
#   (Optional) an RPL row if its cache is present.
#
# KEY PROTOCOL (spec, enforced here):
#   * threshold tau = 95th percentile of unknownness on the KNOWN VAL set;
#     accept when u(x) <= tau (uses known data only).
#   * All four scores read the SAME cached logits/features.
#   * CSA uses ONLY the 10 known logits (for PROSER too), keeping closed-set
#     classification and rejection distinct.
#
# LINKS: scores/*, evaluation/{metrics,thresholds,failure_analysis}, methods/*.
# =============================================================================

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import json
import argparse
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils import load_config, ensure_dir
from data.cifar10 import CIFAR10_CLASSES
from scores import msp, mls, energy, mahalanobis
from methods.proser import proser_detection_score
from methods.rpl import rpl_unknownness
from evaluation.metrics import (closed_set_accuracy, auroc_known_vs_unknown,
                                acceptance_rejection_rates)
from evaluation.thresholds import calibrate_threshold
from evaluation.failure_analysis import collect_accepted_unknown_failures


def _load_cache(cfg, run_name):
    path = os.path.join(cfg["paths"]["cache_dir"], f"{run_name}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def _score_all_splits(cache, score_fn, needs_features=False, maha_params=None):
    """Apply a score function to val/test/near/far, returning a dict of arrays.

    Uses the SAME cached logits (and features for Mahalanobis) for every split so
    all scores operate on identical examples.
    """
    out = {}
    for split in ["val", "test", "near", "far"]:
        logits = cache[f"{split}_logits"]
        feats = cache[f"{split}_features"] if needs_features else None
        if maha_params is not None:
            out[split] = mahalanobis.score(logits, features=feats, params=maha_params)
        else:
            out[split] = score_fn(logits, features=feats)
    return out


def _evaluate_score(cache, scores_by_split, cfg):
    """Given per-split unknownness scores, compute AUROC + calibrated rejection.

    Returns a dict with near/far/all AUROC and the operating-point metrics at the
    validation-calibrated threshold (accept when u <= tau).
    """
    tau = calibrate_threshold(scores_by_split["val"], cfg["evaluation"]["tpr_target"])
    known_test = scores_by_split["test"]; near = scores_by_split["near"]; far = scores_by_split["far"]
    allu = np.concatenate([near, far])

    res = {
        "auroc_near": auroc_known_vs_unknown(known_test, near),
        "auroc_far":  auroc_known_vs_unknown(known_test, far),
        "auroc_all":  auroc_known_vs_unknown(known_test, allu),
        "threshold_tau": tau,
    }
    res.update({"near_" + k: v for k, v in
                acceptance_rejection_rates(known_test, near, tau).items()})
    res.update({"far_" + k: v for k, v in
                acceptance_rejection_rates(known_test, far, tau).items()})
    return res


def evaluate_all(cfg, include_rpl=False):
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    num_classes = cfg["known"]["num_classes"]

    # =====================================================================
    # TABLE A: four post-hoc scores on the FROZEN VANILLA model.
    # =====================================================================
    vanilla = _load_cache(cfg, "vanilla")
    if vanilla is None:
        raise FileNotFoundError("Missing cache for 'vanilla'; run extract_outputs first.")

    # Fit Mahalanobis on UNAUGMENTED CIFAR-10 TRAIN features (spec).
    maha_params = mahalanobis.fit_mahalanobis(
        vanilla["train_features"], vanilla["train_labels"], num_classes,
        eps=cfg["evaluation"]["mahalanobis_eps"])

    score_specs = {
        "MSP":         (msp.score, False, None),
        "MLS":         (mls.score, False, None),
        "Energy":      (energy.score, False, None),
        "Mahalanobis": (None, True, maha_params),
    }
    table_a = {}
    dist_cache = {}                                     # keep scores for the figure
    for sname, (fn, needs_feat, mp) in score_specs.items():
        s = _score_all_splits(vanilla, fn, needs_features=needs_feat, maha_params=mp)
        table_a[sname] = _evaluate_score(vanilla, s, cfg)
        dist_cache[sname] = s
    csa_vanilla = closed_set_accuracy(vanilla["test_logits"], vanilla["test_labels"])

    # =====================================================================
    # TABLE B: Vanilla / GCSC / PROSER with MLS (+ PROSER placeholder score).
    # =====================================================================
    table_b = {}
    # Vanilla with MLS
    s_v = _score_all_splits(vanilla, mls.score)
    table_b["Vanilla (MLS)"] = {"csa": csa_vanilla, **_evaluate_score(vanilla, s_v, cfg)}

    # GCSC with MLS
    gcsc = _load_cache(cfg, "gcsc")
    if gcsc is not None:
        s_g = _score_all_splits(gcsc, mls.score)
        table_b["GCSC (MLS)"] = {
            "csa": closed_set_accuracy(gcsc["test_logits"], gcsc["test_labels"]),
            **_evaluate_score(gcsc, s_g, cfg)}

    # PROSER with MLS on the 10 known logits (CSA from known logits only, spec).
    proser = _load_cache(cfg, "proser")
    if proser is not None:
        s_p_mls = _score_all_splits(proser, mls.score)
        table_b["PROSER (MLS)"] = {
            "csa": closed_set_accuracy(proser["test_logits"], proser["test_labels"]),
            **_evaluate_score(proser, s_p_mls, cfg)}
        # PROSER placeholder-based detection score (dummy vs known).
        s_p_ph = {sp: proser_detection_score(proser[f"{sp}_logits"], proser[f"{sp}_dummy"])
                  for sp in ["val", "test", "near", "far"]}
        table_b["PROSER (placeholder)"] = {
            "csa": closed_set_accuracy(proser["test_logits"], proser["test_labels"]),
            **_evaluate_score(proser, s_p_ph, cfg)}

    # Optional RPL row
    if include_rpl:
        rpl = _load_cache(cfg, "rpl")
        if rpl is not None:
            s_r = {sp: rpl_unknownness(rpl[f"{sp}_dist"]) for sp in ["val","test","near","far"]}
            # CSA for RPL: argmax of distance-logits (spec's known-class score).
            table_b["RPL"] = {
                "csa": closed_set_accuracy(rpl["test_logits"], rpl["test_labels"]),
                **_evaluate_score(rpl, s_r, cfg)}

    # =====================================================================
    # FIGURE: score distributions for MSP, MLS, Mahalanobis (compact multi-panel).
    # =====================================================================
    _plot_score_distributions(dist_cache, ["MSP", "MLS", "Mahalanobis"], results_dir)

    # =====================================================================
    # FAILURE ANALYSIS: Vanilla MLS threshold; >=3 near + >=3 far accepted unknowns.
    # =====================================================================
    tau_mls = calibrate_threshold(s_v["val"], cfg["evaluation"]["tpr_target"])
    failures = {
        "near": collect_accepted_unknown_failures(
            vanilla["near_logits"], s_v["near"], tau_mls,
            list(vanilla["near_names"]), CIFAR10_CLASSES, k=3),
        "far": collect_accepted_unknown_failures(
            vanilla["far_logits"], s_v["far"], tau_mls,
            list(vanilla["far_names"]), CIFAR10_CLASSES, k=3),
        "vanilla_mls_threshold": tau_mls,
    }

    out = {"table_A_posthoc_scores_vanilla": table_a,
           "vanilla_csa": csa_vanilla,
           "table_B_model_comparison_mls": table_b,
           "failure_analysis": failures}
    with open(os.path.join(results_dir, "task4_results.json"), "w") as f:
        json.dump(_json_safe(out), f, indent=2)
    _write_csvs(table_a, table_b, results_dir)
    print(f"[done] Task 4 evaluation written to {results_dir}")
    return out


def _plot_score_distributions(dist_cache, score_names, results_dir):
    """Multi-panel histogram of known-test vs near vs far unknownness scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(score_names), figsize=(5 * len(score_names), 4))
    if len(score_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, score_names):
        s = dist_cache[name]
        ax.hist(s["test"], bins=40, alpha=0.5, label="known (test)", density=True)
        ax.hist(s["near"], bins=40, alpha=0.5, label="near unknown", density=True)
        ax.hist(s["far"], bins=40, alpha=0.5, label="far unknown", density=True)
        ax.set_title(f"{name} unknownness"); ax.set_xlabel("u(x)"); ax.set_ylabel("density")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_distributions.png"), dpi=150)
    plt.close()


def _write_csvs(table_a, table_b, results_dir):
    import csv
    # Table A
    with open(os.path.join(results_dir, "task4_table_A_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "auroc_near", "auroc_far", "auroc_all",
                    "near_fpr@95tpr", "far_fpr@95tpr",
                    "near_reject_rate", "far_reject_rate"])
        for name, r in table_a.items():
            w.writerow([name, f"{r['auroc_near']:.4f}", f"{r['auroc_far']:.4f}",
                        f"{r['auroc_all']:.4f}", f"{r['near_fpr_at_95tpr']:.4f}",
                        f"{r['far_fpr_at_95tpr']:.4f}", f"{r['near_unknown_rejection_rate']:.4f}",
                        f"{r['far_unknown_rejection_rate']:.4f}"])
    # Table B
    with open(os.path.join(results_dir, "task4_table_B_models.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_score", "csa", "auroc_near", "auroc_far", "auroc_all",
                    "near_fpr@95tpr", "far_fpr@95tpr"])
        for name, r in table_b.items():
            w.writerow([name, f"{r['csa']:.4f}", f"{r['auroc_near']:.4f}",
                        f"{r['auroc_far']:.4f}", f"{r['auroc_all']:.4f}",
                        f"{r['near_fpr_at_95tpr']:.4f}", f"{r['far_fpr_at_95tpr']:.4f}"])


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
    parser.add_argument("--include_rpl", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_all(cfg, include_rpl=args.include_rpl)
