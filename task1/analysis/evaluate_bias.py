# =============================================================================
# analysis/evaluate_bias.py
# -----------------------------------------------------------------------------
# PURPOSE: all the SCORING logic — the metrics that turn raw predictions into
# the numbers the report needs:
#   * top-1 accuracy, macro-F1, mean maximum confidence  (clean baseline + every
#     intervention);
#   * prediction consistency (fraction of images whose predicted class is
#     unchanged after an intervention);
#   * shape-bias and coverage for the cue-conflict experiment.
#
# WHY SEPARATE FROM data/ AND models/:
#   The repo is deliberately staged as generate -> represent -> SCORE. Keeping
#   metrics here means the exact same predictions can be scored consistently for
#   every model and every intervention, and the definitions live in one place.
#
# LINKS TO OTHER FILES:
#   - Receives logits/predictions from models/backbones.py.
#   - Its shape-bias routine consumes the metadata from
#     data/make_cue_conflicts.py (which image has which shape/texture label).
#   - Called throughout scripts/run_task1.py.
# =============================================================================

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score      # macro-F1 implementation


# -----------------------------------------------------------------------------
# Core prediction utilities
# -----------------------------------------------------------------------------
def logits_to_pred_and_conf(logits: torch.Tensor):
    """Turn raw logits into (predicted_class, max_softmax_confidence).

    ML CONCEPT — SOFTMAX CONFIDENCE:
      Softmax converts logits into a probability distribution over classes. The
      MAXIMUM softmax probability is the model's confidence in its top choice.
      'Mean maximum confidence' (averaged over images) summarises how sure a
      model is on average — useful because a model can stay accurate but become
      less/more confident under an intervention, or (importantly) stay very
      confident even after patch-shuffling destroys the object (spec: "A
      confident prediction after patch shuffling is not necessarily a sensible
      prediction").
    """
    probs = F.softmax(logits, dim=1)          # logits -> probabilities per class
    conf, pred = probs.max(dim=1)             # top probability and its class idx
    return pred.cpu(), conf.cpu()


def top1_accuracy(pred, labels):
    """Fraction of images whose predicted class equals the ground-truth label."""
    pred = torch.as_tensor(pred)
    labels = torch.as_tensor(labels)
    return (pred == labels).float().mean().item()


def macro_f1(pred, labels):
    """Macro-averaged F1 across classes.

    ML CONCEPT — WHY MACRO-F1 (not just accuracy):
      Accuracy can look good while a model quietly fails on a minority class.
      Macro-F1 averages the per-class F1 (harmonic mean of precision & recall)
      giving every class EQUAL weight regardless of size, so it exposes uneven
      per-class performance. The spec asks for both top-1 and macro-F1 precisely
      because they can disagree.
    """
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    return f1_score(labels, pred, average="macro")


def mean_max_confidence(conf):
    """Average of the per-image maximum softmax probabilities."""
    return float(np.mean(np.asarray(conf)))


def prediction_consistency(pred_clean, pred_transformed):
    """Fraction of images whose predicted class is UNCHANGED by an intervention.

    ML CONCEPT — PREDICTION CONSISTENCY (spec definition):
      Consistency(delta) = (1/N) * sum 1[ y_hat(x_i) == y_hat(T(x_i)) ].
      It measures behavioural STABILITY of the decision under a controlled
      change, independent of correctness. A model can be consistent but wrong,
      or correct but inconsistent — consistency isolates 'did the intervention
      flip the decision', which is the question the interventions ask.
      NOTE: this compares to the CLEAN prediction, not the ground truth.
    """
    pred_clean = torch.as_tensor(pred_clean)
    pred_transformed = torch.as_tensor(pred_transformed)
    return (pred_clean == pred_transformed).float().mean().item()


# -----------------------------------------------------------------------------
# Bundled metric block used for the clean baseline and every intervention
# -----------------------------------------------------------------------------
def standard_metrics(logits, labels):
    """Compute the (top1, macro_f1, mean_max_conf) triple + return predictions.

    This is the exact metric bundle the spec asks to report for the clean
    baseline "top-1 accuracy, macro-F1, and mean maximum confidence" and reuse
    for every intervention.
    """
    pred, conf = logits_to_pred_and_conf(logits)
    return {
        "top1": top1_accuracy(pred, labels),
        "macro_f1": macro_f1(pred, labels),
        "mean_max_conf": mean_max_confidence(conf),
    }, pred, conf


# -----------------------------------------------------------------------------
# Shape-bias and coverage (cue-conflict experiment)
# -----------------------------------------------------------------------------
def shape_bias_and_coverage(preds, meta):
    """Classify each cue-conflict prediction and compute shape bias + coverage.

    Each prediction is one of three kinds (spec):
      * SHAPE   decision: predicted == the content/shape label;
      * TEXTURE decision: predicted == the style/texture label;
      * OTHER: predicted some third class (neither intended cue).

    Then (spec formulas):
      Shape Bias(%) = N_shape / (N_shape + N_texture) * 100
      Coverage(%)   = (N_shape + N_texture) / N_total  * 100

    ML CONCEPT — WHY BOTH NUMBERS MATTER:
      Shape bias only counts decisions that landed on ONE of the two intended
      classes; it ignores 'other'. So a model that answers 'other' 95% of the
      time could still post a high shape-bias from a handful of decisions —
      statistically fragile. COVERAGE reports what fraction of predictions were
      even interpretable as shape-or-texture, so it tells you how much to TRUST
      the shape-bias number (spec: "Shape bias must be interpreted with
      coverage").
    """
    preds = np.asarray(preds)
    n_shape = n_texture = n_other = 0
    for i, m in enumerate(meta):
        p = int(preds[i])
        if p == m["shape_label"]:
            n_shape += 1                    # model followed the SHAPE cue
        elif p == m["texture_label"]:
            n_texture += 1                  # model followed the TEXTURE cue
        else:
            n_other += 1                    # neither intended class
    n_total = len(meta)
    decisive = n_shape + n_texture          # decisions on an intended cue
    shape_bias = (100.0 * n_shape / decisive) if decisive > 0 else float("nan")
    coverage = (100.0 * decisive / n_total) if n_total > 0 else float("nan")
    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_other,
        "n_total": n_total,
        "shape_bias_pct": shape_bias,
        "coverage_pct": coverage,
    }


def collect_conflict_examples(preds, meta, class_names, k=6):
    """Return a few informative cue-conflict cases (agreement/disagreement/other).

    The spec's Required Evidence includes "A small set of informative
    cue-conflict agreements, disagreements, or failures with model predictions."
    We surface a handful of each category so they can be shown in the report.
    """
    preds = np.asarray(preds)
    shape_cases, texture_cases, other_cases = [], [], []
    for i, m in enumerate(meta):
        p = int(preds[i])
        rec = {
            "index": i,
            "shape": m["shape_name"],
            "texture": m["texture_name"],
            "predicted": class_names[p],
        }
        if p == m["shape_label"] and len(shape_cases) < k:
            shape_cases.append(rec)
        elif p == m["texture_label"] and len(texture_cases) < k:
            texture_cases.append(rec)
        elif len(other_cases) < k:
            other_cases.append(rec)
    return {"followed_shape": shape_cases,
            "followed_texture": texture_cases,
            "followed_other": other_cases}
