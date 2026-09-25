# Task 4 — Open-Set Recognition (CIFAR-10 known, CIFAR-100 unknown)

This repo implements **Task 4 only**: training a closed-set CIFAR-10 classifier
and equipping it to **reject unknowns** drawn from held-out CIFAR-100 classes.
It compares four post-hoc novelty scores (MSP, MLS, Energy, Mahalanobis) on a
frozen Vanilla model, then three trained models (Vanilla, GCSC, PROSER) under a
common score, plus an optional RPL row.

Every file is heavily commented, tying each line to the ML concept behind it.

---

## 1. What each file does

```
task4/
  configs/
    base.yaml               # ALL shared settings (recipe, data, eval protocol)
    vanilla.yaml            # closed-set baseline (base aug)
    gcsc.yaml               # Vanilla + RandAugment(num_ops=2, magnitude=9)
    proser.yaml             # dummy classifiers + placeholder losses
    rpl.yaml                # optional: reciprocal point learning
  data/
    cifar10.py              # known data + transforms (+ optional RandAugment)
    make_splits.py          # stratified 90/10 train/val split (seed 6304)
    cifar100_unknowns.py    # near/far CIFAR-100 unknown groups (eval-only)
  models/
    resnet_cifar.py         # CIFAR ResNet-18 (3x3 stem, no maxpool) + mixup hooks
  scores/
    msp.py  mls.py  energy.py  mahalanobis.py   # the 4 post-hoc novelty scores
  methods/
    vanilla.py  gcsc.py     # cross-entropy training (GCSC flips RandAugment on)
    manifold_mixup.py       # proxy-unknown generator for PROSER
    proser.py               # classifier + data placeholder losses + detect score
    rpl.py                  # optional reciprocal point learning
  cache/                    # saved logits/features (.npz) — created on run
  evaluation/
    metrics.py              # AUROC, FPR@95TPR, acceptance/rejection, CSA
    thresholds.py           # 95th-percentile validation-calibrated threshold
    failure_analysis.py     # accepted-unknown failure cases
  utils.py                  # config (inherit), seeding, device
  train.py                  # train Vanilla/GCSC/PROSER/RPL (select by val acc)
  extract_outputs.py        # cache logits/features for every eval set (once)
  evaluate_osr.py           # Tables A & B, figure, failure analysis
  run_task4.py              # orchestrator: train -> extract -> evaluate
  results/                  # outputs (created on run)
  requirements.txt
  README.md
```

### How the pieces connect
```
configs ─> utils.load_config
data/ (cifar10 + make_splits + cifar100_unknowns)
        │
        ▼
train.py ── CIFAR ResNet-18 + method ── trains Vanilla/GCSC/PROSER/RPL
        │     (select checkpoint by CIFAR-10 val accuracy; no unknowns)
        ▼
results/checkpoints/*.pt
        │
        ▼
extract_outputs.py ── run each fixed model once ── cache/<model>.npz
        │              (train/val/test knowns + near/far unknowns)
        ▼
evaluate_osr.py ── scores + AUROC + calibrated rejection + figure + failures
        │
        ▼
results/ (task4_results.json, task4_table_A_scores.csv,
          task4_table_B_models.csv, score_distributions.png)
```

---

## 2. The steps (mapped to the assignment)

1. **Vanilla** — 10-class ResNet-18, cross-entropy; frozen and used for the
   post-hoc scores; also the PROSER init.
2. **Post-hoc scores** (unknownness u(x), larger = more novel):
   - `u_MSP = 1 - max_k softmax(z)_k`  (confidence)
   - `u_MLS = - max_k z_k`  (logit magnitude; Vaze et al. 2022)
   - `u_Energy = - logsumexp_k z_k`  (all-logit evidence; Liu et al. 2020)
   - `u_Mah = min_c (f-mu_c)^T Σ⁻¹ (f-mu_c)`  (feature distance; shared diagonal
     Σ from unaugmented train features, +1e-6 on the diagonal).
3. **GCSC** — Vanilla + RandAugment; evaluated with MLS (controlled test of
   whether stronger augmentation helps rejection, esp. near unknowns).
4. **PROSER** — append 5 dummy classifiers; classifier-placeholder loss (β=1) +
   data-placeholder loss (γ=0.1) via manifold mixup after layer2/before layer3
   (λ~Beta(2,2)); each mini-batch split in half. Reported with MLS on the 10
   known logits AND with the placeholder-based detection score.
5. **RPL** (optional) — learns per-class reciprocal points + open-space reg.
6. **Evaluation** — AUROC for Known-vs-Near / Far / All; threshold = 95th
   percentile of unknownness on the CIFAR-10 val set (accept when u ≤ τ);
   report CSA, acceptance/rejection rates, and FPR@95TPR; plus the figure and
   failure cases.

---

## 3. How to run
```bash
pip install -r task4/requirements.txt
cd task4

python run_task4.py                 # vanilla + gcsc + proser, then evaluate
python run_task4.py --include_rpl   # also train + evaluate the optional RPL row
python run_task4.py --skip_train    # only re-extract + re-evaluate saved ckpts

# single stages
python train.py --config configs/vanilla.yaml
python train.py --config configs/proser.yaml --init_checkpoint results/checkpoints/vanilla.pt
python extract_outputs.py --config configs/vanilla.yaml --run_name vanilla
python evaluate_osr.py
```
Outputs in `results/`: `task4_results.json`, `task4_table_A_scores.csv`,
`task4_table_B_models.csv`, `score_distributions.png`, and the checkpoints.

### Where it runs (CUDA / Apple M1 / CPU / Colab)
`utils.get_device()` auto-detects **CUDA → MPS → CPU**; you edit nothing.
CIFAR-10/100 auto-download via torchvision (no manual dataset step — unlike
Tasks 2/3). A GPU is strongly recommended: this trains three ResNet-18s for 100
epochs each (plus 50 for PROSER). On Colab set Runtime → GPU.

---

## 4. ⚠️ Manual work / things you must do yourself

1. **Write the 8-page PDF report yourself (required).** AI-usage policy forbids
   AI-written report prose. This code produces the numbers, tables, and figure.
2. **Nothing else is strictly required to run it.** CIFAR-10/100 download
   automatically; every hyperparameter is set in the configs; unknowns are
   evaluation-only and handled for you.
3. **Optional choices you may defend in the report:**
   - Run RPL or not (`--include_rpl`). It's the optional extension.
   - The near/far unknown groups are fixed by the assignment and already encoded
     in `base.yaml` — do NOT change them (spec: the grouping may not be revised).
4. **(If you want faster iteration)** reduce `train.epochs` in `base.yaml` for a
   smoke test — but the graded run should use the specified 100 epochs (50 for
   PROSER). Reducing epochs is a config change, not a code edit.

You do **not** need to edit any model weights, function, or the scores to run it.

---

## 5. "What to Watch For" — how this repo handles it
- **Near vs far are different difficulty.** Every metric is reported separately
  for near and far (AUROC + FPR@95TPR), so you never conflate them.
- **CIFAR-100 is evaluation-only.** Unknowns are loaded solely in
  `extract_outputs.py`/`evaluate_osr.py`; training/selection/threshold use known
  data only.
- **Confidence vs logit magnitude vs aggregate evidence vs feature distance are
  different signals.** All four scores are computed from the SAME cached outputs,
  so you can compare exactly where they agree/disagree.
- **Better closed-set accuracy ≠ better rejection.** Table B reports CSA next to
  near/far OSR so you can see whether they move together (GCSC question).
- **Proxy unknowns ≠ real unknowns.** PROSER's mixup points are interpolations;
  the README/comments flag that they may not cover every direction an unknown can
  come from (spec caution) — visible in the near-vs-far PROSER gap.
- **AUROC vs one-operating-point rejection.** Both are reported (AUROC = ranking
  over all thresholds; FPR@95TPR = behavior at the calibrated τ).

## 6. Reproducibility
Seed **6304** drives the 90/10 split, all training, and the deterministic
unknown-group subsampling. Post-hoc scores are deterministic functions of the
cached outputs. Cite Vaze et al. (2022), Zhou et al. (2021), Liu et al. (2020),
and (if used) Chen et al. (2020) in your report.
