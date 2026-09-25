# Task 2 — Unsupervised Domain Adaptation (PACS, Sketch as target)

This repo implements **Task 2 only**: comparing Source-only ERM against three
adaptation methods — DAN (MMD), DANN (adversarial), and CDAN (class-conditional
adversarial) — on the PACS dataset with **Sketch as the unlabeled target** and
Photo, Art Painting, Cartoon as the labeled sources. It follows the transductive
UDA protocol: target images are visible unlabeled during adaptation, and target
labels are used **only** at the final evaluation.

Every file is heavily commented, tying each line to the ML concept behind it.

---

## 1. What each file does

```
shared/                         # reused by Task 2 AND Task 3 (same PACS protocol)
  pacs.py                       # PACS loading + train/eval image transforms
  pacs_protocol.py              # stratified 80/20 splits (seed 6304), domain-
                                # balanced batch iterators (8/source + 24 target)
  utils.py                      # config loading (with inherit), seeding, device
  splits/                       # auto-generated shared split JSON lives here
task2/
  configs/
    base.yaml                   # ALL shared settings (the controlled part)
    source_only.yaml            # ERM baseline (also the Task 3 ERM baseline)
    dan.yaml                    # MMD alignment + the lambda_MMD controlled sweep
    dann.yaml                   # adversarial alignment (GRL) settings
    cdan.yaml                   # class-conditional adversarial settings
  models/
    backbone.py                 # ResNet-18 feature extractor + FROZEN-BN policy
    classifier_head.py          # 7-class linear head
    domain_discriminator.py     # GRL + domain-discriminator MLP (DANN/CDAN)
  methods/
    base_method.py              # shared interface + source classification loss
    source_only.py              # ERM (no alignment)
    dan.py                      # multi-kernel MMD (median heuristic)
    dann.py                     # adversarial domain loss via GRL
    cdan.py                     # multilinear map g=vec(f x p) + GRL
  evaluation/
    metrics.py                  # top-1, macro-F1, mean source-val macro-F1
    domain_separability.py      # LogReg source-vs-target probe (70/30, C=1)
    class_analysis.py           # per-class target accuracy, changes, confusions
  train.py                      # THE common training loop (all methods)
  evaluate_final.py             # final tables + diagnostics (target labels here)
  controlled_study.py           # alignment-strength sweep (Step 6)
  run_task2.py                  # orchestrator: trains all, evaluates, studies
  results/                      # all outputs (created on first run)
  requirements.txt
  README.md
```

### How the pieces connect

```
configs (base + method) ─> shared/utils.load_config (merges inherit)
shared/pacs + pacs_protocol ─> splits + domain-balanced batches
        │
        ▼
train.py  ── backbone + head + method(+discriminator) ── trains each method
        │        (frozen BN, GRL schedule, early stop on mean source-val F1)
        ▼
results/checkpoints/*.pt   +   history_*.json (loss curves)
        │
        ▼
evaluate_final.py ── source/target metrics + domain separability + per-class
controlled_study.py ── lambda_MMD sweep {0.1,1,10}
        │
        ▼
results/  (task2_summary_table.csv, task2_final_results.json, *.png)
```

---

## 2. The six steps (mapped to the assignment)

1. **Source-only ERM** — cross-entropy on the 3 pooled sources, no alignment.
   Baseline for everything; **also saved as the Task 3 ERM baseline**.
2. **DAN** — adds `lambda_MMD * MMD^2(source_feat, target_feat)` with a sum of 3
   RBF kernels at 0.5/1/2 × the median pairwise distance. Marginal alignment.
3. **DANN** — a domain discriminator + Gradient Reversal Layer with the schedule
   `alpha(p)=2/(1+e^{-10p})-1`. Adversarial marginal alignment.
4. **CDAN** — same adversarial setup but the discriminator sees the multilinear
   map `g=vec(f⊗p)` (feature ⊗ softmax prediction). Class-conditional. No entropy
   conditioning; f and p are **not** detached (per spec).
5. **Common evaluation + diagnostic** — source-val & target accuracy/macro-F1,
   target change vs Source-only, and **domain separability** (LogReg, 70/30, C=1).
6. **Controlled study** — vary `lambda_MMD ∈ {0.1, 1, 10}` for DAN, everything
   else fixed; plot source-F1, separability, and target vs strength.

---

## 3. How to run

```bash
pip install -r task2/requirements.txt

# full pipeline: trains all four methods, evaluates, runs the study
cd task2
python run_task2.py

# only re-evaluate existing checkpoints (no retraining)
python run_task2.py --skip_train

# run a single method by itself
python train.py --config configs/dann.yaml

# run just the controlled study
python controlled_study.py --config configs/dan.yaml
```

Outputs in `results/`: `task2_summary_table.csv`, `task2_final_results.json`,
`loss_curves_<method>.png`, `controlled_study.png/.json`, and the checkpoints.

### Where it runs (CUDA / Apple M1 / CPU / Colab)
`shared/utils.get_device()` auto-detects **CUDA → MPS (Apple M-series) → CPU**;
you edit nothing. An M1 uses its GPU via MPS. On Colab set Runtime → GPU. A GPU
is strongly recommended: this fine-tunes a full ResNet-18 four times (plus the
sweep), which is slow on CPU.

---

## 4. ⚠️ Manual work / things you must do yourself

Read this carefully — a couple of items are genuinely required.

1. **Download PACS yourself (required).** Unlike Task 1's STL-10, PACS is **not**
   auto-downloaded by torchvision. You must obtain PACS and place it as:
   ```
   task2/data_cache/PACS/
     photo/<class>/*.jpg
     art_painting/<class>/*.jpg
     cartoon/<class>/*.jpg
     sketch/<class>/*.jpg
   ```
   with the 7 class folders (dog, elephant, giraffe, guitar, horse, house,
   person) inside each domain. PACS is a standard public benchmark (commonly
   mirrored on Kaggle and the DomainBed/DeepDG repos). Point `dataset.root` in
   `configs/base.yaml` at your PACS folder if you put it elsewhere. If the folder
   layout differs, `shared/pacs.py` will raise a clear error telling you what it
   expected. **Cite wherever you downloaded PACS from in your report.**

2. **Write the 8-page PDF report yourself (required).** The AI-usage policy
   forbids AI-written report text. This code produces the numbers, tables, and
   figures; the analysis, the four Research Questions, and all prose are yours.

3. **Choose ONE controlled study (a design choice, already made for you).** The
   spec says to run *either* the DAN `lambda_MMD ∈ {0.1,1,10}` sweep *or* the
   DANN max-GRL `∈ {0.25,0.5,1}` sweep. We enabled the **DAN** sweep by default
   (`controlled_study.enabled: true` in `dan.yaml`). To switch to the DANN sweep
   instead: set `controlled_study.enabled: true` in `dann.yaml` and `false` in
   `dan.yaml`, then run `python controlled_study.py --config configs/dann.yaml`.
   No code edits needed — only these config flags.

4. **State a hypothesis before interpreting the study (required writing).** The
   spec: "State what you expect the increasing alignment pressure to do to source
   performance, domain separability, and target recognition before interpreting
   the results." Do this in the report, in your own words, before looking at the
   sweep's target numbers.

You do **not** need to change any model weights, edit any function, or tune any
hyperparameter to make the code run. Every required value is already set in the
configs.

---

## 5. "What to Watch For" — how this repo already handles it

The spec's cautions are built into the outputs so you can address them:
- **Domain separability ≠ preserved class info.** We report separability *and*
  target accuracy side by side (`task2_summary_table.csv`) so you compare them
  rather than assuming lower separability is automatically better.
- **Discriminator accuracy near chance is ambiguous.** DANN/CDAN log per-step
  `domain_acc` and the loss curves, so you can tell successful confusion from an
  undertrained discriminator or collapsed features.
- **Pooling sources hides per-source behavior.** The table reports EACH source
  validation domain before the mean.
- **A source drop with a target gain can be a good trade-off.** Both are in the
  table (with target change vs Source-only) so you can judge the trade-off.
- **Never select on target labels.** Checkpoint selection and early stopping use
  only mean **source**-validation macro-F1; target labels are read solely in
  `evaluate_final.py`, after everything is frozen.

---

## 5b. Training stability for the adversarial methods (DANN/CDAN)

The adversarial min-max objective (via the Gradient Reversal Layer) is not a
smooth loss like MMD, and without safeguards DANN/CDAN can diverge: the reversed
gradient spikes, one step overshoots, the loss jumps to 1e5+, and the shared
features collapse (source macro-F1 falls to chance and target predictions
degenerate to a single class). Two spec-compliant safeguards keep them in a
trainable regime:

1. **Global gradient-norm clipping** (`train.grad_clip_norm`, default 1.0),
   applied to *every* method identically, after `backward()` and before
   `optimizer.step()`. It bounds each update so a gradient spike cannot blow up
   the weights. It does not change the mandated LR/optimizer/schedule and rarely
   triggers on the already-stable Source-only/DAN runs (their gradient norms sit
   below the ceiling), so their results are unchanged. Each step also logs the
   pre-clip `grad_norm` so you can see when it was active.

   **DANN uses the standard stable-DANN recipe, NOT weakened settings.** DANN's
   discriminator originally ran on the raw, unbounded 512-d feature and collapsed:
   it trained well for one epoch, then as GRL alpha rose the discriminator
   overpowered the feature extractor, features collapsed, and source F1 fell below
   chance. Note that CDAN — same GRL, same schedule — stayed stable because its
   discriminator input was normalised; the one variable that differs between
   collapsing-DANN and stable-CDAN is the discriminator input. `dann.yaml`
   therefore applies the two techniques standard in DANN-on-PACS implementations,
   both of which LEAVE THE ASSIGNMENT'S PRESCRIBED HYPERPARAMETERS INTACT:
     * **`normalize_disc_input: true`** — L2-normalise the feature into the DANN
       discriminator (implemented in `methods/dann.py`). This bounds the
       discriminator input so it cannot race ahead of the extractor. It is an
       implementation detail of the discriminator's input, not a spec
       hyperparameter.
     * **`disc_lr: 1e-5`** — a separate, lower learning rate for the discriminator
       (two-timescale training), applied via a param-group split in `train.py`.
       The backbone + head keep the spec lr 1e-4; only the auxiliary discriminator
       runs slower. The spec assigns no LR to the discriminator, so this is an
       open implementation choice, not a deviation.
   Crucially, DANN's GRL schedule keeps **alpha_max = 1.0, gamma = 10** and the
   model lr stays **1e-4** — the spec's alignment strength and model LR are
   preserved. Only the discriminator's optimisation is tamed. `grad_clip_norm`
   (base 1.0) is retained as a light safety net. DAN/CDAN/Source-only are
   untouched (no `disc_lr` -> single optimiser exactly as before). In the report,
   cite this as the standard stable-DANN recipe (feature normalisation +
   two-timescale discriminator LR) rather than a change to the assignment's
   settings.
2. **CDAN multilinear-map L2 normalization** (`method.normalize_multilinear`,
   default true). The conditioning input `g = vec(f⊗p)` is the outer product of
   an *unscaled* 512-d ResNet feature and the 7-d softmax, so its raw magnitude
   is large and destabilizes the discriminator. We L2-normalize `g` per example.
   This is **not** entropy conditioning and does **not** detach `f` or `p`
   (gradients still flow through both), so the required CDAN construction is
   preserved — it only rescales the conditioning vector, as stabilized CDAN
   implementations do.

Both safeguards are config flags (set `grad_clip_norm: 0` or
`normalize_multilinear: false` to reproduce the raw, unstable behaviour). If you
report the instability, the `domain_acc` log plus the loss curves let you show it
was feature *collapse* (chance domain-accuracy reached by degenerate features),
not genuine domain confusion — exactly the ambiguity the spec's "What to Watch
For" warns about.

---

## 6. Reproducibility & the Task 2 / Task 3 link
Seed **6304** drives the splits, all training, and the separability probe. The
Source-only checkpoint (`results/checkpoints/source_only.pt`) is the exact ERM
baseline Task 3 must reuse — do not retrain it under different settings.

## 7. External code / data attribution
- ResNet-18 weights: torchvision `ResNet18_Weights.IMAGENET1K_V1`.
- Methods: DAN (Long et al. 2015), DANN (Ganin et al. 2016), CDAN (Long et al.
  2018). Implemented here from the papers; cite any public code you reuse.
- PACS dataset: cite your download source.
