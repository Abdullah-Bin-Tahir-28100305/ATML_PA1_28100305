# Task 3 — Domain Generalization (PACS, Sketch as the UNSEEN target)

This repo implements **Task 3 only**: comparing ERM against two domain-
generalization strategies — **DAN-DG** (align the observed source domains via
pairwise MMD) and **SAM** (seek flat, locally stable minima) — on PACS with
Photo, Art Painting, Cartoon as labeled sources and **Sketch as an unseen
target that is loaded only at final evaluation**.

Every file is heavily commented, tying each line to the ML concept behind it.

---

## 1. What each file does

```
shared/                         # SAME PACS protocol as Task 2 (bundled here so
  pacs.py                       #   Task 3 runs standalone). Loading, transforms.
  pacs_protocol.py              # stratified 80/20 splits (seed 6304), domain-
                                #   balanced source batching, infinite loaders.
  utils.py                      # config loading (inherit), seeding, device.
  splits/                       # auto-generated shared split JSON lives here.
task3/
  configs/
    base.yaml                   # ALL shared settings (mirrors Task 2's protocol)
    erm.yaml                    # ERM baseline (reuses Task 2's Source-only model)
    dan_dg.yaml                 # pairwise source MMD + the lambda_DG sweep
    sam.yaml                    # sharpness-aware minimization settings
  models/
    backbone.py                 # ResNet-18 feature extractor + FROZEN-BN policy
    classifier_head.py          # 7-class linear head
  methods/
    base_method.py              # shared interface + source classification loss
    erm.py                      # cross-entropy on pooled sources (no alignment)
    dan_dg.py                   # average MMD^2 over the 3 source-domain pairs
    sam.py                      # SAM optimizer (two-step) + SAM method wrapper
  selection/
    source_validation.py       # mean / worst source-val metrics (selection)
  evaluation/
    domain_metrics.py           # Sketch metrics + per-class analysis + confusions
    source_domain_separability.py  # 3-way LogReg domain probe (chance = 33.3%)
    sharpness.py                # the common sharpness proxy Delta_sharp
  train.py                      # THE common loop (ERM/DAN-DG/SAM; SAM two-pass)
  evaluate_sketch.py            # final tables + diagnostics (Sketch loaded HERE)
  controlled_study.py           # alignment/sharpness-strength sweep (Step 5)
  run_task3.py                  # orchestrator: trains all, evaluates, studies
  results/                      # all outputs (created on first run)
  requirements.txt
  README.md
```

### How the pieces connect
```
configs ─> shared/utils.load_config (merges inherit)
shared/pacs + pacs_protocol ─> splits + domain-balanced SOURCE batches (no Sketch)
        │
        ▼
train.py ── backbone + head + method ── trains each method
        │     (frozen BN; SAM runs two forward/backward passes)
        ▼
results/checkpoints/*.pt  +  history_*.json (training curves)
        │
        ▼
evaluate_sketch.py ── source-val + SKETCH metrics + separability + sharpness + per-class
controlled_study.py ── lambda_DG sweep {0.1,1,10}
        │
        ▼
results/ (task3_summary_table.csv, task3_final_results.json, *.png)
```

---

## 2. The steps (mapped to the assignment)

1. **ERM** — cross-entropy on the 3 pooled, domain-balanced sources. The SAME
   model as Task 2's Source-only ERM (reuse its checkpoint if you have it).
2. **DAN-DG** — `L_ERM + (lambda_DG/3) * sum_{e<e'} MMD^2(F(X_e), F(X_e'))` over
   the 3 unordered source pairs; same MMD/kernels as Task 2, **never touches
   Sketch**. `lambda_DG = 1`.
3. **SAM** — `min_theta max_{||eps||<=rho} L_ERM(theta+eps)`, `rho = 0.05`, two
   forward/backward passes, AdamW base optimizer, frozen BN on both passes.
4. **Common evaluation + diagnostics** — source-val (mean + worst) and **Sketch**
   accuracy/macro-F1 + Sketch change vs ERM; **source-domain separability**
   (3-way LogReg, chance 33.3%); the **sharpness proxy**; per-class Sketch
   changes vs ERM + confusions.
5. **Controlled study** — vary `lambda_DG ∈ {0.1,1,10}` (default) for DAN-DG,
   everything else fixed; plot source-F1, the method diagnostic, and Sketch.

---

## 3. How to run
```bash
pip install -r task3/requirements.txt
cd task3

# full pipeline
python run_task3.py

# reuse your Task 2 Source-only checkpoint as the ERM baseline (recommended):
python run_task3.py --erm_checkpoint /path/to/task2/results/checkpoints/source_only.pt

# only re-evaluate existing checkpoints
python run_task3.py --skip_train

# a single method
python train.py --config configs/sam.yaml

# just the controlled study
python controlled_study.py --config configs/dan_dg.yaml
```
Outputs in `results/`: `task3_summary_table.csv`, `task3_final_results.json`,
`loss_curves_<method>.png`, `controlled_study.png/.json`, and the checkpoints.

### Where it runs (CUDA / Apple M1 / CPU / Colab)
`shared/utils.get_device()` auto-detects **CUDA → MPS → CPU**; you edit nothing.
A GPU is strongly recommended: SAM does two passes per step, so it is the
slowest of the three. On Colab, set Runtime → GPU.

---

## 4. ⚠️ Manual work / things you must do yourself

1. **Download PACS yourself and place it (required).** Exactly like Task 2:
   ```
   task3/data_cache/PACS/<domain>/<class>/*.jpg
   ```
   domains `photo, art_painting, cartoon, sketch`; classes `dog, elephant,
   giraffe, guitar, horse, house, person`. If you already arranged PACS for
   Task 2, copy that same `data_cache/PACS` folder here (or point
   `dataset.root` in `configs/base.yaml` at it). Cite your PACS source in the report.

2. **(Recommended) Reuse Task 2's ERM checkpoint.** The spec says Task 3's ERM
   must be Task 2's Source-only model, **loaded, not retrained**. Pass
   `--erm_checkpoint <path to task2 source_only.pt>` (or set
   `paths.task2_erm_checkpoint` in `base.yaml`). If you don't, Task 3 trains an
   equivalent ERM under the identical protocol (same splits/seed/architecture),
   which matches Task 2's Source-only — but reusing the exact checkpoint is the
   literal requirement, so prefer the flag. Mention which you did in the report.

3. **Write the 8-page PDF report yourself (required).** AI-usage policy forbids
   AI-written report prose. This code produces numbers/tables/figures only.

4. **Choose ONE controlled study (already chosen for you).** Default is the
   DAN-DG `lambda_DG ∈ {0.1,1,10}` sweep. To use the SAM `rho ∈ {0.01,0.05,0.1}`
   sweep instead: set `controlled_study.enabled: true` in `sam.yaml` and `false`
   in `dan_dg.yaml`, then `python controlled_study.py --config configs/sam.yaml`.
   No code edits — just config flags.

5. **State a hypothesis before interpreting the study (required writing).**

You do **not** need to change any model weights, edit any function, or tune any
hyperparameter to run the code. Every required value is set in the configs.

---

## 5. "What to Watch For" — how this repo already handles it
- **Source invariance ≠ class invariance.** DAN-DG can lower separability while
  also erasing class-discriminative info. We report separability, source
  macro-F1, AND Sketch accuracy together so you can tell them apart.
- **Mean-source can hide a weak source; worst-source needn't predict Sketch.**
  The table reports each source domain, the mean, AND the worst.
- **The sharpness proxy is a LOCAL diagnostic, not global flatness.** Computed
  under the one specified perturbation (rho=0.05, fixed seed-6304 batch, eval
  mode) so all three models are compared on the identical yardstick.
- **DAN vs DAN-DG use the same mechanism, different information.** Task 2's DAN
  sees unlabeled Sketch; Task 3's DAN-DG aligns only observed sources — the code
  keeps the MMD/kernels identical so the comparison is about target access.
- **Never inspect Sketch to make a training choice.** Sketch is loaded only in
  `evaluate_sketch.py`, after all decisions are frozen; selection uses source
  validation only.

## 5b. DAN-DG stability (deviation — declared and defended on source-side grounds)
At the prescribed optimizer (AdamW lr 1e-4) with the mandated `lambda_DG=1`,
DAN-DG's SOURCE training collapsed: classification loss stayed pinned at
~ln(7)=1.95 (uniform-guess cross-entropy) and source-validation macro-F1 sat at
~0.05, below the 1/7 chance line. This is diagnosed entirely from SOURCE-ONLY
signals (the training loss curve and source-val F1) — Sketch is never consulted —
so it is a legitimate model-selection observation, not the target-driven
"post-hoc winner" choice the spec forbids. Mechanism: at lr 1e-4 the
pairwise-source-MMD gradient overpowers the classification gradient in the first
steps and drives features to near-constant (trivially making the three source
domains identical, which erases class information) before the classifier can
learn. Two DAN-DG-only stability measures counter this **while keeping
lambda_DG=1**:
  * **`mmd_warmup_steps` (=150, ~1 epoch):** ramp the MMD weight 0->1 so the
    classifier establishes features before alignment pressure engages (analogous
    to DANN's alpha schedule). The FINAL weight is still lambda_DG=1, so the model
    converges under the prescribed objective.
  * **A lower DAN-DG learning rate (3e-5 vs the shared 1e-4):** smaller steps let
    the classification signal keep pace with the MMD gradient instead of being
    overrun. This is a declared deviation from the shared optimizer, made for
    numerical stability of the joint objective and justified on source-side
    evidence; `lambda_DG` stays at the mandated 1.
ERM and SAM keep the prescribed lr 1e-4 and no warm-up. The controlled study
inherits these DAN-DG settings for every lambda, so the lambda sweep remains a
clean "effect of lambda" comparison under one fixed optimizer. Set
`mmd_warmup_steps: 0` and remove the `train.lr` override to reproduce the
original collapsing run. If DAN-DG still collapses under these measures, that is
itself the RQ2 finding (source alignment removing class-discriminative
information), evidenced by the source-side loss curve.

## 6. Reproducibility & the Task 2 link
Seed **6304** everywhere. The splits are the same JSON Task 2 used (bundled in
`shared/splits/`). ERM is Task 2's Source-only model. Cite DAN (Long 2015),
SAM (Foret 2021), and your PACS source in the report.
