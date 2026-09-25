# Task 1 — Inductive Biases and Feature Representations

This repository implements **Task 1 only** of the assignment: comparing a
convolutional network (ResNet-50), a Vision Transformer (ViT-B/16), and CLIP
(ViT-B/32) under controlled image interventions, to study which visual cues
drive their predictions, which cues stay encoded in their representations, and
how these behaviours relate to architecture and pretraining.

> The code is heavily commented line-by-line and each comment ties the code to
> the underlying ML concept, so you can read the source as a study guide.

---

## 1. What each file is and does

```
task1/
  configs/
    config.yaml              # SINGLE source of truth for every hyperparameter,
                             # seed, path, and experimental-design choice.
  data/
    make_subset.py           # loads STL-10; builds the stratified 80/20
                             # train/val split and the class-balanced 500-image
                             # test evaluation subset; saves image identifiers.
    make_cue_conflicts.py    # AdaIN style transfer -> shape/texture cue-conflict
                             # images; applies the pre-registered rejection rule;
                             # logs accepted/rejected counts.
    transforms.py            # the pixel-space interventions: grayscale, hue
                             # rotation, translation, 4x4 patch shuffle.
  models/
    backbones.py             # ResNet-50 / ViT-B/16 / CLIP wrappers (frozen),
                             # per-model normalization, the linear-probe head +
                             # its training loop, and CLIP zero-shot.
  analysis/
    evaluate_bias.py         # metrics: top-1, macro-F1, mean-max-confidence,
                             # prediction consistency, shape bias + coverage.
    feature_similarity.py    # cosine-stability index I_T (representation drift).
    representation.py        # t-SNE / UMAP 2-D visualization of features.
  scripts/
    run_task1.py             # THE ORCHESTRATOR: runs all six steps in order and
                             # writes everything to results/.
  utils.py                   # shared helpers: seeding, config loading, device.
  results/                   # all outputs land here (created on first run).
  requirements.txt
  README.md
  task1_walkthrough.ipynb    # optional notebook: explains + runs the pipeline.
```

### How the files connect (data flow)

```
config.yaml ─┬─> utils.load_config ──> (every module reads the same knobs)
             │
make_subset ──> stratified split + 500-image eval subset ─┐
                                                          │
transforms ──> grayscale/hue/translate/patch_shuffle ─────┤
make_cue_conflicts ──> stylized shape/texture images ─────┤
                                                          ▼
                                    scripts/run_task1.py (orchestrator)
                                                          │
                                   backbones (frozen feats + heads + CLIP)
                                                          │
                    ┌─────────────────────────────────────┼───────────────────┐
                    ▼                                       ▼                   ▼
        evaluate_bias (metrics,               feature_similarity (I_T)   representation
        consistency, shape bias)                                          (t-SNE/UMAP plots)
                    │                                       │                   │
                    └──────────────► results/ (JSON, CSV, PNG figures) ◄────────┘
```

The staging — **generate transforms first, evaluate second** — is deliberate
and required by the assignment: it guarantees ResNet, ViT and CLIP are compared
on *byte-identical* clean and transformed images.

---

## 2. The six experiments (mapped to the assignment "Steps")

1. **Clean Baseline** — top-1, macro-F1, mean-max-confidence for the three
   trained heads *and* CLIP zero-shot. This is the reference every later
   intervention is compared against (both absolute and relative).
2. **Color Bias** — grayscale (required) + a fixed 90° hue rotation (our chosen
   additional color intervention). Reports accuracy change and prediction
   consistency vs clean.
3. **Shape vs Texture** — AdaIN cue conflicts across ≥6 class pairs, both
   directions, ≥200 valid images after a pre-registered rejection rule. Reports
   shape bias and coverage.
4. **Translation** — shifts {0, 8, 16, 32} px averaged over the four cardinal
   directions, with reflection padding + shifted crop. Produces accuracy and
   consistency vs displacement curves.
5. **Patch Structure** — one non-identity 4×4 permutation (seed 6304), reused
   across models. Reports accuracy drop and consistency.
6. **Representation Analysis** — cosine-stability index `I_T` for grayscale, cue
   conflict, translation, and patch shuffle, plus a t-SNE (or UMAP) plot per
   backbone with color = class and marker = clean/transformed.

---

## 3. How to run

```bash
# 1) create an environment and install deps
pip install -r requirements.txt

# 2) full run (downloads STL-10 automatically on first use; GPU recommended)
python scripts/run_task1.py

# 3) fast smoke test to check everything wires up (tiny subset, few conflicts)
python scripts/run_task1.py --quick

# 4) optional: use authentic pretrained AdaIN decoder weights (see below)
python scripts/run_task1.py --decoder_weights /path/to/adain_decoder.pth
```

### Where it runs (CUDA / Apple M1 / CPU / Colab)

The code auto-detects the best device via `utils.get_device()` — you never edit
anything. It picks, in order: **CUDA** (NVIDIA GPU, e.g. Google Colab) →
**MPS** (Apple Silicon GPU on M1/M2/M3 Macs) → **CPU** (works everywhere).

- **MacBook M1/M2/M3:** just run it — PyTorch uses your Mac's GPU through MPS.
  (There is no CUDA on a Mac; CUDA is NVIDIA-only. MPS is the Mac equivalent.)
  A safety flag (`PYTORCH_ENABLE_MPS_FALLBACK=1`) is set automatically so any op
  without a Metal kernel falls back to CPU instead of erroring.
- **Google Colab (recommended for the full run):** set Runtime → Change runtime
  type → GPU, then `pip install -r requirements.txt` and run the script or open
  `task1_walkthrough.ipynb`. The CUDA path is used automatically. Download the
  `results/` folder before the session ends (Colab disk is temporary).
- **Any CPU-only machine:** it still runs, just slower. Use `--quick` for a fast
  correctness check.

Outputs appear in `results/`:
`task1_results.json` (all numbers), `summary_compact_comparison.csv`,
`cue_conflict_log.json`, `eval_subset_identifiers.json`,
`translation_acc.png`, `translation_consistency.png`, and
`proj_<model>_<transform>.png` projection figures.

---

## 4. ⚠️ Manual work required by you (this is the part your friend meant)

Most of the pipeline is automatic, but a few things genuinely need **your**
hands and judgement. None of them are hard; here they are, most important first.

1. **The 8-page PDF report is 100% yours — and must be.** The assignment's *AI
   Usage Policy* says you may **not** use generative AI to write any part of the
   report; the language, interpretation, and analysis must be entirely your own.
   This code produces the numbers, tables, and figures; **you** write the
   report, answer the four Research Questions, and interpret the results. Do not
   paste model-generated prose into it.

2. **Pretrained AdaIN decoder weights (recommended for best stylizations).**
   Authentic AdaIN (Huang & Belongie 2017) uses a decoder trained on MS-COCO;
   those weights are **not** bundled with torchvision. Two options:
   - **Do nothing:** the code automatically uses a built-in *fallback* stylizer
     (pixel-space statistic matching + style-texture blend). It still creates
     genuine shape-vs-texture conflicts and runs end-to-end. Fine for a first
     pass, but the stylizations look less like Figure 1.
   - **Recommended (authentic AdaIN):** this needs TWO matched files from the
     `naoto0804/pytorch-AdaIN` repo — the decoder AND its VGG encoder — because
     the pretrained decoder only works with that repo's own encoder. Passing the
     decoder alone with a different encoder produces garbage (saturated blobs),
     so the code now REQUIRES both and refuses the authentic path otherwise.
     Download `decoder.pth` and `vgg_normalised.pth`, then run:
     `python scripts/run_task1.py --decoder_weights /path/decoder.pth --vgg_weights /path/vgg_normalised.pth`
     The `cue_conflict_log.json` records `used_pretrained_decoder: true` when the
     authentic path ran. **Cite `naoto0804/pytorch-AdaIN` and Huang & Belongie
     (2017) in this README/report** (assignment requirement).

3. **Look at the generated cue-conflict images before trusting the numbers.**
   The run automatically saves them to **`results/cue_conflict_samples/`**:
   - `montage.png` — a labeled grid of the first 40 conflicts, each captioned
     with its intended SHAPE class and TEXTURE class;
   - `NNN_shape-<..>_tex-<..>.png` — individual PNGs (count set by
     `cue_conflict.n_saved_samples` in `config.yaml`, default 40).
   Open the montage and confirm the content (shape) and style (texture) are both
   still interpretable (spec: "retain interpretable content and style cues").
   GOOD = object outline recognizable AND the other class's texture visibly
   applied; BAD = unrecognizable mush, or looks like a plain photo. If too many
   look degenerate, loosen/tighten the thresholds under `cue_conflict.reject` in
   `config.yaml` and re-run. **Never** filter by model predictions — that would
   violate the rule. (Note: with the default fallback stylizer the images look
   less crisp than the paper's Figure 1; that's expected.)

4. **Sanity-check the STL-10 download.** The first run downloads ~2.5 GB. If you
   are offline or behind a proxy, pre-download it or point `dataset.root` at an
   existing copy.

5. **Pick UMAP vs t-SNE and record it.** Default is t-SNE (no extra install). If
   you prefer UMAP, `pip install umap-learn` and set
   `visualization.method: umap` in the config. Whichever you use, its settings
   are written into every figure title and into the JSON for your report.

6. **Optional design choices are yours to defend.** The config encodes our
   choices (STL-10, 6 class pairs, `style_strength=1.0`, hue rotation of 90°,
   t-SNE). The assignment says to "state a hypothesis and an appropriate metric
   for each experimental choice before interpreting its result" — do that in the
   report; you can change any of these knobs in `config.yaml` if you want to
   defend different ones.

---

## 5. About the `.ipynb` question

The suggested repository structure in the PDF does **not** list any `.ipynb`
file — it lists plain `.py` modules plus `scripts/run_task1.py`. That is fine and
fully sufficient: **the assignment does not require a notebook.** Graders run the
`.py` files and read your PDF report; the code lives in a public GitHub repo.

However, notebooks are convenient for exploring and for showing your work, so
this repo *also* includes an **optional** `task1_walkthrough.ipynb` that simply
imports the same `.py` modules and runs the pipeline cell-by-cell with
explanations. Key points:

- The notebook is **additive**, not a replacement. All real logic lives in the
  `.py` files; the notebook just calls them. This keeps a single source of truth
  (no copy-pasted, drifting code) and matches the suggested structure.
- Everything works **without** the notebook — `python scripts/run_task1.py` is
  the canonical way to reproduce results.
- If your course wants a notebook submission, the notebook is ready; if it wants
  scripts, ignore the notebook. You lose nothing either way.

---

## 6. Reproducibility

Every source of randomness is seeded with **6304** (the seed the assignment
mandates) via `utils.set_seed`: the train/val split, the 500-image subset, the
linear-head initialisation, the patch permutation, and the t-SNE/UMAP layout.
The exact evaluation images are saved as identifiers in
`results/eval_subset_identifiers.json`, and every hyperparameter lives in
`configs/config.yaml`, so the whole pipeline is reproducible.

## 7. External code attribution

- Pretrained backbones: torchvision (`ResNet50_Weights.IMAGENET1K_V2`,
  `ViT_B_16_Weights.IMAGENET1K_V1`) and OpenCLIP (`ViT-B-32`, `pretrained=openai`).
- AdaIN method: Huang & Belongie, *Arbitrary Style Transfer in Real-time with
  Adaptive Instance Normalization*, ICCV 2017. If you download pretrained
  decoder weights, add the exact source repository and license here.
