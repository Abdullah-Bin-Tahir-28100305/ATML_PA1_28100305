# =============================================================================
# data/make_cue_conflicts.py
# -----------------------------------------------------------------------------
# PURPOSE: generate SHAPE-vs-TEXTURE cue-conflict images via AdaIN style
# transfer. A cue-conflict image takes the CONTENT/SHAPE from one class (A) and
# the STYLE/TEXTURE from another (B). A model that then predicts A is
# "shape-biased"; one that predicts B is "texture-biased" (Geirhos et al. 2019).
#
# ML CONCEPT — DISENTANGLING SHAPE AND TEXTURE:
#   On natural images shape and texture almost always agree (a cat has cat
#   shape AND cat fur), so a correct prediction cannot tell you which cue the
#   model used. Cue conflict deliberately puts them in disagreement, turning an
#   otherwise hidden preference into an observable decision. This is the core
#   experimental idea of the whole shape-bias literature.
#
# ML CONCEPT — AdaIN (Huang & Belongie 2017, the optional reading):
#   Adaptive Instance Normalization transfers style by matching per-channel
#   feature statistics. In a VGG feature space it does:
#       AdaIN(c, s) = sigma(s) * (c - mu(c)) / sigma(c) + mu(s)
#   i.e. it re-normalises the CONTENT features to have the per-channel MEAN and
#   STD of the STYLE features. The insight (Huang & Belongie) is that a layer's
#   channel-wise mean/variance largely encode *style/texture*, while the spatial
#   arrangement encodes *content/shape*. A decoder turns the AdaIN'd features
#   back into an image that keeps content geometry but wears the style's texture.
#
# WHY A SEPARATE FILE / STAGE:
#   Same reason as transforms: "Keep transformation generation separate from
#   evaluation." We GENERATE and SAVE the conflict set once (with an accepted/
#   rejected log), then every model is evaluated on the identical saved set.
#
# IMPORTANT — REJECTION RULE (spec):
#   "define a visual rejection rule BEFORE model evaluation and record the
#   accepted and rejected counts. Do not use model predictions to decide which
#   images to retain." Our rule (see _passes_rejection_rule) uses only
#   content-independent image statistics — never any classifier output.
#
# LINKS TO OTHER FILES:
#   - utils: config + seeding.
#   - data/make_subset: to draw content/style images per class.
#   - Consumed by analysis/evaluate_bias.py (shape-bias & coverage) and
#     scripts/run_task1.py.
# =============================================================================

import os
import json
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights

try:
    from utils import load_config, set_seed, ensure_dir, get_device
    from data.make_subset import load_datasets, _labels_of
except ImportError:
    from ..utils import load_config, set_seed, ensure_dir, get_device
    from .make_subset import load_datasets, _labels_of


# -----------------------------------------------------------------------------
# AdaIN core operation
# -----------------------------------------------------------------------------
def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    """Compute per-channel (spatial) mean and std of a feature map (B,C,H,W).

    These are exactly the statistics AdaIN manipulates. The mean/std are taken
    over the SPATIAL dimensions (H,W) for each channel independently — that is
    what "instance" normalization means (per-sample, per-channel).
    """
    B, C = feat.shape[:2]
    feat_flat = feat.view(B, C, -1)                     # flatten H*W
    mean = feat_flat.mean(dim=2).view(B, C, 1, 1)       # per-channel mean
    std = (feat_flat.var(dim=2) + eps).sqrt().view(B, C, 1, 1)  # per-channel std
    return mean, std


def adaptive_instance_normalization(content_feat, style_feat):
    """The AdaIN equation itself: re-style content features with style stats.

    out = sigma(style) * (content - mu(content)) / sigma(content) + mu(style)
    Step by step: whiten the content (subtract its mean, divide by its std) so
    it has zero-mean/unit-var per channel, then 'colour' it with the style's
    mean and std. Content SPATIAL structure is untouched; only per-channel
    statistics (the texture signature) are swapped in.
    """
    c_mean, c_std = _calc_mean_std(content_feat)
    s_mean, s_std = _calc_mean_std(style_feat)
    normalized = (content_feat - c_mean) / c_std        # whiten content
    return normalized * s_std + s_mean                  # apply style stats


# -----------------------------------------------------------------------------
# Encoder: a pretrained VGG-19 truncated at relu4_1 (standard AdaIN encoder)
# -----------------------------------------------------------------------------
# =============================================================================
# AUTHENTIC AdaIN encoder (matches naoto0804/pytorch-AdaIN)
# -----------------------------------------------------------------------------
# CRITICAL COMPATIBILITY NOTE:
#   The pretrained AdaIN DECODER was trained against a SPECIFIC VGG-19 encoder
#   (the repo's `vgg_normalised.pth`), NOT torchvision's ImageNet VGG-19. The two
#   encoders live in different feature spaces and expect different input
#   conventions, so pairing torchvision's encoder with that decoder produces
#   garbage. To get authentic stylization we MUST reproduce that repo's exact
#   encoder architecture and load its weights.
#
#   That encoder:
#     * takes images in the [0,1] range, RGB (the repo normalises inside the net
#       via its first 3x3 conv that maps 3->3, which is why the first layer is a
#       Conv2d(3,3,1)),
#     * is a flat nn.Sequential with ReflectionPad2d + Conv + ReLU blocks,
#     * we slice it to relu4_1 (the first 31 modules), matching the repo.
#
# So the authentic path now needs TWO weight files: the decoder (decoder.pth)
# AND this encoder (vgg_normalised.pth), both from naoto0804/pytorch-AdaIN.
# =============================================================================
def _build_naoto_vgg():
    """Return the naoto0804/pytorch-AdaIN VGG-19 (full), as an nn.Sequential.

    This is the exact module sequence that repo uses for `vgg`; its
    `vgg_normalised.pth` loads straight into it. We only ever run it up to
    relu4_1 (index 31), which is what AdaIN uses.
    """
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),   # relu1_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),  # relu2_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),  # relu3_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),  # relu4_1 (index 31)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),  # relu5_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    )


class VGGEncoder(nn.Module):
    """AdaIN encoder = naoto0804 VGG-19 truncated at relu4_1 (index 31).

    ML CONCEPT — WHY VGG FEATURES:
      AdaIN operates in a *perceptual* feature space. relu4_1 is deep enough to
      capture texture/style statistics but shallow enough to preserve content
      geometry (Huang & Belongie 2017). We use the repo's own VGG so its features
      match what the pretrained decoder was trained to invert.

    Two modes:
      * If `vgg_weights_path` is given and loads, we use the AUTHENTIC repo VGG
        (required to pair with the repo decoder).
      * Otherwise we fall back to torchvision VGG-19 features (used only by the
        decoder-free fallback stylizer, which never touches the repo decoder).
    """
    def __init__(self, vgg_weights_path=None):
        super().__init__()
        self.authentic = False
        if vgg_weights_path and os.path.exists(vgg_weights_path):
            full = _build_naoto_vgg()
            state = torch.load(vgg_weights_path, map_location="cpu")
            full.load_state_dict(state)                 # exact-match load
            # keep modules up to and INCLUDING relu4_1 (index 31)
            self.slice = nn.Sequential(*list(full.children())[:31])
            self.authentic = True
        else:
            # Fallback encoder for the decoder-free path only.
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            self.slice = nn.Sequential(*[vgg[i] for i in range(22)])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        return self.slice(x)


# -----------------------------------------------------------------------------
# Decoder: mirror of the encoder, maps relu4_1 features back to an image.
# -----------------------------------------------------------------------------
class VGGDecoder(nn.Module):
    """A decoder that inverts relu4_1 features into an RGB image.

    NOTE ON PRETRAINED WEIGHTS (read this — it affects your manual work!):
      The AUTHENTIC AdaIN pipeline uses a decoder that Huang & Belongie TRAINED
      on MS-COCO. Those weights are NOT bundled with torchvision. This class
      defines the correct architecture and, if a weights file is provided in the
      config/at the given path, loads it. If no weights are available, we FALL
      BACK to a lightweight stylizer (see stylize_fallback) so the pipeline
      still runs end-to-end and still produces genuine shape/texture conflicts.
      See README "Manual steps" for how to drop in pretrained decoder weights
      for the highest-fidelity stylizations.
    """
    def __init__(self):
        super().__init__()
        # Architecture mirrors the encoder: upsample + conv blocks back to 3ch.
        # ReflectionPad avoids border artefacts (same reason we used reflect pad
        # in the translation transform).
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.decoder(x)


# -----------------------------------------------------------------------------
# High-level stylizer
# -----------------------------------------------------------------------------
class AdaINStylizer:
    """Wraps encoder + AdaIN + decoder into a single .stylize(content, style)."""

    def __init__(self, cfg, decoder_weights_path=None, vgg_weights_path=None):
        self.device = get_device()
        self.alpha = cfg["cue_conflict"]["style_strength"]   # style strength

        # AUTHENTIC AdaIN requires a MATCHED pair: the repo's VGG encoder AND its
        # decoder. We only take the authentic path when BOTH files load; using
        # one without the other produces garbage (different feature spaces).
        self.encoder = VGGEncoder(vgg_weights_path=vgg_weights_path).to(self.device)
        self.decoder = VGGDecoder().to(self.device)
        self.has_decoder_weights = False

        decoder_ok = False
        if decoder_weights_path and os.path.exists(decoder_weights_path):
            state = torch.load(decoder_weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # accept both prefixed ('decoder.x') and bare ('x') key namings
            if not any(k.startswith("decoder.") for k in state):
                state = {f"decoder.{k}": v for k, v in state.items()}
            self.decoder.load_state_dict(state)
            decoder_ok = True

        # authentic path ON only if BOTH encoder and decoder loaded their weights
        self.has_decoder_weights = decoder_ok and self.encoder.authentic
        if decoder_ok and not self.encoder.authentic:
            print("[warn] decoder weights loaded but VGG encoder weights are "
                  "missing/incompatible; the repo decoder needs the repo encoder. "
                  "Falling back to the decoder-free stylizer to avoid garbage. "
                  "Pass --vgg_weights <vgg_normalised.pth> to enable authentic AdaIN.")
        self.decoder.eval()

    @torch.no_grad()                                   # inference only, no grads
    def stylize(self, content, style):
        """Produce a stylized image: content geometry + style texture, in [0,1].

        `alpha` interpolates between the plain content features (alpha=0) and the
        fully AdaIN'd features (alpha=1): t = alpha*AdaIN(c,s) + (1-alpha)*c.
        This is the standard AdaIN 'style strength' knob (our reported choice).
        """
        content = content.to(self.device)
        style = style.to(self.device)
        if content.dim() == 3:
            content = content.unsqueeze(0)
        if style.dim() == 3:
            style = style.unsqueeze(0)

        if self.has_decoder_weights:
            # AUTHENTIC PATH: encode both, AdaIN, interpolate, decode.
            c_feat = self.encoder(content)
            s_feat = self.encoder(style)
            t = adaptive_instance_normalization(c_feat, s_feat)
            t = self.alpha * t + (1 - self.alpha) * c_feat   # style-strength mix
            out = self.decoder(t)
            return out.clamp(0, 1).squeeze(0).cpu()
        else:
            # FALLBACK PATH (no trained decoder available): apply AdaIN directly
            # in *pixel* space by matching per-channel colour statistics, then
            # blend in high-frequency style texture. This still creates a genuine
            # shape(content)/texture(style) conflict — geometry from content,
            # colour+texture statistics from style — which is what the
            # experiment needs. Documented as a fallback in the README.
            return self._stylize_fallback(content, style).squeeze(0).cpu()

    def _stylize_fallback(self, content, style):
        """Decoder-free stylization: pixel-space AdaIN + style texture blend.

        We (1) match content's per-channel mean/std to style's (colour/texture
        statistics transfer, the pixel-space analogue of AdaIN), and (2) add a
        fraction of the style's high-frequency detail (texture) while keeping the
        content's low-frequency structure (shape). Shape stays from content;
        colour + fine texture come from style.
        """
        # (1) Per-channel statistic matching in pixel space.
        c_mean, c_std = _calc_mean_std(content)
        s_mean, s_std = _calc_mean_std(style)
        recolored = (content - c_mean) / c_std * s_std + s_mean

        # (2) Extract style's high-frequency texture via a blur residual.
        #     blur = low-pass; style - blur(style) = high-pass (fine texture).
        k = _gaussian_kernel(5, 1.0).to(self.device)
        def blur(img):
            img_p = F.pad(img, (2, 2, 2, 2), mode="reflect")
            return F.conv2d(img_p, k, groups=3)
        style_hf = style - blur(style)             # style texture detail
        content_lf = blur(recolored)               # content structure (smoothed)
        mixed = content_lf + self.alpha * style_hf # shape + texture blend
        return mixed.clamp(0, 1)


def _gaussian_kernel(ksize, sigma):
    """Build a depthwise 3-channel Gaussian blur kernel (for the fallback)."""
    ax = torch.arange(ksize) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1)  # (3,1,k,k)


# -----------------------------------------------------------------------------
# Rejection rule (MUST NOT use model predictions)
# -----------------------------------------------------------------------------
def _passes_rejection_rule(img: torch.Tensor, content: torch.Tensor, cfg) -> bool:
    """Return True if a stylization is visually usable, using ONLY image stats.

    Two failure modes we reject (spec: define the rule before evaluation):
      * COLLAPSE: the stylizer washed everything to near-constant -> tiny std.
        Such an image has no interpretable content, so it is ambiguous.
      * NO-OP: the stylization barely differs from the content -> style failed
        to apply, so there is effectively no texture conflict to measure.
    Both thresholds are content-independent statistics; NO classifier is
    consulted, satisfying "Do not use model predictions to decide which images
    to retain."
    """
    std = img.std().item()
    change = (img - content).abs().mean().item()
    if std < cfg["cue_conflict"]["reject"]["min_std"]:
        return False                               # collapsed / flat
    if change < cfg["cue_conflict"]["reject"]["min_change"]:
        return False                               # style did not take
    return True


# -----------------------------------------------------------------------------
# Orchestration: build the balanced conflict set
# -----------------------------------------------------------------------------
def generate_cue_conflicts(cfg, decoder_weights_path=None, vgg_weights_path=None):
    """Generate >=200 valid cue-conflict images, balanced over pairs & directions.

    Returns a dict with tensors and metadata, and writes an accept/reject log.

    BALANCING (spec: "balanced across class pairs and directions as closely as
    possible"): we loop over unordered class pairs {A,B}, and for each we make
    BOTH directions — (shape=A,texture=B) and (shape=B,texture=A) — cycling
    through many content/style image instances until we have collected enough
    ACCEPTED conflicts per (pair, direction) cell.
    """
    set_seed(cfg["seed"])
    device = get_device()
    train_ds, _, class_names = load_datasets(cfg)   # use TRAIN images as raw material
    labels = _labels_of(train_ds)
    rng = np.random.default_rng(cfg["seed"])

    # Precompute, per class, a shuffled list of dataset indices to draw from.
    class_to_indices = {}
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        class_to_indices[int(c)] = idx_c.tolist()

    # Choose class pairs. We take the first `num_pairs` unordered pairs in a
    # deterministic order (seeded shuffle of all C(K,2) combinations).
    all_pairs = list(itertools.combinations(range(len(class_names)), 2))
    rng.shuffle(all_pairs)
    pairs = all_pairs[: cfg["cue_conflict"]["num_pairs"]]

    stylizer = AdaINStylizer(cfg, decoder_weights_path, vgg_weights_path)

    # How many accepted conflicts per (pair, direction) to aim for.
    target_total = cfg["cue_conflict"]["target_valid"]
    n_cells = len(pairs) * 2                          # 2 directions per pair
    per_cell = int(np.ceil(target_total / n_cells))  # even quota per cell

    images, meta = [], []
    # We ALSO keep the content (shape-source) image for each accepted conflict.
    # Reason: Step 6 (representation analysis) needs to pair the stylized image
    # with its clean counterpart to compute cosine stability I_T for the
    # cue-conflict transform. The natural clean counterpart is the CONTENT image
    # (same geometry, no style), so storing it here keeps the pairing exact and
    # 1-to-1 (spec lists cue conflict among the transforms for I_T).
    content_images = []
    accepted, rejected = 0, 0

    def draw_image(class_id, cursor):
        """Fetch the (image_tensor) for class `class_id` at position `cursor`."""
        idx_list = class_to_indices[class_id]
        ds_index = idx_list[cursor % len(idx_list)]  # wrap around if exhausted
        img, _ = train_ds[ds_index]
        return img

    for (a, b) in pairs:
        # Two directions: (shape=a, texture=b) and (shape=b, texture=a).
        for (shape_cls, tex_cls) in [(a, b), (b, a)]:
            collected = 0
            cursor_c = 0                              # cursor into content class
            cursor_s = 0                              # cursor into style class
            attempts = 0
            max_attempts = per_cell * 8              # give up cap to avoid loops
            while collected < per_cell and attempts < max_attempts:
                attempts += 1
                content = draw_image(shape_cls, cursor_c); cursor_c += 1
                style = draw_image(tex_cls, cursor_s);   cursor_s += 1
                styl = stylizer.stylize(content, style)   # (3,224,224) in [0,1]
                # Apply the pre-registered rejection rule (no model involved).
                if _passes_rejection_rule(styl, content, cfg):
                    images.append(styl)
                    # store the matching content image (squeezed to 3,H,W) so
                    # stylized[i] pairs with content_images[i] downstream.
                    content_images.append(content.squeeze(0).cpu()
                                          if content.dim() == 4 else content.cpu())
                    meta.append({
                        "shape_label": int(shape_cls),   # ground-truth SHAPE class
                        "texture_label": int(tex_cls),   # ground-truth TEXTURE class
                        "shape_name": class_names[shape_cls],
                        "texture_name": class_names[tex_cls],
                        "pair": [int(a), int(b)],
                    })
                    collected += 1
                    accepted += 1
                else:
                    rejected += 1

    images = torch.stack(images) if images else torch.empty(0)
    content_images = torch.stack(content_images) if content_images else torch.empty(0)

    # Persist an accept/reject log (spec: "record the accepted and rejected
    # counts"). This log is part of the required evidence.
    out_dir = ensure_dir(cfg["paths"]["results_dir"])
    log = {
        "num_pairs": len(pairs),
        "pairs": [[int(a), int(b)] for a, b in pairs],
        "pair_names": [[class_names[a], class_names[b]] for a, b in pairs],
        "per_cell_target": per_cell,
        "accepted": accepted,
        "rejected": rejected,
        "total_valid": int(images.shape[0]) if images.numel() else 0,
        "style_strength_alpha": cfg["cue_conflict"]["style_strength"],
        "used_pretrained_decoder": stylizer.has_decoder_weights,
        "rejection_rule": cfg["cue_conflict"]["reject"],
    }
    with open(os.path.join(out_dir, "cue_conflict_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Save the actual stylized images to disk so they can be VISUALLY INSPECTED
    # (the spec requires the conflicts to "retain interpretable content and style
    # cues"; the human eyeball check happens on these files). Without this the
    # images would only ever exist in memory during the run. See the function's
    # own docstring for what to look for.
    save_cue_conflict_samples(images, meta, out_dir,
                              n_individual=cfg["cue_conflict"].get("n_saved_samples", 40))

    return {"images": images, "content_images": content_images,
            "meta": meta, "log": log, "class_names": class_names}


def save_cue_conflict_samples(images, meta, results_dir, n_individual=40):
    """Write cue-conflict images to disk for the required visual sanity check.

    Creates results_dir/cue_conflict_samples/ containing:
      * montage.png       — a labeled grid of the first up-to-40 conflicts, each
                            captioned with its intended SHAPE class and TEXTURE
                            class, so you can confirm at a glance that both cues
                            are interpretable;
      * <NNN>_shape-<..>_tex-<..>.png — individual PNGs of the first
                            `n_individual` conflicts for closer inspection.
      * ALL images are also saved as a single .npz for programmatic re-checking.

    WHAT TO LOOK FOR (the eyeball check):
      GOOD  -> the object's outline/shape (content class) is still recognizable
               AND the texture/style of the OTHER class is visibly applied.
      BAD   -> unrecognizable mush (over-stylized) or looks like a plain photo
               (style did not apply). The statistical rejection rule already
               drops the worst; this is a human sanity pass on top. Never filter
               by model predictions.
    """
    import numpy as np
    if images is None or images.numel() == 0:
        print("[warn] no cue-conflict images to save.")
        return None
    sample_dir = ensure_dir(os.path.join(results_dir, "cue_conflict_samples"))

    # ---- individual labeled PNGs (first n_individual) ----
    try:
        from PIL import Image
        k = min(n_individual, images.shape[0])
        for i in range(k):
            arr = (images[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            fname = f"{i:03d}_shape-{meta[i]['shape_name']}_tex-{meta[i]['texture_name']}.png"
            Image.fromarray(arr).save(os.path.join(sample_dir, fname))
    except Exception as e:
        print(f"[warn] could not save individual PNGs ({e}); montage still attempted.")

    # ---- labeled montage grid ----
    import matplotlib
    matplotlib.use("Agg")                               # file-only backend
    import matplotlib.pyplot as plt
    n = min(40, images.shape[0])
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = np.atleast_1d(axes).flatten()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_title(f"shape: {meta[i]['shape_name']}\ntex: {meta[i]['texture_name']}",
                         fontsize=6)
    fig.suptitle("Cue-conflict samples — check shape (content) vs texture (style) "
                 "are both interpretable", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    montage_path = os.path.join(sample_dir, "montage.png")
    plt.savefig(montage_path, dpi=150)
    plt.close()

    print(f"[info] saved {min(n_individual, images.shape[0])} sample PNGs + montage "
          f"-> {sample_dir}")
    return sample_dir


if __name__ == "__main__":
    cfg = load_config()
    out = generate_cue_conflicts(cfg)
    print(f"accepted={out['log']['accepted']} rejected={out['log']['rejected']}")
    print(f"valid conflicts={out['log']['total_valid']} (need >= 200)")
