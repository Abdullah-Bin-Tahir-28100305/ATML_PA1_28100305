# =============================================================================
# data/transforms.py
# -----------------------------------------------------------------------------
# PURPOSE: define every CONTROLLED INTERVENTION that operates in the common
# 224x224 pixel space:
#   * grayscale            (required color intervention)
#   * hue_rotation         (our chosen additional color intervention)
#   * translate            (translation experiment: shift + reflection pad + crop)
#   * patch_shuffle        (patch-structure experiment: 4x4 grid permutation)
#
# ML CONCEPT — CONTROLLED INTERVENTION:
#   The spec defines it as changing "one visual factor while preserving others
#   as far as possible." Each function below changes exactly one factor:
#     grayscale/hue -> color, keeping geometry;
#     translate     -> position, keeping appearance;
#     patch_shuffle -> global spatial arrangement, keeping the pixels & local
#                      texture. This lets us attribute any prediction change to
#                      that single factor (a causal, not correlational, probe).
#
# KEY DESIGN RULE (spec): all interventions take and return a tensor in the
# SHARED [0,1], 3x224x224 space, BEFORE any model-specific normalization. That
# is why nothing here subtracts ImageNet means — normalization lives in the
# backbones. This is the mechanism that guarantees "Every model must receive
# the same clean and transformed images."
#
# LINKS TO OTHER FILES:
#   - Called by scripts/run_task1.py to build each intervention's image set.
#   - The patch permutation uses seed 6304 (from config) exactly as the spec
#     demands, so ViT/ResNet/CLIP all see the identical shuffle.
# =============================================================================

import torch
import torch.nn.functional as F        # for grid_sample / padding utilities
import numpy as np


# -----------------------------------------------------------------------------
# 1. GRAYSCALE  (required common color intervention)
# -----------------------------------------------------------------------------
def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image (or batch) to grayscale, kept as 3 channels.

    ML CONCEPT — REMOVING A CUE:
      Grayscale destroys chromatic (color) information while leaving luminance,
      edges and shape intact. If accuracy barely drops, the model was NOT
      relying on color; if it collapses, color was load-bearing. This isolates
      the *contribution of color* to the decision (spec: "Grayscale tests the
      effect of removing color").

    We use the standard ITU-R 601 luminance weights (perceptual brightness):
      Y = 0.299 R + 0.587 G + 0.114 B
    Then we REPLICATE Y across 3 channels so the tensor shape stays 3x224x224 —
    the backbones all expect 3-channel input, so we keep the shape but kill the
    color information.
    """
    # Accept both a single image (C,H,W) and a batch (B,C,H,W) by normalising to
    # a batch, then squeezing back at the end.
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)                       # (C,H,W) -> (1,C,H,W)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]          # split channels
    y = 0.299 * r + 0.587 * g + 0.114 * b        # luminance (B,H,W)
    y = y.unsqueeze(1)                           # (B,1,H,W)
    out = y.repeat(1, 3, 1, 1)                   # replicate to (B,3,H,W)
    return out.squeeze(0) if single else out


# -----------------------------------------------------------------------------
# 2. HUE ROTATION  (our chosen additional color intervention)
# -----------------------------------------------------------------------------
def _rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    """Convert a batch of RGB images in [0,1] to HSV. Shapes: (B,3,H,W)->(B,3,H,W).

    We implement HSV manually (rather than importing kornia) so the code has no
    heavy extra dependency and every arithmetic step is visible/auditable.
    HSV = Hue, Saturation, Value. Rotating H changes the *color* while leaving
    S and V (and therefore geometry/brightness/contrast) untouched.
    """
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)                        # V (value) = max channel
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc                          # chroma
    # Saturation: 0 where the pixel is gray (delta==0); guard divide-by-zero.
    s = torch.where(maxc > 0, delta / (maxc + 1e-8), torch.zeros_like(maxc))
    # Hue: which channel is the max determines the 60-degree sector.
    deltac = delta + 1e-8                         # avoid /0
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac
    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0                           # normalise hue to [0,1)
    h = torch.where(delta == 0, torch.zeros_like(h), h)  # gray -> hue 0
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _rgb_to_hsv: (B,3,H,W) HSV -> (B,3,H,W) RGB in [0,1]."""
    h, s, v = x[:, 0], x[:, 1], x[:, 2]
    i = torch.floor(h * 6.0)                      # which sector (0..5)
    f = h * 6.0 - i                               # fractional part within sector
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).long()                            # wrap sector index
    # For each sector, pick the right (r,g,b) combination.
    conditions = [i == k for k in range(6)]
    r = torch.zeros_like(v); g = torch.zeros_like(v); b = torch.zeros_like(v)
    rs = [v, q, p, p, t, v]                       # r per sector
    gs = [t, v, v, q, p, p]                       # g per sector
    bs = [p, p, t, v, v, q]                       # b per sector
    for k in range(6):
        r = torch.where(conditions[k], rs[k], r)
        g = torch.where(conditions[k], gs[k], g)
        b = torch.where(conditions[k], bs[k], b)
    return torch.stack([r, g, b], dim=1).clamp(0, 1)


def hue_rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate hue by a FIXED number of degrees, preserving geometry.

    ML CONCEPT — CHANGING (not removing) A CUE:
      Grayscale *removes* color; hue rotation *changes* it while keeping shape,
      saturation and brightness. Together they answer two different questions:
      does the model need color at all (grayscale), and is it sensitive to which
      colors appear (hue rotation)? A model with a strong 'color-name' prior
      (e.g. 'yellow-ish blob => banana') will wobble under hue rotation even
      though the object geometry is unchanged.
    """
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    hsv = _rgb_to_hsv(x)                          # to HSV
    shift = (degrees % 360) / 360.0              # convert degrees to [0,1) hue units
    hsv[:, 0] = (hsv[:, 0] + shift) % 1.0        # rotate ONLY the hue channel
    out = _hsv_to_rgb(hsv)                        # back to RGB
    return out.squeeze(0) if single else out


# -----------------------------------------------------------------------------
# 3. TRANSLATION  (position experiment)
# -----------------------------------------------------------------------------
def translate(x: torch.Tensor, shift: int, direction: str) -> torch.Tensor:
    """Shift the image by `shift` pixels in `direction`, using reflection pad
    then a shifted crop (exactly as the spec prescribes).

    ML CONCEPT — TRANSLATION (IN)VARIANCE:
      Convolutions are *approximately* translation-equivariant, but stride,
      pooling, padding, positional encodings (ViT/CLIP) and the training data
      can all make the final classifier position-sensitive. Shifting the object
      and checking whether the prediction survives directly tests that
      sensitivity. We use REFLECTION padding (mirror the border) rather than
      zero padding so we do not inject an artificial black band that itself
      becomes a spurious cue.
    """
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    if shift == 0:
        return x.squeeze(0) if single else x     # 0-shift == identity (baseline)

    # Map a direction to (dy, dx) pixel offsets. Positive dy moves content DOWN.
    d = {"up": (-shift, 0), "down": (shift, 0),
         "left": (0, -shift), "right": (0, shift)}[direction]
    dy, dx = d

    # Reflection-pad by `shift` on all sides so that after we crop with an
    # offset, every output pixel comes from real (mirrored) image content.
    pad = shift
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")  # (B,C,H+2s,W+2s)

    # The crop's top-left corner starts at (pad+dy, pad+dx); cropping a HxW
    # window from there yields the content displaced by (dy,dx).
    H, W = x.shape[-2], x.shape[-1]
    top = pad + dy
    left = pad + dx
    out = xp[:, :, top:top + H, left:left + W]   # shifted crop back to HxW
    return out.squeeze(0) if single else out


def translate_average_prediction_helper(x, shift, directions):
    """Return the list of translated images for ALL directions at one shift.

    The spec says to "average the results across directions." We do NOT average
    pixels; we produce one translated image per direction and let the caller
    (run_task1.py) evaluate each and average the *metrics*. This helper just
    packages the four directional versions for a given shift.
    """
    return [translate(x, shift, dir_) for dir_ in directions]


# -----------------------------------------------------------------------------
# 4. PATCH SHUFFLE  (global-structure experiment)
# -----------------------------------------------------------------------------
def make_patch_permutation(grid: int, seed: int) -> np.ndarray:
    """Create ONE non-identity permutation of grid*grid patch positions.

    ML CONCEPT — GLOBAL vs LOCAL EVIDENCE:
      Shuffling patches keeps every pixel and all *local* texture, but destroys
      the *global* spatial layout (a face's eyes-above-nose ordering, a car's
      wheels-below-body ordering, etc.). If a model still predicts confidently,
      it is leaning on local texture/statistics rather than coherent global
      shape. The spec fixes the permutation with seed 6304 so all models see the
      same shuffle — the comparison is about the models, not the noise.

    'Non-identity' guarantee: we redraw until at least one patch moves, so we do
    not accidentally test the trivial no-op permutation.
    """
    n = grid * grid                               # e.g. 16 patches for a 4x4 grid
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    while True:
        rng.shuffle(perm)
        if not np.array_equal(perm, np.arange(n)):  # ensure it actually shuffles
            return perm.copy()


def patch_shuffle(x: torch.Tensor, grid: int, perm: np.ndarray) -> torch.Tensor:
    """Apply a fixed patch permutation to an image on a grid x grid layout.

    We slice the 224x224 image into grid*grid equal tiles, reorder the tiles
    according to `perm`, and stitch them back. Because the permutation is passed
    in (not generated here), the SAME shuffle is reused across models and — if
    the caller wants — across images.
    """
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    ph, pw = H // grid, W // grid                 # patch height/width

    # Collect the grid*grid patches in row-major (reading) order.
    patches = []
    for gy in range(grid):
        for gx in range(grid):
            patch = x[:, :, gy*ph:(gy+1)*ph, gx*pw:(gx+1)*pw]
            patches.append(patch)

    # Build the output by placing patch `perm[k]` into slot k.
    out = torch.zeros_like(x)
    for k in range(grid * grid):
        gy, gx = divmod(k, grid)                  # destination slot coordinates
        src = patches[perm[k]]                    # source patch chosen by perm
        out[:, :, gy*ph:(gy+1)*ph, gx*pw:(gx+1)*pw] = src
    return out.squeeze(0) if single else out


# Quick visual/shape sanity check when run standalone.
if __name__ == "__main__":
    dummy = torch.rand(3, 224, 224)               # a fake image in [0,1]
    assert to_grayscale(dummy).shape == (3, 224, 224)
    assert hue_rotate(dummy, 90).shape == (3, 224, 224)
    assert translate(dummy, 16, "right").shape == (3, 224, 224)
    perm = make_patch_permutation(4, 6304)
    assert patch_shuffle(dummy, 4, perm).shape == (3, 224, 224)
    print("all transform shape checks passed; perm =", perm)
