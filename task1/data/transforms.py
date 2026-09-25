
import torch
import torch.nn.functional as F        # for grid_sample / padding utilities
import numpy as np


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)                       # (C,H,W) -> (1,C,H,W)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]          # split channels
    y = 0.299 * r + 0.587 * g + 0.114 * b        # luminance (B,H,W)
    y = y.unsqueeze(1)                           # (B,1,H,W)
    out = y.repeat(1, 3, 1, 1)                   # replicate to (B,3,H,W)
    return out.squeeze(0) if single else out



def _rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
   
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)                        # V (value) = max channel
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc                          # chroma

    s = torch.where(maxc > 0, delta / (maxc + 1e-8), torch.zeros_like(maxc))

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
   
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    hsv = _rgb_to_hsv(x)                          # to HSV
    shift = (degrees % 360) / 360.0              # convert degrees to [0,1) hue units
    hsv[:, 0] = (hsv[:, 0] + shift) % 1.0        # rotate ONLY the hue channel
    out = _hsv_to_rgb(hsv)                        # back to RGB
    return out.squeeze(0) if single else out



def translate(x: torch.Tensor, shift: int, direction: str) -> torch.Tensor:
    
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    if shift == 0:
        return x.squeeze(0) if single else x     # 0-shift == identity (baseline)

    d = {"up": (-shift, 0), "down": (shift, 0),
         "left": (0, -shift), "right": (0, shift)}[direction]
    dy, dx = d

    
    pad = shift
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")  # (B,C,H+2s,W+2s)

    
    H, W = x.shape[-2], x.shape[-1]
    top = pad + dy
    left = pad + dx
    out = xp[:, :, top:top + H, left:left + W]   # shifted crop back to HxW
    return out.squeeze(0) if single else out


def translate_average_prediction_helper(x, shift, directions):
   
    return [translate(x, shift, dir_) for dir_ in directions]


# -----------------------------------------------------------------------------
# 4. PATCH SHUFFLE  (global-structure experiment)
# -----------------------------------------------------------------------------
def make_patch_permutation(grid: int, seed: int) -> np.ndarray:
    
    n = grid * grid                               # e.g. 16 patches for a 4x4 grid
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    while True:
        rng.shuffle(perm)
        if not np.array_equal(perm, np.arange(n)):  # ensure it actually shuffles
            return perm.copy()


def patch_shuffle(x: torch.Tensor, grid: int, perm: np.ndarray) -> torch.Tensor:
    
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    ph, pw = H // grid, W // grid                 # patch height/width

    patches = []
    for gy in range(grid):
        for gx in range(grid):
            patch = x[:, :, gy*ph:(gy+1)*ph, gx*pw:(gx+1)*pw]
            patches.append(patch)

    out = torch.zeros_like(x)
    for k in range(grid * grid):
        gy, gx = divmod(k, grid)                  # destination slot coordinates
        src = patches[perm[k]]                    # source patch chosen by perm
        out[:, :, gy*ph:(gy+1)*ph, gx*pw:(gx+1)*pw] = src
    return out.squeeze(0) if single else out


if __name__ == "__main__":
    dummy = torch.rand(3, 224, 224)               # a fake image in [0,1]
    assert to_grayscale(dummy).shape == (3, 224, 224)
    assert hue_rotate(dummy, 90).shape == (3, 224, 224)
    assert translate(dummy, 16, "right").shape == (3, 224, 224)
    perm = make_patch_permutation(4, 6304)
    assert patch_shuffle(dummy, 4, perm).shape == (3, 224, 224)
    print("all transform shape checks passed; perm =", perm)
