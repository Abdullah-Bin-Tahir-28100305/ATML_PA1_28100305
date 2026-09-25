

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



def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    
    B, C = feat.shape[:2]
    feat_flat = feat.view(B, C, -1)                     # flatten H*W
    mean = feat_flat.mean(dim=2).view(B, C, 1, 1)       # per-channel mean
    std = (feat_flat.var(dim=2) + eps).sqrt().view(B, C, 1, 1)  # per-channel std
    return mean, std


def adaptive_instance_normalization(content_feat, style_feat):
   
    c_mean, c_std = _calc_mean_std(content_feat)
    s_mean, s_std = _calc_mean_std(style_feat)
    normalized = (content_feat - c_mean) / c_std        # whiten content
    return normalized * s_std + s_mean                  # apply style stats



def _build_naoto_vgg():
    
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



class VGGDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
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



class AdaINStylizer:

    def __init__(self, cfg, decoder_weights_path=None, vgg_weights_path=None):
        self.device = get_device()
        self.alpha = cfg["cue_conflict"]["style_strength"]   # style strength

        
        self.encoder = VGGEncoder(vgg_weights_path=vgg_weights_path).to(self.device)
        self.decoder = VGGDecoder().to(self.device)
        self.has_decoder_weights = False

        decoder_ok = False
        if decoder_weights_path and os.path.exists(decoder_weights_path):
            state = torch.load(decoder_weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            if not any(k.startswith("decoder.") for k in state):
                state = {f"decoder.{k}": v for k, v in state.items()}
            self.decoder.load_state_dict(state)
            decoder_ok = True

        self.has_decoder_weights = decoder_ok and self.encoder.authentic
        if decoder_ok and not self.encoder.authentic:
            print("[warn] decoder weights loaded but VGG encoder weights are "
                  "missing/incompatible; the repo decoder needs the repo encoder. "
                  "Falling back to the decoder-free stylizer to avoid garbage. "
                  "Pass --vgg_weights <vgg_normalised.pth> to enable authentic AdaIN.")
        self.decoder.eval()

    @torch.no_grad()                                   # inference only, no grads
    def stylize(self, content, style):
       
        content = content.to(self.device)
        style = style.to(self.device)
        if content.dim() == 3:
            content = content.unsqueeze(0)
        if style.dim() == 3:
            style = style.unsqueeze(0)

        if self.has_decoder_weights:

            c_feat = self.encoder(content)
            s_feat = self.encoder(style)
            t = adaptive_instance_normalization(c_feat, s_feat)
            t = self.alpha * t + (1 - self.alpha) * c_feat   # style-strength mix
            out = self.decoder(t)
            return out.clamp(0, 1).squeeze(0).cpu()
        else:
            
            return self._stylize_fallback(content, style).squeeze(0).cpu()

    def _stylize_fallback(self, content, style):
        
        c_mean, c_std = _calc_mean_std(content)
        s_mean, s_std = _calc_mean_std(style)
        recolored = (content - c_mean) / c_std * s_std + s_mean

       
        k = _gaussian_kernel(5, 1.0).to(self.device)
        def blur(img):
            img_p = F.pad(img, (2, 2, 2, 2), mode="reflect")
            return F.conv2d(img_p, k, groups=3)
        style_hf = style - blur(style)             # style texture detail
        content_lf = blur(recolored)               # content structure (smoothed)
        mixed = content_lf + self.alpha * style_hf # shape + texture blend
        return mixed.clamp(0, 1)


def _gaussian_kernel(ksize, sigma):
    ax = torch.arange(ksize) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1)  # (3,1,k,k)



def _passes_rejection_rule(img: torch.Tensor, content: torch.Tensor, cfg) -> bool:
    
    std = img.std().item()
    change = (img - content).abs().mean().item()
    if std < cfg["cue_conflict"]["reject"]["min_std"]:
        return False                               # collapsed / flat
    if change < cfg["cue_conflict"]["reject"]["min_change"]:
        return False                               # style did not take
    return True



def generate_cue_conflicts(cfg, decoder_weights_path=None, vgg_weights_path=None):
   
    set_seed(cfg["seed"])
    device = get_device()
    train_ds, _, class_names = load_datasets(cfg)   # use TRAIN images as raw material
    labels = _labels_of(train_ds)
    rng = np.random.default_rng(cfg["seed"])

    class_to_indices = {}
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        class_to_indices[int(c)] = idx_c.tolist()

    
    all_pairs = list(itertools.combinations(range(len(class_names)), 2))
    rng.shuffle(all_pairs)
    pairs = all_pairs[: cfg["cue_conflict"]["num_pairs"]]

    stylizer = AdaINStylizer(cfg, decoder_weights_path, vgg_weights_path)

    target_total = cfg["cue_conflict"]["target_valid"]
    n_cells = len(pairs) * 2                          # 2 directions per pair
    per_cell = int(np.ceil(target_total / n_cells))  # even quota per cell

    images, meta = [], []
    
    content_images = []
    accepted, rejected = 0, 0

    def draw_image(class_id, cursor):

        idx_list = class_to_indices[class_id]
        ds_index = idx_list[cursor % len(idx_list)]  # wrap around if exhausted
        img, _ = train_ds[ds_index]
        return img

    for (a, b) in pairs:

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

                if _passes_rejection_rule(styl, content, cfg):
                    images.append(styl)
                    
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

    
    save_cue_conflict_samples(images, meta, out_dir,
                              n_individual=cfg["cue_conflict"].get("n_saved_samples", 40))

    return {"images": images, "content_images": content_images,
            "meta": meta, "log": log, "class_names": class_names}


def save_cue_conflict_samples(images, meta, results_dir, n_individual=40):
   
    import numpy as np
    if images is None or images.numel() == 0:
        print("[warn] no cue-conflict images to save.")
        return None
    sample_dir = ensure_dir(os.path.join(results_dir, "cue_conflict_samples"))

    try:
        from PIL import Image
        k = min(n_individual, images.shape[0])
        for i in range(k):
            arr = (images[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            fname = f"{i:03d}_shape-{meta[i]['shape_name']}_tex-{meta[i]['texture_name']}.png"
            Image.fromarray(arr).save(os.path.join(sample_dir, fname))
    except Exception as e:
        print(f"[warn] could not save individual PNGs ({e}); montage still attempted.")

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
