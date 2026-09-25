# =============================================================================
# models/backbones.py
# -----------------------------------------------------------------------------
# PURPOSE: wrap the three pretrained backbones behind ONE common interface so
# the rest of the code can treat them interchangeably:
#   * ResNet-50  (torchvision, ImageNet1K_V2)  -> global-average-pooled feature
#   * ViT-B/16   (torchvision, ImageNet1K_V1)  -> final class token
#   * CLIP ViT-B/32 (OpenCLIP, pretrained='openai') -> normalized image embedding
# Plus: a linear classifier HEAD, the training loop (linear probe), and CLIP
# ZERO-SHOT classification.
#
# ML CONCEPTS ANCHORED HERE:
#   * FROZEN BACKBONE + LINEAR PROBE: we never update the backbone. We extract
#     fixed features and train only a linear layer. This measures the QUALITY of
#     the pretrained representation (how linearly separable it already is),
#     which is precisely what lets us compare architectures/pretraining fairly.
#   * PER-MODEL NORMALIZATION: each backbone was pretrained with specific
#     input statistics. We apply each model's OWN normalization at the last
#     moment, AFTER all interventions, so interventions live in a shared pixel
#     space (spec requirement) yet each model still receives the distribution it
#     expects.
#   * CLIP ZERO-SHOT: CLIP aligns image and text in one embedding space, so we
#     can classify with NO trained head by comparing an image embedding to text
#     embeddings of "a photo of a {class}." This is a fundamentally different
#     decision rule from a trained linear head, and comparing the two is a
#     Research Question in the assignment.
#
# LINKS TO OTHER FILES:
#   - utils: device + seeding (linear-head init uses the seed).
#   - Consumes the shared [0,1] images produced by data/transforms.py and
#     data/make_cue_conflicts.py (it normalizes them internally).
#   - Produces the features that analysis/feature_similarity.py and
#     analysis/representation.py analyze, and the predictions that
#     analysis/evaluate_bias.py scores.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

try:
    from utils import get_device, set_seed
except ImportError:
    from ..utils import get_device, set_seed

# OpenCLIP is imported lazily inside the CLIP wrapper so the ResNet/ViT path
# still works even if open_clip is not installed.


# -----------------------------------------------------------------------------
# Normalization constants
# -----------------------------------------------------------------------------
# ImageNet statistics used by BOTH torchvision backbones. Applied at the very
# end so interventions happen in the raw [0,1] space first.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# CLIP uses its OWN normalization statistics (different from ImageNet). Using
# the wrong stats silently degrades CLIP, so we keep them separate.
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _normalize(x, mean, std):
    """Apply (x - mean) / std with broadcasting, on x's device."""
    return (x - mean.to(x.device)) / std.to(x.device)


# -----------------------------------------------------------------------------
# Base class: a common interface every backbone implements
# -----------------------------------------------------------------------------
class Backbone(nn.Module):
    """Common interface: .extract_features(images_0to1) -> (N, D) feature tensor.

    Every wrapper receives images in the SHARED [0,1] 224x224 space, does its
    OWN normalization internally, and returns a fixed-length feature vector.
    Because they all expose the same method, downstream code is model-agnostic.
    """
    feature_dim: int = None       # set by subclasses; width D of the feature
    name: str = "backbone"        # short label used in result tables

    @torch.no_grad()
    def extract_features(self, images):
        raise NotImplementedError


# -----------------------------------------------------------------------------
# ResNet-50 wrapper
# -----------------------------------------------------------------------------
class ResNetBackbone(Backbone):
    """ResNet-50, feature = global-average-pooled final conv map (2048-D).

    ML CONCEPT — WHAT THE FEATURE IS:
      A CNN builds a stack of convolutional feature maps; the last block outputs
      a 2048x7x7 map. Global Average Pooling (GAP) averages each channel over
      space to a single number, giving a 2048-vector. GAP discards absolute
      position but keeps 'how much of each learned pattern is present', which is
      the standard ResNet embedding fed to its classifier — exactly what the
      spec means by "the global-average-pooled ResNet feature."
    """
    name = "resnet50"

    def __init__(self):
        super().__init__()
        # IMAGENET1K_V2 is the improved-recipe weight set the spec names.
        net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the final fully-connected classifier with Identity so the
        # network OUTPUTS the pre-classifier feature (post-GAP) rather than
        # 1000 ImageNet logits. This is how we 'read off' the representation.
        self.feature_dim = net.fc.in_features        # 2048
        net.fc = nn.Identity()
        self.net = net.eval()                        # inference mode
        for p in self.net.parameters():
            p.requires_grad = False                  # FROZEN backbone

    @torch.no_grad()
    def extract_features(self, images):
        # images: (N,3,224,224) in [0,1] -> normalize with ImageNet stats.
        # Move input to the SAME device as the network so CPU/GPU never mismatch.
        images = images.to(next(self.net.parameters()).device)
        x = _normalize(images, IMAGENET_MEAN, IMAGENET_STD)
        return self.net(x)                           # (N, 2048)


# -----------------------------------------------------------------------------
# ViT-B/16 wrapper
# -----------------------------------------------------------------------------
class ViTBackbone(Backbone):
    """ViT-B/16, feature = final class ([CLS]) token (768-D).

    ML CONCEPT — WHAT THE FEATURE IS:
      A Vision Transformer splits the image into 16x16 patches, embeds each as a
      token, prepends a learnable [CLS] token, adds POSITIONAL ENCODINGS, and
      passes everything through self-attention layers. The final [CLS] token
      aggregates global information via attention and is the vector fed to the
      classifier — the spec's "final ViT class token." Note ViT has explicit
      positional encodings, which is exactly why the translation/patch-shuffle
      experiments are interesting for it.
    """
    name = "vit_b16"

    def __init__(self):
        super().__init__()
        net = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.feature_dim = net.heads.head.in_features  # 768
        # Replace the classification head with Identity so forward() returns the
        # [CLS]-token feature that normally feeds the head.
        net.heads = nn.Identity()
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad = False                    # FROZEN backbone

    @torch.no_grad()
    def extract_features(self, images):
        # Move input to the network's device to avoid CPU/GPU mismatches.
        images = images.to(next(self.net.parameters()).device)
        x = _normalize(images, IMAGENET_MEAN, IMAGENET_STD)
        return self.net(x)                             # (N, 768) = [CLS] token


# -----------------------------------------------------------------------------
# CLIP wrapper (OpenCLIP ViT-B/32, pretrained='openai')
# -----------------------------------------------------------------------------
class CLIPBackbone(Backbone):
    """CLIP image encoder; feature = L2-normalized image embedding (512-D).

    ML CONCEPT — CONTRASTIVE VISION-LANGUAGE PRETRAINING:
      CLIP was trained to pull matching (image, caption) pairs together and push
      mismatched ones apart in a shared embedding space. The result is an image
      embedding that is directly comparable to TEXT embeddings via cosine
      similarity. We L2-normalize it (spec: "the normalized CLIP image
      embedding") because CLIP's decision rule is cosine similarity, which only
      depends on direction, not magnitude.
    """
    name = "clip_vitb32"

    def __init__(self, cfg):
        super().__init__()
        import open_clip                              # lazy import
        self.device = get_device()
        model_name = cfg["clip"]["model_name"]        # 'ViT-B-32'
        pretrained = cfg["clip"]["pretrained"]        # 'openai'
        # create_model_and_transforms returns the model and its OWN preprocess;
        # we only need the model because we normalize manually to keep the shared
        # [0,1] intervention space consistent with the other backbones.
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)  # for zero-shot text
        self.model = self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False                   # FROZEN backbone
        self.feature_dim = self.model.visual.output_dim  # 512
        self.prompt_template = cfg["clip"]["prompt_template"]  # "a photo of a {}."

    @torch.no_grad()
    def extract_features(self, images):
        # Normalize with CLIP's OWN mean/std (not ImageNet's!).
        x = _normalize(images.to(self.device), CLIP_MEAN, CLIP_STD)
        feats = self.model.encode_image(x)            # (N, 512)
        # L2-normalize: put every embedding on the unit sphere so downstream
        # cosine comparisons are well-defined (this is the 'normalized' feature).
        feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def zero_shot_logits(self, images, class_names):
        """CLIP ZERO-SHOT classification with the fixed prompt.

        ML CONCEPT — ZERO-SHOT VIA TEXT PROMPTS:
          We embed each class name as "a photo of a {class}.", L2-normalize the
          text embeddings, and score each image by cosine similarity to every
          class's text embedding. We multiply by CLIP's learned temperature
          (logit_scale) before softmax so the confidences are calibrated the way
          CLIP was trained. NO trained head is involved — the 'classifier' is
          the set of text embeddings. (Spec: "evaluate CLIP zero-shot using the
          fixed prompt 'a photo of a {class}.'" and "compute confidence from the
          softmax over scaled class similarities".)
        """
        # Build and embed the text prompts, one per class.
        prompts = [self.prompt_template.format(c) for c in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        text_feats = self.model.encode_text(tokens)   # (num_classes, 512)
        text_feats = F.normalize(text_feats, dim=-1)  # unit sphere

        img_feats = self.extract_features(images)      # (N, 512), already norm.
        # Cosine similarities = dot products of unit vectors. Scale by CLIP's
        # temperature so the softmax matches CLIP's training-time calibration.
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.t()   # (N, num_classes)
        return logits


# -----------------------------------------------------------------------------
# Linear classifier head (the 'probe')
# -----------------------------------------------------------------------------
class LinearHead(nn.Module):
    """A single linear layer mapping a frozen feature to class logits.

    ML CONCEPT — LINEAR PROBE / LINEAR SEPARABILITY:
      Because the backbone is frozen, all learning happens in THIS one matrix.
      If a linear layer can separate the classes well, the representation
      already encodes the task in a linearly-accessible way. Comparing probe
      accuracy across backbones compares REPRESENTATION quality, not the ability
      to learn new features. Bias term included (affine decision boundary).
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)      # W (num_classes x in_dim) + b

    def forward(self, feats):
        return self.fc(feats)                          # logits (N, num_classes)


# -----------------------------------------------------------------------------
# Feature caching + linear-head training
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_all_features(backbone, dataset, batch_size, device):
    """Run the frozen backbone over an entire dataset once, caching features.

    ML EFFICIENCY NOTE:
      Since the backbone never changes during probe training, its outputs are
      constant. We compute them ONCE and then train the linear head on the cached
      feature vectors — dramatically faster than re-running the CNN/ViT every
      epoch. This is the standard 'extract-then-probe' recipe.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        f = backbone.extract_features(imgs).cpu()      # (b, D) cached on CPU
        feats.append(f)
        labels.append(ys)
    return torch.cat(feats), torch.cat(labels)


def train_linear_head(backbone, train_ds, val_ds, cfg):
    """Train ONE linear head on frozen features with early stopping.

    Implements the spec exactly: AdamW, lr=1e-3, weight_decay=1e-4, <=50 epochs,
    early stopping after 5 epochs without improved VALIDATION accuracy, seed 6304.

    ML CONCEPTS:
      * EARLY STOPPING is a regularizer: we monitor validation accuracy (a proxy
        for generalization) and keep the weights from the best epoch, preventing
        the head from overfitting the training features.
      * ADAMW decouples weight decay from the adaptive gradient step, giving
        cleaner L2 regularization than plain Adam.
    """
    set_seed(cfg["seed"])                              # deterministic head init
    device = get_device()
    backbone = backbone.to(device)

    # 1) Cache features for train and val once (backbone is frozen).
    Xtr, ytr = extract_all_features(backbone, train_ds, cfg["train"]["batch_size"], device)
    Xva, yva = extract_all_features(backbone, val_ds,   cfg["train"]["batch_size"], device)

    # 2) Build the linear head and optimizer.
    head = LinearHead(backbone.feature_dim, cfg["dataset"]["num_classes"]).to(device)
    opt = torch.optim.AdamW(head.parameters(),
                            lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    crit = nn.CrossEntropyLoss()                       # standard multiclass loss

    # 3) Mini-batch training loop over cached features.
    from torch.utils.data import TensorDataset, DataLoader
    tr_loader = DataLoader(TensorDataset(Xtr, ytr),
                           batch_size=cfg["train"]["batch_size"], shuffle=True)

    best_val, best_state, epochs_no_improve = -1.0, None, 0
    history = []
    for epoch in range(cfg["train"]["max_epochs"]):
        head.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()                            # clear old gradients
            loss = crit(head(xb), yb)                  # forward + loss
            loss.backward()                            # backprop into the head only
            opt.step()                                 # AdamW update

        # Validation accuracy for early-stopping decisions.
        head.eval()
        with torch.no_grad():
            val_pred = head(Xva.to(device)).argmax(1).cpu()
        val_acc = (val_pred == yva).float().mean().item()
        history.append({"epoch": epoch, "val_acc": val_acc})

        # Keep the best-so-far weights; count epochs since the last improvement.
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["train"]["early_stop_patience"]:
                break                                  # early stop (patience hit)

    head.load_state_dict(best_state)                   # restore best epoch
    return head, {"best_val_acc": best_val, "history": history}


# -----------------------------------------------------------------------------
# Factory: build all backbones from config
# -----------------------------------------------------------------------------
def build_backbones(cfg):
    """Instantiate the three backbones. Returns a dict name -> Backbone.

    CLIP is optional-safe: if open_clip is missing we warn and skip it so the
    ResNet/ViT pipeline still runs. (You will want CLIP installed for the full
    comparison — see README.)
    """
    out = {"resnet50": ResNetBackbone(), "vit_b16": ViTBackbone()}
    try:
        out["clip_vitb32"] = CLIPBackbone(cfg)
    except Exception as e:
        print(f"[warn] CLIP backbone unavailable ({e}); continuing without it.")
    return out
