

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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _normalize(x, mean, std):
    """Apply (x - mean) / std with broadcasting, on x's device."""
    return (x - mean.to(x.device)) / std.to(x.device)



class Backbone(nn.Module):
   
    feature_dim: int = None       # set by subclasses; width D of the feature
    name: str = "backbone"        # short label used in result tables

    @torch.no_grad()
    def extract_features(self, images):
        raise NotImplementedError



class ResNetBackbone(Backbone):
   
    name = "resnet50"

    def __init__(self):
        super().__init__()
        net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        self.feature_dim = net.fc.in_features        # 2048
        net.fc = nn.Identity()
        self.net = net.eval()                        # inference mode
        for p in self.net.parameters():
            p.requires_grad = False                  # FROZEN backbone

    @torch.no_grad()
    def extract_features(self, images):
        
        images = images.to(next(self.net.parameters()).device)
        x = _normalize(images, IMAGENET_MEAN, IMAGENET_STD)
        return self.net(x)                           # (N, 2048)



class ViTBackbone(Backbone):
   
    name = "vit_b16"

    def __init__(self):
        super().__init__()
        net = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.feature_dim = net.heads.head.in_features  # 768
        
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



class CLIPBackbone(Backbone):
   
    name = "clip_vitb32"

    def __init__(self, cfg):
        super().__init__()
        import open_clip                              # lazy import
        self.device = get_device()
        model_name = cfg["clip"]["model_name"]        # 'ViT-B-32'
        pretrained = cfg["clip"]["pretrained"]        # 'openai'
        
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
       
        prompts = [self.prompt_template.format(c) for c in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        text_feats = self.model.encode_text(tokens)   # (num_classes, 512)
        text_feats = F.normalize(text_feats, dim=-1)  # unit sphere

        img_feats = self.extract_features(images)      # (N, 512), already norm.
        
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.t()   # (N, num_classes)
        return logits


#
class LinearHead(nn.Module):
   
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)      # W (num_classes x in_dim) + b

    def forward(self, feats):
        return self.fc(feats)                          # logits (N, num_classes)



@torch.no_grad()
def extract_all_features(backbone, dataset, batch_size, device):
   
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
    
    set_seed(cfg["seed"])                              # deterministic head init
    device = get_device()
    backbone = backbone.to(device)

    Xtr, ytr = extract_all_features(backbone, train_ds, cfg["train"]["batch_size"], device)
    Xva, yva = extract_all_features(backbone, val_ds,   cfg["train"]["batch_size"], device)

    head = LinearHead(backbone.feature_dim, cfg["dataset"]["num_classes"]).to(device)
    opt = torch.optim.AdamW(head.parameters(),
                            lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    crit = nn.CrossEntropyLoss()                       # standard multiclass loss

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

        head.eval()
        with torch.no_grad():
            val_pred = head(Xva.to(device)).argmax(1).cpu()
        val_acc = (val_pred == yva).float().mean().item()
        history.append({"epoch": epoch, "val_acc": val_acc})

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



def build_backbones(cfg):
   
    out = {"resnet50": ResNetBackbone(), "vit_b16": ViTBackbone()}
    try:
        out["clip_vitb32"] = CLIPBackbone(cfg)
    except Exception as e:
        print(f"[warn] CLIP backbone unavailable ({e}); continuing without it.")
    return out
