

import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_stability(backbone, clean_images, transformed_images, batch_size=64):
    
    assert clean_images.shape[0] == transformed_images.shape[0], \
        "clean and transformed sets must be aligned 1-to-1"
    
    sims = []
    N = clean_images.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        c = clean_images[start:end]
        t = transformed_images[start:end]
        fc = backbone.extract_features(c)      
        ft = backbone.extract_features(t)      

        sim = F.cosine_similarity(fc, ft, dim=1)   
        sims.append(sim.cpu())
    sims = torch.cat(sims)                      
    return sims.mean().item()                   


@torch.no_grad()
def cosine_stability_per_class(backbone, clean_images, transformed_images,
                               labels, num_classes, batch_size=64):
   
    device_sims = []
    N = clean_images.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        fc = backbone.extract_features(clean_images[start:end])
        ft = backbone.extract_features(transformed_images[start:end])
        device_sims.append(F.cosine_similarity(fc, ft, dim=1).cpu())
    sims = torch.cat(device_sims)
    labels = torch.as_tensor(labels)
    per_class = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            per_class[c] = sims[mask].mean().item()
    return per_class
