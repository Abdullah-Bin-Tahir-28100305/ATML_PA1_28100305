# =============================================================================
# analysis/feature_similarity.py
# -----------------------------------------------------------------------------
# PURPOSE: measure REPRESENTATION stability — how much a backbone's feature
# vector moves when an image is transformed, even if the PREDICTION is unchanged.
# Implements the spec's cosine-stability index:
#
#     I_T = (1/N) * sum_i  [ f(x_i) . f(T(x_i)) ] / ( ||f(x_i)|| * ||f(T(x_i))|| )
#
# i.e. the AVERAGE COSINE SIMILARITY between the clean feature and the
# transformed feature, over paired (clean, transformed) images.
#
# ML CONCEPT — REPRESENTATION vs PREDICTION STABILITY:
#   The spec is explicit: "Prediction stability and representation stability
#   answer different questions. A cue may remain encoded even when the
#   classifier does not use it, or a prediction may remain unchanged despite a
#   large representation shift." Prediction consistency (evaluate_bias.py) asks
#   'did the decision flip?'. I_T asks 'did the underlying feature vector move?'.
#   A high I_T means the representation is (nearly) invariant to the transform;
#   a low I_T means the transform changed the encoding a lot even if the final
#   label happened to survive. Comparing the two is a core Research Question.
#
# WHY COSINE (not Euclidean distance):
#   Cosine similarity compares DIRECTION, ignoring magnitude. Linear classifiers
#   and CLIP's cosine rule both care about direction, so cosine is the natural
#   measure of 'has the representation meaningfully changed'. It is also
#   bounded in [-1, 1], making values comparable across backbones of different
#   feature dimensionality.
#
# LINKS TO OTHER FILES:
#   - Consumes features from models/backbones.py (extract_features).
#   - Pairs each clean image with its transformed counterpart produced by
#     data/transforms.py or data/make_cue_conflicts.py.
#   - Results feed the representation-stability table required as evidence.
# =============================================================================

import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_stability(backbone, clean_images, transformed_images, batch_size=64):
    """Compute I_T: mean cosine similarity between paired clean/transformed feats.

    Args:
      backbone: any Backbone (ResNet/ViT/CLIP) exposing extract_features.
      clean_images:       (N,3,224,224) tensor in [0,1].
      transformed_images: (N,3,224,224) tensor in [0,1], ALIGNED row-for-row
                          with clean_images (image i must pair with image i).
    Returns:
      float I_T in [-1, 1].

    We process in batches to bound memory, extracting features for the clean and
    transformed versions of the SAME indices together so the pairing is exact.
    """
    assert clean_images.shape[0] == transformed_images.shape[0], \
        "clean and transformed sets must be aligned 1-to-1"
    # NOTE: we do NOT move images to a device here. Each backbone's
    # extract_features() already moves its input to the backbone's own device
    # (ImageNet backbones) or to CLIP's device internally, so passing CPU tensors
    # is safe and keeps this function backbone-agnostic.
    sims = []
    N = clean_images.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        c = clean_images[start:end]
        t = transformed_images[start:end]
        fc = backbone.extract_features(c)      # (b, D) clean features
        ft = backbone.extract_features(t)      # (b, D) transformed features
        # cosine_similarity computes the per-row cosine; dim=1 => per image.
        sim = F.cosine_similarity(fc, ft, dim=1)   # (b,)
        sims.append(sim.cpu())
    sims = torch.cat(sims)                      # (N,)
    return sims.mean().item()                   # the scalar I_T


@torch.no_grad()
def cosine_stability_per_class(backbone, clean_images, transformed_images,
                               labels, num_classes, batch_size=64):
    """Optional finer view: I_T broken down per class.

    Useful for discussion — some classes' representations may be far more
    transform-stable than others (e.g. rigid objects vs deformable animals).
    Not required by the spec but cheap and informative for the report.
    """
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
