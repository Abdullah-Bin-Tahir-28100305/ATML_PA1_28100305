# =============================================================================
# analysis/representation.py
# -----------------------------------------------------------------------------
# PURPOSE: 2-D VISUALIZATION of high-dimensional representations. For each
# backbone we fit ONE 2-D projection (t-SNE or UMAP) on the COMBINED clean +
# transformed features so both conditions share one coordinate system, then plot
# them with:
#   * COLOR  = ground-truth class,
#   * MARKER = clean vs transformed.
#
# ML CONCEPT — WHY PROJECT AT ALL:
#   Features live in 512/768/2048 dimensions; we cannot see them directly. t-SNE
#   and UMAP are NON-LINEAR dimensionality-reduction methods that place similar
#   high-D points near each other in 2-D. They reveal cluster STRUCTURE:
#     - Do classes form separated clusters (good, linearly-probeable features)?
#     - Do transformed points stay inside their clean class cluster (the
#       transform preserved the representation) or drift away (it disrupted it)?
#
# CRITICAL CAVEATS (straight from the spec, encoded as rules below):
#   * "fit ONE two-dimensional projection to the COMBINED clean and transformed
#     features so that both conditions appear in the same space" -> we fit on the
#     stacked matrix, never separately per condition.
#   * "neither method preserves every distance from the original feature space"
#     -> distances/axes are NOT quantitatively meaningful; only relative
#     grouping is. We therefore never read absolute coordinates.
#   * "do not compare absolute coordinates across projections fitted separately
#     for different backbones" -> each backbone gets its OWN plot; we never
#     overlay coordinates from different projections.
#   * "Keep the image subset and random seed fixed, report the visualization
#     settings" -> seed + settings are passed in and recorded.
#
# LINKS TO OTHER FILES:
#   - Consumes features from models/backbones.py.
#   - Uses the same fixed subset and seed as everything else (utils/config).
#   - Produces the t-SNE/UMAP figures listed under Required Evidence.
# =============================================================================

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # non-interactive backend: save figures to file
import matplotlib.pyplot as plt


def _fit_projection(features_2d_input, method, cfg, seed):
    """Fit a 2-D embedding of a stacked feature matrix. Returns (N,2) array.

    method: 'tsne' or 'umap'. We keep t-SNE as the dependency-free default
    (scikit-learn ships it); UMAP is used only if the umap-learn package is
    present, otherwise we transparently fall back to t-SNE and say so.
    """
    X = features_2d_input                       # (N, D) numpy array
    if method == "umap":
        try:
            import umap                          # optional dependency
            reducer = umap.UMAP(
                n_neighbors=cfg["visualization"]["umap_n_neighbors"],
                min_dist=cfg["visualization"]["umap_min_dist"],
                random_state=seed,               # fixed seed => reproducible layout
            )
            return reducer.fit_transform(X), "umap"
        except Exception as e:
            print(f"[warn] UMAP unavailable ({e}); falling back to t-SNE.")

    # Default / fallback: t-SNE from scikit-learn.
    from sklearn.manifold import TSNE
    import inspect
    # perplexity must be < n_samples; guard for small subsets.
    perplexity = min(cfg["visualization"]["perplexity"], max(5, (X.shape[0] - 1) // 3))
    # scikit-learn renamed the iterations argument from `n_iter` (<=1.4) to
    # `max_iter` (>=1.5). We detect which name this installed version accepts so
    # the code runs on either — a small compatibility shim, not a logic change.
    tsne_params = inspect.signature(TSNE.__init__).parameters
    iter_kw = "max_iter" if "max_iter" in tsne_params else "n_iter"
    kwargs = {
        "n_components": 2,                       # project to 2-D
        "perplexity": perplexity,               # neighborhood size (reported)
        iter_kw: cfg["visualization"]["n_iter"],# optimization steps (reported)
        "init": "pca",                          # PCA init: stabler, faster t-SNE
        "random_state": seed,                   # fixed seed => reproducible
    }
    reducer = TSNE(**kwargs)
    return reducer.fit_transform(X), "tsne"


@torch.no_grad()
def visualize_backbone(backbone, clean_images, transformed_images, labels,
                       class_names, cfg, transform_name, out_dir):
    """Make and save ONE 2-D projection plot for a single backbone.

    Steps:
      1) extract clean + transformed features,
      2) STACK them (clean on top, transformed below) so one projection covers
         both conditions in a shared space,
      3) fit t-SNE/UMAP on the stacked matrix,
      4) split the result back into clean/transformed halves,
      5) scatter: color=class, marker=condition, and save the figure.
    """
    # 1) Features for both conditions.
    fc = backbone.extract_features(clean_images).cpu().numpy()        # (N, D)
    ft = backbone.extract_features(transformed_images).cpu().numpy()  # (N, D)
    N = fc.shape[0]

    # 2) Stack: rows [0:N] are clean, rows [N:2N] are transformed.
    stacked = np.concatenate([fc, ft], axis=0)                        # (2N, D)

    # 3) Fit ONE projection on the combined matrix (spec requirement).
    emb, used = _fit_projection(stacked, cfg["visualization"]["method"], cfg, cfg["seed"])

    # 4) Split back into the two conditions.
    emb_clean = emb[:N]
    emb_trans = emb[N:]
    labels = np.asarray(labels)

    # 5) Plot. Color encodes class; marker 'o' = clean, 'x' = transformed.
    plt.figure(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")                 # 10 distinct colors for 10 classes
    for c in range(len(class_names)):
        m = (labels == c)
        if m.any():
            plt.scatter(emb_clean[m, 0], emb_clean[m, 1],
                        color=cmap(c % 10), marker="o", s=18, alpha=0.7,
                        label=f"{class_names[c]} (clean)")
            plt.scatter(emb_trans[m, 0], emb_trans[m, 1],
                        color=cmap(c % 10), marker="x", s=22, alpha=0.7)
    # Title records the settings so the figure is self-documenting (spec:
    # "report the visualization settings").
    plt.title(f"{backbone.name} — {transform_name}\n"
              f"{used} (seed={cfg['seed']}, "
              f"{'perp='+str(cfg['visualization']['perplexity']) if used=='tsne' else 'n_neighbors='+str(cfg['visualization']['umap_n_neighbors'])}) "
              f"| o=clean  x=transformed")
    plt.xlabel("dim 1 (arbitrary units — distances not quantitative)")
    plt.ylabel("dim 2 (arbitrary units — distances not quantitative)")
    # A compact legend (classes only; markers explained in the title).
    plt.legend(fontsize=7, markerscale=1.2, ncol=2, loc="best")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"proj_{backbone.name}_{transform_name}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname
