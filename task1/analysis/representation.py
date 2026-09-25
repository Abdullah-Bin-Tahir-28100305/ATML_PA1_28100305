

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt


def _fit_projection(features_2d_input, method, cfg, seed):
   
    X = features_2d_input                       
    if method == "umap":
        try:
            import umap                         
            reducer = umap.UMAP(
                n_neighbors=cfg["visualization"]["umap_n_neighbors"],
                min_dist=cfg["visualization"]["umap_min_dist"],
                random_state=seed,               
            )
            return reducer.fit_transform(X), "umap"
        except Exception as e:
            print(f"[warn] UMAP unavailable ({e}); falling back to t-SNE.")

    
    from sklearn.manifold import TSNE
    import inspect

    perplexity = min(cfg["visualization"]["perplexity"], max(5, (X.shape[0] - 1) // 3))
   
    tsne_params = inspect.signature(TSNE.__init__).parameters
    iter_kw = "max_iter" if "max_iter" in tsne_params else "n_iter"
    kwargs = {
        "n_components": 2,                       
        "perplexity": perplexity,               
        iter_kw: cfg["visualization"]["n_iter"],
        "init": "pca",                          
        "random_state": seed,                   
    }
    reducer = TSNE(**kwargs)
    return reducer.fit_transform(X), "tsne"


@torch.no_grad()
def visualize_backbone(backbone, clean_images, transformed_images, labels,
                       class_names, cfg, transform_name, out_dir):
    
    fc = backbone.extract_features(clean_images).cpu().numpy()       
    ft = backbone.extract_features(transformed_images).cpu().numpy()  
    N = fc.shape[0]

    stacked = np.concatenate([fc, ft], axis=0)                        

    emb, used = _fit_projection(stacked, cfg["visualization"]["method"], cfg, cfg["seed"])

    emb_clean = emb[:N]
    emb_trans = emb[N:]
    labels = np.asarray(labels)

    plt.figure(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")                 
    for c in range(len(class_names)):
        m = (labels == c)
        if m.any():
            plt.scatter(emb_clean[m, 0], emb_clean[m, 1],
                        color=cmap(c % 10), marker="o", s=18, alpha=0.7,
                        label=f"{class_names[c]} (clean)")
            plt.scatter(emb_trans[m, 0], emb_trans[m, 1],
                        color=cmap(c % 10), marker="x", s=22, alpha=0.7)
   
    plt.title(f"{backbone.name} — {transform_name}\n"
              f"{used} (seed={cfg['seed']}, "
              f"{'perp='+str(cfg['visualization']['perplexity']) if used=='tsne' else 'n_neighbors='+str(cfg['visualization']['umap_n_neighbors'])}) "
              f"| o=clean  x=transformed")
    plt.xlabel("dim 1 (arbitrary units — distances not quantitative)")
    plt.ylabel("dim 2 (arbitrary units — distances not quantitative)")

    plt.legend(fontsize=7, markerscale=1.2, ncol=2, loc="best")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"proj_{backbone.name}_{transform_name}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname
