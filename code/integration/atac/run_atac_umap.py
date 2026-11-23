import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"

import sys
import gc
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import scanpy.external as sce
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.stats import spearmanr

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

def log(message):
    print(message, flush=True)

log("Loading ATAC matrix")
a = sc.read_h5ad(inp)
assert a.n_obs == 117653
assert a.n_vars == 439167

obs = a.obs.copy()
X = a.X.tocsr()
del a
gc.collect()

assert np.isfinite(X.data).all()
assert X.data.min() >= 0
assert X.nnz > 0
log(f"Input shape: {X.shape}; nnz: {X.nnz}")

# Store depth before binary transformation.
depth = np.asarray(X.sum(axis=1)).ravel()

# Binary peak accessibility.
X.data[:] = 1
X.eliminate_zeros()

# Remove peaks detected in fewer than 30 cells.
detected = np.asarray(X.getnnz(axis=0)).ravel()
keep = detected >= 30
log(f"Peaks retained: {keep.sum()} / {len(keep)}")

X = X[:, keep].tocsr()
X = X.astype(np.float32)
X.sum_duplicates()
del detected, keep
gc.collect()

# Sparse TF-IDF. Do not convert the peak matrix to dense.
log("Computing TF-IDF")
tfidf = TfidfTransformer(
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False,
)
X = tfidf.fit_transform(X)
del tfidf
gc.collect()

# Sparse truncated SVD / LSI.
log("Computing 51 LSI components")
svd = TruncatedSVD(
    n_components=51,
    n_iter=5,
    random_state=42,
)
lsi = svd.fit_transform(X).astype(np.float32)
variance = svd.explained_variance_ratio_.tolist()
del X, svd
gc.collect()

# Inspect whether component 1 is dominated by read depth.
rho = spearmanr(
    lsi[:, 0], np.log1p(depth)
).statistic
log(f"LSI1 vs log depth Spearman rho: {rho:.4f}")

if abs(rho) > 0.5:
    selected = lsi[:, 1:51]
    log("Excluding depth-associated LSI1")
else:
    selected = lsi[:, :50]
    log("Retaining LSI1")

# Standardize each LSI component.
scale = selected.std(axis=0)
scale[scale == 0] = 1
selected = (
    (selected - selected.mean(axis=0)) / scale
).astype(np.float32)

# Lightweight AnnData containing only metadata and embeddings.
b = sc.AnnData(obs=obs)
b.obsm["X_atac_lsi"] = selected

log("Running Harmony")
sce.pp.harmony_integrate(
    b,
    key="batch",
    basis="X_atac_lsi",
    adjusted_basis="X_atac_harmony",
    max_iter_harmony=20,
)

# UMAP before Harmony.
log("Computing uncorrected ATAC UMAP")
sc.pp.neighbors(
    b,
    n_neighbors=30,
    use_rep="X_atac_lsi",
    key_added="lsi_neighbors",
    random_state=42,
)
sc.tl.umap(
    b,
    neighbors_key="lsi_neighbors",
    min_dist=0.3,
    random_state=42,
)
b.obsm["X_umap_lsi"] = b.obsm["X_umap"].copy()

# Free uncorrected neighbor graph before computing the next one.
for key in ["lsi_neighbors_distances",
            "lsi_neighbors_connectivities"]:
    if key in b.obsp:
        del b.obsp[key]

log("Computing Harmony ATAC UMAP")
sc.pp.neighbors(
    b,
    n_neighbors=30,
    use_rep="X_atac_harmony",
    key_added="harmony_neighbors",
    random_state=42,
)
sc.tl.umap(
    b,
    neighbors_key="harmony_neighbors",
    min_dist=0.3,
    random_state=42,
)
b.obsm["X_umap_harmony"] = b.obsm["X_umap"].copy()

# Save compact coordinates and metadata.
log("Saving coordinates")
result = b.obs.copy()
result["atac_lsi_umap1"] = b.obsm["X_umap_lsi"][:, 0]
result["atac_lsi_umap2"] = b.obsm["X_umap_lsi"][:, 1]
result["atac_harmony_umap1"] = b.obsm["X_umap_harmony"][:, 0]
result["atac_harmony_umap2"] = b.obsm["X_umap_harmony"][:, 1]

result.to_csv(out / "ATAC_UMAP_METADATA.csv.gz")

np.savez_compressed(
    out / "ATAC_EMBEDDINGS.npz",
    lsi=b.obsm["X_atac_lsi"],
    harmony=b.obsm["X_atac_harmony"],
    umap_lsi=b.obsm["X_umap_lsi"],
    umap_harmony=b.obsm["X_umap_harmony"],
)

# Plot all cells using existing annotations.
def plot_umap(coords, labels, title, filename):
    labels = labels.fillna("Unassigned").astype(str)
    categories = labels.value_counts().index.tolist()
    palette = plt.get_cmap("gist_ncar")
    colors = palette(
        np.linspace(0, 0.95, len(categories))
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    for i, category in enumerate(categories):
        mask = (labels.to_numpy() == category)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=0.5,
            alpha=0.65,
            c=[colors[i]],
            rasterized=True,
            label=f"{category} ({mask.sum():,})",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        markerscale=7,
        fontsize=7,
        frameon=False,
    )
    fig.savefig(
        out / filename,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

for embedding, coords in [
    ("LSI", b.obsm["X_umap_lsi"]),
    ("Harmony", b.obsm["X_umap_harmony"]),
]:
    for label in [
        "trbi_level2",
        "gse267584_best_state",
        "source_GSE",
        "gse267584_mapping_status",
    ]:
        if label in result.columns:
            log(f"Plotting {embedding} / {label}")
            plot_umap(
                coords,
                result[label],
                f"ATAC {embedding} UMAP: {label}",
                f"ATAC_{embedding}_{label}.png",
            )

with open(out / "ANALYSIS_SETTINGS.json", "w") as f:
    json.dump({
        "cells": len(result),
        "initial_peaks": 439167,
        "minimum_peak_cells": 30,
        "lsi_components": 50,
        "lsi1_depth_spearman": float(rho),
        "lsi1_excluded": bool(abs(rho) > 0.5),
        "batch_key": "batch",
        "neighbors": 30,
        "umap_min_dist": 0.3,
        "random_seed": 42,
        "lsi_explained_variance_ratio": variance,
    }, f, indent=2)

log("COMPLETE")
