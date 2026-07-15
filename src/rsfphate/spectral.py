"""Spectral clustering utilities used by the public RSF-PHATE pipeline."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian


def yu_shi(
    similarity: np.ndarray,
    n_clusters: int,
    n_iter: int = 100, #numerin de iteraciones para la rotación de Procrustes
    tol: float = 1e-6,
    eps: float = 1e-12,
    drop_isolated: bool = False,
) -> np.ndarray:
    """Discretize a similarity matrix with the Yu-Shi spectral method."""
    S = np.asarray(similarity, dtype=float)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S = 0.5 * (S + S.T)
    S[S < 0] = 0.0

    n_samples = S.shape[0]
    degree = S.sum(axis=1)
    isolated = degree <= eps

    if np.any(isolated):
        if drop_isolated:
            keep = np.where(~isolated)[0]
            drop = np.where(isolated)[0]
            S_reduced = S[np.ix_(keep, keep)]
        else:
            S = S + eps * np.eye(n_samples)
            keep = np.arange(n_samples)
            drop = np.array([], dtype=int)
            S_reduced = S
    else:
        keep = np.arange(n_samples)
        drop = np.array([], dtype=int)
        S_reduced = S

    n_keep = len(keep)
    if n_keep == 0:
        return np.zeros(n_samples, dtype=int)
    if n_clusters > n_keep:
        raise ValueError(f"n_clusters={n_clusters} exceeds the number of non-isolated nodes ({n_keep}).")

    lap = csgraph_laplacian(S_reduced, normed=True)
    lap = 0.5 * (lap + lap.T)
    _, eigenvectors = eigh(lap, subset_by_index=[0, n_clusters - 1])
    embedding = eigenvectors / np.maximum(norm(eigenvectors, axis=1, keepdims=True), eps)

    labels_reduced = np.argmax(embedding, axis=1)
    indicator = np.zeros((n_keep, n_clusters), dtype=float)
    indicator[np.arange(n_keep), labels_reduced] = 1.0

    for _ in range(n_iter):
        M = embedding.T @ indicator
        A, _, B_t = np.linalg.svd(M, full_matrices=False)
        rotation = B_t.T @ A.T
        transformed = embedding @ rotation
        labels_new = np.argmax(transformed, axis=1)
        if np.mean(labels_new != labels_reduced) < tol:
            labels_reduced = labels_new
            break
        labels_reduced = labels_new
        indicator.fill(0.0)
        indicator[np.arange(n_keep), labels_reduced] = 1.0

    labels = np.zeros(n_samples, dtype=int)
    labels[keep] = labels_reduced
    labels[drop] = 0
    return labels
