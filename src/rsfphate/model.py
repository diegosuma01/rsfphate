"""Public estimator API for RSF-PHATE survival clustering."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering

from .forest import RFGAPSurvival, SurvPageRankPHATE, diffusion_similarity
from .spectral import yu_shi


def to_survival_array(time, event) -> np.ndarray:
    """Build a structured survival target array compatible with scikit-survival."""
    time_arr = np.asarray(time, dtype=float)
    event_arr = np.asarray(event, dtype=bool)
    if time_arr.shape[0] != event_arr.shape[0]:
        raise ValueError("time and event must have the same length.")
    y = np.zeros(time_arr.shape[0], dtype=[("event", bool), ("time", float)])
    y["event"] = event_arr
    y["time"] = time_arr
    return y


class RSFPhate(BaseEstimator, ClusterMixin):
    """End-to-end estimator for RSF-PHATE survival clustering.

    The estimator fits a path-aware random survival forest, builds a path-based
    proximity matrix, applies diffusion smoothing, embeds the samples with PHATE,
    and clusters the smoothed similarity matrix.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_estimators: int = 100,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
        lca_weight: float = -1.0,
        diffusion_time: float = 3.0,
        teleportation: float = 1.0,
        phate_n_components: int = 2,
        phate_knn: int = 5,
        phate_t: int | str = "auto",
        embedding_input: Literal["proximity", "smoothed_proximity"] = "proximity",
        cluster_method: Literal["yu-shi", "spectral"] = "yu-shi",
        random_state: int | None = 42,
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.lca_weight = lca_weight
        self.diffusion_time = diffusion_time
        self.teleportation = teleportation
        self.phate_n_components = phate_n_components
        self.phate_knn = phate_knn
        self.phate_t = phate_t
        self.embedding_input = embedding_input
        self.cluster_method = cluster_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, time, event):
        X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        y = to_survival_array(time, event)

        self.rf_model_ = RFGAPSurvival(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            lca_weight=self.lca_weight,
            matrix_type="dense",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.rf_model_.fit(X_array, y)

        self.proximity_ = np.asarray(self.rf_model_.get_proximities(), dtype=np.float32)
        self.smoothed_proximity_ = np.asarray(
            diffusion_similarity(self.proximity_, self.diffusion_time, self.teleportation),
            dtype=np.float32,
        )

        self.phate_operator_ = SurvPageRankPHATE(
            beta=self.teleportation,
            smooth=True,
            tdiff=self.diffusion_time,
            n_components=self.phate_n_components,
            t=self.phate_t,
            knn=self.phate_knn,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        embedding_source = (
            self.proximity_
            if self.embedding_input == "proximity"
            else self.smoothed_proximity_
        )
        self.embedding_ = self.phate_operator_.fit_transform(embedding_source)

        if self.cluster_method == "yu-shi":
            self.labels_ = yu_shi(self.smoothed_proximity_, self.n_clusters)
        elif self.cluster_method == "spectral":
            self.labels_ = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="precomputed",
                random_state=self.random_state,
            ).fit_predict(self.smoothed_proximity_)
        else:
            raise ValueError(f"Unknown cluster_method: {self.cluster_method}")

        return self

    def fit_predict(self, X, time, event):
        self.fit(X, time, event)
        return self.labels_

    def fit_transform(self, X, time, event):
        self.fit(X, time, event)
        return self.embedding_
