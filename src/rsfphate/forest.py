"""Path-aware survival-forest proximity utilities."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from phate import PHATE
from rfgap import RFGAP as _orig_RFGAP
from scipy import sparse
from sklearn.utils.validation import check_is_fitted
from sksurv.ensemble import RandomSurvivalForest

DEFAULT_LCA_WEIGHT = -1.0


def _normalize_matrix(H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a positive kernel to unit diagonal."""
    diag = np.diag(H)
    norm = np.sqrt(np.clip(np.outer(diag, diag), eps, None))
    return H / norm


def diffusion_similarity(
    S0: np.ndarray,
    t_param: Union[int, float],
    beta: float = 1.0,
) -> np.ndarray:
    """Apply heat-kernel diffusion with teleportation to a similarity matrix."""
    A = 0.5 * (S0 + S0.T)
    n = A.shape[0]
    T = np.ones((n, n), dtype=float) / float(n)
    A = beta * A + (1.0 - beta) * T

    if t_param == 0:
        return A

    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    lam_max = max(float(eigenvalues.max()), 1.0)
    t_scaled = float(t_param) / lam_max
    H = (eigenvectors * np.exp(-t_scaled * eigenvalues)) @ eigenvectors.T
    return _normalize_matrix(H)


class RFGAPSurvival(RandomSurvivalForest):
    """Random survival forest with cached path information for path-based proximities."""

    def __init__(
        self,
        prox_method: str = "rfgapLCA",
        prediction_type: str = "survival",
        matrix_type: str = "dense",
        triangular: bool = True,
        non_zero_diagonal: bool = True,
        force_symmetric: bool = True,
        lca_weight: float = DEFAULT_LCA_WEIGHT,
        cache_paths: bool = True,
        **rf_kwargs,
    ):
        super().__init__(**rf_kwargs)
        self.proxy = _orig_RFGAP(
            prox_method="rfgap",
            matrix_type=matrix_type,
            non_zero_diagonal=non_zero_diagonal,
            **rf_kwargs,
        )
        self.prox_method = prox_method
        self.matrix_type = matrix_type
        self.triangular = triangular
        self.non_zero_diagonal = non_zero_diagonal
        self.prediction_type = prediction_type
        self.force_symmetric = force_symmetric
        self.lca_weight = float(lca_weight)
        self.cache_paths = bool(cache_paths)
        self._P_list = None
        self._li_list = None
        self._max_depths = None

    def _precompute_paths(self) -> None:
        n_trees = len(self.estimators_)
        self._P_list = [None] * n_trees
        self._li_list = [None] * n_trees
        self._max_depths = np.zeros(n_trees, dtype=np.int32)
        for t, estimator in enumerate(self.estimators_):
            path_matrix = estimator.decision_path(self._X_all).tocsr()
            path_lengths = np.asarray(path_matrix.getnnz(axis=1)).ravel()
            self._P_list[t] = path_matrix
            self._li_list[t] = path_lengths
            self._max_depths[t] = int(path_lengths.max() - 1)

    def _pairwise_edge_distance_col(self, tree_index: int, sample_index: int) -> np.ndarray:
        path_matrix = self._P_list[tree_index]
        path_lengths = self._li_list[tree_index]
        sample_path = path_matrix.getrow(sample_index).T
        common_nodes = path_matrix @ sample_path
        common_nodes = np.asarray(common_nodes.toarray()).ravel()
        return path_lengths + path_lengths[sample_index] - 2.0 * common_nodes

    def fit(self, X, y, sample_weight=None, x_test=None):
        X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        super().fit(X_array, y, sample_weight)
        self.leaf_matrix = self.apply(X_array)
        self.proxy.estimators_ = self.estimators_
        self.proxy.leaf_matrix = self.leaf_matrix

        if x_test is None:
            self._X_all = X_array
        else:
            self.leaf_matrix_test = self.apply(x_test)
            self.leaf_matrix = np.concatenate((self.leaf_matrix, self.leaf_matrix_test), axis=0)
            self._X_all = np.vstack([X_array, x_test])

        if self.prox_method != "rfgapLCA":
            raise NotImplementedError("Only prox_method='rfgapLCA' is implemented.")

        self.oob_indices = self.proxy.get_oob_indices(X_array)
        self.in_bag_counts = self.proxy.get_in_bag_counts(X_array)

        if x_test is not None:
            n_test = int(np.shape(x_test)[0])
            self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
            self.in_bag_counts = np.concatenate((self.in_bag_counts, np.zeros((n_test, self.n_estimators))))

        self.in_bag_indices = 1 - self.oob_indices
        if self.cache_paths:
            self._precompute_paths()
        return self

    def _prox_vector_one(self, sample_index: int):
        n_samples, _ = self.leaf_matrix.shape
        oob_trees = np.nonzero(self.oob_indices[sample_index, :])[0]
        in_bag_trees = np.nonzero((1 - self.oob_indices[sample_index, :]))[0]

        if len(oob_trees) == 0:
            return [1.0], [sample_index], [sample_index]

        terminals = self.leaf_matrix[sample_index, :]
        matches = (terminals == self.leaf_matrix) & ((1 - self.oob_indices).astype(bool))
        match_counts = np.where(matches, self.in_bag_counts, 0)

        ks = np.sum(match_counts, axis=0)
        ks[ks == 0] = 1
        sim_counts = match_counts.astype(float)

        if self.lca_weight < 0:
            for t in oob_trees:
                denom = int(2 * self._max_depths[t]) if self._max_depths is not None else 0
                if denom <= 0:
                    sim_counts[:, t] = 0.0
                    continue
                edge_distance = self._pairwise_edge_distance_col(t, sample_index)
                similarity = 1.0 - (edge_distance / float(denom))
                similarity = np.clip(similarity, 0.0, 1.0)
                sim_counts[:, t] = similarity * self.in_bag_counts[:, t]
        elif self.lca_weight > 0:
            for t in oob_trees:
                denom = int(2 * self._max_depths[t]) if self._max_depths is not None else 0
                if denom <= 0:
                    continue
                edge_distance = self._pairwise_edge_distance_col(t, sample_index)
                similarity = self.lca_weight * (1.0 - (edge_distance / float(denom)))
                similarity = similarity * self.in_bag_counts[:, t]
                col = sim_counts[:, t]
                np.copyto(col, np.clip(similarity, 0.0, 1.0), where=(col == 0.0))

        if self.non_zero_diagonal and len(in_bag_trees) > 0:
            raw_in = match_counts[sample_index, in_bag_trees] / ks[in_bag_trees]
            diag_val = float(np.sum(raw_in) / len(in_bag_trees))
            for t in in_bag_trees:
                sim_counts[sample_index, t] = diag_val

        ks2 = np.sum(sim_counts, axis=0)
        ks2[ks2 == 0] = 1
        prox_vec = np.sum(sim_counts[:, oob_trees] / ks2[oob_trees], axis=1) / float(len(oob_trees))

        if self.non_zero_diagonal:
            max_value = float(np.max(prox_vec))
            if max_value > 0:
                prox_vec = prox_vec / max_value
            prox_vec[sample_index] = 1.0

        cols = np.nonzero(prox_vec)[0]
        rows = np.full(cols.shape, sample_index, dtype=int)
        data = prox_vec[cols]
        return data.tolist(), rows.tolist(), cols.tolist()

    def get_proximities(self):
        check_is_fitted(self)
        n_samples, _ = self.leaf_matrix.shape
        all_data, all_rows, all_cols = [], [], []
        for i in range(n_samples):
            data, rows, cols = self._prox_vector_one(i)
            all_data.extend(data)
            all_rows.extend(rows)
            all_cols.extend(cols)
        prox_sparse = sparse.csr_matrix(
            (np.array(all_data), (np.array(all_rows), np.array(all_cols))),
            shape=(n_samples, n_samples),
        )
        if self.force_symmetric:
            prox_sparse = (prox_sparse + prox_sparse.transpose()) * 0.5
        if self.matrix_type == "dense":
            return prox_sparse.toarray()
        return prox_sparse


class SurvPageRankPHATE(PHATE):
    """PHATE wrapper that optionally stores a diffusion-smoothed similarity matrix."""

    def __init__(self, beta: float = 1.0, smooth: bool = True, tdiff: float = 1.0, **phate_kwargs: Any):
        super().__init__(**phate_kwargs)
        self.beta = float(beta)
        self.smooth = bool(smooth)
        self.tdiff = float(tdiff)
        self.smt_proximity: Optional[np.ndarray] = None
        self.rf_model: Optional[RFGAPSurvival] = None

    def fit(self, S: Union[np.ndarray, sparse.spmatrix]) -> "SurvPageRankPHATE":
        arr = S.toarray() if sparse.issparse(S) else np.asarray(S)
        if self.smooth and self.tdiff > 0:
            self.smt_proximity = diffusion_similarity(arr, self.tdiff, self.beta)
        else:
            self.smt_proximity = arr
        return super().fit(S)
