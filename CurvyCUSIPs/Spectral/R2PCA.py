# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4400158

"""
- The original paper mentions the ability to handle datasets with variable feature dimensions over time by comparing only shared features. 
  This implementation assumes fixed dimensions for simplicity and does not address this explicitly.

- There was an emphasis on the importance of careful initialization of principal components in the first window which makes sense for dataset with 
  potential regime shifts. This implementation takes the easy road and initializes based on the first available window. A todo item is adaptive initialization 
  based on user-defined periods or regime detection.

- Low hanging fruit: batch processing or sparse matrix stuff for covar matrix and eigendecomp idk the math really, test cases lol
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return np.dot(u, v)


class R2PCA(BaseEstimator, TransformerMixin):
    """
    Robust Rolling PCA (R2-PCA).

    Parameters
    ----------
    n_components : int, optional (default=2)
        Number of principal components to compute.

    window_size : int, optional (default=12)
        Rolling window size for PCA (in 'time' dimension).

    copy : bool, optional (default=True)
        If False, data passed to fit are overwritten (not recommended here).

    svd_solver : str, optional (default='auto')
        SVD solver to use. Supported values are
        {'auto', 'full', 'arpack', 'randomized', 'covariance_eigh'}.

    Attributes
    ----------
    components_ : ndarray of shape (T, n_components, D)
        Principal components (eigenvectors) for each time step.

    explained_variance_ : ndarray of shape (T, n_components)
        The variance explained by each of the selected components at each time step.

    mean_ : ndarray of shape (D,)
        Per-feature empirical mean, estimated from the training set.
        (Currently stored but not actively used in transform/inverse_transform
         in this snippet.)

    n_features_in_ : int
        Number of features seen during fit.

    Notes
    -----
    Input data is assumed to be 3D: (F, T, D)
      - F: number of funds (assets)
      - T: number of time periods
      - D: number of features
    """

    def __init__(
        self,
        n_components=2,
        window_size=12,
        copy=True,
        svd_solver="auto",
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.copy = copy
        self.svd_solver = svd_solver

    def fit(self, X, y=None):
        """
        Fit the R2PCA model on X.

        Parameters
        ----------
        X : ndarray of shape (F, T, D)
            Input data with F assets, T time steps, and D features.

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Input data X must be 3D with shape (F, T, D). Got shape {X.shape}.")
        F, T, D = X.shape
        self.n_features_in_ = D

        # If you want to do global centering, you can store mean here:
        self.mean_ = np.mean(X, axis=(0, 1))  # (D,)

        # Allocate storage
        self.components_ = np.zeros((T, self.n_components, D), dtype=float)
        self.explained_variance_ = np.zeros((T, self.n_components), dtype=float)

        prev_pcs = None

        # Rolling window loop
        for t in range(T):
            start = max(0, t - self.window_size + 1)
            X_window = X[:, start : t + 1, :]  # shape (F, window_length, D)

            # 1) Compute average covariance matrix across F assets
            cov_sum = np.zeros((D, D), dtype=float)
            valid_funds = 0
            for f_idx in range(F):
                data_fund = X_window[f_idx]  # shape (window_length, D)

                if data_fund.shape[0] > 1:
                    cov_fund = np.cov(data_fund, rowvar=False)
                    cov_sum += cov_fund
                    valid_funds += 1

            if valid_funds > 0:
                avg_cov = cov_sum / valid_funds
            else:
                avg_cov = np.zeros((D, D), dtype=float)

            # 2) Eigen-decomposition of avg_cov
            eigvals, eigvecs = np.linalg.eigh(avg_cov)
            idx_sorted = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx_sorted]
            eigvecs = eigvecs[:, idx_sorted]

            # Keep top n_components
            eigvals = eigvals[: self.n_components]
            eigvecs = eigvecs[:, : self.n_components]  # shape (D, n_components)
            eigvecs = eigvecs.T  # shape (n_components, D)

            # Normalize eigenvectors
            for i in range(self.n_components):
                norm = np.linalg.norm(eigvecs[i])
                if norm > 0:
                    eigvecs[i] /= norm

            # 3) Sign flip & reordering
            if prev_pcs is not None:
                new_order = []
                used_prev_idx = set()
                tmp_eigvecs = np.copy(eigvecs)

                for i in range(self.n_components):
                    sims = [abs(_cosine_similarity(tmp_eigvecs[i], prev_pcs[j])) for j in range(self.n_components)]
                    best_j = np.argmax(sims)
                    # If there's a tie in usage, pick next best.
                    if best_j in used_prev_idx:
                        sorted_sims_idx = np.argsort(sims)[::-1]
                        for k in sorted_sims_idx:
                            if k not in used_prev_idx:
                                best_j = k
                                break
                    used_prev_idx.add(best_j)

                    # Flip sign if dot < 0
                    dot_val = np.dot(tmp_eigvecs[i], prev_pcs[best_j])
                    if dot_val < 0:
                        tmp_eigvecs[i] = -tmp_eigvecs[i]

                    new_order.append((i, best_j))

                # Reorder so that new_eigvecs[j] matches whichever i gave best similarity
                new_eigvecs = np.zeros_like(tmp_eigvecs)
                for curr_i, matched_j in new_order:
                    new_eigvecs[matched_j] = tmp_eigvecs[curr_i]

                eigvecs = new_eigvecs

            self.components_[t] = eigvecs  # (n_components, D)
            self.explained_variance_[t] = eigvals

            prev_pcs = eigvecs

        return self

    def transform(self, X, y=None):
        """
        Project X onto rolling principal components.

        Parameters
        ----------
        X : ndarray of shape (F, T, D)

        Returns
        -------
        X_transformed : ndarray of shape (F, T, n_components)
        """
        X = np.asarray(X)
        F, T, D = X.shape

        if T != self.components_.shape[0]:
            raise ValueError("Number of time steps in X does not match fitted components_. " f"Got T={T}, expected {self.components_.shape[0]}.")
        if D != self.n_features_in_:
            raise ValueError(f"Number of features in X (D={D}) does not match fit (D={self.n_features_in_}).")

        X_transformed = np.zeros((F, T, self.n_components), dtype=float)

        for t in range(T):
            pcs_t = self.components_[t]  # (n_components, D)
            # (F, D) dot (D, n_components) => (F, n_components)
            X_transformed[:, t, :] = X[:, t, :].dot(pcs_t.T)

        return X_transformed

    def inverse_transform(self, X_transformed):
        """
        Map data from the rolling principal components space back to the original feature space.

        Parameters
        ----------
        X_transformed : ndarray of shape (F, T, n_components)
            Rolling PCA scores.

        Returns
        -------
        X_reconstructed : ndarray of shape (F, T, D)
            Reconstructed data in original feature space.
        """
        X_transformed = np.asarray(X_transformed)
        F, T, C = X_transformed.shape

        if T != self.components_.shape[0]:
            raise ValueError(
                "Number of time steps in X_transformed does not match fitted components_. " f"Got T={T}, expected {self.components_.shape[0]}."
            )
        if C != self.n_components:
            raise ValueError(f"Number of components in X_transformed (C={C}) " f"does not match n_components={self.n_components}.")

        # (F, T, D) reconstruction
        X_reconstructed = np.zeros((F, T, self.n_features_in_), dtype=float)

        # For each time step, multiply (F, n_components) by (n_components, D)
        for t in range(T):
            pcs_t = self.components_[t]  # shape (n_components, D)
            # shape (F, n_components) dot shape (n_components, D) => (F, D)
            X_reconstructed[:, t, :] = X_transformed[:, t, :].dot(pcs_t)

        # If you had subtracted a global or time-specific mean in transform,
        # you would add it back here. For example:
        #
        # X_reconstructed[:, t, :] += self.mean_
        #
        # if the data was originally mean-centered.
        #
        # But currently, we did NOT subtract mean in transform(),
        # so we do not add it here.

        return X_reconstructed

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (F, T, D)

        Returns
        -------
        X_new : ndarray of shape (F, T, n_components)
        """
        return self.fit(X, y=y).transform(X)

    def get_covariance(self):
        """
        Not directly meaningful in rolling PCA context, but kept
        for compatibility with scikit-learn's PCA.
        """
        raise NotImplementedError("Covariance over time-varying components is not defined for rolling PCA.")
