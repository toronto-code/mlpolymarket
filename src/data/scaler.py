"""Feature scaling for sequence data. Fit on train; apply to val/test."""

from __future__ import annotations

import numpy as np


class SequenceScaler:
    """
    Per-feature mean/std over the training set. Applied to each (sample, time, feature).
    Fit on X_train (n, seq_len, n_features); transform preserves shape.
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceScaler":
        self._mean = np.nanmean(X, axis=(0, 1)).astype(np.float32)
        self._std = np.nanstd(X, axis=(0, 1)).astype(np.float32)
        self._std[self._std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Normalise X.

        If ``inplace=True`` and X is float32, the subtraction and division are
        done in-place. This halves peak memory on large tensors where X can be
        many GB (the default ``(X - mean) / std`` would otherwise allocate two
        full-size temporaries).
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Scaler not fitted")
        if inplace and X.dtype == np.float32:
            # Broadcasting over the last axis works with in-place ops.
            np.subtract(X, self._mean, out=X)
            np.divide(X, self._std, out=X)
            return X
        return ((X - self._mean) / self._std).astype(np.float32)

    def fit_transform(self, X: np.ndarray, inplace: bool = False) -> np.ndarray:
        return self.fit(X).transform(X, inplace=inplace)
