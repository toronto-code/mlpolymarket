"""Baseline predictors: last price and VWAP (no ML)."""

from __future__ import annotations

import numpy as np

# Feature order: yes_price, size, order_flow, vwap, price_std_5
IDX_PRICE = 0
IDX_VWAP = 3


class LastPriceBaseline:
    """Predict next price = last observed price in the sequence."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Last time step, price feature
        return X[:, -1, IDX_PRICE].astype(np.float32)


class VWAPBaseline:
    """Predict next price = VWAP of the sequence (last time step's VWAP)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, -1, IDX_VWAP].astype(np.float32)
