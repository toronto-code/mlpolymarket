"""Evaluation metrics for next-price prediction."""

from __future__ import annotations

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    is_direction: bool = False,
    last_price: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute MAE, RMSE; if regression (next_price), also direction accuracy and simple profit.
    If is_direction (target is 0/1 direction), report accuracy and optional Brier.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = {}

    if is_direction:
        out["accuracy"] = float(np.mean((y_pred >= 0.5) == (y_true >= 0.5)))
        out["brier"] = float(np.mean((y_pred - y_true) ** 2))
        return out

    # Regression (next price)
    out["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    out["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Direction: (pred > last) == (true > last). Use last_price if provided, else median as proxy.
    if last_price is not None:
        last_price = np.asarray(last_price, dtype=np.float64)
    else:
        last_price = np.median(y_true)
    pred_up = y_pred > last_price
    true_up = y_true > last_price
    out["direction_accuracy"] = float(np.mean(pred_up == true_up))

    # Simple profit: assume we "buy" at predicted price and "sell" at actual; PnL = (y_true - y_pred) per contract.
    # So profit per contract = y_true - y_pred; sum = total (not scaled).
    out["profit_simple_per_contract"] = float(np.mean(y_true - y_pred))
    return out


def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    for k, v in metrics.items():
        print(f"  {prefix}{k}: {v:.4f}")
