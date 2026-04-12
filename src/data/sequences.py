"""Build feature sequences and targets for next-price prediction."""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _order_flow(df: pd.DataFrame) -> pd.Series:
    """Net yes-side volume: positive = more yes buying."""
    net = np.where(df["taker_side"].str.lower() == "yes", df["size"], -df["size"])
    return pd.Series(net, index=df.index)


def build_sequences(
    trades: pd.DataFrame,
    seq_len: int = 32,
    min_trades_per_market: int = 100,
    target_type: Literal["next_price", "return_5", "direction_5"] = "next_price",
    target_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    For each trade (after seq_len), build:
    - X: (seq_len, n_features) per sample. Features: yes_price, size, order_flow, vwap, price_std_5 (rolling 5-trade volatility).
    - y: next price (or return / direction over horizon).
    - ticker_index: which market (for optional market-based split).
    - timestamps: created_time of the *prediction* point (for time-based split).

    Returns:
        X: (n_samples, seq_len, n_features) float32
        y: (n_samples,) float32
        ticker_indices: (n_samples,) int
        timestamps: (n_samples,) datetime index
    """
    trades = trades.sort_values(["ticker", "created_time"]).reset_index(drop=True)
    trades["order_flow"] = _order_flow(trades)
    # VWAP and other rolling stats per market
    trades["vwap"] = np.nan
    trades["price_std_5"] = np.nan
    for _, g in trades.groupby("ticker", sort=False):
        idx = g.index
        v = g["yes_price"] * g["size"]
        cum_v = v.cumsum().shift(1)
        cum_s = g["size"].cumsum().shift(1)
        trades.loc[idx, "vwap"] = (cum_v / cum_s.replace(0, np.nan)).fillna(g["yes_price"]).values
        # Rolling std of price over last 5 trades (min_periods=1 avoids NaNs at start)
        trades.loc[idx, "price_std_5"] = g["yes_price"].rolling(5, min_periods=1).std().fillna(0).values

    feature_cols = ["yes_price", "size", "order_flow", "vwap", "price_std_5"]
    n_features = len(feature_cols)

    # Collect one numpy array per ticker rather than one tiny view per window.
    # This avoids building a Python list with millions of entries (each a numpy
    # view object carrying ~200 B of Python overhead), which is often the largest
    # RAM spike before np.stack is even called.
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ticker_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []

    for ticker, g in trades.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < seq_len + target_horizon or n < min_trades_per_market:
            continue

        # n_windows: number of valid (window, target) pairs.
        # Original loop was range(seq_len, n - target_horizon), so the count is:
        n_windows = n - seq_len - target_horizon
        if n_windows <= 0:
            continue

        arr = g[feature_cols].values.astype(np.float32)  # (n, n_features)
        prices = g["yes_price"].values
        times = g["created_time"]

        # sliding_window_view shape: (n - seq_len + 1, n_features, seq_len)
        # wins[k, f, t] == arr[k + t, f]  →  transpose to (n_windows, seq_len, n_features)
        wins = sliding_window_view(arr, seq_len, axis=0)
        # np.ascontiguousarray forces a real copy so arr can be freed immediately.
        X_ticker = np.ascontiguousarray(wins[:n_windows].transpose(0, 2, 1))
        del wins, arr  # release the view + the source array

        # Vectorised targets — i in range(seq_len, n - target_horizon)
        i_arr = np.arange(seq_len, n - target_horizon)  # length == n_windows
        if target_type == "next_price":
            y_ticker = prices[i_arr + target_horizon - 1].astype(np.float32)
        elif target_type == "return_5":
            p0 = prices[i_arr - 1]
            p1 = prices[i_arr + target_horizon - 1]
            y_ticker = np.where(p0 != 0, (p1 - p0) / p0, 0.0).astype(np.float32)
        else:  # direction_5
            p0 = prices[i_arr - 1]
            p1 = prices[i_arr + target_horizon - 1]
            y_ticker = (p1 > p0).astype(np.float32)

        X_parts.append(X_ticker)
        y_parts.append(y_ticker)
        ticker_parts.append(np.full(n_windows, ticker))
        ts_parts.append(times.iloc[i_arr].values)

    if not X_parts:
        return (
            np.zeros((0, seq_len, n_features), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            pd.DatetimeIndex([]),
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    timestamps = pd.DatetimeIndex(np.concatenate(ts_parts))
    tickers = np.concatenate(ticker_parts)
    uniq, ticker_indices = np.unique(tickers, return_inverse=True)
    ticker_indices = ticker_indices.astype(np.int64)

    return X, y, ticker_indices, timestamps


def time_based_split(
    timestamps: pd.DatetimeIndex,
    test_start_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Train and test indices; test is the last fraction of time."""
    n = len(timestamps)
    ts = pd.Series(range(n), index=timestamps).sort_index()
    cut = int(n * (1 - test_start_frac))
    train_idx = np.arange(0, cut)
    test_idx = np.arange(cut, n)
    return train_idx, test_idx


def time_based_split_three_way(
    timestamps: pd.DatetimeIndex,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train / validation / test by time order. Fractions must sum to 1.
    Validation used for model selection and early stopping; test used once for reporting.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    n = len(timestamps)
    ts = pd.Series(range(n), index=timestamps).sort_index()
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train_idx = np.arange(0, t1)
    val_idx = np.arange(t1, t2)
    test_idx = np.arange(t2, n)
    return train_idx, val_idx, test_idx
