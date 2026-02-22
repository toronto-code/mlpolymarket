"""Build feature sequences and targets for next-price prediction."""

from typing import Literal

import numpy as np
import pandas as pd


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

    X_list = []
    y_list = []
    ticker_list = []
    ts_list = []

    for ticker, g in trades.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < seq_len + target_horizon or len(g) < min_trades_per_market:
            continue
        arr = g[feature_cols].values.astype(np.float32)
        prices = g["yes_price"].values
        times = g["created_time"]

        for i in range(seq_len, len(g) - target_horizon):
            # Features: last seq_len trades
            X_list.append(arr[i - seq_len : i])
            # Target
            if target_type == "next_price":
                y_list.append(float(prices[i + target_horizon - 1]))
            elif target_type == "return_5":
                p0 = prices[i - 1]
                p1 = prices[i + target_horizon - 1]
                y_list.append((p1 - p0) / p0 if p0 else 0.0)
            else:  # direction_5
                p0 = prices[i - 1]
                p1 = prices[i + target_horizon - 1]
                y_list.append(1.0 if p1 > p0 else 0.0)
            ticker_list.append(ticker)
            ts_list.append(times.iloc[i])

    if not X_list:
        return (
            np.zeros((0, seq_len, n_features), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            pd.DatetimeIndex([]),
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    timestamps = pd.DatetimeIndex(ts_list)
    tickers = np.array(ticker_list)
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
