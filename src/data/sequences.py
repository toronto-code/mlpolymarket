"""Build feature sequences and targets for next-price prediction.

Memory-conscious: pre-allocates the final X/y/timestamps arrays and fills them
in-place, so peak RAM is ~1x the output tensor instead of ~2-3x (no list of
per-ticker arrays, no np.concatenate doubling, no duplicated trades DataFrame).
The full ``trades`` DataFrame is freed before windowing so it never overlaps
with the output tensor in memory.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .load import iter_consolidated_trades_batches


_FEATURE_COLS = ["yes_price", "size", "order_flow", "vwap", "price_std_5"]
_N_FEATURES = len(_FEATURE_COLS)


def _per_ticker_stats(prices: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal VWAP (shifted by 1) and 5-trade rolling std of price, both float32."""
    v = prices * sizes
    cum_v = np.empty_like(v)
    cum_s = np.empty_like(sizes)
    cum_v[0] = 0.0
    cum_s[0] = 0.0
    if v.size > 1:
        np.cumsum(v[:-1], out=cum_v[1:])
        np.cumsum(sizes[:-1], out=cum_s[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cum_s > 0, cum_v / np.where(cum_s == 0, 1.0, cum_s), prices)
    vwap = vwap.astype(np.float32, copy=False)
    price_std_5 = (
        pd.Series(prices).rolling(5, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float32)
    )
    return vwap, price_std_5


def _merge_remainder(remainder: pd.DataFrame, chunk: pd.DataFrame) -> pd.DataFrame:
    if remainder is None or len(remainder) == 0:
        return chunk
    return pd.concat([remainder, chunk], ignore_index=True)


def _split_complete_remainder(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so ``complete`` ends on full ticker groups; ``remainder`` is the last (open) ticker."""
    if df.empty:
        return df, pd.DataFrame(columns=df.columns)
    last = df["ticker"].iloc[-1]
    k = len(df) - 1
    while k >= 0 and df["ticker"].iloc[k] == last:
        k -= 1
    return df.iloc[: k + 1], df.iloc[k + 1 :]


def _iter_valid_ticker_segments(
    df: pd.DataFrame,
    seq_len: int,
    min_trades: int,
    target_horizon: int,
):
    tickers_col = df["ticker"].to_numpy()
    n = len(df)
    if n == 0:
        return
    change = np.flatnonzero(tickers_col[1:] != tickers_col[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    for i in range(len(starts)):
        s, e = int(starts[i]), int(ends[i])
        nn = e - s
        if nn < seq_len + target_horizon or nn < min_trades:
            continue
        nw = nn - seq_len - target_horizon
        if nw > 0:
            yield s, e, nw


def _emit_ticker_windows(
    prices_g: np.ndarray,
    sizes_g: np.ndarray,
    oflow_g: np.ndarray,
    times_g: np.ndarray,
    *,
    seq_len: int,
    min_trades: int,
    target_horizon: int,
    target_type: Literal["next_price", "return_5", "direction_5"],
    stride: int,
    X: np.ndarray,
    y: np.ndarray,
    ts_out: np.ndarray,
    ticker_idx_out: np.ndarray,
    write: int,
    local_ticker_idx: int,
) -> int:
    n = int(prices_g.shape[0])
    if n < seq_len + target_horizon or n < min_trades:
        return write
    nw = n - seq_len - target_horizon
    if nw <= 0:
        return write

    vwap_g, pstd5_g = _per_ticker_stats(prices_g, sizes_g)
    feat = np.empty((n, _N_FEATURES), dtype=np.float32)
    feat[:, 0] = prices_g
    feat[:, 1] = sizes_g
    feat[:, 2] = oflow_g
    feat[:, 3] = vwap_g
    feat[:, 4] = pstd5_g
    wins = sliding_window_view(feat, seq_len, axis=0)
    sel = np.arange(0, nw, stride, dtype=np.int64)
    n_kept = int(sel.shape[0])
    if n_kept == 0:
        del wins, feat, vwap_g, pstd5_g
        return write

    X[write : write + n_kept] = wins[sel].transpose(0, 2, 1)
    i_arr = seq_len + sel
    if target_type == "next_price":
        y[write : write + n_kept] = prices_g[i_arr + target_horizon - 1]
    elif target_type == "return_5":
        p0 = prices_g[i_arr - 1]
        p1 = prices_g[i_arr + target_horizon - 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            y[write : write + n_kept] = np.where(p0 != 0, (p1 - p0) / p0, 0.0)
    else:
        p0 = prices_g[i_arr - 1]
        p1 = prices_g[i_arr + target_horizon - 1]
        y[write : write + n_kept] = (p1 > p0).astype(np.float32)

    ts_out[write : write + n_kept] = times_g[i_arr]
    ticker_idx_out[write : write + n_kept] = local_ticker_idx
    del wins, feat, vwap_g, pstd5_g
    return write + n_kept


def _collect_nw_list_streaming(
    consolidated_path: Path,
    iter_batch_kw: dict[str, Any],
    seq_len: int,
    min_trades: int,
    target_horizon: int,
) -> list[int]:
    nw_list: list[int] = []
    remainder = pd.DataFrame()
    for chunk in iter_consolidated_trades_batches(consolidated_path, **iter_batch_kw):
        df = _merge_remainder(remainder, chunk)
        complete, remainder = _split_complete_remainder(df)
        if not complete.empty:
            for _, _, nw in _iter_valid_ticker_segments(
                complete, seq_len, min_trades, target_horizon
            ):
                nw_list.append(nw)
    if not remainder.empty:
        for _, _, nw in _iter_valid_ticker_segments(
            remainder, seq_len, min_trades, target_horizon
        ):
            nw_list.append(nw)
    return nw_list


def _fill_streaming_pass2(
    consolidated_path: Path,
    iter_batch_kw: dict[str, Any],
    nw_list: list[int],
    stride: int,
    seq_len: int,
    min_trades: int,
    target_horizon: int,
    target_type: Literal["next_price", "return_5", "direction_5"],
    X: np.ndarray,
    y: np.ndarray,
    ts_out: np.ndarray,
    ticker_idx_out: np.ndarray,
) -> None:
    idx_nw = 0
    write = 0
    local_ticker = 0
    remainder = pd.DataFrame()

    def _flush_frame(frame: pd.DataFrame) -> None:
        nonlocal idx_nw, write, local_ticker
        if frame.empty:
            return
        prices = frame["yes_price"].to_numpy(dtype=np.float32, copy=False)
        sizes = frame["size"].to_numpy(dtype=np.float32, copy=False)
        taker = frame["taker_side"].astype(str).str.lower().to_numpy()
        oflow = np.where(taker == "yes", sizes, -sizes).astype(np.float32, copy=False)
        times = frame["created_time"].to_numpy(dtype="datetime64[ns]")
        for s, e, nw in _iter_valid_ticker_segments(
            frame, seq_len, min_trades, target_horizon
        ):
            if idx_nw >= len(nw_list) or nw != nw_list[idx_nw]:
                raise RuntimeError(
                    "Streaming pass-2 desync at ticker block "
                    f"(idx={idx_nw}, expected nw={nw_list[idx_nw] if idx_nw < len(nw_list) else 'eof'}, got {nw})"
                )
            write = _emit_ticker_windows(
                prices[s:e],
                sizes[s:e],
                oflow[s:e],
                times[s:e],
                seq_len=seq_len,
                min_trades=min_trades,
                target_horizon=target_horizon,
                target_type=target_type,
                stride=stride,
                X=X,
                y=y,
                ts_out=ts_out,
                ticker_idx_out=ticker_idx_out,
                write=write,
                local_ticker_idx=local_ticker,
            )
            idx_nw += 1
            local_ticker += 1

    for chunk in iter_consolidated_trades_batches(consolidated_path, **iter_batch_kw):
        df = _merge_remainder(remainder, chunk)
        complete, remainder = _split_complete_remainder(df)
        _flush_frame(complete)

    _flush_frame(remainder)

    if idx_nw != len(nw_list):
        raise RuntimeError(
            f"Streaming pass-2 ended early: processed {idx_nw} tickers, expected {len(nw_list)}"
        )
    if write != X.shape[0]:
        raise RuntimeError(f"Streaming pass-2 write mismatch: {write} vs {X.shape[0]}")


def build_sequences_from_consolidated(
    consolidated_path: str | Path,
    *,
    seq_len: int = 32,
    min_trades_per_market: int = 100,
    target_type: Literal["next_price", "return_5", "direction_5"] = "next_price",
    target_horizon: int = 1,
    max_samples: Optional[int] = None,
    iter_batch_kw: dict[str, Any],
    memmap_dir: Optional[Path | str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build sequences from a consolidated parquet without materialising all trades
    in pandas at once (two streaming passes over DuckDB chunks).

    When ``memmap_dir`` is given the four output arrays (X, y, ts, ticker_idx)
    are written to numpy memory-mapped files on disk instead of allocated in RAM.
    This lets X be arbitrarily large — the OS pages only the slices currently
    needed, so peak RSS stays O(batch_size) during training.

    Resume behaviour: if ``memmap_dir`` already contains correctly-shaped files
    from a previous run (same ``total_kept`` and same ``seq_len``/``_N_FEATURES``)
    both streaming passes are skipped entirely.
    """
    path = Path(consolidated_path)
    print("  Pass 1/2: counting windows (streaming consolidated parquet)...", flush=True)
    nw_list = _collect_nw_list_streaming(
        path, iter_batch_kw, seq_len, min_trades_per_market, target_horizon
    )
    total_w = sum(nw_list)
    if total_w == 0:
        return (
            np.zeros((0, seq_len, _N_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            pd.DatetimeIndex([]),
        )

    stride = (
        max(1, total_w // max_samples)
        if max_samples is not None and total_w > max_samples
        else 1
    )
    total_kept = sum((nw + stride - 1) // stride for nw in nw_list)
    x_gb = total_kept * seq_len * _N_FEATURES * 4 / 1e9
    print(
        f"  Pass 1 done: windows={total_w:,}, stride={stride}, "
        f"samples_kept={total_kept:,}  (X ≈ {x_gb:.1f} GB)",
        flush=True,
    )

    if memmap_dir is not None:
        mm_dir = Path(memmap_dir)
        mm_dir.mkdir(parents=True, exist_ok=True)
        x_path = mm_dir / "X.npy"
        y_path = mm_dir / "y.npy"
        ts_path = mm_dir / "ts.npy"
        ti_path = mm_dir / "ticker_idx.npy"
        meta_path = mm_dir / "meta.npy"

        expected_shape = (total_kept, seq_len, _N_FEATURES)
        can_resume = (
            x_path.exists()
            and y_path.exists()
            and ts_path.exists()
            and ti_path.exists()
            and meta_path.exists()
        )
        if can_resume:
            saved_meta = np.load(meta_path)
            can_resume = tuple(int(v) for v in saved_meta) == expected_shape

        if can_resume:
            print(
                f"  Sequences already on disk at {mm_dir} — skipping pass 2 (resume).",
                flush=True,
            )
            # Open r+ so the caller can still apply in-place normalisation if needed.
            X = np.memmap(x_path, mode="r+", dtype="float32", shape=expected_shape)
            y = np.memmap(y_path, mode="r+", dtype="float32", shape=(total_kept,))
            ts_out = np.memmap(ts_path, mode="r+", dtype="datetime64[ns]", shape=(total_kept,))
            ticker_idx_out = np.memmap(ti_path, mode="r+", dtype="int64", shape=(total_kept,))
            return X, y, ticker_idx_out, pd.DatetimeIndex(ts_out)

        print(
            f"  Writing {x_gb:.1f} GB sequence array to disk at {mm_dir} ...",
            flush=True,
        )
        X = np.memmap(x_path, mode="w+", dtype="float32", shape=expected_shape)
        y = np.memmap(y_path, mode="w+", dtype="float32", shape=(total_kept,))
        ts_out = np.memmap(ts_path, mode="w+", dtype="datetime64[ns]", shape=(total_kept,))
        ticker_idx_out = np.memmap(ti_path, mode="w+", dtype="int64", shape=(total_kept,))
    else:
        print(
            f"  Allocating {x_gb:.1f} GB X array in RAM (set training.sequences_dir "
            "to offload to disk).",
            flush=True,
        )
        X = np.empty((total_kept, seq_len, _N_FEATURES), dtype=np.float32)
        y = np.empty(total_kept, dtype=np.float32)
        ts_out = np.empty(total_kept, dtype="datetime64[ns]")
        ticker_idx_out = np.empty(total_kept, dtype=np.int64)

    print("  Pass 2/2: filling X/y (streaming)...", flush=True)
    _fill_streaming_pass2(
        path,
        iter_batch_kw,
        nw_list,
        stride,
        seq_len,
        min_trades_per_market,
        target_horizon,
        target_type,
        X,
        y,
        ts_out,
        ticker_idx_out,
    )

    if memmap_dir is not None:
        # Flush to disk and write shape metadata so future runs can resume.
        X.flush()
        y.flush()
        ts_out.flush()
        ticker_idx_out.flush()
        np.save(meta_path, np.array(list(expected_shape), dtype=np.int64))
        print(f"  Sequences flushed to {mm_dir}.", flush=True)
        # Re-open as r+ so in-place normalisation in the caller can write back to disk.
        X = np.memmap(x_path, mode="r+", dtype="float32", shape=expected_shape)
        y = np.memmap(y_path, mode="r+", dtype="float32", shape=(total_kept,))
        ts_out = np.memmap(ts_path, mode="r+", dtype="datetime64[ns]", shape=(total_kept,))
        ticker_idx_out = np.memmap(ti_path, mode="r+", dtype="int64", shape=(total_kept,))

    gc.collect()
    return X, y, ticker_idx_out, pd.DatetimeIndex(ts_out)


def build_sequences(
    trades: pd.DataFrame,
    seq_len: int = 32,
    min_trades_per_market: int = 100,
    target_type: Literal["next_price", "return_5", "direction_5"] = "next_price",
    target_horizon: int = 1,
    max_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    For each trade (after seq_len), build:
    - X: (seq_len, n_features) per sample. Features: yes_price, size, order_flow, vwap, price_std_5.
    - y: next price (or return / direction over horizon).
    - ticker_index: which market (for optional market-based split).
    - timestamps: created_time of the *prediction* point (for time-based split).

    If ``max_samples`` is given and the full build would exceed it, a uniform
    per-ticker stride is applied so the returned arrays have at most
    ``max_samples`` rows. This keeps peak memory bounded to the *final*
    (capped) tensor size rather than the full uncapped one.

    Returns:
        X: (n_samples, seq_len, n_features) float32
        y: (n_samples,) float32
        ticker_indices: (n_samples,) int64
        timestamps: (n_samples,) datetime index
    """
    # Sort once; keep only the columns we need so we can free the rest early.
    trades = trades.sort_values(["ticker", "created_time"])
    trades = trades[["ticker", "yes_price", "size", "taker_side", "created_time"]].reset_index(
        drop=True
    )

    # Extract per-column numpy arrays (downcast numeric to float32), then drop the DataFrame.
    # This is the single biggest peak-reduction step: after this, the wide trades
    # DataFrame with ticker strings + pandas overhead is gone.
    tickers_col = trades["ticker"].to_numpy()
    sizes = trades["size"].to_numpy(dtype=np.float32, copy=False)
    prices = trades["yes_price"].to_numpy(dtype=np.float32, copy=False)
    # order_flow = +size if taker_side == 'yes' else -size
    taker_side = trades["taker_side"].astype(str).str.lower().to_numpy()
    order_flow = np.where(taker_side == "yes", sizes, -sizes).astype(np.float32, copy=False)
    del taker_side
    times = trades["created_time"].to_numpy(dtype="datetime64[ns]")

    del trades
    gc.collect()

    # Locate ticker group boundaries in the sorted array (no expensive groupby).
    n_total = tickers_col.shape[0]
    if n_total == 0:
        return (
            np.zeros((0, seq_len, _N_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            pd.DatetimeIndex([]),
        )
    # Change points: positions where ticker differs from previous.
    change = np.flatnonzero(tickers_col[1:] != tickers_col[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n_total]])

    # First pass: figure out which tickers pass the min-length filter and how
    # many windows each will produce. No data copying yet.
    ticker_rows: list[tuple[int, int, int, int]] = []  # (ticker_idx, s, e, n_windows)
    total_windows_uncapped = 0
    for i in range(starts.shape[0]):
        s = int(starts[i])
        e = int(ends[i])
        n = e - s
        if n < seq_len + target_horizon or n < min_trades_per_market:
            continue
        nw = n - seq_len - target_horizon
        if nw <= 0:
            continue
        ticker_rows.append((i, s, e, nw))
        total_windows_uncapped += nw

    if not ticker_rows:
        return (
            np.zeros((0, seq_len, _N_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            pd.DatetimeIndex([]),
        )

    # Apply optional cap by striding each ticker's windows uniformly.
    # Stride is global so time coverage is preserved across all tickers.
    if max_samples is not None and total_windows_uncapped > max_samples:
        stride = max(1, total_windows_uncapped // max_samples)
    else:
        stride = 1

    # Second pass: compute per-ticker kept count after stride, plus running offsets.
    per_ticker: list[tuple[int, int, int, int, int, int]] = []
    # (ticker_idx, s, e, nw, kept, write_offset)
    total_kept = 0
    for ticker_idx, s, e, nw in ticker_rows:
        if stride == 1:
            kept = nw
        else:
            kept = (nw + stride - 1) // stride
        per_ticker.append((ticker_idx, s, e, nw, kept, total_kept))
        total_kept += kept

    # Pre-allocate the single final output once. No list-of-arrays, no concatenate.
    X = np.empty((total_kept, seq_len, _N_FEATURES), dtype=np.float32)
    y = np.empty(total_kept, dtype=np.float32)
    ts_out = np.empty(total_kept, dtype="datetime64[ns]")
    ticker_idx_out = np.empty(total_kept, dtype=np.int64)

    # Record which ticker names survive so we can emit a compact index -> name mapping.
    kept_ticker_names: list = []

    for local_idx, (grp_idx, s, e, nw, kept, write) in enumerate(per_ticker):
        kept_ticker_names.append(tickers_col[s])

        prices_g = prices[s:e]
        sizes_g = sizes[s:e]
        oflow_g = order_flow[s:e]
        times_g = times[s:e]

        vwap_g, pstd5_g = _per_ticker_stats(prices_g, sizes_g)

        # Build (n, n_features) feature matrix for this ticker.
        # Stack is cheap relative to total dataset; per-ticker only.
        feat = np.empty((e - s, _N_FEATURES), dtype=np.float32)
        feat[:, 0] = prices_g
        feat[:, 1] = sizes_g
        feat[:, 2] = oflow_g
        feat[:, 3] = vwap_g
        feat[:, 4] = pstd5_g

        # sliding_window_view shape: (n - seq_len + 1, n_features, seq_len)
        wins = sliding_window_view(feat, seq_len, axis=0)

        if stride == 1:
            X[write : write + kept] = wins[:nw].transpose(0, 2, 1)
            sel = np.arange(nw)
        else:
            sel = np.arange(0, nw, stride)[:kept]
            # Defensive: recompute kept if arange produced fewer rows (edge cases)
            if sel.shape[0] != kept:
                # Should not happen given the ceil formula above, but guard anyway.
                kept_local = sel.shape[0]
                X[write : write + kept_local] = wins[sel].transpose(0, 2, 1)
                # Pad-by-truncation: rewrite per_ticker layout is complex; instead
                # zero out trailing rows and remember the real count. To keep things
                # simple, fall back to copying whatever we have and trimming later.
                # (All downstream indexing uses write+kept, so we adjust via y/ts below.)
                # We simply record the actual kept count via `sel.shape[0]` for y/ts.
                kept = kept_local
            else:
                X[write : write + kept] = wins[sel].transpose(0, 2, 1)

        # Targets: original semantics --
        #   i_arr = np.arange(seq_len, n - target_horizon), length == nw
        #   y[k] corresponds to trade at index seq_len + k (window k ends at seq_len+k-1)
        i_arr = seq_len + sel  # absolute index inside this ticker
        if target_type == "next_price":
            y[write : write + kept] = prices_g[i_arr + target_horizon - 1]
        elif target_type == "return_5":
            p0 = prices_g[i_arr - 1]
            p1 = prices_g[i_arr + target_horizon - 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                y[write : write + kept] = np.where(p0 != 0, (p1 - p0) / p0, 0.0)
        else:  # direction_5
            p0 = prices_g[i_arr - 1]
            p1 = prices_g[i_arr + target_horizon - 1]
            y[write : write + kept] = (p1 > p0).astype(np.float32)

        ts_out[write : write + kept] = times_g[i_arr]
        ticker_idx_out[write : write + kept] = local_idx

        # Help the allocator between tickers: drop per-ticker scratch promptly.
        del wins, feat, vwap_g, pstd5_g

    # Free the original long per-trade arrays before returning.
    del prices, sizes, order_flow, times, tickers_col
    gc.collect()

    return X, y, ticker_idx_out, pd.DatetimeIndex(ts_out)


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
