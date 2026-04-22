#!/usr/bin/env python3
"""
Next-price prediction on Polymarket (past year). Train/val/test split by time.
Validation used for early stopping and model selection; test used once for final metrics.
Saves best model weights and metrics to output/.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

# Ensure project root is on path when running as script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import yaml

from src.data.load import load_polymarket_trades, load_sports_ticker_ids_file
from src.data.scaler import SequenceScaler
from src.data.sequences import (
    build_sequences,
    build_sequences_from_consolidated,
    time_based_split_three_way,
)
from src.eval.metrics import compute_metrics, print_metrics
from src.models.baselines import LastPriceBaseline, VWAPBaseline
from src.models.mlp import MLPModel
from src.models.lstm import LSTMModel


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train next-price models on Polymarket (past year). Saves to output/."
    )
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root dir containing data/polymarket/trades and .../blocks",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--max-markets", type=int, default=None, help="Cap markets for debugging")
    parser.add_argument("--no-mlp", action="store_true", help="Skip MLP")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM")
    parser.add_argument("--no-save", action="store_true", help="Do not write models or metrics to disk")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    data_root = Path(args.data_dir or root)
    out_dir = Path(args.output_dir or cfg.get("output", {}).get("dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    trades_path = data_root / data_cfg.get("polymarket_trades", "data/polymarket/trades")
    blocks_path = data_root / data_cfg.get("polymarket_blocks", "data/polymarket/blocks")
    last_n_months = data_cfg.get("last_n_months", 12)
    exclude_sports = data_cfg.get("exclude_sports", False)
    sports_tickers_file = data_cfg.get("polymarket_sports_tickers_file")
    max_rows = data_cfg.get("max_rows")
    batch_by_month = data_cfg.get("batch_by_month", False)
    consolidated_key = data_cfg.get("polymarket_consolidated")
    consolidated_path = (data_root / consolidated_key) if consolidated_key else None

    exclude_tickers: set[str] = set()
    if exclude_sports and sports_tickers_file:
        sports_path = data_root / sports_tickers_file
        exclude_tickers = load_sports_ticker_ids_file(sports_path)
        if exclude_tickers:
            print("Excluding {} sports/blocked tickers (from {}).".format(len(exclude_tickers), sports_path))
        else:
            print("exclude_sports is true but no tickers in {} (run scripts/generate_sports_tickers.py?).".format(sports_path))

    seq_cfg = cfg.get("sequence", {})
    seq_len = seq_cfg.get("length", 32)
    min_trades = seq_cfg.get("min_trades_per_market", 100)
    target_cfg = cfg.get("target", {})
    target_type = target_cfg.get("type", "next_price")
    target_horizon = target_cfg.get("horizon", 1)
    max_samples = data_cfg.get("max_samples")

    # Optional disk-backed sequence store: X/y written as numpy memmap so they
    # never reside in RAM as a monolithic allocation.  Peak RSS during training
    # stays O(batch_size) regardless of dataset size.
    train_cfg_early = cfg.get("training", {})
    sequences_dir_rel = train_cfg_early.get("sequences_dir")
    memmap_dir: Path | None = (out_dir / sequences_dir_rel) if sequences_dir_rel else None

    consolidated_exists = consolidated_path is not None and consolidated_path.exists()
    trades = None

    print("Loading Polymarket trades (last {} months)...".format(last_n_months))
    try:
        if consolidated_exists:
            print(
                "  Consolidated parquet found — using streaming loader (no full-RAM pandas materialisation).",
                flush=True,
            )
        elif consolidated_path is not None:
            trades = load_polymarket_trades(
                trades_path,
                consolidated_path=consolidated_path,
                last_n_months=last_n_months,
                exclude_tickers=exclude_tickers if exclude_tickers else None,
                max_rows=max_rows,
            )
        elif blocks_path.exists():
            trades = load_polymarket_trades(
                trades_path,
                blocks_dir=blocks_path,
                last_n_months=last_n_months,
                exclude_tickers=exclude_tickers if exclude_tickers else None,
                max_rows=max_rows,
                batch_by_month=batch_by_month,
            )
        else:
            trades = load_polymarket_trades(
                trades_path,
                last_n_months=last_n_months,
                exclude_tickers=exclude_tickers if exclude_tickers else None,
                max_rows=max_rows,
                batch_by_month=batch_by_month,
            )
    except FileNotFoundError as e:
        print(e)
        print("Point --data-dir to the repo that contains data/polymarket/trades and blocks.")
        return

    if not consolidated_exists:
        if trades is None or trades.empty:
            print("No trades in the requested window. Check path and last_n_months.")
            return
        if args.max_markets:
            tickers = trades["ticker"].unique()[: args.max_markets]
            trades = trades[trades["ticker"].isin(tickers)]
    elif args.max_markets:
        print("Note: --max-markets is ignored when training from an on-disk consolidated parquet.")

    print("Building sequences...")
    if consolidated_exists:
        iter_kw = {
            "min_price": 0.01,
            "max_price": 0.99,
            "start_date": None,
            "end_date": None,
            "last_n_months": last_n_months,
            "exclude_tickers": exclude_tickers if exclude_tickers else None,
            "max_rows": max_rows,
            "chunk_rows": int(data_cfg.get("consolidated_chunk_rows", 350_000)),
            "duckdb_memory_limit": str(data_cfg.get("duckdb_memory_limit", "4GB")),
        }
        X, y, _, timestamps = build_sequences_from_consolidated(
            consolidated_path,
            seq_len=seq_len,
            min_trades_per_market=min_trades,
            target_type=target_type,
            target_horizon=target_horizon,
            max_samples=max_samples,
            iter_batch_kw=iter_kw,
            memmap_dir=memmap_dir,
        )
    else:
        X, y, _, timestamps = build_sequences(
            trades,
            seq_len=seq_len,
            min_trades_per_market=min_trades,
            target_type=target_type,
            target_horizon=target_horizon,
            max_samples=max_samples,
        )
        del trades
    gc.collect()
    n_samples, _, n_features = X.shape
    if n_samples == 0:
        print("No sequence samples produced. Check data window, min_trades, and filters.")
        return
    x_gb = n_samples * seq_len * n_features * 4 / 1e9
    print(
        "  samples={:,}, seq_len={}, n_features={}  (X ~= {:.2f} GB)".format(
            n_samples, seq_len, n_features, x_gb
        )
    )
    if max_samples:
        print("  max_samples cap: {:,}".format(max_samples))

    split_cfg = cfg.get("split", {})
    train_frac = split_cfg.get("train_frac", 0.6)
    val_frac = split_cfg.get("val_frac", 0.2)
    test_frac = split_cfg.get("test_frac", 0.2)
    train_idx, val_idx, test_idx = time_based_split_three_way(
        timestamps, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )
    print("  train={}, val={}, test={}".format(len(train_idx), len(val_idx), len(test_idx)))

    # Contiguous splits: the time-based split returns np.arange ranges, so we
    # can use plain slicing to get *views* instead of fancy-indexing copies.
    # This saves one full copy of X (several GB on the 12-month run).
    t1 = int(len(train_idx))
    t2 = t1 + int(len(val_idx))
    X_train = X[:t1]
    X_val = X[t1:t2]
    X_test = X[t2:]
    y_train = y[:t1]
    y_val = y[t1:t2]
    y_test = y[t2:]

    normalize = cfg.get("normalize_features", True)
    if normalize:
        scaler = SequenceScaler()
        if memmap_dir is not None:
            # Avoid double-normalising across runs: write a flag file after the
            # first normalisation so resuming runs skip it and just reload stats.
            norm_flag = memmap_dir / "normalized.flag"
            scaler_cache = memmap_dir / "scaler.npz"
            if norm_flag.exists() and scaler_cache.exists():
                data = np.load(scaler_cache)
                scaler._mean = data["mean"]
                scaler._std = data["std"]
                print("  features normalised (loaded cached scaler — memmap already scaled)")
            else:
                # In-place: writes directly into the memmap files on disk.
                # Reads all of X_train once to compute mean/std (O(1) peak RAM).
                scaler.fit(X_train)
                scaler.transform(X_train, inplace=True)
                scaler.transform(X_val, inplace=True)
                scaler.transform(X_test, inplace=True)
                np.savez(scaler_cache, mean=scaler._mean, std=scaler._std)
                norm_flag.touch()
                print("  features normalised in-place on disk (scaler cached)")
        else:
            scaler.fit(X_train)
            scaler.transform(X_train, inplace=True)
            scaler.transform(X_val, inplace=True)
            scaler.transform(X_test, inplace=True)
            print("  features normalized (mean/std on train, in-place)")

    last_price_test = X_test[:, -1, 0]

    train_cfg = cfg.get("training", {})
    device = train_cfg.get("device", "cpu")
    batch_size = train_cfg.get("batch_size", 256)
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("lr", 1e-3)
    patience = train_cfg.get("early_stopping_patience", 7)
    model_seed = train_cfg.get("seed", seed)

    mlp_cfg = cfg.get("mlp", {})
    lstm_cfg = cfg.get("lstm", {})
    models_cfg = cfg.get("models", {})

    results = {}
    val_results = {}
    saved_artifacts = {}
    last_price_val = X_val[:, -1, 0]

    if models_cfg.get("baseline_last_price", True):
        m = LastPriceBaseline()
        m.fit(X_train, y_train)
        pred_test = m.predict(X_test)
        pred_val = m.predict(X_val)
        results["last_price"] = compute_metrics(y_test, pred_test, last_price=last_price_test)
        val_results["last_price"] = compute_metrics(y_val, pred_val, last_price=last_price_val)

    if models_cfg.get("baseline_vwap", True):
        m = VWAPBaseline()
        m.fit(X_train, y_train)
        pred_test = m.predict(X_test)
        pred_val = m.predict(X_val)
        results["vwap"] = compute_metrics(y_test, pred_test, last_price=last_price_test)
        val_results["vwap"] = compute_metrics(y_val, pred_val, last_price=last_price_val)

    if models_cfg.get("mlp", True) and not args.no_mlp:
        print("Training MLP...")
        m = MLPModel(
            seq_len=seq_len,
            n_features=n_features,
            hidden=tuple(mlp_cfg.get("hidden", [128, 64, 32])),
            dropout=mlp_cfg.get("dropout", 0.2),
            lr=lr,
            device=device,
            max_epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            seed=model_seed,
        )
        m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        pred_test = m.predict(X_test)
        pred_val = m.predict(X_val)
        results["mlp"] = compute_metrics(y_test, pred_test, last_price=last_price_test)
        val_results["mlp"] = compute_metrics(y_val, pred_val, last_price=last_price_val)
        if not args.no_save and cfg.get("output", {}).get("save_models", True):
            state = m.get_state_dict()
            torch.save(state, out_dir / "mlp_best.pt")
            saved_artifacts["mlp"] = str(out_dir / "mlp_best.pt")
        # Release the fitted MLP (and its internal torch tensors) before the
        # LSTM allocates its own — avoids holding two models plus the data.
        del m
        gc.collect()

    if models_cfg.get("lstm", True) and not args.no_lstm:
        print("Training LSTM...")
        m = LSTMModel(
            seq_len=seq_len,
            n_features=n_features,
            hidden=lstm_cfg.get("hidden", 128),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.2),
            lr=lr,
            device=device,
            max_epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            seed=model_seed,
        )
        m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        pred_test = m.predict(X_test)
        pred_val = m.predict(X_val)
        results["lstm"] = compute_metrics(y_test, pred_test, last_price=last_price_test)
        val_results["lstm"] = compute_metrics(y_val, pred_val, last_price=last_price_val)
        if not args.no_save and cfg.get("output", {}).get("save_models", True):
            state = m.get_state_dict()
            torch.save(state, out_dir / "lstm_best.pt")
            saved_artifacts["lstm"] = str(out_dir / "lstm_best.pt")
        del m
        gc.collect()

    print("\n--- Test set metrics ---\n")
    for name, metrics in results.items():
        print("{}:".format(name))
        print_metrics(metrics)
        print()

    if "last_price" in results and len(results) > 1:
        base_mae = results["last_price"]["mae"]
        base_dir = results["last_price"]["direction_accuracy"]
        print("Vs last-price baseline:")
        for name, metrics in results.items():
            if name == "last_price":
                continue
            mae_imp = (base_mae - metrics["mae"]) / base_mae * 100 if base_mae else 0.0
            dir_imp = (metrics["direction_accuracy"] - base_dir) * 100
            print("  {}: MAE {:.1f}% vs baseline, direction acc {:.1f} pp".format(name, mae_imp, dir_imp))

    if cfg.get("output", {}).get("save_metrics", True) and not args.no_save:
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "config": {k: v for k, v in cfg.items() if k != "output"},
                    "val_metrics": val_results,
                    "test_metrics": results,
                    "saved_models": saved_artifacts,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                },
                f,
                indent=2,
            )
        print("\nWrote {} and model checkpoints to {}.".format(metrics_path, out_dir))


if __name__ == "__main__":
    main()
