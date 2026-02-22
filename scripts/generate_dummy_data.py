#!/usr/bin/env python3
"""Generate dummy Kalshi-style Parquet trades for testing run.py without the full dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / "kalshi" / "trades"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_markets = 20
    trades_per_market = 200
    rows = []
    for m in range(n_markets):
        ticker = f"DUMMY-{m}"
        price = 50.0
        for t in range(trades_per_market):
            price = np.clip(price + np.random.randn() * 3, 5, 95)
            size = int(np.random.lognormal(2, 1)) + 1
            side = "yes" if np.random.rand() > 0.5 else "no"
            rows.append({
                "trade_id": f"{ticker}-{t}",
                "ticker": ticker,
                "size": size,
                "yes_price": int(round(price)),
                "no_price": 100 - int(round(price)),
                "taker_side": side,
                "created_time": pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(minutes=m * 1000 + t),
            })
    df = pd.DataFrame(rows)
    path = out_dir / "dummy_trades.parquet"
    df.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(df)} rows, {n_markets} markets)")


if __name__ == "__main__":
    main()
