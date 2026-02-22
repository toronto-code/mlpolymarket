#!/usr/bin/env python3
"""Generate dummy Polymarket-style Parquet trades for testing run.py without the full dataset."""

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent.parent
    out_trades = root / "data" / "polymarket" / "trades"
    out_blocks = root / "data" / "polymarket" / "blocks"
    out_trades.mkdir(parents=True, exist_ok=True)
    out_blocks.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_markets = 20
    trades_per_market = 200
    base_block = 40_000_000
    rows = []
    for m in range(n_markets):
        # ticker = asset_id (outcome token id) as string
        ticker = str(1000 + m)
        price = 0.5
        for t in range(trades_per_market):
            price = np.clip(price + np.random.randn() * 0.05, 0.05, 0.95)
            size_tokens = max(1, int(np.random.lognormal(2, 1)) * 1_000_000)  # 6 decimals
            usdc_amount = int(price * size_tokens)
            is_buy = np.random.rand() > 0.5
            block = base_block + m * 1000 + t
            rows.append({
                "block_number": block,
                "transaction_hash": f"0x{t:064x}"[:66],
                "log_index": t,
                "order_hash": f"0x{m:032x}{t:032x}",
                "maker": "0x" + "a" * 40,
                "taker": "0x" + "b" * 40,
                "maker_asset_id": 0 if is_buy else int(ticker),
                "taker_asset_id": int(ticker) if is_buy else 0,
                "maker_amount": usdc_amount if is_buy else size_tokens,
                "taker_amount": size_tokens if is_buy else usdc_amount,
                "fee": 0,
            })
    trades_df = pd.DataFrame(rows)

    # Blocks: block_number -> timestamp
    blocks = trades_df[["block_number"]].drop_duplicates().sort_values("block_number")
    blocks["timestamp"] = pd.to_datetime("2023-06-01", utc=True) + pd.to_timedelta(blocks["block_number"] - base_block, unit="s")
    blocks.to_parquet(out_blocks / "blocks.parquet", index=False)

    trades_df.to_parquet(out_trades / "dummy_trades.parquet", index=False)
    print(f"Wrote {out_trades / 'dummy_trades.parquet'} ({len(trades_df)} rows, {n_markets} markets)")
    print(f"Wrote {out_blocks / 'blocks.parquet'} ({len(blocks)} blocks)")


if __name__ == "__main__":
    main()
