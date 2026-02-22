"""Load Kalshi and Polymarket trade data from Parquet."""

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd


def load_kalshi_trades(
    data_dir: str | Path,
    *,
    min_yes_price: int = 1,
    max_yes_price: int = 99,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load Kalshi trades from Parquet directory. One row per trade."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Kalshi trades directory not found: {path}")
    pattern = str(path / "*.parquet")
    if not list(path.glob("*.parquet")):
        pattern = str(path / "**" / "*.parquet")

    con = duckdb.connect()
    ticker_filter = ""
    if tickers is not None:
        ticker_list = ", ".join(f"'{t}'" for t in tickers)
        ticker_filter = f" AND ticker IN ({ticker_list})"

    query = f"""
    SELECT
        trade_id,
        ticker,
        count AS size,
        yes_price,
        no_price,
        taker_side,
        created_time
    FROM read_parquet('{pattern}', hive_partitioning=0)
    WHERE yes_price BETWEEN {min_yes_price} AND {max_yes_price}
    {ticker_filter}
    ORDER BY ticker, created_time
    """
    df = con.execute(query).df()
    con.close()

    if "created_time" in df.columns and df["created_time"].dtype == object:
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
    return df


def load_kalshi_markets(
    data_dir: str | Path,
    *,
    status: Optional[str] = None,
    resolved_only: bool = False,
) -> pd.DataFrame:
    """Load Kalshi market metadata."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Kalshi markets directory not found: {path}")
    pattern = str(path / "*.parquet")
    if not list(path.glob("*.parquet")):
        pattern = str(path / "**" / "*.parquet")

    con = duckdb.connect()
    where = "1=1"
    if status:
        where += f" AND status = '{status}'"
    if resolved_only:
        where += " AND result IN ('yes', 'no')"

    query = f"""
    SELECT ticker, event_ticker, title, status, result, volume, open_interest,
           created_time, open_time, close_time
    FROM read_parquet('{pattern}', hive_partitioning=0)
    WHERE {where}
    """
    df = con.execute(query).df()
    con.close()
    return df


def load_polymarket_blocks(data_dir: str | Path) -> pd.DataFrame:
    """Load Polymarket block_number -> timestamp mapping."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Polymarket blocks directory not found: {path}")
    pattern = str(path / "*.parquet")
    if not list(path.glob("*.parquet")):
        pattern = str(path / "**" / "*.parquet")
    con = duckdb.connect()
    df = con.execute(
        f"SELECT block_number, timestamp FROM read_parquet('{pattern}', hive_partitioning=0)"
    ).df()
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_polymarket_trades(
    data_dir: str | Path,
    blocks_dir: Optional[str | Path] = None,
    *,
    min_price: float = 0.01,
    max_price: float = 0.99,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    last_n_months: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Polymarket CTF Exchange trades from Parquet.
    Derives price from maker/taker amounts (maker_asset_id=0 means USDC).
    Returns DataFrame with columns: ticker, yes_price (0-1), size, taker_side, created_time.
    Optional time filters: start_date, end_date (inclusive), or last_n_months from latest trade.
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Polymarket trades directory not found: {path}")
    files = list(path.glob("*.parquet")) or list(path.glob("**/*.parquet"))
    files = [f for f in files if not f.name.startswith("._")]
    if not files:
        return pd.DataFrame(
            columns=[
                "block_number", "transaction_hash", "log_index", "order_hash",
                "maker", "taker", "maker_asset_id", "taker_asset_id",
                "maker_amount", "taker_amount", "fee",
            ]
        )
    files_str = ", ".join(f"'{f}'" for f in sorted(files))

    con = duckdb.connect()
    query = f"""
    SELECT
        block_number,
        transaction_hash,
        log_index,
        order_hash,
        maker,
        taker,
        maker_asset_id,
        taker_asset_id,
        maker_amount,
        taker_amount,
        fee
    FROM read_parquet([{files_str}], hive_partitioning=0)
    """
    df = con.execute(query).df()
    con.close()

    if df.empty:
        return pd.DataFrame(columns=["ticker", "yes_price", "size", "taker_side", "created_time"])

    # Normalize asset IDs (may be stored as string)
    for col in ("maker_asset_id", "taker_asset_id"):
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
    for col in ("maker_amount", "taker_amount", "fee"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)

    # is_buy: maker gives USDC (maker_asset_id == 0)
    is_buy = df["maker_asset_id"] == 0
    # Price in [0,1]: USDC per outcome token. When is_buy: maker gives USDC, taker gives tokens -> price = maker_amount/taker_amount
    df["yes_price"] = np.where(
        is_buy,
        np.where(df["taker_amount"] > 0, df["maker_amount"] / df["taker_amount"], np.nan),
        np.where(df["maker_amount"] > 0, df["taker_amount"] / df["maker_amount"], np.nan),
    )
    # Size: outcome tokens traded (6 decimals in amounts)
    df["size"] = np.where(
        is_buy,
        df["taker_amount"] / 1e6,
        df["maker_amount"] / 1e6,
    ).astype(np.float64)
    # Market = outcome token asset id (same for all trades in one outcome)
    df["ticker"] = np.where(is_buy, df["taker_asset_id"].astype(str), df["maker_asset_id"].astype(str))
    df["taker_side"] = np.where(is_buy, "yes", "no")

    # Filter valid prices
    df = df.loc[
        (df["yes_price"] >= min_price) & (df["yes_price"] <= max_price)
    ].copy()

    # Timestamp: join blocks if provided
    if blocks_dir is not None:
        blocks = load_polymarket_blocks(blocks_dir)
        df = df.merge(blocks, on="block_number", how="left")
        df = df.rename(columns={"timestamp": "created_time"})
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
    else:
        # No blocks: use block_number as proxy for ordering (monotonic)
        df["created_time"] = pd.to_datetime(df["block_number"], unit="s", origin="unix", utc=True)

    df = df.sort_values(["ticker", "created_time"]).reset_index(drop=True)

    if start_date is not None or end_date is not None or last_n_months is not None:
        if last_n_months is not None:
            cutoff = df["created_time"].max() - pd.DateOffset(months=last_n_months)
            df = df.loc[df["created_time"] >= cutoff].copy()
        if start_date is not None:
            df = df.loc[df["created_time"] >= start_date].copy()
        if end_date is not None:
            df = df.loc[df["created_time"] <= end_date].copy()
        df = df.reset_index(drop=True)

    return df[["ticker", "yes_price", "size", "taker_side", "created_time"]]
