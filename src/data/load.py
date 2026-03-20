"""Load Kalshi and Polymarket trade data from Parquet."""

import re
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# Keywords (lowercased) in question/slug that mark a market as sports.
SPORTS_KEYWORDS = re.compile(
    r"\b(sport|nfl|nba|mlb|nhl|super\s*bowl|world\s*cup|soccer|football|basketball|"
    r"baseball|hockey|ufc|boxing|tennis|golf|olympics|f1|nascar|premier\s*league|"
    r"champions\s*league|world\s*series|stanley\s*cup|playoff|mvp|championship)\b",
    re.I,
)


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


def load_polymarket_markets(data_dir: str | Path) -> pd.DataFrame:
    """Load Polymarket market metadata (question, slug, condition_id, etc.)."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Polymarket markets directory not found: {path}")
    files = list(path.glob("*.parquet")) or list(path.glob("**/*.parquet"))
    files = [f for f in files if not f.name.startswith("._")]
    if not files:
        return pd.DataFrame()
    files_str = ", ".join(f"'{f}'" for f in sorted(files))
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet([{files_str}], hive_partitioning=0)").df()
    con.close()
    return df


def get_sports_condition_ids_from_markets(markets_df: pd.DataFrame) -> set[str]:
    """Return set of condition_id (or id) for markets that look like sports from question/slug."""
    if markets_df.empty:
        return set()
    out = set()
    id_col = "condition_id" if "condition_id" in markets_df.columns else "id"
    if id_col not in markets_df.columns:
        return out
    for _, row in markets_df.iterrows():
        q = str(row.get("question", "") or "")
        s = str(row.get("slug", "") or "")
        if SPORTS_KEYWORDS.search(q) or SPORTS_KEYWORDS.search(s):
            out.add(str(row[id_col]))
    return out


def load_sports_ticker_ids_file(filepath: str | Path) -> set[str]:
    """Load a set of ticker (asset) IDs to exclude, one per line. Skip blanks and # comments."""
    path = Path(filepath)
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                out.add(line)
    return out


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
    exclude_tickers: Optional[set[str]] = None,
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
    n_files = len(files)
    print(f"  Reading {n_files} parquet files...", flush=True)
    files_str = ", ".join(f"'{f}'" for f in sorted(files))
    # Fast path: when blocks are available, push timestamp join + time filtering down into DuckDB.
    # This avoids materializing the full (potentially huge) trade dataset into pandas first.
    if blocks_dir is not None and Path(blocks_dir).exists():
        bpath = Path(blocks_dir)
        bfiles = list(bpath.glob("*.parquet")) or list(bpath.glob("**/*.parquet"))
        bfiles = [f for f in bfiles if not f.name.startswith("._")]
        if bfiles:
            bfiles_str = ", ".join(f"'{f}'" for f in sorted(bfiles))

            # Format optional date filters as UTC timestamps that DuckDB can parse.
            start_utc = (
                pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%d %H:%M:%S") if start_date is not None else None
            )
            end_utc = (
                pd.to_datetime(end_date, utc=True).strftime("%Y-%m-%d %H:%M:%S") if end_date is not None else None
            )

            ticker_excl_sql = ""
            if exclude_tickers:
                tickers_list = ", ".join(f"'{t}'" for t in sorted(exclude_tickers))
                ticker_excl_sql = f" AND ticker NOT IN ({tickers_list})"

            blocks_where_sql = ""
            if last_n_months is not None:
                # DuckDB interval syntax can be brittle across versions.
                # Compute the cutoff timestamp in Python by fetching only the
                # max blocks timestamp (cheap) and subtracting months.
                con_max = duckdb.connect()
                # DuckDB uses a local temp directory (default: ".tmp") for intermediates.
                # Ensure it points to a writable path so training doesn't fail if the
                # repo directory isn't writable by the current user.
                con_max.execute("PRAGMA temp_directory='/tmp'")
                max_ts = con_max.execute(
                    f"SELECT max(timestamp) FROM read_parquet([{bfiles_str}], hive_partitioning=0)"
                ).fetchone()[0]
                con_max.close()
                if max_ts is None:
                    return pd.DataFrame(columns=["ticker", "yes_price", "size", "taker_side", "created_time"])
                cutoff_dt = pd.to_datetime(max_ts, utc=True) - pd.DateOffset(months=int(last_n_months))
                cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
                blocks_where_sql += f" AND TRY_CAST(timestamp AS TIMESTAMP) >= TIMESTAMP '{cutoff_str}'"
            if start_utc is not None:
                blocks_where_sql += f" AND TRY_CAST(timestamp AS TIMESTAMP) >= TIMESTAMP '{start_utc}'"
            if end_utc is not None:
                blocks_where_sql += f" AND TRY_CAST(timestamp AS TIMESTAMP) <= TIMESTAMP '{end_utc}'"

            con = duckdb.connect()
            con.execute("PRAGMA temp_directory='/tmp'")
            con.execute("PRAGMA threads=1")
            query = f"""
            WITH blocks_filtered AS (
                SELECT
                    block_number,
                    TRY_CAST(timestamp AS TIMESTAMP) AS created_time
                FROM read_parquet([{bfiles_str}], hive_partitioning=0)
                WHERE 1=1
                {blocks_where_sql}
            ),
            raw AS (
                SELECT
                    b.created_time AS created_time,
                    CASE
                        WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0
                            THEN CAST(t.taker_asset_id AS VARCHAR)
                        ELSE CAST(t.maker_asset_id AS VARCHAR)
                    END AS ticker,
                    CASE
                        WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0 THEN 'yes'
                        ELSE 'no'
                    END AS taker_side,
                    CASE
                        WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0 THEN
                            CASE WHEN TRY_CAST(t.taker_amount AS DOUBLE) > 0
                                THEN TRY_CAST(t.maker_amount AS DOUBLE) / TRY_CAST(t.taker_amount AS DOUBLE)
                                ELSE NULL
                            END
                        ELSE
                            CASE WHEN TRY_CAST(t.maker_amount AS DOUBLE) > 0
                                THEN TRY_CAST(t.taker_amount AS DOUBLE) / TRY_CAST(t.maker_amount AS DOUBLE)
                                ELSE NULL
                            END
                    END AS yes_price,
                    CASE
                        WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0
                            THEN TRY_CAST(t.taker_amount AS DOUBLE) / 1e6
                        ELSE
                            TRY_CAST(t.maker_amount AS DOUBLE) / 1e6
                    END AS size
                FROM read_parquet([{files_str}], hive_partitioning=0) t
                JOIN read_parquet([{bfiles_str}], hive_partitioning=0) b
                ON t.block_number = b.block_number
            )
            SELECT ticker, yes_price, size, taker_side, created_time
            FROM raw
            WHERE yes_price BETWEEN {min_price} AND {max_price}
            {ticker_excl_sql}
            ORDER BY ticker, created_time
            """
            df = con.execute(query).df()
            con.close()

            if df.empty:
                return pd.DataFrame(columns=["ticker", "yes_price", "size", "taker_side", "created_time"])

            df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
            print(f"  Loaded {len(df):,} rows.", flush=True)
            return df[["ticker", "yes_price", "size", "taker_side", "created_time"]]

    # Fallback (no blocks available): original behavior (compute timestamps in pandas).
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='/tmp'")
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
    print(f"  Loaded {len(df):,} rows.", flush=True)

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
    # Price in [0,1]
    df["yes_price"] = np.where(
        is_buy,
        np.where(df["taker_amount"] > 0, df["maker_amount"] / df["taker_amount"], np.nan),
        np.where(df["maker_amount"] > 0, df["taker_amount"] / df["maker_amount"], np.nan),
    )
    # Size in tokens (amounts are 1e6-scaled)
    df["size"] = np.where(is_buy, df["taker_amount"] / 1e6, df["maker_amount"] / 1e6).astype(np.float64)
    df["ticker"] = np.where(is_buy, df["taker_asset_id"].astype(str), df["maker_asset_id"].astype(str))
    df["taker_side"] = np.where(is_buy, "yes", "no")

    df = df.loc[(df["yes_price"] >= min_price) & (df["yes_price"] <= max_price)].copy()

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

    if exclude_tickers:
        df = df.loc[~df["ticker"].isin(exclude_tickers)].copy()
        df = df.reset_index(drop=True)

    return df[["ticker", "yes_price", "size", "taker_side", "created_time"]]
