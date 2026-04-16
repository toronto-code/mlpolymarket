"""Load Kalshi and Polymarket trade data from Parquet."""

import os
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


def _load_from_consolidated(
    consolidated_path: Path,
    *,
    min_price: float = 0.01,
    max_price: float = 0.99,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    last_n_months: Optional[int] = None,
    exclude_tickers: Optional[set[str]] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fast path: load from a single pre-joined consolidated Parquet file produced
    by scripts/prepare_data.py.  This skips the expensive multi-shard DuckDB
    join entirely — typical load time drops from hours to seconds.
    """
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='/tmp'")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute(f"PRAGMA threads={max(1, min(4, os.cpu_count() or 1))}")

    where: list[str] = [
        f"yes_price BETWEEN {min_price} AND {max_price}"
    ]

    if last_n_months is not None:
        max_ts = con.execute(
            f"SELECT max(created_time) FROM read_parquet('{consolidated_path}')"
        ).fetchone()[0]
        if max_ts is not None:
            cutoff = (
                pd.to_datetime(max_ts, utc=True)
                - pd.DateOffset(months=int(last_n_months))
            )
            where.append(
                f"created_time >= TIMESTAMP '{cutoff.strftime('%Y-%m-%d %H:%M:%S')}'"
            )

    if start_date is not None:
        s = pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%d %H:%M:%S")
        where.append(f"created_time >= TIMESTAMP '{s}'")

    if end_date is not None:
        e = pd.to_datetime(end_date, utc=True).strftime("%Y-%m-%d %H:%M:%S")
        where.append(f"created_time <= TIMESTAMP '{e}'")

    if exclude_tickers:
        tl = ", ".join(f"'{t}'" for t in sorted(exclude_tickers))
        where.append(f"ticker NOT IN ({tl})")

    limit_sql = f"LIMIT {int(max_rows)}" if max_rows is not None else ""
    where_sql = " AND ".join(where)

    query = f"""
    SELECT ticker, yes_price, size, taker_side, created_time
    FROM read_parquet('{consolidated_path}')
    WHERE {where_sql}
    {limit_sql}
    """
    df = con.execute(query).df()
    con.close()

    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "yes_price", "size", "taker_side", "created_time"]
        )

    df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
    print(f"  Loaded {len(df):,} rows from consolidated file.", flush=True)
    return df[["ticker", "yes_price", "size", "taker_side", "created_time"]]


def load_polymarket_trades(
    data_dir: str | Path,
    blocks_dir: Optional[str | Path] = None,
    *,
    consolidated_path: Optional[str | Path] = None,
    min_price: float = 0.01,
    max_price: float = 0.99,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    last_n_months: Optional[int] = None,
    exclude_tickers: Optional[set[str]] = None,
    max_rows: Optional[int] = None,
    batch_by_month: bool = False,
    _files_override: Optional[list[Path]] = None,
) -> pd.DataFrame:
    """
    Load Polymarket CTF Exchange trades from Parquet.
    Derives price from maker/taker amounts (maker_asset_id=0 means USDC).
    Returns DataFrame with columns: ticker, yes_price (0-1), size, taker_side, created_time.
    Optional time filters: start_date, end_date (inclusive), or last_n_months from latest trade.

    Fast path: if consolidated_path points to a file produced by
    scripts/prepare_data.py, loading takes seconds instead of hours.
    """
    # ------------------------------------------------------------------ #
    # Fast path: consolidated file (pre-joined, single parquet)           #
    # ------------------------------------------------------------------ #
    if consolidated_path is not None:
        cp = Path(consolidated_path)
        if cp.exists():
            print(f"  Using consolidated file: {cp}", flush=True)
            return _load_from_consolidated(
                cp,
                min_price=min_price,
                max_price=max_price,
                start_date=start_date,
                end_date=end_date,
                last_n_months=last_n_months,
                exclude_tickers=exclude_tickers,
                max_rows=max_rows,
            )
        else:
            print(
                f"  WARNING: consolidated_path set but file not found: {cp}\n"
                f"  Run scripts/prepare_data.py first, or unset polymarket_consolidated.\n"
                f"  Falling back to raw shard loading (slow).",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # Slow path: raw shards                                               #
    # ------------------------------------------------------------------ #
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Polymarket trades directory not found: {path}")
    files = _files_override or (list(path.glob("*.parquet")) or list(path.glob("**/*.parquet")))
    files = [f for f in files if not f.name.startswith("._")]
    if not files:
        return pd.DataFrame(
            columns=[
                "block_number", "transaction_hash", "log_index", "order_hash",
                "maker", "taker", "maker_asset_id", "taker_asset_id",
                "maker_amount", "taker_amount", "fee",
            ]
        )
    limit_sql = f"LIMIT {int(max_rows)}" if max_rows is not None else ""
    n_files = len(files)
    print(f"  Reading {n_files} parquet files...", flush=True)
    files_sorted = sorted(files)

    def _files_for_month_window(all_files: list[Path], ws: pd.Timestamp, we: pd.Timestamp) -> list[Path]:
        """
        Best-effort pruning of Parquet shard list for a [ws, we] window.
        Supports common layouts:
        - Hive partitions: .../year=2026/month=03/...
        - Filenames containing YYYY-MM or YYYY_MM
        Falls back to returning all files if no pattern matches.
        """
        ws = pd.to_datetime(ws, utc=True)
        we = pd.to_datetime(we, utc=True)
        months = pd.period_range(ws.to_period("M"), we.to_period("M"), freq="M")
        month_keys = {(p.year, p.month) for p in months}

        hive_re = re.compile(r"(?:^|[\\/])year=(\d{4})(?:[\\/])month=(\d{1,2})(?:[\\/]|$)")
        name_re = re.compile(r"(\d{4})[-_](\d{2})")

        pruned: list[Path] = []
        matched_any = False
        for f in all_files:
            s = str(f)
            m = hive_re.search(s)
            if m:
                matched_any = True
                y = int(m.group(1))
                mo = int(m.group(2))
                if (y, mo) in month_keys:
                    pruned.append(f)
                continue
            m2 = name_re.search(f.name)
            if m2:
                matched_any = True
                y = int(m2.group(1))
                mo = int(m2.group(2))
                if (y, mo) in month_keys:
                    pruned.append(f)
        return pruned if matched_any and pruned else all_files

    files_str = ", ".join(f"'{f}'" for f in files_sorted)
    # Fast path: when blocks are available, push timestamp join + time filtering down into DuckDB.
    # This avoids materializing the full (potentially huge) trade dataset into pandas first.
    if blocks_dir is not None and Path(blocks_dir).exists():
        bpath = Path(blocks_dir)
        bfiles = list(bpath.glob("*.parquet")) or list(bpath.glob("**/*.parquet"))
        bfiles = [f for f in bfiles if not f.name.startswith("._")]
        if bfiles:
            bfiles_str = ", ".join(f"'{f}'" for f in sorted(bfiles))

            # Optional: load month-by-month to reduce DuckDB peak memory.
            # We ensure non-overlapping windows by using inclusive ends of:
            #   end_inclusive = next_month_start - 1 second
            if batch_by_month:
                if max_rows is not None:
                    raise ValueError(
                        "batch_by_month=True is not compatible with max_rows: "
                        "to preserve 'every data point', remove max_rows from config."
                    )

                # Compute the max blocks timestamp once; we use it to turn last_n_months
                # into a concrete window.
                con_max = duckdb.connect()
                con_max.execute("PRAGMA temp_directory='/tmp'")
                con_max.execute("PRAGMA memory_limit='6GB'")
                max_ts = con_max.execute(
                    f"SELECT max(timestamp) FROM read_parquet([{bfiles_str}], hive_partitioning=0)"
                ).fetchone()[0]
                con_max.close()
                if max_ts is None:
                    return pd.DataFrame(
                        columns=["ticker", "yes_price", "size", "taker_side", "created_time"]
                    )

                max_dt = pd.to_datetime(max_ts, utc=True)

                if start_date is not None:
                    total_start = pd.to_datetime(start_date, utc=True)
                elif last_n_months is not None:
                    total_start = max_dt - pd.DateOffset(months=int(last_n_months))
                else:
                    raise ValueError("batch_by_month=True requires start_date or last_n_months.")

                if end_date is not None:
                    total_end = pd.to_datetime(end_date, utc=True)
                elif last_n_months is not None:
                    total_end = max_dt
                else:
                    raise ValueError("batch_by_month=True requires end_date or last_n_months.")

                # We later convert timestamps to seconds precision in SQL strings, so keep
                # boundaries consistent here.
                total_start = total_start.floor("s")
                total_end = total_end.floor("s")
                # Some pandas operations may return tz-naive timestamps; ensure UTC tz-awareness.
                if getattr(total_start, "tz", None) is None:
                    total_start = total_start.tz_localize("UTC")
                if getattr(total_end, "tz", None) is None:
                    total_end = total_end.tz_localize("UTC")
                if total_end < total_start:
                    return pd.DataFrame(
                        columns=["ticker", "yes_price", "size", "taker_side", "created_time"]
                    )

                # Build non-overlapping monthly windows covering [total_start, total_end].
                windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
                cur = total_start
                while cur <= total_end:
                    # Avoid pandas Period->Timestamp conversion (it can drop timezone info).
                    month_start = cur.replace(
                        day=1,
                        hour=0,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )
                    next_month_start = month_start + pd.DateOffset(months=1)
                    end_inclusive = min(total_end, next_month_start - pd.Timedelta(seconds=1))
                    if end_inclusive < cur:
                        break
                    windows.append((cur, end_inclusive))
                    cur = end_inclusive + pd.Timedelta(seconds=1)

                print(
                    f"  Loading trades month-by-month ({len(windows)} windows) to reduce DuckDB peak memory...",
                    flush=True,
                )

                dfs: list[pd.DataFrame] = []
                for i, (ws, we) in enumerate(windows):
                    print(f"    Window {i + 1}/{len(windows)}: {ws} -> {we}", flush=True)
                    # Reduce runtime: point DuckDB only at the subset of shards likely to
                    # contain this window (if the on-disk layout encodes dates).
                    month_files = _files_for_month_window(files_sorted, ws, we)
                    if month_files is not files_sorted:
                        print(f"      Using {len(month_files):,}/{len(files_sorted):,} parquet shards for this window.", flush=True)
                    part = load_polymarket_trades(
                        data_dir,
                        blocks_dir=blocks_dir,
                        min_price=min_price,
                        max_price=max_price,
                        start_date=ws,
                        end_date=we,
                        last_n_months=None,
                        exclude_tickers=exclude_tickers,
                        max_rows=None,
                        batch_by_month=False,
                        _files_override=month_files,
                    )
                    if not part.empty:
                        dfs.append(part)

                if not dfs:
                    return pd.DataFrame(
                        columns=["ticker", "yes_price", "size", "taker_side", "created_time"]
                    )

                # Concatenate in window order; build_sequences will re-sort per ticker.
                df_all = pd.concat(dfs, ignore_index=True)
                return df_all[["ticker", "yes_price", "size", "taker_side", "created_time"]]

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
                con_max.execute("PRAGMA memory_limit='6GB'")
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
            # Use available CPU to speed up scanning many Parquet shards.
            con.execute(f"PRAGMA threads={max(1, min(4, os.cpu_count() or 1))}")
            con.execute("PRAGMA memory_limit='6GB'")
            query = f"""
            WITH blocks_filtered AS (
                SELECT
                    block_number,
                    TRY_CAST(timestamp AS TIMESTAMP) AS created_time
                FROM read_parquet([{bfiles_str}], hive_partitioning=0)
                WHERE 1=1
                {blocks_where_sql}
            ),
            trades_filtered AS (
                SELECT
                    tf.block_number,
                    tf.ticker,
                    tf.yes_price,
                    tf.size,
                    tf.taker_side
                FROM (
                    SELECT
                        t.block_number,
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
                ) tf
                WHERE tf.yes_price BETWEEN {min_price} AND {max_price}
                {ticker_excl_sql}
                {limit_sql}
            )
            SELECT
                tf.ticker,
                tf.yes_price,
                tf.size,
                tf.taker_side,
                b.created_time AS created_time
            FROM trades_filtered tf
            JOIN blocks_filtered b
            ON tf.block_number = b.block_number
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
    {limit_sql}
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
