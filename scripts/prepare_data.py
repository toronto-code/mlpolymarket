#!/usr/bin/env python3
"""
One-time data preparation: consolidate 40 000+ raw Parquet shards into a
single pre-joined Parquet file.

Run this ONCE on the VM before any training.  Every subsequent training run
(including sweeps) will load the single consolidated file instead of scanning
tens of thousands of shards, cutting load time from hours to seconds.

Usage:
    cd ~/mlpolymarket
    source .venv/bin/activate
    screen -S prepare
    python scripts/prepare_data.py --data-dir ~/prediction-market-analysis
    # Ctrl+A then D to detach — this can take 1-3 hours depending on data size.

Output:
    <data-dir>/data/polymarket/consolidated/trades_with_timestamps.parquet
    (~2-8 GB, contains: ticker, yes_price, size, taker_side, created_time)

After this completes, add to your config:
    data:
      polymarket_consolidated: data/polymarket/consolidated/trades_with_timestamps.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


def build_file_lists(
    trades_dir: Path, blocks_dir: Path
) -> tuple[list[Path], list[Path]]:
    trade_files = list(trades_dir.glob("*.parquet")) or list(
        trades_dir.glob("**/*.parquet")
    )
    trade_files = [f for f in trade_files if not f.name.startswith("._")]
    block_files = list(blocks_dir.glob("*.parquet")) or list(
        blocks_dir.glob("**/*.parquet")
    )
    block_files = [f for f in block_files if not f.name.startswith("._")]
    return sorted(trade_files), sorted(block_files)


def get_block_time_range(
    block_files: list[Path], con: duckdb.DuckDBPyConnection
) -> tuple[pd.Timestamp, pd.Timestamp]:
    bf_str = ", ".join(f"'{f}'" for f in block_files)
    row = con.execute(
        f"SELECT min(timestamp), max(timestamp) "
        f"FROM read_parquet([{bf_str}], hive_partitioning=0)"
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("No block data found — check blocks directory.")
    return pd.to_datetime(row[0], utc=True), pd.to_datetime(row[1], utc=True)


def month_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start.floor("s")
    end = end.floor("s")
    while cur <= end:
        ms = cur.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nxt = ms + pd.DateOffset(months=1)
        we = min(end, nxt - pd.Timedelta(seconds=1))
        if we < cur:
            break
        windows.append((cur, we))
        cur = we + pd.Timedelta(seconds=1)
    return windows


def consolidate_window(
    ws: pd.Timestamp,
    we: pd.Timestamp,
    tf_str: str,
    bf_str: str,
    min_price: float,
    max_price: float,
    out_path: Path,
    con: duckdb.DuckDBPyConnection,
) -> int:
    ws_s = ws.strftime("%Y-%m-%d %H:%M:%S")
    we_s = we.strftime("%Y-%m-%d %H:%M:%S")

    query = f"""
    COPY (
        WITH blocks_filtered AS (
            SELECT
                block_number,
                TRY_CAST(timestamp AS TIMESTAMP) AS created_time
            FROM read_parquet([{bf_str}], hive_partitioning=0)
            WHERE TRY_CAST(timestamp AS TIMESTAMP)
                BETWEEN TIMESTAMP '{ws_s}' AND TIMESTAMP '{we_s}'
        )
        SELECT
            CASE
                WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0
                    THEN CAST(t.taker_asset_id AS VARCHAR)
                ELSE CAST(t.maker_asset_id AS VARCHAR)
            END AS ticker,
            CASE
                WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0 THEN
                    CASE WHEN TRY_CAST(t.taker_amount AS DOUBLE) > 0
                        THEN TRY_CAST(t.maker_amount AS DOUBLE)
                             / TRY_CAST(t.taker_amount AS DOUBLE)
                        ELSE NULL END
                ELSE
                    CASE WHEN TRY_CAST(t.maker_amount AS DOUBLE) > 0
                        THEN TRY_CAST(t.taker_amount AS DOUBLE)
                             / TRY_CAST(t.maker_amount AS DOUBLE)
                        ELSE NULL END
            END AS yes_price,
            CASE
                WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0
                    THEN TRY_CAST(t.taker_amount AS DOUBLE) / 1e6
                ELSE TRY_CAST(t.maker_amount AS DOUBLE) / 1e6
            END AS size,
            CASE
                WHEN COALESCE(TRY_CAST(t.maker_asset_id AS BIGINT), -1) = 0
                    THEN 'yes'
                ELSE 'no'
            END AS taker_side,
            b.created_time
        FROM read_parquet([{tf_str}], hive_partitioning=0) t
        JOIN blocks_filtered b ON t.block_number = b.block_number
        WHERE yes_price BETWEEN {min_price} AND {max_price}
    ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    count = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]
    return int(count)


def prepare(
    data_dir: Path,
    output_path: Path,
    months: int,
    min_price: float,
    max_price: float,
    memory_limit: str,
    threads: int,
    resume: bool,
) -> None:
    trades_dir = data_dir / "data/polymarket/trades"
    blocks_dir = data_dir / "data/polymarket/blocks"

    if not trades_dir.exists():
        raise FileNotFoundError(f"Trades dir not found: {trades_dir}")
    if not blocks_dir.exists():
        raise FileNotFoundError(f"Blocks dir not found: {blocks_dir}")

    trade_files, block_files = build_file_lists(trades_dir, blocks_dir)
    if not trade_files:
        raise FileNotFoundError(f"No parquet files found in {trades_dir}")
    if not block_files:
        raise FileNotFoundError(f"No parquet files found in {blocks_dir}")

    print(
        f"Trade shards : {len(trade_files):,}",
        f"\nBlock shards : {len(block_files):,}",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf_str = ", ".join(f"'{f}'" for f in trade_files)
    bf_str = ", ".join(f"'{f}'" for f in block_files)

    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='/tmp'")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")

    print("Scanning block timestamp range...", flush=True)
    min_dt, max_dt = get_block_time_range(block_files, con)
    cutoff_dt = max_dt - pd.DateOffset(months=months)
    start_dt = max(min_dt, cutoff_dt)
    print(f"  Window : {start_dt} → {max_dt}", flush=True)

    windows = month_windows(start_dt, max_dt)
    print(f"  Monthly windows: {len(windows)}", flush=True)

    temp_dir = output_path.parent / "_temp"
    temp_dir.mkdir(exist_ok=True)
    temp_files: list[Path] = []
    total_rows = 0

    for i, (ws, we) in enumerate(windows):
        temp_path = temp_dir / f"month_{i:02d}.parquet"
        temp_files.append(temp_path)

        if resume and temp_path.exists():
            try:
                count = con.execute(
                    f"SELECT count(*) FROM read_parquet('{temp_path}')"
                ).fetchone()[0]
                if count > 0:
                    print(
                        f"  [{i + 1}/{len(windows)}] {ws.date()} → {we.date()} "
                        f"SKIPPED (exists, {count:,} rows)",
                        flush=True,
                    )
                    total_rows += count
                    continue
                else:
                    print(
                        f"  [{i + 1}/{len(windows)}] {ws.date()} → {we.date()} "
                        f"temp file empty, re-running...",
                        flush=True,
                    )
                    temp_path.unlink()
            except Exception:
                print(
                    f"  [{i + 1}/{len(windows)}] {ws.date()} → {we.date()} "
                    f"temp file corrupt, re-running...",
                    flush=True,
                )
                temp_path.unlink()

        print(
            f"  [{i + 1}/{len(windows)}] {ws.date()} → {we.date()} ...",
            end=" ",
            flush=True,
        )
        count = consolidate_window(
            ws, we, tf_str, bf_str, min_price, max_price, temp_path, con
        )
        total_rows += count
        print(f"{count:,} rows", flush=True)

    print(f"\nMerging {len(temp_files)} monthly files → {output_path} ...", flush=True)
    existing = [str(f) for f in temp_files if f.exists()]
    if not existing:
        print("ERROR: no temp files produced.", file=sys.stderr)
        sys.exit(1)

    existing_str = ", ".join(f"'{f}'" for f in existing)
    con.execute(
        f"COPY (SELECT * FROM read_parquet([{existing_str}])) "
        f"TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    final_count = con.execute(
        f"SELECT count(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]
    con.close()

    print(f"Done. {final_count:,} rows → {output_path}", flush=True)
    size_gb = output_path.stat().st_size / 1e9
    print(f"File size : {size_gb:.2f} GB", flush=True)

    # Clean up temp files only on success
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(
        "\nNext step: add this to your config YAML under 'data:':\n"
        f"  polymarket_consolidated: {output_path.relative_to(data_dir)}\n"
        "Then run training normally — data loads in seconds.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time consolidation of raw Polymarket Parquet shards into a "
            "single pre-joined file.  Run once; all training runs load from it."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root of prediction-market-analysis repo (contains data/polymarket/)",
    )
    parser.add_argument(
        "--months", type=int, default=12, help="Months of history to consolidate"
    )
    parser.add_argument("--min-price", type=float, default=0.01)
    parser.add_argument("--max-price", type=float, default=0.99)
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory limit (raise on larger VMs, e.g. '16GB')",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="DuckDB thread count"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip monthly temp files that already exist (resume after failure)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = (
        data_dir / "data/polymarket/consolidated/trades_with_timestamps.parquet"
    )

    print(f"Data dir : {data_dir}")
    print(f"Output   : {output_path}")
    print(f"Months   : {args.months}")
    print(f"Memory   : {args.memory_limit}")
    print(f"Threads  : {args.threads}")
    print(f"Resume   : {args.resume}")
    print()

    prepare(
        data_dir=data_dir,
        output_path=output_path,
        months=args.months,
        min_price=args.min_price,
        max_price=args.max_price,
        memory_limit=args.memory_limit,
        threads=args.threads,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
