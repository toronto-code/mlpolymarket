#!/usr/bin/env python3
"""
Generate data/polymarket/sports_tickers.txt for use with exclude_sports.

Reads markets from prediction-market-analysis (question/slug), detects sports,
then fetches Polymarket Gamma API to get outcome token IDs for those markets
and writes one ticker ID per line. Run once with --data-dir pointing at the
dataset root; then run.py with exclude_sports: true will use the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.data.load import (
    load_polymarket_markets,
    get_sports_condition_ids_from_markets,
)

try:
    import requests
except ImportError:
    requests = None


def fetch_token_ids_for_conditions(condition_ids: set[str]) -> set[str]:
    """Resolve condition_id -> clob token IDs via Polymarket Gamma API."""
    if not condition_ids or not requests:
        return set()
    token_ids = set()
    # Gamma API returns markets; filter by conditionId and collect clobTokenIds
    url = "https://gamma-api.polymarket.com/markets"
    try:
        r = requests.get(url, params={"limit": 5000}, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("Warning: could not fetch Gamma API ({}). Write sports_tickers.txt manually.".format(e))
        return set()
    if isinstance(data, dict):
        data = data.get("data") or data.get("markets") or data.get("results") or []
    if not isinstance(data, list):
        data = []
    for m in data:
        cid = m.get("conditionId") or m.get("condition_id")
        if not cid or cid not in condition_ids:
            continue
        tokens = m.get("clobTokenIds") or m.get("clob_token_ids") or m.get("tokens")
        if isinstance(tokens, list):
            for t in tokens:
                if t:
                    token_ids.add(str(t))
        elif isinstance(tokens, str):
            for t in tokens.split(","):
                t = t.strip()
                if t:
                    token_ids.add(t)
    return token_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate sports_tickers.txt for exclude_sports.")
    ap.add_argument("--data-dir", type=Path, required=True, help="Root containing data/polymarket/markets")
    ap.add_argument("--out", type=Path, default=None, help="Output file (default: data-dir/polymarket/sports_tickers.txt)")
    args = ap.parse_args()
    data_root = args.data_dir
    markets_dir = data_root / "data" / "polymarket" / "markets"
    if not markets_dir.exists():
        print("Markets dir not found: {}".format(markets_dir))
        sys.exit(1)
    out_path = args.out or (data_root / "data" / "polymarket" / "sports_tickers.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading markets from {}...".format(markets_dir))
    markets = load_polymarket_markets(markets_dir)
    sports_cids = get_sports_condition_ids_from_markets(markets)
    print("Found {} sports condition IDs (by question/slug).".format(len(sports_cids)))
    if not sports_cids:
        print("No sports markets detected. Writing empty file.")
        out_path.write_text("")
        return
    print("Fetching token IDs from Polymarket Gamma API...")
    token_ids = fetch_token_ids_for_conditions(sports_cids)
    print("Resolved {} outcome token IDs.".format(len(token_ids)))
    with open(out_path, "w") as f:
        for t in sorted(token_ids):
            f.write(t + "\n")
    print("Wrote {}.".format(out_path))


if __name__ == "__main__":
    main()
