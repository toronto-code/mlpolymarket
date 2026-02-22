#!/usr/bin/env python3
"""
Run multiple training runs with different configs; select best by validation metric.
Each run writes to output_root/<run_name>/; best run is reported at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins for leaf values."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep over configs; pick best by validation metric."
    )
    parser.add_argument("--config", default="config/sweep.yaml", help="Sweep config YAML")
    parser.add_argument("--data-dir", default=None, help="Data root (passed to run.py)")
    parser.add_argument("--dry-run", action="store_true", help="Print configs only, do not run")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sweep_cfg = load_config(root / args.config)
    base_path = root / sweep_cfg.get("base_config", "config/default.yaml")
    base = load_config(base_path)
    output_root = Path(sweep_cfg.get("output_root", "output/sweep"))
    runs = sweep_cfg.get("runs", [])
    metric = sweep_cfg.get("metric", "mae")
    minimize = sweep_cfg.get("minimize", True)
    selection_model = sweep_cfg.get("selection_model", "lstm")

    if not runs:
        print("No runs in sweep config.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    run_dirs = []
    for i, run_spec in enumerate(runs):
        if isinstance(run_spec, dict):
            name = run_spec.get("name", "run_{}".format(i))
            overrides = {k: v for k, v in run_spec.items() if k != "name"}
        else:
            name = "run_{}".format(i)
            overrides = {}
        merged = deep_merge(base, overrides)
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
        run_dirs.append((name, run_dir, config_path))

    if args.dry_run:
        for name, run_dir, config_path in run_dirs:
            print("Would run: {} -> {}".format(name, run_dir))
        return

    for name, run_dir, config_path in run_dirs:
        print("\n--- Run: {} ---\n".format(name))
        cmd = [
            sys.executable,
            str(root / "run.py"),
            "--config", str(config_path),
            "--output-dir", str(run_dir),
        ]
        if args.data_dir:
            cmd.extend(["--data-dir", args.data_dir])
        subprocess.run(cmd, cwd=str(root))
        print("Finished {}.\n".format(name))

    # Load metrics from each run; select best by val metric for selection_model
    best_name = None
    best_val = None
    results = []
    for name, run_dir, _ in run_dirs:
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            results.append({"name": name, "val": None, "test": None})
            continue
        with open(metrics_file) as f:
            data = json.load(f)
        val_metrics = data.get("val_metrics", {})
        test_metrics = data.get("test_metrics", {})
        val_val = val_metrics.get(selection_model, {}).get(metric)
        test_val = test_metrics.get(selection_model, {}).get(metric)
        results.append({"name": name, "val": val_val, "test": test_val, "data": data})
        if val_val is None:
            continue
        if best_val is None:
            best_val = val_val
            best_name = name
        else:
            if minimize and val_val < best_val:
                best_val = val_val
                best_name = name
            elif not minimize and val_val > best_val:
                best_val = val_val
                best_name = name

    print("\n--- Sweep summary ---")
    print("Metric: {} ({}). Selection model: {}.\n".format(
        metric, "minimize" if minimize else "maximize", selection_model))
    for r in results:
        v = r.get("val")
        t = r.get("test")
        vstr = "{:.4f}".format(v) if v is not None else "N/A"
        tstr = "{:.4f}".format(t) if t is not None else "N/A"
        mark = " [best]" if r["name"] == best_name else ""
        print("  {}  val_{}={}  test_{}={}{}".format(r["name"], metric, vstr, metric, tstr, mark))
    if best_name:
        print("\nBest run by val {}: {} (test {} = {:.4f})".format(
            metric, best_name, metric,
            next((r["test"] for r in results if r["name"] == best_name), None) or float("nan")))
        print("Artifacts: {}.".format(output_root / best_name))

    summary_path = output_root / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "metric": metric,
                "minimize": minimize,
                "selection_model": selection_model,
                "best_run": best_name,
                "best_val": best_val,
                "runs": [{"name": r["name"], "val": r.get("val"), "test": r.get("test")} for r in results],
            },
            f,
            indent=2,
        )
    print("Summary written to {}.".format(summary_path))


if __name__ == "__main__":
    main()
