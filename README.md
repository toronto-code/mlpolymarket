# Next-price prediction (Polymarket)

Short-horizon price prediction on **Polymarket** trade data: next trade price from order flow and recent prices. Train/validation/test split by time; validation used for early stopping and model selection.

## What it does

- Loads Polymarket CTF trades from Parquet (compatible with [prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis)), optionally restricted to the **past year** (`last_n_months: 12`).
- Builds sequences (default length 32) with features: `yes_price`, `size`, `order_flow`, `vwap`. Target: next trade `yes_price`.
- **Train / val / test** by time (default 60/20/20). No lookahead; test is used once for reporting.
- Optional **feature normalization** (per-feature mean/std on train).
- Models: last-price baseline, VWAP baseline, **MLP**, **LSTM** (configurable depth, dropout, LR). Early stopping on validation loss.
- Saves best model weights (`mlp_best.pt`, `lstm_best.pt`) and **metrics** (val + test) to `output/` or a chosen dir.
- **Sweep**: run multiple configs (e.g. different sequence length, LSTM size); select best by validation MAE and report that run’s test metrics.

## Setup

```bash
cd /path/to/mlpolymarket
uv sync
```

(Or `pip install -e .` with the project’s dependencies.)

For a plain venv setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas pyarrow duckdb scikit-learn torch pyyaml tqdm
```

## Data

- **Real data (recommended)**: In the [prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) repo run `make setup`, then point this repo at it with `--data-dir`.
  - This gives you full **Polymarket** and **Kalshi** datasets on disk.
  - This project is configured by default to use **Polymarket only** and the **last 12 months** of trades (`data.last_n_months: 12` in `config/default.yaml`).
  - **Exclude sports**: set `data.exclude_sports: true` in `config/default.yaml` and create `data/polymarket/sports_tickers.txt` (one outcome-token ID per line). Generate it once with:
    ```bash
    pip install requests  # optional, for the script
    python scripts/generate_sports_tickers.py --data-dir /path/to/prediction-market-analysis
    ```
    The script uses your markets Parquet (question/slug) to detect sports and the Polymarket Gamma API to resolve token IDs.

### Path sanity check (before every long run)

Always verify expected data paths exist on the VM:

```bash
ls ~/prediction-market-analysis/data/polymarket/trades | head
ls ~/prediction-market-analysis/data/polymarket/blocks | head
```

If either path is missing, do not start training.

### Consolidated parquet (recommended for reliability and speed)

Build a pre-joined consolidated parquet once (per VM/dataset copy):

```bash
python scripts/prepare_data.py \
  --data-dir ~/prediction-market-analysis \
  --months 12 \
  --memory-limit 24GB \
  --threads 8 \
  --resume
```

Then verify:

```bash
ls -lh ~/prediction-market-analysis/data/polymarket/consolidated/trades_with_timestamps.parquet
```

With `data.polymarket_consolidated` set in config, `run.py` uses the streaming consolidated loader and avoids repeated raw-shard joins.

## Single run

```bash
# Polymarket, last 12 months, train/val/test, save to output/
python run.py --data-dir /path/to/prediction-market-analysis

# Custom output dir (e.g. for a sweep sub-run)
python run.py --data-dir /path/to/data --output-dir output/exp1

# Debug: fewer markets, skip LSTM
python run.py --data-dir /path/to/data --max-markets 100 --no-lstm
```

Config: `config/default.yaml` (data path, `last_n_months`, sequence length, split fractions, normalization, MLP/LSTM hyperparameters, output dir).

## Multiple runs (sweep / optimization)

```bash
python sweep.py --config config/sweep.yaml --data-dir /path/to/data
```

- Reads `config/sweep.yaml`: base config plus a list of runs (each run = base + overrides).
- For each run, writes a merged config to `output/sweep/<run_name>/config.yaml` and runs `run.py` with that config and output dir.
- After all runs, loads each `output/sweep/<run_name>/metrics.json`, selects the **best run** by validation MAE for the chosen model (default: LSTM), and prints a summary plus test metric for the best run.
- Full results: `output/sweep/sweep_summary.json` and per-run dirs.

Edit `config/sweep.yaml` to add or change runs (e.g. different `sequence.length`, `lstm.hidden`, `training.lr`).

### Detached sweep launch (safe for laptop close / SSH drops)

```bash
cd ~/mlpolymarket
DATA_DIR=~/prediction-market-analysis ./scripts/run_background.sh sweep
```

Check status:

```bash
pgrep -af "python.*sweep.py|python.*run.py"
tail -n 120 "$(ls -t output/logs/sweep_*.log | head -n 1)"
```

### Smoke test before expensive sweeps

This verifies metrics writing end-to-end.

```bash
cat > /tmp/smoke.yaml <<'YAML'
seed: 42
data:
  polymarket_consolidated: data/polymarket/consolidated/trades_with_timestamps.parquet
  last_n_months: 1
  max_rows: 200000
sequence:
  min_trades_per_market: 20
training:
  epochs: 1
  early_stopping_patience: 1
models:
  baseline_last_price: true
  baseline_vwap: true
  mlp: false
  lstm: false
output:
  save_models: false
  save_metrics: true
YAML

python run.py --config /tmp/smoke.yaml --data-dir ~/prediction-market-analysis --output-dir /tmp/metrics_smoke
ls -lh /tmp/metrics_smoke/metrics.json
```

If `metrics.json` exists, recording pipeline is healthy.

### 32GB RAM note

On 32GB VMs, a no-cap 12-month run can attempt very large in-RAM arrays. For reliability:

- Set `training.sequences_dir` (disk-backed sequence store), and/or
- Set `data.max_samples` to a safe cap for sweeps.

Use full-data no-cap runs only with sufficient RAM/disk and expected runtime.

## Output layout

After a single run:

- `output/metrics.json`: config snapshot, `val_metrics`, `test_metrics`, `n_train`/`n_val`/`n_test`, paths to saved models.
- `output/mlp_best.pt`, `output/lstm_best.pt`: PyTorch state dicts (if `save_models: true`).

After a sweep:

- `output/sweep/<run_name>/`: same as above for each run.
- `output/sweep/sweep_summary.json`: best run name, best val metric, and per-run val/test metrics.

## Project layout

```
config/
  default.yaml   # data (last_n_months, paths), sequence, split, models, training, mlp/lstm, output
  sweep.yaml     # base_config, runs (name + overrides), metric, selection_model
run.py           # single run: load Polymarket (past year), build seq, train/val/test, save
sweep.py         # multi-run: merge configs, run.py per run, pick best by val metric
src/
  data/          # load_polymarket_trades (with last_n_months), build_sequences, SequenceScaler, time_based_split_three_way
  models/        # LastPriceBaseline, VWAPBaseline, MLPModel, LSTMModel (fit with explicit val, get/load_state_dict)
  eval/          # compute_metrics, print_metrics

```

## Notes

- All splits are **by time** so the test set is strictly after train and val.
- Features can be **normalized** (mean/std on train) before training; recommended for the neural nets.
- Polymarket prices are in [0, 1]. Metrics (MAE, direction accuracy, etc.) are in the same scale.
