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

## Data

- **Real data (recommended)**: In the [prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) repo run `make setup`, then point this repo at it with `--data-dir`.
  - This gives you full **Polymarket** and **Kalshi** datasets on disk.
  - This project is configured by default to use **Polymarket only** and the **last 12 months** of trades (`data.last_n_months: 12` in `config/default.yaml`).
  - To exclude sports-betting markets, filter them out when preparing the Polymarket dataset in `prediction-market-analysis` (e.g., dropping markets whose titles/categories indicate sports) so that `data/polymarket/trades/` only contains non-sports markets.

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
