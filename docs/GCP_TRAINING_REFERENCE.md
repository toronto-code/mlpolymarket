# GCP training reference (from transcript baaa8766)

Quick reference for running the mlpolymarket pipeline on Google Cloud. Source: Cursor chat that built the pipeline and set up the VM.

---

## VM specs (updated baseline)

- **Region:** northamerica-northeast2 (Toronto)
- **Zone:** northamerica-northeast2-b (or another zone where your instance exists)
- **Machine type:** use enough RAM for your run mode
  - 32 GB RAM VM works with consolidated streaming + safer sweep settings
  - Full-data no-cap sweeps may require significantly more RAM
- **Boot disk:** 150 GB minimum (more if keeping large outputs)
- **Example VM names from recent runs:** `instance-20260428-034135`, `instance-20260415-144541`

Replace `YOUR_VM_NAME`, `YOUR_ZONE`, `YOUR_PROJECT_ID` below with your actual values.

---

## 1. SSH into the VM

**From Mac:**

```bash
gcloud compute ssh YOUR_VM_NAME --zone=northamerica-northeast2-a --project=YOUR_PROJECT_ID
```

Or in Console: **Compute Engine → VM instances → SSH**.

---

## 2. On the VM: install tools (once)

```bash
sudo apt-get update
sudo apt-get install -y git zstd aria2 python3-pip python3-venv screen ripgrep
```

---

## 3. Download the dataset (once; ~36 GB download, ~70 GB after extract)

```bash
cd ~
git clone --depth 1 https://github.com/Jon-Becker/prediction-market-analysis.git
cd prediction-market-analysis
bash scripts/download.sh
```

- Runs on the VM; data stays on the VM.
- Ignore `LIBARCHIVE.xattr.com.apple.provenance` messages; extraction is fine.
- If you see `aria2c not found, falling back to curl`, install `aria2` first for faster download.

Sanity check before any training:

```bash
ls ~/prediction-market-analysis/data/polymarket/trades | head
ls ~/prediction-market-analysis/data/polymarket/blocks | head
```

If either path is missing, stop and fix data setup before running `run.py`/`sweep.py`.

---

## 4. Clone mlpolymarket

```bash
cd ~
git clone https://github.com/toronto-code/mlpolymarket.git
cd mlpolymarket
```

---

## 5. Python env and dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas pyarrow duckdb scikit-learn torch pyyaml tqdm
```

If you get `ModuleNotFoundError: No module named 'src.data'`, ensure `run.py` adds the project root to `sys.path` (see repo history / “Fix: add project root to path for src imports”).

---

## 5b. Build consolidated parquet once (recommended)

This avoids repeated expensive raw-shard joins and is the most reliable path for large sweeps.

```bash
cd ~/mlpolymarket
source .venv/bin/activate
python scripts/prepare_data.py \
  --data-dir ~/prediction-market-analysis \
  --months 12 \
  --memory-limit 24GB \
  --threads 8 \
  --resume
```

Confirm output:

```bash
ls -lh ~/prediction-market-analysis/data/polymarket/consolidated/trades_with_timestamps.parquet
```

---

## 6. Run training

Single run (Polymarket, last 12 months):

```bash
python run.py --data-dir ~/prediction-market-analysis
```

Sweep (multiple configs, pick best by validation):

```bash
python sweep.py --config config/sweep.yaml --data-dir ~/prediction-market-analysis
```

Rough time: ~30 min–2 hours depending on data size.

---

## 6b. Run training so it survives disconnects (recommended)

If you run training in the browser SSH window and close your laptop/tab, the SSH session can drop and terminate the Python process. Use one of these approaches.

### Option A: Run the background script inside `screen` (most reliable)

So the job is never in the SSH session—it runs under `screen`, which keeps going after you disconnect.

```bash
sudo apt-get install -y screen
cd ~/mlpolymarket
screen -S train
DATA_DIR=~/prediction-market-analysis ./scripts/run_background.sh sweep
```

Then **immediately** detach so closing your laptop doesn’t kill anything: press **`Ctrl+A`** then **`D`**.

Check later (new SSH, then):

```bash
tail -f ~/mlpolymarket/output/logs/sweep_*.log
# or: pgrep -af sweep.py
```

### Option B: `screen` with interactive run (see live output)

```bash
sudo apt-get install -y screen
cd ~/mlpolymarket
screen -S train
source .venv/bin/activate
python sweep.py --config config/sweep.yaml --data-dir ~/prediction-market-analysis
```

Detach (keeps running): press **`Ctrl+A`** then **`D`**. Re-attach later: `screen -r train`.

### Option C: background script only (no screen)

```bash
cd ~/mlpolymarket
chmod +x scripts/run_background.sh
DATA_DIR=~/prediction-market-analysis ./scripts/run_background.sh sweep
```

Then watch logs: `tail -f output/logs/sweep_*.log`. To see if still running: `pgrep -af sweep.py`.

---

## 6c. Preflight checks that prevent wasted runs

### Smoke test (must produce metrics)

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

If `metrics.json` exists, recording is healthy.

### Early sentinel check (first 2-5 min of sweep)

```bash
LOG=$(ls -t output/logs/sweep_*.log | head -n 1)
tail -n 120 "$LOG"
grep -nE "Consolidated parquet found|Wrote .*metrics.json|not found|Point --data-dir|Traceback|ERROR" "$LOG"
```

You want:

- `Consolidated parquet found`
- later `Wrote .../metrics.json`

You do not want:

- `trades directory not found`
- `Point --data-dir ...`
- `Traceback`

### 32GB RAM safety note

If logs show huge memory estimates (for example `X ~= 179 GB`), stop and rerun with safer config:

- enable disk-backed sequences (`training.sequences_dir`), and/or
- set `data.max_samples` for sweeps.

---

## 7. Copy results to your Mac (from Mac terminal)

```bash
gcloud compute scp --recurse YOUR_VM_NAME:~/mlpolymarket/output ./mlpolymarket-output --zone=northamerica-northeast2-a --project=YOUR_PROJECT_ID
```

Example (from transcript):

```bash
gcloud compute scp --recurse instance-20260222-194709:~/mlpolymarket/output ./mlpolymarket-output --zone=northamerica-northeast2-a --project=YOUR_PROJECT_ID
```

---

## 8. Stop the VM when done

**Console:** Compute Engine → VM instances → select VM → **Stop**.

- Stopped VM: you pay for disk only (~$6.60/month for 150 GB).
- To run again: Start VM, SSH in; data is still there, no re-download. Run `git pull` in `~/mlpolymarket` if you pushed changes.

---

## Local (Mac) workflow when you change code

```bash
cd ~/mlpolymarket   # or /Users/rupertkahng/mlpolymarket
git add .
git commit -m "Describe what you changed"
git push
```

Then on the VM:

```bash
cd ~/mlpolymarket
git pull
source .venv/bin/activate
python run.py --data-dir ~/prediction-market-analysis
```

---

## Transcript location

Full chat transcript (raw JSONL):

`~/.cursor/projects/Users-rupertkahng-polybot-feb2026/agent-transcripts/baaa8766-4606-4414-bdc4-10d208426f46/baaa8766-4606-4414-bdc4-10d208426f46.jsonl`
