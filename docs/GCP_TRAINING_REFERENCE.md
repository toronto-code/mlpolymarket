# GCP training reference (from transcript baaa8766)

Quick reference for running the mlpolymarket pipeline on Google Cloud. Source: Cursor chat that built the pipeline and set up the VM.

---

## VM specs (from that chat)

- **Region:** northamerica-northeast2 (Toronto)
- **Zone:** northamerica-northeast2-a (or -b, -c; if “Any” fails, try a specific zone or us-central1)
- **Machine type:** e2-standard-4 (4 vCPU, 16 GB RAM)
- **Boot disk:** 150 GB, Ubuntu 22.04 LTS
- **Example VM name from chat:** `instance-20260222-194709`

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
sudo apt-get install -y git zstd aria2 python3-pip python3-venv
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
