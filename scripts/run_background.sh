#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create it first:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r <your deps>"
  exit 1
fi

DATA_DIR="${DATA_DIR:-$HOME/prediction-market-analysis}"
MODE="${1:-sweep}" # sweep | run

timestamp="$(date +"%Y%m%d_%H%M%S")"
log_dir="${ROOT_DIR}/output/logs"
mkdir -p "$log_dir"

case "$MODE" in
  sweep)
    cmd=(.venv/bin/python sweep.py --config config/sweep.yaml --data-dir "$DATA_DIR")
    ;;
  run)
    cmd=(.venv/bin/python run.py --data-dir "$DATA_DIR")
    ;;
  *)
    echo "Usage: $0 [sweep|run]"
    exit 2
    ;;
esac

log_file="${log_dir}/${MODE}_${timestamp}.log"
pid_file="${log_dir}/${MODE}_${timestamp}.pid"

echo "Starting: ${cmd[*]}"
echo "Log: $log_file"

nohup "${cmd[@]}" >"$log_file" 2>&1 </dev/null &
pid="$!"
echo "$pid" > "$pid_file"

echo "PID: $pid"
echo "To watch logs:"
echo "  tail -f \"$log_file\""
echo "To check if still running:"
echo "  ps -p $pid"
