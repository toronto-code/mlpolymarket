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
# Optional: override YAML (paths relative to repo root or absolute).
SWEEP_CONFIG="${SWEEP_CONFIG:-config/sweep.yaml}"
CONFIG="${CONFIG:-}" # for run mode only; if empty, run.py uses its default config
MODE="${1:-sweep}" # sweep | run

timestamp="$(date +"%Y%m%d_%H%M%S")"
log_dir="${ROOT_DIR}/output/logs"
mkdir -p "$log_dir"

case "$MODE" in
  sweep)
    cmd=(.venv/bin/python sweep.py --config "$SWEEP_CONFIG" --data-dir "$DATA_DIR")
    ;;
  run)
    if [[ -n "$CONFIG" ]]; then
      cmd=(.venv/bin/python run.py --config "$CONFIG" --data-dir "$DATA_DIR")
    else
      cmd=(.venv/bin/python run.py --data-dir "$DATA_DIR")
    fi
    ;;
  *)
    echo "Usage: $0 [sweep|run]"
    exit 2
    ;;
esac

log_file="${log_dir}/${MODE}_${timestamp}.log"
pid_file="${log_dir}/${MODE}_${timestamp}.pid"

if [[ "$MODE" == "sweep" ]]; then
  echo "SWEEP_CONFIG=$SWEEP_CONFIG"
else
  [[ -n "$CONFIG" ]] && echo "CONFIG=$CONFIG" || echo "CONFIG=(default: config/default.yaml)"
fi
echo "Starting: ${cmd[*]}"
echo "Log: $log_file"

# Run in a new session (setsid) so SSH disconnect / SIGHUP never reaches this process.
# nohup + redirect stdin so it's fully detached from the terminal.
setsid nohup "${cmd[@]}" >"$log_file" 2>&1 </dev/null &
pid="$!"
echo "$pid" > "$pid_file"
disown -h 2>/dev/null || true

echo "PID: $pid (session leader; Python may be a child)"
echo "To watch logs:"
echo "  tail -f \"$log_file\""
echo "To check if still running:"
echo "  pgrep -af sweep.py   # or  pgrep -af run.py"
