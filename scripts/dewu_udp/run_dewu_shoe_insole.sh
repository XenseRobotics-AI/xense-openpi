#!/usr/bin/env bash
# Launch the dewu-shoe-insole run. Single-instance via flock; `exec` makes the
# python process inherit the lock and keep this script's PID, so a SIGINT sent
# to this PID reaches main.py's graceful-shutdown handler directly.
set -euo pipefail

REPO=/home/vertax/xense-openpi
PY=/home/vertax/miniforge3/envs/lerobot-xense/bin/python
LOCK=/tmp/dewu-shoe-insole.lock

cd "$REPO"
mkdir -p logs

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "dewu-shoe-insole already running, refusing to start" >&2
    exit 1
fi

LOG="logs/dewu-$(date +%F_%H%M%S).log"
# Mirror output to the terminal and the log file. tee runs in a process
# substitution, so `exec` still hands this PID to python and SIGINT reaches it.
exec > >(tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
exec "$PY" -m examples.bi_flexiv_rizon4_rt.main --args.run dewu-shoe-insole
