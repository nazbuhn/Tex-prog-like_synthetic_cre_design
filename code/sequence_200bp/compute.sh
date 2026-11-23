#!/usr/bin/env bash
set -euo pipefail
C1_ROOT="${1:?run}"; C1_STAGE="${2:?stage}"
C1_ENV="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["env"])' "$C1_ROOT/SETTINGS.json")"
export KERAS_BACKEND=torch PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 MPLBACKEND=Agg WANDB_MODE=disabled
unset PYTHONPATH PYTHONHOME
mkdir -p "$C1_ROOT/private" "$C1_ROOT/matplotlib_cache"
export MPLCONFIGDIR="$C1_ROOT/matplotlib_cache"
C1_LOCK="$C1_ROOT/${C1_STAGE}_${SLURM_ARRAY_TASK_ID:-single}.lock"
exec 7>"$C1_LOCK"
flock -n 7 || { echo 'This stage/shard is already running.' >&2; exit 1; }
echo "Run: $C1_ROOT ; Stage: $C1_STAGE ; Job: ${SLURM_JOB_ID:-unknown} ; Shard: ${SLURM_ARRAY_TASK_ID:-none}"
"$C1_ENV/bin/python" -u "$C1_ROOT/workflow.py" "$C1_ROOT" "$C1_STAGE" "${SLURM_ARRAY_TASK_ID:-0}"
