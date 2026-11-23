#!/usr/bin/env bash
set -euo pipefail
RUN="$1"
STAGE="$2"
shift 2
# Slurm executes a spool copy of this script. Resolve the package from the
# persistent run directory (<package>/runs/<run>), not this script's location.
PACKAGE="$(cd -- "$RUN/../.." && pwd)"
[[ -f "$PACKAGE/workflow.py" && -f "$PACKAGE/SETUP_COMPUTE.sh" ]] || {
    echo "Cannot locate analysis package from run directory: $RUN" >&2
    exit 1
}
export KERAS_BACKEND=torch MPLBACKEND=Agg PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20261112
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLCONFIGDIR="$RUN/matplotlib_cache" NUMBA_CACHE_DIR="$RUN/numba_cache"
export CRESTED_DATA_DIR="$RUN/database_cache"
export PATH='/vast/projects/saargill/cell-type-specific/nazbuhn/immgen/c1_core_analysis_tools/envs/meme_cpu_v1/bin':"$PATH"
unset PYTHONPATH PYTHONHOME
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
PY="$PACKAGE/.analysis-env/bin/python"
trap 'printf "Stage %s failed; inspect %s/logs. Resume with RESUME_BETTY.sh\n" "$STAGE" "$RUN" >&2' ERR
if [[ "$STAGE" = prepare ]]; then
    bash "$PACKAGE/SETUP_COMPUTE.sh" "$PACKAGE"
fi
if [[ "$STAGE" = dispatch ]]; then
    if [[ "${1:-resume}" = resume ]]; then
        exec "$PY" "$PACKAGE/control.py" advance --run "$RUN"
    else
        exec "$PY" "$PACKAGE/control.py" advance --run "$RUN" --check "$1"
    fi
elif [[ "$STAGE" = predict || "$STAGE" = explain || "$STAGE" = modisco || "$STAGE" = reports ]]; then
    TASK="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])])' "$1" "$SLURM_ARRAY_TASK_ID")"
    exec "$PY" "$PACKAGE/workflow.py" "$STAGE" --run "$RUN" --task "$TASK"
elif [[ "$STAGE" = cluster ]]; then
    "$PY" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["methods"]))' "$RUN/CONFIG.json" |
    while IFS= read -r METHOD; do
        "$PY" "$PACKAGE/workflow.py" cluster --run "$RUN" --method "$METHOD"
    done
else
    "$PY" "$PACKAGE/workflow.py" "$STAGE" --run "$RUN"
fi
if [[ "$STAGE" != finish ]]; then
    exec "$PY" "$PACKAGE/control.py" advance --run "$RUN"
fi
