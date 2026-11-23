#!/usr/bin/env bash
set -euo pipefail
PANELS_RUN="${1:?run folder required}"
PANELS_STAGE="${2:?stage required}"
PANELS_ENV='/vast/projects/saargill/cell-type-specific/nazbuhn/immgen/crested_cluster6/envs/crested_fresh_py312_torch_v1'
export KERAS_BACKEND=torch WANDB_MODE=disabled MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONPATH PYTHONHOME
mkdir -p "$PANELS_RUN/matplotlib_cache"
export MPLCONFIGDIR="$PANELS_RUN/matplotlib_cache"
if [ "$PANELS_STAGE" = train ]; then
    PANELS_PANEL="$("$PANELS_ENV/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["panels"][int(sys.argv[2])])' "$PANELS_RUN/SETTINGS.json" "$SLURM_ARRAY_TASK_ID")"
    mkdir -p "$PANELS_RUN/models/$PANELS_PANEL"
    exec 7>"$PANELS_RUN/models/$PANELS_PANEL/execution.lock"
else
    exec 7>"$PANELS_RUN/${PANELS_STAGE}_execution.lock"
fi
flock -n 7 || exit 1
finish_panels() {
    PANELS_CODE=$?
    trap - EXIT
    if [ "$PANELS_CODE" -ne 0 ]; then
        "$PANELS_ENV/bin/python" "$PANELS_RUN/collect.py" "$PANELS_RUN" || true
    fi
    exit "$PANELS_CODE"
}
trap finish_panels EXIT
echo "RUN $PANELS_RUN; STAGE $PANELS_STAGE; JOB ${SLURM_JOB_ID:-unknown}; PANEL ${PANELS_PANEL:-none}"
case "$PANELS_STAGE" in
    prepare)
        export CUDA_VISIBLE_DEVICES=''
        "$PANELS_ENV/bin/python" -u "$PANELS_RUN/prepare.py" "$PANELS_RUN"
        ;;
    train)
        "$PANELS_ENV/bin/python" -u "$PANELS_RUN/train.py" "$PANELS_RUN" "$PANELS_PANEL"
        ;;
    collect)
        "$PANELS_ENV/bin/python" -u "$PANELS_RUN/collect.py" "$PANELS_RUN"
        ;;
    *) exit 2 ;;
esac
