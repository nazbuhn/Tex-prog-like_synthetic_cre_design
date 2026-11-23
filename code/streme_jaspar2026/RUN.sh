#!/usr/bin/env bash
set -euo pipefail
RUN="$1"
PACKAGE="$2"
export OMP_NUM_THREADS=1
exec python3 "$PACKAGE/reannotate.py" --run "$RUN"
