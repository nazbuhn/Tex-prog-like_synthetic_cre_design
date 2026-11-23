#!/usr/bin/env bash
set -euo pipefail
PACKAGE="${1:?package}"; RUN="${2:?run}"; STREME="${3:?streme}"; PY="${4:?python}"
export PATH="$(dirname "$STREME"):$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" "$PACKAGE/run_streme.py" --package "$PACKAGE" --out "$RUN" --streme "$STREME"
