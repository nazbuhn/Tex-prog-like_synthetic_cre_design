#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$PACKAGE/.submission.lock"
flock -n 9 || { echo 'Submission already in progress.'; exit 1; }
python3 "$PACKAGE/references.py" fetch
python3 "$PACKAGE/control.py" new
