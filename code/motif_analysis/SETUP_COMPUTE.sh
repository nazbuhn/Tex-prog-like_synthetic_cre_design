#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$1"
BASE_PY='/vast/projects/saargill/cell-type-specific/nazbuhn/immgen/crested_cluster6/envs/crested_fresh_py312_torch_v1/bin/python'
[[ -x "$BASE_PY" ]] || { echo 'Frozen CREsted environment missing.'; exit 1; }
if [[ ! -x "$PACKAGE/.analysis-env/bin/python" ]]; then
    "$BASE_PY" -m venv --system-site-packages "$PACKAGE/.analysis-env"
fi
PY="$PACKAGE/.analysis-env/bin/python"
# A new venv does not automatically inherit another venv's site-packages.
# Read the frozen environment through a .pth file; never modify that directory.
"$PY" - "$BASE_PY" <<'PY'
from pathlib import Path
import subprocess,sys,sysconfig
paths=subprocess.check_output([sys.argv[1],'-c',
    'import sysconfig; print(sysconfig.get_path("purelib")); print(sysconfig.get_path("platlib"))'],text=True).splitlines()
Path(sysconfig.get_path('purelib'),'frozen_training_environment.pth').write_text('\n'.join(dict.fromkeys(paths))+'\n')
PY
# Reuse prior analysis extras read-only when available.
PRIOR='/vast/projects/saargill/cell-type-specific/nazbuhn/immgen/C1_CRESTED_ANALYSIS_BETTY_V1/.analysis-env/bin/python'
if [[ -x "$PRIOR" ]]; then
    "$PY" - "$PRIOR" <<'PY'
from pathlib import Path
import subprocess,sys,sysconfig
p=subprocess.check_output([sys.argv[1],'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip()
Path(sysconfig.get_path('purelib'),'previous_analysis_extras.pth').write_text(p+'\n')
PY
fi
# Constraints protect the trained model stack. Only this new environment can be
# changed; no installation writes to the original training/design environments.
"$BASE_PY" - "$PACKAGE/MODEL_CONSTRAINTS.txt" <<'PY'
import importlib.metadata as m, sys
from pathlib import Path
pins=[]
for n in ['crested','torch','keras','numpy','scipy','h5py','anndata','scanpy']:
    try:
        pins.append(n+'=='+m.version(n))
    except m.PackageNotFoundError:
        if n not in {'anndata', 'scanpy'}:
            raise
        print('Analysis dependency absent from frozen environment: '+n)
Path(sys.argv[1]).write_text('\n'.join(pins)+'\n')
PY
if ! "$PY" -c 'import crested, modiscolite, memelite, logomaker, scanpy, anndata; import crested.tl.modisco; import importlib.metadata as m; assert m.version("modisco-lite")=="3.0.0"'; then
    "$PY" -m pip install --constraint "$PACKAGE/MODEL_CONSTRAINTS.txt" \
        'crested[motif]' 'modisco-lite==3.0.0' logomaker 'scanpy==1.10.3'
fi
"$PY" -m pip freeze > "$PACKAGE/ANALYSIS_ENVIRONMENT.txt"
"$PY" - "$BASE_PY" <<'PY'
import importlib.metadata as m, json, subprocess, sys
names=['crested','torch','keras','numpy','scipy','h5py']
code='import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in '+repr(names)+'}))'
old=json.loads(subprocess.check_output([sys.argv[1],'-c',code],text=True))
assert old=={n:m.version(n) for n in names}, 'Model environment changed; stop.'
import crested.tl.modisco
PY
