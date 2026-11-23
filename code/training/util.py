import csv
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def clean(obj):
    import math
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    if hasattr(obj, 'tolist'):
        return clean(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def verify(folder, receipt='COMPLETE.json'):
    folder = Path(folder)
    r = read(folder / receipt)
    hashes = r.get('sha256', r)
    for name, want in hashes.items():
        need(sha(folder / name) == want, 'Changed file: ' + str(folder / name))
    return r


def checkpoint(folder, files, **fields):
    folder = Path(folder)
    dump(folder / 'COMPLETE.json', dict(sha256={str(n): sha(folder / n) for n in files}, **fields))


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def table(path, data):
    data = [clean(x) for x in data]
    opener = gzip.open if str(path).endswith('.gz') else open
    fields = list(dict.fromkeys(k for row in data for k in row))
    with opener(path, 'wt') as f:
        writer = csv.DictWriter(f, fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)


def progress(folder, stage, **kw):
    obj = dict(stage=stage, time_UTC=datetime.now(timezone.utc).isoformat(), **kw)
    dump(Path(folder) / 'PROGRESS.json', obj)
    print('PROGRESS ' + json.dumps(clean(obj)), flush=True)
