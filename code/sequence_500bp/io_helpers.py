import csv
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def dump(path,x):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');temp.replace(path)


def rows(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt') as f:return list(csv.DictReader(f,delimiter='\t'))


def table(path,data):
    data=[clean(x) for x in data]
    fields=list(dict.fromkeys(k for row in data for k in row))
    path=Path(path);temp=path.with_name(path.name+'.partial')
    with temp.open('w') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(data)
    temp.replace(path)


def progress(root,stage,**kw):
    x=dict(stage=stage,time_UTC=datetime.now(timezone.utc).isoformat(),**kw)
    dump(Path(root)/'PROGRESS.json',x)
    print('PROGRESS '+json.dumps(clean(x)),flush=True)


def verify_bundle(root):
    for n,h in read(Path(root)/'BUNDLE_SHA256.json').items():
        need(sha(Path(root)/n)==h,'Frozen bundle changed: '+n)


def checkpoint(root,tag,files,**kw):
    root=Path(root)
    dump(root/(tag+'.json'),dict(sha256={str(Path(p).relative_to(root)):sha(p) for p in files},**kw))


def reuse(root,tag):
    root=Path(root);p=root/(tag+'.json')
    if not p.exists():return False
    for n,h in read(p)['sha256'].items():need(sha(root/n)==h,'Committed output changed: '+n)
    print('REUSE '+tag,flush=True);return True


def receipt_file(folder,receipt,name):
    folder=Path(folder);r=read(folder/receipt)
    hashes=r.get('sha256',r)
    need(name in hashes,'Receipt missing '+name)
    need(sha(folder/name)==hashes[name],'Source file changed: '+str(folder/name))
    return hashes[name]
