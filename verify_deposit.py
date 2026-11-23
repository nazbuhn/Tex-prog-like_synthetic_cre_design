"""Verify deposited file identity using SHA256SUMS.txt (standard library only)."""
from pathlib import Path
import hashlib
import sys

def main():
    root=Path(__file__).resolve().parent
    failed=[]
    checked=0
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1)
        path=root/name
        if not path.is_file():
            failed.append(name+' (missing)')
            continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):
                h.update(b)
        checked+=1
        if h.hexdigest()!=expected:
            failed.append(name+' (checksum differs)')
    if failed:
        print('\n'.join(failed))
        raise SystemExit(1)
    print(f'PASS: {checked} files match their deposited checksums.')
    print('File identity does not establish scientific validity or fresh-environment execution; see README.md for scope limits.')

if __name__=='__main__':
    main()
