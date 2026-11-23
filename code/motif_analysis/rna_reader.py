# Reused verified sparse RNA reader; original source hash recorded in SOURCE_NOTES.txt.
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import h5py
from scipy import sparse

def need(ok, message):
    if not ok: raise RuntimeError(message)

def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)

def strings(node):
    if isinstance(node, h5py.Dataset):
        return np.asarray(node.asstr()[:], dtype=str)
    if 'categories' in node and 'codes' in node:
        cats = strings(node['categories']); codes = node['codes'][:]
        return np.array([cats[c] if c >= 0 else '' for c in codes])
    raise ValueError('Not a string column')

def axis_names(group):
    key = group.attrs.get('_index', '_index')
    if isinstance(key, bytes): key = key.decode()
    return strings(group[key])

def matrix_blocks(node, nobs, nvars):
    """Yield sparse/dense blocks with their orientation, without materializing the matrix."""
    if isinstance(node, h5py.Dataset):
        need(node.shape == (nobs, nvars), 'Dense RNA shape differs')
        for lo in range(0, nobs, 512):
            yield 'rows', lo, np.asarray(node[lo:min(lo+512, nobs), :])
        return
    kind = node.attrs.get('encoding-type', '')
    if isinstance(kind, bytes): kind = kind.decode()
    need(tuple(node.attrs['shape']) == (nobs, nvars), 'RNA matrix axes differ')
    need(kind in ['csr_matrix', 'csc_matrix'], 'Unsupported RNA encoding: '+str(kind))
    rows = kind == 'csr_matrix'; length = nobs if rows else nvars
    blocksize = 512 if rows else 128
    ptr = node['indptr'][:]
    for lo in range(0, length, blocksize):
        hi = min(lo+blocksize, length); beg, end = int(ptr[lo]), int(ptr[hi])
        data = node['data'][beg:end]; idx = node['indices'][beg:end]
        constructor = sparse.csr_matrix if rows else sparse.csc_matrix
        shape = (hi-lo, nvars) if rows else (nobs, hi-lo)
        block = constructor((data, idx, ptr[lo:hi+1]-beg), shape=shape)
        block.sum_duplicates(); block.eliminate_zeros()
        yield 'rows' if rows else 'columns', lo, block

def extract_markers(path, ids, requested):
    rejected = []
    with h5py.File(path, 'r') as f:
        obs = axis_names(f['obs']); lookup = pd.Index(obs)
        need(lookup.is_unique, 'Duplicate RNA cell identifiers')
        positions = lookup.get_indexer(ids)
        need((positions >= 0).all(), 'RNA does not contain every exact source-qualified cell ID')
        for layer, varpath in [('layers/counts', 'var'), ('raw/X', 'raw/var'), ('X', 'var')]:
            if layer not in f or varpath not in f: continue
            var = f[varpath]; nvars = len(axis_names(var)); choices = []
            for field in var:
                try: names = strings(var[field])
                except (ValueError, TypeError, UnicodeDecodeError): continue
                if names.shape != (nvars,): continue
                hits = len({x.lower() for x in names} & {x.lower() for x in requested})
                choices.append((hits, field, names))
            if not choices: continue
            choices.sort(key=lambda x: (-x[0], x[1]))
            hits, field, names = choices[0]
            need(hits >= min(10, len(requested)), 'Cannot identify RNA gene symbols')
            available = [g for g in requested if g.lower() in {s.lower() for s in names}]
            wanted = {g.lower():j for j,g in enumerate(available)}
            src = [i for i,s in enumerate(names) if str(s).lower() in wanted]
            dst = [wanted[str(names[i]).lower()] for i in src]
            feature_map = sparse.csr_matrix((np.ones(len(src)), (src,dst)), shape=(nvars,len(available)))
            values = np.zeros((len(obs), len(available)), dtype=np.float64)
            totals = np.zeros(len(obs), dtype=np.float64)
            reason = None
            for orientation, lo, block in matrix_blocks(f[layer], len(obs), nvars):
                data = block.data if sparse.issparse(block) else block.ravel()
                if (not np.isfinite(data).all() or (data < 0).any() or
                        not np.allclose(data, np.rint(data), rtol=0, atol=1e-6)):
                    reason = 'not finite nonnegative integer counts'; break
                if orientation == 'rows':
                    hi = lo+block.shape[0]
                    totals[lo:hi] = np.asarray(block.sum(axis=1)).ravel()
                    sub = block @ feature_map
                    values[lo:hi] = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
                else:
                    totals += np.asarray(block.sum(axis=1)).ravel()
                    sub = block @ feature_map[lo:lo+block.shape[1]]
                    values += sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
                if lo % (512*20) == 0: log(f'RNA {layer}: {orientation} through {lo}')
            if reason:
                rejected.append(dict(matrix=layer, reason=reason)); log('Rejected '+layer); continue
            need((totals[positions] > 0).all(), 'Matched cells have empty RNA libraries')
            return values[positions], totals[positions], available, dict(
                file=str(path), matrix=layer, symbol_field=field, gene_axis=varpath,
                all_selected_source_values_integer_verified=True, rejected_sources=rejected)
    raise RuntimeError('No verified raw RNA counts: '+str(rejected))
