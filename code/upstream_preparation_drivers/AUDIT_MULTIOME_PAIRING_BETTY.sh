#!/usr/bin/env bash
# Run this saved file with Bash from the Betty login terminal.
set -euo pipefail
PAIR_BASE='/vast/projects/saargill/cell-type-specific/nazbuhn'
PAIR_RAW="$PAIR_BASE/scenicplus_inputs/00_raw_geo"
PAIR_TRANSFER="$PAIR_RAW/_transfer_audits/ATAC_upload_20261020T192733_88495"
PAIR_SOURCE="$PAIR_BASE/rbi2_upload_jobs_data/TRBI_ANNOTATED_RNA/runs/import_20261020T171724_1179750"
PAIR_ENV="$PAIR_BASE/rbi2_upload_jobs_data/REASSEMBLED_RNA/envs/rna_merge_py311"
PAIR_ROOT="$PAIR_BASE/scenicplus_inputs/01_pairing_review"
command -v sbatch >/dev/null || { echo 'Run this script from your interactive Betty login terminal.'; exit 1; }
test -f "$PAIR_TRANSFER/TRANSFER_VERIFICATION.json" && test -f "$PAIR_SOURCE/RUN_MANIFEST.json" || { echo 'Completed transfer verification or annotated RNA manifest is missing.'; exit 1; }
test -x "$PAIR_ENV/bin/python" || { echo "Existing RNA environment missing: $PAIR_ENV"; exit 1; }
mkdir -p "$PAIR_ROOT/runs"
PAIR_RUN="$PAIR_ROOT/runs/pairing_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$PAIR_RUN"
if [ ! -f "$PAIR_ROOT/README.md" ]; then
  printf '# Uploaded RNA–ATAC pairing review\n\nRead-only audit of all online-annotated upload cells and verified multiome inputs. Run-specific methods and reports are retained below.\n' > "$PAIR_ROOT/README.md"
fi
cat > "$PAIR_RUN/audit_multiome_pairing.py" <<'PAIR_SOURCE_0'
"""Read-only pairing audit of the frozen online-annotated RNA and GEO multiome inputs."""
import concurrent.futures
import csv
import datetime
import gc
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy import io, sparse
from threadpoolctl import threadpool_limits


def say(*v):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *v, flush=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, value):
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def text(v):
    return v.decode('utf-8') if isinstance(v, bytes) else str(v)


def table(p, d):
    t = p.with_name(p.name + '.tmp')
    compression = dict(method='gzip', mtime=0) if p.name.endswith('.gz') else None
    d.to_csv(t, sep='\t', index=False, compression=compression)
    t.replace(p)


def canonical_names(ids, symbols, refs):
    require(len(set(ids)) == len(ids), 'Duplicate RNA feature IDs')
    names = list(symbols)
    for r in refs:
        ens, target = r['Ensembl_ID'], r['symbol']
        aliases = set(r['globally_unique_aliases'].split('|'))
        for i, (fid, old) in enumerate(zip(ids, symbols)):
            if old == target:
                continue
            matched_id = ens != '-' and re.sub(r'\.\d+$', '', fid) == ens
            matched_alias = old in aliases
            if not (matched_id or matched_alias):
                continue
            require(not (matched_alias and ens != '-' and re.sub(r'\.\d+$', '', fid) != ens),
                    'Alias/Ensembl disagreement: ' + old)
            require(names[i] in (old, target), 'Conflicting canonical gene rename')
            names[i] = target
    return names


def load_combined(raw, spec):
    """Return cells x features, retaining original feature order and raw integer counts."""
    if spec['matrix_format'] == '10x_H5':
        with h5py.File(raw / spec['matrix'], 'r') as f:
            m = f['matrix']
            features = m['features']
            barcodes = [text(x) for x in m['barcodes'][...]]
            ft = pd.DataFrame({k: [text(x) for x in features[v][...]] for k, v in
                               [('id', 'id'), ('symbol', 'name'), ('type', 'feature_type')]})
            shape = tuple(int(x) for x in m['shape'][...])
            # 10x genes x cells CSC becomes cells x genes CSR without a dense copy.
            x = sparse.csr_matrix((m['data'][...], m['indices'][...], m['indptr'][...]), shape=shape[::-1])
    else:
        with gzip.open(raw / spec['features'], 'rt') as f:
            rows = list(csv.reader(f, delimiter='\t'))
        require(all(len(r) >= 3 for r in rows), 'MTX features need ID, symbol, feature type')
        ft = pd.DataFrame([r[:3] for r in rows], columns=['id', 'symbol', 'type'])
        with gzip.open(raw / spec['barcodes'], 'rt') as f:
            barcodes = f.read().splitlines()
        with gzip.open(raw / spec['matrix'], 'rb') as f:
            original = io.mmread(f)
        require(sparse.issparse(original), 'Expected coordinate sparse MTX')
        for start in range(0, len(original.data), 2_000_000):
            v = original.data[start:start + 2_000_000]
            require(np.isfinite(v).all() and (v >= 0).all() and np.equal(v, np.floor(v)).all(),
                    'Noninteger/negative/nonfinite original MTX counts')
        x = original.T.tocsr()
        del original
    require(x.shape == (len(barcodes), len(ft)), 'Combined matrix dimensions disagree')
    require(len(set(barcodes)) == len(barcodes) and all(b and b == b.strip() for b in barcodes),
            'Empty/duplicate/whitespace matrix barcodes')
    x.check_format(full_check=True)
    for start in range(0, len(x.data), 2_000_000):
        v = x.data[start:start + 2_000_000]
        require(np.isfinite(v).all() and (v >= 0).all() and np.equal(v, np.floor(v)).all(),
                'Invalid combined matrix counts')
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    return x, ft, barcodes


def audit_matrix(raw, spec, obs, reference_counts, gene_names, refs):
    say('READ COMBINED MATRIX', spec['source_GSE'], spec['RNA_library'])
    x, ft, barcodes = load_combined(raw, spec)
    bi = pd.Index(barcodes)
    at = bi.get_indexer(obs.original_barcode.astype(str))
    present = at >= 0
    result = pd.DataFrame({'Cell_ID': obs.index.astype(str),
                           'matrix_barcode_present': present,
                           'RNA_counts_exact_match': False,
                           'ATAC_peak_count_sum': np.zeros(len(obs), dtype=np.int64),
                           'ATAC_detected_peaks': np.zeros(len(obs), dtype=np.int64)})
    types = ft['type'].to_numpy()
    gex, peaks = types == 'Gene Expression', types == 'Peaks'
    require(gex.any() and peaks.any(), 'Combined matrix must contain Gene Expression AND Peaks')
    selected = x[at[present]]
    peak_counts = selected[:, peaks]
    result.loc[present, 'ATAC_peak_count_sum'] = np.asarray(peak_counts.sum(axis=1)).ravel().astype(np.int64)
    result.loc[present, 'ATAC_detected_peaks'] = peak_counts.getnnz(axis=1)
    names = canonical_names(ft.loc[gex, 'id'].tolist(), ft.loc[gex, 'symbol'].tolist(), refs)
    gene_index = {g: i for i, g in enumerate(gene_names)}
    targets = np.array([gene_index.get(g, -1) for g in names])
    inside = targets >= 0
    # Combine duplicate symbols exactly as Step 04 did, using its frozen alias table.
    projection = sparse.csr_matrix((np.ones(inside.sum(), dtype=np.int64),
                                   (np.flatnonzero(inside), targets[inside])), shape=(len(names), len(gene_names)))
    g = selected[:, gex].astype(np.int64)
    mapped = (g @ projection).tocsr()
    extra_nonzero = g[:, ~inside].getnnz(axis=1) > 0
    delta = mapped - reference_counts[present]
    delta.eliminate_zeros()
    exact = (delta.getnnz(axis=1) == 0) & ~extra_nonzero
    result.loc[present, 'RNA_counts_exact_match'] = exact
    missing = sorted(set(gene_names) - set(names))
    extra = sorted(set(names) - set(gene_names))
    info = dict(source_GSE=spec['source_GSE'], source_library=spec['RNA_library'],
                matrix=spec['matrix'], original_matrix_cells=len(barcodes),
                original_GEX_features=int(gex.sum()), original_peak_features=int(peaks.sum()),
                uploaded_cells=len(obs), barcode_present_cells=int(present.sum()),
                exact_RNA_count_cells=int(result.RNA_counts_exact_match.sum()),
                cells_with_positive_peak_counts=int(result.ATAC_peak_count_sum.gt(0).sum()),
                absent_RNA_gene_symbols=missing, extra_original_gene_symbols=extra,
                gene_comparison='All original GEX counts: frozen Step04 TCR-gamma renames, duplicate-symbol sums; extra nonzero genes cause mismatch')
    say('MATRIX AUDIT', spec['source_GSE'], spec['RNA_library'],
        'barcodes', int(present.sum()), '/', len(obs), 'exact RNA', int(exact.sum()),
        'positive ATAC', info['cells_with_positive_peak_counts'])
    return result, info


def scan_fragment(path_string, wanted, out_string, signature):
    path, out = Path(path_string), Path(out_string)
    meta = out.with_suffix('.json')
    if out.is_file() and meta.is_file():
        old = json.loads(meta.read_text())
        require(old['signature'] == signature and sha(out) == old['output_SHA256'], 'Fragment checkpoint changed')
        say('REUSE FRAGMENT SCAN', path.name)
        return old
    wanted = set(wanted)
    seen = {}
    n = matched = 0
    say('SCAN FRAGMENTS', path.name)
    with gzip.open(path, 'rt', encoding='ascii') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\r\n').split('\t')
            require(len(fields) >= 5 and bool(fields[0]) and bool(fields[3]), 'Malformed fragment row: ' + path.name)
            start, stop, count = int(fields[1]), int(fields[2]), int(fields[4])
            require(start >= 0 and stop > start and count > 0, 'Invalid fragment coordinate/count: ' + path.name)
            b = fields[3]
            if b in wanted:
                if b not in seen:
                    seen[b] = [0, 0]
                seen[b][0] += 1
                seen[b][1] += count
                matched += 1
            n += 1
            if n % 5_000_000 == 0:
                say('FRAGMENT PROGRESS', path.name, n, 'rows')
    require(n > 0, 'Empty fragment file: ' + path.name)
    rows = [(b, v[0], v[1]) for b, v in sorted(seen.items())]
    table(out, pd.DataFrame(rows, columns=['original_barcode', 'fragment_rows', 'fragment_read_support_sum']))
    record = dict(signature=signature, file=str(path), fragment_rows=n, matched_rows=matched,
                  matched_wanted_barcodes=len(seen), output_SHA256=sha(out))
    dump(meta, record)
    say('FRAGMENT SCAN COMPLETE', path.name, n, 'rows;', len(seen), 'wanted barcodes')
    return record


def expected_fragment(spec, paths):
    """Filename-based candidate only. Known condition conflicts remain explicitly unresolved."""
    name = Path(spec['matrix']).name
    if spec['source_GSE'] == 'GSE230363':
        fragment = name.replace('_filtered_feature_bc_matrix.h5', '_atac_fragments.tsv.gz')
    elif spec['source_GSE'] == 'GSE255042':
        fragment = 'GSM8062933_Naive_P14_atac_fragments.tsv.gz'
    elif spec['source_GSE'] == 'GSE267584':
        suffix = name.split('_', 1)[1].replace('.filtered_matrix.mtx.gz', '.atac_fragments.tsv.gz')
        hits = [p for p in paths if Path(p).name.split('_', 1)[-1] == suffix]
        require(len(hits) == 1, 'Expected exactly one condition-filename candidate: ' + name)
        return hits[0]
    elif spec['source_GSE'] == 'GSE290618':
        fragment = {'gut': 'GSM8818256_atac_fragments_gut.tsv.gz',
                    'tumor': 'GSM8818258_atac_fragments_tumor.tsv.gz'}[spec['RNA_library']]
    else:
        return ''
    hits = [p for p in paths if Path(p).name == fragment]
    require(len(hits) == 1, 'Expected fragment missing/ambiguous: ' + fragment)
    return hits[0]


def confidence_summary(obs):
    records = []
    for key, d in obs.groupby(['source_GSE', 'source_library', 'trbi_submission_scope', 'trbi_level2'],
                               observed=True, dropna=False):
        v = pd.to_numeric(d.trbi_confidence_score, errors='raise')
        record = dict(zip(['source_GSE', 'source_library', 'submission_scope', 'fine_label'], key))
        record.update(cells=len(d), confidence_available=int(v.notna().sum()),
                      confidence_missing=int(v.isna().sum()),
                      confidence_p05=float(v.quantile(.05)) if v.notna().any() else np.nan,
                      confidence_median=float(v.median()) if v.notna().any() else np.nan,
                      confidence_below_0_5=int(v.lt(.5).sum()), confidence_below_0_8=int(v.lt(.8).sum()),
                      confidence_below_0_9=int(v.lt(.9).sum()))
        records.append(record)
    return pd.DataFrame(records)


def run(run):
    cfg = json.loads((run / 'INPUT_PATHS.json').read_text())
    raw, audit, source = (Path(cfg[k]) for k in ['raw', 'transfer_audit', 'annotated_run'])
    frozen = json.loads((run / 'SOURCES_SHA256.json').read_text())
    for name, want in frozen.items():
        require(sha(run / name) == want, 'Saved source/settings changed: ' + name)
    plan = json.loads((audit / 'UPLOAD_PLAN.json').read_text())
    verified = json.loads((audit / 'TRANSFER_VERIFICATION.json').read_text())
    require(verified['status'] == 'PASSED' and verified['uploaded_files_verified'] == len(plan['files']),
            'Expected successful transfer verification')
    files = json.loads((audit / 'FILES_SHA256.json').read_text())
    require(set(files) == {r['destination'] for r in plan['files']}, 'Transfer file list differs from plan')
    prior = json.loads((source / 'RUN_MANIFEST.json').read_text())
    h5ad = source / 'ALL_UPLOADED_CELLS_WITH_TRBI.h5ad'
    say('VERIFY ANNOTATED RNA')
    source_sha = sha(h5ad)
    require(prior['status'] == 'COMPLETE' and source_sha == prior['output_SHA256'][h5ad.name],
            'Annotated RNA differs from its completed import')
    # Revalidate inputs on every resume; extracted fragment hashes are from the successful transfer job.
    input_hashes = {}
    for relative, item in sorted(files.items()):
        p = raw / relative
        require(p.is_file() and p.stat().st_size == item['bytes'], 'Input file missing/wrong size: ' + str(p))
        require(sha(p) == item['SHA256'], 'Input checksum changed: ' + str(p))
        input_hashes[relative] = item['SHA256']
    fragment_paths = [r['destination'] for r in plan['files'] if r['kind'] == 'fragments']
    for item in verified['extracted_fragment_members']:
        p = Path(item['path'])
        relative = p.resolve().relative_to(raw.resolve()).as_posix()
        require(p.stat().st_size == item['bytes'] and sha(p) == item['SHA256'], 'Extracted fragment changed: ' + relative)
        input_hashes[relative] = item['SHA256']
        fragment_paths.append(relative)
    require(len(fragment_paths) == len(set(fragment_paths)), 'Repeated fragment paths')
    sig = dict(annotated_h5ad_SHA256=source_sha, input_SHA256=input_hashes, source_SHA256=frozen,
               input_paths_SHA256=sha(run / 'INPUT_PATHS.json'), transfer_plan_SHA256=sha(audit / 'UPLOAD_PLAN.json'),
               transfer_verification_SHA256=sha(audit / 'TRANSFER_VERIFICATION.json'))
    signature = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
    if (run / 'AUDIT_INPUTS.json').exists():
        require(json.loads((run / 'AUDIT_INPUTS.json').read_text()) == sig, 'Inputs changed; start a new audit run')
    else:
        dump(run / 'AUDIT_INPUTS.json', sig)
    if (run / 'RUN_MANIFEST.json').exists():
        manifest = json.loads((run / 'RUN_MANIFEST.json').read_text())
        require(manifest['signature'] == signature, 'Completed run signature changed')
        for name, want in manifest['output_SHA256'].items():
            require(sha(run / name) == want, 'Completed output changed: ' + name)
        say('ALREADY COMPLETE; verified audit outputs')
        return
    with h5py.File(h5ad, 'r') as f:
        obs, var = read_elem(f['obs']), read_elem(f['var'])
        require(obs.index.astype(str).tolist() == obs.Cell_ID.astype(str).tolist(), 'Cell_ID/index differ')
        require(len(obs) == prior['cells'] and len(var) == prior['genes'] and obs.index.is_unique,
                'Unexpected RNA dimensions/IDs')
        group = f['layers/counts']
        require(group.attrs['encoding-type'] == 'csr_matrix', 'Expected raw counts in CSR')
        counts = sparse.csr_matrix((group['data'][...], group['indices'][...], group['indptr'][...]),
                                    shape=tuple(group.attrs['shape']))
    require(not obs.duplicated(['source_GSE', 'source_library', 'original_barcode']).any(),
            'A barcode is assigned to more than one uploaded cell in a capture library')
    with open(run / 'FROZEN_STEP04_TCRG_ALIASES.tsv') as f:
        refs = list(csv.DictReader(f, delimiter='\t'))
    expected = set(zip(obs.source_GSE.astype(str), obs.source_library.astype(str)))
    require(expected == {(s['source_GSE'], s['RNA_library']) for s in plan['matrices']}, 'Matrix/RNA library sets differ')
    cache = run / 'checkpoints'
    cache.mkdir(exist_ok=True)
    blocks, matrix_info = [], []
    for spec in plan['matrices']:
        key = spec['source_GSE'] + '__' + spec['RNA_library']
        pos = np.flatnonzero(obs.source_GSE.astype(str).eq(spec['source_GSE']) &
                             obs.source_library.astype(str).eq(spec['RNA_library']))
        p, meta = cache / (key + '.tsv.gz'), cache / (key + '.json')
        if p.exists() and meta.exists():
            info = json.loads(meta.read_text())
            require(info['signature'] == signature and sha(p) == info['output_SHA256'], 'Matrix checkpoint changed')
            block = pd.read_csv(p, sep='\t', keep_default_na=False)
            require(block.Cell_ID.tolist() == obs.index[pos].tolist(), 'Checkpoint IDs differ')
            for c in ['matrix_barcode_present', 'RNA_counts_exact_match']:
                require(block[c].isin([True, False]).all(), 'Invalid checkpoint boolean')
            say('REUSE MATRIX AUDIT', key)
        else:
            block, info = audit_matrix(raw, spec, obs.iloc[pos], counts[pos], var.index.astype(str).tolist(), refs)
            table(p, block)
            info.update(signature=signature, output_SHA256=sha(p))
            dump(meta, info)
        blocks.append(block)
        matrix_info.append(info)
        gc.collect()
    del counts
    joined = pd.concat(blocks, ignore_index=True).set_index('Cell_ID').reindex(obs.index)
    require(joined.index.equals(obs.index) and joined.matrix_barcode_present.notna().all(), 'Cell accounting failed')
    workers = int(cfg.get('fragment_workers', 4))
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for relative in sorted(fragment_paths):
            gse = relative.split('/')[0]
            wanted = obs.loc[obs.source_GSE.astype(str).eq(gse), 'original_barcode'].astype(str).unique().tolist()
            out = cache / (Path(relative).name + '.barcodes.tsv.gz')
            tasks.append(pool.submit(scan_fragment, str(raw / relative), wanted, str(out), signature))
        for future in concurrent.futures.as_completed(tasks):
            future.result()
    fragment_tables = {}
    for relative in sorted(fragment_paths):
        p = cache / (Path(relative).name + '.barcodes.tsv.gz')
        d = pd.read_csv(p, sep='\t', keep_default_na=False).set_index('original_barcode')
        require(d.index.is_unique, 'Duplicate fragment summary barcode')
        fragment_tables[relative] = d
    joined['candidate_fragment_file'] = ''
    joined['fragment_assignment_status'] = 'NO_FRAGMENT_FILE_IN_UPLOAD'
    joined['candidate_fragment_rows'] = np.zeros(len(joined), dtype=np.int64)
    joined['candidate_fragment_read_support_sum'] = np.zeros(len(joined), dtype=np.int64)
    overlaps = []
    for spec in plan['matrices']:
        mask = obs.source_GSE.astype(str).eq(spec['source_GSE']) & obs.source_library.astype(str).eq(spec['RNA_library'])
        sub = obs.loc[mask]
        candidates = [p for p in fragment_paths if p.split('/')[0] == spec['source_GSE']]
        expected_file = expected_fragment(spec, candidates)
        conflict = sub.condition_identity_status.astype(str).str.startswith('UNRESOLVED').any()
        for relative in candidates:
            d = fragment_tables[relative].reindex(sub.original_barcode.astype(str)).fillna(0)
            overlaps.append(dict(source_GSE=spec['source_GSE'], source_library=spec['RNA_library'],
                                 fragment_file=relative, cells=len(sub), exact_barcode_overlap=int(d.fragment_rows.gt(0).sum()),
                                 filename_candidate=relative == expected_file,
                                 condition_conflict_retained=bool(conflict),
                                 interpretation='Overlap is diagnostic, not a cross-library assignment rule'))
        if expected_file:
            d = fragment_tables[expected_file].reindex(sub.original_barcode.astype(str)).fillna(0)
            joined.loc[mask, 'candidate_fragment_file'] = expected_file
            joined.loc[mask, 'candidate_fragment_rows'] = d.fragment_rows.to_numpy(dtype=np.int64)
            joined.loc[mask, 'candidate_fragment_read_support_sum'] = d.fragment_read_support_sum.to_numpy(dtype=np.int64)
            status = np.where(d.fragment_rows.gt(0), 'EXACT_BARCODE_IN_EXPECTED_FILE', 'NO_EXACT_BARCODE_IN_EXPECTED_FILE')
            if conflict:
                status = np.full(len(sub), 'UNRESOLVED_CONDITION_CONFLICT_CANDIDATE_ONLY', dtype=object)
            joined.loc[mask, 'fragment_assignment_status'] = status
    joined['matrix_pairing_verified'] = (joined.matrix_barcode_present & joined.RNA_counts_exact_match &
                                          joined.ATAC_peak_count_sum.gt(0))
    joined['matrix_and_fragment_pairing_verified'] = (joined.matrix_pairing_verified &
                                                       joined.fragment_assignment_status.eq('EXACT_BARCODE_IN_EXPECTED_FILE'))
    # Working scope records the submitted CD8 restriction, not an independently confirmed lineage for those jobs.
    joined['working_CD8_scope'] = (obs.trbi_import_status.astype(str).eq('RETURNED_LABEL') &
                                  (obs.trbi_submission_scope.astype(str).eq('CD8_ONLY') |
                                   obs.trbi_level1.astype(str).eq('CD8')))
    cols = ['source_GSE', 'source_library', 'original_barcode', 'upload_job', 'sample_ID', 'HTO_tag', 'HTO_sample',
            'condition_identity_status', 'trbi_import_status', 'trbi_submission_scope', 'trbi_level1', 'trbi_level2',
            'trbi_confidence_score', 'trbi_discovery_score', 'trbi_discovery', 'trbi_QC_filter']
    output = obs[[c for c in cols if c in obs]].join(joined)
    output.insert(0, 'Cell_ID', output.index.astype(str))
    require(len(output) == len(obs) and output.Cell_ID.tolist() == obs.index.tolist(), 'Lost/reordered uploaded cells')
    table(run / 'ALL_CELLS_PAIRING_REVIEW.tsv.gz', output)
    rows = []
    for key, d in output.groupby(['source_GSE', 'source_library'], observed=True):
        rows.append(dict(source_GSE=key[0], source_library=key[1], uploaded_cells=len(d),
                         barcode_present_cells=int(d.matrix_barcode_present.sum()),
                         exact_RNA_count_cells=int(d.RNA_counts_exact_match.sum()),
                         positive_ATAC_cells=int(d.ATAC_peak_count_sum.gt(0).sum()),
                         matrix_pairing_verified_cells=int(d.matrix_pairing_verified.sum()),
                         candidate_fragment_present_cells=int(d.candidate_fragment_rows.gt(0).sum()),
                         matrix_and_fragment_verified_cells=int(d.matrix_and_fragment_pairing_verified.sum()),
                         working_CD8_scope_cells=int(d.working_CD8_scope.sum()),
                         CD8_matrix_and_fragment_verified_cells=int((d.working_CD8_scope & d.matrix_and_fragment_pairing_verified).sum()),
                         fragment_assignment_status=';'.join(sorted(d.fragment_assignment_status.unique()))))
    summary = pd.DataFrame(rows)
    table(run / 'PAIRING_SUMMARY_BY_LIBRARY.tsv', summary)
    table(run / 'FRAGMENT_BARCODE_OVERLAP_BY_LIBRARY.tsv', pd.DataFrame(overlaps))
    table(run / 'LABEL_CONFIDENCE_SUMMARY.tsv', confidence_summary(obs))
    table(run / 'DISCOVERY_VALUES_BY_LIBRARY.tsv', obs.groupby(
        ['source_GSE', 'source_library', 'trbi_import_status', 'trbi_discovery'], observed=True, dropna=False
    ).size().reset_index(name='cells'))
    dump(run / 'MATRIX_SOURCE_REVIEW.json', matrix_info)
    report = dict(status='COMPLETE', cells=len(obs), genes=len(var), libraries=len(summary),
                  matrix_pairing_verified_cells=int(output.matrix_pairing_verified.sum()),
                  matrix_and_fragment_verified_cells=int(output.matrix_and_fragment_pairing_verified.sum()),
                  working_CD8_scope_cells=int(output.working_CD8_scope.sum()),
                  all_cells_retained=True, source_H5AD_and_labels_unchanged=True,
                  no_new_QC_or_confidence_cutoff=True, no_SCENICPLUS_run_yet=True,
                  unresolved_condition_libraries=['GSM8269394', 'GSM8269395', 'GSM8269396'],
                  W8_W9_fragments_not_in_upload=True,
                  fragment_count_note='fragment_rows counts supplied records; read_support_sum sums column five; neither is a new ATAC QC filter')
    dump(run / 'AUDIT_SUMMARY.json', report)
    names = ['ALL_CELLS_PAIRING_REVIEW.tsv.gz', 'PAIRING_SUMMARY_BY_LIBRARY.tsv',
             'FRAGMENT_BARCODE_OVERLAP_BY_LIBRARY.tsv', 'LABEL_CONFIDENCE_SUMMARY.tsv',
             'DISCOVERY_VALUES_BY_LIBRARY.tsv', 'MATRIX_SOURCE_REVIEW.json', 'AUDIT_SUMMARY.json']
    dump(run / 'RUN_MANIFEST.json', dict(status='COMPLETE', signature=signature,
                                        output_SHA256={n: sha(run / n) for n in names}))
    with zipfile.ZipFile(run / 'SHARE_MULTIOME_PAIRING_REVIEW.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names[1:] + ['RUN_MANIFEST.json', 'README_RUN.md', 'AUDIT_INPUTS.json', 'ENVIRONMENT_CHECK.txt']:
            if (run / name).is_file():
                z.write(run / name, name)
    with open(run.parent.parent / 'README.md', 'a') as f:
        f.write('\n- ' + datetime.datetime.now(datetime.timezone.utc).isoformat() +
                ' Pairing review completed; all cells retained; see AUDIT_SUMMARY.json and library table. Run: ' + str(run) + '\n')
    say('COMPLETE', json.dumps(report))
    print(summary.to_string(index=False), flush=True)
    say('SHARE', run / 'SHARE_MULTIOME_PAIRING_REVIEW.zip')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(Path(sys.argv[1]).resolve())
PAIR_SOURCE_0
cat > "$PAIR_RUN/FROZEN_STEP04_TCRG_ALIASES.tsv" <<'PAIR_SOURCE_1'
symbol	NCBI_GeneID	Ensembl_ID	globally_unique_aliases	NCBI_modification_date
Trgc1	107573	ENSMUSG00000076749	Cgamma1|Gm17004|Tcrg-C1	20260910
Trgc2	107574	ENSMUSG00000076752	Gm16603|Tcrg-C2	20260910
Trgc3	107575	ENSMUSG00000091682	Gm17356|Tcrg-C3	20260910
Trgc4	107576	ENSMUSG00000076757	Tcrg-C4	20260910
Trgv1	21632	ENSMUSG00000076755	Gm16602|Tcrg-V1	20260910
Trgv3	21637	ENSMUSG00000076750	Tcrg-V3|vgamma3	20260812
Trgv4	21638	ENSMUSG00000076745	Tcrg-V4|Vg4	20250924
Trgv5	21639	ENSMUSG00000076747	Tcrg-V5	20260812
Trgv6	21640	-	Tcrg-V6	20250924
Trgv7	21641	ENSMUSG00000076744	Tcrg-V7|Vg7	20260910
PAIR_SOURCE_1
cat > "$PAIR_RUN/README_RUN.md" <<'PAIR_SOURCE_2'
# RNA–ATAC pairing and online-label review

This is a read-only audit of the 161,773 uploaded RNA cells, their online T-RBI annotations, and the 23 combined multiome matrices in the verified ATAC upload. It creates reports and checkpoints, not a new expression object or a SCENIC+ analysis. Original H5ADs, matrices, fragments and labels are opened read-only. Every uploaded cell is retained in the audit, including portal-QC exclusions and any missing matches.

## Inputs

- Annotated RNA: `rbi2_upload_jobs_data/TRBI_ANNOTATED_RNA/runs/import_20261020T171724_1179750/ALL_UPLOADED_CELLS_WITH_TRBI.h5ad` under the Betty project base.
- Verified ATAC: `scenicplus_inputs/00_raw_geo`.
- Transfer audit: `_transfer_audits/ATAC_upload_20261020T192733_88495` within the raw ATAC folder. Job 9195514 completed successfully and verified all 97 selected files, then extracted the two known GSE290618 fragment members.
- Existing project RNA environment: `rbi2_upload_jobs_data/REASSEMBLED_RNA/envs/rna_merge_py311`; no installations or package changes.

The audit checks the annotated H5AD against the successful import's SHA256 and checks all transferred/extracted input hashes. The source code and the frozen Step 04 TCR-gamma alias table are also hashed. Resume requires identical inputs and code.

## Matrix pairing test

Each uploaded cell is matched by its exact `original_barcode` within its recorded source library. Bare barcodes are never joined globally; W8/W9 HTO tags remain metadata within their original capture wells. Duplicate cell/barcode assignments within a source library stop the audit.

The full Gene Expression vector in each combined matrix is compared against the original `layers/counts` RNA vector for that barcode. The ten frozen Step 04 TCR-gamma ID/alias rules and duplicate-symbol sums are repeated. There is no capitalization guessing, gene imputation, count rounding, normalization reversal, or use of normalized X for this test. Missing genes are treated as zero only for an exact count comparison; extra original genes with any nonzero expression cause a mismatch. The script records feature-universe differences explicitly.

The paired ATAC vector is the Peaks feature subset of the *same source matrix column*. Its supplied count sum and number of nonzero peak features are reported. A matrix pair is marked verified only when the barcode is present, the complete RNA count vector matches, and the ATAC peak vector has positive counts. This is an identity/availability check, not an ATAC quality threshold or biological singlet validation. No peak matrices from different libraries are combined or assumed to use the same peak universe.

## Fragment coverage test

Four CPU workers stream the 21 available gzip fragment files. Every non-comment record is checked for at least five tab-delimited fields, valid nonnegative start/end coordinates with end > start, and a positive integer fifth-column support count. Reading through EOF also checks the gzip stream. For exact uploaded barcode strings, the script counts supplied fragment records and sums column five. It neither deduplicates fragment records nor calls those support sums unique molecules.

The library-by-fragment overlap table includes every fragment file in that series. This helps reveal missing barcodes or incorrect candidates; overlap across independent libraries is *not* used to infer an assignment. Candidate fragments follow the original paired sample/condition filenames. No suffix removal, reverse complement, or ATAC-to-GEX barcode conversion is guessed. A missing exact barcode is reported for review.

GSE267584 GSM8269394, GSM8269395 and GSM8269396 have previously documented GEO description/filename condition conflicts. Their combined-matrix pairing can be verified independently, but fragment filenames are reported as candidates only. Even complete barcode coverage does not resolve these condition identities. The script does not silently assign their fragments for downstream analysis.

W8/W9 combined peak-count matrices are included; their processed fragment files are absent from this selected upload. The audit records matrix pairing separately from fragment availability. Missing fragments do not imply an unpaired RNA assay.

## Labels and cell scope

The confidence report gives per-library/fine-label counts, missing confidence, 5th percentile, median, and numbers below 0.5, 0.8 and 0.9 for descriptive review. These are sensitivity summaries, not accepted-label cutoffs. Discovery strings remain verbatim; missing/NA discovery results are not converted to FALSE.

`working_CD8_scope` records returned fine labels in the CD8-only submissions plus returned `level1 == CD8` cells in the All-T submission. The CD8-only scope is inherited from the upload restriction; it is not an independently returned broad-lineage assignment. No cell is selected or removed based on this flag.

## Outputs

- `ALL_CELLS_PAIRING_REVIEW.tsv.gz`: original cell IDs and key provenance/online labels plus all matrix, fragment-candidate and scope flags, in original RNA order.
- `PAIRING_SUMMARY_BY_LIBRARY.tsv`: uploaded/matched/exact-count/positive-ATAC/fragment-coverage counts.
- `FRAGMENT_BARCODE_OVERLAP_BY_LIBRARY.tsv`: all within-series fragment candidates, including unexpected overlaps.
- `LABEL_CONFIDENCE_SUMMARY.tsv`, `DISCOVERY_VALUES_BY_LIBRARY.tsv`: descriptive online-label review.
- `MATRIX_SOURCE_REVIEW.json`: source features and exact-comparison details.
- `AUDIT_SUMMARY.json`, `AUDIT_INPUTS.json`, `RUN_MANIFEST.json`: explicit accounting and fingerprints. COMPLETE means the audit finished; consult the match counts to assess the result.
- `SHARE_MULTIOME_PAIRING_REVIEW.zip`: small aggregate reports and methods, excluding the large per-cell table.

Each matrix and fragment scan has an atomic, hashed checkpoint. A stopped run resumes from completed checkpoints using its saved `SUBMIT_OR_RESUME.sh`; a partial matrix or fragment scan restarts only that unit. Inputs are rehashed before checkpoint reuse. The submit script checks the prior job and holds a submission lock; the compute job holds its own run lock. Paths are passed explicitly so Slurm's spool directory is never used as the project path.

## Resources and use

Run the saved bootstrap with Bash in the Betty login terminal. It submits one `genoa-std-mem` CPU job: 8 CPUs, 5632 MiB per CPU (44 GiB), four fragment workers, four-hour limit. Matrices are read sequentially. BLAS/OpenMP threads are limited to one to avoid nested oversubscription. This does not use or require a GPU. Most of the work is reading sparse input matrices and decompressing fragment text; runtime depends on storage and fragment sizes. It prints progress and appends preparation, submission, start, completion or failure entries to `scenicplus_inputs/01_pairing_review/README.md`.

Review the resulting library table before choosing any CD8/label-confidence or ATAC-QC subset and before building SCENIC+ inputs. Full ATAC QC, consensus peaks, topic models and regulatory-network inference remain subsequent work.
PAIR_SOURCE_2
cat > "$PAIR_RUN/SOURCES_SHA256.json" <<'PAIR_SOURCE_3'
{
  "audit_multiome_pairing.py": "55caffb04ef8326de3605ee9ae9d9e84eda37080b15fc7ba5bdcbc007ed0b1f0",
  "FROZEN_STEP04_TCRG_ALIASES.tsv": "69fabafe0e226a22c8e3562c9a2a5bba1f489868ee66f4452307e4be63b24a9e",
  "README_RUN.md": "544362b69440e39d0386ec05ec467a3d942314cb067373405cad046638b728a3"
}
PAIR_SOURCE_3

python3 - "$PAIR_RUN" "$PAIR_BASE" "$PAIR_RAW" "$PAIR_TRANSFER" "$PAIR_SOURCE" "$PAIR_ENV" <<'PAIR_SAVE_PATHS'
import json,pathlib,sys
run,base,raw,transfer_audit,annotated_run,env=sys.argv[1:]
pathlib.Path(run,'INPUT_PATHS.json').write_text(json.dumps(dict(base=base,raw=raw,transfer_audit=transfer_audit,annotated_run=annotated_run,env=env,fragment_workers=4),indent=2)+'\n')
PAIR_SAVE_PATHS
cat > "$PAIR_RUN/RUN_ON_COMPUTE.sh" <<'PAIR_COMPUTE'
#!/usr/bin/env bash
set -euo pipefail
PAIR_RUN="$(cd -- "${1:?Pass the saved run directory}" && pwd)"
PAIR_ROOT="$(dirname -- "$(dirname -- "$PAIR_RUN")")"
test -n "${SLURM_JOB_ID:-}" || { echo 'Use SUBMIT_OR_RESUME.sh to run on a compute node.'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
ulimit -c 0
exec 9>"$PAIR_RUN/.compute.lock"
flock -n 9 || { echo 'Another pairing job holds this run lock.'; exit 1; }
trap 'PAIR_RC=$?; if [ "$PAIR_RC" -ne 0 ]; then printf "\n- %s Pairing review stopped with exit %s; run %s. Completed checkpoints retained; source data and labels preserved.\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_RC" "$PAIR_RUN" >> "$PAIR_ROOT/README.md"; echo "Stopped. Share log tails; resume with $PAIR_RUN/SUBMIT_OR_RESUME.sh"; fi' EXIT
PAIR_BASE="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["base"])' "$PAIR_RUN/INPUT_PATHS.json")"
PAIR_ENV="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["env"])' "$PAIR_RUN/INPUT_PATHS.json")"
PAIR_MAMBA="$PAIR_BASE/immgent_annotation_betty/bin/micromamba"
export MAMBA_ROOT_PREFIX="$PAIR_BASE/rbi2_upload_jobs_data/REASSEMBLED_RNA/mamba"
eval "$("$PAIR_MAMBA" shell hook -s bash)"
micromamba activate "$PAIR_ENV"
unset PYTHONPATH PYTHONHOME
printf '\n- %s Started/resumed pairing audit job %s; run %s. Full RNA count checks and exact fragment-barcode coverage; no cell filtering.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "$PAIR_RUN" >> "$PAIR_ROOT/README.md"
echo "RUN: $PAIR_RUN"
echo "Slurm job: $SLURM_JOB_ID; allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}; fragment workers: 4"
"$PAIR_ENV/bin/python" - "$PAIR_RUN" <<'PAIR_ENV_CHECK'
import importlib.metadata,pathlib,sys
from anndata.experimental import read_elem
expected={'anndata':'0.10.9','h5py':'3.11.0','numpy':'1.26.4','pandas':'2.2.3','scipy':'1.13.1'}
versions={n:importlib.metadata.version(n) for n in expected}
if versions!=expected:raise RuntimeError('Existing RNA environment differs: '+str(versions))
pathlib.Path(sys.argv[1],'ENVIRONMENT_CHECK.txt').write_text(sys.version+'\n'+repr(versions)+'\n')
print('Existing RNA environment checked; no installation or mutation.')
PAIR_ENV_CHECK
"$PAIR_ENV/bin/python" -u "$PAIR_RUN/audit_multiome_pairing.py" "$PAIR_RUN"
printf '%s\n' "$PAIR_RUN" > "$PAIR_ROOT/latest_complete_run.txt"
echo "Share: $PAIR_RUN/SHARE_MULTIOME_PAIRING_REVIEW.zip"
PAIR_COMPUTE
cat > "$PAIR_RUN/SUBMIT_OR_RESUME.sh" <<'PAIR_SUBMIT'
#!/usr/bin/env bash
set -euo pipefail
PAIR_RUN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 8>"$PAIR_RUN/.submit.lock"
flock -n 8 || { echo 'Another submission is in progress.'; exit 1; }
if [ -f "$PAIR_RUN/SLURM_JOB_ID.txt" ]; then
  PAIR_OLD="$(cat "$PAIR_RUN/SLURM_JOB_ID.txt")"
  [[ "$PAIR_OLD" =~ ^[0-9]+$ ]] || { echo 'Unexpected saved job ID.'; exit 1; }
  if PAIR_QUEUE="$(squeue -h -j "$PAIR_OLD" -o '%i' 2>"$PAIR_RUN/last_squeue.err")"; then
    if [ -n "$PAIR_QUEUE" ]; then echo "Already queued/running: $PAIR_QUEUE"; exit 0; fi
  elif ! grep -qi 'invalid job id' "$PAIR_RUN/last_squeue.err"; then
    cat "$PAIR_RUN/last_squeue.err" >&2
    echo 'Could not check the prior job; no duplicate submitted.'; exit 1
  fi
fi
PAIR_OPTS=(--parsable --job-name=multiome_pairing --partition=genoa-std-mem --nodes=1 --ntasks=1
  --cpus-per-task=8 --mem-per-cpu=5632M --time=04:00:00 --chdir="$PAIR_RUN"
  --output="$PAIR_RUN/slurm_%j.out" --error="$PAIR_RUN/slurm_%j.err" --export=ALL)
if [ -n "${IMMGENT_SLURM_ACCOUNT:-}" ]; then PAIR_OPTS+=(--account="$IMMGENT_SLURM_ACCOUNT"); fi
PAIR_JOB="$(sbatch "${PAIR_OPTS[@]}" "$PAIR_RUN/RUN_ON_COMPUTE.sh" "$PAIR_RUN")"
PAIR_JOB="${PAIR_JOB%%;*}"
[[ "$PAIR_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected submission response: $PAIR_JOB"; exit 1; }
printf '%s\n' "$PAIR_JOB" > "$PAIR_RUN/SLURM_JOB_ID.txt"
printf '\n- %s Submitted pairing audit job %s; run %s.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_JOB" "$PAIR_RUN" >> "$(dirname -- "$(dirname -- "$PAIR_RUN")")/README.md"
echo "Submitted pairing audit job: $PAIR_JOB"
echo "Check: squeue -j $PAIR_JOB"
echo "Logs: tail -n 40 \"$PAIR_RUN/slurm_${PAIR_JOB}.out\" \"$PAIR_RUN/slurm_${PAIR_JOB}.err\""
echo "Resume: bash \"$PAIR_RUN/SUBMIT_OR_RESUME.sh\""
echo "Run folder: $PAIR_RUN"
PAIR_SUBMIT
printf '%s\n' "$PAIR_RUN" > "$PAIR_ROOT/latest_started_run.txt"
printf '\n- %s Prepared pairing audit %s. Source RNA: %s. Transfer audit: %s. See README_RUN.md.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_RUN" "$PAIR_SOURCE" "$PAIR_TRANSFER" >> "$PAIR_ROOT/README.md"
bash "$PAIR_RUN/SUBMIT_OR_RESUME.sh"
