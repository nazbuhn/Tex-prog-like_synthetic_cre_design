#!/usr/bin/env bash
bash <<'SCPLUS_PREP_INSTALL'
set -euo pipefail
SP_BASE="${SCPLUS_BASE:-/vast/projects/saargill/cell-type-specific/nazbuhn}"
SP_ROOT="$SP_BASE/scenicplus_inputs/05_matrix_preparation"
SP_ANNOTATED="$SP_BASE/immgen/rbi2_upload_jobs_data/TRBI_ANNOTATED_RNA/runs/import_20261020T171724_1179750"
SP_LINKS="$SP_BASE/scenicplus_inputs/04_cell_link_manifest/runs/links_20261020T204930_19268"
SP_RAW="$SP_BASE/scenicplus_inputs/00_raw_geo"
SP_AUDIT="$SP_RAW/_transfer_audits/ATAC_upload_20261020T192733_88495"
command -v sbatch >/dev/null || { echo 'Run this script on the Betty Slurm login node.' >&2; exit 1; }
for SP_PATH in "$SP_ANNOTATED/ALL_UPLOADED_CELLS_WITH_TRBI.h5ad" "$SP_LINKS/RUN_MANIFEST.json" "$SP_AUDIT/UPLOAD_PLAN.json"; do
  test -f "$SP_PATH" || { echo "Required input missing after the move: $SP_PATH" >&2; exit 1; }
done
mkdir -p "$SP_ROOT/runs"
SP_RUN="$SP_ROOT/runs/prep_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$SP_RUN"
if [ ! -f "$SP_ROOT/README.md" ]; then
  printf '# SCENIC+ paired matrix preparation\n\nSource RNA moved under immgen/. Preserve all master annotations and original source data.\n' > "$SP_ROOT/README.md"
fi
cat > "$SP_RUN/prepare_scenicplus_inputs.py" <<'SP_FROZEN_SOURCE_0'
"""Prepare real paired matrix inputs; no network inference or new ATAC QC cutoff."""
import csv
import datetime
import gc
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import zipfile

import anndata as ad
from anndata.experimental import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy import io, sparse
from threadpoolctl import threadpool_limits


def need(ok, message):
    if not ok:
        raise ValueError(message)


def say(*values):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *values, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(p, value):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def table(p, frame):
    tmp = p.with_name(p.name + '.tmp')
    frame.to_csv(tmp, sep='\t', index=False,
                 compression=dict(method='gzip', mtime=0) if p.name.endswith('.gz') else None)
    tmp.replace(p)


def strings(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def boolean(values):
    out = values.astype(str).str.lower()
    need(out.isin(['true', 'false']).all(), 'Malformed saved boolean: ' + str(values.name))
    return out.eq('true')


def relative_file(raw, value):
    p = Path(value)
    need(value and not p.is_absolute() and '..' not in p.parts, 'Unsafe relative input: ' + value)
    return raw / p


def csr_digest(x):
    x = x.tocsr(copy=False)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    h = hashlib.sha256(np.asarray(x.shape, dtype='<i8').tobytes())
    for arr in (x.indptr, x.indices, x.data):
        for start in range(0, len(arr), 1_000_000):
            # Original matrices are integer, so canonicalize integer storage width.
            h.update(np.asarray(arr[start:start + 1_000_000], dtype='<i8').tobytes())
    return h.hexdigest()


def read_sparse(group):
    need(group.attrs['encoding-type'] == 'csr_matrix', 'Expected CSR H5AD matrix')
    return sparse.csr_matrix((group['data'][...], group['indices'][...], group['indptr'][...]),
                             shape=tuple(group.attrs['shape']))


def compare_sparse(group, expected):
    need(tuple(group.attrs['shape']) == expected.shape, 'Output matrix shape changed')
    for name in ['data', 'indices', 'indptr']:
        arr = getattr(expected, name)
        need(len(group[name]) == len(arr), 'Output sparse structure changed: ' + name)
        for start in range(0, len(arr), 1_000_000):
            need(np.array_equal(group[name][start:start + 1_000_000], arr[start:start + 1_000_000]),
                 'Output matrix values changed: ' + name)


def region(feature_id, symbol):
    matches = []
    for s in (feature_id, symbol):
        m = re.fullmatch(r'([^:\s]+):(\d+)-(\d+)', s)
        if m is None:
            m = re.fullmatch(r'([^\s]+)_(\d+)_(\d+)', s)
        if m:
            chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
            need(start >= 0 and end > start, 'Invalid peak interval: ' + s)
            matches.append((chrom, start, end))
    need(matches and len(set(matches)) == 1, 'Missing/ambiguous peak coordinate: ' + feature_id + ' / ' + symbol)
    chrom, start, end = matches[0]
    return chrom, start, end, f'{chrom}:{start}-{end}'


def load_peak_matrix(raw, spec, cells):
    if spec['matrix_format'] == '10x_H5':
        with h5py.File(relative_file(raw, spec['matrix']), 'r') as f:
            m = f['matrix']; ft = m['features']
            barcodes = strings(m['barcodes'][...])
            features = pd.DataFrame({k: strings(ft[v][...]) for k, v in
                                    [('id', 'id'), ('symbol', 'name'), ('type', 'feature_type')]})
            full = sparse.csr_matrix((m['data'][...], m['indices'][...], m['indptr'][...]),
                                      shape=tuple(int(v) for v in m['shape'][...])[::-1])
    else:
        with gzip.open(relative_file(raw, spec['features']), 'rt') as f:
            rows = list(csv.reader(f, delimiter='\t'))
        need(all(len(r) >= 3 for r in rows), 'Missing MTX feature types')
        features = pd.DataFrame([r[:3] for r in rows], columns=['id', 'symbol', 'type'])
        with gzip.open(relative_file(raw, spec['barcodes']), 'rt') as f:
            barcodes = f.read().splitlines()
        with gzip.open(relative_file(raw, spec['matrix']), 'rb') as f:
            original = io.mmread(f)
        need(sparse.issparse(original), 'Expected sparse coordinate Matrix Market input')
        full = original.T.tocsr(); del original
    need(full.shape == (len(barcodes), len(features)), 'Combined matrix dimensions disagree')
    need(pd.Index(barcodes).is_unique, 'Duplicate original matrix barcodes')
    at = pd.Index(barcodes).get_indexer(cells.original_barcode.astype(str))
    need((at >= 0).all(), 'Selected cell missing from original matrix')
    mask = features.type.eq('Peaks').to_numpy()
    need(mask.any(), 'No ATAC peak features')
    x = full[at][:, mask].tocsr(); del full
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    for start in range(0, len(x.data), 1_000_000):
        v = x.data[start:start + 1_000_000]
        need(np.isfinite(v).all() and (v >= 0).all() and np.equal(v, np.floor(v)).all(), 'Invalid ATAC counts')
    need(not len(x.data) or x.data.max() <= np.iinfo(np.int32).max, 'ATAC entry exceeds int32')
    x = x.astype(np.int32)
    ft = features.loc[mask].reset_index(drop=True)
    coords = [region(a, b) for a, b in zip(ft.id, ft.symbol)]
    var = pd.DataFrame(coords, columns=['chrom', 'start', 'end', 'region_id'])
    var['original_feature_id'] = ft.id.to_numpy(); var['original_feature_name'] = ft.symbol.to_numpy()
    need(var.region_id.is_unique, 'Duplicate exact peak coordinates; inspect before combining')
    var.index = pd.Index(var.region_id, name='region_index')
    totals = np.asarray(x.sum(axis=1)).ravel().astype(np.int64)
    detected = np.diff(x.indptr).astype(np.int64)
    need(np.array_equal(totals, cells.ATAC_peak_count_sum.to_numpy(dtype=np.int64)), 'ATAC totals differ from completed original pairing audit')
    need(np.array_equal(detected, cells.ATAC_detected_peaks.to_numpy(dtype=np.int64)), 'Detected peaks differ from completed pairing audit')
    need((totals > 0).all(), 'Previously paired cell has no ATAC signal')
    return x, var, totals, detected


def fragment_columns(cells, raw, ranking, settings):
    d = cells.copy()
    d['scplus_fragment_file'] = ''
    d['scplus_fragment_barcode'] = ''
    d['scplus_fragment_status'] = 'MATRIX_ONLY_NO_FRAGMENT_FILE'
    valid = d.fragment_link_status.eq('VALIDATED_ORIGINAL_FRAGMENT_ID')
    for filename, group in d.loc[valid].groupby('fragment_file', observed=True):
        p = Path(filename)
        # Raw ATAC remains at the same location; reject unrelated paths.
        p.resolve().relative_to(raw.resolve())
        need(p.is_file(), 'Validated fragment file missing: ' + str(p))
        d.loc[group.index, 'scplus_fragment_file'] = str(p)
        d.loc[group.index, 'scplus_fragment_barcode'] = group.fragment_barcode.to_numpy()
    d.loc[valid, 'scplus_fragment_status'] = 'VALIDATED_ORIGINAL_FRAGMENT_ID'
    provisional = d.fragment_link_status.eq('CONDITION_CONFLICT_CANDIDATE_ONLY')
    need(set(d.loc[provisional, 'source_library']) <= set(settings['provisional_condition_pairs']), 'Unexpected provisional library')
    for lib, expected_atac in settings['provisional_condition_pairs'].items():
        mask = provisional & d.source_library.eq(lib)
        if not mask.any():
            continue
        r = ranking.loc[ranking.source_library.eq(lib)]
        need(len(r) == 1 and boolean(r.best_match_is_filename_candidate).all(), 'Reviewed capture support is missing: ' + lib)
        filename = str(r.iloc[0].filename_candidate_fragment)
        need(Path(filename).name.startswith(expected_atac + '_'), 'Unexpected candidate ATAC accession: ' + lib)
        need(d.loc[mask, 'candidate_fragment_file'].eq(filename).all(), 'Candidate fragment differs from capture review')
        need(pd.to_numeric(d.loc[mask, 'candidate_fragment_rows']).gt(0).all(), 'Provisional cell lacks cached fragment records')
        need(d.loc[mask, 'condition_identity_status'].str.startswith('UNRESOLVED').all(), 'Condition uncertainty was lost')
        p = relative_file(raw, filename)
        need(p.is_file(), 'Provisional fragment file missing: ' + str(p))
        d.loc[mask, 'scplus_fragment_file'] = str(p)
        d.loc[mask, 'scplus_fragment_barcode'] = d.loc[mask, 'original_barcode']
        d.loc[mask, 'scplus_fragment_status'] = 'PROVISIONAL_CAPTURE_LINK_CONDITION_UNKNOWN'
    known = valid | provisional | d.fragment_link_status.eq('MATRIX_PAIRED_FRAGMENT_FILE_NOT_IN_UPLOAD')
    need(known.all(), 'Unexpected unclassified fragment-link status')
    need(d.loc[d.scplus_fragment_file.ne(''), 'scplus_fragment_barcode'].ne('').all(), 'Missing fragment barcode')
    need(not d.loc[d.scplus_fragment_file.ne('')].duplicated(['scplus_fragment_file', 'scplus_fragment_barcode']).any(), 'Repeated fragment link across selected cells')
    return d


def write_plot(path, totals, detected, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(np.log10(1 + totals), bins=60)
    ax[0].set(xlabel='log10(1 + supplied peak counts)', ylabel='Cells')
    ax[1].hexbin(np.log10(1 + totals), np.log10(1 + detected), gridsize=55, mincnt=1, bins='log')
    ax[1].set(xlabel='log10(1 + supplied peak counts)', ylabel='log10(1 + detected supplied peaks)')
    fig.suptitle(title + ' — diagnostic; no new ATAC cutoff applied', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def share_reports(run, output_names):
    with zipfile.ZipFile(run / 'SHARE_SCENICPLUS_INPUT_PREPARATION.zip.partial', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['PREPARED_LIBRARY_SUMMARY.tsv', 'EXACT_PEAK_SET_OVERLAP.tsv', 'W8_W9_PEAK_COMPATIBILITY.tsv',
                     'EXISTING_QC_METADATA_COLUMNS.tsv', 'PREPARATION_SUMMARY.json', 'RUN_MANIFEST.json',
                     'MANIFEST_INPUTS.json', 'INPUT_PATHS.json', 'SETTINGS.json', 'README_RUN.md',
                     'SOURCES_SHA256.json', 'ENVIRONMENT_EXPLICIT.txt'] + [n for n in output_names if n.startswith('plots/')]:
            if (run / name).is_file():
                z.write(run / name, name)
    (run / 'SHARE_SCENICPLUS_INPUT_PREPARATION.zip.partial').replace(run / 'SHARE_SCENICPLUS_INPUT_PREPARATION.zip')


def prepare(run):
    cfg = json.loads((run / 'INPUT_PATHS.json').read_text())
    settings = json.loads((run / 'SETTINGS.json').read_text())
    frozen = json.loads((run / 'SOURCES_SHA256.json').read_text())
    for name, expected in frozen.items():
        need(sha(run / name) == expected, 'Frozen preparation source changed: ' + name)
    raw, source, links, transfer = (Path(cfg[k]).resolve() for k in ['raw', 'annotated_run', 'link_run', 'transfer_audit'])
    source_path = source / 'ALL_UPLOADED_CELLS_WITH_TRBI.h5ad'
    say('VERIFY MOVED ANNOTATED RNA', source_path)
    source_sha = sha(source_path)
    need(source_sha == settings['annotated_SHA256'], 'Moved annotated RNA differs from reviewed object')
    sm = json.loads((source / 'RUN_MANIFEST.json').read_text())
    need(sm['status'] == 'COMPLETE' and sm['output_SHA256'][source_path.name] == source_sha, 'Source import manifest differs')
    lm = json.loads((links / 'RUN_MANIFEST.json').read_text())
    need(lm['status'] == 'COMPLETE', 'Cell-link input run is incomplete')
    cell_name = 'ALL_UPLOADED_CELL_RNA_ATAC_LINKS.tsv.gz'
    rank_name = 'GSE267584_BEST_FRAGMENT_DEPTH_MATCH.tsv'
    for name, want in [(cell_name, settings['cell_links_SHA256']), (rank_name, settings['ranking_SHA256'])]:
        need(sha(links / name) == want == lm['output_SHA256'][name], 'Reviewed cell-link input changed: ' + name)
    plan_path = transfer / 'UPLOAD_PLAN.json'
    need(sha(plan_path) == settings['transfer_plan_SHA256'], 'Transfer matrix plan changed')
    plan = json.loads(plan_path.read_text())
    file_meta = json.loads((transfer / 'FILES_SHA256.json').read_text())
    matrix_hashes = {}
    for spec in plan['matrices']:
        for field in ['matrix', 'features', 'barcodes']:
            relative = spec.get(field, '')
            if not relative:
                continue
            p = relative_file(raw, relative)
            need(p.stat().st_size == file_meta[relative]['bytes'], 'Matrix input size changed: ' + relative)
            matrix_hashes[relative] = sha(p)
            need(matrix_hashes[relative] == file_meta[relative]['SHA256'], 'Matrix input checksum changed: ' + relative)
    inputs = dict(annotated_SHA256=source_sha, cell_links_SHA256=settings['cell_links_SHA256'],
                  ranking_SHA256=settings['ranking_SHA256'], matrix_SHA256=matrix_hashes,
                  source_SHA256=frozen, resolved_paths=cfg)
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if (run / 'MANIFEST_INPUTS.json').exists():
        need(json.loads((run / 'MANIFEST_INPUTS.json').read_text()) == inputs, 'Inputs changed; prepare a new run')
    else:
        dump(run / 'MANIFEST_INPUTS.json', inputs)
    if (run / 'RUN_MANIFEST.json').exists():
        previous = json.loads((run / 'RUN_MANIFEST.json').read_text())
        need(previous['signature'] == signature, 'Completed run signature changed')
        for name, want in previous['output_SHA256'].items():
            need(sha(run / name) == want, 'Completed output changed: ' + name)
        share_reports(run, previous['output_SHA256'])
        say('ALREADY COMPLETE; prepared outputs verified')
        return

    cells = pd.read_csv(links / cell_name, sep='\t', keep_default_na=False).set_index('Cell_ID', drop=False)
    ranking = pd.read_csv(links / rank_name, sep='\t', keep_default_na=False)
    need(cells.index.is_unique and len(cells) == settings['expected_cells'], 'Unexpected source cell accounting')
    need(boolean(cells.matrix_pairing_verified).all(), 'Source includes unverified matrix pairs')
    mask = boolean(cells.working_CD8_scope)
    need(int(mask.sum()) == settings['expected_CD8'], 'Working CD8 scope differs from reviewed count')
    accounting = cells[['Cell_ID', 'source_GSE', 'source_library', 'trbi_import_status', 'trbi_level1']].copy()
    accounting['scplus_in_working_CD8_scope'] = mask
    accounting['scope_reason'] = np.where(mask, 'WORKING_CD8_PENDING_ATAC_QC',
        np.where(cells.trbi_import_status.eq('RETURNED_LABEL'), 'OTHER_PORTAL_T_LINEAGE', 'PORTAL_QC_EXCLUDED'))
    table(run / 'ALL_CELL_SCOPE_ACCOUNTING.tsv.gz', accounting)
    selected = fragment_columns(cells.loc[mask], raw, ranking, settings)
    need(len(plan['matrices']) == settings['expected_libraries'], 'Unexpected matrix-library count')
    keys = {(x['source_GSE'], x['RNA_library']) for x in plan['matrices']}
    need(keys == set(zip(selected.source_GSE, selected.source_library)), 'Matrix and candidate source-library sets differ')
    with h5py.File(source_path, 'r') as f:
        source_obs = read_elem(f['obs']); source_var = read_elem(f['var'])
    need(source_obs.index.astype(str).tolist() == cells.index.tolist(), 'Master and cell-link ordering disagree')
    need(len(source_var) == settings['expected_genes'], 'Unexpected gene count')
    for name in ['source_GSE', 'source_library', 'original_barcode'] + [c for c in cells if c.startswith('trbi_') and c in source_obs]:
        a = source_obs[name].astype(str).to_numpy(); b = cells[name].astype(str).to_numpy()
        # CSV may represent missing numeric scores as empty; compare numerics separately.
        if name.endswith('_score'):
            need(np.allclose(pd.to_numeric(source_obs[name], errors='coerce'), pd.to_numeric(cells[name], errors='coerce'), equal_nan=True), 'Source score mismatch: ' + name)
        else:
            need(np.array_equal(a, b), 'Source metadata mismatch: ' + name)

    (run / 'ATAC_BY_LIBRARY').mkdir(exist_ok=True)
    (run / 'plots').mkdir(exist_ok=True)
    summaries, metrics, region_sets, output_names = [], [], {}, ['ALL_CELL_SCOPE_ACCOUNTING.tsv.gz']
    for spec in plan['matrices']:
        key = spec['source_GSE'] + '__' + spec['RNA_library']
        d = selected.loc[selected.source_GSE.eq(spec['source_GSE']) & selected.source_library.eq(spec['RNA_library'])]
        name = 'ATAC_BY_LIBRARY/' + key + '.h5ad'
        checkpoint = run / 'ATAC_BY_LIBRARY' / (key + '.checkpoint.json')
        metric_name = 'ATAC_BY_LIBRARY/' + key + '.CELL_QC.tsv.gz'
        bed_name = 'ATAC_BY_LIBRARY/' + key + '.regions.bed'
        plot_name = 'plots/' + key + '.png'
        files = [name, metric_name, bed_name, plot_name]
        if checkpoint.exists():
            cp = json.loads(checkpoint.read_text())
            need(cp['signature'] == signature, 'ATAC checkpoint inputs changed: ' + key)
            for n, want in cp['output_SHA256'].items():
                need(sha(run / n) == want, 'ATAC checkpoint changed: ' + n)
            with h5py.File(run / name, 'r') as f:
                var = read_elem(f['var']); saved_obs = read_elem(f['obs'])
            need(saved_obs.index.tolist() == d.index.tolist(), 'Checkpoint cell ordering differs')
            q = pd.read_csv(run / metric_name, sep='\t', keep_default_na=False)
            summary = cp['summary']
            say('REUSE PREPARED ATAC', key)
        else:
            say('EXTRACT PAIRED ATAC', key, 'cells', len(d))
            x, var, totals, detected = load_peak_matrix(raw, spec, d)
            obs = source_obs.loc[d.index].copy()
            obs['scplus_ATAC_QC_status'] = 'PENDING_FULL_ATAC_QC'
            obj = ad.AnnData(X=x, obs=obs, var=var)
            obj.uns['count_semantics'] = 'Original supplied ARC peak counts (cut-site counts); not reconstructed fragment records.'
            obj.uns['source_combined_matrix'] = str(relative_file(raw, spec['matrix']))
            obj.uns['peak_coordinates'] = 'Original 0-based half-open intervals; no re-calling, merging or zero padding.'
            partial = run / (name + '.partial')
            obj.write_h5ad(partial, compression='gzip', compression_opts=1)
            with h5py.File(partial, 'r') as f:
                compare_sparse(f['X'], x)
                need(read_elem(f['obs']).index.tolist() == d.index.tolist(), 'Written ATAC cell IDs changed')
                need(read_elem(f['var']).index.tolist() == var.index.tolist(), 'Written ATAC region IDs changed')
            partial.replace(run / name)
            q = d[['Cell_ID', 'source_GSE', 'source_library', 'sample_ID', 'trbi_level2',
                   'scplus_fragment_status', 'condition_identity_status']].copy()
            q['native_peak_count_sum'] = totals; q['native_detected_peaks'] = detected
            mito = var.chrom.str.lower().isin(['chrm', 'mt', 'm']).to_numpy()
            q['fraction_counts_in_supplied_mitochondrial_peaks'] = np.asarray(x[:, mito].sum(axis=1)).ravel() / totals
            q['ATAC_QC_status'] = 'PENDING_FULL_ATAC_QC'
            q['TSS_enrichment_recomputed'] = False
            q['global_FRiP_recomputed'] = False
            q['nucleosome_signal_recomputed'] = False
            table(run / metric_name, q)
            var[['chrom', 'start', 'end']].to_csv(run / bed_name, sep='\t', header=False, index=False)
            write_plot(run / plot_name, totals, detected, key)
            regions = set(var.index)
            fingerprint = hashlib.sha256(('\n'.join(sorted(regions)) + '\n').encode()).hexdigest()
            summary = dict(source_GSE=spec['source_GSE'], source_library=spec['RNA_library'], cells=len(d),
                           native_peaks=len(var), ordered_region_SHA256=hashlib.sha256(('\n'.join(var.index) + '\n').encode()).hexdigest(),
                           peak_set_SHA256=fingerprint, count_SHA256=csr_digest(x),
                           median_peak_counts=float(np.median(totals)), median_detected_peaks=float(np.median(detected)),
                           ATAC_h5ad=name, original_fragment_status=';'.join(sorted(d.scplus_fragment_status.unique())),
                           new_ATAC_QC_filter_applied=False)
            dump(checkpoint, dict(signature=signature, summary=summary, output_SHA256={n: sha(run / n) for n in files}))
            del obj, x; gc.collect()
        region_sets[key] = set(var.index)
        summaries.append(summary); metrics.append(q); output_names += files + [str(checkpoint.relative_to(run))]
    summary = pd.DataFrame(summaries)
    summary['exact_peak_set_group'] = 'PEAKSET_' + summary.peak_set_SHA256.str[:16]
    table(run / 'PREPARED_LIBRARY_SUMMARY.tsv', summary)
    overlaps = []
    keys = list(region_sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = len(region_sets[a] & region_sets[b]); na = len(region_sets[a]); nb = len(region_sets[b])
            overlaps.append(dict(library_A=a, library_B=b, peaks_A=na, peaks_B=nb, exact_shared_peaks=shared,
                                 fraction_A=shared / na, fraction_B=shared / nb,
                                 Jaccard=shared / (na + nb - shared), identical_measured_peak_sets=region_sets[a] == region_sets[b]))
    overlap = pd.DataFrame(overlaps)
    table(run / 'EXACT_PEAK_SET_OVERLAP.tsv', overlap)
    w8, w9 = 'GSE278342__GSM8545120', 'GSE278342__GSM8545124'
    ww = overlap.loc[((overlap.library_A == w8) & (overlap.library_B == w9)) |
                     ((overlap.library_A == w9) & (overlap.library_B == w8))].copy()
    if w8 in region_sets and w9 in region_sets:
        shared = region_sets[w8] & region_sets[w9]
        for key, label in [(w8, 'W8'), (w9, 'W9')]:
            name = summary.loc[summary.source_library.eq(key.split('__')[1]), 'ATAC_h5ad'].iloc[0]
            obj = ad.read_h5ad(run / name)
            total = np.asarray(obj.X.sum(axis=1)).ravel()
            retained = np.asarray(obj.X[:, obj.var_names.isin(shared)].sum(axis=1)).ravel() / total
            ww[label + '_median_count_fraction_in_exact_shared_peaks'] = float(np.median(retained))
            ww[label + '_p05_count_fraction_in_exact_shared_peaks'] = float(np.quantile(retained, .05))
            del obj; gc.collect()
    table(run / 'W8_W9_PEAK_COMPATIBILITY.tsv', ww)
    q = pd.concat(metrics, ignore_index=True).set_index('Cell_ID').reindex(selected.index)
    need(q.index.tolist() == selected.index.tolist() and q.native_peak_count_sum.notna().all(), 'ATAC cell accounting failed')
    table(run / 'ALL_CANDIDATE_MATRIX_QC.tsv.gz', q.reset_index())
    group_for = summary.set_index(['source_GSE', 'source_library']).exact_peak_set_group.to_dict()
    selected['scplus_native_peak_set_group'] = [group_for[(a, b)] for a, b in zip(selected.source_GSE, selected.source_library)]
    selected['scplus_RNA_source_path'] = str(source_path)
    selected['scplus_ATAC_QC_status'] = 'PENDING_FULL_ATAC_QC'
    table(run / 'PREPARED_CELL_LINKS.tsv.gz', selected.reset_index(drop=True))

    say('PREPARE RNA COPY: preserved normalized X and raw/counts snapshot; no normalization repeated')
    with h5py.File(source_path, 'r') as f:
        positions = source_obs.index.get_indexer(selected.index)
        counts = read_sparse(f['layers/counts'])[positions]
        normalized = read_sparse(f['X'])[positions]
    need(counts.shape == normalized.shape == (len(selected), len(source_var)), 'RNA preparation shape mismatch')
    obs = source_obs.loc[selected.index].copy()
    for name in [c for c in selected if c.startswith('scplus_')]:
        obs[name] = selected[name].to_numpy()
    rna = ad.AnnData(X=normalized, obs=obs, var=source_var.copy())
    rna.raw = ad.AnnData(X=counts, obs=obs.copy(), var=source_var.copy())
    rna.uns['scplus_expression_scale'] = 'X: unchanged ln(1+CP10k); raw.X: original integer RNA counts from master layers/counts.'
    rna.uns['scplus_scope'] = 'Working CD8 candidates only; all genes; full ATAC QC still pending. Master preserves all uploaded cells.'
    rna.uns['annotated_master_path'] = str(source_path)
    rna.uns['annotated_master_SHA256'] = source_sha
    partial = run / 'RNA_CD8_CANDIDATES_FOR_SCENICPLUS.h5ad.partial'
    rna.write_h5ad(partial, compression='gzip', compression_opts=1)
    with h5py.File(partial, 'r') as f:
        compare_sparse(f['X'], normalized); compare_sparse(f['raw/X'], counts)
        saved_obs = read_elem(f['obs'])
        need(saved_obs.index.tolist() == obs.index.tolist(), 'Written RNA cell IDs changed')
        need(read_elem(f['raw/var']).index.tolist() == source_var.index.tolist(), 'Raw RNA gene set changed')
        for c in source_obs:
            a = saved_obs[c].astype(str).to_numpy(); b = obs[c].astype(str).to_numpy()
            need(np.array_equal(a, b), 'Original annotation/metadata changed: ' + c)
    partial.replace(run / 'RNA_CD8_CANDIDATES_FOR_SCENICPLUS.h5ad')
    del rna, counts, normalized; gc.collect()
    qc_cols = [c for c in source_obs if re.search(r'tss|nucleos|frip|blacklist|doublet|mito|step11|hto', c, re.I)]
    inventory = pd.DataFrame([dict(column=c, nonmissing_cells=int(source_obs.loc[selected.index, c].notna().sum()),
                                   note='Preserved source field; name alone does not establish ATAC QC meaning') for c in qc_cols],
                             columns=['column', 'nonmissing_cells', 'note'])
    table(run / 'EXISTING_QC_METADATA_COLUMNS.tsv', inventory)
    report = dict(status='COMPLETE', source_cells=len(cells), working_CD8_candidates=len(selected),
                  genes=len(source_var), prepared_libraries=len(summary),
                  source_cells_outside_working_CD8_scope=int((~mask).sum()),
                  fragment_status_counts={str(k): int(v) for k, v in selected.scplus_fragment_status.value_counts().items()},
                  identical_peak_set_groups=int(summary.peak_set_SHA256.nunique()),
                  exact_peaks_shared_by_all_libraries=len(set.intersection(*region_sets.values())),
                  master_data_and_annotations_unchanged=True, raw_counts_and_normalized_X_fully_verified=True,
                  no_new_ATAC_QC_cutoff_applied=True, no_peaks_recalled_or_zero_padded=True,
                  SCENICPLUS_inference_has_run=False,
                  next_step='Use measured peak compatibility and QC availability to configure ATAC QC/topic modeling; do not treat candidates as final QC-passing cells.')
    dump(run / 'PREPARATION_SUMMARY.json', report)
    output_names += ['PREPARED_LIBRARY_SUMMARY.tsv', 'EXACT_PEAK_SET_OVERLAP.tsv', 'W8_W9_PEAK_COMPATIBILITY.tsv',
                     'ALL_CANDIDATE_MATRIX_QC.tsv.gz', 'PREPARED_CELL_LINKS.tsv.gz',
                     'RNA_CD8_CANDIDATES_FOR_SCENICPLUS.h5ad', 'EXISTING_QC_METADATA_COLUMNS.tsv', 'PREPARATION_SUMMARY.json']
    # Sources were opened read-only. Detect any concurrent replacement before declaring completion.
    need(sha(source_path) == source_sha, 'Annotated master changed during preparation')
    dump(run / 'RUN_MANIFEST.json', dict(status='COMPLETE', signature=signature,
         output_SHA256={n: sha(run / n) for n in output_names}))
    share_reports(run, output_names)
    say('COMPLETE', json.dumps(report))
    print(summary[['source_GSE', 'source_library', 'cells', 'native_peaks', 'exact_peak_set_group']].to_string(index=False), flush=True)
    if len(ww):
        print(ww.to_string(index=False), flush=True)
    say('SHARE', run / 'SHARE_SCENICPLUS_INPUT_PREPARATION.zip')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        prepare(Path(sys.argv[1]).resolve())
SP_FROZEN_SOURCE_0
cat > "$SP_RUN/README_RUN.md" <<'SP_FROZEN_SOURCE_1'
# SCENIC+ input preparation on Betty — Step 05, 2026-10-21

## Purpose

Continue from the completed annotated RNA and verified RNA–ATAC links after the user moved RNA/annotation directories under `immgen/`. Prepare actual paired ATAC count objects and a SCENIC+-compatible RNA input copy. No original data, completed run metadata or annotations are modified. No author request or SRA reconstruction is needed for this stage.

Run `PREPARE_SCENICPLUS_INPUTS_BETTY.sh` on the Betty login node. It saves all code/methods/settings and submits the compute job. Analysis and installation run on the compute node. The saved submit script resumes the same run, avoids duplicate active jobs and retains per-library checkpoints.

## Paths

Base: `/vast/projects/saargill/cell-type-specific/nazbuhn` (override only with `SCPLUS_BASE`).

Moved annotated master: `immgen/rbi2_upload_jobs_data/TRBI_ANNOTATED_RNA/runs/import_20261020T171724_1179750/ALL_UPLOADED_CELLS_WITH_TRBI.h5ad`.

Existing ATAC: `scenicplus_inputs/00_raw_geo`; transfer audit: `_transfer_audits/ATAC_upload_20261020T192733_88495`.

Completed cell links: `scenicplus_inputs/04_cell_link_manifest/runs/links_20261020T204930_19268`.

New outputs: `scenicplus_inputs/05_matrix_preparation/runs/prep_<UTC>_<pid>/`.

Old absolute paths recorded in completed manifests are historical provenance. They are not edited or followed blindly. This stage records resolved current paths in its own INPUT_PATHS.json. It verifies the moved H5AD against the exact previously reviewed SHA256, so a move is accepted while an unreviewed replacement is rejected.

## Environment and resources

Use a fresh environment under the unmoved `scenicplus_inputs/envs/scplus_input_prep_py311_v1` rather than relying on relocated Conda prefixes. Use the standalone micromamba binary from the new `immgen/immgent_annotation_betty/bin` location when available; a few explicit software-directory fallbacks are checked. No existing environment is changed. The new environment installs pinned Python 3.11, numpy 1.26.4, scipy 1.13.1, pandas 2.2.3, h5py 3.11.0, anndata 0.10.9, zarr 2.18.7, matplotlib 3.9.4 and threadpoolctl 3.6.0. Save the explicit resolved environment.

Slurm: genoa-std-mem, one task, eight CPUs, 5632M per CPU (44 GiB), eight-hour limit. No GPU. This is a preparation environment, not an assertion that the SCENIC+ inference software has been installed. Matrix exports are sequential to bound memory; numeric worker thread pools are limited to one to avoid oversubscription. The allocation provides memory headroom rather than claiming eight concurrent extraction workers.

## Biological and cell scope

The master retains all 161,773 uploaded cells and 32,245 genes. This stage creates a working CD8 input containing 160,766 candidates: returned CD8-only submissions plus All-T cells assigned CD8. The 818 portal-QC exclusions and 189 other T-lineage assignments remain in the original master and in ALL_CELL_SCOPE_ACCOUNTING.tsv.gz. This is an analysis-scope selection, not a newly fitted lineage classifier or ATAC filter. CD8-only submissions do not independently prove broad lineage. Keep label confidence/discovery, author/sample metadata and saved QC flags.

Carry GSM8269394/395/396 as provisional capture-supported pairs under neutral GSM IDs, with biological conditions unresolved. Verify their expected fragment accessions and saved best-depth-match support before recording provisional paths. Do not modify the old manifest's candidate-only statuses or relabel conditions. These provisional links are explicitly distinct from validated original-fragment links. The matrix pairing is already verified for all selected cells.

W8/W9 remain matrix-only; retain their existing original peak counts and exact paired RNA identities. No new peak counts are inferred from overlaps or from absent measurements.

## Outputs and validation

- `ATAC_BY_LIBRARY/<GSE>__<library>.h5ad`: original selected cells by original measured peaks, sparse integer counts, original RNA/ImmGen metadata. Canonical `chr:start-end` names express the same original intervals; original feature identifiers/names remain in var. Duplicate exact coordinates or ambiguous coordinates stop the run rather than silently merge them.
- Each library has a region BED, per-cell matrix metrics, diagnostic plot and independently hashed completion checkpoint. The write is read back and every ATAC sparse value is compared. Per-cell totals and detected peaks must match the completed pairing audit.
- `PREPARED_LIBRARY_SUMMARY.tsv` and `EXACT_PEAK_SET_OVERLAP.tsv`: ordered/unordered peak fingerprints and exact shared-coordinate coverage. Group equality means identical measured peak sets, not biological batch integration or automatic topic-model selection. Intersecting names does not recount fragments in new regions.
- `W8_W9_PEAK_COMPATIBILITY.tsv`: exact peak overlap and the fractions of each well's observed counts retained in those exact shared peaks. No shared-feature result is assumed in advance and no forced merge occurs.
- `PREPARED_CELL_LINKS.tsv.gz`: selected cells, original reviewed links, current source path and explicit validated/provisional/matrix-only status. Preserve the GSE230363 underscore suffixes in fragment IDs.
- `RNA_CD8_CANDIDATES_FOR_SCENICPLUS.h5ad`: all supplied genes for selected cells; unchanged normalized `.X`; `.raw.X` contains original integer counts from the master's counts layer. This copy does not duplicate the counts in another layer. Every written RNA X/raw value and original metadata column is checked; no normalization is repeated. The master remains unchanged.
- `EXISTING_QC_METADATA_COLUMNS.tsv` reports potential source QC fields without treating their names as validated metrics.
- `SHARE_SCENICPLUS_INPUT_PREPARATION.zip` contains small summaries, methods, provenance and plots, not the large count objects or full per-cell tables.

Available matrix diagnostics include supplied peak count depth, number of detected supplied peaks and counts in supplied mitochondrial peaks. These are not global fragment QC or independent cell-quality approval. TSS enrichment, global FRiP and nucleosome signal are NOT recalculated from peak matrices. No new ATAC depth cutoff, blacklist filter, gene filter, confidence threshold, batch correction or doublet model is applied here. Original RNA QC selections are retained.

## Resume and reproducibility

Do not rerun the installer to resume an interrupted run; use its saved SUBMIT_OR_RESUME.sh. Reverify source H5AD and consumed matrix hashes on resume. Reuse only per-library outputs with matching source signature and checked output hashes. Incomplete RNA-copy writing restarts that copy; it does not rerun completed library exports. A failed checksum stops the run; never remove checksum records to bypass it. Complete runs verify their outputs before returning without rebuilding. The small share ZIP can be regenerated if interrupted after the manifest was written.

Project and run activity logs are separate from the frozen README_RUN.md. Preparation, submission, start, failure and completion append to the stage README and RUN_ACTIVITY.md. No action mutates the frozen README after its source hash is recorded.

## Next scientific stage

Review actual peak compatibility and available QC, then configure fragment-derived QC/common regions for the fragment-supported branch and matrix-based QC/topic inputs for W8/W9. A nonzero ATAC vector is not full QC. Fit and compare topic models, derive topic/DAR candidate regions, perform motif enrichment with matching mouse coordinate resources and run paired SCENIC+ inference. Check reference assembly compatibility before combining studies. Use full global Cell_IDs and preserve original capture/sample identities; upload chunks are not replicates. Do not train regulatory models on reference-atlas expression or arbitrary batch-corrected values.

Sources checked 2026-10-21: [pycisTopic existing-matrix entry](https://pycistopic.readthedocs.io/en/latest/faqs.html), [pycisTopic API](https://pycistopic.readthedocs.io/en/latest/api.html), [SCENIC+ input preparation](https://scenicplus.readthedocs.io/en/latest/human_cerebellum.html). The SCENIC+ run version will be pinned separately when installing its inference dependencies.
SP_FROZEN_SOURCE_1
cat > "$SP_RUN/SETTINGS.json" <<'SP_FROZEN_SOURCE_2'
{
  "expected_cells": 161773,
  "expected_CD8": 160766,
  "expected_genes": 32245,
  "expected_libraries": 23,
  "annotated_SHA256": "420de5756416246a86e2909ac0c49aa621b6648499a0858f053eeec466d313d3",
  "cell_links_SHA256": "3b1fdd749fed8fbf9e6bce3ddf1aaeea23976eefb7911b94e539d19c463295e3",
  "ranking_SHA256": "e257df6d86c5969640ae6a7e3e134631e97f2721fe9a69cb0883c9a7143ea238",
  "transfer_plan_SHA256": "d3dec0e0e0cbd0101c2c570f2b8a135c9f8d1896331447ee9f03b136f9525b2c",
  "provisional_condition_pairs": {
    "GSM8269394": "GSM8269405",
    "GSM8269395": "GSM8269406",
    "GSM8269396": "GSM8269407"
  }
}
SP_FROZEN_SOURCE_2
cat > "$SP_RUN/SOURCES_SHA256.json" <<'SP_FROZEN_SOURCE_3'
{
  "prepare_scenicplus_inputs.py": "0c3fc70a942047aa2aa4bdc6e115b82e8b9362739c1ec9c9c0f381fdffd9d778",
  "README_RUN.md": "6e2a134f099ff4c406b20d727f9bdc5227b05b6f9754ed841a0e5ee3403b0dd3",
  "SETTINGS.json": "97efba037a79f59b9d24ab1159e9ad32793a2df14635c3a734ea53f867991b73"
}
SP_FROZEN_SOURCE_3

python3 - "$SP_RUN" "$SP_BASE" "$SP_RAW" "$SP_ANNOTATED" "$SP_LINKS" "$SP_AUDIT" <<'SP_SAVE_PATHS'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1])
keys=['base','raw','annotated_run','link_run','transfer_audit']
(run/'INPUT_PATHS.json').write_text(json.dumps(dict(zip(keys,sys.argv[2:])),indent=2)+'\n')
SP_SAVE_PATHS
printf '%s\n' "$SP_BASE" > "$SP_RUN/PROJECT_BASE.txt"
cat > "$SP_RUN/RUN_ON_COMPUTE.sh" <<'SP_COMPUTE'
#!/usr/bin/env bash
set -euo pipefail
SP_RUN="$(cd -- "${1:?Pass the saved run folder}" && pwd)"
SP_ROOT="$(dirname -- "$(dirname -- "$SP_RUN")")"
SP_BASE="$(cat "$SP_RUN/PROJECT_BASE.txt")"
test -n "${SLURM_JOB_ID:-}" || { echo 'Use SUBMIT_OR_RESUME.sh to run on a compute node.' >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
SP_ENV="$SP_BASE/scenicplus_inputs/envs/scplus_input_prep_py311_v1"
export MAMBA_ROOT_PREFIX="$SP_BASE/scenicplus_inputs/mamba"
export MPLCONFIGDIR="$SP_ROOT/cache/matplotlib"
mkdir -p "$SP_BASE/scenicplus_inputs/envs" "$MPLCONFIGDIR"
sp_log() {
  printf '\n- %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$SP_ROOT/README.md" >> "$SP_RUN/RUN_ACTIVITY.md"
}
trap 'SP_RC=$?; if [ "$SP_RC" -ne 0 ]; then sp_log "STOPPED job ${SLURM_JOB_ID:-unknown}, exit $SP_RC; run $SP_RUN. Master unchanged; resume with saved SUBMIT_OR_RESUME.sh."; fi' EXIT
exec 9>"$SP_RUN/.compute.lock"
flock -n 9 || { echo 'This run already holds a compute lock.' >&2; exit 1; }
sp_log "START job $SLURM_JOB_ID; run $SP_RUN. Frozen methods: README_RUN.md."
echo "RUN: $SP_RUN"
echo "Job $SLURM_JOB_ID; CPUs allocated ${SLURM_CPUS_PER_TASK:-unknown}; sequential matrix exports."
SP_MAMBA=''
for SP_CANDIDATE in "$SP_BASE/immgen/immgent_annotation_betty/bin/micromamba" "$SP_BASE/software/micromamba/bin/micromamba" "$SP_BASE/software/bin/micromamba"; do
  if [ -x "$SP_CANDIDATE" ]; then SP_MAMBA="$SP_CANDIDATE"; break; fi
done
if [ -z "$SP_MAMBA" ] && command -v micromamba >/dev/null; then SP_MAMBA="$(command -v micromamba)"; fi
test -n "$SP_MAMBA" || { echo 'Standalone micromamba was not found at the moved location or in PATH. Share this log; no environment changed.' >&2; exit 1; }
(
  flock -w 180 8 || { echo 'Preparation environment is in use by another setup.' >&2; exit 1; }
  if [ ! -f "$SP_ENV/PREP_ENV_READY_v1" ]; then
    if [ -f "$SP_ENV/conda-meta/history" ]; then SP_OP=install; else SP_OP=create; fi
    "$SP_MAMBA" "$SP_OP" -y -p "$SP_ENV" --override-channels -c conda-forge --strict-channel-priority \
      'python=3.11' 'numpy=1.26.4' 'scipy=1.13.1' 'pandas=2.2.3' 'h5py=3.11.0' \
      'anndata=0.10.9' 'zarr=2.18.7' 'matplotlib=3.9.4' 'threadpoolctl=3.6.0'
    "$SP_ENV/bin/python" -c 'import anndata,h5py,numpy,pandas,scipy,matplotlib; from anndata.experimental import read_elem; print("Preparation imports passed")'
    touch "$SP_ENV/PREP_ENV_READY_v1"
  fi
) 8>"$SP_BASE/scenicplus_inputs/envs/.input_prep_env.lock"
eval "$("$SP_MAMBA" shell hook -s bash)"
micromamba activate "$SP_ENV"
unset PYTHONPATH PYTHONHOME
"$SP_MAMBA" list -p "$SP_ENV" --explicit > "$SP_RUN/ENVIRONMENT_EXPLICIT.txt"
"$SP_ENV/bin/python" -u "$SP_RUN/prepare_scenicplus_inputs.py" "$SP_RUN"
printf '%s\n' "$SP_RUN" > "$SP_ROOT/latest_complete_run.txt"
sp_log "COMPLETE/VERIFIED job $SLURM_JOB_ID; paired input copies prepared; full ATAC QC and network inference remain subsequent. Run $SP_RUN."
echo "Share: $SP_RUN/SHARE_SCENICPLUS_INPUT_PREPARATION.zip"
SP_COMPUTE
cat > "$SP_RUN/SUBMIT_OR_RESUME.sh" <<'SP_SUBMIT'
#!/usr/bin/env bash
set -euo pipefail
SP_RUN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SP_ROOT="$(dirname -- "$(dirname -- "$SP_RUN")")"
exec 9>"$SP_RUN/.submission.lock"
flock -n 9 || { echo 'A submission for this run is already being prepared.' >&2; exit 1; }
if [ -f "$SP_RUN/SLURM_JOB_ID.txt" ]; then
  SP_OLD="$(cat "$SP_RUN/SLURM_JOB_ID.txt")"
  [[ "$SP_OLD" =~ ^[0-9]+$ ]] || { echo 'Malformed saved Slurm job ID.' >&2; exit 1; }
  if SP_QUEUE="$(squeue -h -j "$SP_OLD" -o '%i' 2>"$SP_RUN/last_squeue.err")"; then
    if [ -n "$SP_QUEUE" ]; then echo "Already queued/running: $SP_OLD"; exit 0; fi
  elif ! grep -qi 'invalid job id' "$SP_RUN/last_squeue.err"; then
    cat "$SP_RUN/last_squeue.err" >&2; exit 1
  fi
fi
SP_OPTS=(--parsable --job-name=SCplus_inputs --partition=genoa-std-mem --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem-per-cpu=5632M --time=08:00:00 --chdir="$SP_RUN" \
  --output="$SP_RUN/slurm_%j.out" --error="$SP_RUN/slurm_%j.err" --export=ALL)
if [ -n "${SCPLUS_SLURM_ACCOUNT:-${IMMGENT_SLURM_ACCOUNT:-}}" ]; then SP_OPTS+=(--account="${SCPLUS_SLURM_ACCOUNT:-$IMMGENT_SLURM_ACCOUNT}"); fi
SP_JOB="$(sbatch "${SP_OPTS[@]}" "$SP_RUN/RUN_ON_COMPUTE.sh" "$SP_RUN")"
SP_JOB="${SP_JOB%%;*}"
[[ "$SP_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected Slurm job ID: $SP_JOB" >&2; exit 1; }
printf '%s\n' "$SP_JOB" > "$SP_RUN/SLURM_JOB_ID.txt"
printf '\n- %s Submitted preparation job %s; run %s.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SP_JOB" "$SP_RUN" >> "$SP_ROOT/README.md"
echo "Submitted SCENIC+ input preparation job: $SP_JOB"
echo "Logs: tail -n 40 \"$SP_RUN/slurm_${SP_JOB}.out\" \"$SP_RUN/slurm_${SP_JOB}.err\""
echo "Resume: bash \"$SP_RUN/SUBMIT_OR_RESUME.sh\""
echo "Run folder: $SP_RUN"
SP_SUBMIT
python3 - "$SP_RUN" <<'SP_FREEZE_RUNNERS'
import hashlib,json,pathlib,sys
run=pathlib.Path(sys.argv[1])
p=run/'SOURCES_SHA256.json'
d=json.loads(p.read_text())
for n in ['RUN_ON_COMPUTE.sh','SUBMIT_OR_RESUME.sh','INPUT_PATHS.json','PROJECT_BASE.txt']:
    d[n]=hashlib.sha256((run/n).read_bytes()).hexdigest()
p.write_text(json.dumps(d,indent=2)+'\n')
SP_FREEZE_RUNNERS
printf '%s\n' "$SP_RUN" > "$SP_ROOT/latest_started_run.txt"
printf '\n- %s Prepared new input-copy run %s. New immgen/ paths recorded; original sources and annotations preserved.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SP_RUN" >> "$SP_ROOT/README.md"
bash "$SP_RUN/SUBMIT_OR_RESUME.sh"
SCPLUS_PREP_INSTALL
