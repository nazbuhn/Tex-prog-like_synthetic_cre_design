#!/usr/bin/env bash
bash <<'SCPLUS_COMMON_INSTALL'
set -euo pipefail
SP_BASE="${SCPLUS_BASE:-/vast/projects/saargill/cell-type-specific/nazbuhn}"
SP_ROOT="$SP_BASE/scenicplus_inputs/07_common_matrix"
SP_PREV="$SP_BASE/scenicplus_inputs/06_consensus_qc/runs/consensus_20261021T170938_114096"
SP_MAP="$SP_BASE/immgen/gse267584_reference_mapping/runs/map_20261021T161413_3496882/ALL_CELLS_GSE267584_REFERENCE_MAPPING.tsv.gz"
command -v sbatch >/dev/null || { echo 'Run this installer on the Betty Slurm login terminal.' >&2; exit 1; }
test -f "$SP_PREV/RUN_MANIFEST.json" && test -f "$SP_MAP" || { echo 'Completed consensus/QC or RNA reference mapping is missing.' >&2; exit 1; }
SP_QC_ENV="$SP_BASE/scenicplus_inputs/envs/scplus_fragment_qc_py311_v1"
SP_RNA_ENV="$SP_BASE/scenicplus_inputs/envs/scplus_input_prep_py311_v1"
test -x "$SP_QC_ENV/bin/python" && test -x "$SP_RNA_ENV/bin/python" || { echo 'An existing QC/RNA environment is missing; no new environment was installed.' >&2; exit 1; }
mkdir -p "$SP_ROOT/runs"
SP_RUN="$SP_ROOT/runs/common_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$SP_RUN"
if [ ! -f "$SP_ROOT/README.md" ]; then
  printf '# SCENIC+ common-region matrices after ATAC QC\n\nW8/W9 excluded. Original RNA, labels, fragments and completed peak calling are read-only.\n' > "$SP_ROOT/README.md"
fi
cat > "$SP_RUN/scplus_common_matrix.py" <<'SP_COMMON_FROZEN_SOURCE_0'
"""Betty Step 07: select ATAC-quality cells, count shared regions, preserve RNA labels."""
import argparse
import concurrent.futures as cf
import datetime
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse


def need(ok, message):
    if not ok:
        raise ValueError(message)


def say(*v):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *v, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n'); q.replace(p)


def table(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    d.to_csv(q, sep='\t', index=False,
             compression={'method': 'gzip', 'mtime': 0} if p.name.endswith('.gz') else None)
    q.replace(p)


def load_table(p):
    return pd.read_csv(p, sep='\t', keep_default_na=False, dtype={'Cell_ID': str})


def save_sparse(p, x):
    q = p.with_name(p.name + '.tmp.npz')
    sparse.save_npz(q, x, compressed=False); q.replace(p)


def checkpoint(run, marker, signature, files, info=None):
    dump(marker, dict(signature=signature,
        output_SHA256={str(p.relative_to(run)): sha(p) for p in files}, info=info or {}))


def reuse(run, marker, signature):
    if not marker.exists():
        return None
    d = json.loads(marker.read_text())
    need(d['signature'] == signature, 'Checkpoint settings changed: ' + str(marker))
    for n, want in d['output_SHA256'].items():
        need((run/n).is_file() and sha(run/n) == want, 'Checkpoint changed: ' + n)
    return d['info']


def frozen(run):
    hashes = json.loads((run/'SOURCES_SHA256.json').read_text())
    for n, want in hashes.items():
        need(sha(run/n) == want, 'Frozen source/settings changed: ' + n)
    sig = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return json.loads((run/'INPUT_PATHS.json').read_text()), json.loads((run/'SETTINGS.json').read_text()), sig


def select_cells(d, settings):
    d = d.copy()
    f = pd.to_numeric(d.unique_fragments_count, errors='coerce')
    t = pd.to_numeric(d.tss_enrichment, errors='coerce')
    p = pd.to_numeric(d.fraction_of_fragments_in_peaks, errors='coerce')
    measured = np.isfinite(f) & np.isfinite(t) & np.isfinite(p)
    d['scplus_pass_primary_ATAC_QC'] = (measured & (f >= settings['min_nuclear_fragments']) &
        (t >= settings['min_TSS']) & (p >= settings['min_FRIP'])).astype(bool)
    d['scplus_pass_stricter_depth_QC'] = (d.scplus_pass_primary_ATAC_QC & (f >= 3000)).astype(bool)
    reasons = []
    for valid, fv, tv, pv in zip(measured, f, t, p):
        fail = []
        if not valid: fail.append('NONFINITE_OR_MISSING_QC')
        if fv < settings['min_nuclear_fragments']: fail.append('LOW_NUCLEAR_FRAGMENTS')
        if tv < settings['min_TSS']: fail.append('LOW_TSS')
        if pv < settings['min_FRIP']: fail.append('LOW_FRIP')
        reasons.append(';'.join(fail) or 'PASS')
    d['scplus_ATAC_QC_reason'] = reasons
    d['final_ATAC_QC_status'] = np.where(d.scplus_pass_primary_ATAC_QC, 'PASS_PRIMARY', 'FAIL_PRIMARY')
    return d


def selection(run, paths, settings, signature):
    old = reuse(run, run/'SELECTION_COMPLETE.json', signature)
    if old is not None:
        say('REUSE QC SELECTION', old); return
    prev = Path(paths['consensus_run'])
    manifest = json.loads((prev/'RUN_MANIFEST.json').read_text())
    need(manifest['status'] == 'COMPLETE', 'Consensus/QC run is not complete')
    for name, want in settings['confirmed_inputs_SHA256'].items():
        need(manifest['output_SHA256'][name] == want and sha(prev/name) == want,
             'Completed consensus input changed: ' + name)
    d = load_table(prev/'ALL_CANDIDATE_ATAC_QC.tsv.gz')
    need(len(d) == settings['expected_candidates'] and d.Cell_ID.is_unique, 'Candidate scope/IDs changed')
    need(d.scplus_capture.nunique() == settings['expected_captures'], 'Capture scope changed')
    need(not d.source_GSE.eq('GSE278342').any(), 'W8/W9 unexpectedly included')
    d = select_cells(d, settings)
    # Keep both annotations as separate fields. No terminal phenotype is inferred here.
    mp = Path(paths['reference_mapping'])
    need(mp.is_file() and sha(mp) == settings['reference_mapping_SHA256'], 'Completed reference-mapping sidecar missing/changed')
    m = load_table(mp)
    need(m.Cell_ID.is_unique and set(d.Cell_ID).issubset(m.Cell_ID), 'Reference mapping has missing/duplicate IDs')
    cols = ['Cell_ID'] + [c for c in m.columns if c.startswith('gse267584_')]
    need('gse267584_best_state' in cols, 'Reference state column missing')
    need(not (set(cols) - {'Cell_ID'}).intersection(d.columns), 'Reference columns already exist; review source')
    d = d.merge(m[cols], on='Cell_ID', how='left', sort=False, validate='one_to_one')
    table(run/'ALL_CANDIDATE_QC_DECISIONS.tsv.gz', d)
    kept = d.loc[d.scplus_pass_primary_ATAC_QC].sort_values(['scplus_capture', 'Cell_ID']).reset_index(drop=True)
    need(len(kept) > 0 and kept.scplus_capture.nunique() == settings['expected_captures'], 'QC removed an entire capture; review before counting')
    table(run/'RETAINED_CELLS.tsv.gz', kept)
    table(run/'EXCLUDED_CELLS.tsv.gz', d.loc[~d.scplus_pass_primary_ATAC_QC])
    for group, name in [(['scplus_capture'], 'QC_RETENTION_BY_CAPTURE.tsv'),
                        (['scplus_capture', 'trbi_level2'], 'QC_RETENTION_BY_IMMGEN_STATE.tsv'),
                        (['scplus_capture', 'gse267584_best_state'], 'QC_RETENTION_BY_REFERENCE_STATE.tsv')]:
        x = d.copy()
        for c in ['scplus_pass_primary_ATAC_QC', 'scplus_pass_stricter_depth_QC']: x[c] = x[c].astype(int)
        x = x.groupby(group, dropna=False, observed=True).agg(candidates=('Cell_ID', 'size'),
            primary=('scplus_pass_primary_ATAC_QC', 'sum'), stricter_depth=('scplus_pass_stricter_depth_QC', 'sum')).reset_index()
        table(run/name, x)
    info = dict(candidates=len(d), retained=len(kept), excluded=len(d)-len(kept), captures=kept.scplus_capture.nunique(),
                stricter_depth=int(kept.scplus_pass_stricter_depth_QC.sum()))
    dump(run/'SELECTION_SUMMARY.json', info)
    files = [run/n for n in ['ALL_CANDIDATE_QC_DECISIONS.tsv.gz', 'RETAINED_CELLS.tsv.gz', 'EXCLUDED_CELLS.tsv.gz',
        'QC_RETENTION_BY_CAPTURE.tsv', 'QC_RETENTION_BY_IMMGEN_STATE.tsv', 'QC_RETENTION_BY_REFERENCE_STATE.tsv', 'SELECTION_SUMMARY.json']]
    checkpoint(run, run/'SELECTION_COMPLETE.json', signature, files, info)
    say('QC SELECTION COMPLETE', info)


def region_index(bed):
    regions = pd.read_csv(bed, sep='\t', names=['Chromosome', 'Start', 'End'], dtype={'Chromosome': str})
    need(len(regions) > 0 and not regions.duplicated().any(), 'Empty/repeated consensus regions')
    need(((regions.Start >= 0) & (regions.End > regions.Start)).all(), 'Invalid BED coordinates')
    index = {}
    for chrom, frame in regions.groupby('Chromosome', sort=False):
        starts, ends = frame.Start.to_numpy(np.int64), frame.End.to_numpy(np.int64)
        need(np.all(starts[1:] >= ends[:-1]), 'Counter requires sorted nonoverlapping consensus: ' + chrom)
        index[chrom] = (starts, ends, frame.index.to_numpy(np.int64))
    return regions, index


def overlaps_in_chunk(d, bc_index, index, shape):
    """Same interval-overlap, one-unique-molecule semantics as pycisTopic's PyRanges join.

    Consensus intervals do not overlap. A long fragment can overlap multiple regions;
    it contributes one to EACH intersected region, irrespective of read-support score.
    Coordinates are zero-based half-open; touching endpoints are not overlaps.
    """
    cells = bc_index.get_indexer(d.Barcode)
    selected = cells >= 0
    d = d.loc[selected].copy(); cells = cells[selected]
    row_parts, col_parts = [], []
    in_peak = np.zeros(shape[1], dtype=np.int64)
    nuclear = np.bincount(cells, minlength=shape[1]).astype(np.int64)
    for chrom, positions in d.groupby('Chromosome', sort=False).indices.items():
        if chrom not in index:
            continue  # A primary chromosome can legitimately contain no consensus regions.
        starts, ends, ids = index[chrom]
        positions = np.asarray(positions)
        a = d.Start.to_numpy()[positions]; b = d.End.to_numpy()[positions]
        need(np.all((a >= 0) & (b > a)), 'Invalid fragment interval')
        lo = np.searchsorted(ends, a, side='right')
        hi = np.searchsorted(starts, b, side='left')
        n = hi - lo
        need((n >= 0).all(), 'Invalid overlap search')
        hit = n > 0
        in_peak += np.bincount(cells[positions[hit]], minlength=shape[1])
        if not hit.any(): continue
        lo, n = lo[hit], n[hit]
        offsets = np.repeat(np.cumsum(n) - n, n)
        r = np.repeat(lo, n) + np.arange(int(n.sum())) - offsets
        row_parts.append(ids[r]); col_parts.append(np.repeat(cells[positions[hit]], n))
    if not row_parts:
        x = sparse.csr_matrix(shape, dtype=np.int32)
    else:
        rr, cc = np.concatenate(row_parts), np.concatenate(col_parts)
        x = sparse.coo_matrix((np.ones(len(rr), dtype=np.int32), (rr, cc)), shape=shape).tocsr()
    return x, nuclear, in_peak


def count_capture(run, capture, signature):
    paths, settings, actual_sig = frozen(run); need(actual_sig == signature, 'Worker signature changed')
    out = run/'counts_by_capture'/capture; out.mkdir(parents=True, exist_ok=True)
    if reuse(run, out/'COMPLETE.json', signature) is not None:
        say('REUSE SHARED-REGION COUNTS', capture); return
    prev = Path(paths['consensus_run'])
    fp = prev/'captures'/capture/'candidate_nuclear.fragments.tsv.gz'
    receipt = json.loads((prev/'captures'/capture/'FRAGMENTS_COMPLETE.json').read_text())
    rel = str(fp.relative_to(prev))
    need(sha(fp) == receipt['output_SHA256'][rel], 'Prepared fragment checkpoint changed: ' + capture)
    kept = load_table(run/'RETAINED_CELLS.tsv.gz'); kept = kept.loc[kept.scplus_capture.eq(capture)].reset_index(drop=True)
    need(kept.scplus_fragment_barcode.is_unique, 'Duplicate fragment barcodes within capture')
    bc = pd.Index(kept.scplus_fragment_barcode.astype(str))
    regions, index = region_index(prev/'CONSENSUS_REGIONS.mm10.bed')
    shape = (len(regions), len(kept)); x = sparse.csr_matrix(shape, dtype=np.int32)
    nuclear = np.zeros(len(kept), np.int64); in_peak = nuclear.copy(); rows = 0
    reader = pd.read_csv(fp, sep='\t', header=None, names=['Chromosome', 'Start', 'End', 'Barcode', 'ReadSupport'],
        dtype={'Chromosome': str, 'Start': np.int64, 'End': np.int64, 'Barcode': str, 'ReadSupport': np.int64},
        chunksize=settings['fragment_chunk_rows'])
    say('COUNT SHARED REGIONS', capture, len(kept), 'cells')
    for chunk in reader:
        y, nf, ip = overlaps_in_chunk(chunk, bc, index, shape)
        x += y; nuclear += nf; in_peak += ip; rows += len(chunk)
        if rows % (settings['fragment_chunk_rows'] * 10) == 0:
            say('COUNT PROGRESS', capture, rows, 'fragment rows', x.nnz, 'nonzero matrix entries')
    x.sum_duplicates(); x.sort_indices()
    need(np.array_equal(nuclear, pd.to_numeric(kept.unique_fragments_count).to_numpy(np.int64)),
         'Nuclear molecule counts disagree with completed QC: ' + capture)
    need(np.array_equal(in_peak, pd.to_numeric(kept.unique_fragments_in_peaks_count).to_numpy(np.int64)),
         'Fragments-in-peaks disagree with completed official pycisTopic QC: ' + capture)
    need(x.nnz > 0 and np.all(x.data > 0), 'Empty/invalid count matrix: ' + capture)
    # Overlap count sums can exceed unique fragments-in-peaks when one molecule spans two peaks.
    counts = np.asarray(x.sum(axis=0)).ravel()
    need(np.all(counts >= in_peak), 'Matrix overlap counts below unique in-peak molecules')
    save_sparse(out/'COUNTS.regions_by_cells.npz', x)
    table(out/'CELL_ORDER.tsv.gz', kept[['Cell_ID', 'scplus_capture', 'scplus_fragment_barcode']])
    metrics = kept[['Cell_ID']].copy(); metrics['nuclear_molecules'] = nuclear
    metrics['unique_molecules_in_consensus'] = in_peak; metrics['fragment_region_overlaps'] = counts
    table(out/'COUNT_VERIFICATION.tsv.gz', metrics)
    checkpoint(run, out/'COMPLETE.json', signature,
        [out/'COUNTS.regions_by_cells.npz', out/'CELL_ORDER.tsv.gz', out/'COUNT_VERIFICATION.tsv.gz'],
        dict(capture=capture, cells=len(kept), regions=len(regions), nnz=x.nnz, scanned_rows=rows,
             nuclear_and_in_peak_molecule_counts_match_official_QC=True))
    say('CAPTURE COUNT COMPLETE', capture, x.nnz, 'entries; molecule counts match QC')


def combine(run, paths, settings, signature):
    marker = run/'CISTOPIC_COMPLETE.json'
    if reuse(run, marker, signature) is not None:
        say('REUSE COMBINED COUNTS AND CISTOPIC OBJECT'); return
    from pycisTopic.cistopic_class import create_cistopic_object
    kept = load_table(run/'RETAINED_CELLS.tsv.gz')
    regions, _ = region_index(Path(paths['consensus_run'])/'CONSENSUS_REGIONS.mm10.bed')
    support = np.zeros(len(regions), np.int64)
    captures = sorted(kept.scplus_capture.unique())
    for capture in captures:
        out = run/'counts_by_capture'/capture
        need(reuse(run, out/'COMPLETE.json', signature) is not None, 'Capture incomplete: ' + capture)
        x = sparse.load_npz(out/'COUNTS.regions_by_cells.npz')
        support += np.diff(x.indptr)
        del x
    use = support >= settings['min_cells_per_peak']
    need(use.any(), 'No peaks pass global cell support')
    regions['cells_with_fragment'] = support; regions['retained_for_topics'] = use
    table(run/'ALL_CONSENSUS_REGION_SUPPORT.tsv.gz', regions)
    retained_regions = regions.loc[use].copy()
    retained_regions[['Chromosome', 'Start', 'End']].to_csv(run/'TOPIC_REGIONS.mm10.bed', sep='\t', header=False, index=False)
    parts, order = [], []
    for capture in captures:
        out = run/'counts_by_capture'/capture
        x = sparse.load_npz(out/'COUNTS.regions_by_cells.npz')
        parts.append(x[use].tocsc()); order.extend(load_table(out/'CELL_ORDER.tsv.gz').Cell_ID)
    need(order == kept.Cell_ID.tolist(), 'Capture concatenation changed retained cell order')
    matrix = sparse.hstack(parts, format='csr', dtype=np.int32); del parts; gc.collect()
    need((np.asarray(matrix.sum(axis=0)).ravel() > 0).all(), 'A cell has no fragments after region support filtering')
    save_sparse(run/'ATAC_COUNTS.regions_by_cells.npz', matrix)
    names = (retained_regions.Chromosome + ':' + retained_regions.Start.astype(str) + '-' + retained_regions.End.astype(str)).tolist()
    need(len(set(names)) == len(names), 'Repeated final region IDs')
    table(run/'ATAC_REGION_ORDER.tsv.gz', pd.DataFrame({'region': names}))
    fragments = {c: str(Path(paths['consensus_run'])/'captures'/c/'candidate_nuclear.fragments.tsv.gz') for c in captures}
    say('CREATE OFFICIAL CISTOPIC OBJECT', matrix.shape, matrix.nnz, 'entries')
    obj = create_cistopic_object(matrix, cell_names=order, region_names=names, min_frag=1, min_cell=1,
        is_acc=1, path_to_fragments=fragments, project='CD8_without_W8_W9', tag_cells=False)
    need(obj.cell_names == order and obj.region_names == names, 'pycisTopic changed cell/region order')
    metadata = kept.set_index('Cell_ID', drop=False).copy()
    metadata['sample_id'] = metadata.scplus_capture
    metadata['barcode'] = metadata.scplus_fragment_barcode
    obj.add_cell_data(metadata)
    need(obj.cell_data.loc[order, 'barcode'].tolist() == kept.scplus_fragment_barcode.tolist(), 'Fragment barcode metadata changed')
    p = run/'CISTOPIC_QC_PASSED.pkl'; q = p.with_name(p.name + '.tmp')
    with open(q, 'wb') as f: pickle.dump(obj, f, protocol=4)
    q.replace(p)
    info = dict(retained_cells=len(kept), consensus_regions=len(regions), topic_regions=len(names),
                minimum_cells_per_topic_region=settings['min_cells_per_peak'], nonzero_counts=matrix.nnz,
                full_Cell_IDs_preserved=True, W8_W9_included=False)
    dump(run/'MATRIX_SUMMARY.json', info)
    checkpoint(run, marker, signature, [run/n for n in ['ALL_CONSENSUS_REGION_SUPPORT.tsv.gz',
        'TOPIC_REGIONS.mm10.bed', 'ATAC_COUNTS.regions_by_cells.npz', 'ATAC_REGION_ORDER.tsv.gz',
        'CISTOPIC_QC_PASSED.pkl', 'MATRIX_SUMMARY.json']], info)
    say('COMBINED COUNTS AND CISTOPIC COMPLETE', info)


def csr_equal(a, b):
    a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
    return a.shape == b.shape and (a != b).nnz == 0


def marker_summary(run, rna, kept):
    # Descriptive evidence only. No automatic Tex-terminal call or universal single-gene gate.
    panels = dict(exhaustion_context=['Pdcd1', 'Tox', 'Lag3', 'Tigit', 'Nr4a1', 'Nr4a2'],
        terminal_associated=['Havcr2', 'Entpd1', 'Cd101'], progenitor_associated=['Tcf7', 'Lef1', 'Slamf6', 'Il7r', 'Id3'],
        effector_comparison=['Cx3cr1', 'Gzmb', 'Prf1', 'Ifng', 'Tbx21', 'Eomes'],
        residency_comparison=['Itgae', 'Cxcr6', 'Znf683', 'Runx3'], cycling=['Mki67', 'Top2a'])
    genes = [g for panel in panels.values() for g in panel]
    found = [g for g in genes if g in rna.var_names]
    table(run/'MARKER_GENE_AVAILABILITY.tsv', pd.DataFrame([
        dict(panel=p, gene=g, present_in_RNA=g in found) for p, gl in panels.items() for g in gl]))
    if not found: return
    expression = sparse.csr_matrix(rna[:, found].X)
    detection = sparse.csr_matrix(rna.raw[:, found].X > 0, dtype=np.float64)
    rows = []
    for state_col in ['trbi_level2', 'gse267584_best_state']:
        group = pd.MultiIndex.from_frame(kept[['scplus_capture', state_col]].astype(str))
        labels, groups = pd.factorize(group, sort=True)
        membership = sparse.csr_matrix((np.ones(len(kept)), (labels, np.arange(len(kept)))), shape=(len(groups), len(kept)))
        n = np.bincount(labels)
        means = (membership @ expression).toarray() / n[:, None]
        detected = (membership @ detection).toarray() / n[:, None]
        for i, (capture, state) in enumerate(groups):
            for j, g in enumerate(found):
                rows.append(dict(capture=capture, annotation_source=state_col, state=state, cells=int(n[i]),
                    small_group_lt20=bool(n[i] < 20), gene=g, mean_log1p_CPM10k=float(means[i, j]),
                    fraction_detected_counts=float(detected[i, j])))
    table(run/'TERMINAL_EXHAUSTION_MARKER_REVIEW.tsv.gz', pd.DataFrame(rows))


def aligned_h5ad(run, paths, settings, signature):
    import anndata as ad
    marker = run/'ALIGNED_H5AD_COMPLETE.json'
    if reuse(run, marker, signature) is not None:
        say('REUSE ALIGNED RNA/ATAC H5AD FILES'); return
    prep = Path(paths['preparation_run']); p = prep/'RNA_CD8_CANDIDATES_FOR_SCENICPLUS.h5ad'
    need(sha(prep/'RUN_MANIFEST.json') == settings['preparation_manifest_SHA256'], 'Step 05 manifest changed')
    manifest = json.loads((prep/'RUN_MANIFEST.json').read_text())
    need(manifest['status'] == 'COMPLETE' and sha(p) == manifest['output_SHA256'][p.name], 'Prepared RNA changed')
    kept = load_table(run/'RETAINED_CELLS.tsv.gz'); ids = kept.Cell_ID.tolist()
    source = ad.read_h5ad(p, backed='r')
    need(source.obs_names.is_unique and set(ids).issubset(source.obs_names), 'Retained cell IDs missing from RNA')
    need(source.raw is not None and source.var_names.equals(source.raw.var_names), 'Expected full raw-count RNA is missing')
    say('EXTRACT MATCHED RNA', len(ids), 'cells; normalized X and raw counts preserved')
    rna = source[ids, :].to_memory()
    metadata = kept.set_index('Cell_ID', drop=False)
    # Original annotation and provenance columns remain their original typed values.
    for c in metadata:
        if c not in rna.obs or c in ['final_ATAC_QC_status', 'scplus_ATAC_QC_status']:
            rna.obs[c] = metadata.loc[ids, c].to_numpy()
        elif c.startswith('trbi_'):
            a, b = rna.obs[c].astype(str).to_numpy(), metadata.loc[ids, c].astype(str).to_numpy()
            same = np.array_equal(a, b)
            if not same and pd.api.types.is_numeric_dtype(rna.obs[c]):
                av = pd.to_numeric(rna.obs[c], errors='coerce').to_numpy()
                bv = pd.to_numeric(metadata.loc[ids, c], errors='coerce').to_numpy()
                same = np.allclose(av, bv, rtol=1e-12, atol=1e-12, equal_nan=True)
            need(same, 'ImmGen annotation/score differs between source and QC: ' + c)
    rna.obs['scplus_ATAC_QC_status'] = 'PASS_PRIMARY'
    rna.uns['scplus_step07'] = dict(ATAC_QC=settings['primary_QC_description'],
        count_layer='raw.X', normalized_X_unchanged=True, terminal_phenotype='NOT_ASSIGNED_BY_THIS_STEP',
        reference_mapping='GSE267584 sidecar retained separately from trbi labels', run=str(run))
    marker_summary(run, rna, kept)
    out = run/'RNA_QC_PASSED.h5ad'; q = out.with_name(out.name + '.tmp')
    rna.write_h5ad(q, compression='gzip', compression_opts=1)
    check = ad.read_h5ad(q, backed='r')
    need(check.obs_names.tolist() == ids and check.var_names.equals(source.var_names), 'Written RNA axes changed')
    for start in range(0, len(ids), 4096):
        end = min(start + 4096, len(ids)); block = ids[start:end]
        need(csr_equal(check.X[start:end], source[block].X), 'Normalized RNA changed')
        need(csr_equal(check.raw.X[start:end], source[block].raw.X), 'RNA raw counts changed')
    check.file.close(); source.file.close(); q.replace(out)
    say('MATCHED RNA VERIFIED: all normalized values and raw counts unchanged')
    var = pd.read_csv(run/'ATAC_REGION_ORDER.tsv.gz', sep='\t', keep_default_na=False)
    regions = pd.read_csv(run/'TOPIC_REGIONS.mm10.bed', sep='\t', names=['chrom', 'start', 'end'])
    regions.index = var.region.tolist()
    x = sparse.load_npz(run/'ATAC_COUNTS.regions_by_cells.npz').transpose().tocsr()
    atac = ad.AnnData(X=x, obs=rna.obs.copy(), var=regions)
    atac.uns['count_semantics'] = 'Unique fragment-region overlaps; read-support counts ignored; zero-based half-open regions.'
    atac.uns['assembly'] = 'GRCm38/mm10'
    atac.uns['scplus_step07'] = dict(min_cells_per_region=settings['min_cells_per_peak'], consensus_run=paths['consensus_run'])
    outa = run/'ATAC_QC_PASSED.h5ad'; qa = outa.with_name(outa.name + '.tmp')
    atac.write_h5ad(qa, compression='gzip', compression_opts=1)
    check = ad.read_h5ad(qa, backed='r')
    need(check.obs_names.tolist() == ids and check.var_names.tolist() == var.region.tolist(), 'ATAC axes changed')
    for start in range(0, len(ids), 4096):
        need(csr_equal(check.X[start:start+4096], x[start:start+4096]), 'Written ATAC counts changed')
    check.file.close(); qa.replace(outa)
    files = [run/n for n in ['RNA_QC_PASSED.h5ad', 'ATAC_QC_PASSED.h5ad', 'MARKER_GENE_AVAILABILITY.tsv',
        'TERMINAL_EXHAUSTION_MARKER_REVIEW.tsv.gz'] if (run/n).exists()]
    checkpoint(run, marker, signature, files, dict(cells=len(ids), RNA_genes=rna.n_vars, ATAC_regions=atac.n_vars,
        RNA_X_and_raw_counts_exactly_preserved=True, identical_RNA_ATAC_cell_order=True))
    say('ALIGNED RNA/ATAC H5AD COMPLETE', len(ids), 'cells')


def finish(run, paths, settings, signature):
    infos = {}
    for name in ['SELECTION', 'CISTOPIC', 'ALIGNED_H5AD']:
        infos[name] = reuse(run, run/(name+'_COMPLETE.json'), signature)
        need(infos[name] is not None, 'Incomplete stage: ' + name)
    summary = dict(status='COMPLETE', **infos['CISTOPIC'], primary_ATAC_QC=settings['primary_QC_description'],
        strict_depth_sensitivity_cells=infos['SELECTION']['stricter_depth'],
        RNA_X_and_raw_counts_exactly_preserved=True, identical_RNA_ATAC_cell_order=True,
        ImmGen_and_reference_labels_kept_separately=True, terminal_phenotype_assigned=False,
        next_step='Fit/evaluate cisTopic topic models, then differential accessible regions, motif enrichment and SCENIC+ network inference.')
    dump(run/'SUMMARY.json', summary)
    outputs = {}
    for name in ['SELECTION', 'CISTOPIC', 'ALIGNED_H5AD']:
        outputs.update(json.loads((run/(name+'_COMPLETE.json')).read_text())['output_SHA256'])
    outputs['SUMMARY.json'] = sha(run/'SUMMARY.json')
    dump(run/'RUN_MANIFEST.json', dict(status='COMPLETE', signature=signature, output_SHA256=outputs))
    small = ['README_RUN.md', 'RUN_ACTIVITY.md', 'SETTINGS.json', 'INPUT_PATHS.json', 'SOURCES_SHA256.json',
        'RUN_MANIFEST.json', 'SUMMARY.json', 'MATRIX_SUMMARY.json', 'SELECTION_SUMMARY.json',
        'QC_RETENTION_BY_CAPTURE.tsv', 'QC_RETENTION_BY_IMMGEN_STATE.tsv', 'QC_RETENTION_BY_REFERENCE_STATE.tsv',
        'MARKER_GENE_AVAILABILITY.tsv', 'TERMINAL_EXHAUSTION_MARKER_REVIEW.tsv.gz',
        'ENVIRONMENT_QC_PIP_FREEZE.txt', 'ENVIRONMENT_RNA_PIP_FREEZE.txt']
    out = run/'SHARE_SCENICPLUS_COMMON_MATRIX.zip'; tmp = out.with_name(out.name+'.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in small:
            if (run/n).is_file(): z.write(run/n, n)
    tmp.replace(out)
    say('COMPLETE', summary); say('SHARE', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    parser.add_argument('stage', choices=['select', 'count', 'combine', 'h5ad', 'finish'])
    parser.add_argument('--capture')
    args = parser.parse_args(); run = args.run.resolve(); paths, settings, signature = frozen(run)
    if args.stage == 'select': selection(run, paths, settings, signature)
    elif args.stage == 'count':
        if args.capture:
            count_capture(run, args.capture, signature)
        else:
            import subprocess
            kept = load_table(run/'RETAINED_CELLS.tsv.gz')
            def task(c):
                subprocess.run([sys.executable, '-u', str(run/'scplus_common_matrix.py'), str(run), 'count', '--capture', c], check=True)
            with cf.ThreadPoolExecutor(max_workers=settings['count_workers']) as ex:
                for f in cf.as_completed([ex.submit(task, c) for c in sorted(kept.scplus_capture.unique())]): f.result()
    elif args.stage == 'combine': combine(run, paths, settings, signature)
    elif args.stage == 'h5ad': aligned_h5ad(run, paths, settings, signature)
    else: finish(run, paths, settings, signature)
SP_COMMON_FROZEN_SOURCE_0
cat > "$SP_RUN/README_RUN.md" <<'SP_COMMON_FROZEN_SOURCE_1'
# SCENIC+ Step 07: common-region ATAC counts and matched RNA on Betty

This stage follows the completed consensus/QC run `consensus_20261021T170938_114096` (repaired job 9273013, completed in 47m44s). Its reports contain 133,059 measured CD8 candidates in 21 captures and 439,446 nonoverlapping, blacklist-filtered 500-bp mm10 regions. W8/W9 stay excluded from this working analysis, not from the annotated master. No original data, annotation, reference mapping, peak call or consensus file is overwritten.

## QC decision

Keep cells with at least **1,000 unique primary nuclear fragments, TSS enrichment >=5, and consensus FRiP >=0.20**. These are analysis choices informed by the per-capture plots, not universal cutoffs guaranteed by SCENIC+. Preserve a stricter sensitivity flag requiring 3,000 fragments with the same TSS/FRiP cutoffs. Retention is summarized by capture and by each annotation system, including rejected cells and explicit reasons.

The earlier tested 1000/TSS3/FRiP0.10 profile retained 117,665 cells; the earlier 3000/TSS5/FRiP0.20 profile retained 93,560. Neither is the exact count for the new primary profile. Step 07 calculates and prints that count. The 3,000-fragment cutoff disproportionately removes cells from shallower captures (notably GSM7220515 and GSM8269392), despite high median TSS. We retain those cells if they meet the primary quality requirements, and retain depth information for sensitivity analyses. Do not blindly apply the diagnostic Otsu thresholds, which can separate good-quality populations.

FRiP is calculated against a broad consensus derived from these same candidates; its high value is not independent proof of quality. TSS enrichment, depth, previous RNA/doublet review, and subsequent topic/cell structure remain relevant. No upper-depth exclusion, repeat doublet filtering, or library-specific biological selection is introduced here. No evidence in the current review requires repeating annotation or peak calling. Thresholds can be reconsidered if subsequent data expose a specific problem.

## Common-region counting

Use the existing, checksum-verified `candidate_nuclear.fragments.tsv.gz` for each capture, with the previously validated exact fragment barcode. Do not strip `-1_1` ... `-1_7` suffixes. Map through the existing full `Cell_ID`; overlapping barcode strings in different captures are different cells.

Count one unique fragment molecule for every consensus interval it overlaps. Read-support counts are not multiplied into the matrix. Coordinates are zero-based half-open; touching endpoints do not overlap. A fragment spanning multiple distinct regions contributes once to each region. This matches the interval-join/count semantics in the frozen official pycisTopic `create_cistopic_object_from_fragments` implementation. A streaming sparse counter avoids that function's dense pandas pivot at this dataset's size. The counter is tested against PyRanges joins and official pycisTopic fragment-in-peak QC on synthetic cases, and the real job requires exact agreement with each retained cell's already-completed nuclear molecule and unique-in-peak molecule counts.

Process three captures concurrently, checkpointing successful capture matrices. Each capture uses the same complete 439,446-region axis. Then keep regions with a fragment in at least **10 retained cells globally**, rather than independently applying that filter within each capture. This is a permissive sparse-feature support filter; it does not select peaks by terminal-state labels. Report the support of every consensus region. No differing native peak sets are zero-padded or exact-boundary-intersected.

Create an official pycisTopic object using `create_cistopic_object` on the sparse region-by-cell count matrix, without adding new Cell_ID suffixes. Keep exact fragment barcode in `barcode`, capture in `sample_id`, and the original biological metadata in their original columns. The fragment-path dictionary is keyed by capture. Future fragment operations must use this explicit barcode metadata; do not reconstruct fragment barcodes by splitting full RNA Cell_ID strings.

Create matching `RNA_QC_PASSED.h5ad` and `ATAC_QC_PASSED.h5ad` with identical cell IDs/order. RNA X remains the existing log1p(10,000-scaled counts), and RNA `.raw.X` remains the original counts; neither is recalculated. Read back and compare every normalized RNA value, raw count and ATAC count. The RNA object retains all 32,245 supplied genes. The ATAC X contains integer fragment-region overlaps, without RNA-style normalization. The standalone sparse NPZ is **regions x cells**; ATAC AnnData X is **cells x regions**.

## Annotations and terminal exhaustion

Keep `trbi_*` labels unchanged. Add the already-completed GSE267584 mapping sidecar by exact Cell_ID, keeping `gse267584_*` fields separate. Exact reference-overlap, same-study, low-support and provenance flags remain available. Reference labels and transfer scores are not independent experimental validation of ImmGen calls, and a best transferred label is not a calibrated probability.

This stage does **not** assign a terminally exhausted phenotype. CD8.R, CD8.S, or a transferred `Tex-term` label alone does not settle that definition. Preserve the full range of quality-passing CD8 states for regulatory comparisons, including progenitor/intermediate, effector, resident-like and cycling populations. Do not restrict the entire regulatory model to a preselected terminal label.

Produce descriptive per-capture/per-annotation marker means and detection fractions for:

- Exhaustion context: Pdcd1, Tox, Lag3, Tigit, Nr4a1, Nr4a2.
- Terminal-associated evidence: Havcr2, Entpd1, Cd101.
- Progenitor-associated evidence: Tcf7, Lef1, Slamf6, Il7r, Id3.
- Effector, residency and cycling comparison genes, to help distinguish other plausible programs.

These panels are a review aid, not an automatic classifier or a validated scoring model. A single missing transcript can reflect dropout; low Tcf7 alone does not establish terminal exhaustion, and activated effectors can express inhibitory receptors. Interpret coordinated exhaustion/terminal programs, relative progenitor programs, tissue/chronic-antigen context and coherent cell populations. Keep uncertain cells uncertain. Terminal exhaustion can retain cytotoxic-gene expression, so low Gzmb/Prf1 is not a universal requirement. Static RNA/ATAC cannot establish irreversible functional exhaustion; use an operational **terminal-exhausted-like** phenotype until stronger validation supports the claim. Small groups (<20 cells) are flagged. Group means can conceal mixed populations; review or subcluster relevant populations before final phenotype assignment. Do not use inferred regulons to define the phenotype and then claim the same regulons independently validate it.

The 12,890 candidate cells in GSE267584 captures GSM8269394/9395/9396 retain unresolved condition identities. Quality-passing cells can remain in an exploratory regulatory atlas, but these captures must not be assigned tumor/infection conditions for biological comparisons until resolved. RNA mapping does not resolve that metadata conflict.

## Execution, recovery and outputs

Run only on Betty. Request 24 CPUs and 5632 MiB/CPU (132 GiB), with a 24-hour limit; this limit is not a runtime estimate. Three count workers are active during counting. Existing QC and AnnData environments are read-only and checked by importing the actual required functions. No package installation, MACS repair or GPU is needed. The minimal QC environment supports object construction here; topic inference and full SCENIC+ inference still require a suitable subsequent environment.

Transfer `SCENICPLUS_COMMON_MATRIX_BETTY.sh`, then execute it with bash. It creates a new run under `scenicplus_inputs/07_common_matrix/runs/`. Resume an interrupted run with its saved `SUBMIT_OR_RESUME.sh`; rerunning the installer creates another run. Successful capture counts and subsequent stages are checksum-verified and reused; an interrupted individual capture restarts that capture. Saved resume checks prevent duplicate active submissions. Frozen `README_RUN.md`, scripts and settings are never appended to; runtime notes go to the root README and RUN_ACTIVITY.md.

Main outputs:

- `RNA_QC_PASSED.h5ad`, `ATAC_QC_PASSED.h5ad`: identically ordered cells, preserved labels.
- `CISTOPIC_QC_PASSED.pkl`: official pycisTopic object; no fitted topic model yet.
- `ATAC_COUNTS.regions_by_cells.npz`, `ATAC_REGION_ORDER.tsv.gz`, `RETAINED_CELLS.tsv.gz`: explicit axes.
- `ALL_CANDIDATE_QC_DECISIONS.tsv.gz`, rejected cells, capture/state retention summaries.
- `ALL_CONSENSUS_REGION_SUPPORT.tsv.gz`, `TOPIC_REGIONS.mm10.bed`.
- `TERMINAL_EXHAUSTION_MARKER_REVIEW.tsv.gz`, gene-availability table, manifest and summaries.
- `SHARE_SCENICPLUS_COMMON_MATRIX.zip`: small reports for review, excluding large matrices/objects.

Topic-model fitting/evaluation, topic/DAR selection, motif enrichment, TF-region-gene inference and regulon validation are subsequent steps. This stage creates the matched input objects, not the final SCENIC+ networks.

## Primary methods references

- Official pycisTopic tutorial: https://pycistopic.readthedocs.io/en/latest/notebooks/human_cerebellum.html
- Frozen counting/object implementation: https://github.com/aertslab/pycisTopic/blob/787ce422a37f5975b0ebb9e7b19eeaed44847501/src/pycisTopic/cistopic_class.py
- Beltra et al., Immunity 2020, exhaustion differentiation stages: https://doi.org/10.1016/j.immuni.2020.04.014
- Hudson et al., Immunity 2019, transitory versus more terminal exhausted cells: https://doi.org/10.1016/j.immuni.2019.11.002
- Miller et al., Nature Immunology 2019, progenitor/terminal exhausted tumor T-cell subsets: https://www.nature.com/articles/s41590-019-0312-6
SP_COMMON_FROZEN_SOURCE_1
cat > "$SP_RUN/SETTINGS.json" <<'SP_COMMON_FROZEN_SOURCE_2'
{
  "expected_candidates": 133059,
  "expected_captures": 21,
  "min_nuclear_fragments": 1000,
  "min_TSS": 5,
  "min_FRIP": 0.2,
  "primary_QC_description": "unique primary nuclear fragments >=1000; TSS enrichment >=5; consensus FRiP >=0.20",
  "min_cells_per_peak": 10,
  "fragment_chunk_rows": 1000000,
  "count_workers": 3,
  "pycisTopic_commit": "787ce422a37f5975b0ebb9e7b19eeaed44847501",
  "preparation_manifest_SHA256": "e8fe09d0af81190b25a753dab197d0518f47eb950889049067bf1b2d1f5fee67",
  "reference_mapping_SHA256": "1c9e521d69d4d288e4c432fbadea3622c29c0847853c28375c04d8444c4714b4",
  "confirmed_inputs_SHA256": {
    "CONSENSUS_REGIONS.mm10.bed": "37b94e5a05d99a0b9caa9cf52c6d6c318f7fc4782975fac10b795106e3ae3c0d",
    "ALL_CANDIDATE_ATAC_QC.tsv.gz": "b19fda16b5498189ac04539a5fee54b54adb955be8e756b120f534c37466bd19"
  }
}
SP_COMMON_FROZEN_SOURCE_2

python3 - "$SP_RUN" "$SP_PREV" "$SP_MAP" <<'SP_COMMON_PATHS'
import json,pathlib,sys
run,prev=map(pathlib.Path,sys.argv[1:3])
paths=json.loads((prev/'INPUT_PATHS.json').read_text())
assert json.loads((prev/'RUN_MANIFEST.json').read_text())['status']=='COMPLETE', 'Prior run is incomplete'
(run/'INPUT_PATHS.json').write_text(json.dumps(dict(consensus_run=str(prev),preparation_run=paths['preparation_run'],reference_mapping=sys.argv[3]),indent=2)+'\n')
SP_COMMON_PATHS
printf '%s\n' "$SP_BASE" > "$SP_RUN/PROJECT_BASE.txt"
cat > "$SP_RUN/RUN_ON_COMPUTE.sh" <<'SP_COMMON_COMPUTE'
#!/usr/bin/env bash
set -euo pipefail
SP_RUN="$(cd -- "${1:?Pass the saved run folder}" && pwd)"
SP_ROOT="$(dirname -- "$(dirname -- "$SP_RUN")")"
SP_BASE="$(cat "$SP_RUN/PROJECT_BASE.txt")"
test -n "${SLURM_JOB_ID:-}" || { echo 'Use SUBMIT_OR_RESUME.sh from the login node.' >&2; exit 1; }
exec 9>"$SP_RUN/.compute.lock"
flock -n 9 || { echo 'This run is already computing.' >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export POLARS_MAX_THREADS=2 ARROW_NUM_THREADS=2 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
unset PYTHONPATH PYTHONHOME
SP_QC_PY="$SP_BASE/scenicplus_inputs/envs/scplus_fragment_qc_py311_v1/bin/python"
SP_RNA_PY="$SP_BASE/scenicplus_inputs/envs/scplus_input_prep_py311_v1/bin/python"
sp_log() { printf '\n- %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$SP_ROOT/README.md" >> "$SP_RUN/RUN_ACTIVITY.md"; }
trap 'SP_RC=$?; if [ "$SP_RC" -ne 0 ]; then sp_log "STOPPED job $SLURM_JOB_ID exit $SP_RC; completed capture counts are checkpointed. Resume $SP_RUN/SUBMIT_OR_RESUME.sh"; fi' EXIT
sp_log "START job $SLURM_JOB_ID; primary ATAC QC: nuclear fragments >=1000, TSS >=5, FRiP >=0.20. $SP_RUN"
echo "RUN: $SP_RUN"
echo "Slurm job $SLURM_JOB_ID; allocated CPUs ${SLURM_CPUS_PER_TASK:-unknown}; 3 independent fragment-counting workers. No GPU."
"$SP_QC_PY" -c 'import numpy,pandas,scipy; from pycisTopic.cistopic_class import create_cistopic_object; print("Existing pycisTopic counting/object imports passed")'
"$SP_RNA_PY" -c 'import anndata,h5py,numpy,pandas,scipy; print("Existing RNA/AnnData imports passed")'
"$SP_QC_PY" -m pip freeze > "$SP_RUN/ENVIRONMENT_QC_PIP_FREEZE.txt"
"$SP_RNA_PY" -m pip freeze > "$SP_RUN/ENVIRONMENT_RNA_PIP_FREEZE.txt"
"$SP_QC_PY" -u "$SP_RUN/scplus_common_matrix.py" "$SP_RUN" select
"$SP_QC_PY" -u "$SP_RUN/scplus_common_matrix.py" "$SP_RUN" count
"$SP_QC_PY" -u "$SP_RUN/scplus_common_matrix.py" "$SP_RUN" combine
"$SP_RNA_PY" -u "$SP_RUN/scplus_common_matrix.py" "$SP_RUN" h5ad
"$SP_QC_PY" -u "$SP_RUN/scplus_common_matrix.py" "$SP_RUN" finish
printf '%s\n' "$SP_RUN" > "$SP_ROOT/latest_complete_run.txt"
sp_log "COMPLETE job $SLURM_JOB_ID; aligned RNA/ATAC and cisTopic object ready. ImmGen/reference labels preserved; terminal phenotype not assigned. $SP_RUN"
echo "SHARE: $SP_RUN/SHARE_SCENICPLUS_COMMON_MATRIX.zip"
SP_COMMON_COMPUTE
cat > "$SP_RUN/SUBMIT_OR_RESUME.sh" <<'SP_COMMON_SUBMIT'
#!/usr/bin/env bash
set -euo pipefail
SP_RUN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SP_ROOT="$(dirname -- "$(dirname -- "$SP_RUN")")"
exec 9>"$SP_RUN/.submission.lock"
flock -n 9 || { echo 'Submission already in progress.' >&2; exit 1; }
if [ -f "$SP_RUN/RUN_MANIFEST.json" ] && [ -f "$SP_RUN/SHARE_SCENICPLUS_COMMON_MATRIX.zip" ]; then
  echo "Already completed: $SP_RUN/SHARE_SCENICPLUS_COMMON_MATRIX.zip"; exit 0
fi
if [ -f "$SP_RUN/SLURM_JOB_ID.txt" ]; then
  SP_OLD="$(cat "$SP_RUN/SLURM_JOB_ID.txt")"
  [[ "$SP_OLD" =~ ^[0-9]+$ ]] || { echo 'Malformed saved Slurm job ID.' >&2; exit 1; }
  if SP_Q="$(squeue -h -j "$SP_OLD" -o '%i' 2>"$SP_RUN/last_squeue.err")"; then
    if [ -n "$SP_Q" ]; then echo "Already queued/running: $SP_OLD"; exit 0; fi
  elif ! grep -qi 'invalid job id' "$SP_RUN/last_squeue.err"; then cat "$SP_RUN/last_squeue.err" >&2; exit 1
  fi
fi
SP_OPTS=(--parsable --job-name=SCplus_common --partition=genoa-std-mem --nodes=1 --ntasks=1 \
  --cpus-per-task=24 --mem-per-cpu=5632M --time=24:00:00 --chdir="$SP_RUN" \
  --output="$SP_RUN/slurm_%j.out" --error="$SP_RUN/slurm_%j.err" --export=ALL)
if [ -n "${SCPLUS_SLURM_ACCOUNT:-${IMMGENT_SLURM_ACCOUNT:-}}" ]; then SP_OPTS+=(--account="${SCPLUS_SLURM_ACCOUNT:-$IMMGENT_SLURM_ACCOUNT}"); fi
SP_JOB="$(sbatch "${SP_OPTS[@]}" "$SP_RUN/RUN_ON_COMPUTE.sh" "$SP_RUN")"; SP_JOB="${SP_JOB%%;*}"
[[ "$SP_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $SP_JOB" >&2; exit 1; }
printf '%s\n' "$SP_JOB" > "$SP_RUN/SLURM_JOB_ID.txt"
printf '\n- %s Submitted common-region matrix job %s; %s.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SP_JOB" "$SP_RUN" >> "$SP_ROOT/README.md"
echo "Submitted SCENIC+ common-region matrix job: $SP_JOB"
echo "Check: sacct -j $SP_JOB --format=JobID,State,Elapsed,ExitCode"
echo "Logs: tail -n 40 \"$SP_RUN/slurm_${SP_JOB}.out\" \"$SP_RUN/slurm_${SP_JOB}.err\""
echo "Resume: bash \"$SP_RUN/SUBMIT_OR_RESUME.sh\""
echo "Run folder: $SP_RUN"
SP_COMMON_SUBMIT
python3 - "$SP_RUN" <<'SP_COMMON_FREEZE'
import hashlib,json,pathlib,sys
run=pathlib.Path(sys.argv[1]);files=['scplus_common_matrix.py','SETTINGS.json','README_RUN.md','INPUT_PATHS.json','PROJECT_BASE.txt','RUN_ON_COMPUTE.sh','SUBMIT_OR_RESUME.sh']
(run/'SOURCES_SHA256.json').write_text(json.dumps({n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n')
SP_COMMON_FREEZE
printf '%s\n' "$SP_RUN" > "$SP_ROOT/latest_started_run.txt"
printf '\n- %s Created common-region matrix run %s. Methods frozen in README_RUN.md.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SP_RUN" >> "$SP_ROOT/README.md"
bash "$SP_RUN/SUBMIT_OR_RESUME.sh"
SCPLUS_COMMON_INSTALL
