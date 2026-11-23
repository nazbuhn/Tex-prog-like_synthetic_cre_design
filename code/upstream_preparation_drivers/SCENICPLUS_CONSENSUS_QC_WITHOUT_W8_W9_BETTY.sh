#!/usr/bin/env bash
bash <<'SCPLUS_CONSENSUS_INSTALL'
set -euo pipefail
SP_BASE="${SCPLUS_BASE:-/vast/projects/saargill/cell-type-specific/nazbuhn}"
SP_ROOT="$SP_BASE/scenicplus_inputs/06_consensus_qc"
SP_PREP_ROOT="$SP_BASE/scenicplus_inputs/05_matrix_preparation"
command -v sbatch >/dev/null || { echo 'Run this script on the Betty Slurm login node.' >&2; exit 1; }
SP_PREP=''
for SP_POINTER in latest_complete_run.txt latest_started_run.txt; do
  if [ -f "$SP_PREP_ROOT/$SP_POINTER" ]; then SP_PREP="$(cat "$SP_PREP_ROOT/$SP_POINTER")"; break; fi
done
test -n "$SP_PREP" && test -f "$SP_PREP/RUN_MANIFEST.json" || { echo 'Completed Step 05 preparation is missing.' >&2; exit 1; }
python3 - "$SP_PREP" <<'SP_CHECK_PREP'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
s=json.loads((p/'PREPARATION_SUMMARY.json').read_text())
assert s['status']=='COMPLETE' and s['working_CD8_candidates']==160766 and s['prepared_libraries']==23, 'Unexpected preparation summary; share it.'
SP_CHECK_PREP
mkdir -p "$SP_ROOT/runs"
SP_RUN="$SP_ROOT/runs/consensus_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$SP_RUN"
if [ ! -f "$SP_ROOT/README.md" ]; then
  printf '# SCENIC+ consensus peaks and fragment QC\n\nW8/W9 excluded from this analysis. Master RNA, labels and original fragments remain unchanged.\n' > "$SP_ROOT/README.md"
fi
cat > "$SP_RUN/scplus_consensus_qc.py" <<'SP_CONSENSUS_FROZEN_SOURCE_0'
"""Betty: fragment-supported consensus peaks and pycisTopic QC; no final cell filter."""
import argparse
import concurrent.futures as cf
import contextlib
import datetime
import gzip
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def say(*v):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *v, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n'); q.replace(p)


def table(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    d.to_csv(q, sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0} if p.name.endswith('.gz') else None)
    q.replace(p)


def checkpoint(root, marker, signature, paths, info=None):
    dump(marker, {'signature': signature, 'output_SHA256': {str(Path(p).relative_to(root)): sha(p) for p in paths}, 'info': info or {}})


def reuse(root, marker, signature):
    if not marker.exists():
        return None
    d = json.loads(marker.read_text()); need(d['signature'] == signature, 'Checkpoint settings changed: ' + str(marker))
    for n, want in d['output_SHA256'].items():
        need((root/n).is_file() and sha(root/n) == want, 'Checkpoint output changed: ' + n)
    return d['info']


def primary_sizes(p):
    d = pd.read_csv(p, sep='\t', names=['Chromosome', 'End'])
    allowed = ['chr' + str(i) for i in range(1, 20)] + ['chrX', 'chrY']
    d = d[d.Chromosome.isin(allowed)].copy(); d['Start'] = 0
    need(len(d) == 21, 'Expected mm10 primary nuclear chromosomes chr1-19,X,Y')
    return d[['Chromosome', 'Start', 'End']]


def make_tss(gtf, sizes, out):
    """GTF is one-based inclusive; BED TSS is zero-based, strand-aware."""
    rows = {}; lens = dict(zip(sizes.Chromosome, sizes.End))
    with gzip.open(gtf, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            a = line.rstrip().split('\t')
            if len(a) != 9 or a[2] != 'transcript' or a[0] not in lens:
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]*)";', a[8]))
            if attrs.get('gene_type') != 'protein_coding' or attrs.get('transcript_type') != 'protein_coding':
                continue
            pos = int(a[3]) - 1 if a[6] == '+' else int(a[4]) - 1
            need(a[6] in ['+', '-'] and 0 <= pos < lens[a[0]], 'Bad GTF TSS')
            # Same TSS on same strand is counted once, even if many transcripts share it.
            rows.setdefault((a[0], pos, a[6]), attrs.get('gene_name', attrs['gene_id']))
    need(len(rows) > 0, 'No protein-coding TSS on primary mm10 chromosomes')
    frame = pd.DataFrame([(c, p, p+1, g, '.', st, 'protein_coding') for (c, p, st), g in sorted(rows.items())],
                         columns=['# Chromosome', 'Start', 'End', 'Gene', 'Score', 'Strand', 'Transcript_type'])
    table(out, frame)
    return len(frame)


def resources(run, settings, signature):
    rd = run/'references'; rd.mkdir(exist_ok=True)
    marker = rd/'COMPLETE.json'
    old = reuse(run, marker, signature)
    if old is not None:
        return primary_sizes(rd/'mm10.chrom.sizes')
    for n, spec in settings['resources'].items():
        p = rd/n
        if not p.exists() or (spec.get('SHA256') and sha(p) != spec['SHA256']):
            say('DOWNLOAD REFERENCE', n)
            tmp = p.with_name(p.name+'.download')
            with urllib.request.urlopen(spec['url'], timeout=120) as f, open(tmp, 'wb') as o:
                while True:
                    b = f.read(1024*1024)
                    if not b:
                        break
                    o.write(b)
            if spec.get('SHA256'):
                need(sha(tmp) == spec['SHA256'], 'Genome resource checksum mismatch: ' + n)
            tmp.replace(p)
    sizes = primary_sizes(rd/'mm10.chrom.sizes')
    with gzip.open(rd/'mm10-blacklist.v2.bed.gz', 'rt') as f:
        black = pd.read_csv(f, sep='\t', header=None)
    black = black.loc[black[0].isin(sizes.Chromosome), [0, 1, 2]]
    need(len(black) > 0, 'Empty mm10 blacklist')
    black.to_csv(rd/'mm10-blacklist.primary.bed', sep='\t', index=False, header=False)
    n = make_tss(rd/'gencode.vM25.annotation.gtf.gz', sizes, rd/'mm10.M25.protein_coding.tss.bed')
    files = [p for p in rd.iterdir() if p.is_file() and p.name != marker.name and not p.name.endswith('.download')]
    manifest = {p.name: {'SHA256': sha(p), 'bytes': p.stat().st_size,
                         'url': settings['resources'].get(p.name, {}).get('url', 'derived')} for p in files}
    dump(rd/'RESOURCE_MANIFEST.json', {'assembly': 'GRCm38/mm10', 'protein_coding_unique_TSS': n, 'files': manifest})
    checkpoint(run, marker, signature, files+[rd/'RESOURCE_MANIFEST.json'])
    return sizes


def plan(run, paths, settings):
    prep = Path(paths['preparation_run'])
    manifest = json.loads((prep/'RUN_MANIFEST.json').read_text())
    summary = json.loads((prep/'PREPARATION_SUMMARY.json').read_text())
    need(manifest['status'] == summary['status'] == 'COMPLETE', 'Step 05 is not complete')
    need(summary['working_CD8_candidates'] == settings['expected_before_exclusion'], 'Unexpected Step 05 CD8 scope')
    for n in ['PREPARED_CELL_LINKS.tsv.gz', 'PREPARATION_SUMMARY.json', 'PREPARED_LIBRARY_SUMMARY.tsv']:
        need(sha(prep/n) == manifest['output_SHA256'][n], 'Step 05 output changed: '+n)
    d = pd.read_csv(prep/'PREPARED_CELL_LINKS.tsv.gz', sep='\t', keep_default_na=False)
    need(len(d) == settings['expected_before_exclusion'] and d.Cell_ID.is_unique, 'Unexpected Step 05 cells/IDs')
    required = ['source_GSE', 'source_library', 'trbi_level2', 'condition_identity_status', 'scplus_fragment_status',
                'scplus_fragment_barcode', 'scplus_fragment_file']
    need(set(required).issubset(d.columns), 'Required Step 05 metadata is missing')
    excluded = d.source_GSE.isin(settings['excluded_GSE'])
    need(int(excluded.sum()) == settings['expected_excluded'], 'W8/W9 count changed')
    table(run/'EXCLUDED_FROM_THIS_RUN.tsv.gz', d.loc[excluded])
    d = d.loc[~excluded].copy()
    need(len(d) == settings['expected_cells'], 'Unexpected fragment-supported CD8 count')
    need(d.scplus_fragment_status.isin(['VALIDATED_ORIGINAL_FRAGMENT_ID', 'PROVISIONAL_CAPTURE_LINK_CONDITION_UNKNOWN']).all(),
         'Included cells have unsupported fragment status')
    need(d.scplus_fragment_barcode.ne('').all() and d.trbi_level2.ne('').all(), 'Missing barcode or imported annotation')
    d['scplus_capture'] = d.source_GSE + '__' + d.source_library
    need(d.scplus_capture.nunique() == settings['expected_libraries'], 'Unexpected capture count')
    need(not d.duplicated(['scplus_capture', 'scplus_fragment_barcode']).any(), 'Repeated fragment barcode within capture')
    raw = Path(paths['raw']).resolve()
    for capture, x in d.groupby('scplus_capture', sort=True):
        need(x.scplus_fragment_file.nunique() == 1, 'Several fragment files within '+capture)
        p = Path(x.scplus_fragment_file.iloc[0]).resolve(); p.relative_to(raw)
        need(p.is_file(), 'Fragment file missing: '+str(p))
    fingerprints = {str(Path(x).resolve()): {'bytes': Path(x).stat().st_size, 'mtime_ns': Path(x).stat().st_mtime_ns} for x in sorted(d.scplus_fragment_file.unique())}
    fp = run/'SOURCE_FRAGMENT_FINGERPRINTS.json'
    if fp.exists():
        need(json.loads(fp.read_text()) == fingerprints, 'Original fragment file size/time changed; start a new run after review')
    else:
        dump(fp, fingerprints)
    table(run/'CANDIDATE_CELL_METADATA.tsv.gz', d)
    return d


def bulk_specs(frame, min_cells):
    specs = [{'bulk_id': 'ALL', 'state': 'ALL_CANDIDATE_CD8', 'cells': len(frame), 'role': 'capture_pooled'}]
    for state, x in frame.groupby('trbi_level2', sort=True):
        if len(x) >= min_cells:
            sid = 'STATE_' + hashlib.sha256(state.encode()).hexdigest()[:16]
            specs.append({'bulk_id': sid, 'state': state, 'cells': len(x), 'role': 'state_within_capture'})
    return specs


def filter_capture(run, capture, frame, sizes, signature, settings):
    """Stream original once. Keep exact barcodes, nuclear coordinates and one molecule per row."""
    out = run/'captures'/capture; out.mkdir(parents=True, exist_ok=True)
    src = Path(frame.scplus_fragment_file.iloc[0]); stat = src.stat()
    token = hashlib.sha256((signature+str(src)+str(stat.st_size)+str(stat.st_mtime_ns)).encode()).hexdigest()
    marker = out/'FRAGMENTS_COMPLETE.json'
    old = reuse(run, marker, token)
    if old is not None:
        say('REUSE FRAGMENTS/PSEUDOBULKS', capture); return old
    specs = bulk_specs(frame, settings['min_state_pseudobulk_cells'])
    state_keys = {x['state']: x['bulk_id'] for x in specs[1:]}
    bcs = dict(zip(frame.scplus_fragment_barcode, frame.trbi_level2))
    lens = dict(zip(sizes.Chromosome, sizes.End)); seen = set(); unique = 0; primary_rows = 0; excluded_chrom_rows = 0
    row_counts = {s['bulk_id']: 0 for s in specs}
    filtered = out/'candidate_nuclear.fragments.tsv.gz'
    allpaths = [filtered] + [out/(s['bulk_id']+'.bedpe.gz') for s in specs]
    temporary = [p.with_name(p.name+'.partial') for p in allpaths]
    for q in temporary:
        q.unlink(missing_ok=True)
    # Fragments are sorted by chromosome/start. Aggregate duplicate molecule rows at each start.
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(gzip.open(q, 'wt', compresslevel=1)) for q in temporary]
        fragments_fh, bulk_fhs = handles[0], dict(zip(row_counts, handles[1:]))
        coord = None; bucket = {}; closed_chroms = set(); last_chrom = None; last_start = -1
        def flush():
            nonlocal unique
            if coord is None:
                return
            chrom, start = coord
            for (end, bc), support in sorted(bucket.items()):
                fragments_fh.write(f'{chrom}\t{start}\t{end}\t{bc}\t{support}\n')
                bedline = f'{chrom}\t{start}\t{end}\n'
                bulk_fhs['ALL'].write(bedline); row_counts['ALL'] += 1
                k = state_keys.get(bcs[bc])
                if k:
                    bulk_fhs[k].write(bedline); row_counts[k] += 1
                seen.add(bc); unique += 1
        with gzip.open(src, 'rt') as f:
            for i, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue
                a = line.rstrip('\n').split('\t')
                need(len(a) >= 4, 'Malformed fragment row in '+str(src))
                chrom, bc = a[0], a[3]
                if bc in bcs:
                    if chrom not in lens:
                        excluded_chrom_rows += 1
                    else:
                        start, end = int(a[1]), int(a[2]); support = int(a[4]) if len(a) > 4 and a[4] != '.' else 1
                        need(0 <= start < end <= lens[chrom] and support > 0, 'Invalid mm10 fragment in '+capture)
                        if chrom != last_chrom:
                            need(chrom not in closed_chroms, 'Fragment chromosomes are unsorted: '+capture)
                            if last_chrom is not None:
                                closed_chroms.add(last_chrom)
                            last_chrom, last_start = chrom, -1
                        need(start >= last_start, 'Fragment starts are unsorted: '+capture)
                        last_start = start
                        key = (chrom, start)
                        if key != coord:
                            flush(); bucket.clear(); coord = key
                        bucket[(end, bc)] = bucket.get((end, bc), 0) + support
                        primary_rows += 1
                if i % 5_000_000 == 0:
                    say('FRAGMENT STREAM', capture, i, 'source rows;', unique, 'nuclear molecules written')
            flush()
    need(unique > 0, 'No nuclear fragments in '+capture+'; check chromosome names')
    need(src.stat().st_size == stat.st_size and src.stat().st_mtime_ns == stat.st_mtime_ns, 'Source changed during read')
    for tmp, p in zip(temporary, allpaths):
        tmp.replace(p)
    info = {'capture': capture, 'source_file': str(src), 'source_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'candidate_cells': len(frame), 'cells_with_nuclear_fragments': len(seen), 'nuclear_unique_molecules': unique,
            'source_candidate_primary_rows': primary_rows, 'candidate_nonprimary_rows_excluded': excluded_chrom_rows,
            'pseudobulks': [dict(s, molecule_rows=row_counts[s['bulk_id']], bedpe=str((out/(s['bulk_id']+'.bedpe.gz')).relative_to(run))) for s in specs]}
    table(out/'BARCODE_MAP.tsv.gz', frame)
    dump(out/'FRAGMENT_PREPARATION.json', info)
    checkpoint(run, marker, token, allpaths+[out/'BARCODE_MAP.tsv.gz', out/'FRAGMENT_PREPARATION.json'], info)
    say('FRAGMENTS READY', capture, unique, 'molecules;', len(specs), 'pseudobulks')
    return info


def call_peak(run, capture, spec, signature):
    out = run/'peaks'/capture/spec['bulk_id']; out.mkdir(parents=True, exist_ok=True)
    marker = out/'COMPLETE.json'
    old = reuse(run, marker, signature)
    if old is not None:
        say('REUSE PEAK CALL', capture, spec['bulk_id']); return old
    need(spec['molecule_rows'] > 0, 'Empty pseudobulk')
    name = capture+'__'+spec['bulk_id']
    argv = ['macs2', 'callpeak', '--treatment', str(run/spec['bedpe']), '--name', name, '--outdir', str(out),
            '--format', 'BEDPE', '--gsize', 'mm', '--qvalue', '0.05', '--nomodel', '--shift', '73',
            '--extsize', '146', '--keep-dup', 'all', '--call-summits', '--nolambda']
    say('MACS2', capture, spec['bulk_id'], 'cells', spec['cells'])
    dump(out/'COMMAND.json', argv)
    with open(out/'macs2.log', 'w') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    p = out/(name+'_peaks.narrowPeak'); need(p.exists(), 'MACS2 did not produce a narrowPeak file')
    n = sum(1 for line in p.open() if line.strip())
    info = dict(capture=capture, **spec, peaks=n, narrowPeak=str(p.relative_to(run)))
    checkpoint(run, marker, signature, [p, out/'COMMAND.json', out/'macs2.log'], info)
    return info


def consensus(run, peaks, sizes, signature, settings):
    marker = run/'CONSENSUS_COMPLETE.json'
    old = reuse(run, marker, signature)
    if old is not None:
        return old
    import pyranges as pr
    from pycisTopic.iterative_peak_calling import get_consensus_peaks
    names = ['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand', 'FC_summit', '-log10_pval', '-log10_qval', 'Summit']
    d = {}
    for x in peaks:
        if not x['peaks']:
            continue
        f = pd.read_csv(run/x['narrowPeak'], sep='\t', header=None, names=names)
        f = f.loc[f.Chromosome.isin(sizes.Chromosome)].copy()
        need((f.Summit >= 0).all(), 'Invalid MACS2 summit')
        if len(f):
            d[x['capture']+'__'+x['bulk_id']] = pr.PyRanges(f)
    need(d, 'No successful pseudobulk peaks to form a consensus')
    say('BUILD CONSENSUS', len(d), 'pseudobulks')
    result = get_consensus_peaks(d, peak_half_width=settings['peak_half_width'], chromsizes=sizes,
                                path_to_blacklist=str(run/'references/mm10-blacklist.primary.bed')).df
    result = result[['Chromosome', 'Start', 'End']].sort_values(['Chromosome', 'Start', 'End']).reset_index(drop=True)
    need(len(result) > 0 and not result.duplicated().any(), 'Empty/repeated consensus peaks')
    need((result.End - result.Start == 2*settings['peak_half_width']).all(), 'Unexpected official consensus width')
    for chrom, x in result.groupby('Chromosome', observed=True):
        need((x.Start.iloc[1:].to_numpy() >= x.End.iloc[:-1].to_numpy()).all(), 'Consensus regions overlap: '+str(chrom))
    need(len(pr.PyRanges(result).overlap(pr.read_bed(str(run/'references/mm10-blacklist.primary.bed')))) == 0, 'Blacklist overlap survived')
    p = run/'CONSENSUS_REGIONS.mm10.bed'; result.to_csv(p, sep='\t', index=False, header=False)
    info = {'consensus_peaks': len(result), 'peak_width': 2*settings['peak_half_width'], 'nonempty_pseudobulks': len(d),
            'pycisTopic_commit': settings['pycisTopic_commit'], 'BED_SHA256': sha(p)}
    dump(run/'CONSENSUS_SUMMARY.json', info)
    checkpoint(run, marker, signature, [p, run/'CONSENSUS_SUMMARY.json'], info)
    return info


def qc_capture(run, capture, signature):
    """Use official pycisTopic metric functions; skip density KDE used only to color plots."""
    import polars as pl
    from pycisTopic.fragments import (read_fragments_to_polars_df, read_bed_to_polars_df,
        get_fragments_per_cb, get_fragments_in_peaks, get_insert_size_distribution)
    from pycisTopic.gene_annotation import read_tss_annotation_from_bed
    from pycisTopic.tss_profile import get_tss_profile
    out = run/'qc'/capture; out.mkdir(parents=True, exist_ok=True)
    marker = out/'COMPLETE.json'
    if reuse(run, marker, signature) is not None:
        say('REUSE QC', capture); return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    say('LOAD FRAGMENTS FOR QC', capture)
    fragments = read_fragments_to_polars_df(str(run/'captures'/capture/'candidate_nuclear.fragments.tsv.gz'))
    # The streaming stage already collapsed duplicate molecule records per barcode/start/end.
    stats = get_fragments_per_cb(fragments, min_fragments_per_cb=0, collapse_duplicates=True).to_pandas()
    regions = read_bed_to_polars_df(str(run/'CONSENSUS_REGIONS.mm10.bed'))
    inpeaks = get_fragments_in_peaks(fragments, regions).to_pandas()
    stats = stats.merge(inpeaks, on='CB', how='left', validate='one_to_one')
    for k in ['unique_fragments_in_peaks_count', 'total_fragments_in_peaks_count']:
        stats[k] = stats[k].fillna(0).astype(np.int64)
    stats['fraction_of_fragments_in_peaks'] = stats.unique_fragments_in_peaks_count / stats.unique_fragments_count
    get_insert_size_distribution(fragments).write_parquet(out/'insert_size_distribution.parquet')
    say('COMPUTE TSS ENRICHMENT', capture)
    tss = read_tss_annotation_from_bed(str(run/'references/mm10.M25.protein_coding.tss.bed'))
    enrichment, sample_profile, cell_profile = get_tss_profile(fragments, tss, flank_window=2000,
        smoothing_rolling_window=10, minimum_signal_window=100, tss_window=50, min_norm=0.2, use_genomic_ranges=True)
    enrichment = enrichment.to_pandas()
    stats = stats.merge(enrichment, on='CB', how='left', validate='one_to_one')
    stats['tss_enrichment'] = stats.tss_enrichment.fillna(0)
    # Large per-cell TSS profile is not needed downstream; preserve aggregate + cell enrichment.
    del cell_profile, fragments
    sample_profile.write_parquet(out/'tss_aggregate_profile.parquet')
    frame = pd.read_csv(run/'captures'/capture/'BARCODE_MAP.tsv.gz', sep='\t', keep_default_na=False)
    merged = frame.merge(stats, left_on='scplus_fragment_barcode', right_on='CB', how='left', validate='one_to_one')
    merged['ATAC_QC_measurement_status'] = np.where(merged.CB.notna(), 'MEASURED_PENDING_REVIEW', 'NO_PRIMARY_NUCLEAR_FRAGMENTS')
    need(len(merged) == len(frame) and merged.Cell_ID.tolist() == frame.Cell_ID.tolist(), 'QC lost/reordered cells')
    table(out/'CELL_QC.tsv.gz', merged)
    checkpoint(run, marker, signature, [out/'CELL_QC.tsv.gz', out/'insert_size_distribution.parquet', out/'tss_aggregate_profile.parquet'], {'capture': capture})
    say('QC COMPLETE', capture, len(merged), 'candidates accounted for')


def summarize(run, candidates, consensus_info, signature):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pycisTopic.topic_binarization import threshold_otsu
    frames = []
    for capture in sorted(candidates.scplus_capture.unique()):
        frames.append(pd.read_csv(run/'qc'/capture/'CELL_QC.tsv.gz', sep='\t', keep_default_na=False))
    allqc = pd.concat(frames, ignore_index=True)
    need(len(allqc) == len(candidates) and allqc.Cell_ID.is_unique and set(allqc.Cell_ID) == set(candidates.Cell_ID), 'Incomplete QC accounting')
    rows = []; thresholds = []
    plotdir = run/'plots'; plotdir.mkdir(exist_ok=True)
    for capture, d in allqc.groupby('scplus_capture', sort=True):
        counts = pd.to_numeric(d.unique_fragments_count, errors='coerce')
        pic = pd.to_numeric(d.unique_fragments_in_peaks_count, errors='coerce')
        tss = pd.to_numeric(d.tss_enrichment, errors='coerce')
        frip = pd.to_numeric(d.fraction_of_fragments_in_peaks, errors='coerce')
        measured = counts.notna()
        need(measured.any(), 'No measured nuclear fragment QC for '+capture)
        need(((frip[measured] >= 0) & (frip[measured] <= 1)).all(), 'FRiP outside [0,1]')
        v1 = np.log10(pic[pic >= 100].to_numpy()+1)
        v2 = tss[tss >= 1].to_numpy()
        auto_c = float(10**threshold_otsu(v1, nbins=5000)-1) if len(v1)>1 and np.ptp(v1)>0 else None
        auto_t = float(threshold_otsu(v2, nbins=5000)) if len(v2)>1 and np.ptp(v2)>0 else None
        thresholds.append(dict(capture=capture, Otsu_unique_in_peak_fragments=auto_c, Otsu_TSS=auto_t, applied=False))
        idx = d.index
        allqc.loc[idx, 'review_profile_1000frag_TSS3_FRIP10pct'] = (counts >= 1000) & (tss >= 3) & (frip >= .1)
        allqc.loc[idx, 'review_profile_3000frag_TSS5_FRIP20pct'] = (counts >= 3000) & (tss >= 5) & (frip >= .2)
        allqc.loc[idx, 'review_profile_Otsu_FRIP10pct'] = ((pic > auto_c) & (tss > auto_t) & (frip >= .1)) if auto_c is not None and auto_t is not None else False
        rows.append(dict(capture=capture, candidates=len(d), measured=int(measured.sum()),
            median_unique_nuclear_fragments=float(counts.median()), median_TSS=float(tss.median()), median_FRIP=float(frip.median()),
            review_1000_TSS3_FRIP10=int(allqc.loc[idx, 'review_profile_1000frag_TSS3_FRIP10pct'].sum()),
            review_3000_TSS5_FRIP20=int(allqc.loc[idx, 'review_profile_3000frag_TSS5_FRIP20pct'].sum())))
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].scatter(np.log10(counts+1), tss, s=2, alpha=.25, rasterized=True)
        ax[0].set(xlabel='log10(1 + unique nuclear fragments)', ylabel='TSS enrichment')
        ax[1].scatter(np.log10(counts+1), frip, s=2, alpha=.25, rasterized=True)
        ax[1].set(xlabel='log10(1 + unique nuclear fragments)', ylabel='FRiP in shared consensus')
        ax[2].hist(tss[np.isfinite(tss)], bins=60)
        ax[2].set(xlabel='TSS enrichment', ylabel='Cells')
        fig.suptitle(capture+' — diagnostic only; no cell filter applied'); fig.tight_layout()
        fig.savefig(plotdir/(capture+'_QC.png'), dpi=120); plt.close(fig)
    allqc['final_ATAC_QC_status'] = 'PENDING_THRESHOLD_REVIEW'
    table(run/'ALL_CANDIDATE_ATAC_QC.tsv.gz', allqc)
    table(run/'LIBRARY_QC_SUMMARY.tsv', pd.DataFrame(rows))
    table(run/'AUTOMATIC_THRESHOLDS_DIAGNOSTIC_ONLY.tsv', pd.DataFrame(thresholds))
    cols = ['scplus_capture', 'trbi_level2']
    bystate = allqc.groupby(cols, sort=True).agg(candidates=('Cell_ID', 'size'),
        profile_1000_TSS3_FRIP10=('review_profile_1000frag_TSS3_FRIP10pct', 'sum'),
        profile_3000_TSS5_FRIP20=('review_profile_3000frag_TSS5_FRIP20pct', 'sum')).reset_index()
    table(run/'QC_PROFILE_COUNTS_BY_STATE.tsv', bystate)
    summary = dict(status='COMPLETE', candidate_cells=len(candidates), captures=len(rows),
        excluded_W8_W9=int(json.loads((run/'SETTINGS.json').read_text())['expected_excluded']),
        **consensus_info, final_cell_filter_applied=False, original_RNA_annotations_unchanged=True,
        uncertain_GSE267584_condition_cells=int(candidates.scplus_fragment_status.eq('PROVISIONAL_CAPTURE_LINK_CONDITION_UNKNOWN').sum()),
        next_step='Review per-library ATAC QC, then count retained cells in consensus regions, fit cisTopic models and run motif enrichment/SCENIC+.')
    dump(run/'SUMMARY.json', summary)
    outputs = ['CANDIDATE_CELL_METADATA.tsv.gz', 'EXCLUDED_FROM_THIS_RUN.tsv.gz', 'PSEUDOBULK_PEAK_SUMMARY.tsv',
        'CONSENSUS_REGIONS.mm10.bed', 'CONSENSUS_SUMMARY.json', 'ALL_CANDIDATE_ATAC_QC.tsv.gz', 'LIBRARY_QC_SUMMARY.tsv',
        'AUTOMATIC_THRESHOLDS_DIAGNOSTIC_ONLY.tsv', 'QC_PROFILE_COUNTS_BY_STATE.tsv', 'SUMMARY.json']
    dump(run/'RUN_MANIFEST.json', {'status': 'COMPLETE', 'signature': signature, 'output_SHA256': {n: sha(run/n) for n in outputs}})
    write_share(run)
    say('COMPLETE', json.dumps(summary)); say('SHARE', run/'SHARE_SCENICPLUS_CONSENSUS_QC.zip')


def write_share(run):
    plotdir = run/'plots'
    small = ['SUMMARY.json', 'RUN_MANIFEST.json', 'README_RUN.md', 'RUN_ACTIVITY.md', 'INPUT_PATHS.json', 'SETTINGS.json',
        'SOURCES_SHA256.json', 'PREPARATION_MANIFEST_SNAPSHOT.json', 'LIBRARY_QC_SUMMARY.tsv', 'PSEUDOBULK_PEAK_SUMMARY.tsv',
        'CONSENSUS_SUMMARY.json', 'AUTOMATIC_THRESHOLDS_DIAGNOSTIC_ONLY.tsv', 'QC_PROFILE_COUNTS_BY_STATE.tsv', 'references/RESOURCE_MANIFEST.json']
    out = run/'SHARE_SCENICPLUS_CONSENSUS_QC.zip'
    tmp = out.with_name(out.name+'.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in small:
            if (run/n).is_file():
                z.write(run/n, n)
        for p in sorted(plotdir.glob('*.png')):
            z.write(p, str(p.relative_to(run)))
    tmp.replace(out)

def main(run):
    settings = json.loads((run/'SETTINGS.json').read_text()); paths = json.loads((run/'INPUT_PATHS.json').read_text())
    sources = json.loads((run/'SOURCES_SHA256.json').read_text())
    for n, want in sources.items():
        need(sha(run/n) == want, 'Frozen run file changed: '+n)
    need(sha(Path(paths['preparation_run'])/'RUN_MANIFEST.json') == sha(run/'PREPARATION_MANIFEST_SNAPSHOT.json'), 'Step 05 manifest changed')
    signature = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if (run/'RUN_MANIFEST.json').exists():
        old = json.loads((run/'RUN_MANIFEST.json').read_text())
        need(old['status'] == 'COMPLETE' and old['signature'] == signature, 'Completed run signature changed')
        for n, want in old['output_SHA256'].items():
            need(sha(run/n) == want, 'Completed output changed: '+n)
        if not (run/'SHARE_SCENICPLUS_CONSENSUS_QC.zip').is_file():
            write_share(run)
        say('ALREADY COMPLETE; verified'); return
    candidates = plan(run, paths, settings)
    sizes = resources(run, settings, signature)
    captures = []
    # Three streaming workers; no large fragment tables are held in memory here.
    with cf.ThreadPoolExecutor(max_workers=settings['stream_workers']) as ex:
        fs = [ex.submit(filter_capture, run, capture, frame, sizes, signature, settings)
              for capture, frame in candidates.groupby('scplus_capture', sort=True)]
        for f in cf.as_completed(fs):
            captures.append(f.result())
    peaks = []
    with cf.ThreadPoolExecutor(max_workers=settings['macs_workers']) as ex:
        fs = [ex.submit(call_peak, run, c['capture'], spec, signature) for c in sorted(captures, key=lambda v: v['capture']) for spec in c['pseudobulks']]
        for f in cf.as_completed(fs):
            peaks.append(f.result())
    peaks.sort(key=lambda x: (x['capture'], x['bulk_id']))
    table(run/'PSEUDOBULK_PEAK_SUMMARY.tsv', pd.DataFrame(peaks))
    info = consensus(run, peaks, sizes, signature, settings)
    # Separate processes release the large fragment/TSS allocations after every capture.
    for capture in sorted(candidates.scplus_capture.unique()):
        subprocess.run([sys.executable, '-u', str(run/'scplus_consensus_qc.py'), str(run), '--qc-capture', capture, '--signature', signature], check=True)
    need(sha(Path(paths['preparation_run'])/'RUN_MANIFEST.json') == sha(run/'PREPARATION_MANIFEST_SNAPSHOT.json'), 'Step 05 changed during this run')
    summarize(run, candidates, info, signature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    parser.add_argument('--qc-capture'); parser.add_argument('--signature')
    args = parser.parse_args()
    if args.qc_capture:
        qc_capture(args.run.resolve(), args.qc_capture, args.signature)
    else:
        main(args.run.resolve())
SP_CONSENSUS_FROZEN_SOURCE_0
cat > "$SP_RUN/smoke_test_qc.py" <<'SP_CONSENSUS_FROZEN_SOURCE_1'
"""Small synthetic metric/consensus check on the compute node, before reading real fragments."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr
from pycisTopic.fragments import get_fragments_per_cb, get_fragments_in_peaks
from pycisTopic.tss_profile import get_tss_profile
from pycisTopic.iterative_peak_calling import get_consensus_peaks
from pycisTopic.topic_binarization import threshold_otsu

pl.enable_string_cache()
f = pl.DataFrame({'Chromosome': ['chr1']*24, 'Start': [9990+i*3 for i in range(24)],
                  'End': [10020+i*3 for i in range(24)], 'Name': ['A-1_1']*12+['B-1_1']*12, 'Score': [1]*24})
f = f.with_columns(pl.col('Chromosome').cast(pl.Categorical), pl.col('Name').cast(pl.Categorical),
                   pl.col('Start').cast(pl.Int32), pl.col('End').cast(pl.Int32), pl.col('Score').cast(pl.Int32))
r = pl.DataFrame({'Chromosome': ['chr1'], 'Start': [9500], 'End': [10500]}).with_columns(
    pl.col('Chromosome').cast(pl.Categorical), pl.col('Start').cast(pl.Int32), pl.col('End').cast(pl.Int32))
s = get_fragments_per_cb(f, min_fragments_per_cb=0).to_pandas()
p = get_fragments_in_peaks(f, r).to_pandas()
assert s.unique_fragments_count.sum() == 24 and p.unique_fragments_in_peaks_count.sum() == 24
annotation = pl.DataFrame({'Chromosome': ['chr1'], 'Start': [10000], 'Strand': ['+']}).with_columns(
    pl.col('Chromosome').cast(pl.Categorical), pl.col('Start').cast(pl.Int32))
e, aggregate, per_cell = get_tss_profile(f, annotation)
assert len(e) == 2 and np.isfinite(e.get_column('tss_enrichment').to_numpy()).all()
assert float(threshold_otsu(np.array([1.,2.,3.,8.,9.]), nbins=20)) > 0
with tempfile.TemporaryDirectory(prefix='scplus_qc_smoke_') as td:
    black = Path(td)/'black.bed'; black.write_text('chr1\t50000\t51000\n')
    peaks = pr.PyRanges(pd.DataFrame({'Chromosome': ['chr1','chr1'], 'Start': [9800,19800], 'End': [10200,20200],
        'Name': ['a','b'], 'Score': [10,20], 'Summit': [200,200]}))
    sizes = pd.DataFrame({'Chromosome': ['chr1'], 'Start': [0], 'End': [100000]})
    c = get_consensus_peaks({'synthetic': peaks}, 250, sizes, str(black)).df
    assert len(c) == 2 and (c.End-c.Start == 500).all()
print('FUNCTIONAL QC SMOKE TEST PASSED: exact suffix barcodes, TSS, fragment counts and consensus intervals.', flush=True)
SP_CONSENSUS_FROZEN_SOURCE_1
cat > "$SP_RUN/requirements_qc.txt" <<'SP_CONSENSUS_FROZEN_SOURCE_2'
numpy==1.26.4
scipy==1.12.0
pandas==1.5.3
matplotlib==3.6.3
polars==0.20.13
pyarrow==15.0.0
pyranges==0.0.127
ncls==0.0.68
sorted-nearest==0.0.39
scikit-learn==1.3.2
requests==2.32.3
tqdm==4.66.5
setuptools-scm==8.1.0
SP_CONSENSUS_FROZEN_SOURCE_2
cat > "$SP_RUN/README_RUN.md" <<'SP_CONSENSUS_FROZEN_SOURCE_3'
# SCENIC+ on Betty: consensus peaks and ATAC QC, without W8/W9

This is the next stage after completed Step 05 matrix preparation. All real data processing runs in a Betty Slurm CPU job. Source files, annotated master RNA, Step 05 objects and annotations are read-only. The separate GSE267584 RNA reference-mapping project is independent.

## Scope

The supplied Step 05 report contains 160,766 working CD8 candidates in 23 captures. Exclude GSE278342 W8 and W9 only from this run (27,707 cells), leaving **133,059 CD8 candidates in 21 captures**, from GSE230363, GSE255042, GSE267584 and GSE290618. These are candidates pending ATAC QC. Portal-excluded/other-lineage cells were already outside this working scope; they stay in the annotated master. GSE262306/GSE334878 are not added.

The 12,890 GSE267584 cells in RNA captures GSM8269394/9395/9396 retain `PROVISIONAL_CAPTURE_LINK_CONDITION_UNKNOWN`. Existing accession/processed-file capture links are used provisionally, as in Step 05; biological tumor/infection condition identity remains unresolved. Use neutral capture IDs. Neither peak calling nor label grouping assigns a biological condition. Exclude these captures from condition-specific biological comparisons until resolved.

## Methods

All four included studies report mm10/GRCm38. Freeze UCSC mm10 chromosome sizes, GENCODE M25 (GRCm38.p6) annotation, and the mm10 ENCODE blacklist v2. Reference downloads are checksum-checked. Derive strand-aware zero-based TSS from protein-coding transcripts; deduplicate identical chromosome/TSS/strand positions. There is no liftOver or chromosome-name guessing.

Keep candidate fragments on chr1–19, chrX and chrY. Mitochondrial and nonprimary rows are counted in preparation reports and omitted from both peak calling and QC. Consequently depth, duplication, nucleosome signal and FRiP use **primary nuclear fragments**. Original fragment barcode strings—including GSE230363 `-1_1`…`-1_7` suffixes—are unchanged. `Cell_ID` is joined through the exact per-capture barcode map. Duplicate molecule rows with identical chromosome/start/end/barcode are collapsed and read-support counts summed; support counts are not multiplied into the peak-calling input. No original file or index is edited. Reading derived gzipped files sequentially does not require tabix indexes.

Export one pooled pseudobulk per capture and one per imported `trbi_level2` state with at least 50 cells within that capture. Small states remain represented in the pooled capture pseudobulk and are not removed. Library IDs refer to capture libraries, not upload chunks or inferred biological replicates. RNA labels guide peak discovery, not final regulon conclusions; library/state grouping also retains accessibility shared across RNA states.

Call MACS2 2.2.9.1 peaks with mouse effective genome size, BEDPE paired-fragment input, q=0.05, keep-dup all, no model, call summits, no lambda, and the pycisTopic tutorial's shift/extsize settings (BEDPE uses fragment lengths). Use official pycisTopic summit-based iterative consensus construction with half-width 250 and blacklist filtering, normalizing peak scores across pseudobulks. Its current implementation outputs 500-bp intervals. Verify nonoverlapping intervals and no blacklist intersections. Native peak matrices are not combined by zero-padding or exact-boundary intersection.

Calculate pycisTopic fragment depth, duplication ratio, nucleosome signal, consensus FRiP, insert-size distribution and TSS enrichment separately per capture. TSS settings match the tutorial: flank 2000, rolling window 10, background tails 100, central window 50, minimum background 0.2. Use official metric component functions without density KDE, which only colors diagnostic plots. Store aggregate TSS profiles and per-cell scores; do not retain the large per-cell 4001-position profiles. All candidates are accounted for, including any with no primary nuclear fragments.

Generate diagnostic Otsu cutoffs and two explicit sensitivity profiles (1000 nuclear fragments/TSS3/FRiP0.10; 3000/TSS5/FRiP0.20), with preservation counts per state. These are **review flags only**: no final ATAC cell filter is applied. Initial pseudobulks use RNA-reviewed candidates, following the tutorial's peak-discovery-before-final-QC order. Review QC before creating the final common-region count matrix. Marked low-quality cells will not automatically enter topic modeling; if QC reveals major contamination, assess whether peak recall is needed.

## Software and execution

Frozen pycisTopic commit is recorded in SETTINGS.json, with its archive SHA256. A fresh, purpose-specific environment contains the metric/consensus imports and MACS2. It intentionally omits unused topic/network dependencies; do not treat it as a complete SCENIC+ installation or use package-wide pip-check as its validation. pandas 1.5.3 is the compatible bugfix release used here, rather than upstream's exact 1.5.0 metadata pin. Source packages are installed with --no-deps; functional tests exercise the actual imported APIs before real data is read. A later topic/network stage will use its own environment. No existing conda environment is changed.

Request 24 CPUs, 5632 MiB per CPU (132 GiB total), and a 24-hour wall limit on genoa-std-mem. This wall limit is not a runtime estimate. Three concurrent streaming readers and four concurrent MACS2 calls; QC runs one capture at a time with up to eight data-library threads to control memory. No GPU is needed for this stage.

Freeze source/settings hashes after writing all scripts. README_RUN.md is immutable. Runtime notes append only to RUN_ACTIVITY.md and the project README.md. The job resolves the completed Step 05 pointer under the current scenicplus_inputs directory. It checks the saved Step 05 manifest and candidate metadata before running. Original fragment size/time fingerprints are frozen on first execution; completed derived files are checksum-checked on resume. If an original changes, stop for review rather than silently reusing earlier peaks.

## Run and resume

Transfer SCENICPLUS_CONSENSUS_QC_WITHOUT_W8_W9_BETTY.sh to Betty, then run it with bash. It creates a dated run under:

`/vast/projects/saargill/cell-type-specific/nazbuhn/scenicplus_inputs/06_consensus_qc/runs/`

The installer creates a new run each time. To resume a stopped run, use its saved **SUBMIT_OR_RESUME.sh**. Do not rerun the installer for a routine resume. Completed fragment exports, peak calls, consensus construction and capture QC are checkpointed; an interrupted individual stage restarts that stage. Submitting the saved resume script while its job is active does not create a duplicate job.

Main outputs: CONSENSUS_REGIONS.mm10.bed, CANDIDATE_CELL_METADATA.tsv.gz, ALL_CANDIDATE_ATAC_QC.tsv.gz, LIBRARY_QC_SUMMARY.tsv, QC_PROFILE_COUNTS_BY_STATE.tsv, plots, SUMMARY.json, and RUN_MANIFEST.json. Share **SHARE_SCENICPLUS_CONSENSUS_QC.zip** for threshold review. Fragments and other large objects are not placed in the share archive.

This job performs peak discovery and QC. Topic modeling, topic/region selection, motif enrichment, TF–region–gene inference, and regulon validation remain subsequent SCENIC+ stages. Original ImmGen labels are retained throughout; no new RNA annotation or batch correction is performed here.

## Primary references

- https://pycistopic.readthedocs.io/en/latest/notebooks/human_cerebellum.html
- https://github.com/aertslab/pycisTopic
- https://www.gencodegenes.org/mouse/release_M25.html
- https://github.com/Boyle-Lab/Blacklist
- GEO family records GSE230363, GSE255042, GSE267584 and GSE290618: cached original mm10 assembly statements were checked before choosing genome resources.
SP_CONSENSUS_FROZEN_SOURCE_3
cat > "$SP_RUN/SETTINGS.json" <<'SP_CONSENSUS_FROZEN_SOURCE_4'
{
  "expected_before_exclusion": 160766,
  "expected_excluded": 27707,
  "expected_cells": 133059,
  "expected_libraries": 21,
  "excluded_GSE": [
    "GSE278342"
  ],
  "min_state_pseudobulk_cells": 50,
  "peak_half_width": 250,
  "stream_workers": 3,
  "macs_workers": 4,
  "pycisTopic_commit": "787ce422a37f5975b0ebb9e7b19eeaed44847501",
  "pycisTopic_archive_url": "https://codeload.github.com/aertslab/pycisTopic/tar.gz/787ce422a37f5975b0ebb9e7b19eeaed44847501",
  "pycisTopic_archive_SHA256": "fb9aa7cf12827220d6c063374cf28c03af0efc921126b14f1ea733c60b67b887",
  "resources": {
    "mm10.chrom.sizes": {
      "url": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes",
      "SHA256": "55463d59f400b2c8260d100f4b521800ee773fa9711fe3f6dad9c7887db2574f"
    },
    "gencode.vM25.annotation.gtf.gz": {
      "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz",
      "SHA256": "e6c9704cb5832b8a1c71431ecac9ea30057377c72cdc9a12dc5d0405ab066d5a"
    },
    "mm10-blacklist.v2.bed.gz": {
      "url": "https://raw.githubusercontent.com/Boyle-Lab/Blacklist/61a04d2c5e49341d76735d485c61f0d1177d08a8/lists/mm10-blacklist.v2.bed.gz",
      "SHA256": "febafb843c6df492f9a9fc418f8796762ee899d9864330fb509ae2d38ddc0b46"
    }
  }
}
SP_CONSENSUS_FROZEN_SOURCE_4
cat > "$SP_RUN/GEO_ASSEMBLY_EVIDENCE.json" <<'SP_CONSENSUS_FROZEN_SOURCE_5'
{
  "GSE230363": {
    "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE230363",
    "local_cached_SOFT_SHA256": "6f4d7f211cf243629460cd9a06e11846fa9b6519f9c5d29cbd7559ef6ee6bfec",
    "assembly_statements": [
      "!Sample_data_processing = Assembly: GRCm38/mm10"
    ]
  },
  "GSE255042": {
    "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255042",
    "local_cached_SOFT_SHA256": "408b6e23d31cdd437119b5720bd697a951be11e0cbd00e9afda7e64b09be9377",
    "assembly_statements": [
      "!Sample_data_processing = Assembly: GRCm38/mm10"
    ]
  },
  "GSE267584": {
    "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267584",
    "local_cached_SOFT_SHA256": "5b448296c992deefec3ce5e52bea37eec8a13860624e74aa4dc1bb6db49f3827",
    "assembly_statements": [
      "!Sample_data_processing = Assembly: mm10"
    ]
  },
  "GSE290618": {
    "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290618",
    "local_cached_SOFT_SHA256": "4be12edf4c84a478db008c9a6e7ca651926c0dd8aa239eeefebbb75932c4c5cf",
    "assembly_statements": [
      "!Sample_data_processing = Assembly: mm10"
    ]
  }
}
SP_CONSENSUS_FROZEN_SOURCE_5
cat > "$SP_RUN/SOURCES_SHA256.json" <<'SP_CONSENSUS_FROZEN_SOURCE_6'
{
  "scplus_consensus_qc.py": "f8bc0a4940a24198b1ea14c81f6f35b9c99222e16e66b53881186a7bbbc6f4dd",
  "smoke_test_qc.py": "513f8f1b9ea37b1756923d75edfc53eb1fb29f044fcf2e130ef75da63ce69762",
  "requirements_qc.txt": "8479b750bf49c2cd0ca2883f2e05f86cb01ea37d5edad963ebe24f829287dd09",
  "README_RUN.md": "4bcea6b1f8001a9a6eda5b2d13a754b1440772eb34be99c20cf3ca94675100f5",
  "SETTINGS.json": "339169ff5f982e833d3213b3c9cd72910b0ae944e832fa58d02a18f4858f0cec",
  "GEO_ASSEMBLY_EVIDENCE.json": "7513f64b2cdfea63f622805ecc9c51353cff97a2b8d60ddf45f11a563119832b"
}
SP_CONSENSUS_FROZEN_SOURCE_6

printf '%s\n' "$SP_BASE" > "$SP_RUN/PROJECT_BASE.txt"
cp "$SP_PREP/RUN_MANIFEST.json" "$SP_RUN/PREPARATION_MANIFEST_SNAPSHOT.json"
python3 - "$SP_RUN" "$SP_BASE" "$SP_PREP" <<'SP_PATHS'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1]);base=sys.argv[2]
(run/'INPUT_PATHS.json').write_text(json.dumps(dict(base=base,preparation_run=sys.argv[3],raw=base+'/scenicplus_inputs/00_raw_geo'),indent=2)+'\n')
SP_PATHS
cat > "$SP_RUN/RUN_ON_COMPUTE.sh" <<'SP_COMPUTE'
#!/usr/bin/env bash
set -euo pipefail
SP_RUN="$(cd -- "${1:?Pass the saved run folder}" && pwd)"
SP_ROOT="$(dirname -- "$(dirname -- "$SP_RUN")")"
SP_BASE="$(cat "$SP_RUN/PROJECT_BASE.txt")"
test -n "${SLURM_JOB_ID:-}" || { echo 'Use SUBMIT_OR_RESUME.sh.' >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export POLARS_MAX_THREADS=8 ARROW_NUM_THREADS=8 MPLBACKEND=Agg PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
SP_ENV="$SP_BASE/scenicplus_inputs/envs/scplus_fragment_qc_py311_v1"
export MAMBA_ROOT_PREFIX="$SP_BASE/scenicplus_inputs/mamba"
export PIP_CACHE_DIR="$SP_ROOT/cache/pip" MPLCONFIGDIR="$SP_ROOT/cache/matplotlib"
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$SP_BASE/scenicplus_inputs/envs" "$SP_RUN/software"
sp_log() {
  printf '\n- %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$SP_ROOT/README.md" >> "$SP_RUN/RUN_ACTIVITY.md"
}
trap 'SP_RC=$?; if [ "$SP_RC" -ne 0 ]; then sp_log "STOPPED job $SLURM_JOB_ID exit $SP_RC; resume saved run $SP_RUN. No final QC selection or network inference completed."; fi' EXIT
exec 9>"$SP_RUN/.compute.lock"
flock -n 9 || { echo 'This run is already computing.' >&2; exit 1; }
sp_log "START job $SLURM_JOB_ID; $SP_RUN. W8/W9 excluded. Frozen methods README_RUN.md."
echo "RUN: $SP_RUN"
echo "Job $SLURM_JOB_ID; ${SLURM_CPUS_PER_TASK:-unknown} CPUs allocated; 3 fragment streams / 4 MACS2 peak calls; QC one capture at a time, up to 8 data threads."
SP_MAMBA=''
for SP_M in "$SP_BASE/immgen/immgent_annotation_betty/bin/micromamba" "$SP_BASE/software/micromamba/bin/micromamba" "$SP_BASE/software/bin/micromamba"; do
  if [ -x "$SP_M" ]; then SP_MAMBA="$SP_M"; break; fi
done
if [ -z "$SP_MAMBA" ] && command -v micromamba >/dev/null; then SP_MAMBA="$(command -v micromamba)"; fi
test -n "$SP_MAMBA" || { echo 'Standalone micromamba missing; share this log.' >&2; exit 1; }
(
  flock -w 180 8 || { echo 'Environment setup is locked by another job.' >&2; exit 1; }
  if [ ! -f "$SP_ENV/SCPLUS_FRAGMENT_ENV_READY_v1" ]; then
    if [ -f "$SP_ENV/conda-meta/history" ]; then SP_OP=install; else SP_OP=create; fi
    "$SP_MAMBA" "$SP_OP" -y -p "$SP_ENV" --override-channels -c conda-forge -c bioconda --strict-channel-priority \
      'python=3.11.8' 'pip=24.3.1' 'setuptools=75.6.0' 'wheel=0.45.1' 'numpy=1.26.4' 'macs2=2.2.9.1'
    "$SP_ENV/bin/python" -m pip install -r "$SP_RUN/requirements_qc.txt"
    "$SP_ENV/bin/python" - "$SP_RUN" <<'SP_FETCH_CODE'
import hashlib,json,pathlib,urllib.request,sys
run=pathlib.Path(sys.argv[1]);settings=json.loads((run/'SETTINGS.json').read_text())
p=run/'software/pycisTopic.tar.gz'
if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=settings['pycisTopic_archive_SHA256']:
    q=p.with_name(p.name+'.partial')
    with urllib.request.urlopen(settings['pycisTopic_archive_url'],timeout=120) as f,q.open('wb') as o:o.write(f.read())
    assert hashlib.sha256(q.read_bytes()).hexdigest()==settings['pycisTopic_archive_SHA256'],'pycisTopic source checksum mismatch'
    q.replace(p)
SP_FETCH_CODE
    # Only metric/consensus imports are installed here; later network/topic tools get their own environment.
    "$SP_ENV/bin/python" -m pip install --no-deps --no-warn-conflicts --no-build-isolation "$SP_RUN/software/pycisTopic.tar.gz" 'pyscenic==0.12.1'
    "$SP_ENV/bin/python" -u "$SP_RUN/smoke_test_qc.py"
    "$SP_ENV/bin/macs2" --version
    cp "$SP_RUN/requirements_qc.txt" "$SP_ENV/SCPLUS_FRAGMENT_REQUIREMENTS.txt"
    "$SP_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pycisTopic_commit"])' "$SP_RUN/SETTINGS.json" > "$SP_ENV/SCPLUS_FRAGMENT_CODE_COMMIT.txt"
    touch "$SP_ENV/SCPLUS_FRAGMENT_ENV_READY_v1"
  fi
) 8>"$SP_BASE/scenicplus_inputs/envs/.fragment_qc_env.lock"
cmp -s "$SP_RUN/requirements_qc.txt" "$SP_ENV/SCPLUS_FRAGMENT_REQUIREMENTS.txt" || { echo 'Environment requirements differ; use a new environment version.' >&2; exit 1; }
eval "$("$SP_MAMBA" shell hook -s bash)"
micromamba activate "$SP_ENV"
unset PYTHONPATH PYTHONHOME
"$SP_ENV/bin/python" - "$SP_RUN" "$SP_ENV" <<'SP_CHECK_ENV'
import importlib.metadata,json,pathlib,sys
run,env=map(pathlib.Path,sys.argv[1:])
for line in (run/'requirements_qc.txt').read_text().splitlines():
    package,version=line.strip().split('==')
    assert importlib.metadata.version(package)==version, 'Environment package version changed: '+package
assert (env/'SCPLUS_FRAGMENT_CODE_COMMIT.txt').read_text().strip()==json.loads((run/'SETTINGS.json').read_text())['pycisTopic_commit'], 'Environment pycisTopic commit changed'
SP_CHECK_ENV
"$SP_ENV/bin/python" -u "$SP_RUN/smoke_test_qc.py"
"$SP_MAMBA" list -p "$SP_ENV" --explicit > "$SP_RUN/ENVIRONMENT_CONDA_EXPLICIT.txt"
"$SP_ENV/bin/python" -m pip freeze > "$SP_RUN/ENVIRONMENT_PIP_FREEZE.txt"
"$SP_ENV/bin/python" -u "$SP_RUN/scplus_consensus_qc.py" "$SP_RUN"
printf '%s\n' "$SP_RUN" > "$SP_ROOT/latest_complete_run.txt"
sp_log "COMPLETE job $SLURM_JOB_ID; shared peaks + ATAC QC ready. Final QC thresholds, topics, motifs and network inference remain. $SP_RUN"
echo "Share: $SP_RUN/SHARE_SCENICPLUS_CONSENSUS_QC.zip"
SP_COMPUTE
cat > "$SP_RUN/SUBMIT_OR_RESUME.sh" <<'SP_SUBMIT'
#!/usr/bin/env bash
set -euo pipefail
SP_RUN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SP_ROOT="$(dirname -- "$(dirname -- "$SP_RUN")")"
exec 9>"$SP_RUN/.submission.lock"
flock -n 9 || { echo 'Submission already in progress.' >&2; exit 1; }
if [ -f "$SP_RUN/RUN_MANIFEST.json" ] && [ -f "$SP_RUN/SHARE_SCENICPLUS_CONSENSUS_QC.zip" ]; then
  echo "This run has completed. Reports: $SP_RUN/SHARE_SCENICPLUS_CONSENSUS_QC.zip"; exit 0
fi
if [ -f "$SP_RUN/SLURM_JOB_ID.txt" ]; then
  SP_OLD="$(cat "$SP_RUN/SLURM_JOB_ID.txt")"
  [[ "$SP_OLD" =~ ^[0-9]+$ ]] || { echo 'Malformed job ID.' >&2; exit 1; }
  if SP_QUEUE="$(squeue -h -j "$SP_OLD" -o '%i' 2>"$SP_RUN/last_squeue.err")"; then
    if [ -n "$SP_QUEUE" ]; then echo "Already queued/running: $SP_OLD"; exit 0; fi
  elif ! grep -qi 'invalid job id' "$SP_RUN/last_squeue.err"; then cat "$SP_RUN/last_squeue.err" >&2; exit 1
  fi
fi
SP_OPTS=(--parsable --job-name=SCplus_ATAC_QC --partition=genoa-std-mem --nodes=1 --ntasks=1 \
  --cpus-per-task=24 --mem-per-cpu=5632M --time=24:00:00 --chdir="$SP_RUN" \
  --output="$SP_RUN/slurm_%j.out" --error="$SP_RUN/slurm_%j.err" --export=ALL)
if [ -n "${SCPLUS_SLURM_ACCOUNT:-${IMMGENT_SLURM_ACCOUNT:-}}" ]; then SP_OPTS+=(--account="${SCPLUS_SLURM_ACCOUNT:-$IMMGENT_SLURM_ACCOUNT}"); fi
SP_JOB="$(sbatch "${SP_OPTS[@]}" "$SP_RUN/RUN_ON_COMPUTE.sh" "$SP_RUN")"; SP_JOB="${SP_JOB%%;*}"
[[ "$SP_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $SP_JOB" >&2; exit 1; }
printf '%s\n' "$SP_JOB" > "$SP_RUN/SLURM_JOB_ID.txt"
printf '\n- %s Submitted consensus/QC job %s; %s.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SP_JOB" "$SP_RUN" >> "$SP_ROOT/README.md"
echo "Submitted consensus peaks + ATAC QC job: $SP_JOB"
echo "Check: sacct -j $SP_JOB --format=JobID,State,Elapsed,ExitCode"
echo "Logs: tail -n 40 \"$SP_RUN/slurm_${SP_JOB}.out\" \"$SP_RUN/slurm_${SP_JOB}.err\""
echo "Resume: bash \"$SP_RUN/SUBMIT_OR_RESUME.sh\""
echo "Run folder: $SP_RUN"
SP_SUBMIT
python3 - "$SP_RUN" <<'SP_FREEZE'
import hashlib,json,pathlib,sys
run=pathlib.Path(sys.argv[1]);d=json.loads((run/'SOURCES_SHA256.json').read_text())
for n in ['RUN_ON_COMPUTE.sh','SUBMIT_OR_RESUME.sh','PROJECT_BASE.txt','INPUT_PATHS.json','PREPARATION_MANIFEST_SNAPSHOT.json']:
    d[n]=hashlib.sha256((run/n).read_bytes()).hexdigest()
(run/'SOURCES_SHA256.json').write_text(json.dumps(d,indent=2)+'\n')
SP_FREEZE
printf '%s\n' "$SP_RUN" > "$SP_ROOT/latest_started_run.txt"
printf '\n- %s Created consensus/QC run %s from %s. Scope: 133059 CD8 candidates, 21 captures; W8/W9 excluded only here.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SP_RUN" "$SP_PREP" >> "$SP_ROOT/README.md"
bash "$SP_RUN/SUBMIT_OR_RESUME.sh"
SCPLUS_CONSENSUS_INSTALL
