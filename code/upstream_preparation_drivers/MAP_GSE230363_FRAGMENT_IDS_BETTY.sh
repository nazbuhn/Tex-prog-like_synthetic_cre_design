#!/usr/bin/env bash
# Run this saved file with Bash from the Betty login terminal.
set -euo pipefail
PAIR_BASE='/vast/projects/saargill/cell-type-specific/nazbuhn'
PAIR_RAW="$PAIR_BASE/scenicplus_inputs/00_raw_geo"
PAIR_TRANSFER="$PAIR_RAW/_transfer_audits/ATAC_upload_20261020T192733_88495"
PAIR_SOURCE="$PAIR_BASE/rbi2_upload_jobs_data/TRBI_ANNOTATED_RNA/runs/import_20261020T171724_1179750"
PAIR_ENV="$PAIR_BASE/rbi2_upload_jobs_data/REASSEMBLED_RNA/envs/rna_merge_py311"
PAIR_ROOT="$PAIR_BASE/scenicplus_inputs/03_GSE230363_fragment_map"
command -v sbatch >/dev/null || { echo 'Run this script from your interactive Betty login terminal.'; exit 1; }
test -f "$PAIR_TRANSFER/TRANSFER_VERIFICATION.json" && test -f "$PAIR_SOURCE/RUN_MANIFEST.json" || { echo 'Completed transfer verification or annotated RNA manifest is missing.'; exit 1; }
test -x "$PAIR_ENV/bin/python" || { echo "Existing RNA environment missing: $PAIR_ENV"; exit 1; }
mkdir -p "$PAIR_ROOT/runs"
PAIR_RUN="$PAIR_ROOT/runs/map_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$PAIR_RUN"
if [ ! -f "$PAIR_ROOT/README.md" ]; then
  printf '# GSE230363 original fragment barcode map\n\nFull-file validation of the observed LCMV _1 through _7 suffixes; save original fragment IDs without rewriting source files. Run-specific methods and reports are retained below.\n' > "$PAIR_ROOT/README.md"
fi
cat > "$PAIR_RUN/map_gse230363_fragment_ids.py" <<'LCMV_MAP_SOURCE_0'
"""Validate LCMV suffixes over complete fragment files and save original-ID maps."""
import concurrent.futures
import datetime
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


def say(*v):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *v, flush=True)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    temp = p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temp.replace(p)


def save_table(p, frame):
    temp = p.with_name(p.name+'.tmp')
    frame.to_csv(temp, sep='\t', index=False,
                 compression=dict(method='gzip', mtime=0) if p.name.endswith('.gz') else None)
    temp.replace(p)


def scan(spec, targets, run_string, signature):
    run = Path(run_string)
    key = spec['RNA_library']
    checkpoint = run/'checkpoints'/(key+'.tsv.gz')
    meta = checkpoint.with_suffix('.json')
    if checkpoint.exists() and meta.exists():
        old = json.loads(meta.read_text())
        require(old['signature']==signature and sha(checkpoint)==old['output_SHA256'], 'Checkpoint changed: '+key)
        say('REUSE COMPLETE LIBRARY', key)
        return old
    path = Path(spec['fragment_path'])
    with h5py.File(spec['matrix_path'], 'r') as f:
        matrix_barcodes = [b.decode() if isinstance(b,bytes) else str(b) for b in f['matrix/barcodes'][...]]
    require(len(matrix_barcodes)==len(set(matrix_barcodes)), 'Duplicate combined-matrix barcodes: '+key)
    require(targets.original_barcode.is_unique and targets.Cell_ID.is_unique, 'Duplicate target IDs: '+key)
    wanted = set(targets.original_barcode)
    matrix_set = set(matrix_barcodes)
    require(wanted<=matrix_set, 'Uploaded RNA barcode missing from validated combined matrix: '+key)
    require(all(re.fullmatch('[ACGT]{16}-1', b) for b in matrix_barcodes), 'Unexpected matrix barcode convention: '+key)
    suffix = '_'+str(spec['LCMV_index'])
    pattern = re.compile(r'^([ACGT]{16}-1)'+re.escape(suffix)+r'$')
    before = path.stat()
    seen, counts = {}, {}
    rows = matrix_rows = wanted_rows = 0
    say('FULL FRAGMENT SCAN', key, 'expected suffix', suffix, path.name)
    with gzip.open(path, 'rt', encoding='ascii') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.rstrip('\r\n').split('\t')
            require(len(fields)>=5, 'Unexpected fragment schema in '+key)
            original = fields[3]
            match = pattern.fullmatch(original)
            require(match is not None, 'Unexpected fragment suffix or barcode; no alternative conversion attempted: '+key+' '+original)
            start, end, support = int(fields[1]), int(fields[2]), int(fields[4])
            require(bool(fields[0]) and start>=0 and end>start and support>0, 'Invalid fragment coordinates/support: '+key)
            barcode = match.group(1)
            if barcode in seen:
                require(seen[barcode]==original, 'Nonunique fragment-to-matrix barcode mapping: '+key)
            else:
                seen[barcode] = original
            if barcode in matrix_set:
                matrix_rows += 1
            if barcode in wanted:
                if barcode not in counts:
                    counts[barcode] = [0,0]
                counts[barcode][0] += 1
                counts[barcode][1] += support
                wanted_rows += 1
            rows += 1
            if rows % 5_000_000 == 0:
                say('FRAGMENT PROGRESS', key, rows, 'rows;', len(counts), '/', len(wanted), 'uploaded barcodes found')
    after = path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), 'Fragment file changed during scan: '+key)
    require(rows>0, 'Empty fragment file: '+key)
    out = targets.copy()
    out['fragment_file'] = str(path)
    out['source_combined_matrix'] = spec['matrix_path']
    out['expected_library_suffix'] = suffix
    # Missing targets retain no invented fragment ID: candidates are stored in their own column.
    original_barcodes = out.original_barcode.astype(str)
    out['candidate_fragment_barcode'] = original_barcodes+suffix
    out['original_fragment_barcode'] = original_barcodes.map(seen).fillna('')
    out['fragment_rows'] = original_barcodes.map(lambda b: counts.get(b,[0,0])[0]).astype(np.int64)
    out['fragment_read_support_sum'] = original_barcodes.map(lambda b: counts.get(b,[0,0])[1]).astype(np.int64)
    out['fragment_pairing_status'] = np.where(out.fragment_rows.gt(0), 'VALIDATED_ORIGINAL_FRAGMENT_ID', 'NO_FRAGMENT_RECORD_FOUND')
    found = out.fragment_rows.gt(0)
    require(out.loc[found,'original_fragment_barcode'].tolist()==out.loc[found,'candidate_fragment_barcode'].tolist(),
            'Observed and expected fragment IDs differ: '+key)
    require(out.loc[found,'original_fragment_barcode'].is_unique, 'Two uploaded cells map to one fragment ID: '+key)
    save_table(checkpoint, out)
    info = dict(signature=signature, output_SHA256=sha(checkpoint), source_GSE='GSE230363', source_library=key,
                LCMV_index=spec['LCMV_index'], expected_fragment_suffix=suffix, uploaded_cells=len(targets),
                fragment_matched_cells=int(found.sum()), missing_fragment_cells=int((~found).sum()),
                original_matrix_cells=len(matrix_barcodes), total_fragment_records=rows,
                total_unique_original_fragment_barcodes=len(seen),
                original_matrix_barcodes_with_fragments=len(matrix_set & set(seen)),
                fragment_records_for_matrix_barcodes=matrix_rows, fragment_records_for_uploaded_cells=wanted_rows,
                barcode_collisions=0, all_fragment_barcodes_follow_exact_expected_suffix=True,
                fragment_file=str(path), matrix_path=spec['matrix_path'], source_files_unchanged=True)
    dump(meta, info)
    say('LIBRARY COMPLETE', key, int(found.sum()), '/', len(targets), 'uploaded cells mapped')
    return info


def main(run):
    cfg = json.loads((run/'INPUT_PATHS.json').read_text())
    settings = json.loads((run/'SETTINGS.json').read_text())
    frozen = json.loads((run/'SOURCES_SHA256.json').read_text())
    for name, want in frozen.items():
        require(sha(run/name)==want, 'Saved code/settings changed: '+name)
    raw, audit, source = (Path(cfg[k]) for k in ['raw','transfer_audit','annotated_run'])
    prior = json.loads((source/'RUN_MANIFEST.json').read_text())
    h5ad = source/'ALL_UPLOADED_CELLS_WITH_TRBI.h5ad'
    source_sha = sha(h5ad)
    require(prior['status']=='COMPLETE' and source_sha==prior['output_SHA256'][h5ad.name], 'Annotated RNA changed since import')
    require(prior['cells']==settings['all_uploaded_cells'], 'Unexpected source RNA scope')
    plan = json.loads((audit/'UPLOAD_PLAN.json').read_text())
    checksums = json.loads((audit/'FILES_SHA256.json').read_text())
    with h5py.File(h5ad,'r') as f:
        obs = read_elem(f['obs'])
    require(obs.Cell_ID.astype(str).tolist()==obs.index.astype(str).tolist(), 'Cell_ID/index mismatch')
    target = obs.loc[obs.source_GSE.astype(str).eq('GSE230363')].copy()
    require(len(target)==settings['GSE230363_cells'], 'Unexpected GSE230363 cell count')
    expected = settings['library_counts']
    actual = target.groupby('source_library',observed=True).size().to_dict()
    require(actual==expected, 'GSE230363 source library counts differ from the frozen RNA export')
    require(not target.duplicated(['source_library','original_barcode']).any(), 'Duplicate uploaded capture/barcode IDs')
    specs, input_hashes = [], {}
    for item in plan['matrices']:
        if item['source_GSE']!='GSE230363':
            continue
        matrix = raw/item['matrix']
        match = re.search(r'_LCMV([1-7])_filtered_feature_bc_matrix\.h5$',matrix.name)
        require(match is not None, 'Unexpected GSE230363 matrix filename')
        fragment = matrix.with_name(matrix.name.replace('_filtered_feature_bc_matrix.h5','_atac_fragments.tsv.gz'))
        for p in [matrix,fragment]:
            relative = p.relative_to(raw).as_posix()
            record = checksums[relative]
            say('VERIFY INPUT',p.name)
            require(p.stat().st_size==record['bytes'] and sha(p)==record['SHA256'], 'Input differs from verified transfer: '+relative)
            input_hashes[relative] = record['SHA256']
        specs.append(dict(RNA_library=item['RNA_library'],LCMV_index=int(match.group(1)),
                          matrix_path=str(matrix),fragment_path=str(fragment)))
    require(len(specs)==7 and {s['RNA_library'] for s in specs}==set(expected) and
            {s['LCMV_index'] for s in specs}==set(range(1,8)), 'Expected all seven distinct LCMV libraries')
    inputs = dict(annotated_h5ad_SHA256=source_sha, input_SHA256=input_hashes, source_SHA256=frozen,
                  input_paths_SHA256=sha(run/'INPUT_PATHS.json'))
    signature = hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    if (run/'MAPPING_INPUTS.json').exists():
        require(json.loads((run/'MAPPING_INPUTS.json').read_text())==inputs, 'Inputs changed; use a new run')
    else:
        dump(run/'MAPPING_INPUTS.json',inputs)
    if (run/'RUN_MANIFEST.json').exists():
        manifest = json.loads((run/'RUN_MANIFEST.json').read_text())
        require(manifest['signature']==signature, 'Completed run signature changed')
        for name,want in manifest['output_SHA256'].items():
            require(sha(run/name)==want, 'Completed mapping output changed: '+name)
        say('ALREADY COMPLETE; all saved maps verified')
        return
    (run/'checkpoints').mkdir(exist_ok=True)
    summaries = []
    columns = ['Cell_ID','source_GSE','source_library','original_barcode','sample_ID','upload_job',
               'trbi_import_status','trbi_submission_scope','trbi_level1','trbi_level2','trbi_confidence_score','trbi_discovery']
    target = target[[c for c in columns if c in target]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(cfg.get('fragment_workers',4))) as pool:
        tasks = [pool.submit(scan,s,target.loc[target.source_library.astype(str).eq(s['RNA_library'])],str(run),signature) for s in specs]
        for future in concurrent.futures.as_completed(tasks):
            summaries.append(future.result())
    parts = [pd.read_csv(run/'checkpoints'/(s['RNA_library']+'.tsv.gz'),sep='\t',keep_default_na=False) for s in specs]
    mapping = pd.concat(parts,ignore_index=True).set_index('Cell_ID').reindex(target.index)
    require(mapping.index.equals(target.index) and mapping.original_barcode.tolist()==target.original_barcode.tolist(), 'Cell order/accounting mismatch')
    mapping.insert(0,'Cell_ID',mapping.index.astype(str))
    known = mapping.original_fragment_barcode.ne('')
    require(not mapping.loc[known].duplicated(['fragment_file','original_fragment_barcode']).any(), 'Fragment-to-cell map is not one-to-one')
    for c in target:
        # TSV serialisation may represent numeric missingness as an empty string; exact ID and label strings are preserved.
        if c not in ['trbi_confidence_score','Cell_ID']:
            require(mapping[c].astype(str).tolist()==target[c].astype(str).tolist(), 'Source metadata changed: '+c)
    save_table(run/'GSE230363_ORIGINAL_FRAGMENT_CELL_ID_MAP.tsv.gz',mapping)
    summary = pd.DataFrame(summaries).sort_values('LCMV_index')
    save_table(run/'GSE230363_FRAGMENT_MAPPING_SUMMARY.tsv',summary.drop(columns=['signature','output_SHA256']))
    report = dict(status='COMPLETE',selected_cells=len(mapping),libraries=7,
                  matched_cells=int(known.sum()),missing_cells=int((~known).sum()),
                  mapping_result='ALL_SELECTED_CELLS_MAPPED' if known.all() else 'REVIEW_MISSING_FRAGMENT_CELLS',
                  all_fragment_records_checked=True,barcode_collisions=0,
                  source_RNA_ATAC_matrices_fragments_indexes_labels_unchanged=True,
                  no_new_cell_filtering_or_SCENICPLUS_run=True,
                  matrix_pairing_evidence='Prior audit verified exact full RNA counts and positive paired peak counts for every target cell',
                  mapping_direction='Original fragment barcode (including _LCMVindex suffix), within original fragment file, to exact uploaded Cell_ID',
                  original_fragment_files_rewritten=False,original_indexes_preserved=True)
    dump(run/'MAPPING_SUMMARY.json',report)
    names = ['GSE230363_ORIGINAL_FRAGMENT_CELL_ID_MAP.tsv.gz','GSE230363_FRAGMENT_MAPPING_SUMMARY.tsv','MAPPING_SUMMARY.json']
    dump(run/'RUN_MANIFEST.json',dict(status='COMPLETE',signature=signature,output_SHA256={n:sha(run/n) for n in names}))
    with zipfile.ZipFile(run/'SHARE_GSE230363_FRAGMENT_MAPPING.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names[1:]+['RUN_MANIFEST.json','README_RUN.md','MAPPING_INPUTS.json']:
            z.write(run/name,name)
    with open(run.parent.parent/'README.md','a') as f:
        f.write('\n- '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' Completed full GSE230363 suffix-map check: '+str(int(known.sum()))+'/'+str(len(mapping))+' selected cells found. Sources unchanged; see '+str(run)+'\n')
    say('COMPLETE',json.dumps(report))
    print(summary[['source_library','uploaded_cells','fragment_matched_cells','missing_fragment_cells','expected_fragment_suffix']].to_string(index=False),flush=True)
    say('SHARE',run/'SHARE_GSE230363_FRAGMENT_MAPPING.zip')


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
LCMV_MAP_SOURCE_0
cat > "$PAIR_RUN/README_RUN.md" <<'LCMV_MAP_SOURCE_1'
# GSE230363 original fragment ID map

The completed RNA–ATAC audit verified paired peak matrices and exact complete RNA count vectors for all 161,773 uploaded cells. GSE230363 accounts for 72,362 of them. Its exact fragment-string lookup failed because the fragment files carry an extra library suffix: matrix barcodes end in `-1`, whereas the sampled fragment IDs end in `-1_1` through `-1_7` for LCMV1 through LCMV7.

This suffix difference was observed in the actual HPC format-probe report, not inferred from example synthetic data. The probe read 50,000 fragment records per library; it did not establish complete coverage. Its earlier diagnostic regex removed terminal `-digits` only, so it did not test the newly observed `_digits` suffix. Neither a barcode-sequence translation nor a reverse complement is indicated by the suffix finding; neither is attempted here.

## Validation and outputs

The next job streams all seven original GSE230363 fragment files through EOF. Every barcode must match exactly sixteen A/C/G/T bases followed by `-1_<expected LCMV index>`. Unknown formats stop the job. It verifies the suffix index against the original combined-matrix filename, checks the complete barcode transformation is one-to-one within the library, and checks fragment coordinate/support fields. Background barcodes absent from the filtered matrix are allowed and counted, not assigned to uploaded cells.

Each uploaded cell must have its exact original matrix barcode in its own combined H5. After suffix handling, fragment coverage is reported separately for every selected cell. Missing cells remain in the output with an explicit status and no invented observed fragment ID. No cell is silently removed. `COMPLETE` means the check finished; `mapping_result` distinguishes complete target coverage from targets needing review.

Outputs include `GSE230363_ORIGINAL_FRAGMENT_CELL_ID_MAP.tsv.gz`, a row for every one of the 72,362 selected cells in original RNA order. It records full uploaded Cell_ID, original matrix barcode, original fragment barcode (including its `_n` suffix), original fragment path, library, online labels, fragment-record count and fifth-column support sum. Counts refer to supplied records and support, not independently deduplicated molecules. The map preserves library boundaries and is designed for downstream label alignment using original fragment IDs. It never creates a global bare-barcode join.

The original fragment files are **not rewritten**, compressed differently, or reindexed. Existing fragments, indexes, combined matrices, annotated RNA and labels are unchanged. The job creates a map and reports only; it does not start SCENIC+, create a consensus peak set or topic model, choose a new QC/confidence cutoff, or decide the other series' outstanding issues.

`GSE230363_FRAGMENT_MAPPING_SUMMARY.tsv` gives whole-file coverage by library; `MAPPING_SUMMARY.json` gives overall coverage; `RUN_MANIFEST.json` records output checksums. `SHARE_GSE230363_FRAGMENT_MAPPING.zip` contains aggregate reports and methods, excluding the larger per-cell map.

## Reproducibility and resources

The job verifies the annotated H5AD against its successful import's SHA256, all seven matrix and fragment inputs against the frozen transfer hashes, and the expected library cell counts against the frozen export. The source/settings and input signature are saved. Each complete library scan has an atomic hashed checkpoint. The saved `SUBMIT_OR_RESUME.sh` resumes a stopped run without repeating complete library scans; inputs are revalidated first. Changing sources or inputs requires a new run. Submission and compute locks prevent two jobs operating on the same saved run. Project READMEs record preparation, submission, start, failure or completion.

It uses the existing project RNA environment without installation: four CPU workers, four allocated CPUs, 5632 MiB per CPU (22 GiB), two-hour limit on genoa-std-mem. It reads only RNA metadata from the annotated H5AD; it does not load the large normalized expression or raw-count matrices. The prior complete RNA-count GSE230363 fragment-map check is retained as the matrix-identity evidence.
LCMV_MAP_SOURCE_1
cat > "$PAIR_RUN/SETTINGS.json" <<'LCMV_MAP_SOURCE_2'
{
  "all_uploaded_cells": 161773,
  "GSE230363_cells": 72362,
  "library_counts": {
    "GSM7220513": 9521,
    "GSM7220515": 16931,
    "GSM7220517": 8042,
    "GSM7220519": 9355,
    "GSM7220521": 10625,
    "GSM7220523": 9052,
    "GSM7220525": 8836
  }
}
LCMV_MAP_SOURCE_2
cat > "$PAIR_RUN/SOURCES_SHA256.json" <<'LCMV_MAP_SOURCE_3'
{
  "map_gse230363_fragment_ids.py": "85f0dcbf571d69c6c4b8d6a151449a3f3eefaf7a125d6844e650d75939b6bbae",
  "README_RUN.md": "99e21879ce8d7ba53861293c8f6a5e1c885382b41f5483a610a3aa31c458625f",
  "SETTINGS.json": "37f00c789cdc334c0dac21a18620c39d1dc5366ffc081f4533369c77ff41df65"
}
LCMV_MAP_SOURCE_3

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
trap 'PAIR_RC=$?; if [ "$PAIR_RC" -ne 0 ]; then printf "\n- %s GSE230363 fragment-map check stopped with exit %s; run %s. Completed checkpoints retained; source data and labels preserved.\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_RC" "$PAIR_RUN" >> "$PAIR_ROOT/README.md"; echo "Stopped. Share log tails; resume with $PAIR_RUN/SUBMIT_OR_RESUME.sh"; fi' EXIT
PAIR_BASE="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["base"])' "$PAIR_RUN/INPUT_PATHS.json")"
PAIR_ENV="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["env"])' "$PAIR_RUN/INPUT_PATHS.json")"
PAIR_MAMBA="$PAIR_BASE/immgent_annotation_betty/bin/micromamba"
export MAMBA_ROOT_PREFIX="$PAIR_BASE/rbi2_upload_jobs_data/REASSEMBLED_RNA/mamba"
eval "$("$PAIR_MAMBA" shell hook -s bash)"
micromamba activate "$PAIR_ENV"
unset PYTHONPATH PYTHONHOME
printf '\n- %s Started/resumed GSE230363 fragment-map check job %s; run %s. Full fragment scans validate the observed LCMV suffix rule and exact uploaded cell mapping; no cell filtering.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "$PAIR_RUN" >> "$PAIR_ROOT/README.md"
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
"$PAIR_ENV/bin/python" -u "$PAIR_RUN/map_gse230363_fragment_ids.py" "$PAIR_RUN"
printf '%s\n' "$PAIR_RUN" > "$PAIR_ROOT/latest_complete_run.txt"
echo "Share: $PAIR_RUN/SHARE_GSE230363_FRAGMENT_MAPPING.zip"
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
PAIR_OPTS=(--parsable --job-name=LCMV_fragment_map --partition=genoa-std-mem --nodes=1 --ntasks=1
  --cpus-per-task=4 --mem-per-cpu=5632M --time=02:00:00 --chdir="$PAIR_RUN"
  --output="$PAIR_RUN/slurm_%j.out" --error="$PAIR_RUN/slurm_%j.err" --export=ALL)
if [ -n "${IMMGENT_SLURM_ACCOUNT:-}" ]; then PAIR_OPTS+=(--account="$IMMGENT_SLURM_ACCOUNT"); fi
PAIR_JOB="$(sbatch "${PAIR_OPTS[@]}" "$PAIR_RUN/RUN_ON_COMPUTE.sh" "$PAIR_RUN")"
PAIR_JOB="${PAIR_JOB%%;*}"
[[ "$PAIR_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected submission response: $PAIR_JOB"; exit 1; }
printf '%s\n' "$PAIR_JOB" > "$PAIR_RUN/SLURM_JOB_ID.txt"
printf '\n- %s Submitted GSE230363 fragment-map check job %s; run %s.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_JOB" "$PAIR_RUN" >> "$(dirname -- "$(dirname -- "$PAIR_RUN")")/README.md"
echo "Submitted GSE230363 fragment-map check job: $PAIR_JOB"
echo "Check: squeue -j $PAIR_JOB"
echo "Logs: tail -n 40 \"$PAIR_RUN/slurm_${PAIR_JOB}.out\" \"$PAIR_RUN/slurm_${PAIR_JOB}.err\""
echo "Resume: bash \"$PAIR_RUN/SUBMIT_OR_RESUME.sh\""
echo "Run folder: $PAIR_RUN"
PAIR_SUBMIT
printf '%s\n' "$PAIR_RUN" > "$PAIR_ROOT/latest_started_run.txt"
printf '\n- %s Prepared GSE230363 fragment-map check %s. Source RNA: %s. Transfer audit: %s. See README_RUN.md.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_RUN" "$PAIR_SOURCE" "$PAIR_TRANSFER" >> "$PAIR_ROOT/README.md"
bash "$PAIR_RUN/SUBMIT_OR_RESUME.sh"
