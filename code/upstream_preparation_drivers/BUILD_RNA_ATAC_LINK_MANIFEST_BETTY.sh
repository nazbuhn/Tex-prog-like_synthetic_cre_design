#!/usr/bin/env bash
# Run this saved file with Bash from the Betty login terminal.
set -euo pipefail
PAIR_BASE='/vast/projects/saargill/cell-type-specific/nazbuhn'
PAIR_RAW="$PAIR_BASE/scenicplus_inputs/00_raw_geo"
PAIR_TRANSFER="$PAIR_RAW/_transfer_audits/ATAC_upload_20261020T192733_88495"
PAIR_SOURCE="$PAIR_BASE/rbi2_upload_jobs_data/TRBI_ANNOTATED_RNA/runs/import_20261020T171724_1179750"
PAIR_ENV="$PAIR_BASE/rbi2_upload_jobs_data/REASSEMBLED_RNA/envs/rna_merge_py311"
PAIR_ROOT="$PAIR_BASE/scenicplus_inputs/04_cell_link_manifest"
command -v sbatch >/dev/null || { echo 'Run this script from your interactive Betty login terminal.'; exit 1; }
test -f "$PAIR_TRANSFER/TRANSFER_VERIFICATION.json" && test -f "$PAIR_SOURCE/RUN_MANIFEST.json" || { echo 'Completed transfer verification or annotated RNA manifest is missing.'; exit 1; }
test -x "$PAIR_ENV/bin/python" || { echo "Existing RNA environment missing: $PAIR_ENV"; exit 1; }
mkdir -p "$PAIR_ROOT/runs"
PAIR_RUN="$PAIR_ROOT/runs/links_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir "$PAIR_RUN"
if [ ! -f "$PAIR_ROOT/README.md" ]; then
  printf '# Unified RNA–ATAC cell-link manifest\n\nCombine verified original fragment IDs and inspect cached GSE267584 capture-depth profiles. Preserve all uploaded cells. Run-specific methods and reports are retained below.\n' > "$PAIR_ROOT/README.md"
fi
cat > "$PAIR_RUN/build_atac_cell_manifest.py" <<'ATAC_LINK_SOURCE_0'
"""Combine validated links and compare cached ATAC depth profiles, without raw-fragment rescans."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def require(ok, message):
    if not ok:
        raise ValueError(message)


def say(*v):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *v, flush=True)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    t=path.with_name(path.name+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    t.replace(path)


def table(path, frame):
    t=path.with_name(path.name+'.tmp')
    frame.to_csv(t,sep='\t',index=False,compression=dict(method='gzip',mtime=0) if path.name.endswith('.gz') else None)
    t.replace(path)


def checked_table(folder, name, manifest):
    require(sha(folder/name)==manifest['output_SHA256'][name], 'Saved output changed: '+str(folder/name))
    return pd.read_csv(folder/name,sep='\t',keep_default_na=False)


def boolean(frame, name):
    require(frame[name].isin([True,False]).all(), 'Invalid boolean column: '+name)
    return frame[name].astype(bool)


def combine(old, lcmv, raw, rna_path, expected_lcmv):
    require(old.Cell_ID.is_unique and lcmv.Cell_ID.is_unique, 'Duplicate cell IDs')
    source=old.set_index('Cell_ID',drop=False)
    new=lcmv.set_index('Cell_ID',drop=False)
    scope=source.source_GSE.eq('GSE230363')
    require(set(new.index)==set(source.index[scope]) and len(new)==expected_lcmv, 'LCMV/source cell scope differs')
    require(boolean(source,'matrix_pairing_verified').all(), 'Some original matrix pairs were not verified')
    require(new.fragment_pairing_status.eq('VALIDATED_ORIGINAL_FRAGMENT_ID').all(), 'LCMV full-file map has missing cells')
    for c in ['source_GSE','source_library','original_barcode','sample_ID','upload_job','trbi_import_status',
              'trbi_submission_scope','trbi_level1','trbi_level2','trbi_discovery']:
        if c in new and c in source:
            require(source.loc[new.index,c].astype(str).tolist()==new[c].astype(str).tolist(), 'LCMV/source metadata differ: '+c)
    out=source.copy()
    out['RNA_h5ad_path']=str(rna_path)
    out['fragment_file']=''
    out['fragment_barcode']=''
    out['fragment_record_count']=np.zeros(len(out),dtype=np.int64)
    out['fragment_read_support_sum']=np.zeros(len(out),dtype=np.int64)
    out['fragment_link_status']='NO_VALIDATED_FRAGMENT_LINK'
    accepted=boolean(source,'matrix_and_fragment_pairing_verified')
    for i in out.index[accepted]:
        relative=str(source.at[i,'candidate_fragment_file'])
        require(relative and not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe/missing candidate fragment path')
        out.at[i,'fragment_file']=str(raw/relative)
    out.loc[accepted,'fragment_barcode']=out.loc[accepted,'original_barcode'].to_numpy()
    out.loc[accepted,'fragment_record_count']=out.loc[accepted,'candidate_fragment_rows'].to_numpy(dtype=np.int64)
    out.loc[accepted,'fragment_read_support_sum']=out.loc[accepted,'candidate_fragment_read_support_sum'].to_numpy(dtype=np.int64)
    out.loc[accepted,'fragment_link_status']='VALIDATED_ORIGINAL_FRAGMENT_ID'
    out.loc[new.index,'fragment_file']=new.fragment_file.to_numpy()
    out.loc[new.index,'fragment_barcode']=new.original_fragment_barcode.to_numpy()
    out.loc[new.index,'fragment_record_count']=new.fragment_rows.to_numpy(dtype=np.int64)
    out.loc[new.index,'fragment_read_support_sum']=new.fragment_read_support_sum.to_numpy(dtype=np.int64)
    out.loc[new.index,'fragment_link_status']='VALIDATED_ORIGINAL_FRAGMENT_ID'
    unresolved=source.fragment_assignment_status.eq('UNRESOLVED_CONDITION_CONFLICT_CANDIDATE_ONLY')
    out.loc[unresolved,'fragment_link_status']='CONDITION_CONFLICT_CANDIDATE_ONLY'
    absent=source.fragment_assignment_status.eq('NO_FRAGMENT_FILE_IN_UPLOAD')
    out.loc[absent,'fragment_link_status']='MATRIX_PAIRED_FRAGMENT_FILE_NOT_IN_UPLOAD'
    out['fragment_link_validated']=out.fragment_link_status.eq('VALIDATED_ORIGINAL_FRAGMENT_ID')
    out['working_CD8_with_validated_fragment_link']=boolean(out,'working_CD8_scope') & out.fragment_link_validated
    linked=out.fragment_link_validated
    require(out.loc[linked,'fragment_file'].ne('').all() and out.loc[linked,'fragment_barcode'].ne('').all(), 'Validated link lacks original IDs')
    require(out.loc[linked,'fragment_record_count'].gt(0).all(), 'Validated link lacks fragment records')
    require(not out.loc[linked].duplicated(['fragment_file','fragment_barcode']).any(), 'Two cells share an original file/barcode link')
    require(out.index.tolist()==source.index.tolist() and len(out)==len(source), 'Cell accounting/order changed')
    for c in source:
        require(out[c].astype(str).tolist()==source[c].astype(str).tolist(), 'Existing source column changed: '+c)
    return out.reset_index(drop=True)


def profiles(cells, raw, pairing, plan, manifest):
    """Compare cut-site counts inside peaks to fragment-record depth; descriptive identity evidence only."""
    candidates=sorted(r['destination'] for r in plan['files'] if r['kind']=='fragments' and r['destination'].startswith('GSE267584/'))
    require(len(candidates)==11, 'Expected eleven GSE267584 fragment checkpoints')
    cached, hashes={},{}
    for relative in candidates:
        p=pairing/'checkpoints'/(Path(relative).name+'.barcodes.tsv.gz')
        meta=json.loads(p.with_suffix('.json').read_text())
        require(meta['signature']==manifest['signature'] and sha(p)==meta['output_SHA256'], 'Fragment checkpoint changed: '+str(p))
        d=pd.read_csv(p,sep='\t',keep_default_na=False).set_index('original_barcode')
        require(d.index.is_unique, 'Duplicate cached fragment barcode')
        for c in ['fragment_rows','fragment_read_support_sum']:
            values=pd.to_numeric(d[c],errors='raise')
            require(np.isfinite(values).all() and values.ge(0).all(), 'Invalid fragment checkpoint counts')
        cached[relative]=d
        hashes[str(p)]=meta['output_SHA256']
    records=[]
    subset=cells.loc[cells.source_GSE.eq('GSE267584')]
    for lib,d in subset.groupby('source_library',observed=True):
        peak=pd.to_numeric(d.ATAC_peak_count_sum,errors='raise').to_numpy(dtype=float)
        require(np.isfinite(peak).all() and (peak>0).all(), 'Invalid paired ATAC peak totals')
        candidate_files=d.candidate_fragment_file.unique().tolist()
        require(len(candidate_files)==1, 'Multiple filename candidates in one RNA library')
        filename_candidate=candidate_files[0]
        for relative,frame in cached.items():
            f=frame.reindex(d.original_barcode).fillna(0)
            rows=f.fragment_rows.to_numpy(dtype=float)
            variable=len(d)>2 and np.ptp(peak)>0 and np.ptp(rows)>0
            rho=float(spearmanr(peak,rows).statistic) if variable else np.nan
            pearson=float(pearsonr(np.log1p(peak),np.log1p(rows)).statistic) if variable else np.nan
            records.append(dict(source_library=lib,cells=len(d),fragment_file=relative,
                                filename_candidate=relative==filename_candidate,
                                exact_barcode_present_cells=int((rows>0).sum()),
                                median_fragment_records=float(np.median(rows)),median_ATAC_peak_counts=float(np.median(peak)),
                                Spearman_peak_counts_vs_fragment_rows=rho,Pearson_log1p_peak_counts_vs_fragment_rows=pearson,
                                condition_conflict=d.condition_identity_status.str.startswith('UNRESOLVED').any()))
    comparisons=pd.DataFrame(records)
    ranked=[]
    for lib,d in comparisons.groupby('source_library',observed=True):
        d=d.sort_values('Spearman_peak_counts_vs_fragment_rows',ascending=False,na_position='last')
        best,second=d.iloc[0],d.iloc[1]
        named=d.loc[d.filename_candidate].iloc[0]
        ranked.append(dict(source_library=lib,cells=int(best.cells),
                           best_depth_match_fragment=best.fragment_file,
                           best_Spearman=best.Spearman_peak_counts_vs_fragment_rows,
                           second_depth_match_fragment=second.fragment_file,
                           second_Spearman=second.Spearman_peak_counts_vs_fragment_rows,
                           Spearman_gap=best.Spearman_peak_counts_vs_fragment_rows-second.Spearman_peak_counts_vs_fragment_rows,
                           filename_candidate_fragment=named.fragment_file,
                           filename_candidate_Spearman=named.Spearman_peak_counts_vs_fragment_rows,
                           best_match_is_filename_candidate=bool(best.filename_candidate),
                           condition_conflict=bool(best.condition_conflict),
                           interpretation='DESCRIPTIVE_CAPTURE_IDENTITY_CHECK_NO_AUTOMATIC_REASSIGNMENT'))
    return comparisons,pd.DataFrame(ranked),hashes


def main(run):
    cfg=json.loads((run/'INPUT_PATHS.json').read_text())
    frozen=json.loads((run/'SOURCES_SHA256.json').read_text())
    settings=json.loads((run/'SETTINGS.json').read_text())
    for n,want in frozen.items():
        require(sha(run/n)==want, 'Saved source/settings changed: '+n)
    base,raw,source,transfer=(Path(cfg[k]) for k in ['base','raw','annotated_run','transfer_audit'])
    saved=run/'RESOLVED_INPUT_PATHS.json'
    if saved.exists():
        paths=json.loads(saved.read_text())
    else:
        pairing=(base/'scenicplus_inputs/01_pairing_review/latest_complete_run.txt').read_text().strip()
        lcmv=cfg.get('lcmv_run',str(base/'scenicplus_inputs/03_GSE230363_fragment_map/runs/map_20261020T202751_3911499'))
        paths=dict(pairing_run=pairing,lcmv_run=lcmv)
        dump(saved,paths)
    pairing,lcmv=Path(paths['pairing_run']),Path(paths['lcmv_run'])
    pm=json.loads((pairing/'RUN_MANIFEST.json').read_text())
    lm=json.loads((lcmv/'RUN_MANIFEST.json').read_text())
    require(pm['status']=='COMPLETE' and lm['status']=='COMPLETE', 'Previous input run incomplete')
    require(pm['signature']==settings['pairing_signature'] and lm['signature']==settings['lcmv_signature'], 'Input runs differ from the reviewed actual reports')
    pinputs=json.loads((pairing/'AUDIT_INPUTS.json').read_text())
    linputs=json.loads((lcmv/'MAPPING_INPUTS.json').read_text())
    require(pinputs['annotated_h5ad_SHA256']==linputs['annotated_h5ad_SHA256'], 'Input runs used different RNA objects')
    h5ad=source/'ALL_UPLOADED_CELLS_WITH_TRBI.h5ad'
    say('VERIFY SOURCE RNA OBJECT AND SAVED CELL MAPS')
    require(sha(h5ad)==pinputs['annotated_h5ad_SHA256'], 'Annotated RNA changed since successful pairing')
    old=checked_table(pairing,'ALL_CELLS_PAIRING_REVIEW.tsv.gz',pm)
    correction=checked_table(lcmv,'GSE230363_ORIGINAL_FRAGMENT_CELL_ID_MAP.tsv.gz',lm)
    require(len(old)==settings['expected_cells'], 'Unexpected full uploaded cell count')
    corrected=combine(old,correction,raw,h5ad,settings['expected_lcmv_cells'])
    plan=json.loads((transfer/'UPLOAD_PLAN.json').read_text())
    require(sha(transfer/'UPLOAD_PLAN.json')==pinputs['transfer_plan_SHA256'], 'Transfer plan changed')
    say('COMPARE EXISTING GSE267584 FRAGMENT DEPTH CHECKPOINTS; NO FRAGMENT RESCAN')
    comparisons,ranked,cache_hashes=profiles(corrected,raw,pairing,plan,pm)
    inputs=dict(pairing_run=str(pairing),lcmv_run=str(lcmv),source_RNA_SHA256=pinputs['annotated_h5ad_SHA256'],
                original_pairing_table_SHA256=pm['output_SHA256']['ALL_CELLS_PAIRING_REVIEW.tsv.gz'],
                LCMV_map_SHA256=lm['output_SHA256']['GSE230363_ORIGINAL_FRAGMENT_CELL_ID_MAP.tsv.gz'],
                fragment_checkpoint_SHA256=cache_hashes,source_SHA256=frozen,input_paths_SHA256=sha(run/'INPUT_PATHS.json'))
    if (run/'MANIFEST_INPUTS.json').exists():
        require(json.loads((run/'MANIFEST_INPUTS.json').read_text())==inputs, 'Inputs changed; use a new run')
    else:
        dump(run/'MANIFEST_INPUTS.json',inputs)
    if (run/'RUN_MANIFEST.json').exists():
        m=json.loads((run/'RUN_MANIFEST.json').read_text())
        for n,want in m['output_SHA256'].items():
            require(sha(run/n)==want, 'Completed output changed: '+n)
        say('ALREADY COMPLETE; all saved outputs verified')
        return
    table(run/'ALL_UPLOADED_CELL_RNA_ATAC_LINKS.tsv.gz',corrected)
    table(run/'GSE267584_FRAGMENT_DEPTH_CORRELATIONS.tsv',comparisons)
    table(run/'GSE267584_BEST_FRAGMENT_DEPTH_MATCH.tsv',ranked)
    summary=corrected.groupby(['source_GSE','source_library','fragment_link_status'],observed=True).agg(
        cells=('Cell_ID','size'),working_CD8_cells=('working_CD8_scope','sum'),
        working_CD8_with_validated_fragment_link=('working_CD8_with_validated_fragment_link','sum')).reset_index()
    table(run/'RNA_ATAC_LINK_SUMMARY_BY_LIBRARY.tsv',summary)
    report=dict(status='COMPLETE',all_uploaded_cells=len(corrected),
                validated_original_fragment_links=int(corrected.fragment_link_validated.sum()),
                working_CD8_with_validated_fragment_link=int(corrected.working_CD8_with_validated_fragment_link.sum()),
                condition_conflict_candidate_only_cells=int(corrected.fragment_link_status.eq('CONDITION_CONFLICT_CANDIDATE_ONLY').sum()),
                matrix_paired_but_fragment_file_not_in_upload_cells=int(corrected.fragment_link_status.eq('MATRIX_PAIRED_FRAGMENT_FILE_NOT_IN_UPLOAD').sum()),
                all_cells_original_order_and_labels_preserved=True,no_source_expression_or_fragment_files_changed=True,
                depth_comparisons_automatically_reassign_samples=False,new_ATAC_QC_or_SCENICPLUS_run=False)
    require(report['validated_original_fragment_links']==settings['expected_validated_fragment_links'], 'Unexpected validated-link accounting')
    dump(run/'LINK_MANIFEST_SUMMARY.json',report)
    names=['ALL_UPLOADED_CELL_RNA_ATAC_LINKS.tsv.gz','GSE267584_FRAGMENT_DEPTH_CORRELATIONS.tsv',
           'GSE267584_BEST_FRAGMENT_DEPTH_MATCH.tsv','RNA_ATAC_LINK_SUMMARY_BY_LIBRARY.tsv','LINK_MANIFEST_SUMMARY.json']
    dump(run/'RUN_MANIFEST.json',dict(status='COMPLETE',output_SHA256={n:sha(run/n) for n in names}))
    with zipfile.ZipFile(run/'SHARE_RNA_ATAC_LINK_MANIFEST.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in names[1:]+['RUN_MANIFEST.json','README_RUN.md','MANIFEST_INPUTS.json']:
            z.write(run/n,n)
    with open(run.parent.parent/'README.md','a') as f:
        f.write('\n- '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' Combined original RNA/fragment links for all uploaded cells; retained unresolved cases and added fragment-depth capture diagnostics. Run: '+str(run)+'\n')
    say('COMPLETE',json.dumps(report))
    print(ranked.to_string(index=False),flush=True)
    say('SHARE',run/'SHARE_RNA_ATAC_LINK_MANIFEST.zip')


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
ATAC_LINK_SOURCE_0
cat > "$PAIR_RUN/README_RUN.md" <<'ATAC_LINK_SOURCE_1'
# Unified RNA–ATAC cell-link manifest and GSE267584 capture check

This step combines the completed initial pairing audit with the completed GSE230363 suffix mapping. The actual HPC reports confirm paired peak matrices for all 161,773 cells and full fragment-ID coverage for all 72,362 GSE230363 cells. The latter map validated every fragment record, found no barcode collisions and preserved all original source files.

The unified output is `ALL_UPLOADED_CELL_RNA_ATAC_LINKS.tsv.gz`, containing all 161,773 cells in original RNA order. It retains the online labels, original audit fields, source library IDs and condition-conflict flags. New columns record the existing RNA H5AD path, original fragment file and barcode, supplied fragment-record/support counts, and the current fragment-link status. There is no new expression object, normalization, cell selection, merging of peak universes, ATAC-QC decision or SCENIC+ run.

The new `fragment_link_status` and `fragment_link_validated` fields are authoritative for this combined manifest. Earlier audit columns, such as `matrix_and_fragment_pairing_verified`, are retained as historical evidence and are not retroactively edited; GSE230363 had zero exact matches before the suffix map.

## Expected accounting

- 121,176 cells currently have validated original fragment links: 72,362 GSE230363, 11,484 GSE255042, 34,512 GSE267584, and 2,818 GSE290618.
- Of these, 120,169 are in the existing working CD8 scope. This is an availability count before new ATAC QC or a fine-label confidence cutoff, not a final analysis subset. CD8-only upload scope remains inherited from the original selection, rather than a new independently returned broad lineage.
- 12,890 GSE267584 cells in GSM8269394/395/396 retain their condition conflicts and candidate-only fragments.
- 27,707 W8/W9 cells retain verified paired peak matrices but no fragment file in the upload.

All portal-excluded/non-CD8 cells remain present with their original labels/status. A validated fragment link is a data-identity/availability result, not a new cell-quality acceptance decision.

## Why another count check is needed for GSE267584

The original audit found that selected barcodes also occur in the *other* GSE267584 fragment files, often as low-count background. For example, GSM8269394 has 4,312/4,312 barcodes in its filename candidate, but 4,258/4,312 also occur in another library's fragment file. Presence alone is therefore insufficient evidence for reassigning the three conflicting libraries.

This step uses the already saved fragment-count checkpoints, not another raw-fragment scan. For each of eleven GSE267584 RNA/paired-ATAC libraries, it compares each cell's supplied peak-count sum with that barcode's fragment-record count in each of the eleven fragment files. It reports Spearman correlation, Pearson correlation of log1p totals, coverage and median depth, plus the best and second-best depth matches. The transformation is used only for diagnostic correlation, never to replace expression or count values. Fragment records and peak cut-site sums are related depth measures but need not be numerically equal because their definitions and genomic scope differ.

`GSE267584_FRAGMENT_DEPTH_CORRELATIONS.tsv` contains all 121 comparisons. `GSE267584_BEST_FRAGMENT_DEPTH_MATCH.tsv` summarizes the ranked candidates. There are no automatic correlation cutoffs, reassigned cells, assigned biological conditions, or changes to the unresolved libraries. These are descriptive capture-identity diagnostics, not biological lineage validation or independent replicate statistics. Even if a fragment capture is identified, a conflicting GEO condition name remains a separate metadata issue.

## Integrity and reproducibility

The prior pairing and LCMV run signatures are pinned to the actual reviewed reports. The all-cell pairing table and LCMV per-cell map are checked against their completed output manifests. Both runs must name the same annotated RNA SHA256, and the current source H5AD is verified against that digest. Existing GSE267584 fragment checkpoints must match the original pairing run's signature and their per-file output hashes. The transfer plan is checked against the original pairing inputs. No raw fragments, matrix counts or annotations are recomputed.

The source/settings and resolved input paths are frozen. Saved submission/run locks prevent duplicate active work on one run. The saved `SUBMIT_OR_RESUME.sh` can repeat this small report-building step and verify completed outputs. READMEs record preparation, submission, start, failure or completion. The existing RNA environment is used without package installation.

Resources: two CPUs, 5632 MiB per CPU (11 GiB), one-hour wall-time limit. Most work is verifying existing files, joining tables, and computing the small correlation table; no fragment-decompression scan is performed. `SHARE_RNA_ATAC_LINK_MANIFEST.zip` contains aggregate reports, methods and input provenance, excluding the larger all-cell table.
ATAC_LINK_SOURCE_1
cat > "$PAIR_RUN/SETTINGS.json" <<'ATAC_LINK_SOURCE_2'
{
  "expected_cells": 161773,
  "expected_lcmv_cells": 72362,
  "expected_validated_fragment_links": 121176,
  "pairing_signature": "b28db9e69eb215bdd27981cf2aa369f916093c43b804355cde1b43538afc8546",
  "lcmv_signature": "0dccdd382bb1b671d3f1f5d770be9e25fb2ff754711cbb015d3b0c9214f0daeb"
}
ATAC_LINK_SOURCE_2
cat > "$PAIR_RUN/SOURCES_SHA256.json" <<'ATAC_LINK_SOURCE_3'
{
  "build_atac_cell_manifest.py": "d88f8823e34de31c5e709761a7ec07cba0d32c7dc0a15aa0b29ee6f21dc9e46f",
  "README_RUN.md": "bce0e968e2268a2a771ecba00f8beadb3fdfa01ee5fcbfe55c77c039983c4309",
  "SETTINGS.json": "8d27375e7b12895a6e6d5af9bad514e44fd9bfa1c8e956e46fbb9ca7e3782d91"
}
ATAC_LINK_SOURCE_3

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
trap 'PAIR_RC=$?; if [ "$PAIR_RC" -ne 0 ]; then printf "\n- %s RNA–ATAC cell-link manifest stopped with exit %s; run %s. Completed checkpoints retained; source data and labels preserved.\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_RC" "$PAIR_RUN" >> "$PAIR_ROOT/README.md"; echo "Stopped. Share log tails; resume with $PAIR_RUN/SUBMIT_OR_RESUME.sh"; fi' EXIT
PAIR_BASE="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["base"])' "$PAIR_RUN/INPUT_PATHS.json")"
PAIR_ENV="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["env"])' "$PAIR_RUN/INPUT_PATHS.json")"
PAIR_MAMBA="$PAIR_BASE/immgent_annotation_betty/bin/micromamba"
export MAMBA_ROOT_PREFIX="$PAIR_BASE/rbi2_upload_jobs_data/REASSEMBLED_RNA/mamba"
eval "$("$PAIR_MAMBA" shell hook -s bash)"
micromamba activate "$PAIR_ENV"
unset PYTHONPATH PYTHONHOME
printf '\n- %s Started/resumed RNA–ATAC cell-link manifest job %s; run %s. Combine verified cell links and compare cached ATAC depth profiles; no cell filtering.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "$PAIR_RUN" >> "$PAIR_ROOT/README.md"
echo "RUN: $PAIR_RUN"
echo "Slurm job: $SLURM_JOB_ID; allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}; cached count comparisons; no fragment rescan"
"$PAIR_ENV/bin/python" - "$PAIR_RUN" <<'PAIR_ENV_CHECK'
import importlib.metadata,pathlib,sys
from anndata.experimental import read_elem
expected={'anndata':'0.10.9','h5py':'3.11.0','numpy':'1.26.4','pandas':'2.2.3','scipy':'1.13.1'}
versions={n:importlib.metadata.version(n) for n in expected}
if versions!=expected:raise RuntimeError('Existing RNA environment differs: '+str(versions))
pathlib.Path(sys.argv[1],'ENVIRONMENT_CHECK.txt').write_text(sys.version+'\n'+repr(versions)+'\n')
print('Existing RNA environment checked; no installation or mutation.')
PAIR_ENV_CHECK
"$PAIR_ENV/bin/python" -u "$PAIR_RUN/build_atac_cell_manifest.py" "$PAIR_RUN"
printf '%s\n' "$PAIR_RUN" > "$PAIR_ROOT/latest_complete_run.txt"
echo "Share: $PAIR_RUN/SHARE_RNA_ATAC_LINK_MANIFEST.zip"
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
PAIR_OPTS=(--parsable --job-name=RNA_ATAC_links --partition=genoa-std-mem --nodes=1 --ntasks=1
  --cpus-per-task=2 --mem-per-cpu=5632M --time=01:00:00 --chdir="$PAIR_RUN"
  --output="$PAIR_RUN/slurm_%j.out" --error="$PAIR_RUN/slurm_%j.err" --export=ALL)
if [ -n "${IMMGENT_SLURM_ACCOUNT:-}" ]; then PAIR_OPTS+=(--account="$IMMGENT_SLURM_ACCOUNT"); fi
PAIR_JOB="$(sbatch "${PAIR_OPTS[@]}" "$PAIR_RUN/RUN_ON_COMPUTE.sh" "$PAIR_RUN")"
PAIR_JOB="${PAIR_JOB%%;*}"
[[ "$PAIR_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected submission response: $PAIR_JOB"; exit 1; }
printf '%s\n' "$PAIR_JOB" > "$PAIR_RUN/SLURM_JOB_ID.txt"
printf '\n- %s Submitted RNA–ATAC cell-link manifest job %s; run %s.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_JOB" "$PAIR_RUN" >> "$(dirname -- "$(dirname -- "$PAIR_RUN")")/README.md"
echo "Submitted RNA–ATAC cell-link manifest job: $PAIR_JOB"
echo "Check: squeue -j $PAIR_JOB"
echo "Logs: tail -n 40 \"$PAIR_RUN/slurm_${PAIR_JOB}.out\" \"$PAIR_RUN/slurm_${PAIR_JOB}.err\""
echo "Resume: bash \"$PAIR_RUN/SUBMIT_OR_RESUME.sh\""
echo "Run folder: $PAIR_RUN"
PAIR_SUBMIT
printf '%s\n' "$PAIR_RUN" > "$PAIR_ROOT/latest_started_run.txt"
printf '\n- %s Prepared RNA–ATAC cell-link manifest %s. Source RNA: %s. Transfer audit: %s. See README_RUN.md.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIR_RUN" "$PAIR_SOURCE" "$PAIR_TRANSFER" >> "$PAIR_ROOT/README.md"
bash "$PAIR_RUN/SUBMIT_OR_RESUME.sh"
