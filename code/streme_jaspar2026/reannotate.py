"""Match the frozen all-196 STREME outputs to the existing JASPAR2026 database."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

BASE = Path('/vast/projects/saargill/cell-type-specific/nazbuhn/immgen')
DB = BASE/'C1_CRESTED_JASPAR2026_BETTY_V2/references/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt'
DB_SHA = 'ae4f77a17256c7b8b7b73402810c1ecf4fc411e9ad7832cdc87d586dee9a4a15'
TOMTOM = BASE/'c1_core_analysis_tools/envs/meme_cpu_v1/bin/tomtom'
EXPECTED = {
    'vs_initial':'4facb71cbdcce55a0bf217761de9969b575ea87d47ddb4a3bf09fedf6c46bd80',
    'vs_dinucleotide':'664cfb3246b474fb7d5635a87c1b33341568dd02bf5c8d2e41fd6f7f50474eaf',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def profiles(path):
    result = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith('MOTIF '):
            fields = line.split(maxsplit=2)
            current = fields[1]
            if current in result:
                raise RuntimeError('Duplicate motif ID: '+current)
            result[current] = {'name':fields[2] if len(fields)>2 else ''}
        elif current and 'letter-probability matrix:' in line:
            width = re.search(r'\bw=\s*(\d+)', line)
            evalue = re.search(r'\bE=\s*(\S+)', line)
            if width:
                result[current]['width'] = int(width.group(1))
            if evalue:
                result[current]['E'] = float(evalue.group(1))
    return result

def find_source(comparison):
    package = BASE/'C1_STREME_ALL196_BETTY_V1'
    candidates = []
    if (package/'LATEST_RUN.txt').is_file():
        run = Path((package/'LATEST_RUN.txt').read_text().strip())
        candidates.append(run/comparison/'streme.txt')
    candidates.append(BASE/'C1_MODISCO_ALL196_BETTY_V1/inputs'/f'STREME_{comparison}.meme')
    for path in candidates:
        if path.is_file() and sha(path) == EXPECTED[comparison]:
            return path
    raise RuntimeError('Frozen all-196 STREME source not found or checksum differs: '+comparison)

def write_table(path, rows, fields):
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    run = args.run
    run.mkdir(parents=True,exist_ok=True)
    if not DB.is_file() or sha(DB) != DB_SHA:
        raise RuntimeError('The completed workflow\'s frozen JASPAR2026 database is missing or differs.')
    database = profiles(DB)
    if len(database) != 1019:
        raise RuntimeError('Expected all 1,019 JASPAR2026 profiles.')
    if not TOMTOM.is_file():
        raise FileNotFoundError(TOMTOM)
    version = subprocess.check_output([str(TOMTOM),'-version'],text=True).strip()
    summary, matches, manifest = [], [], {}
    for comparison in EXPECTED:
        source = find_source(comparison)
        query = run/f'STREME_{comparison}.meme'
        shutil.copy2(source,query)
        queries = profiles(query)
        if len(queries) != 20 or any('E' not in x for x in queries.values()):
            raise RuntimeError('Expected twenty original STREME outputs with E-values.')
        command = [str(TOMTOM),'-oc',str(run/comparison),'-thresh','0.05',str(query),str(DB)]
        print('Matching',comparison,': 20 STREME motifs against 1,019 JASPAR2026 profiles.',flush=True)
        with (run/f'{comparison}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (run/comparison/'tomtom.tsv').open() as f:
            reader = csv.DictReader((s for s in f if s.strip() and not s.startswith('#')),delimiter='\t')
            hits = list(reader)
        for hit in hits:
            if hit['Query_ID'] not in queries or hit['Target_ID'] not in database:
                raise RuntimeError('Unexpected motif ID in Tomtom output.')
            if float(hit['q-value']) > .05:
                raise RuntimeError('Unexpected above-threshold match.')
            matches.append(dict(comparison=comparison,**hit,
                                target_name=database[hit['Target_ID']]['name']))
        for motif, info in queries.items():
            selected = sorted([h for h in hits if h['Query_ID']==motif],
                              key=lambda h:(float(h['q-value']),float(h['p-value']),h['Target_ID']))
            best = float(selected[0]['q-value']) if selected else None
            tied = [h for h in selected if abs(float(h['q-value'])-best)<=max(1e-15,abs(best)*1e-9)] if selected else []
            summary.append(dict(comparison=comparison,motif=motif,width=info['width'],
                STREME_E=info['E'],STREME_E_below_0p05=info['E']<.05,
                significant_JASPAR2026_matches=len(selected),
                lowest_Tomtom_q='' if best is None else best,
                best_q_profile_IDs=';'.join(h['Target_ID'] for h in tied),
                best_q_profile_names=';'.join(database[h['Target_ID']]['name'] for h in tied),
                match_status='significant reference match' if selected else 'no reference match at q<=0.05'))
        manifest[comparison] = dict(source=str(source),input_sha256=sha(query),command=command)
    fields=list(summary[0])
    write_table(run/'ALL40_STREME_JASPAR2026_SUMMARY.tsv',summary,fields)
    enriched=[row for row in summary if row['STREME_E_below_0p05']]
    if len(enriched)!=11:
        raise RuntimeError('Expected the same five plus six enriched STREME outputs.')
    write_table(run/'ENRICHED11_STREME_JASPAR2026_SUMMARY.tsv',enriched,fields)
    match_fields=['comparison','Query_ID','Target_ID','target_name','Optimal_offset','p-value','E-value',
                  'q-value','Overlap','Query_consensus','Target_consensus','Orientation']
    write_table(run/'ALL_SIGNIFICANT_JASPAR2026_MATCHES.tsv',matches,match_fields)
    note = dict(status='complete',tomtom_version=version,database=str(DB),database_sha256=sha(DB),
        database_profiles=len(database),query_patterns=40,STREME_enriched_outputs=11,
        enriched_outputs_with_reference_match=sum(row['significant_JASPAR2026_matches']>0 for row in enriched),
        inputs=manifest,STREME_discovery_rerun=False,FIMO_rerun=False,model_inference_performed=False,
        interpretation='Reference motif similarity only; not a unique TF assignment or experimental binding evidence.')
    (run/'RESULT.json').write_text(json.dumps(note,indent=2))
    archive=run/'SHARE_C1_STREME_JASPAR2026.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p!=archive and not p.name.startswith('slurm_'):
                z.write(p,str(p.relative_to(run)))
    print(json.dumps(note,indent=2),flush=True)
    print('DOWNLOAD:',archive,flush=True)

if __name__=='__main__':
    main()
