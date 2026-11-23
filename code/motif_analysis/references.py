"""Public JASPAR 2026 references and identifier-based mouse TF candidates. No model imports."""
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import urllib.request

P = Path(__file__).resolve().parent
FILES = {
    'JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt':
        'https://jaspar2026.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt',
    'JASPAR2026.sqlite3':
        'https://mencius-pub.uio.no/JASPAR/JASPAR_metadata/2026/JASPAR2026.sqlite3',
    'HOM_MouseHumanSequence.rpt':
        'https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt',
}
MEME = next(iter(FILES))


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def table(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def meme_profiles(path):
    profiles = {}
    lines = Path(path).read_text().splitlines()
    need(lines and lines[0].startswith('MEME version'), 'Not a MEME database: ' + str(path))
    current = None
    for i, line in enumerate(lines):
        if line.startswith('MOTIF '):
            parts = line.split(maxsplit=2)
            current = parts[1]
            need(re.fullmatch(r'MA\d+\.\d+', current) is not None and current not in profiles,
                 'Invalid or duplicate JASPAR motif ID: ' + current)
            profiles[current] = dict(name=parts[2] if len(parts) > 2 else '', width=None)
        if line.startswith('letter-probability matrix:'):
            need(current is not None, 'Matrix has no motif ID')
            match = re.search(r'\bw\s*=\s*(\d+)', line)
            need(match is not None, 'No matrix width')
            width = int(match.group(1))
            profiles[current]['width'] = width
            rows = lines[i+1:i+1+width]
            need(len(rows) == width, 'Truncated PFM')
            for row in rows:
                values = list(map(float, row.split()))
                need(len(values) == 4 and all(0 <= x <= 1 for x in values)
                     and abs(sum(values)-1) < .02, 'Invalid probability row')
    need(all(v['width'] for v in profiles.values()), 'A motif lacks its matrix')
    return profiles


def normalize(value):
    return re.sub('[^a-z0-9]', '', value.lower())


def uniprot_ids(value):
    # Do not infer a TF from a familiar sequence or edit a TF gene name.
    result = set()
    for token in re.split(r'[,;|\s]+', value.strip()):
        token = re.sub(r'-\d+$', '', token.strip())
        if re.fullmatch(r'[A-Z0-9]{6,10}', token):
            result.add(token)
    return result


def mouse_homology(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter='\t')
        headers = {normalize(x): x for x in reader.fieldnames or []}
        def col(*names):
            matches = [headers[n] for n in names if n in headers]
            need(len(matches) == 1, 'Missing/ambiguous MGI column: ' + str(names))
            return matches[0]
        group = col('dbclasskey', 'homologeneid')
        tax = col('ncbitaxonid')
        symbol = col('symbol')
        swiss = [v for k, v in headers.items() if k.startswith('swissprot')]
        need(len(swiss) == 1, 'Cannot identify MGI SWISS-PROT column')
        groups = collections.defaultdict(list)
        for row in reader:
            if row[tax].strip() in {'10090', '9606'}:
                groups[row[group]].append(dict(tax=row[tax].strip(), symbol=row[symbol].strip(),
                                              proteins=uniprot_ids(row[swiss[0]])))
    proteins = collections.defaultdict(set)
    mouse_symbols = collections.defaultdict(set)
    human_symbols = collections.defaultdict(set)
    for key, rows in groups.items():
        mice = {r['symbol'] for r in rows if r['tax'] == '10090'}
        for row in rows:
            if row['tax'] == '10090':
                mouse_symbols[row['symbol'].casefold()].add(row['symbol'])
                candidates = {row['symbol']}
                route = 'MGI_mouse_UniProt'
            else:
                human_symbols[row['symbol'].casefold()].update(mice)
                candidates = mice
                route = 'MGI_human_UniProt_to_mouse_homology'
            for protein in row['proteins']:
                proteins[protein].update((symbol, route, key) for symbol in candidates)
    need(len(mouse_symbols) > 1000 and len(proteins) > 1000, 'MGI report appears incomplete')
    return proteins, mouse_symbols, human_symbols


def sql_metadata(path, profiles):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    need(db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'SQLite integrity check failed')
    tables = {r[0].upper(): r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    need({'MATRIX','MATRIX_PROTEIN','MATRIX_SPECIES','MATRIX_ANNOTATION'} <= tables.keys(),
         'Unexpected JASPAR SQLite schema: ' + str(sorted(tables)))
    for table_name, required in [('MATRIX', {'ID','BASE_ID','VERSION','COLLECTION','NAME'}),
                                 ('MATRIX_PROTEIN', {'ID','ACC'}),
                                 ('MATRIX_SPECIES', {'ID','TAX_ID'}),
                                 ('MATRIX_ANNOTATION', {'ID','TAG','VAL'})]:
        actual = {r[1].upper() for r in db.execute('PRAGMA table_info("'+tables[table_name]+'")')}
        need(required <= actual, 'JASPAR columns differ: ' + table_name)
    index = collections.defaultdict(list)
    for row in db.execute('SELECT ID,BASE_ID,VERSION,COLLECTION,NAME FROM MATRIX'):
        index[str(row[1])+'.'+str(int(row[2]))].append(row)
    results = {}
    for motif in profiles:
        rows = index[motif]
        need(len(rows) == 1, 'JASPAR version absent or duplicated in SQLite: ' + motif)
        internal, _, _, collection, name = rows[0]
        need(collection.upper() == 'CORE', 'Non-CORE motif in CORE file')
        proteins = {str(r[0]) for r in db.execute('SELECT ACC FROM MATRIX_PROTEIN WHERE ID=?',(internal,))}
        species = {str(r[0]) for r in db.execute('SELECT TAX_ID FROM MATRIX_SPECIES WHERE ID=?',(internal,))}
        annotations = list(db.execute('SELECT TAG,VAL FROM MATRIX_ANNOTATION WHERE ID=?',(internal,)))
        tax_groups = [str(v).lower() for k,v in annotations if str(k).lower() == 'tax_group']
        need(tax_groups and all('vertebrat' in v for v in tax_groups), 'Non-vertebrate profile: '+motif)
        results[motif] = dict(name=name, proteins=proteins, species=species)
    db.close()
    return results


def build_mapping(profiles, metadata, homology):
    proteins, mouse_symbols, human_symbols = homology
    mapping, audit = [], []
    for motif, profile in profiles.items():
        meta = metadata[motif]
        candidates = set()
        for accession in meta['proteins']:
            for protein in uniprot_ids(accession):
                for symbol, route, group in proteins.get(protein, set()):
                    candidates.add(symbol)
                    audit.append(dict(Motif_name=motif, TF=symbol, route=route,
                                      source_identifier=protein, homology_group=group))
        # Only use exact approved source-species symbols as a documented fallback.
        # Retain documented one-to-many human/mouse homologies as candidate sets.
        if not candidates:
            components = meta['name'].split('::')
            for component in components:
                component = re.sub(r'\(var\.\d+\)$','',component).strip().casefold()
                if '10090' in meta['species'] and len(mouse_symbols.get(component,set())) == 1:
                    for symbol in mouse_symbols[component]:
                        candidates.add(symbol)
                        audit.append(dict(Motif_name=motif,TF=symbol,route='MGI_mouse_approved_symbol_unique_casefold',source_identifier=component,homology_group=''))
                if '9606' in meta['species']:
                    for symbol in human_symbols.get(component,set()):
                        candidates.add(symbol)
                        audit.append(dict(Motif_name=motif,TF=symbol,route='MGI_human_approved_symbol_to_mouse_homology',source_identifier=component,homology_group=''))
        mapping.append(dict(Motif_name=motif, JASPAR_name=meta['name'],
            Mouse_JASPAR_MGI_candidates=', '.join(sorted(candidates)),
            source_taxa=';'.join(sorted(meta['species'])),
            UniProt_accessions=';'.join(sorted(meta['proteins'])),
            status='mapped' if candidates else 'no_verified_mouse_assignment',
            complex_profile='::' in meta['name']))
    return mapping, audit


def fetch(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / 'REFERENCE_MANIFEST.json'
    if frozen.exists():
        verify(folder)
        print('Previously frozen reference files verified. No download needed.')
        return
    for name, url in FILES.items():
        dest = folder / name
        if dest.exists():
            print('Using supplied file:', dest, flush=True)
            continue
        print('Downloading official reference:', url, flush=True)
        tmp = dest.with_name(name + '.download.partial')
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'C1-CREsted-reference-preflight/2'})
            with urllib.request.urlopen(req, timeout=60) as response, tmp.open('wb') as handle:
                total = 0
                while True:
                    block = response.read(1024**2)
                    if not block:
                        break
                    total += len(block)
                    need(total < 2_000_000_000, 'Reference download unexpectedly large')
                    handle.write(block)
            tmp.replace(dest)
        except Exception as error:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f'Download failed: {url}\n{error}\nDownload the file on your Mac and upload it to {folder}. No Slurm job has been submitted.') from error
    profiles = meme_profiles(folder / MEME)
    need(len(profiles) == 1019, f'Expected 1,019 JASPAR2026 CORE vertebrate motifs, found {len(profiles)}')
    metadata = sql_metadata(folder / 'JASPAR2026.sqlite3', profiles)
    homology = mouse_homology(folder / 'HOM_MouseHumanSequence.rpt')
    mapping, audit = build_mapping(profiles, metadata, homology)
    need(sum(r['status']=='mapped' for r in mapping) >= 100,
         'Unexpectedly few identifier-supported mouse mappings; inspect reference formats')
    table(folder / 'JASPAR2026_MOUSE_TF_MAPPING.tsv',mapping,list(mapping[0]))
    table(folder / 'JASPAR2026_MOUSE_MAPPING_AUDIT.tsv',audit,
          ['Motif_name','TF','route','source_identifier','homology_group'])
    names = list(FILES) + ['JASPAR2026_MOUSE_TF_MAPPING.tsv','JASPAR2026_MOUSE_MAPPING_AUDIT.tsv']
    dump(frozen,dict(release=2026, collection='CORE', tax_group='vertebrates', profiles=len(profiles),
         mapped_profiles=sum(r['status']=='mapped' for r in mapping),
         unmapped_profiles=sum(r['status']!='mapped' for r in mapping),
         frozen_UTC=datetime.now(timezone.utc).isoformat(),
         sources=FILES, files={n:sha(folder/n) for n in names},
         mapping_rule='JASPAR protein identifiers to MGI mouse genes or MGI human-to-mouse homology; source-species approved-symbol fallback; no TF assigned from sequence resemblance alone.',
         limitations=['One-to-many homologies remain candidate sets.',
                     'All 1,019 profiles enter Tomtom/FIMO, including profiles with no verified mouse TF assignment.',
                     'Complex profiles can have multiple TF subunits; this does not prove cooperation.',
                     'This is a documented JASPAR/MGI adaptation, not the identical Aerts motif-to-TF collection.']))
    print('REFERENCES READY:', frozen, flush=True)


def verify(folder):
    folder=Path(folder)
    record=json.loads((folder/'REFERENCE_MANIFEST.json').read_text())
    need(record['release']==2026 and record['profiles']==1019,'Unexpected reference release')
    for name, expected in record['files'].items():
        need(sha(folder/name)==expected, 'Frozen reference changed: '+name)
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['fetch','verify'])
    parser.add_argument('--folder',type=Path,default=P/'references')
    args=parser.parse_args()
    if args.action=='fetch':
        fetch(args.folder)
    else:
        print(json.dumps(verify(args.folder),indent=2))
