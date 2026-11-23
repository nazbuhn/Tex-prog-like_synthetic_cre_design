#!/usr/bin/env python3
"""Sequence-only STREME analysis. Standard library; no model/attribution imports."""
import argparse
import csv
import hashlib
import json
import subprocess
import traceback
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def fasta(path):
    seqs = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            name = line[1:].split()[0]
            if name in seqs:
                raise ValueError('Duplicate FASTA identifier: ' + name)
            seqs[name] = ''
        elif line.strip():
            if name is None:
                raise ValueError('Sequence before FASTA header')
            seqs[name] += line.strip()
    return seqs


def check_inputs(package):
    manifest = json.loads((package / 'INPUT_MANIFEST.json').read_text())
    for rel, expected in manifest['file_sha256'].items():
        actual = hashlib.sha256((package / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Input checksum mismatch: ' + rel)
    before = fasta(package / 'inputs/INITIAL_196.fa')
    after = fasta(package / 'inputs/SELECTED_196.fa')
    if len(before) != 196 or before.keys() != after.keys():
        raise ValueError('Expected exactly 196 matched initial/selected designs')
    if any(len(s) != 1000 or set(s) - set('ACGT')
           for data in (before, after) for s in data.values()):
        raise ValueError('Every insert must be 1000 bp with A/C/G/T only')
    with (package / 'inputs/DESIGN_METADATA.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if len(rows) != 196 or {r['design_id'] for r in rows} != set(before):
        raise ValueError('Metadata does not match FASTA identifiers')
    counts = {}
    for r in rows:
        key = r['arm'] + '_retention_' + r['retention_fraction']
        counts[key] = counts.get(key, 0) + 1
    return dict(designs=196, sequence_length=1000, group_counts=counts,
                parent_start_families=len({r['parent_family'] for r in rows}),
                input_manifest_sha256=hashlib.sha256(
                    (package / 'INPUT_MANIFEST.json').read_bytes()).hexdigest())


def summarize_xml(path, comparison):
    tree = ET.parse(path).getroot()
    rows = []
    keys = ['id', 'alt', 'width', 'train_pos_count', 'train_neg_count',
            'train_pvalue', 'test_pos_count', 'test_neg_count',
            'test_pvalue', 'test_evalue']
    for motif in tree.iter('motif'):
        row = dict(comparison=comparison)
        row.update({k: motif.get(k, '') for k in keys})
        value = motif.get('test_evalue')
        try:
            significant = str(float(value) < 0.05)
        except (ValueError, TypeError):
            significant = 'not_available'
        row['internal_test_E_lt_0.05'] = significant
        rows.append(row)
    return rows


def main(args):
    package = args.package.resolve()
    verified = check_inputs(package)
    if args.check_inputs:
        print(json.dumps(verified, indent=2))
        return
    if args.out is None or args.streme is None:
        raise ValueError('--out and --streme are required for execution')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'JOB_RESULT.json').exists() or any((out / s).exists()
            for s in ['vs_initial', 'vs_dinucleotide']):
        raise ValueError('Existing analysis output found; no overwrite attempted')
    try:
        version = subprocess.check_output([str(args.streme), '--version'],
                                          text=True, stderr=subprocess.STDOUT).strip()
        if version != '5.5.9':
            raise ValueError('Expected the existing STREME 5.5.9, found: ' + version)
        write_json(out / 'RUN_METADATA.json', dict(
            started_UTC=stamp(), streme_version=version,
            streme_executable=str(args.streme), inputs=verified,
            parameters=dict(order=1, minw=6, maxw=20, nmotifs=20,
                            hofract=0.1, seed=20261111),
            interpretation='All-library exploratory motif discovery. Related designs and paired stages are not independent biological replicates. STREME internal holdout is not a family-level confirmation split.'))
        all_rows = []
        for label in ['vs_initial', 'vs_dinucleotide']:
            write_json(out / 'PROGRESS.json', dict(stage=label, time_UTC=stamp()))
            cmd = [str(args.streme), '--dna', '--p',
                   str(package / 'inputs/SELECTED_196.fa'), '--order', '1',
                   '--minw', '6', '--maxw', '20', '--nmotifs', '20',
                   '--hofract', '0.1', '--seed', '20261111',
                   '--o', str(out / label)]
            if label == 'vs_initial':
                cmd += ['--n', str(package / 'inputs/INITIAL_196.fa')]
            write_json(out / (label + '_COMMAND.json'), cmd)
            print('Starting ' + label, flush=True)
            with (out / (label + '.log')).open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            for name in ['streme.xml', 'streme.txt', 'streme.html']:
                if not (out / label / name).is_file():
                    raise ValueError('Missing STREME output: ' + label + '/' + name)
            all_rows.extend(summarize_xml(out / label / 'streme.xml', label))
        columns = ['comparison', 'id', 'alt', 'width', 'train_pos_count',
                   'train_neg_count', 'train_pvalue', 'test_pos_count',
                   'test_neg_count', 'test_pvalue', 'test_evalue',
                   'internal_test_E_lt_0.05']
        with (out / 'MOTIF_SUMMARY.tsv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t')
            writer.writeheader()
            writer.writerows(all_rows)
        write_json(out / 'JOB_RESULT.json', dict(status='complete', time_UTC=stamp(),
            designs=196, comparisons=2, motifs_reported=len(all_rows),
            internal_E_lt_0_05=sum(r['internal_test_E_lt_0.05'] == 'True' for r in all_rows)))
        write_json(out / 'PROGRESS.json', dict(stage='complete', time_UTC=stamp()))
        archive = out / 'SHARE_C1_STREME_ALL196.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p != archive:
                    z.write(p, str(p.relative_to(out)))
            for name in ['INPUT_MANIFEST.json', 'README.txt', 'run_streme.py']:
                z.write(package / name, 'provenance/' + name)
            for p in sorted((package / 'inputs').glob('*')):
                z.write(p, 'inputs/' + p.name)
        print('COMPLETE: ' + str(archive), flush=True)
    except Exception as error:
        write_json(out / 'JOB_RESULT.json', dict(status='failed', time_UTC=stamp(),
                                                error=str(error)))
        (out / 'FAILURE.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--streme', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    main(parser.parse_args())
