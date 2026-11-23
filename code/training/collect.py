from pathlib import Path
import fcntl
import sys
import zipfile
from util import read, dump, rows, table, verify, sha


def main(root):
    root = Path(root)
    with (root / 'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        settings = read(root / 'SETTINGS.json')
        statuses, metrics, sources = [], [], []
        files = ['SETTINGS.json', 'METHODS.txt', 'LOCAL_VALIDATION.json', 'BUNDLE_SHA256.json',
                 'PREPARATION_COMPLETE.json', 'NATIVE_NUMERIC_CHECK.json', 'FOXP3_RESTORATION.tsv',
                 'MATCHED_CELL_LABELS.tsv.gz', 'PROGRESS.json']
        for panel in settings['panels']:
            folder = root / 'models' / panel
            complete = (folder / 'COMPLETE.json').exists()
            if complete:
                verify(folder)
            summary = read(folder / 'SUMMARY.json') if (folder / 'SUMMARY.json').exists() else dict(status='incomplete')
            statuses.append(dict(panel=panel, complete_receipt=complete, **summary))
            for name in ['SUMMARY.json', 'BEST.json', 'BASELINE.json', 'HISTORY.json', 'PROGRESS.json',
                         'NATIVE_NUMERIC_CHECK.json', 'VALIDATION_SELECTIVITY.tsv', 'TEST_SELECTIVITY.tsv',
                         'TEST_ALL_GROUPS_SENSITIVITY.tsv', 'TEST_TOP_REGIONS.tsv.gz', 'TRAINING_SEEDS.tsv.gz', 'SEED_TARGETS.json']:
                files.append(str(Path('models') / panel / name))
            for name in ['DATASET.json', 'SELECTION_TARGETS.tsv', 'TOPIC_CELL_ASSOCIATIONS.tsv', 'CAPTURE_GROUPS.json', 'INITIALIZATION.json']:
                files.append(str(Path('data') / panel / name))
            for split in ['VALIDATION', 'TEST']:
                path = folder / (split+'_SELECTIVITY.tsv')
                if path.exists():
                    for row in rows(path):
                        metrics.append(dict(panel=panel, split=split.lower(), **row))
            sources.append(dict(panel=panel, data_receipt=sha(root / 'data' / panel / 'COMPLETE.json')
                     if (root / 'data' / panel / 'COMPLETE.json').exists() else None))
        table(root / 'ALL_PANEL_SELECTIVITY.tsv', metrics)
        finished = all(x['complete_receipt'] for x in statuses)
        dump(root / 'SUMMARY.json', dict(status='complete' if finished else 'partial', models=statuses,
             sources=sources, interpretation='Compare supported target-specific ranking, calibration and capture consistency within each task. AP across different label panels does not define a universal winner.',
             steps_to_design=['Review validation-selected targets and measured test precision.',
                  'Select a biologically acceptable target and competitors; shared target unions are reported separately.',
                  'Review TRAINING_SEEDS and capture support, then test200bp context dependence before sequence optimization.'],
             synthetic_sequences_designed=False, original_runs_modified=False))
        files += ['ALL_PANEL_SELECTIVITY.tsv', 'SUMMARY.json']
        for path in sorted((root / 'jobs').glob('*.out')) + sorted((root / 'jobs').glob('*.err')):
            dest = root / 'jobs' / (path.name+'.tail.txt')
            dest.write_text('\n'.join(path.read_text(errors='replace').splitlines()[-100:])+'\n')
            files.append(str(dest.relative_to(root)))
        temporary = root / 'SHARE_BORZOI_TARGET_PANELS.zip.partial'
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in sorted(set(files)):
                if (root / name).is_file():
                    z.write(root / name, name)
        temporary.replace(root / 'SHARE_BORZOI_TARGET_PANELS.zip')
        print('SHARE ' + str(root / 'SHARE_BORZOI_TARGET_PANELS.zip'), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
