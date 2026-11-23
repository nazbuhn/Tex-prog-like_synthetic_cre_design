from pathlib import Path
from datetime import datetime, timezone
import fcntl
import os
import re
import shutil
import subprocess
import sys
from util import need, read, dump, verify


def terminal(job):
    p = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State'],
                       capture_output=True, text=True, timeout=20)
    need(p.returncode == 0, 'Accounting unavailable; no retry submitted')
    # -X and -j restrict this result to this allocation/array. Include pending
    # array ranges as well as individual tasks; ignoring a range could duplicate it.
    records = [r.split('|') for r in p.stdout.splitlines() if len(r.split('|')) >= 2 and r.split('|')[1].strip()]
    need(records, 'No unambiguous accounting rows; inspect before retrying')
    return [r[1].split()[0].rstrip('+') for r in records]


def job(root, stage, resources, dependency=None, resume=False):
    folder = root / 'jobs'
    folder.mkdir(exist_ok=True)
    identity = folder / (stage+'_JOB_ID.txt')
    journal = folder / (stage+'_SUBMISSION.json')
    if identity.exists():
        recorded = identity.read_text().strip()
        need(recorded.isdigit(), 'Invalid recorded job ID')
        if not resume:
            print('Existing '+stage+' job '+recorded+'; no duplicate submitted.', flush=True)
            return recorded
        states = terminal(recorded)
        terminal_states = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
        if any(s not in terminal_states for s in states):
            print('Existing '+stage+' job remains active; keeping it.', flush=True)
            return recorded
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        identity.rename(folder / (identity.name+'.previous_'+stamp))
        if journal.exists():
            journal.rename(folder / (journal.name+'.previous_'+stamp))
    need(not journal.exists(), 'Earlier submission has an unknown outcome: '+str(journal))
    cmd = ['sbatch', '--parsable', '--job-name=Borzoi_panels_'+stage, '--nodes=1', '--ntasks=1',
           '--chdir='+str(root), '--output='+str(folder/(stage+'_%A_%a.out')),
           '--error='+str(folder/(stage+'_%A_%a.err'))] + resources
    if dependency:
        cmd += ['--dependency='+dependency, '--kill-on-invalid-dep=yes']
    if os.environ.get('IMMGENT_SLURM_ACCOUNT'):
        cmd.append('--account='+os.environ['IMMGENT_SLURM_ACCOUNT'])
    cmd += [str(root / 'compute.sh'), str(root), stage]
    dump(journal, dict(status='submission_started_outcome_not_known', command=cmd))
    print('Calling sbatch for '+stage+'; waiting for controller...', flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (folder / (stage+'_SUBMIT.stdout')).write_text(p.stdout)
    (folder / (stage+'_SUBMIT.stderr')).write_text(p.stderr)
    print(p.stderr, flush=True)
    lines = p.stdout.strip().splitlines()
    matched = re.fullmatch(r'(\d+)(?:;\S+)?', lines[-1]) if lines else None
    need(p.returncode == 0 and matched is not None, 'No confirmed ID. Inspect submission files before retrying.')
    value = matched.group(1)
    identity.write_text(value+'\n')
    dump(journal, dict(status='submitted', job_id=value, command=cmd))
    print('SUBMITTED '+stage+' '+value, flush=True)
    return value


def main(source, all_topics=False, resume=False):
    source = Path(source).resolve()
    manifest = verify(source, 'BUNDLE_SHA256.json')
    settings = read(source / 'DEFAULTS.json')
    settings['topic_ks'] = [20, 40, 80, 100] if all_topics else [40]
    settings['panels'] = ['reference_parent', 'immgent', 'atac_C1'] + ['topics_K'+str(k) for k in settings['topic_ks']]
    root = Path(settings['run'] + ('_all_topics' if all_topics else ''))
    for path in [Path(settings['reference']) / 'full/PREPARE_COMPLETE.json',
                 Path(settings['old_borzoi']) / 'PREPARE_COMPLETE.json', Path(settings['env']) / 'bin/python']:
        need(path.is_file(), 'Required completed input missing: '+str(path))
    for k in settings['topic_ks']:
        need((Path(settings['topic_audit']) / ('K'+str(k)) / 'COMPLETE.json').is_file(), 'Saved topic audit cache is missing for K'+str(k))
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'submission.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another launcher is active; no duplicate submitted.')
            return
        if (root / 'BUNDLE_SHA256.json').exists():
            need(read(root / 'BUNDLE_SHA256.json') == manifest, 'Another bundle is frozen in this run')
            verify(root, 'BUNDLE_SHA256.json')
            need(read(root / 'SETTINGS.json') == settings, 'Frozen settings differ')
        else:
            for name in [*manifest, 'BUNDLE_SHA256.json']:
                shutil.copy2(source / name, root / name)
            dump(root / 'SETTINGS.json', settings)
        (root / 'data').mkdir(exist_ok=True)
        (root / 'models').mkdir(exist_ok=True)
        if (root / 'data/COMPLETE.json').exists():
            verify(root / 'data')
            preparation = None
        else:
            preparation = job(root, 'prepare', ['--partition=genoa-std-mem', '--cpus-per-task=16', '--mem-per-cpu=5632M', '--time=08:00:00'], resume=resume)
        missing = [i for i, panel in enumerate(settings['panels']) if not (root / 'models' / panel / 'COMPLETE.json').exists()]
        for i, panel in enumerate(settings['panels']):
            if i not in missing:
                verify(root / 'models' / panel)
        if missing:
            array = ','.join(map(str, missing))+'%1'
            gpu = job(root, 'train', ['--partition=dgx-b200', '--cpus-per-task=28', '--gpus=1', '--array='+array, '--time=24:00:00'],
                      dependency='afterok:'+preparation if preparation else None, resume=resume)
        else:
            gpu = None
        job(root, 'collect', ['--partition=genoa-std-mem', '--cpus-per-task=4', '--mem-per-cpu=5632M', '--time=01:00:00'],
            dependency='afterany:'+gpu if gpu else None, resume=resume)
    print('Run: '+str(root), flush=True)
    print('One full B200 at a time. Panels: '+', '.join(settings['panels']), flush=True)
    print('Check: bash "'+str(root / 'CHECK_STATUS.sh')+'" "'+str(root)+'"', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], '--all-topics' in sys.argv[2:], '--resume' in sys.argv[2:])
