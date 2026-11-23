"""Lightweight Slurm orchestration. No GPU work or large data reads at submission."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ACCOUNT = 'saargill-cell-type-specific'


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, x):
    p = Path(p)
    temp = p.with_name(p.name + '.partial')
    temp.write_text(json.dumps(x, indent=2) + '\n')
    temp.replace(p)


def check_package():
    import hashlib
    for name, expected in read(P / 'BUNDLE_SHA256.json').items():
        h = hashlib.sha256((P / name).read_bytes()).hexdigest()
        if h != expected:
            raise RuntimeError('Package checksum changed: ' + name)


def job(run, stage, args=(), gpu=False, array=None, dependency=None, large_cpu=False):
    cfg = read(run / 'CONFIG.json')
    command = ['sbatch', '--parsable', '--account=' + ACCOUNT, '--nodes=1', '--ntasks=1',
        '--chdir=' + str(run), '--output=' + str(run / 'logs' / (stage + '_%A_%a.out')),
        '--error=' + str(run / 'logs' / (stage + '_%A_%a.err')),
        '--job-name=C1J26_' + stage]
    if gpu:
        command += ['--partition=b200-mig45', '--qos=mig', '--cpus-per-task=6',
                    '--mem=48G', '--gres=gpu:1', '--time=' + cfg['gpu_time']]
    else:
        cpus = int(cfg.get('cpu_modisco_threads',16)) if (stage == 'modisco' or stage == 'cluster') else (4 if large_cpu else 2)
        command += ['--partition=genoa-std-mem', '--qos=genoa-std',
                    '--cpus-per-task=' + str(cpus), '--mem=' + str(cpus*5632) + 'M',
                    '--time=' + cfg['cpu_time']]
    if array:
        command += ['--array=' + array]
    if dependency:
        command += ['--dependency=afterany:' + str(dependency)]
    command += [str(P / 'RUN_STAGE.sh'), str(run), stage] + list(map(str, args))
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    # PARCC filters may print banners. --parsable job ID is a numeric line.
    lines = [s.strip().split(';')[0] for s in result.stdout.splitlines()]
    ids = [s for s in lines if s.isdigit()]
    if len(ids) != 1:
        raise RuntimeError('Unexpected sbatch response: ' + result.stdout)
    jid = ids[0]
    with (run / 'SUBMISSIONS.jsonl').open('a') as f:
        f.write(json.dumps(dict(job_id=jid, stage=stage, command=command,
                              time=datetime.now(timezone.utc).isoformat())) + '\n')
    print(stage + ': ' + jid, flush=True)
    return jid


def active(run):
    path = run / 'SUBMISSIONS.jsonl'
    if not path.exists():
        return []
    ids = [readrow['job_id'] for readrow in map(json.loads, path.read_text().splitlines())]
    current = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', ''))
    result = subprocess.run(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%T'],
                            text=True, capture_output=True, check=True, timeout=30)
    return [line for line in result.stdout.splitlines()
            if line.split('|')[0].split('_')[0] in ids and
               line.split('|')[0].split('_')[0] != current]


def queue_array(run, phase, ids, gpu):
    cfg = read(run / 'CONFIG.json')
    last = None
    for lo in range(0, len(ids), cfg['array_batch_size']):
        chunk = ids[lo:lo+cfg['array_batch_size']]
        path = run / ('TASK_IDS_' + phase + '_' + str(time.time_ns()) + '.json')
        dump(path, chunk)
        throttle = cfg['gpu_array_concurrency'] if gpu else 2
        last = job(run, phase, [path], gpu=gpu,
                   array=f'0-{len(chunk)-1}%{throttle}', dependency=last)
    # afterany lets the dispatcher audit actual output files even if an array fails.
    job(run, 'dispatch', [phase], dependency=last)


def missing(run, phase):
    # Imported only on compute nodes; submission stays standard-library only.
    import workflow as w
    if phase == 'predict':
        return [i for i in range(len(read(run / 'PREDICT_TASKS.json')))
                if not w.valid(run / 'predictions' / f'{i:05d}.json')]
    if phase == 'explain':
        return [i for i in range(len(read(run / 'EXPLAIN_TASKS.json')))
                if not w.task_done(run, i)]
    if phase == 'modisco':
        return [i for i, x in enumerate(read(run / 'MODISCO_TASKS.json'))
                if not w.valid(run / 'modisco' / x['method'] /
                               (x['label'] + '_modisco_results.json'))]
    if phase == 'reports':
        return [i for i,x in enumerate(read(run / 'MODISCO_TASKS.json'))
                if not w.valid(run / 'reports' / x['method'] / (x['label']+'.json'))]
    raise RuntimeError('Unknown phase ' + phase)


def advance(run, checked_phase=None):
    import workflow as w
    cfg = read(run / 'CONFIG.json')
    if checked_phase:
        lost = missing(run, checked_phase)
        if lost:
            dump(run / 'NEEDS_RESUME.json', dict(stage=checked_phase, incomplete_tasks=lost))
            raise RuntimeError(f'{len(lost)} incomplete {checked_phase} tasks. '
                               'Inspect logs, then run RESUME_BETTY.sh. No automatic retry loop.')
    if not w.valid(run / 'PREPARE_COMPLETE.json'):
        job(run, 'prepare', large_cpu=True)
    elif not w.valid(run / 'GPU_PREFLIGHT_COMPLETE.json'):
        job(run, 'gpucheck', gpu=True)
    elif missing(run, 'predict'):
        queue_array(run, 'predict', missing(run, 'predict'), True)
    elif not w.valid(run / 'SELECT_COMPLETE.json'):
        job(run, 'select', large_cpu=True)
    elif missing(run, 'explain'):
        queue_array(run, 'explain', missing(run, 'explain'), True)
    elif not (run / 'MODISCO_TASKS.json').exists():
        job(run, 'assemble', large_cpu=True)
    elif missing(run, 'modisco'):
        queue_array(run, 'modisco', missing(run, 'modisco'), False)
    elif missing(run, 'reports'):
        queue_array(run, 'reports', missing(run, 'reports'), False)
    elif not all(w.valid(run / 'clustered' / m / 'COMPLETE.json') for m in cfg['methods']):
        job(run, 'cluster', large_cpu=True)
    elif not w.valid(run / 'RNA_COMPLETE.json'):
        job(run, 'rna', large_cpu=True)
    elif not w.valid(run / 'ANALYSIS_COMPLETE.json'):
        job(run, 'finish', large_cpu=True)
    elif not (run / 'SHARE_C1_CRESTED_JASPAR2026_FULL_ANALYSIS.zip').is_file():
        job(run, 'finish', large_cpu=True)
    else:
        print('Complete: ' + str(run / 'SHARE_C1_CRESTED_JASPAR2026_FULL_ANALYSIS.zip'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['new', 'advance', 'resume', 'status'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--check')
    a = parser.parse_args()
    if a.action == 'new':
        check_package()
        import references
        references.verify(P / 'references')
        old_package = P.parent / 'C1_CRESTED_ANALYSIS_BETTY_V1'
        if (old_package / 'LATEST_RUN.txt').exists():
            old_run = Path((old_package / 'LATEST_RUN.txt').read_text().strip())
            if active(old_run):
                raise RuntimeError('The previous C1 CREsted workflow still has active jobs. No duplicate workflow submitted; inspect those exact jobs first.')
        if (P / 'LATEST_RUN.txt').exists():
            print('Existing run found. Use CHECK_STATUS.sh or RESUME_BETTY.sh; no duplicate created.')
            return
        run = P / 'runs' / ('native_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + str(os.getpid()))
        (run / 'logs').mkdir(parents=True)
        (run / 'CONFIG.json').write_bytes((P / 'CONFIG.json').read_bytes())
        (run / 'PACKAGE_SHA256.json').write_bytes((P / 'BUNDLE_SHA256.json').read_bytes())
        (P / 'LATEST_RUN.txt').write_text(str(run) + '\n')
        job(run, 'prepare', large_cpu=True)
        print('Run: ' + str(run))
        return
    run = a.run or Path((P / 'LATEST_RUN.txt').read_text().strip())
    if a.action == 'resume':
        if active(run):
            raise RuntimeError('This run still has active jobs. No duplicate submitted.')
        # Setup may have failed before creating the analysis interpreter.
        # Re-enter prepare directly in that case; dispatch itself needs it.
        if not (P / '.analysis-env/bin/python').is_file() or not (run / 'PREPARE_COMPLETE.json').is_file():
            job(run, 'prepare', large_cpu=True)
        else:
            # Dispatch on CPU: large completeness checks do not run on login.
            job(run, 'dispatch', ['resume'])
    elif a.action == 'advance':
        advance(run, a.check)
    else:
        print('Run: ' + str(run))
        print('Active jobs: ' + str(active(run)))
        if (run / 'ANALYSIS_COMPLETE.json').exists():
            print((run / 'ANALYSIS_COMPLETE.json').read_text())
        for name in ['FAILURE.json', 'NEEDS_RESUME.json']:
            if (run / name).exists():
                print('Saved diagnostic (may predate a successful resume): ' + (run / name).read_text())
        archive = run / 'SHARE_C1_CRESTED_JASPAR2026_FULL_ANALYSIS.zip'
        print(('Archive: ' if archive.is_file() else 'Expected archive (not created yet): ') + str(archive))


if __name__ == '__main__':
    main()
