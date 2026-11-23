"""Bounded-concurrency array, dependency-safe submission, resumable receipts."""
import fcntl
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from io_helpers import need, read, dump, verify_bundle, reuse


def job_active(job):
    p = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'], capture_output=True,
                       text=True, timeout=25, check=True)
    return bool(p.stdout.strip())


def submit_one(root, s, stage, dependency=None, array=False):
    record_path = root/f'SUBMISSION_{stage}.json'
    if record_path.exists():
        old = read(record_path)
        need(old.get('job_id'), 'Submission response uncertain: inspect '+str(record_path)+'; do not submit duplicate jobs')
        return old['job_id']
    gpu = stage in ['preflight', 'design', 'evaluate']
    args = ['sbatch', '--parsable', '--job-name=C1lib_'+stage, '--nodes=1', '--ntasks=1',
            '--chdir='+str(root), '--kill-on-invalid-dep=yes',
            '--output='+str(root/'logs'/f'{stage}_%A_%a.out'),
            '--error='+str(root/'logs'/f'{stage}_%A_%a.err')]
    if gpu:
        args += ['--partition=dgx-b200', '--gpus=1', '--cpus-per-task=28', '--mem-per-cpu=8G',
                 '--time='+('24:00:00' if stage in ['design', 'evaluate'] else '02:00:00')]
    else:
        args += ['--partition=genoa-std-mem', '--cpus-per-task=8', '--mem-per-cpu=5632M', '--time=02:00:00']
    if array:
        count = s['designs']//s['designs_per_shard']
        args += [f'--array=0-{count-1}%{s["max_concurrent_GPUs"]}']
    if dependency:
        args += ['--dependency='+dependency]
    if os.environ.get('IMMGENT_SLURM_ACCOUNT'):
        args += ['--account='+os.environ['IMMGENT_SLURM_ACCOUNT']]
    args += [str(root/'compute.sh'), str(root), stage]
    record = dict(stage=stage, status='submission_started_no_confirmed_ID', command=args)
    dump(record_path, record)
    print('Submitting '+stage+'; waiting for Slurm response...', flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
    p = subprocess.run(args, capture_output=True, text=True, env=env)
    (root/f'SUBMIT_{stage}.stdout').write_text(p.stdout); (root/f'SUBMIT_{stage}.stderr').write_text(p.stderr)
    if p.stderr:
        print(p.stderr, flush=True)
    lines = p.stdout.strip().splitlines()
    match = re.fullmatch(r'(\d+)(?:;\S+)?', lines[-1]) if lines else None
    need(p.returncode == 0 and match is not None, 'No confirmed job ID; inspect submission logs before retry')
    job = match.group(1); dump(record_path, dict(record, status='submitted', job_id=job))
    print('SUBMITTED '+stage+' '+job, flush=True)
    return job


def main(package, resume=False):
    package = Path(package).resolve(); verify_bundle(package); s = read(package/'SETTINGS.json')
    root = Path(s['run']); root.mkdir(parents=True, exist_ok=True)
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'BUNDLE_SHA256.json').exists():
            need(read(root/'BUNDLE_SHA256.json') == read(package/'BUNDLE_SHA256.json'), 'Different bundle in frozen run')
            verify_bundle(root)
        else:
            for name in read(package/'BUNDLE_SHA256.json'):
                shutil.copy2(package/name, root/name)
            shutil.copy2(package/'BUNDLE_SHA256.json', root/'BUNDLE_SHA256.json')
        (root/'logs').mkdir(exist_ok=True); (root/'private').mkdir(exist_ok=True)
        if reuse(root, 'EVALUATION_COMPLETE'):
            print('Already complete:', root/'SHARE_C1_500BP_PILOT32_METRICS.zip'); return
        existing = list(root.glob('SUBMISSION_*.json'))
        for p in existing:
            need(read(p).get('job_id'), 'Uncertain earlier submission: '+str(p))
        if resume and existing:
            # No automatic cancellation. Invalid-dependency children are submitted
            # with kill-on-invalid-dep so a failed parent cannot strand them forever.
            for p in existing:
                if job_active(read(p)['job_id']):
                    print('Existing pipeline still queued/running; no duplicate submission.'); return
            archive = root/'submission_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            archive.mkdir(parents=True)
            for p in existing:
                p.rename(archive/p.name)
        prepare = submit_one(root, s, 'prepare')
        preflight = submit_one(root, s, 'preflight', 'afterok:'+prepare)
        designs = submit_one(root, s, 'design', 'afterok:'+preflight, array=True)
        evaluation = submit_one(root, s, 'evaluate', 'afterok:'+designs)
        submit_one(root, s, 'collect', 'afterany:'+evaluation)
        print('Check: bash "'+str(root/'CHECK_STATUS.sh')+'"')


if __name__ == '__main__':
    main(sys.argv[1], '--resume' in sys.argv[2:])
