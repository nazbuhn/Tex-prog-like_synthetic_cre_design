import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
import time
import inspect
import importlib.metadata as metadata
import numpy as np
from core import objective, qc
from design_core import insert, dna, DESIGN_LEN, N_SINGLE_MUTANTS
from native_design import evolve, native_api
from io_helpers import need, read, dump, rows, table, sha, reuse, checkpoint, progress, verify_bundle, receipt_file


def predict(scorer, contexts, sequences):
    seq = np.asarray(sequences, np.uint8)
    if seq.ndim == 1:
        seq = seq[None]
    # Limit transient host memory even for the complete native mutant batch.
    output = []
    for begin in range(0, len(seq), 32):
        block = seq[begin:begin+32]
        output.append(scorer.predict(insert(contexts, block)).reshape(len(block), len(contexts), -1))
    return np.concatenate(output)


def preflight(root, s, scorer):
    if reuse(root, 'NATIVE_COMPLETE'):
        return
    source = Path(s['source']); model = source/'models/atac_C1'
    receipt_file(model, 'COMPLETE.json', 'PREDICTIONS.npz')
    with np.load(model/'PREDICTIONS.npz') as z:
        ids = z['validation_ids'][:4]; expected = z['validation_predictions'][:4]
    tokens = np.load(source/'tokens2048.npy', mmap_mode='r')
    actual = scorer.predict(tokens[ids])
    need(np.allclose(actual, expected, rtol=.003, atol=.0003), 'Checkpoint prediction replay failed')
    z = np.load(root/'private/INPUTS.npz'); initial = z['initial']; contexts = z['context_tokens']
    natural = predict(scorer, contexts, initial[:s['designs']//2])
    reference = np.median(natural[..., scorer.target], axis=0)
    need((reference > 0).all(), 'Zero reference activity')
    np.save(root/'private/REFERENCE_PRIMARY.npy', reference)
    floor = np.maximum(.25*predict(scorer, contexts[:2], initial[0])[0, :, scorer.target], .25*reference[:2])
    begin = time.monotonic()
    result = evolve(scorer, contexts[:2], initial[0], floor, steps=3, capture_replay=True)
    seconds = time.monotonic() - begin
    replay = predict(scorer, contexts[:2], result['trajectory'])
    recorded = result['predictions']
    matched = [replay[0]]
    for frames, positions in zip(result['replay_frames'], result['replay_positions']):
        matched.append(np.asarray([scorer.predict(frame)[int(pos)] for frame, pos in zip(frames, positions)]))
    matched = np.asarray(matched)
    repeated = predict(scorer, contexts[:2], result['trajectory'])
    matched_ok = np.allclose(matched, recorded, rtol=1e-5, atol=1e-6)
    repeat_ok = np.allclose(repeated, replay, rtol=1e-5, atol=1e-6)
    isolated_ok = np.allclose(replay, recorded, rtol=s['isolated_replay_rtol'], atol=s['isolated_replay_atol'])
    diagnostics = dict(status='PASS' if matched_ok and repeat_ok and isolated_ok else 'FAIL',
         exact_native_sequence_and_score_mapping=True, preflight_iterations=3,
         same_batch_replay_pass=bool(matched_ok), isolated_repeat_pass=bool(repeat_ok),
         isolated_cross_batch_replay_pass=bool(isolated_ok),
         original_strict_isolated_check_pass=bool(np.allclose(replay, recorded, rtol=1e-5, atol=1e-6)),
         same_batch_max_absolute_difference=float(np.abs(matched-recorded).max()),
         isolated_max_absolute_difference=float(np.abs(replay-recorded).max()),
         isolated_max_relative_difference=float((np.abs(replay-recorded)/np.maximum(np.abs(recorded), 1e-8)).max()),
         same_batch_tolerance=dict(rtol=1e-5, atol=1e-6),
         cross_batch_tolerance=dict(rtol=s['isolated_replay_rtol'], atol=s['isolated_replay_atol']),
         cross_batch_tolerance_basis='Same tolerance as existing saved-checkpoint validation replay; a changed cross-batch numerical gate, not proof of the original failure cause.')
    dump(root/'REPLAY_DIAGNOSTICS.json', diagnostics)
    np.savez_compressed(root/'private/PREFLIGHT_REPLAY.npz', trajectory=result['trajectory'],
                        recorded=recorded, same_batch_replay=matched, isolated_replay=replay, repeated_isolated=repeated)
    progress(root, 'native_replay_diagnostics', **diagnostics)
    need(matched_ok and repeat_ok and isolated_ok,
         'Replay diagnostic failed; see REPLAY_DIAGNOSTICS.json; library remains blocked')
    ise, _ = native_api(); import torch
    dump(root/'NATIVE_PREFLIGHT.json', dict(status='PASS', native_function=ise.__module__+'.'+ise.__name__,
         native_source_sha256=sha(inspect.getsourcefile(ise)), adapter='500bp -> fixed2048bp contexts -> RC mean normalized FRPM',
         target_index=scorer.target, class_names=scorer.names,
         mutations_forward_evaluated_per_step=N_SINGLE_MUTANTS, gradient_shortlist_used=False, known_motifs_used=False,
         checkpoint_replay_max_difference=float(np.abs(actual-expected).max()),
         native_trajectory_replayed=True, replay_diagnostics=diagnostics,
         benchmark_seconds_three_steps=seconds, benchmark_seconds_one_step=seconds/3,
         rough_design_only_GPU_hours=seconds/3*s['steps']*s['designs']/3600,
         estimate_caveat='Three-step warm benchmark; excludes queue, model loads, evaluation and variation in throughput.',
         versions={n: metadata.version(n) for n in ['crested', 'keras', 'torch', 'numpy']}, GPU=torch.cuda.get_device_name(0)))
    checkpoint(root, 'NATIVE_COMPLETE', [root/'NATIVE_PREFLIGHT.json', root/'private/REFERENCE_PRIMARY.npy',
               root/'REPLAY_DIAGNOSTICS.json', root/'private/PREFLIGHT_REPLAY.npz'])
    progress(root, 'native_preflight_passed', **read(root/'NATIVE_PREFLIGHT.json'))


def design(root, s, scorer, shard):
    need(reuse(root, 'PREPARE_COMPLETE') and reuse(root, 'NATIVE_COMPLETE'), 'Preflight incomplete')
    z = np.load(root/'private/INPUTS.npz'); contexts = z['context_tokens'][:2]; starts = z['initial']
    reference = np.load(root/'private/REFERENCE_PRIMARY.npy')[:2]
    plan = [r for r in rows(root/'DESIGN_PLAN.tsv') if int(r['shard']) == shard]
    need(len(plan) == s['designs_per_shard'], 'Shard size differs')
    for r in plan:
        folder = root/'private'/r['design_id']; folder.mkdir(exist_ok=True)
        if reuse(folder, 'DESIGN_COMPLETE'):
            continue
        i = int(r['index']); p = predict(scorer, contexts, starts[i])[0]
        floor = np.maximum(float(r['retention_fraction'])*p[:, scorer.target], .25*reference)
        begin = time.monotonic()
        def report(step):
            progress(folder, 'native_ISE', design=r['design_id'], step=step, total_steps=s['steps'],
                     variants_per_step=N_SINGLE_MUTANTS, elapsed_seconds=round(time.monotonic()-begin, 1))
        result = evolve(scorer, contexts, starts[i], floor, s['steps'], report)
        step = int(result['selected_step']); selected = result['trajectory'][step]
        result_path = folder/'RESULT.partial.npz'
        np.savez_compressed(result_path, **result); result_path.replace(folder/'RESULT.npz')
        history = [dict(design_id=r['design_id'], iteration=j, sequence_500bp=dna(seq),
                   objective=float(objective(result['predictions'][j], scorer.target, floor)),
                   target_floor_met=bool((result['predictions'][j, :, scorer.target] >= floor-1e-7).all()),
                   selected=j == step) for j, seq in enumerate(result['trajectory'])]
        table(folder/'PRIVATE_TRAJECTORY.tsv', history)
        dump(folder/'DESIGN_SUMMARY.json', dict(design_id=r['design_id'], native_iterations=s['steps'],
             selected_step=step, selected_feasible=bool(result['feasible']),
             mutations_from_initial=int((selected != starts[i]).sum()),
             seconds=time.monotonic()-begin, **qc(selected)))
        checkpoint(folder, 'DESIGN_COMPLETE', [folder/'RESULT.npz', folder/'PRIVATE_TRAJECTORY.tsv', folder/'DESIGN_SUMMARY.json'])
        progress(folder, 'complete', **read(folder/'DESIGN_SUMMARY.json'))
    dump(root/f'SHARD_{shard:03d}_COMPLETE.json', dict(shard=shard, designs=len(plan)))


def main(root, stage, shard=None):
    verify_bundle(root); s = read(root/'SETTINGS.json')
    need(s['target_len'] == DESIGN_LEN, 'Settings and design code length disagree')
    if stage == 'prepare':
        from prepare_library import main as prepare
        prepare(root); return
    if stage == 'collect':
        from evaluate import collect
        collect(root); return
    import torch
    torch.set_num_threads(8)
    need(torch.cuda.is_available(), 'GPU required; refusing silent CPU fallback')
    from scorers import primary
    scorer = primary(s, 'cuda')
    if stage == 'preflight':
        preflight(root, s, scorer)
    elif stage == 'design':
        design(root, s, scorer, int(shard))
    elif stage == 'evaluate':
        from evaluate import main as evaluate
        evaluate(root, s, scorer)
    else:
        raise ValueError('Unknown stage '+stage)


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
