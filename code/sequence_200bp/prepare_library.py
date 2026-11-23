from pathlib import Path
import shutil
import numpy as np
from design_core import shared_context_ids, dna
from io_helpers import need, read, sha, reuse, checkpoint, dump, rows, table, progress


def main(root):
    if reuse(root, 'PREPARE_COMPLETE'):
        return
    s = read(root/'SETTINGS.json'); pilot = Path(s['pilot_run']); source = Path(s['source'])
    if s.get('prepared_source_run'):
        previous = Path(s['prepared_source_run'])
        need(reuse(previous, 'PREPARE_COMPLETE'), 'Original preparation receipt missing')
        hashes = read(previous/'PREPARE_COMPLETE.json')['sha256']
        need(hashes == s['expected_prepared_hashes'], 'Original preparation differs from uploaded verified receipt')
        old_settings = read(previous/'SETTINGS.json')
        for key, value in old_settings.items():
            if key != 'run':
                need(s[key] == value, 'Scientific settings changed: '+key)
        (root/'private').mkdir(exist_ok=True)
        for name, digest in hashes.items():
            need(sha(previous/name) == digest, 'Prepared source changed: '+name)
            shutil.copy2(previous/name, root/name)
            need(sha(root/name) == digest, 'Prepared copy differs: '+name)
        dump(root/'PREPARATION_REUSE.json', dict(source_run=str(previous), sha256=hashes,
             starts_regenerated=False, contexts_reselected=False))
        checkpoint(root, 'PREPARE_COMPLETE', [root/name for name in hashes])
        progress(root, 'prepared_inputs_reused', designs=s['designs'], source_run=str(previous))
        return
    need(reuse(pilot, 'PREPARE_COMPLETE') and reuse(pilot, 'COMPLETE'), 'Completed portable pilot required')
    ps = read(pilot/'SETTINGS.json')
    for key in ['source', 'secondary_source', 'expected_primary_weights', 'expected_primary_dataset',
                'expected_secondary_weights', 'expected_secondary_dataset', 'expected_seeds_sha256']:
        need(s[key] == ps[key], 'Pilot/model provenance differs: '+key)
    progress(root, 'verify_training_DNA_and_splits')
    prep = read(source/'PREPARATION_COMPLETE.json')
    for name, key in [('tokens2048.npy', 'tokens_sha256'), ('splits.npy', 'splits_sha256')]:
        need(sha(source/name) == prep[key], 'Training source changed: '+name)
    tokens = np.load(source/'tokens2048.npy', mmap_mode='r'); splits = np.load(source/'splits.npy')
    meta = read(source/'data/atac_C1/DATASET.json')
    need(sha(source/'data/atac_C1/DATASET.json') == s['expected_primary_dataset'], 'Primary metadata changed')
    with np.load(pilot/'private/INPUTS.npz') as z:
        seed_ids = z['seed_ids'].copy(); old_context_ids = z['context_ids'].copy()
        seed_tokens = z['seed_tokens'].copy()
    need(len(seed_ids) == 50 and (splits[seed_ids] == 0).all(), 'Nontraining seeds')
    need(np.array_equal(seed_tokens, tokens[seed_ids]), 'Seed cache/parent axis mismatch')
    rng = np.random.default_rng(s['seed'])
    verified = read(pilot/'INPUT_PROVENANCE.json')['source_SHA256']
    for name in ['rates.npy', 'counts.npy']:
        path = source/'data/atac_C1'/name
        need(str(path) in verified and sha(path) == verified[str(path)], 'Measured accessibility cache changed')
    rates = np.load(source/'data/atac_C1/rates.npy', mmap_mode='r')
    counts = np.load(source/'data/atac_C1/counts.npy', mmap_mode='r')
    need(rates.shape == counts.shape == (len(tokens), len(meta['names'])), 'Rate axes differ')
    nctx = sum(s[k] for k in ['optimization_contexts', 'selection_contexts', 'audit_contexts'])
    ids = shared_context_ids(rates, counts, splits, meta['cells'],
                             np.r_[seed_ids, old_context_ids], nctx, rng)
    need(not np.isin(ids, old_context_ids).any(), 'Old reviewed contexts reused')
    contexts = tokens[ids].copy()
    seeds = rows(pilot/'private/SEEDS.tsv'); starts = []; plan = []
    candidates = [(j, off) for j in range(50) for off in range(0, 301, 20)
                  if seed_tokens[j, 774+off:974+off].max() < 4]
    # Round-robin parents, shuffled offsets: many windows, explicitly only 50 natural parents.
    pools = [[(j, off) for jj, off in candidates if jj == j] for j in range(50)]
    for pool in pools:
        rng.shuffle(pool)
    natural = [pool[t] for t in range(16) for pool in pools if len(pool) > t][:s['designs']//2]
    need(len(natural) == 512, 'Insufficient unique training windows; no resampling fallback')
    for j, off in natural:
        starts.append(seed_tokens[j, 774+off:974+off].copy())
        plan.append(dict(arm='natural', parent_id=seeds[j]['seed_id'], offset=off,
                         supporting_captures=seeds[j]['supporting_captures'],
                         informative_captures=seeds[j]['informative_captures']))
    # GC/composition matched to the natural start pool; zero motif/PWM inputs.
    probabilities = np.bincount(np.asarray(starts).ravel(), minlength=4) / (512*200)
    keys = {dna(seq) for seq in starts}
    for j in range(512):
        for _ in range(100):
            seq = rng.choice(4, 200, p=probabilities).astype(np.uint8)
            if dna(seq) not in keys:
                break
        else:
            raise RuntimeError('Could not generate distinct random starts')
        starts.append(seq); keys.add(dna(seq))
        plan.append(dict(arm='random', parent_id=f'random_{j+1:04d}', offset='',
                         supporting_captures='', informative_captures=''))
    need(len(plan) == s['designs'] == 1024, 'Library size differs')
    retention = np.tile(np.repeat([.25, .5], 256), 2)
    rng.shuffle(retention[:512]); rng.shuffle(retention[512:])
    for j, r in enumerate(plan):
        r.update(index=j, design_id=f'C1L_{j+1:04d}',
                 retention_fraction=float(retention[j]),
                 shard=j // s['designs_per_shard'])
    (root/'private').mkdir(exist_ok=True)
    np.savez_compressed(root/'private/INPUTS.npz', initial=np.asarray(starts),
                        context_tokens=contexts, context_ids=ids, seed_ids=seed_ids,
                        pilot_context_ids=old_context_ids)
    table(root/'DESIGN_PLAN.tsv', plan)
    roles = ['optimization']*s['optimization_contexts'] + ['selection']*s['selection_contexts'] + ['final_audit']*s['audit_contexts']
    table(root/'CONTEXTS.tsv', [dict(context_id=f'fresh_{j+1:02d}', global_index=int(i), role=roles[j],
                                   split='training', reviewed_in_pilot=False) for j, i in enumerate(ids)])
    dump(root/'INPUT_PROVENANCE.json', dict(target_cells=4440, target='Frozen ATAC resolution0.3 C1',
         natural_parent_peaks=50, natural_windows=512, random_starts=512,
         source_tokens_sha256=prep['tokens_sha256'], source_splits_sha256=prep['splits_sha256'],
         training_only=True, prior_pilot_contexts_excluded=old_context_ids,
         motif_inputs_used=False, cell_membership_changed=False, models_retrained=False,
         caveat='Contexts are training-chromosome DNA backgrounds, not independent biological samples. '
                'Natural windows overlap within 50 parent peaks. Discovery and prior tests already inspected.'))
    checkpoint(root, 'PREPARE_COMPLETE', [root/'private/INPUTS.npz', root/'DESIGN_PLAN.tsv',
                root/'CONTEXTS.tsv', root/'INPUT_PROVENANCE.json'])
    progress(root, 'prepared', designs=1024, natural_parent_peaks=50, fresh_contexts=nctx)
