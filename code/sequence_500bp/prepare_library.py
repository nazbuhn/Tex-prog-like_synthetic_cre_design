from pathlib import Path
import numpy as np
from design_core import shared_context_ids, dna, DESIGN_LEN
from io_helpers import need, read, sha, reuse, checkpoint, dump, rows, table, progress


def pilot_initializations(seed_tokens, seeds, designs, rng):
    """Select distinct natural parents and equal composition-matched random starts."""
    if designs != 32 or designs % 4:
        raise ValueError('This pilot requires exactly 32 designs balanced across arms and floors')
    need(len(seed_tokens) == len(seeds) == 50, 'Expected 50 verified natural parent sequences')
    center = (seed_tokens.shape[1] - DESIGN_LEN)//2
    candidates = []
    for parent in range(len(seeds)):
        for offset in (0, -100, 100, -200, 200):
            start = center + offset
            segment = seed_tokens[parent, start:start+DESIGN_LEN]
            if len(segment) == DESIGN_LEN and (segment < 4).all():
                candidates.append((parent, offset, segment.copy()))
                break  # One start per parent, closest available to the center.
    need(len(candidates) >= designs//2, 'Too few 500-bp ACGT-only starting regions')
    chosen = rng.choice(len(candidates), size=designs//2, replace=False)
    starts = []
    plan = []
    for idx in chosen:
        parent, offset, seq = candidates[int(idx)]
        starts.append(seq)
        plan.append(dict(arm='natural', parent_id=seeds[parent]['seed_id'], offset=int(offset),
                         supporting_captures=seeds[parent]['supporting_captures'],
                         informative_captures=seeds[parent]['informative_captures']))
    n_natural = len(starts)
    probabilities = np.bincount(np.asarray(starts).ravel(), minlength=4).astype(float)
    probabilities /= probabilities.sum()
    keys = {dna(seq) for seq in starts}
    for j in range(designs-n_natural):
        for _ in range(100):
            seq = rng.choice(4, DESIGN_LEN, p=probabilities).astype(np.uint8)
            if dna(seq) not in keys:
                break
        else:
            raise RuntimeError('Could not generate distinct random starts')
        starts.append(seq)
        keys.add(dna(seq))
        plan.append(dict(arm='random', parent_id=f'random_{j+1:04d}', offset='',
                         supporting_captures='', informative_captures=''))
    need(len(starts) == designs, 'Incorrect pilot starting sequence count')
    return starts, plan


def main(root):
    if reuse(root, 'PREPARE_COMPLETE'):
        return
    s = read(root/'SETTINGS.json'); pilot = Path(s['pilot_run']); source = Path(s['source'])
    # V3 always prepares new 500-bp inputs; never reuse frozen V1/V2 200-bp arrays.
    need(s.get('prepared_source_run') is None, '500-bp pilot must prepare new inputs')
    need(s['target_len'] == DESIGN_LEN, 'Target length does not match optimizer')
    need(s['designs'] == 32, 'This package is the 32-design plateau pilot')
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
    seeds = rows(pilot/'private/SEEDS.tsv')
    starts, plan = pilot_initializations(seed_tokens, seeds, s['designs'], rng)
    n_natural = s['designs'] // 2
    retention = np.empty(s['designs'], dtype=np.float64)
    for start, stop in [(0, n_natural), (n_natural, s['designs'])]:
        block = np.tile([.25, .5], (stop-start)//2)
        rng.shuffle(block)
        retention[start:stop] = block
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
         natural_parent_peaks=len({r['parent_id'] for r in plan if r['arm'] == 'natural'}), natural_windows=n_natural, random_starts=s['designs']-n_natural,
         source_tokens_sha256=prep['tokens_sha256'], source_splits_sha256=prep['splits_sha256'],
         training_only=True, prior_pilot_contexts_excluded=old_context_ids,
         motif_inputs_used=False, cell_membership_changed=False, models_retrained=False,
         caveat='Contexts are training-chromosome DNA backgrounds, not independent biological samples. '
                'Sixteen non-overlapping parent identities sampled from 50 verified original parents. Prior tests already inspected.'))
    checkpoint(root, 'PREPARE_COMPLETE', [root/'private/INPUTS.npz', root/'DESIGN_PLAN.tsv',
                root/'CONTEXTS.tsv', root/'INPUT_PROVENANCE.json'])
    progress(root, 'prepared', designs=s['designs'], natural_parent_peaks=n_natural, fresh_contexts=nctx)
