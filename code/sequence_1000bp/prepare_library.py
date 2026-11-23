"""Fresh 1000-bp library preparation, never reuse any 200/500-bp design arrays."""
from pathlib import Path
import numpy as np
from design_core import shared_context_ids, dna, DESIGN_LEN
from io_helpers import need, read, sha, reuse, checkpoint, dump, rows, table, progress


def library_initializations(seed_tokens, seeds, designs, rng):
    """Diverse starts: ~one-to-three offset windows per eligible parent, 1:1 random arm.

    All starts are based on the same 50 vetted natural parent peaks as the old pilot.
    Natural windows from the same parent are correlated, never independent replicates.
    """
    need(designs > 0 and designs % 4 == 0, 'Design count must be divisible by four')
    need(seed_tokens.ndim == 2 and seed_tokens.shape[1] == 2048, 'Parent tokens must be 2048bp')
    need(len(seed_tokens) == len(seeds) == 50, 'Expected 50 verified natural parents')
    need(DESIGN_LEN <= 2048, 'Design length exceeds model input')
    center = (seed_tokens.shape[1] - DESIGN_LEN)//2
    offsets = (0, -400, 400, -200, 200, -500, 500, -300, 300, -100, 100)
    pools = []
    for parent in range(50):
        candidates = []
        seen = set()
        for off in offsets:
            start = center + off
            if start < 0 or start + DESIGN_LEN > 2048:
                continue
            seq = seed_tokens[parent, start:start+DESIGN_LEN].copy()
            if (seq < 4).all():
                key = dna(seq)
                if key not in seen:
                    seen.add(key)
                    candidates.append((parent,off,seq))
        pools.append(candidates)
    eligible = [i for i in range(50) if pools[i]]
    need(len(eligible) >= 2, 'No diverse ACGT-only natural parents at 1000bp')
    rng.shuffle(eligible)
    # Round-robin parents for breadth; no replacement of identical parent windows.
    selected = []
    used_keys = set()
    for round_number in range(max(map(len,pools))):
        for parent in eligible:
            if round_number >= len(pools[parent]):
                continue
            pp, off, seq = pools[parent][round_number]
            key = dna(seq)
            if key not in used_keys:
                selected.append((pp,off,seq))
                used_keys.add(key)
            if len(selected) >= designs//2:
                break
        if len(selected) >= designs//2:
            break
    need(len(selected) == designs//2,
         f'Only {len(selected)} distinct natural windows; requested {designs//2}. '
         'Do not silently replace with random DNA.')
    starts = []
    plan = []
    for parent, off, seq in selected:
        starts.append(seq)
        plan.append(dict(arm='natural',parent_id=seeds[parent]['seed_id'],offset=int(off),
                         supporting_captures=seeds[parent]['supporting_captures'],
                         informative_captures=seeds[parent]['informative_captures']))
    probabilities=np.bincount(np.asarray(starts).ravel(),minlength=4).astype(float)
    probabilities/=probabilities.sum()
    for j in range(designs//2):
        for _ in range(100):
            seq=rng.choice(4,DESIGN_LEN,p=probabilities).astype(np.uint8)
            if dna(seq) not in used_keys:
                break
        else:
            raise RuntimeError('Could not generate unique random start')
        starts.append(seq)
        used_keys.add(dna(seq))
        plan.append(dict(arm='random',parent_id=f'random_{j+1:04d}',offset='',
                         supporting_captures='',informative_captures=''))
    need(len(starts) == designs, 'Library size does not match settings')
    return starts, plan


def main(root):
    if reuse(root,'PREPARE_COMPLETE'):
        return
    s=read(root/'SETTINGS.json'); pilot=Path(s['pilot_run']); source=Path(s['source'])
    need(s.get('prepared_source_run') is None, 'Must prepare new 1000bp design inputs')
    need(s['target_len'] == DESIGN_LEN, 'Length setting does not match native optimizer')
    need(s['designs'] % 4 == 0 and s['designs'] % s['designs_per_shard'] == 0,
         'Unbalanced library or incomplete design shard')
    need(reuse(pilot,'PREPARE_COMPLETE') and reuse(pilot,'COMPLETE'), 'Verified portable pilot required')
    ps=read(pilot/'SETTINGS.json')
    for key in ('source','expected_primary_weights','expected_primary_dataset','expected_seeds_sha256'):
        need(s[key] == ps[key], 'Primary model/pilot provenance differs: '+key)
    progress(root,'verify_training_DNA_and_splits')
    prep=read(source/'PREPARATION_COMPLETE.json')
    for name,key in (('tokens2048.npy','tokens_sha256'),('splits.npy','splits_sha256')):
        need(sha(source/name)==prep[key], 'Training source changed: '+name)
    tokens=np.load(source/'tokens2048.npy',mmap_mode='r')
    splits=np.load(source/'splits.npy')
    meta=read(source/'data/atac_C1/DATASET.json')
    need(sha(source/'data/atac_C1/DATASET.json')==s['expected_primary_dataset'], 'Primary metadata changed')
    with np.load(pilot/'private/INPUTS.npz') as z:
        seed_ids=z['seed_ids'].copy()
        old_context_ids=z['context_ids'].copy()
        seed_tokens=z['seed_tokens'].copy()
    need(len(seed_ids)==50 and (splits[seed_ids]==0).all(), 'Parent peaks not training only')
    need(np.array_equal(seed_tokens,tokens[seed_ids]), 'Parent token cache changed')
    verified=read(pilot/'INPUT_PROVENANCE.json')['source_SHA256']
    for name in ('rates.npy','counts.npy'):
        path=source/'data/atac_C1'/name
        need(str(path) in verified and sha(path)==verified[str(path)], 'Measured ATAC source changed: '+name)
    rates=np.load(source/'data/atac_C1/rates.npy',mmap_mode='r')
    counts=np.load(source/'data/atac_C1/counts.npy',mmap_mode='r')
    need(rates.shape==counts.shape==(len(tokens),len(meta['names'])), 'Accessibility matrix shape changed')
    nctx=sum(s[k] for k in ('optimization_contexts','selection_contexts','audit_contexts'))
    rng=np.random.default_rng(s['seed'])
    ids=shared_context_ids(rates,counts,splits,meta['cells'],np.r_[seed_ids,old_context_ids],nctx,rng)
    need(not np.isin(ids,old_context_ids).any(), 'Previously reviewed pilot context reused')
    contexts=tokens[ids].copy()
    seeds=rows(pilot/'private/SEEDS.tsv')
    starts,plan=library_initializations(seed_tokens,seeds,s['designs'],rng)
    n_natural=s['designs']//2
    retention=np.empty(s['designs'],dtype=np.float64)
    for begin,stop in ((0,n_natural),(n_natural,s['designs'])):
        block=np.tile([.25,.5],(stop-begin)//2)
        rng.shuffle(block)
        retention[begin:stop]=block
    for j,r in enumerate(plan):
        r.update(index=j,design_id=f'C1L_{j+1:04d}',retention_fraction=float(retention[j]),
                 shard=j//s['designs_per_shard'])
    (root/'private').mkdir(exist_ok=True)
    np.savez_compressed(root/'private/INPUTS.npz',initial=np.asarray(starts),context_tokens=contexts,
                        context_ids=ids,seed_ids=seed_ids,pilot_context_ids=old_context_ids)
    table(root/'DESIGN_PLAN.tsv',plan)
    roles=['optimization']*s['optimization_contexts']+['selection']*s['selection_contexts']+['final_audit']*s['audit_contexts']
    table(root/'CONTEXTS.tsv',[dict(context_id=f'fresh_{j+1:02d}',global_index=int(i),role=roles[j],
                                    split='training',reviewed_in_pilot=False) for j,i in enumerate(ids)])
    nparents=len({r['parent_id'] for r in plan if r['arm']=='natural'})
    dump(root/'INPUT_PROVENANCE.json',dict(target_cells=4440,target='Frozen ATAC resolution0.3 C1',
         natural_parent_peaks=nparents,natural_windows=n_natural,random_starts=s['designs']-n_natural,
         source_tokens_sha256=prep['tokens_sha256'],source_splits_sha256=prep['splits_sha256'],
         training_only=True,prior_pilot_contexts_excluded=old_context_ids,
         motif_inputs_used=False,cell_membership_changed=False,models_retrained=False,
         caveat='Contexts are TRAINING chromosome backgrounds, not independent biological samples. '
                'Multiple overlapping windows from 50 original natural parent peaks are correlated.'))
    checkpoint(root,'PREPARE_COMPLETE',[root/'private/INPUTS.npz',root/'DESIGN_PLAN.tsv',
              root/'CONTEXTS.tsv',root/'INPUT_PROVENANCE.json'])
    progress(root,'prepared',designs=s['designs'],natural_parent_peaks=nparents,fresh_contexts=nctx)
