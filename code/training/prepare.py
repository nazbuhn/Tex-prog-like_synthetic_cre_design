import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import gc
import shutil
import sys
import numpy as np
from scipy import sparse
from util import need, read, dump, sha, rows, table, verify, checkpoint, progress
from math_core import label_panel, original_states, count_scores


def aggregate(matrix, positions, codes, exposure, n_groups, folder, capture=None):
    total = matrix.shape[1]
    maps = sparse.csr_matrix((np.ones(len(positions), np.int64), (positions, codes)), shape=(total, n_groups))
    denoms = np.bincount(codes, weights=exposure, minlength=n_groups)
    cells = np.bincount(codes, minlength=n_groups)
    need((denoms > 0).all(), 'Group has no library exposure')
    out = np.lib.format.open_memmap(folder / 'counts.partial.npy', mode='w+', dtype=np.int64, shape=(matrix.shape[0], n_groups))
    if capture is not None:
        keys = sorted(set(zip(codes.tolist(), capture.tolist())))
        key_lookup = {k: i for i, k in enumerate(keys)}
        pair = np.array([key_lookup[(g, c)] for g, c in zip(codes, capture)], dtype=int)
        cmap = sparse.csr_matrix((np.ones(len(positions), np.int64), (positions, pair)), shape=(total, len(keys)))
        cap = np.lib.format.open_memmap(folder / 'capture_counts.partial.npy', mode='w+', dtype=np.int64, shape=(matrix.shape[0], len(keys)))
        capmeta = [dict(group=int(g), capture=c, cells=int((pair == i).sum()),
                   exposure=float(exposure[pair == i].sum())) for i, (g, c) in enumerate(keys)]
    for start in range(0, matrix.shape[0], 4096):
        block = matrix[start:start+4096].astype(np.int64)
        out[start:start+len(block.indptr)-1] = (block @ maps).toarray()
        if capture is not None:
            cap[start:start+len(block.indptr)-1] = (block @ cmap).toarray()
        if start % (4096*20) == 0:
            progress(folder, 'aggregate_counts', completed=start, total=matrix.shape[0])
    out.flush()
    del out
    (folder / 'counts.partial.npy').replace(folder / 'counts.npy')
    if capture is not None:
        cap.flush()
        del cap
        (folder / 'capture_counts.partial.npy').replace(folder / 'capture_counts.npy')
        dump(folder / 'CAPTURE_GROUPS.json', capmeta)
    return cells, denoms


def count_panel(root, panel, values, matrix, positions, exposure, capture, splits, target=None):
    import anndata as ad
    import crested
    folder = root / 'data' / panel
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETE.json').exists():
        verify(folder)
        return
    names, codes = label_panel(values, target=target)
    cells, denoms = aggregate(matrix, positions, codes, exposure, len(names), folder, capture)
    counts = np.load(folder / 'counts.npy', mmap_mode='r')
    rates = (counts / denoms[None, :] * 1e6).astype(np.float32)
    reliable = cells >= 200
    need(reliable.sum() >= 3, 'Too few sufficiently sampled groups')
    norm = ad.AnnData(rates[splits == 0][:, reliable].T.copy())
    norm.var_names = np.flatnonzero(splits == 0).astype(str)
    norm.obs_names = np.array(names)[reliable]
    constitutive = crested.pp.normalize_peaks(norm, top_k_percent=.03)
    w = np.asarray(norm.obsm['weights']).reshape(-1)
    need(np.isfinite(w).all() and (w > 0).all(), 'Invalid normalization')
    need(np.allclose(norm.X, rates[splits == 0][:, reliable].T*w[:, None], rtol=1e-5, atol=1e-5), 'Native normalization semantics differ')
    ci = np.asarray(constitutive.index, int)
    need(len(ci) > 0 and (splits[ci] == 0).all(), 'Normalization used nontraining regions')
    weights = np.ones(len(names), np.float32)
    weights[reliable] = w
    reference = float(np.median((rates[ci][:, reliable]*w).mean(axis=0)))
    for j in np.flatnonzero(~reliable):
        mean = float(rates[ci, j].mean())
        need(mean > 0, 'Small output has no constitutive accessibility')
        weights[j] = reference / mean
    np.save(folder / 'rates.npy', rates)
    np.save(folder / 'targets.npy', rates*weights[None, :])
    np.save(folder / 'cell_codes.npy', codes)
    meta = dict(panel=panel, task='count', names=names, cells=cells, exposure=denoms, weights=weights,
                cohort_cells=111347, all_cells_retained=True, fitting_captures='all original cohort captures',
                normalization='Native constitutive-peak scaling, training chromosomes only; >=200-cell groups fit scale; smaller outputs aligned to that scale.',
                independent_capture_holdout=False, biological_labels_validated=False)
    selection = []
    val = splits == 1
    for title, group, positive, _, _ in count_scores(rates, rates, counts, meta):
        train_pos = int(positive[splits == 0].sum())
        val_pos = int(positive[val].sum())
        unknown_name = any(names[j].lower() in ['not classified', 'unknown', 'cd8', 'other_small_atac'] for j in group)
        eligible = int(cells[group].sum()) >= 200 and train_pos >= 50 and val_pos >= 10 and not unknown_name
        selection.append(dict(target=title, training_positives=train_pos, validation_positives=val_pos,
                              eligible_for_checkpoint_selection=eligible))
    meta['checkpoint_targets'] = [x['target'] for x in selection if x['eligible_for_checkpoint_selection']]
    # Do not silently relax evidence thresholds to make a panel appear learnable.
    meta['selection_possible'] = bool(meta['checkpoint_targets'])
    dump(folder / 'DATASET.json', meta)
    table(folder / 'SELECTION_TARGETS.tsv', selection)
    checkpoint(folder, ['DATASET.json', 'counts.npy', 'rates.npy', 'targets.npy', 'cell_codes.npy',
                        'capture_counts.npy', 'CAPTURE_GROUPS.json', 'SELECTION_TARGETS.tsv'])


def topic_panel(root, k, settings, cellmeta, regions):
    folder = root / 'data' / ('topics_K'+str(k))
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'COMPLETE.json').exists():
        verify(folder)
        return
    audit = Path(settings['topic_audit'])
    old = audit / ('K'+str(k))
    receipt = verify(old)
    need('OVERLAP_CACHE.npz' in receipt['sha256'], 'Saved topic cache is not in its receipt')
    source_meta = rows(audit / 'CELL_METADATA.tsv.gz')
    need(len(source_meta) == 111347, 'Topic cohort differs')
    index = {r['Cell_ID']: i for i, r in enumerate(source_meta)}
    positions = np.array([index[r['Cell_ID']] for r in cellmeta])
    summary = read(old / 'SUMMARY.json')
    need(sha(audit / 'CELL_METADATA.tsv.gz') == summary['identity']['metadata_sha256'], 'Saved topic cell axis metadata changed')
    need(read(audit / 'SETTINGS.json')['common_manifest_sha256'] == settings['common_manifest_sha256'], 'Topic and count peak universes differ')
    need(summary['identity']['model_sha256'] == settings['topic_model_sha256'][str(k)], 'A different topic model was audited')
    need(summary['common_regions'] == len(regions), 'Topic peak axis differs')
    with np.load(old / 'OVERLAP_CACHE.npz') as z:
        y = z['region_otsu'].astype(np.float32)
        weights = z['cell_weights'][positions]
        high = z['cell_high'][positions]
    need(y.shape == (len(regions), k) and weights.shape == (111347, k), 'Topic axes differ')
    np.save(folder / 'targets.npy', y)
    names = ['Topic'+str(i+1) for i in range(k)]
    split = np.load(root / 'splits.npy')
    exclusive = (y > .5) & (y.sum(axis=1, keepdims=True) == 1)
    eligible = [names[j] for j in range(k) if int(exclusive[split == 0, j].sum()) >= 50 and int(exclusive[split == 1, j].sum()) >= 10]
    associations = []
    for scheme in ['ImmGenT', 'Reference_parent', 'ATAC_coarse', 'ATAC_fine']:
        labels = np.array([m[scheme] for m in cellmeta])
        for group in sorted(set(labels)):
            pick = labels == group
            for j, topic in enumerate(names):
                associations.append(dict(topic=topic, label_scheme=scheme, label=group, cells=int(pick.sum()),
                    mean_topic_weight=float(weights[pick, j].mean()),
                    fraction_of_high_topic_cells_in_group=float(high[pick, j].sum()/max(1, high[:, j].sum()))))
    table(folder / 'TOPIC_CELL_ASSOCIATIONS.tsv', associations)
    dump(folder / 'DATASET.json', dict(panel='topics_K'+str(k), task='topic', names=names,
         checkpoint_targets=eligible, selection_possible=bool(eligible),
         source_model=summary['identity'], cache_sha256=sha(old / 'OVERLAP_CACHE.npz'),
         cache_axis_source='Verified original common peak order; saved audit metadata matched by exact Cell_ID.',
         labels='Saved Otsu region memberships; overlapping positive labels are retained for BCE training.',
         selection_metric='Macro AP for exclusive-topic regions, not cell-state enhancer activity.',
         caveat='Saved topics were inferred using all regions and117653 cells. Chromosome evaluation predicts fixed transductive topic labels, not an independent discovery/generalization benchmark.'))
    checkpoint(folder, ['targets.npy', 'DATASET.json', 'TOPIC_CELL_ASSOCIATIONS.tsv'])


def main(root):
    root = Path(root)
    verify(root, 'BUNDLE_SHA256.json')
    settings = read(root / 'SETTINGS.json')
    need(settings['batch_size'] == 32, 'This matched experiment fixes batch size at32')
    if (root / 'data/COMPLETE.json').exists():
        verify(root / 'data')
        for panel in settings['panels']:
            verify(root / 'data' / panel)
        return
    need(shutil.disk_usage(root).free >= 100*1024**3, 'Less than100GiB free on the server filesystem; preparation stopped before writing large outputs. This check does not measure your project quota.')
    common = Path(settings['common'])
    need(sha(common / 'RUN_MANIFEST.json') == settings['common_manifest_sha256'], 'Common source manifest differs')
    recorded = read(common / 'RUN_MANIFEST.json')['output_SHA256']
    for name in ['ATAC_COUNTS.regions_by_cells.npz', 'ATAC_REGION_ORDER.tsv.gz', 'RETAINED_CELLS.tsv.gz']:
        progress(root, 'verify', file=name)
        need(sha(common / name) == recorded[name], 'Changed source: '+name)
    reference = Path(settings['reference']) / 'full'
    need(sha(reference / 'PREPARE_COMPLETE.json') == settings['reference_receipt_sha256'], 'Authenticated reference preparation differs')
    verify(reference, 'PREPARE_COMPLETE.json')
    old_borzoi = Path(settings['old_borzoi'])
    verify(old_borzoi, 'PREPARE_COMPLETE.json')
    sequence_info = read(old_borzoi / 'SEQUENCE_CHECK.json')
    need(sequence_info['central_500bp_preserved'], 'Existing sequence preparation differs')
    tokens = old_borzoi / 'tokens2048.npy'
    need(sha(tokens) == sequence_info['tokens_sha256'], 'Cached DNA tokens changed')
    regions = [r['region'] for r in rows(common / 'ATAC_REGION_ORDER.tsv.gz')]
    original = (reference / 'REGIONS.txt').read_text().splitlines()
    need(len(regions) == len(original) == 439167, 'Peak universe changed')
    for peak, window in zip(regions, original):
        c, ab = window.split(':')
        a, b = map(int, ab.split('-'))
        need(b-a == 2114 and peak == f'{c}:{a+807}-{b-807}', 'Sequence and measured peak rows are misaligned')
    splits = np.load(reference / 'splits.npy')
    need(np.array_equal(np.bincount(splits), [356938, 45032, 37197]), 'Chromosome partitions changed')
    for name, path in [('tokens2048.npy', tokens), ('splits.npy', reference / 'splits.npy')]:
        dest = root / name
        if not dest.exists():
            dest.symlink_to(path)
        need(sha(dest) == sha(path), 'Cached-input link differs')
    (root / 'REGIONS.txt').write_text('\n'.join(regions)+'\n')
    cellmeta = rows(root / 'CELL_METADATA.tsv.gz')
    auth = rows(reference / 'CELL_LABELS.tsv.gz')
    authenticated = {r['Cell_ID']: r for r in auth}
    need(len(cellmeta) == len(auth) == len(authenticated) == 111347, 'Cohort differs')
    ordered = [authenticated[r['Cell_ID']] for r in cellmeta]
    restored = original_states(ordered)
    override = sum(r['training_class'] == 'Foxp3_high_TIL' for r in ordered)
    need(override == 983, 'Reviewed Foxp3-high cell membership changed; inspect before proceeding')
    for m, a, state in zip(cellmeta, ordered, restored):
        need(m['scplus_capture'] == a['scplus_capture'], 'Capture differs across label sources')
        need(float(m['unique_fragments_count']) == float(a['unique_fragments_count']), 'Fragment exposure differs across label sources')
        m['Reference_parent'] = state
    table(root / 'MATCHED_CELL_LABELS.tsv.gz', cellmeta)
    table(root / 'FOXP3_RESTORATION.tsv', [dict(state=x, cells=sum(r['training_class'] == 'Foxp3_high_TIL' and r['gse267584_reference_state'] == x for r in ordered)) for x in sorted(set(restored))])
    retained = rows(common / 'RETAINED_CELLS.tsv.gz')
    lookup = {r['Cell_ID']: i for i, r in enumerate(retained)}
    pos = np.array([lookup[m['Cell_ID']] for m in cellmeta])
    exposure = np.array([float(m['unique_fragments_count']) for m in cellmeta])
    capture = np.array([m['scplus_capture'] for m in cellmeta])
    need(np.isfinite(exposure).all() and (exposure > 0).all(), 'Invalid fragment exposure')
    for m, i in zip(cellmeta, pos):
        need(retained[i]['scplus_capture'] == m['scplus_capture'], 'Common matrix cell alignment differs')
    matrix = sparse.load_npz(common / 'ATAC_COUNTS.regions_by_cells.npz').tocsr()
    need(matrix.shape == (439167, 117653) and matrix.dtype.kind in 'iu', 'Unexpected raw ATAC matrix')
    matrix.sum_duplicates()
    need(np.isfinite(matrix.data).all() and (matrix.data >= 0).all(), 'Invalid ATAC counts')
    for panel, values, target in [('reference_parent', restored, None),
                                  ('immgent', [m['ImmGenT'] for m in cellmeta], None),
                                  ('atac_C1', ['BG_R1_'+m['ATAC_fine'] for m in cellmeta],
                                   np.array([m['ATAC_coarse'] == 'C1' for m in cellmeta]))]:
        if target is not None:
            need(target.sum() == 4440, 'Original C1 target changed')
        count_panel(root, panel, values, matrix, pos, exposure, capture, splits, target)
    del matrix
    gc.collect()
    for k in settings['topic_ks']:
        topic_panel(root, k, settings, cellmeta, regions)
    # Verify the numerical repair before initializing any large model.
    from model import build, save, native_numeric_check
    import keras
    import crested
    native_numeric_check(root)
    archive, _ = crested.get_model('borzoi_mouse_rep0')
    need(sha(archive) == settings['pretrained_sha256'], 'Official checkpoint differs from the reviewed Borzoi experiment')
    for panel in settings['panels']:
        folder = root / 'data' / panel
        if (folder / 'INITIALIZATION.json').exists():
            record = read(folder / 'INITIALIZATION.json')
            need(sha(folder / 'INITIAL.weights.h5') == record['sha256'], 'Initial model changed')
            continue
        meta = read(folder / 'DATASET.json')
        keras.utils.set_random_seed(settings['seed'])
        model = build(len(meta['names']), meta['task'], archive, settings['pretrained_sha256'])
        pred = keras.ops.convert_to_numpy(model(np.zeros((1, 2048, 4), np.float32), training=False))
        need(pred.shape == (1, len(meta['names'])) and np.isfinite(pred).all(), 'Native model initialization failed')
        h = save(model, folder, 'INITIAL')
        dump(folder / 'INITIALIZATION.json', dict(sha256=h, pretrained_sha256=settings['pretrained_sha256'],
             parameters=model.count_params(), strict_pretrained_load=True, inherited_fine_tuned_weights=False,
             task=meta['task'], outputs=len(meta['names'])))
        del model, pred
        keras.backend.clear_session()
        gc.collect()
    files = [str(Path(panel) / name) for panel in settings['panels'] for name in ['COMPLETE.json', 'INITIALIZATION.json', 'INITIAL.weights.h5']]
    checkpoint(root / 'data', files, cells=111347, panels=settings['panels'], all_preparations_complete=True)
    dump(root / 'PREPARATION_COMPLETE.json', dict(data_receipt_sha256=sha(root / 'data/COMPLETE.json'),
         restored_Foxp3_cells=983, C1_cells=4440, tokens_sha256=sha(tokens), splits_sha256=sha(root / 'splits.npy'),
         settings_sha256=sha(root / 'SETTINGS.json')))
    progress(root, 'preparation_complete', panels=settings['panels'])


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
