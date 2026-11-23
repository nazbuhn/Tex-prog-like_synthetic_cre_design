import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
import gc
import time
import numpy as np
from util import need, read, dump, verify, checkpoint, sha, table, progress
from math_core import binary, count_scores, pooled, rc_flag, background_indices


def summarize(observed, predicted, counts, meta):
    results, scores = [], {}
    if meta['task'] == 'count':
        for name, group, positive, score, extra in count_scores(observed, predicted, counts, meta):
            results.append(dict(target=name, task='measured_accessibility_selectivity', **extra, **binary(positive, score)))
            scores[name] = (group, positive, score)
    else:
        exclusive = observed.sum(axis=1) == 1
        for j, name in enumerate(meta['names']):
            others = np.delete(np.arange(len(meta['names'])), j)
            score = np.log(np.maximum(predicted[:, j], 1e-7)) - np.log(np.maximum(predicted[:, others].max(axis=1), 1e-7))
            positive = (observed[:, j] > .5) & exclusive
            membership = binary(observed[:, j] > .5, predicted[:, j])
            results.append(dict(target=name, task='exclusive_topic_membership', membership_AP=membership['AP'], **binary(positive, score)))
            scores[name] = ([j], positive, score)
    eligible = [r['AP'] for r in results if r['target'] in meta['checkpoint_targets'] and r['AP'] is not None]
    if meta['task'] == 'count':
        calibration = float(np.square(np.log1p(observed)-np.log1p(predicted)).mean())
    else:
        p = np.clip(predicted, 1e-7, 1-1e-7)
        calibration = float(-(observed*np.log(p)+(1-observed)*np.log1p(-p)).mean())
    return results, scores, dict(macro_selectivity_AP=float(np.mean(eligible)) if eligible else None, calibration_error=calibration)


def capture_support(folder, peak_ids, group, offgroups=None):
    info = read(folder / 'CAPTURE_GROUPS.json')
    matrix = np.load(folder / 'capture_counts.npy', mmap_mode='r')
    result = []
    for capture in sorted({x['capture'] for x in info}):
        targets = [i for i, x in enumerate(info) if x['capture'] == capture and x['group'] in group]
        others = [i for i, x in enumerate(info) if x['capture'] == capture and x['group'] not in group and x['cells'] >= 30 and
                  (offgroups is None or x['group'] in offgroups)]
        cells = sum(info[i]['cells'] for i in targets)
        if cells < 30 or not others:
            continue
        target_count = np.asarray(matrix[np.ix_(peak_ids, targets)]).sum(axis=1)
        exposure = sum(info[i]['exposure'] for i in targets)
        target_rate = target_count/exposure*1e6
        other_rate = np.asarray(matrix[np.ix_(peak_ids, others)])/np.array([info[i]['exposure'] for i in others])[None, :]*1e6
        other_count = np.asarray(matrix[np.ix_(peak_ids, others)]).sum(axis=1)
        fold = (target_rate+.1)/(other_rate.max(axis=1)+.1)
        result.append((capture, target_count, fold, other_count))
    if not result:
        return [dict(informative_captures=0, supporting_captures=0, support_fraction=None) for _ in peak_ids]
    rows = []
    for j in range(len(peak_ids)):
        eligible = [x for x in result if x[1][j]+x[3][j] >= 10]
        support = sum(x[1][j] >= 3 and x[2][j] >= 2 for x in eligible)
        rows.append(dict(informative_captures=len(eligible), supporting_captures=int(support),
                         support_fraction=support/len(eligible) if eligible else None))
    return rows


def main(root, panel):
    import torch
    import keras
    from torch.utils.data import Dataset, DataLoader
    from model import build, stable_count_loss, save, native_numeric_check
    root = Path(root)
    verify(root, 'BUNDLE_SHA256.json')
    settings = read(root / 'SETTINGS.json')
    need(settings['batch_size'] == 32, 'This experiment fixes batch size at32')
    preparation = read(root / 'PREPARATION_COMPLETE.json')
    need(sha(root / 'SETTINGS.json') == preparation['settings_sha256'], 'Settings changed after preparation')
    need(sha(root / 'data/COMPLETE.json') == preparation['data_receipt_sha256'], 'Preparation receipt changed')
    need(sha(root / 'splits.npy') == preparation['splits_sha256'], 'Chromosome split changed')
    data, out = root / 'data' / panel, root / 'models' / panel
    out.mkdir(parents=True, exist_ok=True)
    verify(data)
    meta = read(data / 'DATASET.json')
    if (out / 'COMPLETE.json').exists():
        verify(out)
        return
    if not meta['selection_possible']:
        dump(out / 'SUMMARY.json', dict(status='insufficient_validation_support', panel=panel,
             reason='No >=200-cell target with>=50training and>=10validation selective regions, or no supported exclusive topic.',
             thresholds_relaxed=False, model_trained=False, sequence_design_performed=False))
        checkpoint(out, ['SUMMARY.json'])
        return
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one allocated GPU required')
    props = torch.cuda.get_device_properties(0)
    need('B200' in props.name and props.total_memory > 150e9, 'A full B200 allocation is required')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    need(keras.mixed_precision.global_policy().name == 'float32', 'Expected FP32 training')
    native_numeric_check(out)
    need(sha(root / 'tokens2048.npy') == read(root / 'PREPARATION_COMPLETE.json')['tokens_sha256'], 'DNA cache changed')
    initial = read(data / 'INITIALIZATION.json')
    need(sha(data / 'INITIAL.weights.h5') == initial['sha256'], 'Initial pretrained weights changed')
    tokens = np.load(root / 'tokens2048.npy', mmap_mode='r')
    targets = np.load(data / 'targets.npy', mmap_mode='r')
    rates = np.load(data / 'rates.npy', mmap_mode='r') if meta['task'] == 'count' else targets
    counts = np.load(data / 'counts.npy', mmap_mode='r') if meta['task'] == 'count' else None
    splits = np.load(root / 'splits.npy')
    training_ids, validation_ids = np.flatnonzero(splits == 0), np.flatnonzero(splits == 1)
    eye = np.r_[np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)]

    class Encoded(Dataset):
        def __init__(self, indices, epoch=None):
            self.indices, self.epoch = indices, epoch

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, k):
            i = int(self.indices[k])
            x = eye[tokens[i]]
            if self.epoch is not None and rc_flag(i, self.epoch, settings['seed']):
                x = x[::-1, ::-1].copy()
            return x, np.array(targets[i], dtype=np.float32, copy=True)

    def predict(model, indices, label):
        p = np.empty((len(indices), len(meta['names'])), np.float32)
        loader = DataLoader(Encoded(indices), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
        offset = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                a, b = model(x, training=False), model(torch.flip(x, [1, 2]), training=False)
                if meta['task'] == 'topic':
                    a, b = torch.sigmoid(a), torch.sigmoid(b)
                values = (a+b)/2
                need(torch.isfinite(values).all().item() and (values >= 0).all().item(), 'Invalid predictions')
                p[offset:offset+len(x)] = values.detach().cpu().numpy()
                offset += len(x)
                if offset % 6400 == 0 or offset == len(indices):
                    progress(out, label, completed=offset, total=len(indices))
        if meta['task'] == 'count':
            p /= np.array(meta['weights'])[None, :]
        return p

    def evaluate(model, indices, label):
        prediction = predict(model, indices, label)
        result, scores, overall = summarize(np.asarray(rates[indices]), prediction,
                       np.asarray(counts[indices]) if counts is not None else None, meta)
        return prediction, result, scores, overall

    keras.utils.set_random_seed(settings['seed'])
    model = build(len(meta['names']), meta['task']).to(device)
    model.load_weights(data / 'INITIAL.weights.h5')
    model.trainable = True
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=settings['learning_rate'], eps=1e-7)
    state_path = out / 'STATE.json'
    if state_path.exists():
        saved = read(state_path)
        need(sha(out / saved['resume_file']) == saved['resume_sha256'], 'Resume checkpoint changed')
        state = torch.load(out / saved['resume_file'], weights_only=True, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, wait, plateau, best = saved['epoch'], saved['wait'], saved['plateau'], saved['best']
        need(sha(out / best['weights_file']) == best['sha256'], 'Committed best weights changed')
        history = [r for r in read(out / 'HISTORY.json') if r['epoch'] <= start]
    else:
        _, _, _, baseline = evaluate(model, validation_ids, 'initial_validation')
        h = save(model, out, 'INITIAL_BASELINE')
        best = dict(epoch=0, metrics=baseline, weights_file='INITIAL_BASELINE.weights.h5', sha256=h)
        dump(out / 'BASELINE.json', best)
        start = wait = plateau = 0
        history = []
    for epoch in range(start, settings['max_epochs']):
        if wait >= settings['early_stopping_patience']:
            break
        tick = time.monotonic()
        torch.manual_seed(settings['seed']+epoch)
        generator = torch.Generator().manual_seed(settings['seed']+epoch)
        loader = DataLoader(Encoded(training_ids, epoch), batch_size=32, shuffle=True, generator=generator,
                            num_workers=0, pin_memory=True, drop_last=True)
        used_lr = optimizer.param_groups[0]['lr']
        loss_sum = seen = 0
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            p = model(x, training=True)
            need(torch.isfinite(p).all().item(), 'Nonfinite forward predictions')
            loss = stable_count_loss(y, p) if meta['task'] == 'count' else torch.nn.functional.binary_cross_entropy_with_logits(p, y)
            for term in model.losses:
                loss = loss + term
            need(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            if step == 1:
                need(any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in model.get_layer('stem_conv').parameters()), 'Missing trainable trunk gradient')
                need(any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in model.get_layer('panel_output').parameters()), 'Missing output-head gradient')
            torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())*len(y)
            seen += len(y)
            if step == 1 or step % 200 == 0:
                progress(out, 'training', panel=panel, epoch=epoch+1, max_epochs=settings['max_epochs'],
                         batch=step, total_batches=len(loader), batch_size=32, learning_rate=used_lr,
                         loss=float(loss.detach()), allocated_GPU_GB=torch.cuda.memory_allocated()/1e9)
        _, validation, _, metrics = evaluate(model, validation_ids, 'epoch_validation')
        score = (metrics['macro_selectivity_AP'], -metrics['calibration_error'])
        old_score = (best['metrics']['macro_selectivity_AP'], -best['metrics']['calibration_error'])
        improved = best['epoch'] == 0 or score > old_score
        if improved:
            stem = 'EPOCH_'+str(epoch+1).zfill(3)
            h = save(model, out, stem)
            best = dict(epoch=epoch+1, metrics=metrics, weights_file=stem+'.weights.h5', sha256=h)
            wait = plateau = 0
        else:
            wait += 1
            plateau += 1
        if plateau >= settings['lr_patience']:
            for group in optimizer.param_groups:
                group['lr'] = max(settings['minimum_lr'], group['lr']*.5)
            plateau = 0
        row = dict(epoch=epoch+1, train_loss=loss_sum/seen, **metrics, improved=improved,
                   learning_rate=used_lr, batch_size=32, seconds=time.monotonic()-tick, wait=wait)
        history.append(row)
        dump(out / 'HISTORY.json', history)
        table(out / 'LATEST_VALIDATION.tsv', validation)
        resume_name = 'RESUME_'+str(epoch+1).zfill(3)+'.pt'
        temp = out / (resume_name+'.partial')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), temp)
        temp.replace(out / resume_name)
        previous = read(state_path) if state_path.exists() else None
        dump(state_path, dict(epoch=epoch+1, wait=wait, plateau=plateau, best=best,
             resume_file=resume_name, resume_sha256=sha(out / resume_name)))
        # This run owns these transient optimizer files; the previous committed one
        # is removed only after its successor and receipt are committed.
        if previous and previous['resume_file'] != resume_name:
            (out / previous['resume_file']).unlink(missing_ok=True)
        print('EPOCH ' + __import__('json').dumps(row), flush=True)
    need(sha(out / best['weights_file']) == best['sha256'], 'Selected checkpoint changed')
    dump(out / 'BEST.json', best)
    model.load_weights(out / best['weights_file'])
    vp, vr, _, _ = evaluate(model, validation_ids, 'final_validation')
    table(out / 'VALIDATION_SELECTIVITY.tsv', vr)
    # Freeze the target choices before any test metrics are computed.
    seed_targets = sorted([r for r in vr if r['target'] in meta['checkpoint_targets']],
                          key=lambda r: (r['top50_precision'], r['AP']), reverse=True)[:5]
    dump(out / 'SEED_TARGETS.json', dict(chosen_on='validation only', targets=[r['target'] for r in seed_targets],
         validation_metrics=seed_targets, generating_training_seed_pool_is_not_design_approval=True))
    test_ids = np.flatnonzero(splits == 2)
    tp, tr, tscores, _ = evaluate(model, test_ids, 'test_prediction')
    table(out / 'TEST_SELECTIVITY.tsv', tr)
    if meta['task'] == 'count':
        all_groups, _, _ = summarize(np.asarray(rates[test_ids]), tp, np.asarray(counts[test_ids]),
                                    dict(meta, include_ambiguous_backgrounds=True))
        table(out / 'TEST_ALL_GROUPS_SENSITIVITY.tsv', all_groups)
    np.savez_compressed(out / 'PREDICTIONS.npz', validation_ids=validation_ids, validation_predictions=vp,
                        test_ids=test_ids, test_predictions=tp)
    region_names = np.array((root / 'REGIONS.txt').read_text().splitlines())
    profiles = []
    for target, (group, positive, scores) in tscores.items():
        top = np.lexsort((test_ids, -scores))[:100]
        corroboration = capture_support(data, test_ids[top], group, background_indices(meta, group)) if meta['task'] == 'count' else [{} for _ in top]
        for rank, (j, support) in enumerate(zip(top, corroboration), 1):
            activity = {}
            if meta['task'] == 'count':
                others = background_indices(meta, group)
                activity = dict(predicted_target_FRPM=float(pooled(tp[j:j+1], meta['exposure'], group)[0]),
                     predicted_strongest_other_FRPM=float(tp[j, others].max()),
                     measured_target_FRPM=float(pooled(np.asarray(rates[test_ids[j:j+1]]), meta['exposure'], group)[0]),
                     measured_strongest_other_FRPM=float(rates[test_ids[j], others].max()),
                     measured_target_fragments=int(counts[test_ids[j], group].sum()))
            profiles.append(dict(target=target, rank=rank, region=region_names[test_ids[j]],
                 global_index=int(test_ids[j]), measured_selective=bool(positive[j]),
                 score=float(scores[j]), **activity, **support))
    table(out / 'TEST_TOP_REGIONS.tsv.gz', profiles)
    # Only training regions can enter the future editing seed pool.
    trainp, _, trainscores, _ = evaluate(model, training_ids, 'training_seed_prediction')
    seeds = []
    for selected in seed_targets:
        target = selected['target']
        group, positive, scores = trainscores[target]
        ii = np.flatnonzero(positive)
        ii = ii[np.lexsort((training_ids[ii], -scores[ii]))][:50]
        corroboration = capture_support(data, training_ids[ii], group, background_indices(meta, group)) if meta['task'] == 'count' else [{} for _ in ii]
        for rank, (j, support) in enumerate(zip(ii, corroboration), 1):
            seeds.append(dict(target=target, rank=rank, global_index=int(training_ids[j]), region=region_names[training_ids[j]],
                 split='training', score=float(scores[j]), measured_selective=True, **support))
    table(out / 'TRAINING_SEEDS.tsv.gz', seeds)
    dump(out / 'SUMMARY.json', dict(status='complete', panel=panel, task=meta['task'], best=best,
         cells=111347, outputs=len(meta['names']), input_bp=2048, measured_peak_bp=500, eventual_insert_bp=200,
         epochs_completed=len(history), checkpoint_selection='Validation macro AP of supported selective targets; calibration breaks ties.',
         test_used_for_checkpoint_or_target_selection=False, native_numeric_check='PASS', batch_size=32,
         all_peak_training=True, high_Gini_only_stage=False, seed_targets=[r['target'] for r in seed_targets],
         sequence_design_performed=False, automatic_design_approval=False,
         warnings=['State-selective accessibility is not measured enhancer activity.',
                   'Reference labels are transferred annotations; original Foxp3 subgroup restored to its individual parent labels.',
                   'Saved clustering/topics and consensus peaks were informed by the full dataset; existing test chromosomes have been reviewed.',
                   'Borzoi pretraining overlap with held-out DNA is not excluded.',
                   'Capture support is descriptive; captures are not guaranteed biological replicates and no formal DA p-values are computed.',
                   'Primary count comparisons exclude ambiguous CD8/not-classified/pooled-small groups; TEST_ALL_GROUPS_SENSITIVITY reports them as competitors too. These exclusions do not establish specificity in those cells.',
                   '2048bp input does not establish standalone200bp specificity.',
                   'Topic metrics concern latent DNA programs and are not directly comparable to cell-state metrics.']))
    complete_files = ['SUMMARY.json', 'BEST.json', best['weights_file'], 'VALIDATION_SELECTIVITY.tsv',
         'TEST_SELECTIVITY.tsv', 'PREDICTIONS.npz', 'TEST_TOP_REGIONS.tsv.gz', 'SEED_TARGETS.json',
         'TRAINING_SEEDS.tsv.gz', 'NATIVE_NUMERIC_CHECK.json']
    if meta['task'] == 'count':
        complete_files.append('TEST_ALL_GROUPS_SENSITIVITY.tsv')
    checkpoint(out, complete_files)
    del model, optimizer, trainp
    keras.backend.clear_session()
    gc.collect()
    torch.cuda.empty_cache()
    progress(out, 'complete', panel=panel)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
