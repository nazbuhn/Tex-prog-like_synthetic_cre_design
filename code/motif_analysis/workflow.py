"""Native CREsted enhancer-code workflow. Execute on Betty, not a login-node GPU.

The only project-specific operations are loading the frozen checkpoint/cell axes,
inserting the designed DNA into its saved contexts, and writing audit tables.
Attribution, region selection, motif discovery, clustering and TF selection call
CREsted itself. No sequence-name/recognized-motif rules assign TF identities.
"""
import argparse
from functools import lru_cache
import copy
import csv
import hashlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import random
import subprocess
import shutil
import sys
import time
import zipfile

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parent


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def table(path, records, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=columns).to_csv(path, sep='\t', index=False)


def fasta(path):
    result = {}
    key = None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]
            need(key not in result, 'Duplicate FASTA identifier: ' + key)
            result[key] = ''
        elif line.strip():
            need(key is not None, 'FASTA sequence before header')
            result[key] += line.strip().upper()
    need(all(len(s) == 1000 and set(s) <= set('ACGT') for s in result.values()),
         'Expected unambiguous 1,000-bp inserts')
    return result


def config(run):
    return read(run / 'CONFIG.json')


def verify_package():
    for name, checksum in read(PACKAGE / 'BUNDLE_SHA256.json').items():
        need(sha(PACKAGE / name) == checksum, 'Package changed: ' + name)


def receipt(path, outputs, **extra):
    dump(path, dict(complete=True, files={str(p): sha(p) for p in outputs}, **extra))


def valid(path):
    if not Path(path).is_file():
        return False
    x = read(path)
    return x.get('complete') is True and all(Path(p).is_file() and sha(p) == h
                                           for p, h in x.get('files', {}).items())


def names_meta():
    plan = read(PACKAGE / 'inputs/FROZEN_MODEL_PLAN.json')
    root = Path(plan['settings']['source'])
    need(sha(root / 'data/atac_C1/DATASET.json') == plan['settings']['expected_primary_dataset'],
         'Frozen population metadata changed')
    meta = read(root / 'data/atac_C1/DATASET.json')
    need(len(meta['names']) == 11 and 'TARGET_C1' in meta['names'], 'Wrong target panel')
    return root, meta, plan


@lru_cache(maxsize=4)
def native_api(run):
    import crested
    functions = [crested.tl.contribution_scores, crested.tl.modisco.tfmodisco,
                 crested.tl.modisco.process_patterns, crested.tl.modisco.create_pattern_matrix,
                 crested.tl.modisco.create_pattern_tf_dict, crested.tl.modisco.create_tf_ct_matrix,
                 crested.tl.modisco.calculate_mean_expression_per_cell_type,
                 crested.pp.sort_and_filter_regions_on_specificity]
    versions = {}
    for name in ['crested', 'keras', 'torch', 'numpy', 'scipy', 'h5py', 'modisco-lite', 'memelite', 'scanpy', 'anndata']:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    apis = {f.__name__: dict(signature=str(inspect.signature(f)),
                             source=inspect.getsource(f)) for f in functions}
    path = run / 'NATIVE_API.json'
    current = dict(versions=versions, functions=apis)
    if path.exists():
        need(read(path) == current, 'Analysis environment/API changed since preflight')
    else:
        dump(path, current)
    need(versions['crested'] == '1.10.0', 'This package was prepared for CREsted 1.10.0; inspect another version before adapting.')
    need(versions['modisco-lite'] == '3.0.0', 'This workflow pins modisco-lite 3.0.0; inspect a different version before running.')
    # The public API hard-codes the 25 intervals/25 EIG references. Verify them;
    # do not silently substitute our old attribution algorithm or fewer references.
    src = apis['contribution_scores']['source']
    need('num_steps=25' in src.replace(' ', '') and 'num_baselines=25' in src.replace(' ', ''),
         'Unexpected CREsted IG/EIG defaults. Inspect NATIVE_API.json before continuing.')
    return crested


def load_model(run):
    import torch
    import keras
    need(torch.cuda.is_available(), 'GPU stage must run inside a GPU allocation')
    root, meta, plan = names_meta()
    source = Path(plan['source_run'])
    for filename in ['scorers.py', 'design_core.py', 'core.py', 'io_helpers.py']:
        need(sha(source / filename) == plan['source_module_sha256'][filename],
             'Frozen model-loading implementation changed: ' + filename)
    sys.path.insert(0, str(source))
    from scorers import primary
    settings = dict(plan['settings'], batch_size=config(run)['prediction_batch_size'])
    scorer = primary(settings, 'cuda')
    x = keras.Input(shape=(2048, 4), name='explain_input')
    # A differentiable wrapper of the exact score used for design evaluation.
    y = (scorer.model(x, training=False) +
         scorer.model(keras.ops.flip(x, axis=(1, 2)), training=False)) / 2
    y = y / np.asarray(scorer.weights, dtype=np.float32)
    wrapped = keras.Model(x, y, name='frozen_RC_averaged_unscaled_Borzoi')
    wrapped.eval()
    job = os.environ.get('SLURM_JOB_ID', 'outside_slurm')
    dump(run / 'gpu_runtime' / (job + '.json'), dict(torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        wrapper='forward/RC mean divided by frozen population scale weights'))
    return scorer, wrapped, meta


def onehot(tokens):
    return np.eye(5, 4, dtype=np.float32)[np.asarray(tokens, dtype=np.uint8)]


def design_tokens(contexts, sequence, context):
    x = contexts[context].copy()
    need(x.shape == (2048,), 'Saved context axis differs')
    x[524:1524] = np.array(['ACGT'.index(c) for c in sequence], dtype=np.uint8)
    return x


def contexts_array():
    plan = read(PACKAGE / 'inputs/FROZEN_MODEL_PLAN.json')
    with np.load(Path(plan['source_run']) / 'private/INPUTS.npz') as z:
        return z['context_tokens'].copy()


def prepare(run):
    verify_package()
    c = native_api(run)
    cfg = config(run)
    # Test native multivariate attribution semantics on an exactly solvable
    # linear model before submitting any Borzoi GPU work.
    import keras
    x = keras.Input(shape=(32, 4))
    y = keras.layers.Dense(11, use_bias=False,
            kernel_initializer=keras.initializers.Constant(.01))(keras.layers.Flatten()(x))
    toy = keras.Model(x, y)
    tx = onehot((np.arange(32) % 4)[None, :])
    checks = {}
    for method in cfg['methods']:
        scores, hot = c.tl.contribution_scores(tx, target_idx=None, model=toy,
            method=method, batch_size=4, seed=cfg['seed'], verbose=False)
        need(scores.shape == (1, 11, 32, 4) and np.allclose(scores, .01, atol=1e-6),
             'Native attribution failed an exact linear-gradient test: ' + method)
        checks[method] = dict(known_gradient=.01, passed=True)
    # Check prediction-based mutation effects, including original-base zeros,
    # against an analytically solvable model before real GPU work.
    weights = np.tile(np.arange(1,5,dtype=np.float32),(32,1)).reshape(-1,1) * .01
    weights = np.repeat(weights,11,axis=1)
    mx = keras.Input(shape=(32,4))
    my = keras.layers.Dense(11,use_bias=False,
        kernel_initializer=keras.initializers.Constant(weights))(keras.layers.Flatten()(mx))
    mutation_toy = keras.Model(mx,my)
    mutation_scores,_ = c.tl.contribution_scores(tx,target_idx=None,model=mutation_toy,
        method='mutagenesis',batch_size=8,verbose=False)
    expected = weights.reshape(32,4,11).transpose(2,0,1)
    expected = expected - np.take_along_axis(expected,np.argmax(tx[0],axis=1)[None,:,None],axis=2)
    need(mutation_scores.shape==(1,11,32,4) and
         np.allclose(mutation_scores[0],expected,atol=1e-5),
         'Native saturation mutagenesis failed a linear-model check')
    checks['mutagenesis'] = dict(known_mutation_deltas_passed=True)
    dump(run / 'NATIVE_LINEAR_MODEL_CHECK.json', checks)
    root, meta, plan = names_meta()
    best = read(root / 'models/atac_C1/BEST.json')
    need(best['sha256'] == plan['settings']['expected_primary_weights'] and
         sha(root / 'models/atac_C1' / best['weights_file']) == best['sha256'],
         'Frozen epoch-8 checkpoint missing or changed')
    for filename in ['scorers.py', 'design_core.py', 'core.py', 'io_helpers.py']:
        need(sha(Path(plan['source_run']) / filename) == plan['source_module_sha256'][filename],
             'Frozen loader changed before submission: ' + filename)
    designs = pd.read_csv(PACKAGE / 'inputs/DESIGN_METADATA.tsv', sep='\t')
    initial = fasta(PACKAGE / 'inputs/INITIAL_196.fa')
    selected = fasta(PACKAGE / 'inputs/SELECTED_196.fa')
    need(len(designs) == 196 and set(designs.design_id) == set(initial) == set(selected),
         'Frozen 196-design cohort differs')
    cells = pd.read_csv(PACKAGE / 'inputs/CELL_METADATA.tsv.gz', sep='\t')
    server_cells = pd.read_csv(root / 'CELL_METADATA.tsv.gz', sep='\t')
    need(cells.Cell_ID.is_unique and cells.Cell_ID.equals(server_cells.Cell_ID),
         'Frozen cell order differs from model-fitting cell order')
    codes = np.load(root / 'data/atac_C1/cell_codes.npy')
    need(len(codes) == len(cells) == 111347 and
         np.array_equal(np.bincount(codes, minlength=11), np.array(meta['cells'])),
         'Population codes/counts differ')
    need(np.bincount(codes)[meta['names'].index('TARGET_C1')] == 4440, 'C1 membership differs')
    need(np.array_equal(cells.ATAC_coarse.eq('C1').to_numpy(),
                        codes == meta['names'].index('TARGET_C1')), 'Frozen C1 cell identities differ')
    for tool in ['tomtom', 'fimo']:
        need((Path(cfg['meme_bin']) / tool).is_file(), 'Missing MEME executable: ' + tool)
    tool_versions = {}
    for tool, flag in [('tomtom','-version'),('fimo','--version')]:
        result = subprocess.run([str(Path(cfg['meme_bin']) / tool),flag],
                                text=True,capture_output=True,timeout=30)
        need(result.returncode==0,'Cannot establish MEME tool version: '+tool)
        tool_versions[tool] = (result.stdout+'\n'+result.stderr).strip()
    dump(run / 'MEME_TOOL_VERSIONS.json',tool_versions)
    need(Path(cfg['rna']).is_file(), 'Frozen RNA count file missing')
    import references
    refdir = PACKAGE / 'references'
    ref = references.verify(refdir)
    meme = refdir / references.MEME
    mapping = refdir / 'JASPAR2026_MOUSE_TF_MAPPING.tsv'
    dump(run / 'DATABASE.json', dict(meme=str(meme), mapping=str(mapping),
        meme_sha256=sha(meme), mapping_sha256=sha(mapping),
        collection='JASPAR2026 CORE vertebrate nonredundant',
        reference_manifest=str(refdir / 'REFERENCE_MANIFEST.json'),
        reference_manifest_sha256=sha(refdir / 'REFERENCE_MANIFEST.json'),
        mouse_mapping_column=cfg['mapping_column'], profiles=ref['profiles']))
    shutil.copy2(refdir / 'REFERENCE_MANIFEST.json', run / 'REFERENCE_MANIFEST.json')
    shutil.copy2(refdir / 'JASPAR2026_MOUSE_MAPPING_AUDIT.tsv', run / 'MOUSE_TF_MAPPING_AUDIT.tsv')
    splits = np.load(root / 'splits.npy')
    tokens = np.load(root / 'tokens2048.npy', mmap_mode='r')
    targets = np.load(root / 'data/atac_C1/targets.npy', mmap_mode='r')
    need(tokens.shape == (439167, 2048) and targets.shape == (439167, 11), 'Peak axes differ')
    # Match the CREsted tutorial: rank the mean of observed and predicted values.
    # All original chromosome partitions are included only for interpretation;
    # these motif results are not an independent test of model accuracy.
    indices = np.flatnonzero(np.isin(splits, cfg['natural_splits']))
    np.save(run / 'ELIGIBLE_PEAK_INDICES.npy', indices)
    pred_tasks = [dict(kind='predict', indices=indices[i:i+8192].tolist())
                  for i in range(0, len(indices), 8192)]
    dump(run / 'PREDICT_TASKS.json', pred_tasks)
    dump(run / 'INPUT_IDENTITY.json', dict(checkpoint=plan['settings']['expected_primary_weights'],
        dataset=plan['settings']['expected_primary_dataset'], designs=196,
        cells=len(cells), names=meta['names'], natural_top_k=cfg['natural_top_k'],
        attribution='native CREsted raw per-output gradients; RC-averaged, population scales removed',
        baseline='zeros for IG; 25 whole-input mononucleotide permutations per input for EIG',
        reference_database='JASPAR2026 CORE vertebrates; MGI-verified mouse TF associations',
        reference_database_is_paper_adaptation=True,
        sequence_scope='full 2048 bp explained; central 1000 bp used for TF-MoDISco',
        independence='Interpretation of existing fitted model/data; no independent biological validation'))
    receipt(run / 'PREPARE_COMPLETE.json', [run / 'DATABASE.json', run / 'PREDICT_TASKS.json',
                                          run / 'INPUT_IDENTITY.json', run / 'NATIVE_API.json'])


def predict_task(run, task):
    native_api(run)
    rec = run / 'predictions' / f'{task:05d}.json'
    if valid(rec):
        return
    root, meta, _ = names_meta()
    item = read(run / 'PREDICT_TASKS.json')[task]
    idx = np.asarray(item['indices'])
    scorer, model, _ = load_model(run)
    tokens = np.load(root / 'tokens2048.npy', mmap_mode='r')
    outputs = scorer.predict(tokens[idx])
    # Test differentiable wrapper against the saved original inference path.
    check = model.predict(onehot(tokens[idx[:2]]), batch_size=2, verbose=0)
    need(np.allclose(check, outputs[:2], rtol=.003, atol=.0003), 'Wrapper predictions differ')
    path = rec.with_suffix('.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=idx, predictions=outputs)
    receipt(rec, [path], records=len(idx))


def gpucheck(run):
    """One real-model check before queuing hundreds of native attribution tasks."""
    done = run / 'GPU_PREFLIGHT_COMPLETE.json'
    if valid(done):
        return
    _, meta, _ = names_meta()
    contexts = contexts_array()
    scorer, model, _ = load_model(run)
    initial = fasta(PACKAGE / 'inputs/INITIAL_196.fa')
    selected = fasta(PACKAGE / 'inputs/SELECTED_196.fa')
    row = pd.read_csv(PACKAGE / 'inputs/DESIGN_METADATA.tsv',sep='\t').iloc[0].to_dict()
    did = row['design_id']
    dna = design_tokens(contexts,selected[did],0)
    original = scorer.predict(dna[None,:])
    wrapped = model.predict(onehot(dna[None,:]),verbose=0)
    need(np.allclose(original,wrapped,rtol=.003,atol=.0003),
         'Real-model attribution wrapper differs from the saved design scorer')
    outputs = []
    # These are genuine full native maps and are reused later with the same IDs.
    for stage in ['initial','selected']:
        item = dict(record_id=f'{did}__{stage}__context0',kind='design',population=stage,
            design_id=did,stage=stage,context=0,output_index=-1,
            parent_family=row['parent_family'],arm=row['arm'],
            retention_fraction=float(row['retention_fraction']))
        for method in config(run)['methods']:
            attribute_record(run,item,method,model,meta,contexts,None,
                             {'initial':initial,'selected':selected})
            outputs.append(run / 'maps' / method / (item['record_id']+'.json'))
    receipt(done,outputs,real_model_wrapper_agreement=True,initial_and_final=True,
            methods=config(run)['methods'],population_outputs=11,actual_context=0,
            note='A preflight within the full cohort; all remaining records still required.')


def select(run):
    c = native_api(run)
    import anndata as ad
    cfg = config(run)
    root, meta, _ = names_meta()
    idx = np.load(run / 'ELIGIBLE_PEAK_INDICES.npy')
    prediction = np.full((439167, 11), np.nan, np.float32)
    for j, item in enumerate(read(run / 'PREDICT_TASKS.json')):
        p = run / 'predictions' / f'{j:05d}.json'
        need(valid(p), f'Prediction task {j} missing or invalid')
        with np.load(p.with_suffix('.npz')) as z:
            prediction[z['indices']] = z['predictions']
    need(np.isfinite(prediction[idx]).all(), 'Incomplete peak predictions')
    truth = np.load(root / 'data/atac_C1/targets.npy', mmap_mode='r')[idx]
    scaled_pred = prediction[idx] * np.array(meta['weights'], np.float32)[None, :]
    a = ad.AnnData(truth.T.copy())
    a.obs_names = meta['names']
    a.var_names = idx.astype(str)
    a.layers['combined'] = (a.X + scaled_pred.T) / 2
    selected = c.pp.sort_and_filter_regions_on_specificity(a, top_k=cfg['natural_top_k'],
                    model_name='combined', method='gini', inplace=False)
    regions = (root / 'REGIONS.txt').read_text().splitlines()
    records = []
    for ix, row in selected.var.iterrows():
        records.append(dict(record_id='peak_' + row['Class name'] + '_' + ix,
            kind='natural', population=row['Class name'], output_index=meta['names'].index(row['Class name']),
            peak_index=int(ix), region=regions[int(ix)], rank=int(row['rank']),
            gini_score=float(row['gini_score'])))
    table(run / 'NATURAL_REGIONS.tsv', records)
    # Exact input population ranks are retained. No synthetic design shortlist.
    designs = pd.read_csv(PACKAGE / 'inputs/DESIGN_METADATA.tsv', sep='\t')
    synthetic = []
    for stage in ['initial', 'selected']:
        for row in designs.to_dict('records'):
            for context in cfg['design_contexts']:
                synthetic.append(dict(record_id=f"{row['design_id']}__{stage}__context{context}",
                    kind='design', population=stage, design_id=row['design_id'], stage=stage,
                    context=context, output_index=-1, parent_family=row['parent_family'],
                    arm=row['arm'], retention_fraction=float(row['retention_fraction'])))
    table(run / 'DESIGN_RECORDS.tsv', synthetic)
    tasks = []
    for method in cfg['methods']:
        for collection in [records, synthetic]:
            chunk_size = cfg['design_records_per_task'] if collection is synthetic else cfg['records_per_task']
            for lo in range(0, len(collection), chunk_size):
                tasks.append(dict(kind='attribute', method=method,
                                  records=collection[lo:lo+chunk_size]))
    for row in designs.to_dict('records'):
        tasks.append(dict(kind='mutagenesis', design_id=row['design_id']))
    dump(run / 'EXPLAIN_TASKS.json', tasks)
    receipt(run / 'SELECT_COMPLETE.json', [run / 'NATURAL_REGIONS.tsv', run / 'DESIGN_RECORDS.tsv',
                                          run / 'EXPLAIN_TASKS.json'])


def attribute_record(run, item, method, model, meta, contexts, tokens, fastas):
    c = native_api(run)
    cfg = config(run)
    dest = run / 'maps' / method / (item['record_id'] + '.npz')
    rec = dest.with_suffix('.json')
    if valid(rec):
        return
    if item['kind'] == 'natural':
        dna = np.array(tokens[item['peak_index']], dtype=np.uint8)
        outputs = [int(item['output_index'])]
    else:
        dna = design_tokens(contexts, fastas[item['stage']][item['design_id']], item['context'])
        outputs = list(range(11))
    x = onehot(dna[None, :])
    seed = cfg['seed'] + int(hashlib.sha256(item['record_id'].encode()).hexdigest()[:7], 16)
    # Call the public function, retaining its native 26 integration points and
    # 25 permuted baselines for expected IG. Full 2048-bp input is explained.
    scores, hot = c.tl.contribution_scores(x, target_idx=outputs, model=model, method=method,
        batch_size=cfg['integration_batch_size'], seed=seed,
        transpose=False, simplex_correction=cfg['simplex_correction'], verbose=False)
    need(scores.shape == (1, len(outputs), 2048, 4) and np.array_equal(hot, x),
         'Native contribution axes differ')
    need(np.isfinite(scores).all(), 'Nonfinite native contribution map')
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, hypothetical=scores[0].astype(np.float32), onehot=x[0],
                        output_indices=np.array(outputs), prediction=model.predict(x, verbose=0)[0])
    quality = {}
    if method == 'integrated_grad':
        true_delta = (model.predict(x, verbose=0)[0] -
                      model.predict(np.zeros_like(x), verbose=0)[0])[outputs]
        attribution_sum = (scores[0] * x[0][None, :, :]).sum((1, 2))
        quality = dict(zero_baseline_true_delta=true_delta.tolist(),
            zero_baseline_full_map_sum=attribution_sum.tolist(),
            zero_baseline_completeness_absolute_error=abs(attribution_sum-true_delta).tolist(),
            diagnostic_only=True)
    receipt(rec, [dest], method=method, seed=seed, record=item,
            simplex_correction=cfg['simplex_correction'], integration_points=26,
            baseline_count=25 if method == 'expected_integrated_grad' else 1,
            quality=quality,
            completeness_note='Native hypothetical-gradient output retained; cropped map sums are not completeness tests.')


def explain_task(run, task):
    cfg = config(run)
    c = native_api(run)
    item = read(run / 'EXPLAIN_TASKS.json')[task]
    root, meta, _ = names_meta()
    contexts = contexts_array()
    scorer, model, _ = load_model(run)
    if item['kind'] == 'attribute':
        tokens = np.load(root / 'tokens2048.npy', mmap_mode='r')
        fastas = {stage: fasta(PACKAGE / 'inputs' / (stage.upper() + '_196.fa'))
                  for stage in ['initial', 'selected']}
        for record in item['records']:
            attribute_record(run, record, item['method'], model, meta, contexts, tokens, fastas)
    else:
        did = item['design_id']
        sequence = fasta(PACKAGE / 'inputs/SELECTED_196.fa')[did]
        for context in cfg['design_contexts']:
            dest = run / 'mutagenesis' / f'{did}__context{context}.npz'
            rec = dest.with_suffix('.json')
            if valid(rec):
                continue
            dna = design_tokens(contexts, sequence, context)
            x = onehot(dna[None, :])
            # Native CREsted saturation mutagenesis, all positions and outputs.
            # Store the whole map; final insert summaries use bases 524:1524.
            delta, hot = c.tl.contribution_scores(x, target_idx=None, model=model,
                method='mutagenesis', batch_size=cfg['prediction_batch_size'],
                transpose=False, verbose=False)
            need(delta.shape == (1, 11, 2048, 4) and np.isfinite(delta).all(), 'Invalid ISM map')
            observed_delta = np.take_along_axis(delta[0], dna[None, :, None].clip(0, 3), axis=2)
            known = dna < 4
            need(np.allclose(observed_delta[:, known], 0, atol=1e-3, rtol=0),
                 'Native ISM original-base change is not zero')
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, delta=delta[0].astype(np.float32), onehot=hot[0],
                                prediction=model.predict(x, verbose=0)[0], tokens=dna)
            receipt(rec, [dest], method='native CREsted mutagenesis', design_id=did, context=context,
                    sign='mutant prediction minus intact prediction', explained_positions=2048,
                    designed_positions=1000, population_outputs=meta['names'])
    receipt(run / 'tasks' / f'{task:05d}.json', [], task=task)


def task_done(run, task):
    item = read(run / 'EXPLAIN_TASKS.json')[task]
    if item['kind'] == 'attribute':
        return all(valid(run / 'maps' / item['method'] / (r['record_id'] + '.json'))
                   for r in item['records'])
    return all(valid(run / 'mutagenesis' / f"{item['design_id']}__context{j}.json")
               for j in config(run)['design_contexts'])


def assemble(run):
    cfg = config(run)
    root, meta, _ = names_meta()
    natural = pd.read_csv(run / 'NATURAL_REGIONS.tsv', sep='\t').to_dict('records')
    designs = pd.read_csv(run / 'DESIGN_RECORDS.tsv', sep='\t').to_dict('records')
    groups = {}
    for pop in meta['names']:
        groups[pop] = [(r, int(r['output_index'])) for r in natural if r['population'] == pop]
    # Contexts stay separate to avoid doubling the effective sequence support.
    # Initial and selected are analyzed in full, without selecting best sequences.
    for stage in ['initial', 'selected']:
        for context in cfg['design_contexts']:
            for output, name in enumerate(meta['names']):
                label = f'design_{stage}_context{context}__{name}'
                groups[label] = [(r, output) for r in designs
                                 if r['stage'] == stage and int(r['context']) == context]
    for method in cfg['methods']:
        for label, values in groups.items():
            dest = run / 'contributions' / method
            dest.mkdir(parents=True, exist_ok=True)
            checkpoint = dest / (label + '.json')
            if valid(checkpoint):
                continue
            hot, scores, index = [], [], []
            for item, output in values:
                path = run / 'maps' / method / (item['record_id'] + '.npz')
                need(valid(path.with_suffix('.json')), 'Missing map: ' + str(path))
                with np.load(path) as z:
                    where = np.flatnonzero(z['output_indices'] == output)
                    need(len(where) == 1, 'Output map missing')
                    hot.append(z['onehot'][524:1524].T)
                    scores.append(z['hypothetical'][where[0], 524:1524].T)
                index.append(dict(row=len(index), **item))
            hpath = dest / (label + '_oh.npz')
            spath = dest / (label + '_contrib.npz')
            ipath = dest / (label + '_index.tsv')
            np.savez_compressed(hpath, np.stack(hot))
            np.savez_compressed(spath, np.stack(scores))
            table(ipath, index)
            receipt(checkpoint, [hpath, spath, ipath], records=len(index))
    return list(groups)


def modisco_jobs(run):
    labels = assemble(run)
    jobs = [dict(method=m, label=label) for m in config(run)['methods'] for label in labels]
    dump(run / 'MODISCO_TASKS.json', jobs)


def motif_task(run, task):
    c = native_api(run)
    cfg = config(run)
    item = read(run / 'MODISCO_TASKS.json')[task]
    method, label = item['method'], item['label']
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    out = run / 'modisco' / method
    dest = out / (label + '_modisco_results.h5')
    done = dest.with_suffix('.json')
    if valid(done):
        return
    if dest.exists():
        dest.rename(dest.with_name(dest.name + '.unverified_' + str(time.time_ns())))
    args = dict(contrib_dir=str(run / 'contributions' / method), class_names=[label],
        output_dir=str(out), max_seqlets=cfg['modisco_max_seqlets'],
        min_metacluster_size=100, min_final_cluster_size=20, window=cfg['modisco_window'],
        n_leiden=2, report=False, verbose=True, fdr=cfg['modisco_fdr'],
        sliding_window_size=cfg['modisco_sliding_window'], flank_size=cfg['modisco_flank'])
    if 'top_n_regions' in inspect.signature(c.tl.modisco.tfmodisco).parameters:
        args['top_n_regions'] = None
    c.tl.modisco.tfmodisco(**args)
    need(dest.is_file(), 'Native TF-MoDISco did not produce an H5 result')
    import h5py
    with h5py.File(dest) as h:
        counts = {k: len(h[k]) if k in h else 0 for k in ['pos_patterns', 'neg_patterns']}
    receipt(done, [dest], parameters=args, pattern_counts=counts)


def report_task(run, task):
    """Native reporting cannot force rediscovery after an HTML-only failure."""
    c = native_api(run)
    item = read(run / 'MODISCO_TASKS.json')[task]
    method, label = item['method'], item['label']
    done = run / 'reports' / method / (label + '.json')
    if valid(done):
        return
    path = run / 'modisco' / method / (label + '_modisco_results.h5')
    discovery = path.with_suffix('.json')
    need(valid(discovery), 'Cannot report an incomplete motif-discovery run')
    record = read(discovery)
    if sum(record['pattern_counts'].values()) == 0:
        receipt(done,[path],status='no_patterns_to_report',
                note='No motif logos or TF matches are invented for an empty discovery result.')
        return
    db = read(run / 'DATABASE.json')
    need(sha(db['meme']) == db['meme_sha256'], 'Frozen reference database changed')
    folder = run / 'modisco' / method / (label + '_report')
    if folder.exists():
        folder.rename(folder.with_name(folder.name + '.unverified_' + str(time.time_ns())))
    args = dict(record['parameters'],report=True,meme_db=db['meme'])
    c.tl.modisco.tfmodisco(**args)
    need(valid(discovery), 'Native reporting changed the frozen discovery H5')
    files = [p for p in folder.rglob('*') if p.is_file()]
    need(any(p.suffix == '.html' for p in files), 'Native motif report did not create HTML')
    receipt(done,files,status='native_html_report_complete',parameters=args,
            discovery_h5_sha256=sha(path))


def write_meme(path, patterns):
    with Path(path).open('w') as f:
        f.write('MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n'
                'Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n')
        for name, data in patterns.items():
            p = np.asarray(data['ppm'], float)
            need(p.ndim == 2 and p.shape[1] == 4 and np.all(p >= 0), 'Invalid native PPM')
            p = p / p.sum(1, keepdims=True)
            nsites = max(1, int(data['pattern']['n_seqlets']))
            f.write(f'MOTIF pattern_{name}\nletter-probability matrix: alength= 4 w= {len(p)} nsites= {nsites}\n')
            for row in p:
                f.write(' '.join('%.9g' % x for x in row) + '\n')
            f.write('\n')


def native_cluster(c, files, cfg, mode):
    fn = c.tl.modisco.process_patterns
    params = inspect.signature(fn).parameters
    kwargs = dict(sim_threshold=cfg['sim_threshold'], trim_ic_threshold=cfg['trim_ic_threshold'],
                  discard_ic_threshold=cfg['discard_ic_threshold'], verbose=True)
    if 'clustering' in params:
        kwargs.update(clustering=mode, representative='ic_mean' if mode == 'greedy' else 'n_seqlets',
                      linkage_method='average', sort_by='n_seqlets')
    else:
        need(mode == 'greedy', 'Installed legacy CREsted has no agglomerative implementation')
    return fn(files, **kwargs), kwargs


def scan_known_references(run):
    done = run / 'KNOWN_REFERENCE_FIMO_COMPLETE.json'
    if valid(done):
        return
    cfg = config(run)
    db = read(run / 'DATABASE.json')
    need(sha(db['meme']) == db['meme_sha256'], 'Frozen JASPAR file changed')
    sequences = run / 'ALL_INITIAL_AND_SELECTED.fa'
    if not sequences.exists():
        with sequences.open('w') as f:
            for stage in ['initial', 'selected']:
                for did, seq in fasta(PACKAGE / 'inputs' / (stage.upper() + '_196.fa')).items():
                    f.write(f'>{stage}__{did}\n{seq}\n')
    folder = run / 'known_JASPAR2026_fimo'
    if not (folder / 'fimo.tsv').is_file():
        subprocess.run([str(Path(cfg['meme_bin']) / 'fimo'), '--oc', str(folder),
            '--thresh', '0.0001', '--max-stored-scores', '10000000',
            db['meme'], str(sequences)], check=True)
    hits = pd.read_csv(folder / 'fimo.tsv', sep='\t', comment='#')
    need(set(hits.get('sequence_name', [])).issubset(
        {stage+'__'+did for stage in ['initial','selected']
         for did in fasta(PACKAGE / 'inputs/SELECTED_196.fa')}), 'Unexpected FIMO sequence IDs')
    grouped = hits.groupby(['motif_id','sequence_name']).size() if len(hits) else {}
    profiles = __import__('references').meme_profiles(Path(db['meme']))
    cohort = pd.read_csv(PACKAGE / 'inputs/DESIGN_METADATA.tsv', sep='\t')
    records = []
    for motif, info in profiles.items():
        for stage in ['initial','selected']:
            for item in cohort.to_dict('records'):
                key = (motif,stage+'__'+item['design_id'])
                records.append(dict(motif_id=motif,motif_name=info['name'],stage=stage,
                    design_id=item['design_id'],parent_family=item['parent_family'],
                    arm=item['arm'],retention_fraction=item['retention_fraction'],
                    site_count=int(grouped.get(key,0))))
    path = run / 'ALL196_JASPAR2026_SITES.tsv.gz'
    table(path, records)
    receipt(done, [folder / 'fimo.tsv', path], reference_profiles=len(profiles),
        designed_sequences=196, sequence_stages=2, site_p_threshold=1e-4,
        significance='FIMO raw per-site p and within-scan q are preserved; counts are not independent TF mechanisms or evidence of binding.')


def cluster(run, method):
    c = native_api(run)
    cfg = config(run)
    folder = run / 'clustered' / method
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for job in read(run / 'MODISCO_TASKS.json'):
        if job['method'] != method:
            continue
        path = run / 'modisco' / method / (job['label'] + '_modisco_results.h5')
        need(valid(path.with_suffix('.json')), 'TF-MoDISco output incomplete: ' + str(path))
        files[job['label']] = str(path)
    # Empty-pattern files are legitimate scientific negatives, not missing jobs.
    import h5py
    nonempty = {}
    for label, path in files.items():
        with h5py.File(path) as h:
            if sum(len(h[k]) for k in ['pos_patterns', 'neg_patterns'] if k in h):
                nonempty[label] = path
    need(nonempty, 'No TF-MoDISco patterns found in any population/design run')
    patterns, kwargs = native_cluster(c, nonempty, cfg, 'greedy')
    with (folder / 'PATTERNS.pkl').open('wb') as f:
        pickle.dump(patterns, f)
    dump(folder / 'CLUSTER_PARAMETERS.json', kwargs)
    members = []
    for key, pat in patterns.items():
        for member, value in pat['instances'].items():
            members.append(dict(cluster=key, member=member,
                source=value.get('file_path', ''), seqlets=int(value.get('n_seqlets', 0)),
                contribution_sign='positive' if np.mean(value['contrib_scores']) > 0 else 'negative',
                width=len(value['sequence']), representative=(member == pat['pattern'].get('id'))))
    table(folder / 'CLUSTER_MEMBERS.tsv', members)
    classes = sorted(nonempty)
    matrix = c.tl.modisco.create_pattern_matrix(classes, patterns, normalize=False,
                                               pattern_parameter='seqlet_count')
    active = [key for key, pat in patterns.items()
              if any(v.get('n_seqlets', 0) > 0 for v in pat['classes'].values())]
    need(matrix.shape[1] == len(active), 'Native pattern matrix column alignment differs')
    pd.DataFrame(matrix, index=classes, columns=active).to_csv(folder / 'PATTERN_MATRIX.tsv', sep='\t')
    write_meme(folder / 'REPRESENTATIVES.meme', patterns)
    # Keep source membership/signs: different signs are not declared different TFs.
    fn_params = inspect.signature(c.tl.modisco.process_patterns).parameters
    if 'clustering' in fn_params:
        sensitivity, args = native_cluster(c, nonempty, cfg, 'agglomerative')
        with (folder / 'SENSITIVITY_AGGLOMERATIVE.pkl').open('wb') as f:
            pickle.dump(sensitivity, f)
        table(folder / 'SENSITIVITY_AGGLOMERATIVE_MEMBERS.tsv',
              [dict(cluster=k, member=i) for k, p in sensitivity.items() for i in p['instances']])
        dump(folder / 'SENSITIVITY_AGGLOMERATIVE_PARAMETERS.json', args)
    scan_known_references(run)
    db = read(run / 'DATABASE.json')
    need(sha(db['meme']) == db['meme_sha256'] and sha(db['mapping']) == db['mapping_sha256'],
         'Frozen JASPAR2026 motif database or mouse mapping changed')
    tom = folder / 'tomtom'
    if not (tom / 'tomtom.tsv').is_file():
        subprocess.run([str(Path(cfg['meme_bin']) / 'tomtom'), '-oc', str(tom),
            '-thresh', str(cfg['meme_database_q']), str(folder / 'REPRESENTATIVES.meme'),
            db['meme']], check=True)
    matches = pd.read_csv(tom / 'tomtom.tsv', sep='\t', comment='#')
    match_dict = {k: dict(matches=matches.loc[
        matches.Query_ID.eq('pattern_' + str(k)) &
        (matches['q-value'] <= cfg['meme_database_q']), 'Target_ID'].drop_duplicates().tolist(),
        patterns=list(pat['instances'])) for k, pat in patterns.items()}
    mapping = c.tl.modisco.read_motif_to_tf_file(db['mapping'])
    cols = [cfg['mapping_column']]
    need(set(cols).issubset(mapping.columns), 'Unexpected verified JASPAR/MGI motif-to-TF mapping')
    tf_dict, all_tfs = c.tl.modisco.create_pattern_tf_dict(match_dict, mapping, patterns, cols)
    with (folder / 'PATTERN_TF_CANDIDATES.pkl').open('wb') as f:
        pickle.dump(tf_dict, f)
    table(folder / 'TF_CANDIDATES.tsv', [dict(cluster=k, tf=tf,
          matched_motifs=';'.join(v['matches'])) for k, v in tf_dict.items() for tf in v['tfs']],
          ['cluster', 'tf', 'matched_motifs'])
    (folder / 'TF_GENES.txt').write_text('\n'.join(map(str, all_tfs)) + '\n')
    # Whole-cohort sequence occurrence audit. It complements contribution-based
    # discovery: a FIMO match alone does not establish model importance or binding.
    sequences_path = run / 'ALL_INITIAL_AND_SELECTED.fa'
    if not sequences_path.exists():
        with sequences_path.open('w') as f:
            for stage in ['initial', 'selected']:
                for did, seq in fasta(PACKAGE / 'inputs' / (stage.upper() + '_196.fa')).items():
                    f.write(f'>{stage}__{did}\n{seq}\n')
    fimo = folder / 'fimo_all196'
    if not (fimo / 'fimo.tsv').is_file():
        subprocess.run([str(Path(cfg['meme_bin']) / 'fimo'), '--oc', str(fimo),
            '--thresh', '0.0001', '--max-stored-scores', '10000000',
            str(folder / 'REPRESENTATIVES.meme'), str(sequences_path)], check=True)
    hits = pd.read_csv(fimo / 'fimo.tsv', sep='\t', comment='#')
    cohort = pd.read_csv(PACKAGE / 'inputs/DESIGN_METADATA.tsv', sep='\t')
    occurrences = []
    for key in patterns:
        motif_hits = hits[hits['motif_id'].eq('pattern_' + key)]
        for stage in ['initial', 'selected']:
            for row in cohort.to_dict('records'):
                g = motif_hits[motif_hits['sequence_name'].eq(stage + '__' + row['design_id'])]
                # Exact interval deduplication avoids counting the same match
                # twice solely because a palindromic site hits both strands.
                occurrences.append(dict(cluster=key, stage=stage, design_id=row['design_id'],
                    parent_family=row['parent_family'], arm=row['arm'],
                    retention_fraction=row['retention_fraction'],
                    matched_intervals=len(g[['start', 'stop']].drop_duplicates())))
    table(folder / 'ALL196_PATTERN_OCCURRENCES.tsv', occurrences)
    receipt(folder / 'COMPLETE.json', [folder / 'PATTERNS.pkl', folder / 'CLUSTER_MEMBERS.tsv',
        folder / 'PATTERN_TF_CANDIDATES.pkl', folder / 'TF_GENES.txt', folder / 'PATTERN_MATRIX.tsv'])


def rna(run):
    c = native_api(run)
    from rna_reader import extract_markers
    from scipy import sparse
    import anndata as ad
    cfg = config(run)
    root, meta, _ = names_meta()
    genes = set()
    for method in cfg['methods']:
        genes.update((run / 'clustered' / method / 'TF_GENES.txt').read_text().split())
    if not genes:
        # A valid absence of verified candidates is not a fabricated TF assignment.
        reason = run / 'RNA_NO_MAPPED_CANDIDATES.json'
        dump(reason, dict(reason='No discovered pattern has a verified mouse TF candidate in the frozen JASPAR/MGI mapping.',
                          RNA_assignment_performed=False, candidate_count=0))
        receipt(run / 'RNA_COMPLETE.json', [reason], status='no_verified_TF_candidates')
        return
    cells = pd.read_csv(PACKAGE / 'inputs/CELL_METADATA.tsv.gz', sep='\t')
    codes = np.load(root / 'data/atac_C1/cell_codes.npy')
    raw, totals, available, source = extract_markers(cfg['rna'], cells.Cell_ID.to_numpy(), sorted(genes))
    need(len(available) > 0, 'No motif candidate TF symbols present in raw RNA')
    normalized = sparse.csr_matrix((raw / totals[:, None] * 1e6).astype(np.float32))
    a = ad.AnnData(normalized)
    a.obs_names = cells.Cell_ID.astype(str).to_numpy()
    a.var_names = available
    a.obs['model_population'] = np.array(meta['names'])[codes]
    path = run / 'RNA_TF_CPM.h5ad'
    a.write_h5ad(path, compression='gzip')
    means = c.tl.modisco.calculate_mean_expression_per_cell_type(str(path),
                          cell_type_column='model_population', cpm_normalize=False)
    means = means.reindex(meta['names'])
    need(np.isfinite(means.to_numpy()).all(), 'Incomplete population RNA means')
    means.to_csv(run / 'RNA_MEAN_CPM.tsv', sep='\t')
    source.update(normalization='Per-cell counts / all-gene raw RNA count total * 1e6, then averaged within the exact model population; only candidate columns exported.',
                  candidate_database='JASPAR2026 CORE vertebrates with MGI identifier/homology mapping')
    dump(run / 'RNA_SOURCE.json', source)
    table(run / 'RNA_GENE_AVAILABILITY.tsv',
          [dict(gene=g, detected_in_feature_axis=g in available) for g in sorted(genes)])
    for method in cfg['methods']:
        folder = run / 'clustered' / method
        with (folder / 'PATTERNS.pkl').open('rb') as f:
            original = pickle.load(f)
        with (folder / 'PATTERN_TF_CANDIDATES.pkl').open('rb') as f:
            candidates = pickle.load(f)
        # Native function expects consecutive pattern keys. Only genuine model
        # populations enter the correlation, never design stages or contexts.
        subset, tf_subset, translation = {}, {}, []
        for old, pat in original.items():
            classes = {k: v for k, v in pat['classes'].items() if k in meta['names']}
            if not classes or old not in candidates:
                continue
            new = str(len(subset))
            subset[new] = copy.deepcopy(pat)
            subset[new]['classes'] = classes
            tf_subset[new] = copy.deepcopy(candidates[old])
            tf_subset[new]['pattern_info'] = subset[new]
            translation.append(dict(native_tf_pattern=new, full_cluster=old))
        table(folder / 'RNA_PATTERN_ID_MAP.tsv', translation,
              ['native_tf_pattern', 'full_cluster'])
        if not tf_subset:
            dump(folder / 'RNA_NO_CANDIDATES.json', dict(reason='No motif-matched pattern has natural-population support'))
            continue
        # Legacy CREsted versions have an empty-array indexing bug if no TF
        # passes the initial expression/positive-importance gate. Record that
        # legitimate empty result before calling it; never invent an assignment.
        positive_candidate = any(
            tf in means.columns and any(np.mean(v['contrib_scores']) > 0 and
                means.loc[pop, tf] > 0 for pop, v in subset[k]['classes'].items())
            for k, candidate in tf_subset.items() for tf in candidate['tfs'])
        if not positive_candidate:
            dump(folder / 'RNA_NO_CANDIDATES.json', dict(reason='No expressed candidate has positive natural-population importance'))
            continue
        fn = c.tl.modisco.create_tf_ct_matrix
        params = inspect.signature(fn).parameters
        args = dict(log_transform=True, normalize_pattern_importances=False, normalize_gex=False,
            importance_threshold=0, pattern_parameter='seqlet_count', filter_correlation=True,
            zscore_threshold=2, correlation_threshold=cfg['correlation_threshold'], verbose=True)
        if 'selection' in params:
            args.update(selection='threshold', similarity_metric='pearson')
        # Legacy threshold/Pearson algorithm, not newer NNLS defaults.
        values, annots = fn(tf_subset, subset, means, meta['names'], **args)
        need(values.shape == (11, len(annots), 2), 'Native RNA/TF result axes differ')
        np.savez_compressed(folder / 'TF_POPULATION_MATRIX.npz', values=values,
                            annotations=np.asarray(annots, dtype=str), populations=np.array(meta['names']))
        table(folder / 'RNA_SUPPORTED_TF_PATTERNS.tsv',
              [dict(annotation=s, population=name, mean_expression=float(values[j, i, 0]),
                    pattern_importance=float(values[j, i, 1]))
               for i, s in enumerate(annots) for j, name in enumerate(meta['names'])],
              ['annotation', 'population', 'mean_expression', 'pattern_importance'])
        dump(folder / 'TF_SELECTION_PARAMETERS.json', args)
        correlations = []
        id_map = {r['native_tf_pattern']: r['full_cluster'] for r in translation}
        for key, candidate in tf_subset.items():
            support = np.zeros(11)
            for pop, value in subset[key]['classes'].items():
                sign = 1 if np.mean(value['contrib_scores']) > 0 else -1
                support[meta['names'].index(pop)] = sign * value['n_seqlets']
            for tf in candidate['tfs']:
                correlation = None
                if tf in means:
                    expression = np.log1p(means[tf].to_numpy())
                    if np.std(expression) > 0 and np.std(abs(support)) > 0:
                        correlation = float(np.corrcoef(expression, abs(support))[0, 1])
                correlations.append(dict(cluster=id_map[key], tf=tf,
                    expression_available=tf in means, pearson_r=correlation,
                    native_selected=tf + '_pattern_' + key in annots))
        table(folder / 'ALL_CANDIDATE_TF_CORRELATIONS.tsv', correlations)
    outputs = [run / 'RNA_MEAN_CPM.tsv', run / 'RNA_SOURCE.json', run / 'RNA_GENE_AVAILABILITY.tsv']
    for method in cfg['methods']:
        folder = run / 'clustered' / method
        outputs.extend(p for p in folder.iterdir() if p.name.startswith(('RNA_', 'TF_POPULATION_', 'TF_SELECTION_')))
    receipt(run / 'RNA_COMPLETE.json', outputs)


def figures(run):
    import matplotlib.pyplot as plt
    import logomaker
    cfg = config(run)
    root, meta, _ = names_meta()
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, stem):
        fig.tight_layout()
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / (stem + '.' + ext), dpi=220, bbox_inches='tight')
        plt.close(fig)

    for method in cfg['methods']:
        folder = run / 'clustered' / method
        matrix = pd.read_csv(folder / 'PATTERN_MATRIX.tsv', sep='\t', index_col=0)
        selected = [r for r in matrix.index if not r.startswith('design_')]
        a = matrix.loc[selected].to_numpy()
        fig, ax = plt.subplots(figsize=(max(8, .22*a.shape[1]), 5))
        signed = np.sign(a) * np.log1p(abs(a))
        limit = max(float(abs(signed).max()), 1)
        im = ax.imshow(signed, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(yticks=range(len(selected)), yticklabels=selected,
               xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
               xlabel='CREsted pattern cluster', title=method + ': natural-population motif support')
        fig.colorbar(im, ax=ax, label='Signed log(1 + seqlet count)')
        save(fig, method + '_PATTERN_POPULATIONS')
        # Counts of input sequences differ between natural and synthetic groups.
        # Export a descriptive per-input-sequence scale alongside native counts.
        natural = pd.read_csv(run / 'NATURAL_REGIONS.tsv', sep='\t')
        denominators = natural.groupby('population').size().to_dict()
        denominators.update({name:196 for name in matrix.index if name.startswith('design_')})
        rate = matrix.divide(pd.Series(denominators).reindex(matrix.index), axis=0) * 100
        need(np.isfinite(rate.to_numpy()).all(), 'Pattern input-sequence counts differ')
        rate.to_csv(folder / 'PATTERN_SEQLETS_PER100_INPUTS.tsv', sep='\t')
        for lo in range(0, len(rate.columns), 30):
            block = rate.iloc[:, lo:lo+30]
            fig, ax = plt.subplots(figsize=(max(9,.33*block.shape[1]), max(5,.18*len(block))))
            values = np.sign(block.to_numpy()) * np.log1p(abs(block.to_numpy()))
            limit = max(float(abs(values).max()), 1)
            im = ax.imshow(values, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
            ax.set(yticks=range(len(block)), yticklabels=block.index,
                   xticks=range(block.shape[1]), xticklabels=block.columns,
                   title=method+': natural populations and every initial/final design group')
            ax.tick_params(axis='y',labelsize=6)
            fig.colorbar(im,ax=ax,label='Signed log1p(seqlets per 100 input sequences)')
            save(fig, f'{method}_NATURAL_AND_DESIGNS_{lo//30:03d}')
        # Every representative is plotted; no top-motif-only gallery.
        with (folder / 'PATTERNS.pkl').open('rb') as f:
            patterns = pickle.load(f)
        for lo in range(0, len(patterns), 12):
            chunk = list(patterns.items())[lo:lo+12]
            fig, axes = plt.subplots(len(chunk), 1, figsize=(12, max(3, len(chunk)*1.2)), squeeze=False)
            for ax, (key, pat) in zip(axes[:, 0], chunk):
                p = np.asarray(pat['ppm'])
                ic = 2 + (p * np.log2(np.maximum(p, 1e-12))).sum(1)
                logomaker.Logo(pd.DataFrame(p*ic[:, None], columns=list('ACGT')), ax=ax,
                               color_scheme='classic', font_name='DejaVu Sans')
                ax.set(ylabel='bits', title=f'Cluster {key}', ylim=(0, 2), xlim=(-.5, len(p)-.5))
            save(fig, f'{method}_ALL_REPRESENTATIVES_{lo//12:03d}')
        p = folder / 'TF_POPULATION_MATRIX.npz'
        if p.exists():
            with np.load(p) as z:
                values, annotations = z['values'], z['annotations']
            for lo in range(0, len(annotations), 30):
                v = values[:, lo:lo+30]
                fig, axes = plt.subplots(1, 2, figsize=(max(9, .48*v.shape[1]), 5.5))
                for ax, j, title in zip(axes, [0, 1],
                        ['TF RNA: log1p(mean CPM)', 'Pattern: signed seqlet count']):
                    data = v[:, :, j]
                    scaled = data / np.maximum(abs(data).max(0), 1e-12)
                    im = ax.imshow(scaled, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set(yticks=range(11), yticklabels=meta['names'], title=title,
                        xticks=range(v.shape[1]), xticklabels=annotations[lo:lo+30])
                    ax.tick_params(axis='x', rotation=90, labelsize=7)
                    fig.colorbar(im, ax=ax, shrink=.6, label='Divided by column maximum magnitude')
                save(fig, f'{method}_RNA_SUPPORTED_TFS_{lo//30:03d}')
    # Mutation summaries use every single-base substitution in every final insert.
    rows = []
    ti = meta['names'].index('TARGET_C1')
    comp = [j for j in range(11) if j != ti]
    fastas = fasta(PACKAGE / 'inputs/SELECTED_196.fa')
    for did in fastas:
        for context in cfg['design_contexts']:
            p = run / 'mutagenesis' / f'{did}__context{context}.npz'
            need(valid(p.with_suffix('.json')), 'Missing final-sequence mutation map')
            with np.load(p) as z:
                delta = z['delta'][:, 524:1524, :]
                intact = z['prediction']
                mutants = np.maximum(0, intact[:, None, None] + delta)
                margin = np.log1p(mutants[ti]) - np.log1p(mutants[comp].max(0))
                original = np.log1p(intact[ti]) - np.log1p(intact[comp].max())
                nonself = np.arange(4)[None, :] != z['tokens'][524:1524, None]
                shifts = (margin - original)[nonself]
                c1 = delta[ti][nonself]
                rows.append(dict(design_id=did, context=context, tested_substitutions=len(shifts),
                    largest_C1_decrease=float(max(0, -c1.min())),
                    largest_margin_decrease=float(max(0, -shifts.min())),
                    median_margin_change=float(np.median(shifts)),
                    fraction_margin_reducing=float(np.mean(shifts < 0))))
                fig, axes = plt.subplots(2, 1, figsize=(14, 4.2))
                for ax, data, title in zip(axes, [delta[ti], margin-original],
                    ['Change in C1 prediction', 'Change in log1p(C1) − log1p(strongest competitor)']):
                    lim = max(float(abs(data).max()), 1e-6)
                    im = ax.imshow(data.T, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                                   extent=(0, 1000, 4, 0))
                    ax.set(yticks=np.arange(4)+.5, yticklabels=list('ACGT'), title=title)
                    fig.colorbar(im, ax=ax, shrink=.8)
                axes[1].set_xlabel('Position in 1,000-bp designed sequence')
                fig.suptitle(f'{did}: context {context}; mutant minus intact')
                save(fig, f'ISM_{did}_context{context}')
    table(run / 'ALL196_MUTATION_SUMMARY.tsv', rows)
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for context, g in summary.groupby('context'):
        values = np.sort(g.largest_margin_decrease)
        ax.plot(values, np.arange(1, len(values)+1)/len(values), label=f'Context {context}')
    ax.set(xlabel='Largest decrease in predicted C1 separation from a single mutation',
           ylabel='Fraction of all 196 designs')
    ax.legend()
    save(fig, 'ALL196_SINGLE_MUTATION_EFFECTS')


def finish(run):
    cfg = config(run)
    for method in cfg['methods']:
        need(valid(run / 'clustered' / method / 'COMPLETE.json'), 'Missing native clustering')
    need(valid(run / 'RNA_COMPLETE.json'), 'RNA/TF analysis incomplete')
    need(all(task_done(run, i) for i in range(len(read(run / 'EXPLAIN_TASKS.json')))),
         'Not all full-cohort attribution/mutagenesis records are complete')
    need(all(valid(run / 'reports' / item['method'] / (item['label']+'.json'))
             for item in read(run / 'MODISCO_TASKS.json')),
         'Not all native motif reports are complete')
    # Compare discovered IG and EIG pattern representatives directly, without
    # requiring either pattern to receive a known-TF label.
    agreement = run / 'IG_VS_EIG_PATTERN_TOMTOM'
    if not (agreement / 'tomtom.tsv').is_file():
        subprocess.run([str(Path(cfg['meme_bin']) / 'tomtom'), '-oc', str(agreement),
            '-thresh', str(cfg['meme_database_q']),
            str(run / 'clustered/integrated_grad/REPRESENTATIVES.meme'),
            str(run / 'clustered/expected_integrated_grad/REPRESENTATIVES.meme')],check=True)
    figures(run)
    completeness = []
    for path in (run / 'maps/integrated_grad').glob('*.json'):
        rec = read(path)
        record = rec['record']
        q = rec.get('quality', {})
        for j, (delta, error) in enumerate(zip(q.get('zero_baseline_true_delta', []),
                    q.get('zero_baseline_completeness_absolute_error', []))):
            completeness.append(dict(record_id=record['record_id'], kind=record['kind'],
                output_index=j if record['kind'] == 'design' else record['output_index'],
                true_delta=delta, absolute_error=error,
                exceeds_descriptive_5pct_plus_005=error > .05 + .05*abs(delta)))
    table(run / 'IG_ZERO_BASELINE_COMPLETENESS.tsv', completeness)
    pattern_counts = []
    for job in read(run / 'MODISCO_TASKS.json'):
        p = run / 'modisco' / job['method'] / (job['label'] + '_modisco_results.json')
        rec = read(p)
        pattern_counts.append(dict(method=job['method'], source=job['label'], **rec['pattern_counts']))
    table(run / 'PATTERN_COUNTS_ALL_RUNS.tsv', pattern_counts)
    cluster_summary = []
    for method in cfg['methods']:
        folder = run / 'clustered' / method
        with (folder / 'PATTERNS.pkl').open('rb') as f:
            patterns = pickle.load(f)
        accepted_file = folder / 'ALL_CANDIDATE_TF_CORRELATIONS.tsv'
        accepted = pd.read_csv(accepted_file, sep='\t') if accepted_file.exists() else pd.DataFrame()
        cluster_summary.append(dict(method=method, clustered_patterns=len(patterns),
            native_RNA_supported_TF_pattern_pairs=int(accepted.native_selected.sum())
                if 'native_selected' in accepted else 0))
    dump(run / 'RUN_SUMMARY.json', dict(designs=196, stages=['initial', 'selected'],
        contexts=config(run)['design_contexts'], design_explanations_per_method=784,
        population_outputs=11, natural_regions_per_population=cfg['natural_top_k'],
        saturation_mutagenesis_designs=196, mutation_maps=392,
        substitutions_within_designs=196*2*1000*3, motif_runs=len(pattern_counts),
        methods=cluster_summary, reference_collection='JASPAR2026 CORE vertebrates',
        TF_mapping='Identifier-supported JASPAR/MGI mouse candidates; not the Aerts collection',
        RNA_correlation_selection='Native CREsted threshold/Pearson with r>=0.2 and native positive-importance/expression and z-score gates',
        native_clustering_primary='Paper-era greedy; highest mean-information-content representative',
        secondary_clustering='Native agglomerative, average linkage, most-supported representative if available',
        seed_generation='Existing designed sequences retained; this interpretation run does not generate new seeds or claim to regenerate the library.',
        interpretation='Model contribution, motif similarity and RNA-supported TF candidates. '
                       'No direct measurement of TF binding or synthetic enhancer function.'))
    receipt(run / 'ANALYSIS_COMPLETE.json', [run / 'ALL196_MUTATION_SUMMARY.tsv',
              run / 'KNOWN_REFERENCE_FIMO_COMPLETE.json', run / 'REFERENCE_MANIFEST.json',
              run / 'RNA_COMPLETE.json', run / 'RUN_SUMMARY.json', run / 'PATTERN_COUNTS_ALL_RUNS.tsv'],
            status='complete', designs=196, methods=cfg['methods'],
            natural_top_k_per_population=cfg['natural_top_k'], population_outputs=11,
            biological_validation=False)
    for diagnostic in ['NEEDS_RESUME.json', 'FAILURE.json']:
        (run / diagnostic).unlink(missing_ok=True)
    archive = run / 'SHARE_C1_CRESTED_JASPAR2026_FULL_ANALYSIS.zip'
    with zipfile.ZipFile(archive.with_suffix('.partial.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            relative = p.relative_to(run)
            if (not p.is_file() or p == archive or '.partial' in p.name or
                    any('.unverified_' in part for part in relative.parts)):
                continue
            if relative.parts[0] in ['maps', 'predictions', 'contributions', 'database_cache',
                                     'mutagenesis', 'logs', 'matplotlib_cache', 'numba_cache']:
                continue
            if p.name == 'RNA_TF_CPM.h5ad':
                continue
            z.write(p, str(relative))
    archive.with_suffix('.partial.zip').replace(archive)
    print('COMPLETE. DOWNLOAD: ' + str(archive), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'gpucheck', 'predict', 'select', 'explain', 'assemble',
                         'modisco', 'reports', 'cluster', 'rna', 'finish'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--task', type=int)
    parser.add_argument('--method')
    a = parser.parse_args()
    # C1_NATIVE_PPM_EXPORT_COMPAT_V1
    if a.stage == 'cluster':
        import ppm_export_compat
        ppm_export_compat.install(sys.modules[__name__], a.run)
    if a.stage == 'prepare': prepare(a.run)
    elif a.stage == 'gpucheck': gpucheck(a.run)
    elif a.stage == 'predict': predict_task(a.run, a.task)
    elif a.stage == 'select': select(a.run)
    elif a.stage == 'explain': explain_task(a.run, a.task)
    elif a.stage == 'assemble': modisco_jobs(a.run)
    elif a.stage == 'modisco': motif_task(a.run, a.task)
    elif a.stage == 'reports': report_task(a.run, a.task)
    elif a.stage == 'cluster': cluster(a.run, a.method)
    elif a.stage == 'rna': rna(a.run)
    elif a.stage == 'finish': finish(a.run)


if __name__ == '__main__':
    main()
