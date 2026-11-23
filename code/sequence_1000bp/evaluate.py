import gzip
import json
import zipfile
import numpy as np
from core import ratio, qc, freeze_shortlist, sequence_key
from design_core import margin, robust, dinuc_shuffle, dna, DESIGN_LEN, N_SINGLE_MUTANTS
from io_helpers import need, read, rows, table, dump, checkpoint, reuse, progress, sha


def get_designs(root, s):
    plans = rows(root/'DESIGN_PLAN.tsv'); starts = []; selected = []; endpoints = []; summaries = []; histories = []
    for r in plans:
        folder = root/'private'/r['design_id']
        need(reuse(folder, 'DESIGN_COMPLETE'), 'Incomplete design '+r['design_id'])
        with np.load(folder/'RESULT.npz') as z:
            target = int(z['target_index'])
            starts.append(z['trajectory'][0]); endpoints.append(z['trajectory'][-1])
            selected.append(z['trajectory'][int(z['selected_step'])])
            for step, profile in enumerate(z['predictions']):
                gaps = margin(profile, target)
                histories.append(dict(design_id=r['design_id'], iteration=step,
                     selected=step == int(z['selected_step']),
                     robust_margin=float(robust(gaps)), minimum_margin=float(gaps.min()),
                     median_target=float(np.median(profile[:, target])),
                     median_strongest_offtarget=float(np.median(np.delete(profile, target, axis=-1).max(axis=-1))),
                     target_floor_met=bool((profile[:, target] >= z['floor']-1e-7).all())))
        summaries.append(read(folder/'DESIGN_SUMMARY.json'))
    need(len(plans) == s['designs'], 'Incomplete library')
    table(root/'OPTIMIZATION_TRAJECTORY_METRICS.tsv', histories)
    return plans, np.array(starts), np.array(selected), np.array(endpoints), summaries


def diversity_report(root, plans, selected):
    records = []
    for i, seq in enumerate(selected):
        distances = np.minimum((selected != seq).sum(axis=1), (selected != 3-seq[::-1]).sum(axis=1))
        distances[i] = DESIGN_LEN + 1; nearest = int(np.argmin(distances))
        other_parent = np.array([r['parent_id'] != plans[i]['parent_id'] for r in plans])
        cross = np.where(other_parent, distances, DESIGN_LEN + 1)
        records.append(dict(design_id=plans[i]['design_id'], nearest_design=plans[nearest]['design_id'],
             nearest_Hamming_allowing_RC=int(distances[nearest]),
             nearest_other_parent_Hamming_allowing_RC=int(cross.min())))
    table(root/'SEQUENCE_DIVERSITY.tsv', records)


def make_controls(root, s, starts, selected, endpoints):
    if reuse(root, 'CONTROLS_COMPLETE'):
        return np.load(root/'private/EVALUATION_SEQUENCES.npz')['sequences']
    rng = np.random.default_rng(s['seed']+1); all_seq = []; control_rows = []
    for i, seq in enumerate(selected):
        controls = np.array([dinuc_shuffle(seq, rng) for _ in range(s['audit_shuffles'])])
        all_seq.append(np.concatenate([starts[i:i+1], selected[i:i+1], endpoints[i:i+1], controls]))
        control_rows.append(dict(design_id=f'C1L_{i+1:04d}', requested=len(controls),
             distinct_shuffles=len({dna(x) for x in controls}),
             identical_to_design=sum(np.array_equal(x, seq) for x in controls)))
    all_seq = np.asarray(all_seq)
    np.savez_compressed(root/'private/EVALUATION_SEQUENCES.npz', sequences=all_seq)
    table(root/'SHUFFLE_QC.tsv', control_rows)
    checkpoint(root, 'CONTROLS_COMPLETE', [root/'private/EVALUATION_SEQUENCES.npz', root/'SHUFFLE_QC.tsv'])
    return all_seq


def score_library(root, s, scorer, name, contexts, all_seq):
    from workflow import predict
    tag = 'SCORES_'+name+'_COMPLETE'
    if reuse(root, tag):
        return np.load(root/f'ALL_PROFILES_{name}.npz')['predictions']
    cache = root/'private'/('scores_'+name); cache.mkdir(exist_ok=True)
    predictions = []
    for i, sequences in enumerate(all_seq):
        path = cache/f'{i:04d}.npz'; receipt = cache/f'{i:04d}.json'
        if receipt.exists():
            need(sha(path) == read(receipt)['sha256'], 'Changed cached evaluation predictions')
            p = np.load(path)['predictions']
        else:
            p = predict(scorer, contexts, sequences)
            temp = path.with_suffix('.partial.npz'); np.savez_compressed(temp, predictions=p); temp.replace(path)
            dump(receipt, dict(sha256=sha(path)))
        need(p.shape == (3+s['audit_shuffles'], len(contexts), len(scorer.names)), 'Profile axes differ')
        predictions.append(p)
        if (i+1) % 8 == 0:
            progress(root, 'evaluate_library', model=name, completed=i+1, total=len(all_seq))
    predictions = np.asarray(predictions)
    np.savez_compressed(root/f'ALL_PROFILES_{name}.npz', predictions=predictions,
         design_ids=np.array([f'C1L_{i+1:04d}' for i in range(len(all_seq))]),
         variants=np.array(['initial', 'selected', 'native_endpoint']+[f'dinuc_{i+1:02d}' for i in range(s['audit_shuffles'])]),
         context_ids=np.array([r['context_id'] for r in rows(root/'CONTEXTS.tsv')]),
         class_names=np.array(scorer.names), target_index=np.array(scorer.target))
    checkpoint(root, tag, [root/f'ALL_PROFILES_{name}.npz'])
    return predictions


def model_metrics(profiles, target, reference, retention, indices):
    p = profiles[:, indices]; g = margin(p, target); r = ratio(p, target)
    floor = np.maximum(retention*p[0, :, target], .25*reference[indices])
    initial, final = p[0, :, target], p[1, :, target]
    shuffled_score = robust(g[3:]); score = float(robust(g[1]))
    competitor = np.delete(p[1], target, axis=-1).max(axis=-1)
    return dict(robust_margin=score, initial_robust_margin=float(robust(g[0])),
         margin_gain=score-float(robust(g[0])),
         median_target=float(np.median(final)), minimum_target=float(final.min()),
         median_strongest_offtarget=float(np.median(competitor)),
         median_ratio=float(np.median(r[1])), minimum_ratio=float(r[1].min()),
         contexts_target_highest=int((r[1] > 1).sum()), contexts_at_least_1p5fold=int((r[1] >= 1.5).sum()),
         contexts_at_least_2fold=int((r[1] >= 2).sum()), contexts=len(indices),
         target_floor_all=bool((final >= floor-1e-7).all()),
         minimum_fraction_initial_target=float(np.min(final/np.maximum(initial, 1e-8))),
         above_all_dinuc_controls=bool(score > shuffled_score.max()),
         margin_above_dinuc_mean=score-float(shuffled_score.mean()),
         empirical_control_tail_fraction=float((1+(shuffled_score >= score).sum())/(1+len(shuffled_score))))


def select_before_audit(root, plans, selected, predictions, target, reference, control_rows, s):
    if reuse(root, 'SHORTLIST_FROZEN'):
        return read(root/'SHORTLIST_FROZEN.json')['indices']
    nopt = s['optimization_contexts']; indices = np.arange(nopt, nopt+s['selection_contexts'])
    records = []
    for i, r in enumerate(plans):
        m = model_metrics(predictions[i], target, reference, float(r['retention_fraction']), indices)
        q = qc(selected[i]); control_ok = int(control_rows[i]['distinct_shuffles']) >= 12 and int(control_rows[i]['identical_to_design']) == 0
        eligible = q['sequence_QC_pass'] and control_ok and m['target_floor_all'] and m['minimum_ratio'] >= 1.5 and m['above_all_dinuc_controls']
        records.append(dict(index=i, design_id=r['design_id'], parent_id=r['parent_id'], arm=r['arm'],
             selection_eligible=bool(eligible), selection_score=m['robust_margin'], **q,
             **{'selection_'+k: v for k, v in m.items()}))
    chosen = freeze_shortlist(records, selected, s['shortlist_maximum'])
    table(root/'PRIMARY_SELECTION_METRICS.tsv', records)
    table(root/'FROZEN_SHORTLIST.tsv', [dict(rank=j+1, **records[i]) for j, i in enumerate(chosen)])
    checkpoint(root, 'SHORTLIST_FROZEN', [root/'PRIMARY_SELECTION_METRICS.tsv', root/'FROZEN_SHORTLIST.tsv'],
         indices=chosen, selection_inputs='Primary scorer + four selection contexts only; no final-audit predictions',
         criteria='All4 target/maxoff>=1.5; floor all4; robust margin exceeds all16 dinuc controls; QC; '
                  'min15bp Hamming allowing reverse complement; max4 per natural parent; max64 (or configured shortlist size).')
    return chosen


def main(root, s, scorer):
    if reuse(root,'EVALUATION_COMPLETE'):
        return
    plans, starts, selected, endpoints, design_summaries=get_designs(root,s)
    diversity_report(root,plans,selected)
    all_seq=make_controls(root,s,starts,selected,endpoints)
    contexts=np.load(root/'private/INPUTS.npz')['context_tokens']
    controls=rows(root/'SHUFFLE_QC.tsv')
    p=score_library(root,s,scorer,'Borzoi',contexts,all_seq)
    reference=np.load(root/'private/REFERENCE_PRIMARY.npy')
    nopt=s['optimization_contexts'];nsel=s['selection_contexts']
    # Selection can only see the four designated selection contexts.
    visible=p.copy();visible[:,:,nopt+nsel:,:]=np.nan
    chosen=select_before_audit(root,plans,selected,visible,scorer.target,reference,controls,s)
    del visible
    if s.get('plateau_monitoring'):
        from plateau import trace_predictions,output_plateau
        trace=trace_predictions(root,s,scorer)
        output_plateau(root,s,trace)
    audit=np.arange(nopt+nsel,len(contexts))
    need(len(audit)==s['audit_contexts'],'Unexpected audit contexts')
    reports=[];profiles_rows=[];private_rows=[];fasta=[]
    chosen_set=set(chosen)
    for i,r in enumerate(plans):
        row={**r,**design_summaries[i],'frozen_shortlist':i in chosen_set}
        metrics=model_metrics(p[i],scorer.target,reference,float(r['retention_fraction']),audit)
        row.update({'Borzoi_audit_'+k:v for k,v in metrics.items()})
        for ci in audit:
            for variant,vi in (('initial',0),('selected',1),('native_endpoint',2)):
                for k,output_name in enumerate(scorer.names):
                    profiles_rows.append(dict(design_id=r['design_id'],model='Borzoi',variant=variant,
                            context_id=f'fresh_{ci+1:02d}',output=output_name,
                            FRPM=float(p[i,vi,ci,k])))
        row['computational_gate_pass']=bool(qc(selected[i])['sequence_QC_pass'] and
            int(controls[i]['distinct_shuffles'])>=12 and
            int(controls[i]['identical_to_design'])==0 and
            row['Borzoi_audit_contexts_target_highest']==s['audit_contexts'] and
            row['Borzoi_audit_contexts_at_least_2fold']>=6 and
            row['Borzoi_audit_target_floor_all'] and
            row['Borzoi_audit_above_all_dinuc_controls'])
        row['confirmed_frozen_shortlist']=bool(i in chosen_set and row['computational_gate_pass'])
        reports.append(row)
        private_rows.append(dict(design_id=r['design_id'],initial_1000bp=dna(starts[i]),
                                 selected_1000bp=dna(selected[i]),native_endpoint_1000bp=dna(endpoints[i])))
        fasta.extend(('>'+r['design_id'],dna(selected[i])))
    table(root/'LIBRARY_EVALUATION.tsv',reports)
    table(root/'private/ALL_SEQUENCES.tsv',private_rows)
    (root/'private/ALL_DESIGNS_1000BP.fa').write_text('\n'.join(fasta)+'\n')
    tmp=root/'AUDIT_ALL_OUTPUTS.tsv';table(tmp,profiles_rows)
    with tmp.open('rb') as fileobj,gzip.open(root/'AUDIT_ALL_OUTPUTS.tsv.gz','wb') as z:
        import shutil
        shutil.copyfileobj(fileobj,z)
    tmp.unlink()
    groups={}
    for arm in ('natural','random'):
        for retain in (.25,.5):
            sub=[r for r in reports if r['arm']==arm and float(r['retention_fraction'])==retain]
            groups[f'{arm}_retention{retain}']=dict(designs=len(sub),
                frozen_shortlist=sum(r['frozen_shortlist'] for r in sub),
                confirmed_frozen_shortlist=sum(r['confirmed_frozen_shortlist'] for r in sub),
                median_Borzoi_audit_ratio=float(np.median([r['Borzoi_audit_median_ratio'] for r in sub])) if sub else None)
    summary=dict(status='complete',model='Fine-tuned Borzoi ONLY',design_length=DESIGN_LEN,
        designs=len(plans),native_ISE_steps=s['steps'],substitutions_evaluated_per_step=N_SINGLE_MUTANTS,
        design_mutant_predictions=len(plans)*s['steps']*N_SINGLE_MUTANTS,
        native_CREsted_ISE=True,models_retrained=False,known_motifs_used=False,target_cells=4440,
        natural_parent_peaks=len({r['parent_id'] for r in plans if r['arm']=='natural'}),
        unique_sequences_up_to_reverse_complement=len({sequence_key(x) for x in selected}),
        frozen_shortlist=len(chosen),confirmed_frozen_shortlist=sum(r['confirmed_frozen_shortlist'] for r in reports),
        descriptive_all_library_gate_passes=sum(r['computational_gate_pass'] for r in reports),
        arms=groups,attribution_performed=False,TF_binding_inferred=False,
        interpretation='Borzoi predictions are accessibility FRPM, not measured enhancer activity. '
           'The 14 contexts are training-chromosome DNA backgrounds, not biological replicates; '
           'four contexts select candidates, eight reserved for final audit. Optimization uses two. '
           'Only Borzoi was used; no secondary CREsted predictive model. '
           'No selection reranking on the final audit. Thresholds are computational follow-up criteria.')
    dump(root/'SUMMARY.json',summary)
    plot(root,reports)
    inputs=[root/n for n in ('SUMMARY.json','LIBRARY_EVALUATION.tsv','LIBRARY_OVERVIEW.png',
              'AUDIT_ALL_OUTPUTS.tsv.gz','OPTIMIZATION_TRAJECTORY_METRICS.tsv','SEQUENCE_DIVERSITY.tsv',
              'private/ALL_SEQUENCES.tsv','private/ALL_DESIGNS_1000BP.fa')]
    if s.get('plateau_monitoring'):
        inputs += [root/n for n in ('PILOT_ITERATION_METRICS.tsv','PILOT_PLATEAU_SUMMARY.tsv',
                   'PILOT_PLATEAU.png','PILOT_PLATEAU.pdf','PILOT_INTERPRETATION.md')]
    checkpoint(root,'EVALUATION_COMPLETE',inputs)
    collect(root)


def plot(root,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings=read(root/'SETTINGS.json');steps=settings['steps']
    fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    for arm,color in (('natural','#2473af'),('random','#e58222')):
        sub=[r for r in reports if r['arm']==arm]
        axs[0].scatter([r['Borzoi_audit_median_ratio'] for r in sub],
                       [r['Borzoi_audit_minimum_ratio'] for r in sub],s=14,alpha=.35,label=arm,color=color)
        axs[1].scatter([r['Borzoi_audit_initial_robust_margin'] for r in sub],
                       [r['Borzoi_audit_robust_margin'] for r in sub],s=14,alpha=.35,color=color)
        axs[2].hist([r['selected_step'] for r in sub],bins=np.arange(-.5,steps+1.5,1),
                    histtype='step',linewidth=1.8,color=color,label=arm)
    axs[0].axvline(2,color='gray',linestyle='--')
    axs[0].axhline(1.5,color='gray',linestyle='--')
    axs[0].set(xlabel='Median C1/strongest other',ylabel='Minimum C1/strongest other',title='Fresh-context Borzoi ratios')
    lims=[min(axs[1].get_xlim()[0],axs[1].get_ylim()[0]),
          max(axs[1].get_xlim()[1],axs[1].get_ylim()[1])]
    axs[1].plot(lims,lims,linestyle='--',color='gray')
    axs[1].set(xlabel='Initial robust log1p margin',ylabel='Selected robust log1p margin',title='Borzoi improvement')
    axs[2].set(xlabel='Selected native ISE step',ylabel='Designs',title=f'Best feasible step among 0..{steps}')
    axs[0].legend();fig.suptitle(f'{len(reports):,} designs, 1000-bp ISE | Borzoi only (predictions)')
    fig.tight_layout();fig.savefig(root/'LIBRARY_OVERVIEW.png',dpi=150);plt.close(fig)


def collect(root):
    complete=(root/'EVALUATION_COMPLETE.json').exists()
    if complete:
        need(reuse(root,'EVALUATION_COMPLETE') and reuse(root,'SHORTLIST_FROZEN') and
             reuse(root,'SCORES_Borzoi_COMPLETE'), 'Broken evaluation receipts')
    completed=len(list((root/'private').glob('C1L_*/DESIGN_COMPLETE.json')))
    dump(root/'JOB_RESULT.json',dict(all_stages_completed=complete,designs_completed=completed,
         no_functional_activity_claim=True,summary_present=(root/'SUMMARY.json').exists()))
    names=['REPLAY_DIAGNOSTICS.json','SETTINGS.json','METHODS.txt','LOCAL_VALIDATION.json',
           'BUNDLE_SHA256.json','INPUT_PROVENANCE.json','NATIVE_PREFLIGHT.json',
           'DESIGN_PLAN.tsv','CONTEXTS.tsv','SUMMARY.json','LIBRARY_EVALUATION.tsv',
           'PRIMARY_SELECTION_METRICS.tsv','FROZEN_SHORTLIST.tsv','SHORTLIST_FROZEN.json',
           'SHUFFLE_QC.tsv','LIBRARY_OVERVIEW.png','OPTIMIZATION_TRAJECTORY_METRICS.tsv',
           'SEQUENCE_DIVERSITY.tsv','ALL_PROFILES_Borzoi.npz','AUDIT_ALL_OUTPUTS.tsv.gz',
           'JOB_RESULT.json','PILOT_ITERATION_METRICS.tsv','PILOT_PLATEAU_SUMMARY.tsv',
           'PILOT_PLATEAU.png','PILOT_PLATEAU.pdf','PILOT_INTERPRETATION.md','PILOT_PLATEAU_STATUS.json']
    for archive,paths in [('SHARE_C1_BORZOI_ONLY_1000BP_256_METRICS.zip',[root/n for n in names]),
                          ('PRIVATE_C1_BORZOI_ONLY_1000BP_256_DESIGNS.zip',
                           list((root/'private').rglob('*'))+[root/'DESIGN_PLAN.tsv'])]:
        tmp=root/(archive+'.partial')
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                if p.is_file() and '.partial' not in p.name:
                    z.write(p,str(p.relative_to(root)))
        tmp.replace(root/archive)
    print(json.dumps(read(root/'JOB_RESULT.json')),flush=True)
