"""Post-hoc plateau diagnostics for the 500-bp 32-design native ISE pilot.

All 0..steps variants are predicted in four DNA backgrounds not used for ISE
(although drawn from original training-genome regions). No early stopping or
selection depends on the monitoring results. The secondary model is used as an
additional, imperfect model-agreement check, not an independent wet-lab test.
"""
from pathlib import Path
import numpy as np
from design_core import margin, robust
from io_helpers import checkpoint, dump, read, rows, table, progress, reuse

MODELS = ('Borzoi', 'CREsted500')


def trace_predictions(root, s, scorer, model_name):
    """Return per-design, per-step monitoring scores, cached with a checksum."""
    from workflow import predict
    root = Path(root)
    path = root / 'private' / f'PLATEAU_TRACES_{model_name}.npz'
    tag = f'PLATEAU_TRACES_{model_name}_COMPLETE'
    if reuse(root, tag):
        with np.load(path) as z:
            return {key: z[key].copy() for key in z.files}
    plans = rows(root/'DESIGN_PLAN.tsv')
    with np.load(root/'private/INPUTS.npz') as z:
        ctx = z['context_tokens'][s['optimization_contexts']:s['optimization_contexts']+s['selection_contexts']]
    assert len(ctx) == s['selection_contexts'] > 0
    nstep = s['steps'] + 1
    score = np.empty((len(plans), nstep), dtype=np.float64)
    target = np.empty_like(score)
    strongest_other = np.empty_like(score)
    for i, row in enumerate(plans):
        with np.load(root/'private'/row['design_id']/'RESULT.npz') as z:
            traj = z['trajectory']
        if traj.shape != (nstep, s['target_len']):
            raise RuntimeError(f'Invalid length or iteration history in {row["design_id"]}')
        profiles = predict(scorer, ctx, traj)
        score[i] = robust(margin(profiles, scorer.target))
        target[i] = np.median(profiles[...,scorer.target], axis=1)
        strongest_other[i] = np.median(np.delete(profiles,scorer.target,axis=-1).max(axis=-1), axis=1)
        if (i + 1) % 4 == 0:
            progress(root, 'plateau_scoring', model=model_name, completed=i+1, total=len(plans))
    np.savez_compressed(path, score=score, target=target, strongest_other=strongest_other,
                        design_ids=np.array([r['design_id'] for r in plans]))
    checkpoint(root, tag, [path])
    return dict(score=score, target=target, strongest_other=strongest_other,
                design_ids=np.array([r['design_id'] for r in plans]))


def output_plateau(root, s, borzoi, secondary):
    """Write raw trace TSVs, descriptive plateau statistics, and a readable plot."""
    root = Path(root)
    if reuse(root, 'PILOT_PLATEAU_COMPLETE'):
        return
    plans = rows(root/'DESIGN_PLAN.tsv')
    traces = {'Borzoi': borzoi, 'CREsted500': secondary}
    rows_out = []
    for model, trace in traces.items():
        for i, plan in enumerate(plans):
            for step in range(s['steps']+1):
                rows_out.append(dict(design_id=plan['design_id'], arm=plan['arm'], parent_id=plan['parent_id'],
                     model=model, iteration=step, monitoring_robust_log_margin=float(trace['score'][i,step]),
                     monitoring_median_target_FRPM=float(trace['target'][i,step]),
                     monitoring_median_max_competitor_FRPM=float(trace['strongest_other'][i,step])))
    table(root/'PILOT_ITERATION_METRICS.tsv', rows_out)
    result = []
    thresholds = [0, 30, 50, 75, s['steps']]
    milestones = sorted(set(k for k in thresholds if k <= s['steps']))
    for model, trace in traces.items():
        for arm in ('natural', 'random', 'all'):
            indices = [i for i, plan in enumerate(plans) if arm == 'all' or plan['arm'] == arm]
            if not indices: continue
            data = trace['score'][indices]
            rec = dict(model=model, arm=arm, designs=len(indices), iterations=s['steps'],
                       metric='Robust log1p(C1) minus log1p(max competitor) over four monitoring contexts')
            for step in milestones:
                rec[f'median_margin_at_{step}'] = float(np.median(data[:,step]))
            final = s['steps']
            rec['median_gain_0_to_final'] = float(np.median(data[:,final]-data[:,0]))
            last_window = min(25,final)
            rec[f'median_gain_last_{last_window}_steps'] = float(np.median(data[:,final]-data[:,final-last_window]))
            rec['fraction_improved_gt_0p02_last_window'] = float(np.mean((data[:,final]-data[:,final-last_window]) > .02))
            rec['fraction_declined_lt_minus0p02_last_window'] = float(np.mean((data[:,final]-data[:,final-last_window]) < -.02))
            rec['fraction_best_observed_step_in_last_25'] = float(np.mean(data[:,max(0,final-25):].max(axis=1) >= data.max(axis=1)-1e-8))
            result.append(rec)
    table(root/'PILOT_PLATEAU_SUMMARY.tsv', result)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    steps = np.arange(s['steps']+1)
    for ax, (model, tr) in zip(axes, traces.items()):
        for arm, color in [('natural', '#2563a6'), ('random', '#bc6227')]:
            indices = [i for i,r in enumerate(plans) if r['arm']==arm]
            a = tr['score'][indices]
            med = np.median(a,axis=0)
            lower, upper = np.percentile(a,[25,75],axis=0)
            ax.plot(steps, med, label=f'{arm} (n={len(indices)})', color=color, linewidth=2.2)
            ax.fill_between(steps,lower,upper,color=color,alpha=.13,linewidth=0)
        for cut in (30,50,75):
            if cut <= s['steps']: ax.axvline(cut,linestyle=':',color='#9c9c9c',linewidth=.8)
        ax.set_title('Fine-tuned Borzoi' if model=='Borzoi' else 'Secondary CREsted 500-bp model',fontsize=12)
        ax.set_xlabel('ISE iteration (single-nucleotide edits)')
        ax.grid(axis='y',alpha=.18)
        ax.legend(frameon=False,loc='best',fontsize=9)
    axes[0].set_ylabel('C1 preference (robust log1p margin; higher is better)')
    fig.suptitle(f"500-bp, {s['steps']}-step pilot: does predicted C1 preference plateau?",fontsize=14,weight='semibold')
    fig.text(.5,.01,'Median and interquartile range across designs; four contexts not used for sequence optimization. Model predictions only.',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.045,1,.93])
    fig.savefig(root/'PILOT_PLATEAU.png',dpi=180)
    fig.savefig(root/'PILOT_PLATEAU.pdf')
    plt.close(fig)
    lines = [
      '# 500-bp ISE pilot: how to interpret the trajectories',
      '',
      'The 100 ISE steps always run to completion. There is no early stopping.',
      'The plotted score is positive when C1 is predicted to exceed the strongest competitor.',
      'Scores are calculated in FOUR alternative DNA contexts not used during ISE.',
      'These are original training-genome backgrounds, NOT independent biological validation.',
      'The secondary model provides a second prediction, not ground-truth enhancer activity.',
      '',
      'Use the median gain between steps 75 and 100, and compare with steps 0 to 75.',
      'If both models are flat or declining in the last 25 steps, 100 steps is unlikely to be necessary.',
      'If scores still increase substantially at step 100, a longer pilot may be justified.',
      'If Borzoi continues increasing but the secondary model declines, suspect optimization overfitting.',
      'A universal plateau iteration cannot be concluded solely from n=32 model predictions.',
      '',
      '',
      'Observed pilot data (monitoring contexts):',
    ]
    for r in result:
        final = s['steps']
        late = r.get(f'median_gain_last_{min(25,final)}_steps')
        lines.append(f"{r['model']} / {r['arm']}: n={r['designs']}, "
                     f"median margin at iteration 0={r['median_margin_at_0']:.4f}, "
                     f"iteration {final}={r[f'median_margin_at_{final}']:.4f}; "
                     f"median final-{min(25,final)}-step gain={late:+.4f}.")
    lines.extend([
      '',
      'Saved results: PILOT_ITERATION_METRICS.tsv (all designs/steps); PILOT_PLATEAU_SUMMARY.tsv;',
      'PILOT_PLATEAU.png and PDF; OPTIMIZATION_TRAJECTORY_METRICS.tsv (optimization contexts).',
      ''
    ])
    (root/'PILOT_INTERPRETATION.md').write_text('\n'.join(lines))
    files = [root/n for n in ('PILOT_ITERATION_METRICS.tsv','PILOT_PLATEAU_SUMMARY.tsv','PILOT_PLATEAU.png',
                                'PILOT_PLATEAU.pdf','PILOT_INTERPRETATION.md')]
    checkpoint(root, 'PILOT_PLATEAU_COMPLETE', files)
    dump(root/'PILOT_PLATEAU_STATUS.json', dict(status='complete', designs=len(plans), iterations=s['steps'],
         optimization_contexts=s['optimization_contexts'], monitoring_contexts=s['selection_contexts'],
         biological_validation=False, secondary_model_tested=True))
