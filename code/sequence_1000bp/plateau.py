"""Post-hoc iteration diagnostics on four alternative, training-DNA contexts.

Only the fine-tuned Borzoi is used. Monitoring never determines the selected
sequence and is not independent biological validation. All iterations are retained.
"""
from pathlib import Path
import numpy as np
from design_core import margin,robust
from io_helpers import checkpoint,dump,read,rows,table,progress,reuse,need


def trace_predictions(root,s,scorer):
    from workflow import predict
    root=Path(root)
    path=root/'private/PLATEAU_TRACES_Borzoi.npz'
    tag='PLATEAU_TRACES_Borzoi_COMPLETE'
    if reuse(root,tag):
        with np.load(path) as z:
            return {key:z[key].copy() for key in z.files}
    plans=rows(root/'DESIGN_PLAN.tsv')
    with np.load(root/'private/INPUTS.npz') as z:
        i=s['optimization_contexts'];ctx=z['context_tokens'][i:i+s['selection_contexts']]
    need(len(ctx)==s['selection_contexts']>0,'Monitoring context axes differ')
    steps=s['steps']+1
    score=np.empty((len(plans),steps),dtype=np.float64)
    target=np.empty_like(score);strongest_other=np.empty_like(score)
    for n,r in enumerate(plans):
        with np.load(root/'private'/r['design_id']/'RESULT.npz') as z:
            traj=z['trajectory'].copy()
        need(traj.shape==(steps,s['target_len']), 'Unexpected design trajectory shape')
        profiles=predict(scorer,ctx,traj)
        score[n]=robust(margin(profiles,scorer.target))
        target[n]=np.median(profiles[...,scorer.target],axis=1)
        strongest_other[n]=np.median(np.delete(profiles,scorer.target,axis=-1).max(axis=-1),axis=1)
        if (n+1)%4==0:
            progress(root,'plateau_scoring',model='Borzoi',completed=n+1,total=len(plans))
    np.savez_compressed(path,score=score,target=target,strongest_other=strongest_other,
                        design_ids=np.array([r['design_id'] for r in plans]))
    checkpoint(root,tag,[path])
    return dict(score=score,target=target,strongest_other=strongest_other)


def output_plateau(root,s,trace):
    root=Path(root)
    if reuse(root,'PILOT_PLATEAU_COMPLETE'):
        return
    plans=rows(root/'DESIGN_PLAN.tsv')
    rows_out=[]
    for i,r in enumerate(plans):
        for step in range(s['steps']+1):
            rows_out.append(dict(design_id=r['design_id'],arm=r['arm'],parent_id=r['parent_id'],
                 model='Borzoi',iteration=step,monitoring_robust_log_margin=float(trace['score'][i,step]),
                 monitoring_median_target_FRPM=float(trace['target'][i,step]),
                 monitoring_median_max_competitor_FRPM=float(trace['strongest_other'][i,step])))
    table(root/'PILOT_ITERATION_METRICS.tsv',rows_out)
    records=[]
    milestones=sorted(set(k for k in (0,30,50,75,100,125,150,175,s['steps']) if k<=s['steps']))
    for arm in ('natural','random','all'):
        indices=[i for i,r in enumerate(plans) if arm=='all' or r['arm']==arm]
        if not indices:continue
        data=trace['score'][indices]; final=s['steps']
        rec=dict(model='Borzoi',arm=arm,designs=len(indices),iterations=final,
                 metric='Robust log1p(C1) minus log1p(max competitor) in four nonoptimization contexts')
        for step in milestones:
            rec[f'median_margin_at_{step}']=float(np.median(data[:,step]))
        rec['median_gain_0_to_final']=float(np.median(data[:,final]-data[:,0]))
        last=min(25,final)
        rec[f'median_gain_last_{last}_steps']=float(np.median(data[:,final]-data[:,final-last]))
        rec['fraction_improved_gt_0p02_last_window']=float(np.mean((data[:,final]-data[:,final-last])>.02))
        rec['fraction_declined_lt_minus0p02_last_window']=float(np.mean((data[:,final]-data[:,final-last])<-.02))
        rec['fraction_best_observed_step_in_last_25']=float(np.mean(data[:,max(0,final-25):].max(axis=1)>=data.max(axis=1)-1e-8))
        records.append(rec)
    table(root/'PILOT_PLATEAU_SUMMARY.tsv',records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.2,5.4))
    xx=np.arange(s['steps']+1)
    for arm,color in (('natural','#2563a6'),('random','#bc6227')):
        ii=[i for i,r in enumerate(plans) if r['arm']==arm]
        data=trace['score'][ii]
        med=np.median(data,axis=0); lo,hi=np.percentile(data,[25,75],axis=0)
        ax.plot(xx,med,label=f'{arm} (n={len(ii)})',color=color,linewidth=2)
        ax.fill_between(xx,lo,hi,color=color,alpha=.13,linewidth=0)
    for cut in (30,50,75,100,150,175):
        if cut<s['steps']:ax.axvline(cut,linestyle=':',color='#999999',linewidth=.7)
    ax.set(xlabel='ISE single-base substitution iteration',
           ylabel='C1 preference: robust log1p margin (higher is better)',
           title=f'Borzoi only | {s["target_len"]}-bp, {s["steps"]}-step ISE trajectories')
    ax.grid(axis='y',alpha=.15);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(root/'PILOT_PLATEAU.png',dpi=160)
    fig.savefig(root/'PILOT_PLATEAU.pdf');plt.close(fig)
    lines=['# Borzoi-only ISE plateau monitoring','',
         f'All {s["steps"]} iterations execute; there is no automatic early stopping.',
         'Native optimization selects the best FEASIBLE historical step, not necessarily the endpoint.',
         'The plotted scores use four additional DNA contexts, not used for mutation selection.',
         'These contexts originate from TRAINING genome regions, not held-out biological data.',
         'Do not use final-audit contexts to select a design or hyperparameters.','',
         'Scores are model predictions, not reporter activity or exclusive cell-state expression.','',
         'Median robust log1p margins by arm:']
    for r in records:
        lines.append(f'- {r["arm"]}: n={r["designs"]}; start={r["median_margin_at_0"]:.4f}; '
                     f'end={r[f"median_margin_at_{s["steps"]}"]:.4f}; '
                     f'last 25-step median delta={r[f"median_gain_last_{min(25,s["steps"])}_steps"]:+.4f}')
    (root/'PILOT_INTERPRETATION.md').write_text('\n'.join(lines)+'\n')
    files=[root/n for n in ('PILOT_ITERATION_METRICS.tsv','PILOT_PLATEAU_SUMMARY.tsv',
                           'PILOT_PLATEAU.png','PILOT_PLATEAU.pdf','PILOT_INTERPRETATION.md')]
    checkpoint(root,'PILOT_PLATEAU_COMPLETE',files)
    dump(root/'PILOT_PLATEAU_STATUS.json',dict(status='complete',designs=len(plans),iterations=s['steps'],
         model='Borzoi_only',monitoring_contexts=s['selection_contexts'],biological_validation=False))
