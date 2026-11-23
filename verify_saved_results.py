"""Recompute saved validation/test metrics on CPU, without loading the model.

Requires NumPy. Uses original recorded population/scoring functions. Average
precision is calculated directly at distinct score thresholds, with tied scores
grouped. Saved published metrics are not overwritten.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np

def average_precision(y,s):
    if not 0<y.sum()<len(y):return None
    order=np.argsort(-s,kind='stable')
    labels=y[order];score=s[order]
    ends=np.r_[np.flatnonzero(np.diff(score)),len(score)-1]
    tp=np.cumsum(labels,dtype=np.int64)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*tp/(ends+1)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-data',type=Path,required=True,
                    help='Extracted C1_SOURCE_DATA directory, containing training/')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'code/training'))
    import math_core
    data=args.source_data/'training'
    meta=json.loads((data/'data/DATASET.json').read_text())
    rates=np.load(data/'data/rates.npy',mmap_mode='r',allow_pickle=False)
    counts=np.load(data/'data/counts.npy',mmap_mode='r',allow_pickle=False)
    differences=[]
    with np.load(data/'model/PREDICTIONS.npz',allow_pickle=False) as z:
        for name,key,settings,filename in [
            ('validation','validation',meta,'VALIDATION_SELECTIVITY.tsv'),
            ('primary test','test',meta,'TEST_SELECTIVITY.tsv'),
            ('all-comparator test','test',dict(meta,include_ambiguous_backgrounds=True),'TEST_ALL_GROUPS_SENSITIVITY.tsv')]:
            ids=z[key+'_ids'];pred=z[key+'_predictions']
            with (data/'model'/filename).open() as f:
                saved={r['target']:r for r in csv.DictReader(f,delimiter='\t')}
            actual={}
            for target,group,labels,score,extra in math_core.count_scores(rates[ids],pred,counts[ids],settings):
                observed=average_precision(labels,score)
                expected=float(saved[target]['AP']) if saved[target]['AP'] else None
                assert int(labels.sum())==int(saved[target]['positives']),(name,target,'positive count differs')
                if expected is None:
                    assert observed is None
                else:
                    delta=abs(expected-observed);differences.append(delta)
                    # Small numeric differences were observed between the original
                    # library calculation and CPU recomputation. Preserve and report
                    # them; reject discrepancies above this explicit tolerance.
                    assert delta<1e-7,(name,target,expected,observed)
                for k in [20,50,100,500]:
                    v=math_core.top_counts(labels,score,k)
                    assert np.isclose(v['hits'],float(saved[target][f'top{k}_hits']),rtol=0,atol=1e-10)
                actual[target]=observed
            print(name+': C1 AP = '+str(actual['TARGET_C1'])+'; positive counts and top-k hits verified')
            if key=='validation':
                macro=float(np.mean([actual[t] for t in meta['checkpoint_targets']]))
                expected_macro=json.loads((data/'model/BEST.json').read_text())['metrics']['macro_selectivity_AP']
                assert abs(macro-expected_macro)<1e-7
                print('Validation macro-AP =',macro,'across',meta['checkpoint_targets'])
    print('PASS. Maximum absolute AP discrepancy versus original saved metrics:',max(differences))
    print('Saved reported metrics were not modified; no model inference was performed.')

if __name__=='__main__':main()
