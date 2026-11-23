import numpy as np
from util import need


def label_panel(values, minimum=200, target=None):
    values = np.asarray(values, dtype=object)
    if target is not None:
        values = values.copy()
        values[np.asarray(target, bool)] = 'TARGET_C1'
    unique, counts = np.unique(values, return_counts=True)
    small = set(unique[counts < minimum]) - {'TARGET_C1'}
    # Small ATAC backgrounds are pooled only; biological labels are retained.
    if target is not None:
        values[np.isin(values, list(small))] = 'OTHER_SMALL_ATAC'
    names = sorted(set(values))
    codes = np.array([names.index(x) for x in values], dtype=np.int32)
    return names, codes


def original_states(rows):
    values = [r['gse267584_reference_state'] for r in rows]
    need(not any(x in ['', 'nan', 'None', 'NA', 'Foxp3_high_TIL'] for x in values), 'Original reference states missing')
    return values


def members(names, panel):
    unions = []
    for title, values in ([('CD8.R_or_CD8.S', ['CD8.R', 'CD8.S'])] if panel == 'immgent' else
                          [('TIL_Tex_term_or_Tex_term', ['TIL Tex-term', 'Tex-term'])] if panel == 'reference_parent' else []):
        if set(values).issubset(names):
            unions.append((title, [names.index(x) for x in values]))
    return [(n, [j]) for j, n in enumerate(names)] + unions


def pooled(values, exposure, indices):
    weights = np.asarray(exposure)[indices]
    return np.asarray(values)[:, indices] @ (weights / weights.sum())


def background_indices(meta, group):
    ambiguous = {'not classified', 'unknown', 'cd8', 'other_small_atac'}
    return np.array([j for j, (name, n) in enumerate(zip(meta['names'], meta['cells']))
                     if j not in group and (n >= 50 or meta.get('include_ambiguous_backgrounds', False)) and
                     (meta.get('include_ambiguous_backgrounds', False) or name.lower() not in ambiguous)], dtype=int)


def top_counts(labels, score, k):
    labels = np.asarray(labels, bool)
    k = min(k, len(score))
    cut = np.partition(score, len(score) - k)[len(score) - k]
    high, tied = score > cut, score == cut
    hits = float(labels[high].sum() + (k - high.sum()) * labels[tied].mean())
    return dict(selected=k, hits=hits, precision=hits/k, recall=hits/labels.sum() if labels.any() else None)


def binary(labels, score):
    from sklearn.metrics import average_precision_score
    need(np.isfinite(score).all(), 'Nonfinite selection score')
    positive = np.asarray(labels, bool)
    result = dict(positives=int(positive.sum()), prevalence=float(positive.mean()),
                  AP=float(average_precision_score(positive, score)) if 0 < positive.sum() < len(positive) else None)
    for k in [20, 50, 100, 500]:
        result.update({'top'+str(k)+'_'+key: value for key, value in top_counts(positive, score, k).items()})
    return result


def count_scores(observed, predicted, counts, meta):
    """FRPM values on identical axes; the same measured labels score all models."""
    names = meta['names']
    cells, exposure = np.asarray(meta['cells']), np.asarray(meta['exposure'])
    for name, group in members(names, meta['panel']):
        other = background_indices(meta, group)
        if not len(other):
            continue
        a, p = pooled(observed, exposure, group), pooled(predicted, exposure, group)
        off_a, off_p = observed[:, other].max(axis=1), predicted[:, other].max(axis=1)
        n = counts[:, group].sum(axis=1)
        truth = (n >= 10) & (a > 0) & (a >= 2 * off_a)
        score = np.log1p(p) - np.log1p(off_p)
        yield name, group, truth, score, dict(target_cells=int(cells[group].sum()),
             comparison='target counts>=10 and FRPM>=2fold; strongest nonambiguous>=50cell competitor in primary; all competitors in all-groups sensitivity',
             predicted_target_median=float(np.median(p)),
             off_target_log_MSE=float(np.square(np.log1p(off_a)-np.log1p(off_p)).mean()))


def rc_flag(index, epoch, seed):
    z = (int(index)*0x45d9f3b + int(epoch)*0x9e3779b9 + int(seed)) & 0xffffffff
    return bool(((z ^ (z >> 16))*0x45d9f3b) & 0x10000)
