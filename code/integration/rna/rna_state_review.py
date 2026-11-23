"""Independent RNA clustering/marker review on Betty; source expression is read-only."""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc


def need(ok, message):
    if not ok:
        raise ValueError(message)


def say(*v):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *v, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n'); q.replace(p)


def table(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    d.to_csv(q, sep='\t', index=False,
        compression={'method': 'gzip', 'mtime': 0} if p.name.endswith('.gz') else None)
    q.replace(p)


def write_h5(p, a):
    q = p.with_name(p.stem + '.partial.h5ad')
    a.write_h5ad(q, compression='gzip'); q.replace(p)


def checkpoint(folder, name, signature, files, info=None):
    dump(folder/(name + '.json'), dict(signature=signature,
         output_SHA256={str(p.relative_to(folder)): sha(p) for p in files}, info=info or {}))


def reuse(folder, name, signature):
    p = folder/(name + '.json')
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    need(d['signature'] == signature, 'Checkpoint settings changed: ' + str(p))
    for n, want in d['output_SHA256'].items():
        need((folder/n).is_file() and sha(folder/n) == want, 'Checkpoint changed: ' + n)
    say('REUSE', folder.name, name)
    return True


def digest_matrix(x):
    """Canonical CSR value/axis fingerprint, independent of HDF5 compression."""
    x = sparse.csr_matrix(x, copy=True); x.sum_duplicates(); x.sort_indices(); x.eliminate_zeros()
    h = hashlib.sha256(np.asarray(x.shape, dtype='<i8').tobytes())
    for v, dtype in [(x.indptr, '<i8'), (x.indices, '<i8'), (x.data, '<f8')]:
        for i in range(0, len(v), 1024**2):
            h.update(np.asarray(v[i:i+1024**2], dtype=dtype).tobytes())
    return h.hexdigest()


def provenance(obs, settings):
    obs = obs.copy()
    need({'scplus_capture', 'source_GSE', 'source_library'}.issubset(obs), 'Source provenance is missing')
    for c in ['scplus_capture', 'source_GSE', 'source_library']:
        obs[c] = obs[c].astype(str)
    need(obs.scplus_capture.eq(obs.source_GSE + '__' + obs.source_library).all(), 'Capture provenance disagrees')
    models = settings['confirmed_TIL_captures']
    obs['rna_review_confirmed_solid_TIL'] = obs.scplus_capture.isin(models)
    obs['rna_review_tumor_model'] = obs.scplus_capture.map(models).fillna('NOT_CONFIRMED_SOLID_TIL')
    obs['rna_review_provenance'] = np.where(obs.rna_review_confirmed_solid_TIL,
        'CONFIRMED_SOLID_TUMOR_SOURCE', 'COMPARATOR_NON_TIL')
    unknown = obs.scplus_capture.isin(settings['unresolved_condition_captures'])
    obs.loc[unknown, 'rna_review_provenance'] = 'CONDITION_UNRESOLVED'
    # Source labels are evidence of origin; no transferred state determines TIL status.
    return obs


def marker_scores(a, panels, folder, seed, aliases, extra):
    """Same full-cohort control background for both all-cell and TIL reports."""
    need(a.var_names.is_unique, 'Repeated gene identifiers need review before scoring')
    availability = []; marker_map = {}; methods = {}
    for g in dict.fromkeys([g for v in panels.values() for g in v] + extra):
        if g in a.var_names:
            marker_map[g]=g; methods[g]='EXACT_CANONICAL_SYMBOL'
        else:
            matches=[x for x in aliases.get(g,[]) if x in a.var_names]
            need(len(matches)<=1,'Multiple measured aliases for '+g+'; review rather than combining them')
            marker_map[g]=matches[0] if matches else None
            methods[g]='UNIQUE_VERIFIED_MOUSE_ALIAS' if matches else 'MISSING'
    for program, genes in list(panels.items()) + [('review_only', extra)]:
        found = list(dict.fromkeys(marker_map[g] for g in genes if marker_map[g] is not None))
        for g in genes:
            feature=marker_map[g]
            availability.append(dict(program=program, mouse_symbol=g, measured_feature=feature or '',
                match=methods[g], present=feature is not None,
                detected_cells=int((a[:,feature].X>0).sum()) if feature else 0))
        if program=='review_only':continue
        need(len(found) >= min(3, len(genes)), 'Insufficient marker coverage: ' + program)
        sc.tl.score_genes(a, gene_list=found, score_name='rna_program_' + program,
            ctrl_size=50, n_bins=25, random_state=seed, use_raw=False)
    table(folder/'MARKER_GENE_AVAILABILITY.tsv', pd.DataFrame(availability))
    return marker_map


def pca_stage(a, folder, s, signature):
    if reuse(folder, 'PCA_COMPLETE', signature):
        return ad.read_h5ad(folder/'PCA.h5ad')
    say('HVG / PCA', folder.name, a.shape)
    expressed = np.asarray((a.X > 0).sum(axis=0)).ravel() >= 10
    nuisance = np.array([bool(re.match(r'^(mt-|Rpl|Rps|Tr[abdg][vj])', g)) for g in a.var_names])
    eligible = expressed & ~nuisance
    need(eligible.sum() >= 60, 'Too few expressed eligible genes')
    h = ad.AnnData(X=a.X[:, eligible].copy(), obs=a.obs.copy(), var=a.var.loc[eligible].copy())
    # Seurat dispersion flavor expects log-normalized RNA, unlike seurat_v3.
    h.uns['log1p'] = {'base': None}
    sc.pp.highly_variable_genes(h, flavor='seurat', n_top_genes=min(s['hvg_genes'], h.n_vars),
                               batch_key='source_GSE')
    hvg = h.var.copy(); hvg.insert(0, 'gene', hvg.index)
    table(folder/'HVG_SELECTION.tsv.gz', hvg.reset_index(drop=True))
    h = h[:, h.var.highly_variable].copy()
    sc.pp.scale(h, zero_center=True, max_value=10)
    n_pcs = min(s['pca_components'], h.n_obs - 1, h.n_vars - 1)
    sc.tl.pca(h, n_comps=n_pcs, svd_solver='arpack', random_state=s['seed'])
    r = ad.AnnData(X=sparse.csr_matrix((a.n_obs, 0)), obs=a.obs.copy())
    r.obsm['X_pca'] = h.obsm['X_pca'].astype(np.float32)
    r.uns['pca'] = h.uns['pca']
    table(folder/'PCA_LOADINGS.tsv.gz', pd.DataFrame(h.varm['PCs'], index=h.var_names)
          .rename_axis('gene').reset_index())
    variance = pd.DataFrame({'PC': np.arange(n_pcs)+1, 'variance_ratio': h.uns['pca']['variance_ratio']})
    table(folder/'PCA_VARIANCE.tsv', variance)
    write_h5(folder/'PCA.h5ad', r)
    checkpoint(folder, 'PCA_COMPLETE', signature,
        [folder/n for n in ['PCA.h5ad', 'PCA_LOADINGS.tsv.gz', 'PCA_VARIANCE.tsv', 'HVG_SELECTION.tsv.gz']])
    del h; gc.collect()
    return r


def harmony_stage(r, folder, s, signature):
    if reuse(folder, 'HARMONY_COMPLETE', signature):
        r.obsm['X_pca_harmony'] = np.load(folder/'HARMONY_PCS.npy')
        return
    say('HARMONY', folder.name, 'study only; biological condition is not a correction factor')
    import harmonypy as hm
    n_pcs = min(s['neighbor_pcs'], r.obsm['X_pca'].shape[1])
    meta = pd.DataFrame({'source_GSE': r.obs.source_GSE.astype(str)}, index=r.obs_names)
    ho = hm.run_harmony(r.obsm['X_pca'][:, :n_pcs].astype(np.float64), meta, ['source_GSE'],
        theta=s['harmony_theta'], lamb=1.0, max_iter_harmony=s['harmony_max_iter'],
        nclust=min(100, max(2, int(round(r.n_obs/30)))), random_state=s['seed'], verbose=True)
    z = ho.Z_corr.T.astype(np.float32)
    need(z.shape == (r.n_obs, n_pcs) and np.isfinite(z).all(), 'Invalid Harmony result')
    r.obsm['X_pca_harmony'] = z
    np.save(folder/'HARMONY_PCS.npy', z)
    dump(folder/'HARMONY_DIAGNOSTICS.json', {'objective_harmony': list(map(float, ho.objective_harmony)),
        'iterations': len(ho.objective_harmony)-1, 'batch_key': 'source_GSE',
        'interpretation': 'Sensitivity analysis: study is confounded with biology; mixing is not proof of correctness.'})
    checkpoint(folder, 'HARMONY_COMPLETE', signature,
        [folder/'HARMONY_PCS.npy', folder/'HARMONY_DIAGNOSTICS.json'])


def graph_stage(r, folder, method, s, signature):
    name = method.upper() + '_GRAPH_COMPLETE'
    p = folder/(method + '_GRAPH.h5ad')
    if reuse(folder, name, signature):
        return ad.read_h5ad(p)
    say('NEIGHBORS / LEIDEN / UMAP', folder.name, method)
    e = ad.AnnData(X=sparse.csr_matrix((r.n_obs, 0)), obs=r.obs.copy())
    basis = 'X_pca_harmony' if method == 'harmony' else 'X_pca'
    e.obsm[basis] = r.obsm[basis]
    sc.pp.neighbors(e, n_neighbors=min(s['neighbors'], e.n_obs-1), use_rep=basis,
        n_pcs=min(s['neighbor_pcs'], e.obsm[basis].shape[1]), random_state=s['seed'], metric='euclidean')
    for res in s['resolutions']:
        key = 'rna_' + method + '_leiden_r' + str(res).replace('.', 'p')
        sc.tl.leiden(e, resolution=res, key_added=key, random_state=s['seed'],
                     flavor='leidenalg', directed=False, n_iterations=-1)
    sc.tl.umap(e, min_dist=0.3, spread=1.0, random_state=s['seed'])
    write_h5(p, e)
    checkpoint(folder, name, signature, [p])
    return e


def category_plot(ax, xy, values, title):
    d = pd.Series(np.asarray(values).astype(str)); cats = sorted(d.unique())
    palette = sc.pl.palettes.default_20 + sc.pl.palettes.default_28
    point_size=max(1.4,min(8.0,1500/max(1,len(d))))
    for i, cat in enumerate(cats):
        pick = (d == cat).to_numpy()
        ax.scatter(xy[pick, 0], xy[pick, 1], s=point_size, linewidths=0, rasterized=True,
                   c=palette[i % len(palette)], label=cat)
    if len(cats) <= 26:
        ax.legend(markerscale=4, fontsize=5, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set(title=title, xlabel='UMAP 1', ylabel='UMAP 2'); ax.set_xticks([]); ax.set_yticks([])


def report(a, graphs, folder, panels, s, marker_map):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score
    marker_genes = list(dict.fromkeys([g for genes in panels.values() for g in genes if marker_map[g] is not None] +
                                     [g for g in s['extra_review_genes'] if marker_map[g] is not None]))
    xm = a[:, [marker_map[g] for g in marker_genes]].X.toarray()
    score_cols = [c for c in a.obs if c.startswith('rna_program_')]
    pdf_names = []
    # Labels remain competing evidence, not a training target or proof of batch mixing.
    colors = [c for c in ['source_GSE', 'scplus_capture', 'rna_review_tumor_model',
              'trbi_level2', 'gse267584_best_state'] if c in a.obs]
    comparisons = []
    all_profiles, capture_profiles, review_rows, top_markers = [], [], [], []
    fig, axes=plt.subplots(2,2,figsize=(16,12))
    for j,(method,e) in enumerate(graphs.items()):
        category_plot(axes[0,j],e.obsm['X_umap'],a.obs.source_GSE,method+' | source study')
        category_plot(axes[1,j],e.obsm['X_umap'],e.obs['rna_'+method+'_leiden_r0p6'],method+' | Leiden r=0.6')
    fig.suptitle(folder.name+' | independent UMAP layouts; judge biological evidence, not visual mixing alone')
    fig.tight_layout();fig.savefig(folder/'CORRECTION_OVERVIEW.pdf');fig.savefig(folder/'CORRECTION_OVERVIEW.png',dpi=140);plt.close(fig)
    qc_values={'stored_RNA_counts':np.log10(1+np.asarray(a.raw.X.sum(axis=1)).ravel()),
               'stored_RNA_detected_genes':np.log10(1+np.asarray((a.raw.X>0).sum(axis=1)).ravel())}
    for col in ['unique_fragments_count','tss_enrichment','fraction_of_fragments_in_peaks']:
        if col in a.obs:qc_values[col]=pd.to_numeric(a.obs[col],errors='coerce').to_numpy()
    mito=[c for c in a.obs if re.search('mito|pct_counts_mt|percent_mt',c,re.I)]
    if mito:qc_values[mito[0]]=pd.to_numeric(a.obs[mito[0]],errors='coerce').to_numpy()
    for method, e in graphs.items():
        xy = e.obsm['X_umap']
        clusterkey = 'rna_' + method + '_leiden_r0p6'
        fig, axes = plt.subplots(len(colors)+1, 1, figsize=(12, 5*(len(colors)+1)))
        for ax, color in zip(axes, colors):
            category_plot(ax, xy, a.obs[color], folder.name + ' | ' + method + ' | ' + color)
        category_plot(axes[-1], xy, e.obs[clusterkey], method + ' | Leiden r=0.6 (review resolution)')
        fig.tight_layout(); fig.savefig(folder/(method+'_UMAP_METADATA.pdf')); plt.close(fig)
        pdf_names.append(method+'_UMAP_METADATA.pdf')
        fig, axes = plt.subplots(3, 4, figsize=(17, 12))
        continuous = ['rna_program_'+p for p in panels][:12]
        for ax, col in zip(axes.ravel(), continuous):
            v = a.obs[col].to_numpy(); lo, hi = np.quantile(v, [.02, .98])
            m = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=1.4, linewidths=0, rasterized=True,
                           cmap='viridis', vmin=lo, vmax=max(hi, lo+1e-8))
            ax.set_title(col.replace('rna_program_', ''), fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(m, ax=ax, shrink=.65)
        for ax in axes.ravel()[len(continuous):]: ax.axis('off')
        fig.suptitle(folder.name+' | '+method+' | programs from unchanged log-normalized RNA')
        fig.tight_layout(); fig.savefig(folder/(method+'_UMAP_PROGRAMS.pdf')); plt.close(fig)
        pdf_names.append(method+'_UMAP_PROGRAMS.pdf')
        fig, axes = plt.subplots(3, 4, figsize=(17, 12))
        for ax, gene in zip(axes.ravel(), ['Pdcd1','Tox','Tcf7','Slamf6','Havcr2','Entpd1','Cd101','Cx3cr1','Gzmb','Il7r','Klrg1','Mki67']):
            if gene not in marker_genes:
                ax.axis('off'); continue
            v = xm[:, marker_genes.index(gene)]; hi = max(float(np.quantile(v, .99)), .5)
            m = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=1.4, linewidths=0, rasterized=True, cmap='viridis', vmin=0, vmax=hi)
            ax.set_title(gene); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(m, ax=ax, shrink=.65)
        fig.suptitle(folder.name+' | '+method+' | observed RNA; zero is not a protein-negative gate')
        fig.tight_layout(); fig.savefig(folder/(method+'_UMAP_MARKERS.pdf')); plt.close(fig)
        pdf_names.append(method+'_UMAP_MARKERS.pdf')
        fig,axes=plt.subplots(2,3,figsize=(17,9))
        for ax,(col,v) in zip(axes.ravel(),qc_values.items()):
            finite=np.isfinite(v)
            if not finite.any():ax.axis('off');continue
            lo,hi=np.quantile(v[finite],[.02,.98])
            m=ax.scatter(xy[:,0],xy[:,1],c=v,s=1.4,linewidths=0,rasterized=True,cmap='viridis',vmin=lo,vmax=max(hi,lo+1e-8))
            ax.set_title(col,fontsize=9);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(m,ax=ax,shrink=.65)
        for ax in axes.ravel()[len(qc_values):]:ax.axis('off')
        fig.suptitle(folder.name+' | '+method+' | QC overlays (stored RNA counts/genes shown as log10(1+x))')
        fig.tight_layout();fig.savefig(folder/(method+'_UMAP_QC.pdf'));plt.close(fig)
        capture = a.obs.scplus_capture.astype(str).to_numpy()
        graph = e.obsp['distances'].tocsr()
        same = np.array([(capture[graph.indices[graph.indptr[i]:graph.indptr[i+1]]] == capture[i]).mean()
                         for i in range(e.n_obs)])
        comparisons.append(dict(method=method, median_same_capture_neighbor_fraction=float(np.median(same)),
            mean_same_capture_neighbor_fraction=float(np.mean(same)),
            interpretation='Descriptive; biological conditions are not exchangeable across captures.'))
        table(folder/(method+'_NEIGHBOR_CAPTURE_DIAGNOSTICS.tsv.gz'),
              pd.DataFrame({'Cell_ID': a.obs_names, 'same_capture_neighbor_fraction': same}))
        group = e.obs[clusterkey].astype(str).to_numpy()
        # Full gene summaries calculated on original normalized expression, never corrected PCs.
        linear = a.X.copy(); linear.data = np.expm1(linear.data)
        total = np.asarray(linear.sum(axis=0)).ravel().astype(np.float64)
        groupmeans, fractions, groups = [], [], sorted(set(group), key=lambda x: int(x))
        for cluster in groups:
            ix = group == cluster; n = int(ix.sum()); caps = capture[ix]
            mean = np.asarray(a.X[ix].mean(axis=0)).ravel()
            sums = np.asarray(linear[ix].sum(axis=0)).ravel().astype(np.float64)
            frac = np.asarray((a.X[ix] > 0).mean(axis=0)).ravel()
            rest = (total-sums)/max(1, a.n_obs-n)
            logfc = np.log2((sums/n+.1)/(np.maximum(rest, 0)+.1))
            eligible = (frac >= .1) & (logfc > 0)
            indices = np.flatnonzero(eligible)[np.argsort(logfc[eligible])[::-1][:40]]
            for j in indices:
                top_markers.append(dict(method=method, cluster=cluster, cells=n, gene=a.var_names[j],
                    log2_mean_CP10k_ratio_vs_rest=float(logfc[j]), detection_fraction=float(frac[j]),
                    mean_log1p_CP10k=float(mean[j]), note='Descriptive cluster marker; no replicate-level significance test'))
            m = xm[ix].mean(axis=0); f = (xm[ix] > 0).mean(axis=0)
            groupmeans.append(m); fractions.append(f)
            counts = pd.Series(caps).value_counts()
            row = dict(method=method, cluster=cluster, cells=n, captures=int(len(counts)),
                largest_capture_fraction=float(counts.iloc[0]/n),
                confirmed_TIL_cells=int(a.obs.rna_review_confirmed_solid_TIL.to_numpy()[ix].sum()),
                label_review_status='UNREVIEWED', differentiation_label='', exhaustion_label='',
                residency_label='', activation_cycling_label='', evidence_notes='')
            for col in score_cols: row[col] = float(a.obs.loc[ix, col].mean())
            review_rows.append(row)
            for gene, mv, fv in zip(marker_genes, m, f):
                all_profiles.append(dict(method=method, cluster=cluster, cells=n, gene=gene,
                    measured_feature=marker_map[gene],
                    mean_log1p_CP10k=float(mv), detection_fraction=float(fv)))
            for cap in sorted(set(caps)):
                ci = ix & (capture == cap); cm=xm[ci].mean(axis=0); cf=(xm[ci] > 0).mean(axis=0)
                for gene, mv, fv in zip(marker_genes, cm, cf):
                    capture_profiles.append(dict(method=method, cluster=cluster, scplus_capture=cap,
                        cells=int(ci.sum()), gene=gene, measured_feature=marker_map[gene],
                        mean_log1p_CP10k=float(mv), detection_fraction=float(fv)))
        del linear
        means = np.asarray(groupmeans); fracs=np.asarray(fractions)
        # Per-gene color scale for readability; raw means and detection are in the TSV.
        scaled = (means-means.min(axis=0))/(np.ptp(means,axis=0)+1e-8)
        fig, ax=plt.subplots(figsize=(max(17, len(marker_genes)*.24), max(4, len(groups)*.32)))
        gx, gy=np.meshgrid(np.arange(len(marker_genes)),np.arange(len(groups)))
        m=ax.scatter(gx.ravel(),gy.ravel(),s=fracs.ravel()*100,c=scaled.ravel(),cmap='Reds',vmin=0,vmax=1)
        ax.set_xticks(np.arange(len(marker_genes)),marker_genes,rotation=90,fontsize=7)
        ax.set_yticks(np.arange(len(groups)),groups);ax.invert_yaxis()
        ax.set_title(method+' | dot size: RNA detection fraction; color: per-gene scaled mean; Leiden r=0.6')
        fig.colorbar(m,ax=ax,label='Scaled mean log1p(CP10k)');fig.tight_layout()
        fig.savefig(folder/(method+'_CLUSTER_MARKER_DOTPLOT.pdf'));plt.close(fig)
        pdf_names.append(method+'_CLUSTER_MARKER_DOTPLOT.pdf')
        for label in ['scplus_capture','rna_review_tumor_model','trbi_level2','gse267584_best_state']:
            if label in a.obs:
                out=pd.crosstab(e.obs[clusterkey],a.obs[label],dropna=False).rename_axis('cluster').reset_index()
                table(folder/(method+'_CLUSTER_BY_'+label+'.tsv'),out)
    for resolution in s['resolutions']:
        suffix='_leiden_r'+str(resolution).replace('.','p')
        comparisons.append(dict(resolution=resolution, uncorrected_vs_harmony_adjusted_Rand_index=float(
            adjusted_rand_score(graphs['uncorrected'].obs['rna_uncorrected'+suffix],
                                graphs['harmony'].obs['rna_harmony'+suffix]))))
    dump(folder/'CORRECTION_COMPARISON.json', comparisons)
    table(folder/'CLUSTER_LABEL_REVIEW.tsv',pd.DataFrame(review_rows))
    table(folder/'CLUSTER_MARKER_VALUES.tsv.gz',pd.DataFrame(all_profiles))
    table(folder/'CLUSTER_CAPTURE_MARKER_VALUES.tsv.gz',pd.DataFrame(capture_profiles))
    table(folder/'TOP_DESCRIPTIVE_CLUSTER_MARKERS.tsv.gz',pd.DataFrame(top_markers))
    variance=pd.read_csv(folder/'PCA_VARIANCE.tsv',sep='\t')
    fig,ax=plt.subplots();ax.plot(variance.PC,variance.variance_ratio,marker='.')
    ax.set(xlabel='PC',ylabel='Explained variance ratio',title=folder.name+' | PCA diagnostic; neighbors use first '+str(s['neighbor_pcs'])+' PCs')
    fig.tight_layout();fig.savefig(folder/'PCA_VARIANCE.pdf');plt.close(fig)


def cohort(a, folder, s, panels, signature, marker_map):
    folder.mkdir(parents=True,exist_ok=True)
    if reuse(folder,'COHORT_COMPLETE',signature):
        return json.loads((folder/'SUMMARY.json').read_text())
    before_x=digest_matrix(a.X);before_raw=digest_matrix(a.raw.X)
    old_obs=a.obs.copy()
    r=pca_stage(a,folder,s,signature)
    harmony_stage(r,folder,s,signature)
    graphs={m:graph_stage(r,folder,m,s,signature) for m in ['uncorrected','harmony']}
    if not reuse(folder,'REPORT_COMPLETE',signature):
        report(a,graphs,folder,panels,s,marker_map)
        reports=list(folder.glob('*.pdf'))+list(folder.glob('*.png'))+list(folder.glob('*MARKER*.tsv*'))+list(folder.glob('*CLUSTER*.tsv*'))
        reports+=list(folder.glob('*NEIGHBOR*.tsv*'))+[folder/'CORRECTION_COMPARISON.json']
        checkpoint(folder,'REPORT_COMPLETE',signature,sorted(set(reports)))
    a.obsm['X_pca_rna_review']=r.obsm['X_pca']
    a.obsm['X_pca_harmony_rna_review']=r.obsm['X_pca_harmony']
    for method,e in graphs.items():
        for c in e.obs:
            if c.startswith('rna_') and '_leiden_' in c: a.obs[c]=e.obs[c]
        a.obsm['X_umap_rna_'+method]=e.obsm['X_umap']
        for k,v in e.obsp.items():a.obsp['rna_'+method+'_'+k]=v
    a.uns['rna_state_review']={'source_run_signature':signature,'default_embedding':'UNSELECTED_PENDING_REVIEW',
        'final_labels_assigned':False,'normalization':'Source log1p(CP10k), unchanged; raw.X holds original counts',
        'marker_source':'User-supplied T-cell nomenclature table, translated to mouse RNA with explicit limitations'}
    p=folder/'RNA_WITH_REVIEW_EMBEDDINGS.h5ad'
    write_h5(p,a)
    v=ad.read_h5ad(p,backed='r')
    need(v.obs_names.equals(a.obs_names) and v.var_names.equals(a.var_names),'Saved RNA IDs/genes changed')
    # Compare in chunks, including raw counts; avoids a second whole object in RAM.
    for start in range(0,a.n_obs,2048):
        stop=min(a.n_obs,start+2048)
        need((sparse.csr_matrix(v.X[start:stop])-a.X[start:stop]).nnz==0,'Saved normalized RNA changed')
        need((sparse.csr_matrix(v.raw.X[start:stop])-a.raw.X[start:stop]).nnz==0,'Saved raw RNA changed')
    need(v.raw.var_names.equals(a.raw.var_names),'Saved raw gene names changed')
    for c in old_obs:
        need(np.array_equal(v.obs[c].astype(str).to_numpy(),old_obs[c].astype(str).to_numpy()),'Existing annotation changed: '+c)
    v.file.close()
    need(digest_matrix(a.X)==before_x and digest_matrix(a.raw.X)==before_raw,'RNA expression mutated during review')
    out=a.obs.drop(columns=['Cell_ID'],errors='ignore').copy();out.insert(0,'Cell_ID',a.obs_names)
    for method,e in graphs.items():
        out['UMAP_'+method+'_1']=e.obsm['X_umap'][:,0];out['UMAP_'+method+'_2']=e.obsm['X_umap'][:,1]
    table(folder/'CELL_LABELS_AND_EMBEDDINGS.tsv.gz',out)
    info=dict(cells=a.n_obs,genes=a.n_vars,captures=a.obs.scplus_capture.nunique(),
        confirmed_solid_TIL_cells=int(a.obs.rna_review_confirmed_solid_TIL.sum()),
        normalized_RNA_unchanged=True,raw_counts_unchanged=True,existing_annotations_unchanged=True,
        RNA_X_fingerprint=before_x,raw_count_fingerprint=before_raw,
        final_phenotype_labels_assigned=False,correction_choice='UNSELECTED_PENDING_REVIEW')
    dump(folder/'SUMMARY.json',info)
    checkpoint(folder,'COHORT_COMPLETE',signature,[p,folder/'CELL_LABELS_AND_EMBEDDINGS.tsv.gz',folder/'SUMMARY.json'])
    say('COHORT COMPLETE',folder.name,info)
    return info


def smoke(folder):
    """Actual Scanpy/Harmony/Leiden/UMAP fit on synthetic cells, before the real input."""
    rng=np.random.default_rng(555);n=180;p=120
    x=rng.poisson(1,size=(n,p)).astype(np.float32)
    x[:90,:20]+=rng.poisson(3,size=(90,20));x[90:,20:40]+=rng.poisson(3,size=(90,20))
    a=ad.AnnData(sparse.csr_matrix(x),obs=pd.DataFrame({'source_GSE':['a','b']*(n//2)},index=['test'+str(i) for i in range(n)]))
    sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a);sc.pp.scale(a,max_value=10)
    sc.tl.pca(a,n_comps=12,random_state=555)
    import harmonypy as hm
    z=hm.run_harmony(a.obsm['X_pca'],a.obs,['source_GSE'],nclust=4,max_iter_harmony=3,random_state=555).Z_corr.T
    a.obsm['X_harmony']=z
    sc.pp.neighbors(a,use_rep='X_harmony',n_neighbors=10,random_state=555)
    sc.tl.leiden(a,key_added='test_leiden',flavor='leidenalg',directed=False,random_state=555)
    sc.tl.umap(a,random_state=555,maxiter=30)
    need(np.isfinite(a.obsm['X_umap']).all() and a.obs.test_leiden.nunique()>1,'Functional environment check failed')
    dump(folder/'FUNCTIONAL_SMOKE_TEST.json',dict(status='PASS',synthetic_cells=n,real_data_used=False,
         Harmony=True,Leiden=True,UMAP=True))
    say('FUNCTIONAL SCANPY / HARMONY / LEIDEN / UMAP TEST PASSED')


def main(run):
    s=json.loads((run/'SETTINGS.json').read_text());panels=json.loads((run/'MOUSE_PROGRAMS.json').read_text())
    hashes=json.loads((run/'SOURCES_SHA256.json').read_text())
    for n,want in hashes.items():need(sha(run/n)==want,'Frozen code/settings changed: '+n)
    signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if reuse(run,'RUN_COMPLETE',signature):return
    paths=json.loads((run/'INPUT_PATHS.json').read_text());common=Path(paths['common_run'])
    manifest=json.loads((common/'RUN_MANIFEST.json').read_text())
    need(manifest['status']=='COMPLETE','Source common matrices are incomplete')
    for n,want in s['input_SHA256'].items():
        say('VERIFY SOURCE',n)
        need(sha(common/n)==want,'Reviewed source input changed: '+n)
    sc.settings.n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK','4'))
    if not (run/'FUNCTIONAL_SMOKE_TEST.json').exists():smoke(run)
    say('READ reviewed RNA; preserve source normalization, counts and existing annotations')
    a=ad.read_h5ad(common/'RNA_QC_PASSED.h5ad')
    need(a.shape==(s['expected_cells'],s['expected_genes']),'RNA cohort changed')
    need(sparse.issparse(a.X) and a.raw is not None,'Expected sparse normalized X and raw RNA counts')
    need(a.var_names.equals(a.raw.var_names) and a.obs_names.is_unique,'RNA axes changed')
    need(np.isfinite(a.X.data).all() and (a.X.data>=0).all(),'Invalid normalized expression')
    need(np.isfinite(a.raw.X.data).all() and (a.raw.X.data>=0).all() and
         np.equal(a.raw.X.data,np.floor(a.raw.X.data)).all(),'raw.X does not contain nonnegative integer counts')
    retained=pd.read_csv(common/'RETAINED_CELLS.tsv.gz',sep='\t',dtype=str,keep_default_na=False)
    need(retained.Cell_ID.is_unique and list(a.obs_names)==list(retained.Cell_ID),'RNA and retained IDs/order differ')
    need(not a.obs.source_GSE.astype(str).eq('GSE278342').any(),'W8/W9 unexpectedly included')
    a.obs=provenance(a.obs,s)
    observed=a.obs.groupby('scplus_capture',observed=True).size().to_dict()
    for cap,n in s['expected_TIL_counts'].items():need(observed.get(cap)==n,'Confirmed TIL source/count changed: '+cap)
    need(int(a.obs.rna_review_confirmed_solid_TIL.sum())==s['expected_TIL_cells'],'TIL subset changed')
    table(run/'PROVENANCE_BY_CAPTURE.tsv',a.obs.groupby(['scplus_capture','source_GSE','rna_review_provenance',
          'rna_review_tumor_model'],observed=True).size().rename('cells').reset_index())
    aliases=json.loads((run/'MOUSE_SYMBOL_ALIASES.json').read_text())
    marker_map=marker_scores(a,panels,run,s['seed'],aliases,s['extra_review_genes'])
    # TILs are re-embedded independently, rather than cropped from the combined UMAP.
    til=a[a.obs.rna_review_confirmed_solid_TIL].copy()
    info={}
    info['confirmed_TIL']=cohort(til,run/'confirmed_TIL',s,panels,signature,marker_map)
    del til;gc.collect()
    info['all_paired_CD8']=cohort(a,run/'all_paired_CD8',s,panels,signature,marker_map)
    dump(run/'SUMMARY.json',dict(status='COMPLETE',cohorts=info,
        scope='Step 07 paired CD8 cells; separate from the larger 161773-cell RNA master',
        interpretation='No final phenotype or preferred correction is assigned automatically; review marker and sample support.'))
    reports=[run/'SUMMARY.json',run/'PROVENANCE_BY_CAPTURE.tsv',run/'MARKER_GENE_AVAILABILITY.tsv',
        run/'SETTINGS.json',run/'MOUSE_PROGRAMS.json',run/'MOUSE_SYMBOL_ALIASES.json',run/'README_RUN.md',run/'SOURCES_SHA256.json',
        run/'FUNCTIONAL_SMOKE_TEST.json',run/'STATE_INTERPRETATION.tsv']
    for name in ['confirmed_TIL','all_paired_CD8']:
        folder=run/name
        reports+=list(folder.glob('*.pdf'))+list(folder.glob('*.png'))+list(folder.glob('*.tsv'))+list(folder.glob('*.tsv.gz'))
        reports+=[folder/'SUMMARY.json',folder/'CORRECTION_COMPARISON.json',folder/'HARMONY_DIAGNOSTICS.json']
    outputs={}
    for name in ['confirmed_TIL','all_paired_CD8']:
        d=json.loads((run/name/'COHORT_COMPLETE.json').read_text())
        outputs.update({name+'/'+p:want for p,want in d['output_SHA256'].items()})
    dump(run/'RUN_MANIFEST.json',dict(status='COMPLETE',signature=signature,output_SHA256=outputs,
         input_SHA256=s['input_SHA256'],report_SHA256={str(p.relative_to(run)):sha(p) for p in sorted(set(reports))}))
    reports.append(run/'RUN_MANIFEST.json')
    p=run/'SHARE_RNA_STATE_REVIEW.zip';q=p.with_name(p.name+'.tmp')
    with zipfile.ZipFile(q,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(set(reports)):z.write(f,str(f.relative_to(run)))
    q.replace(p)
    checkpoint(run,'RUN_COMPLETE',signature,[run/'SUMMARY.json',run/'RUN_MANIFEST.json',p])
    say('COMPLETE',run/'SHARE_RNA_STATE_REVIEW.zip')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:smoke(args.run.resolve())
    else:main(args.run.resolve())
