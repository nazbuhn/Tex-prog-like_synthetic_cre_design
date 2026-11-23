import gc,gzip,hashlib,importlib.metadata,json,pathlib,re,runpy,sys,traceback,zipfile
import numpy as np,pandas as pd,anndata as ad,scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
raw,prior,step6,step7,out=map(pathlib.Path,sys.argv[1:6])
SEED=1729
sc.settings.n_jobs=1
PARAMS={'seed':SEED,'normalization_target':10000,'HVG_target':2000,'PCA_max_components':30,'neighbors':15,'UMAP_min_dist':.4,'Leiden_resolution':.6,'Leiden_iterations':2,'model_gate':'RNA counts >500; detected symbols >300; mito <=20%; provisional for plots','gene_exclusions_for_embedding_only':'mitochondrial, ribosomal Rpl/Rps, TCR variable/joining segments; genes expressed in <3 model cells','no_doublet_or_lineage_filter':True}
MARKERS='Cd3d Cd3e Cd3g Trac Trbc1 Trbc2 Cd8a Cd8b1 Cd4 Foxp3 Il2ra Ctla4 Ikzf2 Tnfrsf18 Trdc Trgc1 Trgc2 Trgc3 Trgc4 Sox13 Rorc Zbtb16 Ncr1 Klrb1c Klrd1 Nkg7 Ms4a1 Cd79a Lyz2 Lst1 Csf1r Tcf7 Ccr7 Il7r Pdcd1 Havcr2 Lag3 Tox Gzmb Prf1 Cd69 Itgae Cxcr6 Entpd1'.split()
PRIORITY={'GSM8545120','GSM8545124','GSM8269399','GSM8269393','gut','tumor'}
summaries=[]; all_clusters=[]; errors=[]
def say(*v): print(*v,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def dump(p,v):
 temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n');temp.replace(p)
def truth(s): return s.astype(str).str.lower().eq('true').to_numpy()
def number(s): return pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
def software(): return {p:importlib.metadata.version(p) for p in ['scanpy','anndata','numpy','scipy','pandas','umap-learn','pynndescent','scikit-learn','numba','llvmlite','matplotlib','igraph','leidenalg','texttable']}
VERSIONS=software();CODE_SHA=sha(pathlib.Path(__file__))

def plot_umap(folder,label,q,mask,xy,clusters,expression,raw_markers):
 fig,axes=plt.subplots(3,3,figsize=(15,13));axes=axes.ravel();sub=q.loc[mask].reset_index(drop=True)
 def categorical(ax,values,title):
  names=list(dict.fromkeys(map(str,values)));cmap=plt.get_cmap('tab20',max(1,len(names)))
  for i,name in enumerate(names):
   take=np.asarray(values).astype(str)==name
   ax.scatter(xy[take,0],xy[take,1],c=[cmap(i)],s=2,label=name,rasterized=True)
  ax.set_title(title,fontsize=10)
  if len(names)<=12: ax.legend(markerscale=3,fontsize=6,loc='best')
  else:
   for name in names:
    take=np.asarray(values).astype(str)==name;center=np.median(xy[take],axis=0);ax.text(*center,name,fontsize=7)
 def continuous(ax,values,title,vmax=None):
  values=np.asarray(values,dtype=float);good=np.isfinite(values)
  ax.set_title(title,fontsize=10)
  if not good.any(): ax.text(.5,.5,'Gene/value unavailable',transform=ax.transAxes,ha='center');return
  points=ax.scatter(xy[good,0],xy[good,1],c=values[good],s=2,cmap='viridis',vmin=0,vmax=vmax,rasterized=True)
  fig.colorbar(points,ax=ax,shrink=.7)
 categorical(axes[0],clusters,'Local review clusters; no cell-type assignments')
 continuous(axes[1],number(sub.RNA_doublet_score),'Saved RNA doublet score; display capped at 0.5',.5)
 if 'HashSolo_default' in sub:
  group=np.where(sub.HashSolo_default.eq('Doublet'),'HTO-called doublet',np.where(truth(sub.stable_singlet_posterior_ge_0_9_CANDIDATE),'Stable HTO singlet candidate','Other HTO class'))
  categorical(axes[2],group,'HTO comparison groups')
 elif 'author_celltype_label' in sub and label.startswith('GSE290618'):
  categorical(axes[2],np.where(sub.is_step04_candidate,'Author CD8 candidate','Other author-listed cell'),'Step 04 candidate membership')
 else: continuous(axes[2],number(sub.percent_mito),'Mitochondrial RNA %; provisional gate <=20',20)
 continuous(axes[3],np.log1p(number(sub.RNA_counts)),'log1p total raw RNA counts')
 for ax,g in zip(axes[4:8],['Cd8b1','Foxp3','Trac','Trdc']):continuous(ax,expression[:,MARKERS.index(g)],g+' — log1p RNA counts per 10,000')
 js=[MARKERS.index('Trgc'+str(i)) for i in range(1,5)];measured=np.isfinite(raw_markers[:,js]).any(axis=1)
 sums=np.where(measured,np.nansum(raw_markers[:,js],axis=1),np.nan)
 continuous(axes[8],np.log1p(1e4*sums/number(sub.RNA_counts)),'Summed Trgc1–4 — log1p counts per 10,000')
 for ax in axes:ax.set(xticks=[],yticks=[],xlabel='UMAP 1',ylabel='UMAP 2')
 fig.suptitle(label+' | all cells passing the provisional plot gate; no doublet/lineage exclusions',fontsize=12)
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(folder/'UMAP_REVIEW.png',dpi=145);plt.close(fig)

def dotplot(folder,label,means,fractions,counts,present):
 genes=[g for g in MARKERS if g in present];cl=list(means.index)
 if not genes or not cl:return
 xx,yy=np.meshgrid(np.arange(len(genes)),np.arange(len(cl)))
 fig,ax=plt.subplots(figsize=(max(13,len(genes)*.29),max(4,len(cl)*.32+2)))
 vals=means[genes].to_numpy();sizes=fractions[genes].to_numpy()*75
 pts=ax.scatter(xx.ravel(),yy.ravel(),s=sizes.ravel(),c=vals.ravel(),cmap='viridis',vmin=0,vmax=3)
 ax.set_xticks(range(len(genes)),genes,rotation=90,fontsize=7);ax.set_yticks(range(len(cl)),[str(c)+' (n='+str(counts[c])+')' for c in cl],fontsize=8)
 ax.invert_yaxis();ax.set_title(label+' | dot area = fraction detected; color = mean log1p counts/10k (capped at 3)',fontsize=10)
 fig.colorbar(pts,ax=ax,shrink=.5)
 fig.tight_layout();fig.savefig(folder/'MARKER_DOTPLOT_REVIEW.png',dpi=145);plt.close(fig)

def one(gse,lib,helpers):
 label=gse+'__'+lib;folder=out/label;folder.mkdir(exist_ok=True)
 src=step6/(label+'_all_model_source_cells_NOT_APPROVED.tsv.gz')
 signature={'code_SHA256':CODE_SHA,'step04_manifest_SHA256':sha(prior/label/'manifest.json'),'step06_cells_SHA256':sha(src),'step07_manifest_SHA256':sha(step7/'RUN_MANIFEST.json'),'software':VERSIONS}
 sentinel=folder/'MANIFEST.json'
 if sentinel.exists():
  cache=json.loads(sentinel.read_text())
  if cache['signature']!=signature:raise ValueError('Cached analysis inputs/code/software changed; use a new run directory')
  for name,value in cache['outputs_SHA256'].items():
   if sha(folder/name)!=value:raise ValueError('Cached output changed: '+str(folder/name))
  checkpoint=json.loads((prior/label/'manifest.json').read_text())
  for name,value in checkpoint['output_sha256'].items():
   if sha(prior/label/name)!=value:raise ValueError('Cached checkpoint source changed: '+name)
  summaries.append(cache['summary'])
  if (folder/'CLUSTER_SUMMARY.tsv').exists():all_clusters.extend(pd.read_csv(folder/'CLUSTER_SUMMARY.tsv',sep='\t').to_dict('records'))
  say('REUSED completed library:',label);return
 say('LOAD',label)
 x,genes,expected,m=helpers['load_checkpoint'](prior/label);candidate_ids=set(expected.Cell_ID)
 if gse=='GSE290618':x,expected=helpers['broader_author_population'](lib,prior/label,x,genes,expected,m)
 q=pd.read_csv(src,sep='\t',keep_default_na=False)
 if q.Cell_ID.duplicated().any() or q.Cell_ID.tolist()!=expected.Cell_ID.tolist() or q.original_barcode.tolist()!=expected.original_barcode.tolist():raise ValueError('Step 06 full-source cell identities disagree')
 q=helpers['metrics'](x,genes,q);mask=q.model_mask_RNA_min_and_mito_le20.to_numpy(dtype=bool)
 q['is_step04_candidate']=q.Cell_ID.isin(candidate_ids);q['source_GSE']=gse;q['source_library']=lib
 q['UMAP_1']=np.nan;q['UMAP_2']=np.nan;q['cluster_review']='NOT_EMBEDDED_RNA_QC';q['review_status']='NO_FINAL_CELL_SELECTION'
 n=int(mask.sum())
 if n<100:q.loc[mask,'cluster_review']='NOT_EMBEDDED_INSUFFICIENT_CELLS'
 summary={'GSE':gse,'library':lib,'source_cells':len(q),'plot_gate_cells':n,'step04_candidates_in_plot_gate':int((mask&q.is_step04_candidate).sum()),'review_clusters':0,'status':'INSUFFICIENT_CELLS_FOR_EMBEDDING'}
 if n>=100:
  gi={g:i for i,g in enumerate(genes)};present=[g for g in MARKERS if g in gi]
  values=np.full((n,len(MARKERS)),np.nan,dtype=np.float32)
  values[:,[MARKERS.index(g) for g in present]]=x[mask][:,[gi[g] for g in present]].toarray()
  totals=number(q.loc[mask,'RNA_counts']);expression=np.log1p(values*1e4/totals[:,None]);sub=q.loc[mask].reset_index(drop=True)
  say('NORMALIZE/HVG',label,'cells:',n)
  a=ad.AnnData(x[mask].astype(np.float32),obs=pd.DataFrame(index=sub.Cell_ID),var=pd.DataFrame(index=genes))
  sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
  eligible=np.array([not (g.lower().startswith('mt-') or g.startswith(('Rpl','Rps')) or re.match(r'^Tr[abdg][vj]',g)) for g in genes])
  eligible&=np.asarray((a.X>0).sum(axis=0)).ravel()>=3
  if eligible.sum()<50:raise ValueError('Too few genes for an interpretable embedding')
  a=a[:,eligible].copy();sc.pp.highly_variable_genes(a,n_top_genes=min(2000,a.n_vars),flavor='seurat',subset=True)
  pcs=min(30,a.n_obs-2,a.n_vars-2)
  if pcs<2:raise ValueError('Insufficient PCA dimensions')
  say('PCA',label,'HVGs:',a.n_vars,'PCs:',pcs);sc.tl.pca(a,n_comps=pcs,svd_solver='arpack',random_state=SEED)
  say('NEIGHBORS',label);sc.pp.neighbors(a,n_neighbors=min(15,n-1),n_pcs=pcs,use_rep='X_pca',method='umap',transformer='pynndescent',random_state=SEED)
  say('UMAP',label);sc.tl.umap(a,min_dist=.4,random_state=SEED)
  say('LEIDEN',label);sc.tl.leiden(a,resolution=.6,random_state=SEED,key_added='cluster_review',flavor='leidenalg',n_iterations=2,directed=False)
  xy=np.asarray(a.obsm['X_umap']);clusters=a.obs.cluster_review.astype(str).to_numpy();sub['cluster_review']=clusters
  if xy.shape!=(n,2) or not np.isfinite(xy).all():raise ValueError('Invalid UMAP coordinates')
  q.loc[mask,['UMAP_1','UMAP_2']]=xy;q.loc[mask,'cluster_review']=clusters
  np.save(folder/'PCA_COORDINATES.npy',np.asarray(a.obsm['X_pca']));(folder/'EMBEDDING_HVG_SYMBOLS.txt').write_text('\n'.join(a.var_names)+'\n')
  names=sorted(set(clusters),key=lambda v:int(v));means=pd.DataFrame(index=names,columns=MARKERS,dtype=float);fractions=means.copy();counts={};rows=[]
  for c in names:
   take=clusters==c;counts[c]=int(take.sum());sq=sub.loc[take];scores=number(sq.RNA_doublet_score);finite=scores[np.isfinite(scores)]
   row={'GSE':gse,'library':lib,'cluster_review':c,'cells':int(take.sum()),'step04_candidates':int(sq.is_step04_candidate.sum()),'median_RNA_counts':float(np.median(number(sq.RNA_counts))),'median_mito':float(np.median(number(sq.percent_mito))),'RNA_score_median':float(np.median(finite)) if len(finite) else None,'RNA_score_p95':float(np.quantile(finite,.95)) if len(finite) else None}
   for cut in [.1,.15,.2,.25,.3]:row['RNA_score_gt_'+str(cut)]=int((scores>cut).sum())
   for g in present:
    j=MARKERS.index(g);means.loc[c,g]=float(expression[take,j].mean());fractions.loc[c,g]=float((values[take,j]>0).mean())
   row['Foxp3_ge5']=int((values[take,MARKERS.index('Foxp3')]>=5).sum())
   support=(values[take][:,[MARKERS.index(g) for g in ['Il2ra','Ctla4','Ikzf2','Tnfrsf18']]]>=2).sum(axis=1)
   row['Foxp3_ge5_with_2_support_genes_ge2']=int(((values[take,MARKERS.index('Foxp3')]>=5)&(support>=2)).sum())
   row['Cd4_ge2']=int((values[take,MARKERS.index('Cd4')]>=2).sum());row['Cd8b1_ge2']=int((values[take,MARKERS.index('Cd8b1')]>=2).sum())
   if 'HashSolo_default' in sq:
    row['HTO_called_doublets']=int(sq.HashSolo_default.eq('Doublet').sum());row['stable_HTO_singlet_candidates']=int(truth(sq.stable_singlet_posterior_ge_0_9_CANDIDATE).sum())
   rows.append(row)
  cs=pd.DataFrame(rows);cs.to_csv(folder/'CLUSTER_SUMMARY.tsv',sep='\t',index=False);all_clusters.extend(rows)
  means.index.name='cluster_review';fractions.index.name='cluster_review';means.to_csv(folder/'CLUSTER_MARKER_MEAN_LOG_CPT10K.tsv',sep='\t');fractions.to_csv(folder/'CLUSTER_MARKER_DETECTED_FRACTION.tsv',sep='\t')
  if 'HashSolo_default' in sub:pd.crosstab(sub.cluster_review,sub.HashSolo_default).to_csv(folder/'CLUSTER_HTO_COUNTS.tsv',sep='\t')
  if gse=='GSE290618':pd.crosstab(q.loc[mask,'cluster_review'],q.loc[mask,'author_celltype_label']).to_csv(folder/'CLUSTER_AUTHOR_LABEL_COUNTS.tsv',sep='\t')
  say('PLOTS',label);plot_umap(folder,label,q,mask,xy,clusters,expression,values);dotplot(folder,label,means,fractions,counts,present)
  summary.update(review_clusters=len(names),status='EMBEDDED_FOR_REVIEW')
  del a,values,expression
 cols=['Cell_ID','original_barcode','source_GSE','source_library','author_celltype_label','is_step04_candidate','RNA_counts','detected_symbols','percent_mito','model_mask_RNA_min_and_mito_le20','RNA_doublet_score','RNA_doublet_review','HashSolo_default','stable_singlet_posterior_ge_0_9_CANDIDATE','proposed_sample','proposed_sample_ID','UMAP_1','UMAP_2','cluster_review','review_status']
 q[[c for c in cols if c in q]].to_csv(folder/'CELL_EMBEDDINGS_REVIEW.tsv.gz',sep='\t',index=False)
 outputs={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
 dump(sentinel,{'signature':signature,'summary':summary,'parameters':PARAMS,'missing_marker_genes':[g for g in MARKERS if g not in genes],'outputs_SHA256':outputs,'no_final_cell_selection':True})
 summaries.append(summary);say('DONE',json.dumps(summary));del x,q;gc.collect()

def main():
 for source,key,expected in [(prior,'successful_libraries',23),(step6,'completed_libraries',23),(step7,'reviewed_libraries',23)]:
  m=json.loads((source/'RUN_MANIFEST.json').read_text())
  if m['failures'] or m[key]!=expected:raise ValueError('Prior stage incomplete: '+str(source))
 sm=json.loads((step6/'RUN_MANIFEST.json').read_text());sp=step6/'rna_doublet_review.py'
 if sm['step04']!=str(prior) or json.loads((step7/'RUN_MANIFEST.json').read_text())['source_step06']!=str(step6):raise ValueError('Prior-stage paths do not link consistently')
 if sha(sp)!=sm['code_SHA256']:raise ValueError('Step 06 helper source changed')
 old=sys.argv[:]
 try:
  sys.argv=[str(sp),str(raw),str(prior),sm['step05'],str(out)]
  helpers=runpy.run_path(str(sp),run_name='READ_ONLY_STEP06_HELPERS')
 finally:sys.argv=old
 index=pd.read_csv(prior/'QC_SUMMARY.tsv',sep='\t',keep_default_na=False)
 if len(index)!=23 or index[['GSE','library']].duplicated().any():raise ValueError('Invalid library index')
 # Prioritize the datasets that motivated this review.
 records=list(index.itertuples(index=False));records.sort(key=lambda r:(r.library not in PRIORITY,r.GSE,r.library))
 for r in records:
  try:one(r.GSE,r.library,helpers)
  except Exception as e:errors.append({'GSE':r.GSE,'library':r.library,'error':str(e)});traceback.print_exc()
  gc.collect()
 pd.DataFrame(summaries).to_csv(out/'EMBEDDING_SUMMARY.tsv',sep='\t',index=False)
 cs=pd.DataFrame(all_clusters);cs.to_csv(out/'ALL_CLUSTER_SUMMARIES.tsv',sep='\t',index=False)
 if len(cs):cs.loc[(cs.library=='GSM8269393')&(cs.Foxp3_ge5_with_2_support_genes_ge2>0)].to_csv(out/'KPCGP_FOXP3_CLUSTER_REVIEW.tsv',sep='\t',index=False)
 else:pd.DataFrame().to_csv(out/'KPCGP_FOXP3_CLUSTER_REVIEW.tsv',sep='\t',index=False)
 dump(out/'RUN_MANIFEST.json',{'code_SHA256':CODE_SHA,'parameters':PARAMS,'software':VERSIONS,'source_step04':str(prior),'source_step06':str(step6),'source_step07':str(step7),'completed_libraries':len(summaries),'embedded_libraries':sum(r['status']=='EMBEDDED_FOR_REVIEW' for r in summaries),'failures':errors,'no_doublet_models_refit':True,'cells_approved_for_upload':0})
 with zipfile.ZipFile(out/'SHARE_STEP08_PRIORITY_REVIEW.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for r in summaries:
   if r['library'] in PRIORITY:
    label=r['GSE']+'__'+r['library']
    for name in ['UMAP_REVIEW.png','MARKER_DOTPLOT_REVIEW.png','CLUSTER_SUMMARY.tsv','CLUSTER_MARKER_MEAN_LOG_CPT10K.tsv','CLUSTER_MARKER_DETECTED_FRACTION.tsv','CLUSTER_HTO_COUNTS.tsv','CLUSTER_AUTHOR_LABEL_COUNTS.tsv','MANIFEST.json']:
     p=out/label/name
     if p.exists():z.write(p,label+'/'+name)
  for name in ['EMBEDDING_SUMMARY.tsv','KPCGP_FOXP3_CLUSTER_REVIEW.tsv','RUN_MANIFEST.json']:z.write(out/name,name)
 say('STEP 08 completed:',len(summaries),'libraries; data failures:',len(errors),'; no final cell selection.')
 if errors:raise RuntimeError('Some libraries failed. Use RUN_SAVED_STEP08.sh to resume this run; completed outputs are verified and reused.')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
