import gc,gzip,hashlib,importlib.metadata as md,json,pathlib,runpy,sys,traceback,zipfile
import numpy as np,pandas as pd,scanpy as sc,doubletdetection as dd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

raw,prior,step5,step6,step8,out=map(pathlib.Path,sys.argv[1:7])
PARAMS=dict(n_iters=25,n_components=30,n_top_var_genes=10000,boost_rate=.25,replace=False,clustering_algorithm='louvain',clustering_kwargs={'resolution':4,'directed':False},standard_scaling=True,pseudocount=.1,random_state=1729,n_jobs=1,verbose=False)
P_THRESH=1e-7
summaries=[];cluster_rows=[];hto_rows=[];failures=[]
sc.settings.n_jobs=1
def say(*v):print(*v,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n');temp.replace(p)
def number(s):return pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
def truth(s):
 v=s.astype(str).str.lower()
 if not v.isin(['true','false']).all():raise ValueError('Invalid boolean data')
 return v.eq('true').to_numpy()
def table(p):return pd.read_csv(p,sep='\t',keep_default_na=False,dtype={'Cell_ID':str,'original_barcode':str,'cluster_review':str})
def gztable(q,p):
 with open(p,'wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(q.to_csv(index=False,sep='\t',lineterminator='\n').encode())
def quantile(values,p):
 v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
 return float(np.quantile(v,p)) if len(v) else None
def software():return {p:md.version(p) for p in ['doubletdetection','PhenoGraph','louvain','scanpy','anndata','numpy','scipy','pandas','scikit-learn','igraph','leidenalg','numba','llvmlite','umap-learn','pynndescent']}
VERSIONS=software();CODE_SHA=sha(pathlib.Path(__file__))

def plot(folder,label,q,mask):
 s=q.loc[mask].copy();xy=s[['UMAP_1','UMAP_2']].apply(pd.to_numeric,errors='coerce').to_numpy()
 good=np.isfinite(xy).all(axis=1)
 if not good.any():return
 s=s.loc[good];xy=xy[good]
 fig,axes=plt.subplots(2,3,figsize=(15,9));axes=axes.ravel()
 def continuous(ax,v,title,vmax=None):
  v=np.asarray(v,dtype=float);ok=np.isfinite(v);ax.set_title(title,fontsize=9)
  if ok.any():
   pts=ax.scatter(xy[ok,0],xy[ok,1],c=v[ok],s=2,cmap='viridis',vmin=0,vmax=vmax,rasterized=True);fig.colorbar(pts,ax=ax,shrink=.7)
  else:ax.text(.5,.5,'Unavailable',transform=ax.transAxes,ha='center')
 cl=s.cluster_review.astype(str).to_numpy()
 for i,c in enumerate(sorted(set(cl))):
  take=cl==c;axes[0].scatter(xy[take,0],xy[take,1],s=2,c=[plt.get_cmap('tab20')(i%20)],rasterized=True);axes[0].text(*np.median(xy[take],axis=0),c,fontsize=8)
 axes[0].set_title('Saved Step 08 review clusters',fontsize=9)
 continuous(axes[1],number(s.RNA_doublet_score),'Saved Scrublet score; not rethresholded',.5)
 continuous(axes[2],number(s.DD_vote_p1e7),'DD iteration voting fraction; not a probability',1)
 call=number(s.DD_call_vote_ge_0_9)
 for value,color,name in [(0,'steelblue','Not called by DD'),(1,'orangered','DD positive'),(np.nan,'gray','Unresolved')]:
  take=~np.isfinite(call) if np.isnan(value) else call==value
  if take.any():axes[3].scatter(xy[take,0],xy[take,1],s=2,c=color,label=name,rasterized=True)
 axes[3].set_title('DD p<=1e-7; voting >=0.9; diagnostic only',fontsize=9);axes[3].legend(fontsize=6,markerscale=3)
 continuous(axes[4],np.log1p(number(s.RNA_counts)),'log1p raw RNA depth')
 continuous(axes[5],np.log1p(1e4*number(s.RNA_Mki67)/number(s.RNA_counts)),'Mki67 log1p counts/10k; cycling context')
 for ax in axes:ax.set(xticks=[],yticks=[],xlabel='Saved UMAP 1',ylabel='Saved UMAP 2')
 fig.suptitle(label+' | SECOND CALLER REVIEW; no final cell selection',fontsize=12);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(folder/'DD_COMPARISON_REVIEW.png',dpi=140);plt.close(fig)

def append_cached(folder,m):
 summaries.append(m['summary']);cluster_rows.extend(m['cluster_rows']);hto_rows.extend(m['HTO_rows'])

def one(gse,lib,helpers,review):
 label=gse+'__'+lib;folder=out/label;folder.mkdir(exist_ok=True)
 cp=prior/label;ep=step8/label;mp=ep/'MANIFEST.json';src=step6/(label+'_all_model_source_cells_NOT_APPROVED.tsv.gz')
 policy=review['reviewed_Step08_libraries'][label]
 if sha(mp)!=policy['step08_manifest_SHA256']:raise ValueError('Step 08 clustering does not match reviewed bundle')
 em=json.loads(mp.read_text());cm=json.loads((cp/'manifest.json').read_text())
 if sha(cp/'manifest.json')!=em['signature']['step04_manifest_SHA256']:raise ValueError('Step 04 manifest differs from reviewed checkpoint')
 for name,digest in cm['output_sha256'].items():
  if sha(cp/name)!=digest:raise ValueError('Step 04 output changed: '+name)
 if sha(src)!=em['signature']['step06_cells_SHA256']:raise ValueError('Saved Scrublet source changed')
 embedding=ep/'CELL_EMBEDDINGS_REVIEW.tsv.gz'
 if sha(embedding)!=em['outputs_SHA256']['CELL_EMBEDDINGS_REVIEW.tsv.gz']:raise ValueError('Saved embeddings changed')
 sig={'code_SHA256':CODE_SHA,'review_policy_SHA256':sha(out/'REVIEW_DECISIONS.json'),'step04_manifest_SHA256':sha(cp/'manifest.json'),'step06_table_SHA256':sha(src),'step08_manifest_SHA256':sha(mp),'parameters':PARAMS,'software':VERSIONS}
 completed=folder/'MANIFEST.json'
 if completed.exists():
  m=json.loads(completed.read_text())
  if m['signature']!=sig:raise ValueError('Cached analysis signature differs; use a new run')
  for name,digest in m['outputs_SHA256'].items():
   if sha(folder/name)!=digest:raise ValueError('Cached output changed: '+name)
  append_cached(folder,m);say('REUSED verified library:',label);return
 say('LOAD',label)
 x,genes,expected,cm=helpers['load_checkpoint'](cp);candidate_ids=set(expected.Cell_ID)
 if gse=='GSE290618':x,expected=helpers['broader_author_population'](lib,cp,x,genes,expected,cm)
 q=table(src);e=table(embedding)
 for t in [q,e]:
  if t.Cell_ID.duplicated().any() or t.Cell_ID.tolist()!=expected.Cell_ID.tolist() or t.original_barcode.tolist()!=expected.original_barcode.tolist():raise ValueError('Cell identities/order differ')
 q=helpers['metrics'](x,genes,q);mask=truth(q.model_mask_RNA_min_and_mito_le20);n=int(mask.sum())
 if not np.array_equal(mask,truth(e.model_mask_RNA_min_and_mito_le20)):raise ValueError('Model gate differs from embedding gate')
 for col in ['RNA_counts','detected_symbols','percent_mito','RNA_doublet_score']:
  if not np.allclose(number(q[col]),number(e[col]),rtol=1e-12,atol=1e-12,equal_nan=True):raise ValueError('Saved QC/score differs: '+col)
 for col in ['UMAP_1','UMAP_2','cluster_review']:q[col]=e[col].to_numpy()
 q['is_step04_candidate']=q.Cell_ID.isin(candidate_ids)
 if not np.array_equal(q.is_step04_candidate.to_numpy(),truth(e.is_step04_candidate)):raise ValueError('Candidate membership differs')
 gi={g:i for i,g in enumerate(genes)}
 for gene in ['Mki67','Top2a']:
  q['RNA_'+gene]=np.asarray(x[:,gi[gene]].toarray()).ravel() if gene in gi else np.nan
 for col in ['DD_vote_p1e7','DD_score_negative_mean_log_p','DD_call_vote_ge_0_5','DD_call_vote_ge_0_9','DD_vote_p1e6','DD_call_p1e6_vote_ge_0_5']:q[col]=np.nan
 q['DD_status']='NOT_MODELED_RNA_QC';q.loc[mask,'DD_status']='UNRESOLVED'
 q['review_hold_B_like_cluster']=q.cluster_review.astype(str).isin(policy['review_cluster_holds'])
 q['condition_identity_status']='NO_NEW_DESCRIPTION_FILENAME_CONFLICT_IDENTIFIED'
 if lib in review['GSE267584_condition_conflicts']:q['condition_identity_status']='GEO_DESCRIPTION_FILENAME_CONFLICT_UNRESOLVED'
 status='INSUFFICIENT_CELLS_FOR_SECOND_CALLER';effective=None
 logp=np.empty((0,n));all_scores=np.empty((0,n))
 if n>=100:
  model=x[mask].astype(np.float32)
  expressed=np.asarray(model.sum(axis=0)).ravel()>0;model=model[:,expressed].tocsr()
  effective=dict(PARAMS);effective['n_components']=min(PARAMS['n_components'],model.shape[0]-1,model.shape[1]-1);effective['n_top_var_genes']=min(PARAMS['n_top_var_genes'],model.shape[1])
  if effective['n_components']<2:raise ValueError('Insufficient expressed features for model')
  say('FIT DOUBLETDETECTION',label,'cells:',n,'iterations:',effective['n_iters'],'one thread; progress counts iterations')
  clf=dd.BoostClassifier(**effective);clf.fit(model)
  y09=np.asarray(clf.predict(p_thresh=P_THRESH,voter_thresh=.9),dtype=float).copy();vote=np.asarray(clf.voting_average_,dtype=float).copy()
  score=np.asarray(clf.doublet_score(),dtype=float).copy();y05=np.asarray(clf.predict(p_thresh=P_THRESH,voter_thresh=.5),dtype=float).copy()
  y6=np.asarray(clf.predict(p_thresh=1e-6,voter_thresh=.5),dtype=float).copy();vote6=np.asarray(clf.voting_average_,dtype=float).copy()
  logp=np.asarray(clf.all_log_p_values_);all_scores=np.asarray(clf.all_scores_)
  if logp.shape!=(effective['n_iters'],n) or all_scores.shape!=logp.shape:raise ValueError('Iteration-score dimensions differ')
  for v in [y09,y05,y6]:
   if v.shape!=(n,) or not np.isin(v[np.isfinite(v)],[0,1]).all():raise ValueError('Invalid model labels')
  if ((vote[np.isfinite(vote)]<0)|(vote[np.isfinite(vote)]>1)).any():raise ValueError('Invalid voting fraction')
  for col,values in [('DD_vote_p1e7',vote),('DD_score_negative_mean_log_p',score),('DD_call_vote_ge_0_5',y05),('DD_call_vote_ge_0_9',y09),('DD_vote_p1e6',vote6),('DD_call_p1e6_vote_ge_0_5',y6)]:q.loc[mask,col]=values
  q.loc[mask,'DD_status']=np.where(np.isfinite(y09),'SCORED_NEEDS_REVIEW','UNRESOLVED')
  status='SCORED_NEEDS_REVIEW';del model,clf
 q['final_cell_selection_status']='NO_FINAL_CELL_SELECTION'
 np.save(folder/'DD_ITERATION_LOG_PVALUES.npy',logp);np.save(folder/'DD_ITERATION_SYNTHETIC_FRACTIONS.npy',all_scores)
 gztable(pd.DataFrame({'Cell_ID':q.loc[mask,'Cell_ID']}),folder/'MODEL_CELL_ORDER.tsv.gz')
 keepcols=['Cell_ID','original_barcode','author_celltype_label','RNA_counts','detected_symbols','percent_mito','RNA_doublet_score','is_step04_candidate','model_mask_RNA_min_and_mito_le20','UMAP_1','UMAP_2','cluster_review','HashSolo_default','stable_singlet_posterior_ge_0_9_CANDIDATE','proposed_sample','proposed_sample_ID','RNA_Mki67','RNA_Top2a','DD_vote_p1e7','DD_score_negative_mean_log_p','DD_call_vote_ge_0_5','DD_call_vote_ge_0_9','DD_vote_p1e6','DD_call_p1e6_vote_ge_0_5','DD_status','review_hold_B_like_cluster','condition_identity_status','final_cell_selection_status']
 gztable(q[[c for c in keepcols if c in q]],folder/'ALL_SOURCE_CELL_SECOND_CALLER_REVIEW.tsv.gz')
 candidates=q.is_step04_candidate.to_numpy();called=number(q.DD_call_vote_ge_0_9)==1;called05=number(q.DD_call_vote_ge_0_5)==1;resolved=np.isfinite(number(q.DD_call_vote_ge_0_9))
 row={'GSE':gse,'library':lib,'source_cells':len(q),'modeled_cells':n,'modeled_step04_candidates':int((mask&candidates).sum()),'DD_positive_vote_ge_0_9':int((mask&called).sum()),'DD_positive_vote_ge_0_5':int((mask&called05).sum()),'DD_positive_p1e6_vote_ge_0_5':int((number(q.DD_call_p1e6_vote_ge_0_5)==1).sum()),'DD_unresolved_modeled':int((mask&~resolved).sum()),'candidate_DD_positive_vote_ge_0_9':int((mask&candidates&called).sum()),'review_B_cluster_cells':int(q.review_hold_B_like_cluster.sum()),'condition_conflict':lib in review['GSE267584_condition_conflicts'],'status':status}
 clusters=[];hto=[]
 for cl,s in q.loc[mask].groupby('cluster_review',sort=False):
  clusters.append({'GSE':gse,'library':lib,'cluster_review':str(cl),'cells':len(s),'step04_candidates':int(s.is_step04_candidate.sum()),'DD_positive_vote_ge_0_9':int((number(s.DD_call_vote_ge_0_9)==1).sum()),'DD_positive_vote_ge_0_5':int((number(s.DD_call_vote_ge_0_5)==1).sum()),'DD_vote_median':quantile(number(s.DD_vote_p1e7),.5),'Scrublet_score_median':quantile(number(s.RNA_doublet_score),.5),'median_RNA_counts':quantile(number(s.RNA_counts),.5),'Mki67_measured':bool(np.isfinite(number(s.RNA_Mki67)).any()),'Mki67_detected_cells':int((number(s.RNA_Mki67)>0).sum()) if np.isfinite(number(s.RNA_Mki67)).any() else None,'review_B_cluster_cells':int(s.review_hold_B_like_cluster.sum())})
 if 'HashSolo_default' in q:
  a=mask&q.HashSolo_default.eq('Doublet').to_numpy();b=mask&truth(q.stable_singlet_posterior_ge_0_9_CANDIDATE)
  for name,group in [('HTO_called_doublets',a),('stable_HTO_singlet_candidates',b)]:
   hto.append({'library':lib,'comparison_group':name,'cells':int(group.sum()),'DD_positive_vote_ge_0_9':int((called&group).sum()),'DD_positive_vote_ge_0_5':int((called05&group).sum()),'DD_unresolved':int((group&~resolved).sum()),'DD_vote_median':quantile(number(q.DD_vote_p1e7)[group],.5),'DD_vote_p95':quantile(number(q.DD_vote_p1e7)[group],.95)})
  vv=number(q.DD_vote_p1e7);use=(a|b)&np.isfinite(vv)
  row['DD_vote_AUROC_relative_to_HTO_groups']=float(roc_auc_score(a[use].astype(int),vv[use])) if a[use].any() and b[use].any() else None
 pd.DataFrame(clusters).to_csv(folder/'CLUSTER_SECOND_CALLER_SUMMARY.tsv',sep='\t',index=False)
 plot(folder,label,q,mask)
 output_names=['DD_ITERATION_LOG_PVALUES.npy','DD_ITERATION_SYNTHETIC_FRACTIONS.npy','MODEL_CELL_ORDER.tsv.gz','ALL_SOURCE_CELL_SECOND_CALLER_REVIEW.tsv.gz','CLUSTER_SECOND_CALLER_SUMMARY.tsv']
 if (folder/'DD_COMPARISON_REVIEW.png').exists():output_names.append('DD_COMPARISON_REVIEW.png')
 m={'signature':sig,'effective_parameters':effective,'summary':row,'cluster_rows':clusters,'HTO_rows':hto,'outputs_SHA256':{n:sha(folder/n) for n in output_names},'no_source_counts_or_exports_modified':True,'no_final_cell_selection':True,'physical_capture_mapping':'GSE278342 W8/W9 identified; remaining supplied libraries modeled separately, physical capture boundaries not independently established'}
 dump(completed,m);append_cached(folder,m);say('DONE',label,json.dumps(row));del x,q,e,logp,all_scores;gc.collect()

def main():
 review=json.loads((out/'REVIEW_DECISIONS.json').read_text())
 m6=json.loads((step6/'RUN_MANIFEST.json').read_text());m8=json.loads((step8/'RUN_MANIFEST.json').read_text())
 if m6['failures'] or m8['failures'] or m8['completed_libraries']!=23:raise ValueError('Earlier run incomplete')
 if m8['source_step04']!=str(prior) or m8['source_step06']!=str(step6) or m6['step05']!=str(step5):raise ValueError('Earlier source paths disagree')
 helper=step6/'rna_doublet_review.py'
 if sha(helper)!=m6['code_SHA256']:raise ValueError('Step 06 helper source changed')
 saved=sys.argv[:]
 try:
  sys.argv=[str(helper),str(raw),str(prior),str(step5),str(out)];helpers=runpy.run_path(str(helper),run_name='step06_read_only_helpers')
 finally:sys.argv=saved
 idx=pd.read_csv(step8/'EMBEDDING_SUMMARY.tsv',sep='\t',keep_default_na=False)
 labels=set(idx.GSE+'__'+idx.library)
 if len(idx)!=23 or labels!=set(review['reviewed_Step08_libraries']):raise ValueError('Reviewed library list differs')
 priority={'GSM8545120':0,'GSM8545124':1,'GSM8269399':2,'GSM8269393':3}
 idx=idx.assign(priority=idx.library.map(priority).fillna(10)).sort_values(['priority','GSE','library'])
 for r in idx.itertuples(index=False):
  try:one(r.GSE,r.library,helpers,review)
  except Exception as exc:failures.append({'GSE':r.GSE,'library':r.library,'error':str(exc)});traceback.print_exc()
  gc.collect()
 pd.DataFrame(summaries).to_csv(out/'SECOND_CALLER_SUMMARY.tsv',sep='\t',index=False)
 pd.DataFrame(cluster_rows).to_csv(out/'CLUSTER_SECOND_CALLER_SUMMARY.tsv',sep='\t',index=False)
 pd.DataFrame(hto_rows).to_csv(out/'HTO_SECOND_CALLER_COMPARISON.tsv',sep='\t',index=False)
 dump(out/'RUN_MANIFEST.json',{'code_SHA256':CODE_SHA,'software':VERSIONS,'parameters':PARAMS,'p_threshold':P_THRESH,'vote_thresholds_compared':[.5,.9],'sensitivity_p_threshold':1e-6,'source_step04':str(prior),'source_step06':str(step6),'source_step08':str(step8),'review_policy_SHA256':sha(out/'REVIEW_DECISIONS.json'),'completed_libraries':len(summaries),'scored_libraries':sum(r['status']=='SCORED_NEEDS_REVIEW' for r in summaries),'failures':failures,'new_exports_created':0,'existing_Step09_exports_modified':False,'no_cells_removed':True,'no_RDS_objects_loaded':True,'no_server_submission':True})
 with zipfile.ZipFile(out/'SHARE_STEP10_SECOND_CALLER_REVIEW.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['SECOND_CALLER_SUMMARY.tsv','CLUSTER_SECOND_CALLER_SUMMARY.tsv','HTO_SECOND_CALLER_COMPARISON.tsv','REVIEW_DECISIONS.json','RUN_MANIFEST.json','README_STEP_10.md']:
   if (out/name).exists():z.write(out/name,name)
  for r in summaries:
   label=r['GSE']+'__'+r['library']
   for name in ['DD_COMPARISON_REVIEW.png','CLUSTER_SECOND_CALLER_SUMMARY.tsv','MANIFEST.json']:
    p=out/label/name
    if p.exists():z.write(p,label+'/'+name)
 say('STEP 10 completed:',len(summaries),'libraries; scored:',sum(r['status']=='SCORED_NEEDS_REVIEW' for r in summaries),'failures:',len(failures),'no final selection or new exports')
 if failures:raise RuntimeError('Some libraries failed; share errors. The saved runner verifies and reuses completed libraries.')

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
