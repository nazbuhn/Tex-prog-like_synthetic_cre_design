import csv,gc,gzip,hashlib,importlib.metadata,json,pathlib,sys,traceback,warnings
import numpy as np,pandas as pd,h5py,anndata as ad,scanpy as sc
from scipy import sparse
from skimage.filters import threshold_minimum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits
raw,prior,review,out=map(pathlib.Path,sys.argv[1:5])
SEED=1729; MITO=20.; RATE=.05
PANELS={'T':['Trac','Cd3d','Cd3e','Cd3g'], 'CD8':['Cd8a','Cd8b1','Cd8b'], 'gdT':['Trdc','Trgc1','Trgc2','Trgc3','Trgc4'], 'B':['Ms4a1','Cd79a','Cd79b'], 'myeloid':['Lyz2','Lst1','Csf1r'], 'NK_receptors':['Ncr1','Klrb1c'], 'other':['Cd4','Foxp3','Zbtb16','Klrd1','Nkg7','Tyrobp','Itgax','Pecam1','Col1a1','Epcam'], 'CD8_states':['Pdcd1','Tox','Havcr2','Lag3','Tcf7','Cxcr6','Entpd1','Gzmb','Prf1','Ccr7','Il7r']}
MARKERS=list(dict.fromkeys(g for v in PANELS.values() for g in v))
summary=[]; agreement=[]; failures=[]; marker_summary=[]; provenance=[]
def say(*s): print(*s,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def dump(p,v): p.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n')
def text(v): return v.decode() if isinstance(v,bytes) else str(v)
def validate(x):
 x=x.tocsr(); x.check_format(full_check=True)
 for k in range(0,len(x.data),1000000):
  v=x.data[k:k+1000000]
  if not np.isfinite(v).all() or (v<0).any() or not np.equal(v,np.floor(v)).all(): raise ValueError('Invalid RNA counts; no rounding permitted')
 x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
 return x

def metrics(x,genes,q):
 total=np.asarray(x.sum(axis=1)).ravel(); detected=np.diff(x.indptr)
 mt=np.array([g.lower().startswith('mt-') for g in genes]); mt_count=np.asarray(x[:,mt].sum(axis=1)).ravel()
 pct=np.divide(100.*mt_count,total,out=np.full(len(total),np.nan),where=total>0)
 if not mt.any(): raise ValueError('No mitochondrial genes')
 if 'RNA_counts' in q:
  if not np.array_equal(total,q.RNA_counts.to_numpy()) or not np.array_equal(detected,q.detected_symbols.to_numpy()): raise ValueError('Checkpoint counts disagree with QC')
  if not np.allclose(pct,pd.to_numeric(q.percent_mito,errors='coerce'),equal_nan=True): raise ValueError('Mito percentages disagree')
 q=q.copy(); q['RNA_counts']=total; q['detected_symbols']=detected; q['percent_mito']=pct
 q['model_mask_RNA_min_and_mito_le20']=(total>500)&(detected>300)&np.isfinite(pct)&(pct<=MITO)
 return q

def load_checkpoint(folder):
 m=json.loads((folder/'manifest.json').read_text())
 for name,value in m['output_sha256'].items():
  if sha(folder/name)!=value: raise ValueError('Changed Step 04 file: '+str(folder/name))
 q=pd.read_csv(folder/'cell_QC.tsv.gz',sep='\t',keep_default_na=False,dtype={'Cell_ID':str,'original_barcode':str})
 with gzip.open(folder/'genes.tsv.gz','rt') as f: genes=[v.rstrip('\r\n') for v in f]
 if len(set(genes))!=len(genes) or q.Cell_ID.duplicated().any() or q.original_barcode.duplicated().any(): raise ValueError('Duplicate gene/cell IDs')
 x=validate(sparse.load_npz(folder/'RNA_counts_cells_by_genes.npz'))
 if x.shape!=(len(q),len(genes)) or x.shape!=(m['cells'],m['canonical_symbols']): raise ValueError('Checkpoint dimensions disagree')
 return x,genes,metrics(x,genes,q),m

def broader_author_population(batch,folder,x_selected,genes,q_selected,m):
 p=raw/'GSE290618'/('GSE290618_'+batch+'_filtered_feature_bc_matrix.h5'); meta=raw/'GSE290618/GSE290618_metadata.tsv.gz'
 for f in [p,meta]:
  if sha(f)!=m['source_sha256'].get(str(f)): raise ValueError('GSE290618 original source changed: '+str(f))
 df=pd.read_csv(meta,sep='\t',keep_default_na=False); df=df.loc[df.batch==batch].copy(); authors=df.iloc[:,0].astype(str)
 if authors.duplicated().any() or not authors.str.endswith('-'+batch).all(): raise ValueError('Author IDs not unique or suffix unexpected')
 bc=[v[:-len('-'+batch)] for v in authors]
 fmap=pd.read_csv(folder/'source_feature_map.tsv.gz',sep='\t',keep_default_na=False)
 with h5py.File(p,'r') as h:
  z=h['matrix']; ft=z['features']; barcodes=[text(v) for v in z['barcodes'][...]]
  if len(set(barcodes))!=len(barcodes): raise ValueError('Duplicate H5 barcodes')
  pos={b:i for i,b in enumerate(barcodes)}
  if any(b not in pos for b in bc): raise ValueError('Author barcode absent from H5')
  mask=np.array([text(v)=='Gene Expression' for v in ft['feature_type'][...]])
  ids=[text(v) for v in ft['id'][...][mask]]; symbols=[text(v) for v in ft['name'][...][mask]]
  if ids!=fmap.source_feature_ID.tolist() or symbols!=fmap.source_symbol.tolist(): raise ValueError('H5 features differ from saved feature map')
  full=sparse.csc_matrix((z['data'][...],z['indices'][...],z['indptr'][...]),shape=tuple(z['shape'][...]))
  full.check_format(full_check=True)
  for k in range(0,len(full.data),1000000):
   v=full.data[k:k+1000000]
   if not np.isfinite(v).all() or (v<0).any() or not np.equal(v,np.floor(v)).all(): raise ValueError('Invalid source counts')
  x=full[:,[pos[b] for b in bc]][mask,:].T.tocsr(); del full
 columns=fmap.canonical_column_0based.to_numpy(dtype=int)
 if not np.array_equal(np.array(genes)[columns],fmap.canonical_symbol.to_numpy()): raise ValueError('Canonical feature map mismatch')
 projection=sparse.csr_matrix((np.ones(len(ids),dtype=np.int64),(np.arange(len(ids)),columns)),shape=(len(ids),len(genes)))
 totals=np.asarray(x.sum(axis=1)).ravel(); x=validate(x.astype(np.int64)@projection)
 if not np.array_equal(totals,np.asarray(x.sum(axis=1)).ravel()): raise ValueError('Gene projection changed RNA totals')
 by_barcode={b:i for i,b in enumerate(bc)}
 rows=[by_barcode[b] for b in q_selected.original_barcode]
 if (x[rows]!=x_selected).nnz: raise ValueError('Reconstructed full author population differs from CD8 checkpoint')
 q=pd.DataFrame({'Cell_ID':['GSE290618__'+batch+'__'+b for b in bc],'original_barcode':bc,'author_cell_ID':authors.to_numpy(),'author_celltype_label':df.celltype_label.to_numpy()})
 return x,metrics(x,genes,q)

def run_model(x,q,genes):
 q=q.copy(); mask=q.model_mask_RNA_min_and_mito_le20.to_numpy(dtype=bool); n=int(mask.sum())
 q['RNA_doublet_score']=np.nan; q['RNA_doublet_review']='NOT_MODELED_RNA_QC'; q.loc[mask,'RNA_doublet_review']='UNRESOLVED_MODEL'
 info={'model_cells':n,'model_status':'UNRESOLVED','automatic_threshold':None,'expected_doublet_rate_assumed':RATE,'expected_rate_verified_from_loading':False,'encapsulation_unit_verified':False,'seed':SEED,'model_gate_mito_percent':MITO,'temporary_API_threshold':.5,'temporary_API_calls_discarded':True,'parameters':{'sim_doublet_ratio':2.0,'stdev_doublet_rate':.02,'n_prin_comps':30,'n_neighbors':int(round(.5*np.sqrt(n))),'use_approx_neighbors':True,'mean_center':True,'normalize_variance':True,'log_transform':False,'metric':'euclidean'}}
 sim=np.array([])
 if n<100:
  info['reason']='Fewer than 100 cells in the provisional model gate'; return q,info,sim
 try:
  a=ad.AnnData(x[mask].copy(),obs=pd.DataFrame(index=q.loc[mask,'Cell_ID']),var=pd.DataFrame(index=genes))
  # A fixed temporary threshold lets Scanpy return scores even when auto thresholding fails.
  # Its predicted_doublet calls are discarded. Only the simulated-score auto threshold is used below.
  with np.errstate(divide='ignore',invalid='ignore'):
   sc.pp.scrublet(a,expected_doublet_rate=RATE,stdev_doublet_rate=.02,sim_doublet_ratio=2.,n_prin_comps=30,use_approx_neighbors=True,threshold=.5,random_state=SEED,verbose=False)
  scores=a.obs.doublet_score.to_numpy(dtype=float); sim=np.asarray(a.uns['scrublet']['doublet_scores_sim'],dtype=float)
  if len(scores)!=n or not np.isfinite(scores).all() or not np.isfinite(sim).all(): raise ValueError('Missing or nonfinite doublet scores')
  q.loc[mask,'RNA_doublet_score']=scores; info['model_status']='SCORED_AUTO_THRESHOLD_UNRESOLVED'
  try:
   threshold=float(threshold_minimum(sim))
   if not np.isfinite(threshold) or not sim.min()<threshold<sim.max(): raise ValueError('Invalid automatic threshold')
   q.loc[mask,'RNA_doublet_review']=np.where(scores>threshold,'ABOVE_AUTO_THRESHOLD','BELOW_AUTO_THRESHOLD')
   info.update(model_status='SCORED_THRESHOLD_NEEDS_REVIEW',automatic_threshold=threshold,simulated_above_threshold_fraction=float((sim>threshold).mean()))
  except Exception as e: info['threshold_error']=str(e)
  info['simulated_cells']=len(sim)
 except Exception as e:
  info['model_error']=str(e); traceback.print_exc()
 return q,info,sim

def add_markers(x,genes,q):
 gi={g:i for i,g in enumerate(genes)}; q=q.copy()
 for g in MARKERS:
  q['RNA_'+g]=np.asarray(x[:,gi[g]].toarray()).ravel() if g in gi else np.nan
 for name,markers in PANELS.items():
  cols=['RNA_'+g for g in markers if g in gi]
  q['diagnostic_'+name+'_genes_detected']=(q[cols]>0).sum(axis=1) if cols else 0
 q['diagnostic_T_and_CD8_detected']=(q.diagnostic_T_genes_detected>=2)&(q.diagnostic_CD8_genes_detected>=1)
 q['diagnostic_Trdc_detected']=q.RNA_Trdc>0
 q['diagnostic_Foxp3_detected']=q.RNA_Foxp3>0
 q['diagnostic_B_2genes'] = q.diagnostic_B_genes_detected>=2
 q['diagnostic_myeloid_2genes']=q.diagnostic_myeloid_genes_detected>=2
 q['doublet_status']=q['RNA_doublet_review']
 q['eligibility_status']='NOT_APPROVED'
 return q,[g for g in MARKERS if g not in gi]

def plot(q,sim,info,label,pdf):
 fig,axes=plt.subplots(1,3,figsize=(14,4)); mask=q.model_mask_RNA_min_and_mito_le20.to_numpy(dtype=bool)
 scores=q.RNA_doublet_score.to_numpy(dtype=float); good=mask&np.isfinite(scores); threshold=info['automatic_threshold']
 bins=np.linspace(0,1,81)
 for values,name,color in [(scores[good],'Observed','steelblue'),(sim,'Simulated doublets','orange')]:
  if len(values): axes[0].hist(values,bins=bins,density=True,alpha=.55,label=name,color=color)
 if threshold is not None: axes[0].axvline(threshold,c='red',label='Automatic threshold')
 axes[0].set(xlabel='RNA doublet score',ylabel='Density'); axes[0].legend(fontsize=7)
 scatter=axes[1].scatter(np.log10(1+q.loc[good,'RNA_counts']),np.log10(1+q.loc[good,'detected_symbols']),c=scores[good],vmin=0,vmax=1,s=2,rasterized=True)
 axes[1].set(xlabel='log10(1+RNA counts)',ylabel='log10(1+detected symbols)'); fig.colorbar(scatter,ax=axes[1],label='RNA doublet score')
 groups=list(PANELS)
 values=[q.loc[mask,'diagnostic_'+g+'_genes_detected'].gt(0).mean() if mask.any() else 0 for g in groups]
 axes[2].barh(groups,values); axes[2].set(xlim=(0,1),xlabel='Fraction with >=1 panel gene detected')
 fig.suptitle(label+' | '+info['model_status']+' | diagnostic only',fontsize=10); fig.tight_layout()
 pdf.savefig(fig); fig.savefig(out/(label+'_RNA_doublet_diagnostics.png'),dpi=140); plt.close(fig)

def one(gse,lib,pdf):
 label=gse+'__'+lib; folder=prior/label
 say('START',label,'(RNA doublet scores; no cells removed)')
 x,genes,q,m=load_checkpoint(folder); selected_ids=q.Cell_ID.tolist()
 source_scope='All Step 04 cells before lineage or HTO selection'
 if gse=='GSE290618':
  x,q=broader_author_population(lib,folder,x,genes,q,m)
  source_scope='All author-listed cells in this tissue file before conventional-CD8 label selection'
 q,info,sim=run_model(x,q,genes); q,missing=add_markers(x,genes,q)
 info.update(GSE=gse,library=lib,source_scope=source_scope,source_cells=len(q),missing_marker_genes=missing,checkpoint_manifest_sha256=sha(folder/'manifest.json'))
 if gse=='GSE278342':
  well='W8' if lib=='GSM8545120' else 'W9'; hp=review/(well+'_HashSolo_calls_NOT_APPROVED.tsv.gz')
  hm=json.loads((review/(well+'_HashSolo_manifest.json')).read_text())
  if hm['source_QC_SHA256']!=sha(folder/'cell_QC.tsv.gz'): raise ValueError('HTO calls use another QC checkpoint')
  h=pd.read_csv(hp,sep='\t',keep_default_na=False)
  if h.Cell_ID.duplicated().any() or set(h.Cell_ID)!=set(q.Cell_ID): raise ValueError('HTO cells do not match RNA')
  h=h.set_index('Cell_ID').loc[q.Cell_ID]
  if h.original_barcode.tolist()!=q.original_barcode.tolist(): raise ValueError('HTO barcode identity/order mismatch')
  for col in ['HashSolo_default','stable_singlet_posterior_ge_0_9_CANDIDATE','proposed_sample','proposed_sample_ID']:
   q[col]=h[col].to_numpy()
  stable=q.stable_singlet_posterior_ge_0_9_CANDIDATE.astype(str).str.lower().eq('true')
  for hto,rna in q[['HashSolo_default','RNA_doublet_review']].drop_duplicates().itertuples(index=False,name=None):
   agreement.append({'well':well,'HTO_default_class':hto,'RNA_review':rna,'cells':int(((q.HashSolo_default==hto)&(q.RNA_doublet_review==rna)).sum())})
  info['HTO_stable_candidates_below_RNA_auto_threshold']=int((stable&q.RNA_doublet_review.eq('BELOW_AUTO_THRESHOLD')).sum())
  info['HTO_calls_SHA256']=sha(hp)
 q['source_GSE']=gse; q['source_library']=lib
 q.to_csv(out/(label+'_all_model_source_cells_NOT_APPROVED.tsv.gz'),sep='\t',index=False)
 by_id=q.set_index('Cell_ID',drop=False)
 if not set(selected_ids)<=set(by_id.index) or by_id.index.duplicated().any(): raise ValueError('Candidate-to-source cell mapping failed')
 selected=by_id.loc[selected_ids].reset_index(drop=True)
 selected.to_csv(out/(label+'_candidate_cells_NOT_APPROVED.tsv.gz'),sep='\t',index=False)
 row={'GSE':gse,'library':lib,'source_cells':len(q),'model_cells':info['model_cells'],'candidate_cells':len(selected),'candidate_modeled':int(selected.model_mask_RNA_min_and_mito_le20.sum()),'above_auto_threshold':int(selected.RNA_doublet_review.eq('ABOVE_AUTO_THRESHOLD').sum()),'below_auto_threshold':int(selected.RNA_doublet_review.eq('BELOW_AUTO_THRESHOLD').sum()),'unresolved_modeled':int((selected.model_mask_RNA_min_and_mito_le20&selected.RNA_doublet_review.eq('UNRESOLVED_MODEL')).sum()),'automatic_threshold':info['automatic_threshold'],'model_status':info['model_status']}
 summary.append(row)
 mask=selected.model_mask_RNA_min_and_mito_le20.to_numpy(dtype=bool)
 mr={'GSE':gse,'library':lib,'modeled_candidates':int(mask.sum()),'missing_marker_genes':','.join(missing)}
 for flag in ['diagnostic_T_and_CD8_detected','diagnostic_Trdc_detected','diagnostic_Foxp3_detected','diagnostic_B_2genes','diagnostic_myeloid_2genes']:
  mr[flag]=int((mask&selected[flag].to_numpy(dtype=bool)).sum())
 marker_summary.append(mr)
 plot(q,sim,info,label,pdf); dump(out/(label+'_manifest.json'),info); provenance.append(info)
 say('DONE',json.dumps(row,default=str)); del x,q,selected; gc.collect()

def main():
 for source,expected in [(prior,23),(review,None)]:
  manifest=json.loads((source/'RUN_MANIFEST.json').read_text())
  if manifest['failures']: raise ValueError('Prior run has failures')
  if expected and manifest['successful_libraries']!=expected: raise ValueError('Expected 23 RNA libraries')
 index=pd.read_csv(prior/'QC_SUMMARY.tsv',sep='\t',keep_default_na=False)
 if len(index)!=23 or index[['GSE','library']].duplicated().any(): raise ValueError('Invalid library index')
 with PdfPages(out/'RNA_doublet_and_marker_diagnostics.pdf') as pdf:
  for r in index.itertuples(index=False):
   try: one(r.GSE,r.library,pdf)
   except Exception as e: failures.append({'GSE':r.GSE,'library':r.library,'error':str(e)}); traceback.print_exc()
   gc.collect()
 pd.DataFrame(summary).to_csv(out/'RNA_DOUBLET_SUMMARY_NOT_APPROVED.tsv',sep='\t',index=False)
 pd.DataFrame(marker_summary).to_csv(out/'MARKER_FLAGS_DIAGNOSTIC.tsv',sep='\t',index=False)
 pd.DataFrame(agreement).to_csv(out/'HTO_RNA_AGREEMENT_DIAGNOSTIC.tsv',sep='\t',index=False)
 dump(out/'RUN_MANIFEST.json',{'code_SHA256':sha(pathlib.Path(__file__)),'step04':str(prior),'step05':str(review),'model_gate':'RNA counts >500, detected symbols >300, mito <=20%; provisional','expected_doublet_rate_assumed':RATE,'automatic_threshold_method':'skimage.filters.threshold_minimum on simulated scores; Scanpy temporary threshold 0.5 calls discarded','no_final_singlet_or_lineage_assignments':True,'software':{p:importlib.metadata.version(p) for p in ['scanpy','anndata','numpy','scipy','pandas','scikit-image']},'completed_libraries':len(summary),'unresolved_threshold_or_model_libraries':sum(r['model_status']!='SCORED_THRESHOLD_NEEDS_REVIEW' for r in summary),'provenance':provenance,'failures':failures,'cells_approved_for_upload':0})
 say('STEP 06 completed libraries:',len(summary),'data failures:',len(failures),'models returning scores:',sum(r['model_status'].startswith('SCORED') for r in summary),'unresolved thresholds/models:',sum(r['model_status']!='SCORED_THRESHOLD_NEEDS_REVIEW' for r in summary))
 if failures: raise RuntimeError('Data checks failed for some libraries; completed reports retained')
 if not any(r['model_status'].startswith('SCORED') for r in summary): raise RuntimeError('No RNA model returned scores; inspect model errors before continuing')
if __name__=='__main__':
 with threadpool_limits(limits=1): main()
