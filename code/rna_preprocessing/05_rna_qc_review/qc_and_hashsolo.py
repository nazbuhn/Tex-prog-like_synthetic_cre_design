import collections,csv,datetime,gc,gzip,hashlib,importlib.metadata,json,os,pathlib,sys,traceback
import numpy as np,pandas as pd,anndata as ad,scanpy as sc,scanpy.external as sce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits
raw,prior,out=map(pathlib.Path,sys.argv[1:4])
SEED=1729; MODEL_MITO_MAX=20.0
PRIORS={'default':(.01,.80,.19),'more_negative':(.05,.75,.20),'more_doublet':(.01,.69,.30)}
MARKERS=['Trac','Cd3d','Cd3e','Cd3g','Cd8a','Cd8b1','Cd8b','Cd4','Foxp3','Trdc','Zbtb16','Ncr1','Ms4a1','Lyz2']
def say(*v): print(*v,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def writejson(p,v): p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def upper_log_mad(values):
 v=np.asarray(values,dtype=float); v=v[np.isfinite(v)&(v>=0)]
 if not len(v): return None
 z=np.log1p(v); med=np.median(z); mad=1.4826*np.median(np.abs(z-med))
 return float(np.expm1(med+3*mad)) if mad>0 else None
def read_qc(folder):
 m=json.loads((folder/'manifest.json').read_text()); p=folder/'cell_QC.tsv.gz'
 if sha(p)!=m['output_sha256']['cell_QC.tsv.gz']: raise ValueError('QC file changed after Step 04: '+str(p))
 q=pd.read_csv(p,sep='\t',dtype={'Cell_ID':str,'original_barcode':str},keep_default_na=False)
 if len(q)!=m['cells'] or q.Cell_ID.duplicated().any() or q.original_barcode.duplicated().any(): raise ValueError('QC dimensions or IDs disagree')
 total=q.RNA_counts.to_numpy(dtype=np.int64); genes=q.detected_symbols.to_numpy(dtype=np.int64)
 mt=pd.to_numeric(q.percent_mito,errors='coerce').to_numpy(); minimum=(total>500)&(genes>300)
 if not np.array_equal(minimum,q.pass_RBI_RNA_minimum.astype(str).str.lower().eq('true').to_numpy()): raise ValueError('Minimum QC flags disagree')
 if int(minimum.sum())!=m['pass_RBI_RNA_minimum']: raise ValueError('Minimum QC count differs from manifest')
 q['percent_mito']=mt; q['RNA_minimum']=minimum
 return q,m
summary=[]; marker_fractions=[]; marker_means=[]; errors=[]; source_metadata=[]; hto_summary=[]; tag_summary=[]
def library_qc(folder,pdf,overview_ax):
 q,m=read_qc(folder); lib=m['library']; gse=m['source_GSE']; mask=q.RNA_minimum.to_numpy(); mt=q.percent_mito.to_numpy()
 row={'GSE':gse,'library':lib,'input_cells':len(q),'RNA_minimum':int(mask.sum())}
 for cutoff in [5,10,15,20,25,30]: row['RNA_min_and_mito_le_'+str(cutoff)]=int((mask&np.isfinite(mt)&(mt<=cutoff)).sum())
 row['log_mito_3_scaled_MAD_upper']=upper_log_mad(mt[mask])
 summary.append(row); source_metadata.append({'library':lib,'step04_manifest_sha256':sha(folder/'manifest.json'),'cell_QC_sha256':sha(folder/'cell_QC.tsv.gz')})
 model=mask&np.isfinite(mt)&(mt<=MODEL_MITO_MAX)
 fractions={'GSE':gse,'library':lib,'cells_in_diagnostic_mask':int(model.sum())}; means=fractions.copy()
 for marker in MARKERS:
  col='RNA_'+marker
  if col in q and model.any():
   values=q.loc[model,col].to_numpy(dtype=float); totals=q.loc[model,'RNA_counts'].to_numpy(dtype=float)
   fractions[marker]=float((values>0).mean()); means[marker]=float(np.log1p(1e4*values/totals).mean())
  else: fractions[marker]=None; means[marker]=None
 marker_fractions.append(fractions); marker_means.append(means)
 fig,axes=plt.subplots(1,3,figsize=(13.5,3.8))
 x=np.log10(1+q.RNA_counts.to_numpy()); y=np.log10(1+q.detected_symbols.to_numpy())
 c=np.clip(mt,0,20); pts=axes[0].scatter(x,y,c=c,s=2,alpha=.5,vmin=0,vmax=20,rasterized=True)
 axes[0].set(xlabel='log10(1 + RNA counts)',ylabel='log10(1 + detected symbols)'); fig.colorbar(pts,ax=axes[0],label='Mito % (color capped at 20)')
 axes[1].hist(x[~mask],bins=45,color='lightgrey',label='Below RNA minimum'); axes[1].hist(x[mask],bins=45,histtype='step',color='black',label='Above RNA minimum')
 axes[1].set(xlabel='log10(1 + RNA counts)',ylabel='Cells'); axes[1].legend(fontsize=7)
 axes[2].hist(mt[mask&np.isfinite(mt)],bins=np.linspace(0,100,101),color='steelblue')
 for cutoff,color in [(10,'orange'),(15,'purple'),(20,'red')]: axes[2].axvline(cutoff,color=color,lw=.9,label=str(cutoff)+'%')
 axes[2].set(xlabel='Mito % among RNA-minimum cells',ylabel='Cells'); axes[2].legend(fontsize=7)
 fig.suptitle(gse+' / '+lib+' — thresholds shown for review; no final cell selection',fontsize=10); fig.tight_layout()
 fig.savefig(out/(gse+'__'+lib+'_QC.png'),dpi=140); pdf.savefig(fig); plt.close(fig)
 overview_ax.scatter(x,y,c=c,s=.6,alpha=.4,vmin=0,vmax=20,rasterized=True)
 overview_ax.set_title(gse+' / '+lib,fontsize=7); overview_ax.set_xlabel('log10(1+RNA)',fontsize=6); overview_ax.set_ylabel('log10(1+genes)',fontsize=6); overview_ax.tick_params(labelsize=6)
 say('QC',lib,'RNA minimum:',row['RNA_minimum'],'mito <=10:',row['RNA_min_and_mito_le_10'],'<=15:',row['RNA_min_and_mito_le_15'],'<=20:',row['RNA_min_and_mito_le_20'])
def classify_counts(counts,index,tags,priors):
 obs=pd.DataFrame(counts,index=index,columns=tags,dtype=np.int64); a=ad.AnnData(obs=obs)
 sce.pp.hashsolo(a,tags,priors=priors,inplace=True)
 result=a.obs[['Classification','negative_hypothesis_probability','singlet_hypothesis_probability','doublet_hypothesis_probability']].copy()
 probabilities=result.iloc[:,1:].to_numpy(dtype=float)
 valid=np.isfinite(probabilities).all(axis=1)&np.isclose(probabilities.sum(axis=1),1,atol=1e-6)
 result.loc[~valid,'Classification']='UNRESOLVED_MODEL'
 return result,valid
def demux(well,gsm,htogsm,pdf):
 folder=prior/('GSE278342__'+gsm); q,m=read_qc(folder)
 source=raw/'GSE278342'/(htogsm+'_'+well+'_HTO_Tag_Counts.csv.gz')
 barcodes=q.original_barcode.tolist()
 if not all(b.endswith('-1') for b in barcodes): raise ValueError('Unexpected RNA barcode suffix')
 bare={b[:-2]:i for i,b in enumerate(barcodes)}
 if len(bare)!=len(barcodes): raise ValueError('Barcode suffix removal is not one-to-one')
 counts=np.zeros((len(q),len(tags)),dtype=np.int64); seen=np.zeros(len(q),dtype=bool); total_disagreements=0
 with gzip.open(source,'rt') as h:
  reader=csv.DictReader(h)
  if set(reader.fieldnames)!=set(['cell_barcode','total']+tags): raise ValueError('Unexpected HTO columns')
  for r in reader:
   if r['cell_barcode'] not in bare: continue
   i=bare[r['cell_barcode']]
   if seen[i]: raise ValueError('Duplicate HTO match for an RNA barcode')
   values=np.array([int(r[t]) for t in tags],dtype=np.int64)
   if (values<0).any(): raise ValueError('Negative HTO count')
   total_disagreements+=int(int(r['total'])!=int(values.sum()))
   counts[i]=values; seen[i]=True
 if not seen.all() or total_disagreements: raise ValueError('HTO barcode coverage or total-vs-tags validation failed')
 mask=q.RNA_minimum.to_numpy()&np.isfinite(q.percent_mito.to_numpy())&(q.percent_mito.to_numpy()<=MODEL_MITO_MAX)
 nonzero=counts.sum(axis=1)>0; fit=mask&nonzero
 if fit.sum()<100: raise ValueError('Fewer than 100 eligible nonzero HTO cells; no diagnostic calls made')
 calls=pd.DataFrame(index=q.Cell_ID); calls.index.name='Cell_ID'
 calls['original_barcode']=barcodes; calls['well']=well; calls['RNA_minimum']=q.RNA_minimum.to_numpy(); calls['model_mask_RNA_min_and_mito_le20']=mask
 for j,t in enumerate(tags): calls[t]=counts[:,j]
 ordered_counts=np.sort(counts,axis=1)
 calls['HTO_total']=counts.sum(axis=1); calls['largest_HTO_count']=ordered_counts[:,-1]; calls['second_largest_HTO_count']=ordered_counts[:,-2]
 calls['largest_to_second_plus1_ratio']=(ordered_counts[:,-1]+1)/(ordered_counts[:,-2]+1)
 all_valid=[]
 for name,priors in PRIORS.items():
  result,valid=classify_counts(counts[fit],q.Cell_ID[fit].tolist(),tags,priors)
  labels=np.full(len(q),'NOT_MODELED_RNA_QC',dtype=object); labels[mask&~nonzero]='Negative_all_zero'; labels[fit]=result.Classification.astype(str).to_numpy()
  calls['HashSolo_'+name]=labels
  all_valid.append(np.array([str(v)!='UNRESOLVED_MODEL' for v in labels]))
  if name=='default':
   for column in ['negative_hypothesis_probability','singlet_hypothesis_probability','doublet_hypothesis_probability']:
    values=np.full(len(q),np.nan); values[fit]=result[column].to_numpy(dtype=float)
    values[mask&~nonzero]=1.0 if column.startswith('negative') else 0.0
    calls[column]=values
 classes=calls.HashSolo_default.to_numpy(); stable=(classes==calls.HashSolo_more_negative.to_numpy())&(classes==calls.HashSolo_more_doublet.to_numpy())
 if not np.isin(classes,tags+['Negative','Negative_all_zero','Doublet','NOT_MODELED_RNA_QC','UNRESOLVED_MODEL']).all(): raise ValueError('Unexpected HashSolo class label')
 stable &= np.logical_and.reduce(all_valid)&mask
 high=stable&np.isin(classes,tags)&(calls.singlet_hypothesis_probability.to_numpy()>=.9)
 calls['same_class_across_3_priors']=stable; calls['stable_singlet_posterior_ge_0_9_CANDIDATE']=high
 calls['proposed_sample']=[sample_map.get(c,'') for c in classes]
 calls['proposed_sample_ID']=[gsm+'__'+c+'__'+sample_map[c] if c in sample_map else '' for c in classes]
 calls['eligibility_status']='NOT_APPROVED'
 calls.to_csv(out/(well+'_HashSolo_calls_NOT_APPROVED.tsv.gz'),sep='\t')
 status=collections.Counter(classes)
 item={'well':well,'GSM':gsm,'RNA_cells':len(q),'RNA_minimum_and_mito_le20':int(mask.sum()),'default_singlet':int(np.isin(classes,tags).sum()),'default_doublet':int((classes=='Doublet').sum()),'default_negative':int(np.isin(classes,['Negative','Negative_all_zero']).sum()),'unresolved_model':int((classes=='UNRESOLVED_MODEL').sum()),'stable_high_conf_singlet_CANDIDATE':int(high.sum()),'stable_singlet_HTO_total_lt20_DIAGNOSTIC':int((high&(counts.sum(axis=1)<20)).sum()),'prior_unstable_or_model_unresolved':int((mask&~stable).sum())}
 hto_summary.append(item)
 for t in tags: tag_summary.append({'well':well,'tag':t,'sample':sample_map[t],'default_singlet':int((classes==t).sum()),'stable_high_conf_singlet_CANDIDATE':int((high&(classes==t)).sum())})
 writejson(out/(well+'_HashSolo_manifest.json'),{'method':'Scanpy 1.10.3 HashSolo; diagnostic alternative to author manual demultiplexing','model_mask':'RNA counts >500 AND detected symbols >300 AND mitochondrial percent <=20; modeling gate only','priors':PRIORS,'zero_HTO_rule':'Eligible all-zero HTO profiles assigned Negative_all_zero without fitting','seed':SEED,'posterior_candidate_threshold':.9,'mapping_rule':'HTO barcode sequence maps uniquely to RNA sequence-1; all RNA cells covered','source_HTO_SHA256':sha(source),'source_QC_SHA256':sha(folder/'cell_QC.tsv.gz'),'sample_map_SHA256':sha(out/'references/GSE278342_HTO_sample_map.tsv'),'classification_counts':dict(status),'all_cells_NOT_APPROVED':True})
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 top=np.sort(counts[mask],axis=1)[:,-2:]; code=np.where(np.isin(classes[mask],tags),0,np.where(classes[mask]=='Doublet',1,2))
 for n,label,color in [(0,'Singlet','steelblue'),(1,'Doublet','red'),(2,'Negative/other','grey')]:
  take=code==n; axes[0].scatter(np.log10(1+top[take,1]),np.log10(1+top[take,0]),s=2,alpha=.45,c=color,label=label,rasterized=True)
 axes[0].set(xlabel='log10(1 + largest HTO count)',ylabel='log10(1 + second-largest HTO count)'); axes[0].legend(fontsize=7)
 default=[int((classes==t).sum()) for t in tags]; hc=[int((high&(classes==t)).sum()) for t in tags]
 axes[1].bar(tags,default,color='lightgrey',label='Default singlet'); axes[1].bar(tags,hc,color='steelblue',label='Stable posterior >=0.9'); axes[1].tick_params(axis='x',labelrotation=60); axes[1].legend(fontsize=7); axes[1].set(ylabel='Cells')
 indices=np.flatnonzero(mask)
 if len(indices)>5000: indices=np.random.default_rng(SEED).choice(indices,5000,replace=False)
 class_order={tag:i for i,tag in enumerate(tags)}
 indices=np.array(sorted(indices,key=lambda i:(class_order.get(str(classes[i]),len(tags)),str(classes[i]),-int(counts[i].max()))),dtype=int)
 image=axes[2].imshow(np.log1p(counts[indices]).T,aspect='auto',interpolation='nearest',cmap='viridis')
 axes[2].set(yticks=np.arange(len(tags)),yticklabels=tags,xlabel='Diagnostic cells sorted by default class'); fig.colorbar(image,ax=axes[2],label='log(1 + raw HTO counts)')
 fig.suptitle(well+' / '+gsm+' — HashSolo diagnostic calls, no cell approval',fontsize=11); fig.tight_layout(); fig.savefig(out/(well+'_HashSolo_diagnostics.png'),dpi=170); pdf.savefig(fig); plt.close(fig)
 say('HTO',json.dumps(item))
def main():
 global tags,sample_map
 run=json.loads((prior/'RUN_MANIFEST.json').read_text())
 if run.get('successful_libraries')!=23 or run.get('failures'): raise ValueError('Expected 23 successful Step 04 libraries')
 index=pd.read_csv(prior/'QC_SUMMARY.tsv',sep='\t',dtype={'GSE':str,'library':str})
 if len(index)!=23 or index[['GSE','library']].duplicated().any(): raise ValueError('Invalid Step 04 summary index')
 ref=pd.read_csv(out/'references/GSE278342_HTO_sample_map.tsv',sep='\t',dtype=str)
 tags=ref.tag.tolist(); sample_map=dict(zip(ref.tag,ref['sample']))
 if len(tags)!=9 or len(set(tags))!=9: raise ValueError('Invalid GEO HTO sample map')
 say('STEP 05: QC review and diagnostic HTO demultiplexing. Raw RNA matrices stay unchanged.')
 overview,axs=plt.subplots(6,4,figsize=(16,22)); axs=axs.ravel()
 with PdfPages(out/'QC_and_HTO_diagnostics.pdf') as pdf:
  for i,r in index.iterrows():
   try: library_qc(prior/(r.GSE+'__'+r.library),pdf,axs[i])
   except Exception as e: errors.append({'library':r.library,'error':str(e)}); say('QC FAILED',r.library,str(e)); traceback.print_exc()
  for ax in axs[len(index):]: ax.axis('off')
  overview.suptitle('RNA quality overview — color is mitochondrial percent, capped at 20',fontsize=13); overview.tight_layout(rect=(0,0,1,.98)); overview.savefig(out/'RNA_QC_overview_23_libraries.png',dpi=150); plt.close(overview)
  for well,gsm,hto in [('W8','GSM8545120','GSM8545123'),('W9','GSM8545124','GSM8545127')]:
   try: demux(well,gsm,hto,pdf)
   except Exception as e: errors.append({'library':well,'error':str(e)}); say('HTO FAILED',well,str(e)); traceback.print_exc()
   gc.collect()
 pd.DataFrame(summary).to_csv(out/'QC_THRESHOLD_COMPARISON.tsv',sep='\t',index=False)
 pd.DataFrame(marker_fractions).to_csv(out/'MARKER_detected_fraction_DIAGNOSTIC.tsv',sep='\t',index=False)
 pd.DataFrame(marker_means).to_csv(out/'MARKER_mean_log_CPT10K_DIAGNOSTIC.tsv',sep='\t',index=False)
 pd.DataFrame(hto_summary).to_csv(out/'HTO_SUMMARY_NOT_APPROVED.tsv',sep='\t',index=False)
 pd.DataFrame(tag_summary).to_csv(out/'HTO_sample_counts_NOT_APPROVED.tsv',sep='\t',index=False)
 writejson(out/'failures.json',errors)
 writejson(out/'RUN_MANIFEST.json',{'source_step04':str(prior),'code_SHA256':sha(pathlib.Path(__file__)),'software':{p:importlib.metadata.version(p) for p in ['scanpy','anndata','numpy','scipy','pandas','matplotlib','scikit-learn','numba']},'source_checkpoints':source_metadata,'hashsolo_priors':PRIORS,'hashsolo_model_mito_max':MODEL_MITO_MAX,'plot_and_marker_masks_not_final_selection':True,'RNA_counts_unchanged':True,'cells_approved_for_upload':0,'failures':errors})
 readme=['', '### Step 05 diagnostic results', '', 'Report directory: `'+str(out)+'`', '',
  'RNA candidates reviewed: '+str(sum(r['input_cells'] for r in summary))+'. Above the strict RNA minimum: '+str(sum(r['RNA_minimum'] for r in summary))+'.', '',
  'Mitochondrial thresholds are compared for review. The <=20% mask is provisional for HashSolo fitting and marker summaries, not a final whitelist.', '',
  '| Well | Modeled cells | Default singlets | Default doublets | Default negatives | Stable singlet candidates with posterior >=0.9 |',
  '| --- | ---: | ---: | ---: | ---: | ---: |']
 for r in hto_summary: readme.append('| '+r['well']+' | '+' | '.join(str(r[k]) for k in ['RNA_minimum_and_mito_le20','default_singlet','default_doublet','default_negative','stable_high_conf_singlet_CANDIDATE'])+' |')
 readme+=['', 'GEO HTO identities are preserved in the report references. Raw RNA matrices were not modified; no cells are approved for upload. RNA doublet analysis and final QC remain pending.', '']
 (out/'RESULTS_FOR_README.md').write_text('\n'.join(readme))
 say('\nSTEP 05 SUMMARY: libraries reviewed:',len(summary),'HTO wells completed:',len(hto_summary),'failures:',len(errors))
 say('Modeling gate <=20% mitochondrial RNA is provisional; QC plots and threshold tables are for review. No final mitochondrial cutoff chosen; RNA doublets remain untested.')
 if errors: raise RuntimeError('Some diagnostics failed; share output. Other reports are preserved.')
if __name__=='__main__':
 np.random.seed(SEED)
 with threadpool_limits(limits=1): main()
