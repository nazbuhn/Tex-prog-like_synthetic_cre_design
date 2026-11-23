import csv,datetime,gzip,hashlib,io,json,pathlib,re,shutil,sys,traceback,zipfile
import numpy as np,pandas as pd,scipy
from scipy import sparse,io as sio
from threadpoolctl import threadpool_limits
raw,prior,step5,step6,step8,out=map(pathlib.Path,sys.argv[1:7])
REQUIRED=set('Trav14-1 Trgv2 Trac Cd3e Foxp3 Izumo1r Cd40lg Dapl1 Cd5 Trat1 Cd3d Zfp683 Themis Esm1 Cd3g Il10 Sdcbp2 Olfr524 Ly6c1 Sox13 Syt13 Gzmk Trdv1 Trdv2-1 Trdv2-2 Trdv3 Trdv4 Trdd1 Trdd2 Trdj1 Trdj2 Trdc Trdv5 Trgv7 Trgv4 Trgv6 Trgv5 Trgj1 Trgc1 Trgv3 Trgj3 Trgc3 Trgc2 Trgj2 Trgv1 Trgj4 Trgc4'.split())
TAG_MAP={'H3':'Naive','H4':'Arm_Spl','H5':'C13_Spl','H6':'Arm_Liver','H7':'C13_Liver','H8':'Arm_IEL','H9':'C13_IEL','H10':'Arm_SG','H11':'C13_SG'}
POLICY={'GSM8545120':{'well':'W8','doublet_enriched_clusters':['5','11'],'myeloid_associated_clusters':['13'],'stable_expected':14400,'core_before_score_hold_expected':14218,'step08_manifest_SHA256':'5091387538082f0ecef923c18300fac71173132e6832e4d1a4986bb2a2548f63'},'GSM8545124':{'well':'W9','doublet_enriched_clusters':['7'],'myeloid_associated_clusters':['10'],'stable_expected':14248,'core_before_score_hold_expected':14121,'step08_manifest_SHA256':'a65a19f8b6c5afcb8198e72d5969892c1fc220c1120aadeef7d601f97d1eae79'}}
SCORE_REVIEW_HOLD=.30
PORTAL='https://rstats.immgen.org/immgenT_integration_analysis/'
exports=[];failures=[];inputs=[]
def say(*v):print(*v,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n');t.replace(p)
def truth(s):
 v=s.astype(str).str.lower()
 if not v.isin(['true','false']).all():raise ValueError('Invalid boolean data')
 return v.eq('true').to_numpy()
def numbers(s):return pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
def table(p):return pd.read_csv(p,sep='\t',keep_default_na=False,dtype={'Cell_ID':str,'original_barcode':str,'cluster_review':str})
def align(q,ids):
 if q.Cell_ID.duplicated().any() or set(q.Cell_ID)!=set(ids):raise ValueError('Source cell identities disagree')
 return q.set_index('Cell_ID',drop=False).loc[ids].reset_index(drop=True)
def gzwrite(p,text=None,matrix=None):
 with open(p,'wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=6) as z:
   if matrix is not None:sio.mmwrite(z,matrix,field='integer',symmetry='general',comment='RNA raw counts; genes by cells; full supplied RNA gene set')
   else:z.write(text.encode('utf-8'))
def gztable(df,p,sep=','):
 gzwrite(p,df.to_csv(index=False,sep=sep,lineterminator='\n'))

def export_well(gsm):
 policy=POLICY[gsm];well=policy['well'];label='GSE278342__'+gsm;folder=prior/label;review=step8/label
 say('CHECK INPUTS',well)
 m=json.loads((folder/'manifest.json').read_text())
 for name,value in m['output_sha256'].items():
  if sha(folder/name)!=value:raise ValueError('Step 04 source changed: '+name)
 mp=review/'MANIFEST.json'
 if sha(mp)!=policy['step08_manifest_SHA256']:raise ValueError('Step 08 run/cluster identity differs from reviewed bundle')
 em=json.loads(mp.read_text())
 if em['signature']['step04_manifest_SHA256']!=sha(folder/'manifest.json'):raise ValueError('Step 04 checkpoint is not the reviewed Step 08 input')
 for name in ['CELL_EMBEDDINGS_REVIEW.tsv.gz','CLUSTER_SUMMARY.tsv']:
  if sha(review/name)!=em['outputs_SHA256'][name]:raise ValueError('Changed Step 08 review file: '+name)
 q=table(folder/'cell_QC.tsv.gz');ids=q.Cell_ID.tolist()
 if q.Cell_ID.duplicated().any() or q.original_barcode.duplicated().any():raise ValueError('Duplicate source IDs')
 q8=align(table(review/'CELL_EMBEDDINGS_REVIEW.tsv.gz'),ids)
 p6=step6/(label+'_all_model_source_cells_NOT_APPROVED.tsv.gz')
 if sha(p6)!=em['signature']['step06_cells_SHA256']:raise ValueError('Step 06 source table changed')
 q6=align(table(p6),ids);hp=step5/(well+'_HashSolo_calls_NOT_APPROVED.tsv.gz')
 sm=json.loads((step6/(label+'_manifest.json')).read_text())
 if sha(hp)!=sm['HTO_calls_SHA256']:raise ValueError('Step 05 HTO calls changed')
 h=align(table(hp),ids);hm=json.loads((step5/(well+'_HashSolo_manifest.json')).read_text())
 if hm['source_QC_SHA256']!=sha(folder/'cell_QC.tsv.gz'):raise ValueError('HTO source QC mismatch')
 ref=step5/'references/GSE278342_HTO_sample_map.tsv'
 if sha(ref)!=hm['sample_map_SHA256']:raise ValueError('HTO identity reference changed')
 refdf=pd.read_csv(ref,sep='\t',keep_default_na=False)
 if dict(zip(refdf.tag,refdf['sample']))!=TAG_MAP:raise ValueError('Unexpected HTO identities')
 for other in [q6,q8,h]:
  if other.original_barcode.tolist()!=q.original_barcode.tolist():raise ValueError('Original barcode mismatch')
 for col in ['RNA_counts','detected_symbols','percent_mito']:
  for other in [q6,q8]:
   if not np.allclose(numbers(q[col]),numbers(other[col]),rtol=1e-12,atol=1e-12,equal_nan=True):raise ValueError('QC values disagree: '+col)
 totals=numbers(q.RNA_counts);detected=numbers(q.detected_symbols);mito=numbers(q.percent_mito)
 minimum=(totals>500)&(detected>300);gate=minimum&np.isfinite(mito)&(mito<=20)
 for other in [q8,h]:
  if not np.array_equal(gate,truth(other.model_mask_RNA_min_and_mito_le20)):raise ValueError('RNA/mitochondrial gate mismatch')
 score=numbers(q6.RNA_doublet_score)
 if not np.allclose(score,numbers(q8.RNA_doublet_score),rtol=1e-12,atol=1e-12,equal_nan=True):raise ValueError('RNA scores disagree')
 tags=h.HashSolo_default.astype(str).to_numpy();same=(tags==h.HashSolo_more_negative.to_numpy())&(tags==h.HashSolo_more_doublet.to_numpy())
 probabilities=h[['negative_hypothesis_probability','singlet_hypothesis_probability','doublet_hypothesis_probability']].apply(pd.to_numeric,errors='coerce').to_numpy()
 valid=np.isfinite(probabilities).all(axis=1)&(probabilities>=0).all(axis=1)&(probabilities<=1).all(axis=1)&np.isclose(probabilities.sum(axis=1),1,atol=1e-6)
 stable=gate&same&np.isin(tags,list(TAG_MAP))&valid&(probabilities[:,1]>=.9)
 for other in [h,q6,q8]:
  if not np.array_equal(stable,truth(other.stable_singlet_posterior_ge_0_9_CANDIDATE)):raise ValueError('Stable HTO calls differ')
 if int(stable.sum())!=policy['stable_expected']:raise ValueError('Stable HTO cell count differs from reviewed result')
 clusters=q8.cluster_review.astype(str).to_numpy();d=np.isin(clusters,policy['doublet_enriched_clusters']);my=np.isin(clusters,policy['myeloid_associated_clusters'])
 core=gate&stable&~d&~my
 if int(core.sum())!=policy['core_before_score_hold_expected']:raise ValueError('Reviewed cluster-hold counts disagree')
 unresolved=~np.isfinite(score);score_hold=score>SCORE_REVIEW_HOLD;keep=core&~unresolved&~score_hold
 if not 0<int(keep.sum())<=25000:raise ValueError('Upload cell count outside 1..25000')
 reasons=[]
 for i in range(len(q)):
  flags=[]
  for condition,text in [(not minimum[i],'RNA_MINIMUM_FAIL'),(not np.isfinite(mito[i]) or mito[i]>20,'MITO_QC_HOLD'),(not stable[i],'NOT_STABLE_HTO_SINGLET'),(d[i],'RNA_HTO_DOUBLET_ENRICHED_CLUSTER_HOLD'),(my[i],'MYELOID_ASSOCIATED_CLUSTER_HOLD'),(unresolved[i],'RNA_SCORE_UNRESOLVED'),(score_hold[i],'RNA_SCORE_GT_0_30_REVIEW_HOLD')]:
   if condition:flags.append(text)
  reasons.append(';'.join(flags) if flags else 'INCLUDED_IN_CORE_UPLOAD')
 decision=pd.DataFrame({'original_Cell_ID':ids,'original_barcode':q.original_barcode,'cluster_review':clusters,'RNA_counts':totals,'detected_symbols':detected,'percent_mito':mito,'HTO_default':tags,'stable_HTO_singlet':stable,'RNA_doublet_score':score,'include_core_upload':keep,'decision_reasons':reasons})
 gztable(decision,out/(well+'_ALL_CELL_DECISIONS.tsv.gz'),sep='\t')
 say('LOAD RNA COUNTS',well)
 x=sparse.load_npz(folder/'RNA_counts_cells_by_genes.npz').tocsr();x.check_format(full_check=True)
 for start in range(0,len(x.data),1000000):
  values=x.data[start:start+1000000]
  if not np.isfinite(values).all() or (values<0).any() or not np.equal(values,np.floor(values)).all():raise ValueError('Invalid raw RNA counts; no rounding permitted')
 x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
 with gzip.open(folder/'genes.tsv.gz','rt') as f:genes=[s.rstrip('\r\n') for s in f]
 if x.shape!=(len(q),len(genes)) or len(set(genes))!=len(genes):raise ValueError('RNA matrix/gene dimensions or uniqueness disagree')
 if any(not g or re.search(r'\s',g) or re.match(r'^ENS[A-Z]*G\d',g) or g.isdigit() for g in genes):raise ValueError('Invalid gene symbols')
 if REQUIRED-set(genes):raise ValueError('Missing required gene rows: '+','.join(sorted(REQUIRED-set(genes))))
 if not np.array_equal(np.asarray(x.sum(axis=1)).ravel(),totals) or not np.array_equal(np.diff(x.indptr),detected):raise ValueError('RNA checkpoint differs from cell QC')
 selected=np.flatnonzero(keep);upload_ids=['GSE278342__'+gsm+'__'+tags[i]+'__'+TAG_MAP[tags[i]]+'__'+str(q.original_barcode.iloc[i]) for i in selected]
 if len(set(upload_ids))!=len(upload_ids) or any(re.search(r'\s',v) or re.match(r'^IGT\d*\.',v) for v in upload_ids):raise ValueError('Invalid upload cell IDs')
 batch='GSE278342__'+gsm+'__'+well
 dest=out/('UPLOAD_'+well+'_CD8_CORE');dest.mkdir()
 matrix=x[selected].T.tocoo()
 say('WRITE',well,'genes:',len(genes),'cells:',len(selected),'RNA entries:',matrix.nnz)
 gzwrite(dest/'matrix.mtx.gz',matrix=matrix);shutil.copy2(folder/'genes.tsv.gz',dest/'genes.tsv.gz');gzwrite(dest/'barcodes.tsv.gz','\n'.join(upload_ids)+'\n')
 cb=pd.DataFrame({'Cell_ID':upload_ids,'batch':[batch]*len(upload_ids)});cb.to_csv(dest/'cell_batch.csv',index=False,lineterminator='\n')
 meta=pd.DataFrame({'Cell_ID':upload_ids,'original_Cell_ID':[ids[i] for i in selected],'source_GSE':'GSE278342','source_GSM':gsm,'encapsulation_run':batch,'HTO_tag':tags[selected],'HTO_sample':[TAG_MAP[t] for t in tags[selected]],'HTO_sample_ID':[gsm+'__'+t+'__'+TAG_MAP[t] for t in tags[selected]],'original_barcode':q.original_barcode.iloc[selected].to_numpy(),'RNA_counts':totals[selected].astype(np.int64),'detected_symbols':detected[selected].astype(np.int64),'percent_mito':mito[selected],'RNA_doublet_score':score[selected],'cluster_review':clusters[selected],'HTO_singlet_posterior':probabilities[selected,1]})
 gztable(meta,out/(well+'_CELL_METADATA_AND_ID_MAP.csv.gz'))
 sample_rows=[]
 for tag,sample in TAG_MAP.items():
  t=tags==tag
  sample_rows.append({'well':well,'source_GSM':gsm,'HTO_tag':tag,'HTO_sample':sample,'stable_HTO_singlets':int((stable&t).sum()),'held_by_reviewed_clusters':int((stable&t&(d|my)).sum()),'additional_score_or_unresolved_holds':int((core&t&(score_hold|unresolved)).sum()),'exported_cells':int((keep&t).sum())})
 pd.DataFrame(sample_rows).to_csv(out/(well+'_SAMPLE_COUNTS.tsv'),sep='\t',index=False)
 say('VALIDATE ROUND TRIP',well)
 with gzip.open(dest/'matrix.mtx.gz','rb') as f:back=sio.mmread(f).tocsc()
 back.sum_duplicates();back.eliminate_zeros();back.sort_indices()
 if back.shape!=matrix.shape or (back!=matrix.tocsc()).nnz:raise ValueError('Saved matrix differs from selected original counts')
 if not np.isfinite(back.data).all() or (back.data<0).any() or not np.equal(back.data,np.floor(back.data)).all():raise ValueError('Saved matrix has invalid counts')
 if not np.array_equal(np.asarray(back.sum(axis=0)).ravel(),totals[selected]) or not (np.diff(back.indptr)>300).all() or not (np.asarray(back.sum(axis=0)).ravel()>500).all():raise ValueError('Saved matrix violates per-cell QC or count conservation')
 with gzip.open(dest/'genes.tsv.gz','rt') as f:saved_genes=f.read().splitlines()
 with gzip.open(dest/'barcodes.tsv.gz','rt') as f:saved_ids=f.read().splitlines()
 saved_batch=pd.read_csv(dest/'cell_batch.csv',keep_default_na=False)
 if saved_genes!=genes or saved_ids!=upload_ids or saved_batch.columns.tolist()!=['Cell_ID','batch'] or saved_batch.Cell_ID.tolist()!=upload_ids or saved_batch.batch.tolist()!=[batch]*len(upload_ids):raise ValueError('Saved genes/barcodes/batch identities or ordering disagree')
 context=('Mouse P14 CD8 T-cell RNA counts from GSE278342, capture well '+well+'. Samples comprise naive cells and LCMV Armstrong/Clone 13 tissue samples, distinguished by H3-H11 hashtags. Stable HTO singlets were retained after RNA quality control and reviewed RNA/HTO cluster and score-outlier holds. Only RNA counts are submitted, with the complete supplied RNA gene set. Capture identity is in batch; hashtag/sample identity is encoded in Cell_ID and preserved in the accompanying local metadata.')
 note='T-RBI upload: '+well+'\nPortal: '+PORTAL+'\nChoose type of T cells: CD8\n\nUpload only these four files from this folder:\nmatrix.mtx.gz\ngenes.tsv.gz\nbarcodes.tsv.gz\ncell_batch.csv\n\nCells: '+str(len(selected))+'; genes: '+str(len(genes))+'; orientation: genes by cells.\nThe original count checkpoints, held cells and all genes are preserved. No normalized/HVG-only matrix is used.\nBatch is the capture well. Hashtag/sample labels are included in the unique Cell_ID values and the separate local metadata/ID map.\nManual RNA-score >0.30 holds are conservative review holds, not established doublet truth. Same-tag or otherwise missed doublets cannot be ruled out by these diagnostics.\n\nExperimental context (under 200 words):\n'+context+'\n\nSubmit one well as one job; do not combine both filesets (combined total exceeds the 25,000-cell limit). Portal permits one job per 24 hours as checked 2026-10-16. Email and data-retention consent are your choices.\n'
 (dest/'READ_BEFORE_UPLOAD.txt').write_text(note)
 validation={'well':well,'source_GSM':gsm,'genes':len(genes),'cells':len(selected),'orientation':'genes_by_cells','all_supplied_RNA_genes_retained':True,'missing_required_genes':[],'full_matrix_roundtrip_matches_original_selected_counts':True,'unique_cell_IDs':True,'batch_is_capture_well':batch,'HTO_sample_in_Cell_ID_and_local_metadata':True,'source_gene_symbols_SHA256':sha(folder/'genes.tsv.gz'),'selection_policy':policy,'RNA_score_review_hold_gt':SCORE_REVIEW_HOLD,'input_step08_manifest_SHA256':sha(mp),'files_SHA256':{n:sha(dest/n) for n in ['matrix.mtx.gz','genes.tsv.gz','barcodes.tsv.gz','cell_batch.csv']}}
 dump(dest/'VALIDATION.json',validation)
 row={'well':well,'source_GSM':gsm,'source_cells':len(q),'stable_HTO_singlets':int(stable.sum()),'held_by_reviewed_clusters_among_stable':int((stable&(d|my)).sum()),'additional_score_or_unresolved_holds':int((core&(score_hold|unresolved)).sum()),'exported_cells':len(selected),'genes':len(genes),'upload_folder':str(dest),'portal_T_cell_type':'CD8','status':'ROUNDTRIP_VALIDATED_CORE_EXPORT'}
 exports.append(row);inputs.append({'well':well,'step04_manifest_SHA256':sha(folder/'manifest.json'),'step05_calls_SHA256':sha(hp),'step06_table_SHA256':sha(p6),'step08_manifest_SHA256':sha(mp)})
 say('EXPORTED',json.dumps(row));del x,matrix,back

def bundle_remaining():
 inventory=[];pdir=raw/'GSE267584'
 for name in ['process_green_multiome.R','1-s2.0-S1074761325001931-mmc2.csv','GSE267584_multiome_postQC.rds','GSE267584_multiome_postQC_withBatch.rds','GSE267584_multiome_preQC_list.rds']:
  p=pdir/name;row={'path':str(p),'exists':p.is_file(),'size_bytes':p.stat().st_size if p.is_file() else None,'mtime_UTC':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat() if p.is_file() else None,'usage':'UNVERIFIED_PROVENANCE_NOT_USED_FOR_CELL_SELECTION'}
  if p.is_file() and p.suffix in ['.R','.csv'] and p.stat().st_size<=5*1024*1024:row['SHA256']=sha(p)
  inventory.append(row)
 pd.DataFrame(inventory).to_csv(out/'GSE267584_EXISTING_FILES_PROVENANCE.tsv',sep='\t',index=False)
 already={'GSM8269393','GSM8269399','GSM8545120','GSM8545124','gut','tumor'}
 idx=pd.read_csv(step8/'EMBEDDING_SUMMARY.tsv',sep='\t',keep_default_na=False)
 with zipfile.ZipFile(out/'SHARE_STEP09_REMAINING_REVIEW.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for r in idx.itertuples(index=False):
   if r.library in already:continue
   label=r.GSE+'__'+r.library;folder=step8/label;m=json.loads((folder/'MANIFEST.json').read_text())
   for name in ['UMAP_REVIEW.png','MARKER_DOTPLOT_REVIEW.png','CLUSTER_SUMMARY.tsv','CLUSTER_MARKER_MEAN_LOG_CPT10K.tsv','CLUSTER_MARKER_DETECTED_FRACTION.tsv']:
    p=folder/name
    if r.status=='EMBEDDED_FOR_REVIEW' and not p.is_file():raise ValueError('Missing remaining review file: '+str(p))
    if p.exists():
     if sha(p)!=m['outputs_SHA256'][name]:raise ValueError('Changed remaining review file: '+str(p))
     z.write(p,label+'/'+name)
   z.write(folder/'MANIFEST.json',label+'/MANIFEST.json')
  for name in ['EMBEDDING_SUMMARY.tsv','ALL_CLUSTER_SUMMARIES.tsv']:z.write(step8/name,name)
  z.write(out/'GSE267584_EXISTING_FILES_PROVENANCE.tsv','GSE267584_EXISTING_FILES_PROVENANCE.tsv')
  for row in inventory:
   if row.get('SHA256'):z.write(row['path'],'source_provenance/'+pathlib.Path(row['path']).name)
  for name in ['EXPORT_SUMMARY.tsv','SELECTION_POLICY.json','W8_SAMPLE_COUNTS.tsv','W9_SAMPLE_COUNTS.tsv']:
   if (out/name).exists():z.write(out/name,name)
 say('Remaining-review bundle created; no RDS objects loaded and no processing script executed.')

def main():
 m=json.loads((step8/'RUN_MANIFEST.json').read_text())
 if m['failures'] or m['completed_libraries']!=23:raise ValueError('Step 08 incomplete or has failures')
 if m['source_step04']!=str(prior) or m['source_step06']!=str(step6):raise ValueError('Step 08 source paths disagree')
 if sha(step8/'embedding_review.py')!=m['code_SHA256']:raise ValueError('Step 08 analysis source changed')
 if json.loads((step6/'RUN_MANIFEST.json').read_text())['step05']!=str(step5):raise ValueError('Step 05 source path differs')
 dump(out/'SELECTION_POLICY.json',{'portal':PORTAL,'policy_scope':'Reviewed W8/W9 only; not applicable to other libraries','well_policies':POLICY,'RNA_score_review_hold_gt':SCORE_REVIEW_HOLD,'RNA_score_hold_rationale':'Conservative manual hold beyond the >99th-percentile region of the stable HTO comparison scores in each reviewed well; not a calibrated doublet probability or a validated universal cutoff','RNA_QC':'counts >500, detected symbols >300, mito <=20%','HTO_QC':'same singlet tag under all three priors and default singlet posterior >=0.9','all_source_cells_and_genes_preserved':True,'other_libraries':'still pending final selection; no exports here','held_studies':['GSE262306','GSE334878'],'excluded_GSMs':['GSM7220527','GSM7220528','GSM8062937','GSM8062934']})
 for gsm in POLICY:
  try:export_well(gsm)
  except Exception as e:failures.append({'GSM':gsm,'error':str(e)});traceback.print_exc()
 pd.DataFrame(exports).to_csv(out/'EXPORT_SUMMARY.tsv',sep='\t',index=False)
 try:bundle_remaining()
 except Exception as e:failures.append({'stage':'remaining_review_bundle','error':str(e)});traceback.print_exc()
 dump(out/'RUN_MANIFEST.json',{'code_SHA256':sha(pathlib.Path(__file__)),'source_step08':str(step8),'source_step08_manifest_SHA256':sha(step8/'RUN_MANIFEST.json'),'input_provenance':inputs,'software':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'validated_core_exports':len(exports),'exported_cells':sum(r['exported_cells'] for r in exports),'source_counts_unchanged':True,'other_libraries_not_finalized':True,'no_server_submission_performed':True,'failures':failures})
 say('STEP 09 validated core exports:',len(exports),'cells:',sum(r['exported_cells'] for r in exports),'failures:',len(failures))
 if failures:raise RuntimeError('Some checks failed. Share errors; earlier inputs and completed outputs are preserved.')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
