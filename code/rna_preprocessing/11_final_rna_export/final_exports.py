import csv,datetime,gzip,hashlib,json,math,pathlib,re,sys,traceback,zipfile
import numpy as np,pandas as pd,scipy
from scipy import sparse,io as sio
from threadpoolctl import threadpool_limits

prior,step8,step9,step10,out=map(pathlib.Path,sys.argv[1:6])
REQUIRED=set('Trav14-1 Trgv2 Trac Cd3e Foxp3 Izumo1r Cd40lg Dapl1 Cd5 Trat1 Cd3d Zfp683 Themis Esm1 Cd3g Il10 Sdcbp2 Olfr524 Ly6c1 Sox13 Syt13 Gzmk Trdv1 Trdv2-1 Trdv2-2 Trdv3 Trdv4 Trdd1 Trdd2 Trdj1 Trdj2 Trdc Trdv5 Trgv7 Trgv4 Trgv6 Trgv5 Trgj1 Trgc1 Trgv3 Trgj3 Trgc3 Trgc2 Trgj2 Trgv1 Trgj4 Trgc4'.split())
TAG_MAP={'H3':'Naive','H4':'Arm_Spl','H5':'C13_Spl','H6':'Arm_Liver','H7':'C13_Liver','H8':'Arm_IEL','H9':'C13_IEL','H10':'Arm_SG','H11':'C13_SG'}
PORTAL='https://rstats.immgen.org/immgenT_integration_analysis/'
library_rows=[];job_rows=[];parts=[];input_records=[]
def say(*v):print(*v,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n');tmp.replace(p)
def table(p):return pd.read_csv(p,sep='\t',keep_default_na=False,dtype={'Cell_ID':str,'original_Cell_ID':str,'original_barcode':str,'cluster_review':str})
def number(s):return pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
def truth(s):
 v=s.astype(str).str.lower()
 if not v.isin(['true','false']).all():raise ValueError('Invalid boolean data')
 return v.eq('true').to_numpy()
def gzwrite(p,text=None,matrix=None):
 with open(p,'wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=6) as z:
   if matrix is not None:sio.mmwrite(z,matrix,field='integer',symmetry='general',comment='Original RNA counts; genes by cells; all supplied RNA genes retained')
   else:z.write(text.encode())
def gztable(q,p,sep='\t'):gzwrite(p,text=q.to_csv(index=False,sep=sep,lineterminator='\n'))

def select(q,label,cfg,old_core=None):
 gate=truth(q.model_mask_RNA_min_and_mito_le20);candidate=truth(q.is_step04_candidate)
 call=number(q.DD_call_vote_ge_0_9);valid=np.isfinite(call)&np.isin(call,[0,1]);positive=call==1
 score=number(q.RNA_doublet_score);cl=q.cluster_review.astype(str).to_numpy()
 exc=cfg['cycling_exception'];cycling=(label==exc['library'])&(cl==exc['cluster'])
 override=cycling&np.isfinite(score)&(score<=exc['retain_if_Scrublet_score_le'])
 dd_hold=positive&~override
 bh=cfg['B_cluster_hold'];b_hold=(label==bh['library'])&(cl==bh['cluster'])
 base=candidate&gate
 if old_core is not None:base&=old_core
 keep=base&valid&~dd_hold&~b_hold
 reasons=[]
 for i in range(len(q)):
  flags=[]
  for condition,text in [(not candidate[i],'OUTSIDE_ORIGINAL_CANDIDATE_SET'),(not gate[i],'RNA_QC_FAIL'),(not valid[i],'DD_UNRESOLVED'),(dd_hold[i],'DD_POSITIVE_VOTE_GE_0_9'),(b_hold[i],'B_LIKE_REVIEW_CLUSTER'),(old_core is not None and not old_core[i],'OUTSIDE_REVIEWED_STEP09_HTO_CORE')]:
   if condition:flags.append(text)
  reasons.append(';'.join(flags) if flags else 'INCLUDED_POST_QC_T_RBI_INPUT')
 return keep,{'gate':gate,'candidate':candidate,'dd_positive':positive,'dd_valid':valid,'dd_hold':dd_hold,'cycling_override':override&positive,'b_hold':b_hold,'base':base,'reasons':reasons}

def load_library(label,cfg):
 gse,lib=label.split('__');folder=prior/label;review=step10/label
 say('VERIFY AND SELECT',label)
 mp=review/'MANIFEST.json'
 if sha(mp)!=cfg['Step10_library_manifest_SHA256'][label]:raise ValueError('Step 10 identity differs from reviewed bundle: '+label)
 m=json.loads(mp.read_text());p=review/'ALL_SOURCE_CELL_SECOND_CALLER_REVIEW.tsv.gz'
 if sha(p)!=m['outputs_SHA256'][p.name]:raise ValueError('Step 10 cell table changed')
 cp=folder/'manifest.json';cm=json.loads(cp.read_text())
 if sha(cp)!=m['signature']['step04_manifest_SHA256']:raise ValueError('Step 04 checkpoint manifest changed')
 if sha(step8/label/'MANIFEST.json')!=m['signature']['step08_manifest_SHA256']:raise ValueError('Step 08 review identity changed')
 for name,digest in cm['output_sha256'].items():
  if sha(folder/name)!=digest:raise ValueError('Source RNA checkpoint changed: '+name)
 q=table(p);q4=table(folder/'cell_QC.tsv.gz')
 if q.Cell_ID.duplicated().any() or q4.Cell_ID.duplicated().any() or q4.original_barcode.duplicated().any():raise ValueError('Duplicate source cell IDs')
 candidate=q.Cell_ID.isin(q4.Cell_ID).to_numpy()
 if not np.array_equal(candidate,truth(q.is_step04_candidate)) or not set(q4.Cell_ID).issubset(set(q.Cell_ID)):raise ValueError('Original candidate set differs')
 sub=q.set_index('Cell_ID',drop=False).loc[q4.Cell_ID].reset_index(drop=True)
 if sub.original_barcode.tolist()!=q4.original_barcode.tolist():raise ValueError('Source barcode mismatch')
 for col in ['RNA_counts','detected_symbols','percent_mito']:
  if not np.allclose(number(sub[col]),number(q4[col]),rtol=1e-12,atol=1e-12,equal_nan=True):raise ValueError('QC mismatch: '+col)
 gate=(number(q.RNA_counts)>500)&(number(q.detected_symbols)>300)&np.isfinite(number(q.percent_mito))&(number(q.percent_mito)<=20)
 if not np.array_equal(gate,truth(q.model_mask_RNA_min_and_mito_le20)):raise ValueError('RNA QC gate mismatch')
 call=number(q.DD_call_vote_ge_0_9);vote=number(q.DD_vote_p1e7);resolved=gate&np.isfinite(call)
 if not np.isin(call[resolved],[0,1]).all() or not np.isfinite(vote[resolved]).all() or not ((vote[resolved]>=0)&(vote[resolved]<=1)).all() or not np.array_equal(call[resolved]==1,vote[resolved]>=.9):raise ValueError('DD calls/votes disagree')
 original=q.Cell_ID.astype(str).to_numpy();upload_ids=original.copy();tags=np.full(len(q),'',dtype=object);samples=tags.copy();old_core=None
 batch=label;batch_verification='SOURCE_LIBRARY_PROXY_NOT_INDEPENDENTLY_VERIFIED_CAPTURE_BOUNDARIES'
 if lib in cfg['Step09_cores']:
  pol=cfg['Step09_cores'][lib];well=pol['well'];batch=label+'__'+well;batch_verification='IDENTIFIED_W8_W9_CAPTURE_WELL'
  old=step9/('UPLOAD_'+well+'_CD8_CORE');vm=json.loads((old/'VALIDATION.json').read_text())
  if vm['input_step08_manifest_SHA256']!=m['signature']['step08_manifest_SHA256']:raise ValueError('Step 09 used another embedding')
  for name,digest in vm['files_SHA256'].items():
   if sha(old/name)!=digest:raise ValueError('Prior Step 09 upload file changed: '+name)
  with gzip.open(old/'barcodes.tsv.gz','rt') as f:old_ids=f.read().splitlines()
  cb=pd.read_csv(old/'cell_batch.csv',keep_default_na=False)
  if len(old_ids)!=pol['expected_cells'] or len(set(old_ids))!=len(old_ids) or cb.columns.tolist()!=['Cell_ID','batch'] or cb.Cell_ID.tolist()!=old_ids or not cb.batch.eq(batch).all():raise ValueError('Step 09 ID/batch check failed')
  tags=q.HashSolo_default.astype(str).to_numpy();samples=np.array([TAG_MAP.get(t,'') for t in tags],dtype=object)
  upload_ids=np.array([label+'__'+t+'__'+TAG_MAP[t]+'__'+b if t in TAG_MAP else cid for t,b,cid in zip(tags,q.original_barcode,original)],dtype=object)
  old_core=np.isin(upload_ids,old_ids)
  expected=gate&truth(q.stable_singlet_posterior_ge_0_9_CANDIDATE)&~q.cluster_review.isin(pol['cluster_holds']).to_numpy()&np.isfinite(number(q.RNA_doublet_score))&(number(q.RNA_doublet_score)<=.3)
  if old_core.sum()!=len(old_ids) or not np.array_equal(old_core,expected):raise ValueError('Prior Step 09 selection does not match reviewed policy')
 keep,flags=select(q,label,cfg,old_core)
 decision=q[['Cell_ID','original_barcode','is_step04_candidate','RNA_counts','detected_symbols','percent_mito','cluster_review','RNA_doublet_score','DD_vote_p1e7','DD_call_vote_ge_0_9','DD_call_vote_ge_0_5']].copy()
 decision['DD_positive_strict']=flags['dd_positive'];decision['cycling_override_retained']=flags['cycling_override']&keep;decision['include_final_input']=keep;decision['decision_reasons']=flags['reasons']
 gztable(decision,out/'CELL_DECISIONS'/(label+'.tsv.gz'))
 # Raw count rows remain in their original Step 04 order, including held rows.
 by_id=pd.Series(np.arange(len(q)),index=q.Cell_ID);q_positions=by_id.loc[q4.Cell_ID].to_numpy(dtype=int);selected=np.flatnonzero(keep[q_positions]);pos=q_positions[selected]
 x=sparse.load_npz(folder/'RNA_counts_cells_by_genes.npz').tocsr();x.check_format(full_check=True)
 for start in range(0,len(x.data),1000000):
  v=x.data[start:start+1000000]
  if not np.isfinite(v).all() or (v<0).any() or not np.equal(v,np.floor(v)).all() or (v>np.iinfo(np.int64).max).any():raise ValueError('Invalid original RNA counts; no rounding allowed')
 x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
 with gzip.open(folder/'genes.tsv.gz','rt') as f:genes=f.read().splitlines()
 if x.shape!=(len(q4),len(genes)) or len(set(genes))!=len(genes):raise ValueError('RNA/gene/cell dimensions differ')
 if REQUIRED-set(genes):raise ValueError('Missing required genes: '+','.join(sorted(REQUIRED-set(genes))))
 if any(not g or re.search(r'\s',g) or re.match(r'^ENS[A-Z]*G\d',g) or g.isdigit() for g in genes):raise ValueError('Invalid gene symbols')
 if not np.array_equal(np.asarray(x.sum(axis=1)).ravel(),number(q4.RNA_counts)) or not np.array_equal(np.diff(x.indptr),number(q4.detected_symbols)):raise ValueError('Original counts disagree with QC')
 canonical=sorted(genes);gi={g:i for i,g in enumerate(genes)};universe=hashlib.sha256(('\n'.join(canonical)+'\n').encode()).hexdigest()
 mode='All T cells' if gse=='GSE290618' else 'CD8'
 meta=pd.DataFrame({'Cell_ID':upload_ids[pos],'original_Cell_ID':original[pos],'original_barcode':q.original_barcode.iloc[pos].to_numpy(),'source_GSE':gse,'source_library':lib,'batch':batch,'batch_boundary_verification':batch_verification,'HTO_tag':tags[pos],'HTO_sample':samples[pos],'sample_ID':[label+'__'+t+'__'+s if s else label for t,s in zip(tags[pos],samples[pos])],'condition_identity_status':'UNRESOLVED_GEO_DESCRIPTION_FILENAME_CONFLICT' if lib in cfg['condition_conflicts'] else 'NO_NEW_DESCRIPTION_FILENAME_CONFLICT_IDENTIFIED','author_celltype_label':q.author_celltype_label.iloc[pos].to_numpy() if 'author_celltype_label' in q else '', 'RNA_counts':number(q.RNA_counts)[pos].astype(np.int64),'detected_symbols':number(q.detected_symbols)[pos].astype(np.int64),'percent_mito':number(q.percent_mito)[pos],'cluster_review':q.cluster_review.iloc[pos].to_numpy(),'Scrublet_score':number(q.RNA_doublet_score)[pos],'DD_vote_p1e7':number(q.DD_vote_p1e7)[pos],'DD_positive_strict':flags['dd_positive'][pos],'DD_positive_vote_ge_0_5_sensitivity':number(q.DD_call_vote_ge_0_5)[pos]==1,'cycling_override_retained':flags['cycling_override'][pos],'portal_type':mode,'RNA_gene_universe_SHA256':universe,'source_step04_row_0based':selected})
 if meta.Cell_ID.duplicated().any() or meta.Cell_ID.str.contains(r'\s').any() or meta.Cell_ID.str.match(r'^IGT\d*\.').any():raise ValueError('Invalid final cell IDs')
 if len(selected):parts.append({'label':label,'mode':mode,'universe':universe,'genes':canonical,'matrix':x[selected][:,[gi[g] for g in canonical]].astype(np.int64).tocsr(),'metadata':meta})
 row={'GSE':gse,'library':lib,'checkpoint_candidate_cells':len(q4),'candidate_RNA_QC_pass':int((flags['candidate']&gate).sum()),'candidate_DD_positive_strict':int((flags['candidate']&gate&flags['dd_positive']).sum()),'cycling_override_retained':int((flags['cycling_override']&keep).sum()),'B_cluster_hold_candidates':int((flags['candidate']&flags['b_hold']).sum()),'Step09_core_cells_before_DD':int(old_core.sum()) if old_core is not None else None,'additional_DD_holds_in_Step09_core':int((old_core&flags['dd_hold']).sum()) if old_core is not None else None,'exported_cells':len(meta),'genes':len(genes),'portal_type':mode,'condition_conflict':lib in cfg['condition_conflicts'],'status':'SELECTED_FOR_FINAL_INPUT' if len(meta) else 'NO_ELIGIBLE_CELLS'}
 library_rows.append(row);input_records.append({'library':label,'Step04_manifest_SHA256':sha(cp),'Step10_manifest_SHA256':sha(mp)})
 say('SELECTED',label,'cells:',len(meta),'cycling overrides:',row['cycling_override_retained']);del x

def write_job(job_id,mode,genes,x,meta,cfg):
 dest=out/'UPLOAD_JOBS'/job_id;dest.mkdir();reports=out/'JOB_REPORTS'/job_id;reports.mkdir()
 if not 0<len(meta)<=cfg['max_cells_per_job']:raise ValueError('Job exceeds cell limit')
 if x.shape!=(len(meta),len(genes)) or not meta.Cell_ID.is_unique:raise ValueError('Job dimensions/IDs differ')
 if not np.array_equal(np.asarray(x.sum(axis=1)).ravel(),meta.RNA_counts.to_numpy()) or not np.array_equal(np.diff(x.indptr),meta.detected_symbols.to_numpy()):raise ValueError('Assembled job lost original counts')
 say('WRITE AND ROUNDTRIP',job_id,'cells:',len(meta),'genes:',len(genes))
 matrix=x.T.tocoo();gzwrite(dest/'matrix.mtx.gz',matrix=matrix);gzwrite(dest/'genes.tsv.gz',text='\n'.join(genes)+'\n');gzwrite(dest/'barcodes.tsv.gz',text='\n'.join(meta.Cell_ID)+'\n');meta[['Cell_ID','batch']].to_csv(dest/'cell_batch.csv',index=False,lineterminator='\n')
 with gzip.open(dest/'matrix.mtx.gz','rb') as f:back=sio.mmread(f).tocsc()
 back.sum_duplicates();back.eliminate_zeros();back.sort_indices()
 if back.shape!=matrix.shape or (back!=matrix.tocsc()).nnz:raise ValueError('Full compressed-matrix round trip differs from original selected counts')
 if not (np.asarray(back.sum(axis=0)).ravel()>500).all() or not (np.diff(back.indptr)>300).all():raise ValueError('Saved job violates minimum QC')
 with gzip.open(dest/'genes.tsv.gz','rt') as f:saved_genes=f.read().splitlines()
 with gzip.open(dest/'barcodes.tsv.gz','rt') as f:saved_ids=f.read().splitlines()
 cb=pd.read_csv(dest/'cell_batch.csv',keep_default_na=False)
 if saved_genes!=genes or saved_ids!=meta.Cell_ID.tolist() or cb.columns.tolist()!=['Cell_ID','batch'] or cb.Cell_ID.tolist()!=saved_ids or cb.batch.tolist()!=meta.batch.tolist():raise ValueError('Saved metadata/genes/cell order mismatch')
 meta=meta.copy();meta['upload_job']=job_id;gztable(meta,reports/'CELL_METADATA_AND_ID_MAP.csv.gz',sep=',')
 if mode=='CD8':context='Mouse P14 CD8 T-cell RNA counts from selected GEO infection and tumor studies. Source-library/capture IDs are supplied as batch labels. W8/W9 cells were hashtag-demultiplexed, with sample identity encoded in Cell_ID. QC and reviewed doublet/contaminant holds were applied. A documented cycling-CD8 exception is retained with sensitivity flags. Some GSE267584 condition descriptions conflict with filenames; those cells retain neutral GSM identities. Tumor P14 cells expanded before in-vivo transfer remain included. All supplied RNA genes are retained; counts are unnormalized. This is one balanced subset of the full retained collection.'
 else:context='Mouse gut/tumor T-cell RNA counts from GSE290618. These are the original author-labelled conventional-CD8 candidates retained after RNA QC and doublet filtering. Earlier review showed mixed T-cell phenotypes, so All T cells is requested to verify lineage before treating them as conventional CD8. Broader author-listed populations were used for doublet modeling only and were not added to these input candidates. All supplied RNA genes are retained; counts are unnormalized. Tissue/source-library IDs are preserved as batch labels.'
 if len(context.split())>200:raise ValueError('Experimental context exceeds word limit')
 (reports/'EXPERIMENTAL_CONTEXT.txt').write_text(context+'\n')
 validation={'job':job_id,'portal_type':mode,'cells':len(meta),'genes':len(genes),'orientation':'genes_by_cells','full_compressed_matrix_matches_original_selected_counts':True,'every_cell_counts_gt_500_and_detected_genes_gt_300':True,'all_supplied_genes_retained_for_every_source_in_this_job':True,'gene_universes_not_intersected_or_zero_padded':True,'gene_universe_SHA256':meta.RNA_gene_universe_SHA256.iloc[0],'source_libraries':sorted(set(meta.source_GSE+'__'+meta.source_library)),'Cell_ID_and_batch_order_verified':True,'cycling_override_cells':int(meta.cycling_override_retained.sum()),'files_SHA256':{name:sha(dest/name) for name in ['matrix.mtx.gz','genes.tsv.gz','barcodes.tsv.gz','cell_batch.csv']}}
 dump(reports/'VALIDATION.json',validation)
 job_rows.append({'job':job_id,'portal_type':mode,'cells':len(meta),'genes':len(genes),'source_libraries':len(validation['source_libraries']),'upload_folder':str(dest),'experimental_context_file':str(reports/'EXPERIMENTAL_CONTEXT.txt'),'validation_file':str(reports/'VALIDATION.json'),'status':'ROUNDTRIP_VALIDATED'})
 return meta

def main():
 cfg=json.loads((out/'SELECTION_POLICY.json').read_text())
 if not 1<=cfg['max_cells_per_job']<=25000:raise ValueError('Invalid job size policy')
 if sha(step10/'RUN_MANIFEST.json')!=cfg['Step10_RUN_MANIFEST_SHA256']:raise ValueError('Step 10 run identity differs from returned results')
 m10=json.loads((step10/'RUN_MANIFEST.json').read_text());m9=json.loads((step9/'RUN_MANIFEST.json').read_text())
 if m10['failures'] or m10['completed_libraries']!=23 or m10['scored_libraries']!=23 or m9['failures'] or m9['validated_core_exports']!=2:raise ValueError('Earlier run incomplete')
 if m10['source_step04']!=str(prior) or m10['source_step08']!=str(step8):raise ValueError('Earlier source paths differ')
 for label in sorted(cfg['Step10_library_manifest_SHA256']):load_library(label,cfg)
 pd.DataFrame(library_rows).to_csv(out/'LIBRARY_SELECTION_SUMMARY.tsv',sep='\t',index=False)
 if not parts:raise ValueError('No retained cells')
 all_ids=[cid for p in parts for cid in p['metadata'].Cell_ID]
 if len(set(all_ids))!=len(all_ids):raise ValueError('Final IDs are not globally unique')
 groups={}
 for p in parts:groups.setdefault((p['mode'],p['universe']),[]).append(p)
 final_metadata=[];group_no={'CD8':0,'All T cells':0}
 for (mode,universe),group in sorted(groups.items()):
  group_no[mode]+=1;genes=group[0]['genes']
  if any(p['genes']!=genes for p in group):raise ValueError('Different supplied gene universes grouped together')
  total=sum(len(p['metadata']) for p in group);k=int(math.ceil(total/cfg['max_cells_per_job']));sizes=np.zeros(k,dtype=int);rng=np.random.default_rng(cfg['allocation_seed'])
  # Distribute each source/hashtag across jobs without losing or duplicating cells.
  for p in group:
   assignment=np.full(len(p['metadata']),-1,dtype=int)
   for sample in sorted(set(p['metadata'].sample_ID)):
    idx=np.flatnonzero(p['metadata'].sample_ID.eq(sample).to_numpy());idx=rng.permutation(idx)
    for i in idx:
     j=int(np.argmin(sizes));assignment[i]=j;sizes[j]+=1
   if (assignment<0).any():raise ValueError('Unassigned retained cell')
   p['assignment']=assignment
  if sizes.max()>cfg['max_cells_per_job'] or sizes.max()-sizes.min()>1:raise ValueError('Unbalanced/oversized job assignment')
  for j in range(k):
   xs=[];ms=[]
   for p in group:
    take=np.flatnonzero(p['assignment']==j)
    if len(take):xs.append(p['matrix'][take]);ms.append(p['metadata'].iloc[take].copy())
   x=sparse.vstack(xs,format='csr');x.sort_indices();meta=pd.concat(ms,ignore_index=True)
   job_id=('CD8' if mode=='CD8' else 'ALLT')+'_G'+str(group_no[mode])+'_J'+str(j+1).zfill(2)
   final_metadata.append(write_job(job_id,mode,genes,x,meta,cfg));del x,meta,xs,ms
 metadata=pd.concat(final_metadata,ignore_index=True)
 if len(metadata)!=len(all_ids) or not metadata.Cell_ID.is_unique or set(metadata.Cell_ID)!=set(all_ids):raise ValueError('Export jobs lost or duplicated retained cells')
 gztable(metadata,out/'ALL_EXPORTED_CELL_METADATA.csv.gz',sep=',')
 pd.DataFrame(job_rows).to_csv(out/'UPLOAD_QUEUE.tsv',sep='\t',index=False)
 metadata.groupby(['source_GSE','source_library','sample_ID','HTO_tag','HTO_sample','upload_job'],dropna=False).size().reset_index(name='exported_cells').to_csv(out/'SAMPLE_JOB_COUNTS.tsv',sep='\t',index=False)
 dump(out/'RUN_MANIFEST.json',{'code_SHA256':sha(pathlib.Path(__file__)),'selection_policy_SHA256':sha(out/'SELECTION_POLICY.json'),'source_step04':str(prior),'source_step09':str(step9),'source_step10':str(step10),'source_provenance':input_records,'completed_libraries':len(library_rows),'nonempty_libraries':len(parts),'validated_upload_jobs':len(job_rows),'exported_cells':len(metadata),'CD8_cells':int(metadata.portal_type.eq('CD8').sum()),'AllT_lineage_verification_cells':int(metadata.portal_type.eq('All T cells').sum()),'cycling_override_retained_cells':int(metadata.cycling_override_retained.sum()),'all_retained_cells_exported_exactly_once':True,'raw_counts_and_prior_exports_unchanged':True,'Step09_exports_superseded_for_new_submissions_by_this_run':True,'no_models_refit':True,'no_server_submission':True,'software':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'failures':[]})
 note='T-RBI upload instructions\nPortal: '+PORTAL+'\n\nUse UPLOAD_QUEUE.tsv. Each upload folder contains exactly the four required files. Select the portal T-cell type shown for that job. Paste the experimental context from its JOB_REPORTS folder. Upload each retained cell once; do not also submit the older Step 09 inputs. No cells were downsampled to meet the size limit. Source batches and hashtag samples were distributed across jobs with their identifiers preserved. Jobs sharing a source batch remain parts of that source, not independent biological replicates.\n\nAs checked 2026-10-19: at most 25,000 cells/job and one submission per 24 hours. Choose email and optional data-retention consent yourself. The script performs no online submission.\n\nAllT jobs contain only the original gut/tumor CD8 candidates; their lineage needs verification from the returned annotations before CD8-specific interpretation. Three GSE267584 condition identities remain unresolved; do not use guessed condition labels. Model calls and the cycling exception remain sensitivity flags in local metadata.\n'
 (out/'READ_BEFORE_UPLOAD.txt').write_text(note)
 with zipfile.ZipFile(out/'SHARE_STEP11_FINAL_REPORTS.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['LIBRARY_SELECTION_SUMMARY.tsv','UPLOAD_QUEUE.tsv','SAMPLE_JOB_COUNTS.tsv','SELECTION_POLICY.json','RUN_MANIFEST.json','READ_BEFORE_UPLOAD.txt','README_STEP_11.md']:
   if (out/name).exists():z.write(out/name,name)
  for p in sorted((out/'JOB_REPORTS').rglob('*')):
   if p.is_file() and p.suffix in ['.json','.txt']:z.write(p,str(p.relative_to(out)))
 say('STEP 11 COMPLETE: libraries:',len(library_rows),'validated upload jobs:',len(job_rows),'cells:',len(metadata),'no models rerun')

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
