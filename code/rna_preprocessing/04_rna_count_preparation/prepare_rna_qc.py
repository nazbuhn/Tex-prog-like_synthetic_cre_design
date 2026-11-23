import collections,csv,datetime,gc,gzip,hashlib,json,os,pathlib,re,socket,sys,traceback
import h5py,numpy as np,scipy
from scipy import sparse,io
from threadpoolctl import threadpool_limits,threadpool_info
raw,prior,out=map(pathlib.Path,sys.argv[1:4])
required=set(('Trav14-1 Trgv2 Trac Cd3e Foxp3 Izumo1r Cd40lg Dapl1 Cd5 Trat1 Cd3d Zfp683 Themis Esm1 Cd3g Il10 Sdcbp2 Olfr524 Ly6c1 Sox13 Syt13 Gzmk Trdv1 Trdv2-1 Trdv2-2 Trdv3 Trdv4 Trdd1 Trdd2 Trdj1 Trdj2 Trdc Trdv5 Trgv7 Trgv4 Trgv6 Trgv5 Trgj1 Trgc1 Trgv3 Trgj3 Trgc3 Trgc2 Trgj2 Trgv1 Trgj4 Trgc4').split())
markers=['Trac','Cd3d','Cd3e','Cd3g','Cd8a','Cd8b1','Cd8b','Cd4','Foxp3','Trdc','Zbtb16','Ncr1','Ms4a1','Lyz2']
allowed={'GSE230363':{'GSM7220513','GSM7220515','GSM7220517','GSM7220519','GSM7220521','GSM7220523','GSM7220525'},'GSE255042':{'GSM8062936'},'GSE267584':{'GSM'+str(i) for i in range(8269389,8269400)},'GSE278342':{'GSM8545120','GSM8545124'}}
def say(*v): print(*v,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
def writejson(p,v): p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def text(v): return v.decode(errors='replace') if isinstance(v,bytes) else str(v)
def checked_counts(x):
 if not sparse.issparse(x): raise ValueError('Expected sparse matrix; inspect dense source separately')
 # Validate original entries before duplicate summation or integer conversion.
 for start in range(0,len(x.data),1000000):
  v=x.data[start:start+1000000]
  if not np.isfinite(v).all() or (v<0).any() or not np.equal(v,np.floor(v)).all(): raise ValueError('Non-integer, negative or non-finite counts; no rounding or reversal of normalization permitted')
  if v.size and v.max()>1000000000: raise ValueError('Implausible count value')
 x=x.astype(np.int64).tocsr(); x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
 x.check_format(full_check=True)
 return x
with open(out/'NCBI_TCRg_subset.tsv') as h: refs=list(csv.DictReader(h,delimiter='\t'))
feature_hashes=json.loads((prior/'feature_file_sha256.json').read_text())
with open(prior/'required_gene_mapping_CANDIDATE.tsv') as h: mapping_checks=list(csv.DictReader(h,delimiter='\t'))
expected_mappings={(gse,gsm,ref['symbol']) for gse,gsms in allowed.items() for gsm in gsms for ref in refs}
observed_mappings={(r['GSE'],r['GSM'],r['required_symbol']) for r in mapping_checks}
if observed_mappings!=expected_mappings or len(mapping_checks)!=len(expected_mappings): raise ValueError('Successful Step 03 mapping table is incomplete or duplicated')
for row in mapping_checks:
 if row['mapping_status'] not in ['ALREADY_PRESENT','ENSEMBL_ID','UNIQUE_NCBI_ALIAS']: raise ValueError('Unresolved Step 03 mapping: '+str(row))
def canonicalize(x,feature_ids,symbols):
 if x.shape[1]!=len(symbols) or len(symbols)!=len(feature_ids): raise ValueError('RNA feature dimensions disagree')
 if len(set(feature_ids))!=len(feature_ids): raise ValueError('Repeated RNA feature IDs; inspect before combining')
 names=list(symbols); mapping=[]
 for ref in refs:
  target=ref['symbol']; ens=ref['Ensembl_ID']; aliases=set(ref['globally_unique_aliases'].split('|'))
  for i,(fid,old) in enumerate(zip(feature_ids,symbols)):
   matched_id=ens!='-' and re.sub(r'\.\d+$','',fid)==ens
   matched_alias=old in aliases
   if old==target: continue
   if not (matched_id or matched_alias): continue
   if matched_alias and ens!='-' and re.sub(r'\.\d+$','',fid)!=ens: raise ValueError('Alias/Ensembl disagreement for '+old)
   if names[i]!=old and names[i]!=target: raise ValueError('Conflicting gene rename')
   names[i]=target; mapping.append([fid,old,target,'ENSEMBL_ID' if matched_id else 'UNIQUE_NCBI_ALIAS'])
 if any(not n or re.search(r'\s',n) or re.match(r'^ENS[A-Z]*G\d',n) or n.isdigit() for n in names): raise ValueError('Invalid gene symbol rows')
 genes=list(dict.fromkeys(names)); gi={g:i for i,g in enumerate(genes)}
 projection=sparse.csr_matrix((np.ones(len(names),dtype=np.int64),(np.arange(len(names)),[gi[n] for n in names])),shape=(len(names),len(genes)))
 before=np.asarray(x.sum(axis=1)).ravel()
 x=(x@projection).tocsr(); x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
 if not np.array_equal(before,np.asarray(x.sum(axis=1)).ravel()): raise ValueError('RNA totals changed while combining duplicate symbols')
 return x,genes,mapping,names
def stats(x,genes):
 total=np.asarray(x.sum(axis=1)).ravel(); detected=np.diff(x.indptr)
 mt=np.array([g.lower().startswith('mt-') for g in genes])
 mtcount=np.asarray(x[:,mt].sum(axis=1)).ravel()
 pct=np.divide(100.0*mtcount,total,out=np.full(len(total),np.nan),where=total>0)
 minimum=(detected>300)&(total>500)
 quantiles={}
 for name,values in [('RNA_counts',total),('detected_symbols',detected),('percent_mito',pct)]:
  good=values[np.isfinite(values)]
  quantiles[name]=dict(zip(['min','p01','p05','p25','median','p75','p95','p99','max'],[float(v) for v in np.quantile(good,[0,.01,.05,.25,.5,.75,.95,.99,1])])) if len(good) else {}
 return total,detected,pct,minimum,int(mt.sum()),quantiles
summaries=[]; failures=[]
def checkpoint(gse,lib,x,feature_ids,symbols,barcodes,author_ids,author_labels,source_paths,details):
 if len(set(barcodes))!=len(barcodes) or any(not b or re.search(r'\s',b) for b in barcodes): raise ValueError('Invalid or duplicated original cell IDs')
 x=checked_counts(x)
 if x.shape!=(len(barcodes),len(symbols)): raise ValueError('Cell/feature dimensions disagree')
 x,genes,mapping,names=canonicalize(x,feature_ids,symbols)
 missing=sorted(required-set(genes))
 if missing: raise ValueError('Required measured gene rows remain missing: '+','.join(missing))
 total,detected,pct,minimum,nmt,quantiles=stats(x,genes)
 if not nmt: raise ValueError('No mitochondrial symbols detected; inspect gene annotations')
 folder=out/(gse+'__'+lib); folder.mkdir()
 sparse.save_npz(folder/'RNA_counts_cells_by_genes.npz',x,compressed=True)
 with gzip.open(folder/'genes.tsv.gz','wt') as h: h.write('\n'.join(genes)+'\n')
 with gzip.open(folder/'source_feature_map.tsv.gz','wt') as h:
  w=csv.writer(h,delimiter='\t'); w.writerow(['source_feature_ID','source_symbol','canonical_symbol','canonical_column_0based'])
  gi={g:i for i,g in enumerate(genes)}; w.writerows([fid,old,new,gi[new]] for fid,old,new in zip(feature_ids,symbols,names))
 marker_present=[m for m in markers if m in genes]
 marker_values=x[:,[genes.index(m) for m in marker_present]].toarray()
 cell_ids=[gse+'__'+lib+'__'+b for b in barcodes]
 if len(set(cell_ids))!=len(cell_ids): raise ValueError('Cell ID prefixing created duplicates')
 with gzip.open(folder/'cell_QC.tsv.gz','wt') as h:
  w=csv.writer(h,delimiter='\t'); w.writerow(['Cell_ID','original_barcode','author_cell_ID','author_celltype_label','source_GSE','source_library','candidate_batch_NOT_VERIFIED','RNA_counts','detected_symbols','percent_mito','pass_RBI_RNA_minimum','doublet_status','eligibility_status']+['RNA_'+m for m in marker_present])
  for i,b in enumerate(barcodes):
   w.writerow([cell_ids[i],b,author_ids[i],author_labels[i],gse,lib,lib,int(total[i]),int(detected[i]),'' if not np.isfinite(pct[i]) else float(pct[i]),str(bool(minimum[i])).lower(),'NOT_TESTED','NOT_APPROVED']+marker_values[i].tolist())
 manifest={'source_GSE':gse,'library':lib,'orientation':'cells_by_genes','source_RNA_features':len(symbols),'canonical_symbols':len(genes),'cells':len(barcodes),'nonzero_counts':int(x.nnz),'total_UMIs':int(total.sum()),'pass_RBI_RNA_minimum':int(minimum.sum()),'minimum_rule':'detected canonical RNA symbols >300 AND RNA counts >500; no other cell filtering applied',
  'mitochondrial_symbols':nmt,'quantiles':quantiles,'renamed_feature_rows':mapping,'duplicate_symbol_excess_rows':len(names)-len(genes),'missing_required_genes':missing,'source_sha256':{str(p):sha(p) for p in source_paths},'source_selection':details,'export_ready':False,
  'pending':['cell eligibility and conventional CD8 purity review','doublet removal','per-library mitochondrial QC review','verified encapsulation-run assignment']+(['HTO demultiplexing, tag identities, negatives and multiplets'] if gse=='GSE278342' else []),
  'output_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
 writejson(folder/'manifest.json',manifest)
 summary={'GSE':gse,'library':lib,'cells':len(barcodes),'RNA_symbols':len(genes),'minimum_QC_pass':int(minimum.sum()),'median_counts':quantiles['RNA_counts']['median'],'median_genes':quantiles['detected_symbols']['median'],'median_percent_mito':quantiles['percent_mito'].get('median'),'status':'NOT_APPROVED'}
 summaries.append(summary)
 say('DONE',lib,'cells:',len(barcodes),'symbols:',len(genes),'minimum RNA QC pass:',int(minimum.sum()),'median counts:',summary['median_counts'],'median genes:',summary['median_genes'],'median mito %:',round(summary['median_percent_mito'],3) if summary['median_percent_mito'] is not None else 'NA')
 say('  Counts quantiles:',quantiles['RNA_counts']); say('  Detected genes quantiles:',quantiles['detected_symbols']); say('  Mito percent quantiles:',quantiles['percent_mito'])
 return None
def mtx_library(gse,gsm,matrix):
 prefix=str(matrix)[:-len('matrix.mtx.gz')]; features=pathlib.Path(prefix+'features.tsv.gz'); bcf=pathlib.Path(prefix+'barcodes.tsv.gz')
 if feature_hashes.get(str(features))!=sha(features): raise ValueError('Feature file differs from successful Step 03 snapshot')
 with gzip.open(features,'rt') as h: ft=list(csv.reader(h,delimiter='\t'))
 if any(len(r)<3 for r in ft): raise ValueError('Feature type column missing')
 with gzip.open(bcf,'rt') as h: bc=[s.rstrip('\r\n') for s in h]
 say('READ',gse,gsm,matrix.name,datetime.datetime.now(datetime.timezone.utc).isoformat())
 with gzip.open(matrix,'rb') as h:
  m=io.mmread(h)
  while h.read(8*1024*1024): pass
 if not sparse.issparse(m) or m.shape!=(len(ft),len(bc)): raise ValueError('MTX is not sparse or dimensions disagree')
 mask=np.array([r[2]=='Gene Expression' for r in ft]); ids=[r[0] for r in ft if r[2]=='Gene Expression']; syms=[r[1] for r in ft if r[2]=='Gene Expression']
 # Validate RNA entries before any sparse conversion can sum duplicate coordinates.
 coo=m.tocoo(copy=False); keep=mask[coo.row]
 selected=sparse.coo_matrix((coo.data[keep],(coo.col[keep],coo.row[keep])),shape=(len(bc),len(ft)))
 selected=checked_counts(selected)[:,mask].tocsr()
 del m,coo,keep
 checkpoint(gse,gsm,selected,ids,syms,bc,['']*len(bc),['']*len(bc),[matrix,features,bcf],{'rule':'All source cellular barcodes retained for numeric QC; no biological or doublet approval','source_matrix_features':len(ft),'source_cells':len(bc),'dropped_non_RNA_features':int((~mask).sum())})
def h5_cd8(batch):
 p=raw/'GSE290618'/('GSE290618_'+batch+'_filtered_feature_bc_matrix.h5'); metadata=raw/'GSE290618/GSE290618_metadata.tsv.gz'
 with gzip.open(metadata,'rt') as h:
  reader=csv.reader(h,delimiter='\t'); head=next(reader); records=list(reader)
 bcol=head.index('batch'); tcol=head.index('celltype_label'); relevant=[r for r in records if r[bcol]==batch]
 labels={'PD1+ CD8','Cytotoxic CD8','CCR7+ CD8'}
 with h5py.File(p,'r') as h:
  m=h['matrix']; ft=m['features']; bc=[text(b) for b in m['barcodes'][...]]
  if len(set(bc))!=len(bc): raise ValueError('H5 has duplicate barcodes')
  bi={b:i for i,b in enumerate(bc)}; chosen=[]; sourceids=[]; authorlabels=[]; seen=set()
  for r in relevant:
   suffix='-'+batch
   if not r[0].endswith(suffix): raise ValueError('Metadata ID has unexpected tissue suffix')
   b=r[0][:-len(suffix)]
   if b not in bi or b in seen: raise ValueError('Metadata barcode not uniquely matched to H5')
   seen.add(b)
   if r[tcol] in labels: chosen.append(bi[b]); sourceids.append(r[0]); authorlabels.append(r[tcol])
  if not chosen: raise ValueError('No author-labelled conventional CD8 candidates')
  types=[text(v) for v in ft['feature_type'][...]]; mask=np.array([v=='Gene Expression' for v in types])
  ids=[text(v) for v,t in zip(ft['id'][...],types) if t=='Gene Expression']; symbols=[text(v) for v,t in zip(ft['name'][...],types) if t=='Gene Expression']
  say('READ GSE290618',batch,'author-labelled CD8 candidates:',len(chosen))
  full=sparse.csc_matrix((m['data'][...],m['indices'][...],m['indptr'][...]),shape=tuple(m['shape'][...]))
  full.check_format(full_check=True)
  # 10x H5 CSC stores one gene row per cell. Validate all original stored entries.
  for start in range(0,len(full.data),1000000):
   v=full.data[start:start+1000000]
   if not np.isfinite(v).all() or (v<0).any() or not np.equal(v,np.floor(v)).all(): raise ValueError('H5 source has invalid counts')
  x=full[:,chosen][mask,:].T.tocsr(); del full
  selected=[bc[i] for i in chosen]
 checkpoint('GSE290618',batch,x,ids,symbols,selected,sourceids,authorlabels,[p,metadata],{'rule':'Author labels PD1+ CD8, Cytotoxic CD8 or CCR7+ CD8; other lineages excluded from this checkpoint','source_cells':len(bc),'author_metadata_cells':len(relevant),'source_cells_without_author_metadata':len(bc)-len(seen),'author_other_lineages':len(relevant)-len(chosen),'selected_CD8_candidates':len(chosen)})
def load_h5ad_sparse(node):
 if not isinstance(node,h5py.Group): raise ValueError('Dense H5AD count layer: inspect separately')
 encoding=text(node.attrs.get('encoding-type',''))
 cls=sparse.csr_matrix if encoding=='csr_matrix' else sparse.csc_matrix if encoding=='csc_matrix' else None
 if cls is None: raise ValueError('Unsupported sparse encoding: '+encoding)
 x=cls((node['data'][...],node['indices'][...],node['indptr'][...]),shape=tuple(node.attrs['shape'])); x.check_format(full_check=True)
 return checked_counts(x)
def check_held_h5ad():
 p=raw/'GSE334878/GSE334878_multiome_rna_final.h5ad'
 with h5py.File(p,'r') as h:
  x=load_h5ad_sparse(h['layers']['raw']); y=load_h5ad_sparse(h['raw']['X'])
  equal=x.shape==y.shape and (x!=y).nnz==0
  var=h['var']; idx=text(var.attrs.get('_index','_index')); syms=[text(v) for v in var[idx][...]]; ids=[text(v) for v in var['gene_ids'][...]]
  if x.shape[1]!=len(syms): raise ValueError('H5AD raw layer and var disagree')
  x,genes,mapping,_=canonicalize(x,ids,syms); totals,detected,pct,minimum,nmt,quantiles=stats(x,genes)
  obs=h['obs']; original_n=obs['n_counts'][...] if 'n_counts' in obs else None
  delta=np.abs(totals-original_n) if original_n is not None else None
  missing=sorted(required-set(genes))
  rawvar=h['raw']['var']; rawidx=text(rawvar.attrs.get('_index','_index'))
  samevar=[text(v) for v in rawvar[rawidx][...]]==syms and [text(v) for v in rawvar['gene_ids'][...]]==ids
  result={'status':'HOLD_NOT_EXPORTED','cells':int(x.shape[0]),'measured_symbols':len(genes),'all_count_entries_nonnegative_integers':True,'layers_raw_equals_raw_X':equal,'layer_and_raw_var_identity_order_agree':samevar,'missing_required_after_verified_aliases':missing,'source_gene_scope':'Only the '+str(len(syms))+' supplied features; full original feature universe not established','n_counts_match_all_cells':bool(np.allclose(totals,original_n,rtol=0,atol=.001)) if original_n is not None else None,'max_abs_difference_from_obs_n_counts':float(delta.max()) if delta is not None else None,'quantiles':quantiles,'minimum_QC_pass':int(minimum.sum()),'no_cells_approved':True}
  if 'n_cells' in var: result['source_var_n_cells_min']=int(np.min(var['n_cells'][...]))
  writejson(out/'GSE334878_HELD_count_diagnostics.json',result)
  say('GSE334878 HELD:',json.dumps(result))
def main():
 say('STEP 04: RNA-only count checkpoints and QC. No uploads or doublet/lineage approval.')
 say('HOST',socket.gethostname(),'LSF job',os.environ.get('LSB_JOBID'),'Python',sys.executable)
 say('VERSIONS numpy',np.__version__,'h5py',h5py.__version__,'scipy',scipy.__version__)
 for gse,gsms in allowed.items():
  for gsm in sorted(gsms):
   try:
    hits=list((raw/gse).glob(gsm+'_*matrix.mtx.gz'))
    if len(hits)!=1: raise ValueError('Expected exactly one selected RNA triplet for '+gsm)
    mtx_library(gse,gsm,hits[0])
   except Exception as e:
    failures.append({'GSE':gse,'library':gsm,'error':str(e)}); say('FAILED',gsm,str(e)); traceback.print_exc()
   gc.collect()
 for batch in ['gut','tumor']:
  try: h5_cd8(batch)
  except Exception as e: failures.append({'GSE':'GSE290618','library':batch,'error':str(e)}); say('FAILED GSE290618',batch,str(e)); traceback.print_exc()
  gc.collect()
 try: check_held_h5ad()
 except Exception as e: failures.append({'GSE':'GSE334878','library':'HELD_diagnostics','error':str(e)}); say('HELD H5AD CHECK FAILED',str(e)); traceback.print_exc()
 if summaries:
  with open(out/'QC_SUMMARY.tsv','w') as h:
   w=csv.DictWriter(h,fieldnames=list(summaries[0]),delimiter='\t'); w.writeheader(); w.writerows(summaries)
 writejson(out/'failures.json',failures)
 writejson(out/'RUN_MANIFEST.json',{'code_sha256':sha(pathlib.Path(__file__)),'prior_successful_step03':str(prior),'source_mapping_sha256':sha(prior/'required_gene_mapping_CANDIDATE.tsv'),'thread_limits':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},'threadpools':[{k:v for k,v in info.items() if not callable(v)} for info in threadpool_info()],'successful_libraries':len(summaries),'failures':failures,'all_output_cells_NOT_APPROVED':True,'GSE262306':'HOLD normalized expression; original counts needed','GSE334878':'HOLD gene coverage/count provenance and experimental-control eligibility; Trim28 excluded','excluded':['GSM7220527','GSM7220528','GSM8062937','GSM8062934']})
 say('\nSUMMARY libraries completed:',len(summaries),'candidate cells:',sum(r['cells'] for r in summaries),'above RNA minimum:',sum(r['minimum_QC_pass'] for r in summaries),'failures:',len(failures))
 say('Final QC SUMMARY:',out/'QC_SUMMARY.tsv'); say('No matrix is approved for ImmGen upload. Doublet removal, HTO demultiplexing, cell eligibility and QC review are next.')
 if failures: raise RuntimeError('Some checks failed. Completed outputs are preserved; share report and failures.json.')
if __name__=='__main__':
 with threadpool_limits(limits=1): main()
