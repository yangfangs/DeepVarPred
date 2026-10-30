#!/usr/bin/env python3
"""New strict-missense full cohort, fixed panel, raw predictor values only.
Source inputs remain read-only. No splitting, imputation, scaling or model fit.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,collections,hashlib,json,re,time,traceback
import numpy as np
import pandas as pd

AA20=set('ACDEFGHIKLMNPQRSTVWY')
LABELS={'Pathogenic':1,'Likely_pathogenic':1,'Pathogenic/Likely_pathogenic':1,'Benign':0,'Likely_benign':0,'Benign/Likely_benign':0}
INFO={'#chr','pos(1-based)','ref','alt','hg19_chr','hg19_pos(1-based)','hg18_chr','hg18_pos(1-based)','clinvar_clnsig','clinvar_Orphanet_id','clinvar_id','clinvar_OMIM_id'}
META=['variant_id','variant_key','gene','chr','pos','ref','alt','label','clinvar_clnsig','aaref','aaalt','ensembl_transcriptids','clinvar_ids','n_filter_rows','n_annotation_rows','sequence_index_0based','reference_match']

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for ch in iter(lambda:f.read(1048576),b''):h.update(ch)
 return h.hexdigest()
def chr_clean(x):return str(x).removeprefix('chr').removesuffix('.0')
def add_key(df):
 pos=pd.to_numeric(df['hg19_pos(1-based)'],errors='coerce')
 valid=pos.notna() & (pos==pos.round())
 result=pd.Series(None,index=df.index,dtype=object)
 result.loc[valid]=df.loc[valid,'hg19_chr'].map(chr_clean)+':'+pos.loc[valid].astype('int64').astype(str)+':'+df.loc[valid,'ref'].str.upper()+'>'+df.loc[valid,'alt'].str.upper()
 df['_key']=result
 return df

def numeric(x):
 vals=[]
 for token in re.split(r'[;|:]',str(x)):
  try:
   v=float(token.strip())
   if np.isfinite(v):vals.append(v)
  except ValueError:pass
 return float(np.mean(vals)) if vals else np.nan

def worker(task):
 gene,srcstr,outstr,panel=task;src=Path(srcstr)/gene;od=Path(outstr)/'_per_gene';od.mkdir(exist_ok=True)
 fp=src/'score'/f'{gene}_filter.txt';cp=src/'score'/f'{gene}_clnsig.txt';bp=src/'ref'/f'{gene}_.bed';fa=src/'ref'/f'{gene}_ref.fa'
 stats={'gene':gene,'source_sha256':{str(p):sha256(p) for p in [fp,cp,bp,fa]}}
 head=pd.read_csv(fp,sep='\t',nrows=0).columns.tolist();present=[c for c in panel if c in head]
 recover_hg19='hg19_pos(1-based)' not in head
 cols=[c for c in ['hg19_chr','hg19_pos(1-based)','ref','alt','clinvar_clnsig'] if c in head]+present
 if recover_hg19:cols+=['#chr','pos(1-based)']
 d=pd.read_csv(fp,sep='\t',usecols=cols,dtype=str,low_memory=False).fillna('');stats['raw_filter_rows']=len(d);stats['raw_feature_columns']=len([c for c in head if c not in INFO]);stats['panel_fields_in_header']=present
 if recover_hg19:
  # Old dynamic missingness filter sometimes dropped the hg19 position metadata.
  # Restore only an unambiguous source mapping at the untouched dbNSFP primary allele.
  mm=['#chr','pos(1-based)','ref','alt','hg19_chr','hg19_pos(1-based)']
  mapping=pd.read_csv(cp,sep='\t',usecols=mm,dtype=str,low_memory=False).fillna('')
  def primary(z):return z['#chr'].map(chr_clean)+':'+z['pos(1-based)'].astype(str)+':'+z.ref.str.upper()+'>'+z.alt.str.upper()
  mapping['_primary']=primary(mapping);mapping=add_key(mapping)
  valid=mapping[mapping._key.notna()].copy();ambiguous=set(valid.groupby('_primary')._key.nunique().loc[lambda x:x>1].index)
  valid=valid[~valid._primary.isin(ambiguous)].drop_duplicates('_primary');valid=valid.set_index('_primary')
  pp=primary(d);d['hg19_pos(1-based)']=pp.map(valid['hg19_pos(1-based)']).fillna('');d['hg19_chr']=pp.map(valid.hg19_chr).fillna('')
  stats['hg19_metadata_recovery']={'method':'Unique untouched clnsig hg38 #chr/pos/ref/alt to hg19 allele mapping; no coordinate inferred','rows_restored':int(d['hg19_pos(1-based)'].ne('').sum()),'ambiguous_primary_keys':len(ambiguous),'rows_unmapped':int(d['hg19_pos(1-based)'].eq('').sum())}

 d['_label']=d.clinvar_clnsig.map(LABELS);d=d[d['_label'].notna()].copy();stats['raw_labeled_rows']=len(d)
 if d.empty:
  stats['status']='no_binary_labeled_rows';stats['retained']=0;(od/f'{gene}.json').write_text(json.dumps(stats));return stats
 add_key(d);keys=set(d['_key'].dropna())
 cc=['hg19_chr','hg19_pos(1-based)','ref','alt','aaref','aaalt','clinvar_clnsig','Ensembl_transcriptid','clinvar_id']
 chead=pd.read_csv(cp,sep='\t',nrows=0).columns.tolist();cc=[c for c in cc if c in chead]
 c=pd.read_csv(cp,sep='\t',usecols=cc,dtype=str,low_memory=False).fillna('');add_key(c);c=c[c['_key'].isin(keys)].copy()
 if not {'aaref','aaalt'}.issubset(c.columns):
  stats['status']='missing_amino_acid_columns';stats['retained']=0;(od/f'{gene}.json').write_text(json.dumps(stats));return stats
 c['_strict_missense']=[a in AA20 and b in AA20 and a!=b for a,b in zip(c.aaref,c.aaalt)];c['_binary_label']=c.clinvar_clnsig.map(LABELS)
 anno={}
 for key,z in c.groupby('_key',sort=False):
  anno[key]={'aa_ok':bool(z['_strict_missense'].all()),'labels':set(z['_binary_label'].dropna().astype(int)),
    'aaref':';'.join(sorted(set(z.aaref))),'aaalt':';'.join(sorted(set(z.aaalt))),
    'ensembl_transcriptids':';'.join(sorted(set(z.get('Ensembl_transcriptid',pd.Series(dtype=str)).astype(str)))),
    'clinvar_ids':';'.join(sorted(set(z.get('clinvar_id',pd.Series(dtype=str)).astype(str)))),'n_annotation_rows':len(z),
    'aa_exclusion_category':'mixed_valid_and_nonmissense' if z['_strict_missense'].any() and not z['_strict_missense'].all() else 'stop_or_stoploss' if ((z.aaref.isin(['X','*']))|(z.aaalt.isin(['X','*']))).any() else 'other_noncanonical_or_equal_aa'}
 bed=bp.read_text().strip().split('\t');chrom=chr_clean(bed[0]);start,end=map(int,bed[1:3]);seq=''.join(l.strip() for l in fa.read_text().splitlines() if not l.startswith('>')).upper()
 stats['reference']={'chr':chrom,'bed_start_0based':start,'bed_end_exclusive':end,'sequence_length':len(seq),'bed_length':end-start,'fasta_path':str(fa),'bed_path':str(bp)}
 rejected=[];rows=[];conf=[]
 if len(seq)!=end-start:
  stats['status']='FASTA_BED_length_mismatch';stats['retained']=0;(od/f'{gene}.json').write_text(json.dumps(stats));return stats
 invalid=d[d['_key'].isna()]
 for _,r in invalid.iterrows():rejected.append({'gene':gene,'variant_key':'','reason':'invalid_genomic_key','n_filter_rows':1})
 for key,z in d[d['_key'].notna()].groupby('_key',sort=False):
  first=z.iloc[0];ls=set(z['_label'].astype(int));a=anno.get(key);reason=None
  if a is None:reason='no_amino_acid_annotation'
  elif len(ls)>1 or len(a['labels'])>1 or (a['labels'] and a['labels']!=ls):reason='opposing_binary_labels'
  elif not a['aa_ok']:reason='non_missense_or_mixed_amino_acid_consequence'
  elif len(first['ref'])!=1 or len(first['alt'])!=1 or first['ref'].upper() not in 'ACGT' or first['alt'].upper() not in 'ACGT' or first['ref']==first['alt']:reason='not_single_nucleotide_substitution'
  if reason:
   rejected.append({'gene':gene,'variant_key':key,'reason':reason,'n_filter_rows':len(z),'aa_category':a['aa_exclusion_category'] if a else ''})
   if reason=='opposing_binary_labels':conf.append({'gene':gene,'variant_key':key,'filter_labels':str(sorted(ls)),'annotation_labels':str(sorted(a['labels']))})
   continue
  pos=int(float(first['hg19_pos(1-based)']));ref=first['ref'].upper();alt=first['alt'].upper();idx=pos-(start+1)
  if chr_clean(first['hg19_chr'])!=chrom or not 0<=idx<len(seq):reason='outside_fasta_interval'
  elif seq[idx]!=ref:reason='reference_base_mismatch'
  if reason:
   rejected.append({'gene':gene,'variant_key':key,'reason':reason,'n_filter_rows':len(z),'fasta_base':seq[idx] if 0<=idx<len(seq) else ''});continue
  row={'variant_id':gene+':'+key,'variant_key':key,'gene':gene,'chr':chrom,'pos':pos,'ref':ref,'alt':alt,'label':int(next(iter(ls))),'clinvar_clnsig':';'.join(sorted(set(z.clinvar_clnsig))),**{k:a[k] for k in ['aaref','aaalt','ensembl_transcriptids','clinvar_ids','n_annotation_rows']},'n_filter_rows':len(z),'sequence_index_0based':idx,'reference_match':True}
  for col in panel:
   if col not in present:row[col]=np.nan;continue
   vals=[numeric(v) for v in z[col]];vals=[v for v in vals if np.isfinite(v)];row[col]=float(np.mean(vals)) if vals else np.nan
  rows.append(row)
 df=pd.DataFrame(rows,columns=META+panel);df.to_parquet(od/f'{gene}.parquet',index=False)
 pd.DataFrame(rejected).to_csv(od/f'{gene}.excluded.csv',index=False)
 pd.DataFrame(conf,columns=['gene','variant_key','filter_labels','annotation_labels']).to_csv(od/f'{gene}.conflicts.csv',index=False)
 stats.update({'status':'ok','retained':len(df),'retained_by_label':df.label.value_counts().to_dict(),'excluded_reasons':dict(collections.Counter(x['reason'] for x in rejected)),'aa_exclusion_categories':dict(collections.Counter(x.get('aa_category','') for x in rejected if x['reason']=='non_missense_or_mixed_amino_acid_consequence')),'n_conflicts':len(conf),'feature_coverage':{c:float(df[c].notna().mean()) if len(df) else 0.0 for c in panel}})
 (od/f'{gene}.json').write_text(json.dumps(stats));return stats

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='/home/bio1/workdata/GeneMutation2/train_all');ap.add_argument('--output',required=True);ap.add_argument('--panel_manifest',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--reuse_valid_gene_outputs',action='store_true');args=ap.parse_args()
 src=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 pmanifest=json.loads(Path(args.panel_manifest).read_text());panel=pmanifest['recommended_fixed_panel'];assert len(panel)==46 and all(c not in INFO for c in panel)
 genes=[]
 for gd in sorted(src.iterdir()):
  if not gd.is_dir():continue
  if all(p.exists() for p in [gd/'score'/f'{gd.name}_filter.txt',gd/'score'/f'{gd.name}_clnsig.txt',gd/'ref'/f'{gd.name}_ref.fa',gd/'ref'/f'{gd.name}_.bed']):genes.append(gd.name)
 t=time.time();stats=[];errors=[];tasks=[]
 for g in genes:
  cached=out/'_per_gene'/f'{g}.json'
  if args.reuse_valid_gene_outputs and cached.exists():stats.append(json.loads(cached.read_text()))
  else:tasks.append((g,str(src),str(out),panel))
 print('reused_valid_gene_outputs',len(stats),'genes_to_process',len(tasks),flush=True)
 print('eligible_genes',len(genes),'workers',args.workers,flush=True)
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(worker,t):t[0] for t in tasks}
  for i,f in enumerate(as_completed(futures),1):
   try:stats.append(f.result())
   except Exception as e:errors.append({'gene':futures[f],'error':repr(e),'traceback':traceback.format_exc()})
   if i%100==0 or i==len(tasks):print('completed',i,'/',len(tasks),'retained',sum(s.get('retained',0) for s in stats),'elapsed_s',round(time.time()-t,1),'errors',len(errors),flush=True)
 data=[pd.read_parquet(out/'_per_gene'/f"{s['gene']}.parquet") for s in stats if s.get('retained',0)>0];d=pd.concat(data,ignore_index=True)
 bad=d.groupby('variant_key').label.nunique();bad=set(bad[bad>1].index)
 opposing=d[d.variant_key.isin(bad)].copy();opposing.to_csv(out/'cross_gene_opposing_labels.csv',index=False);d=d[~d.variant_key.isin(bad)].sort_values(['gene','pos','ref','alt']).reset_index(drop=True)
 d.to_parquet(out/'full_cohort.parquet',index=False);d.to_csv(out/'full_cohort.csv',index=False)
 # Compile audit tables from per-gene outputs; the gene-level outputs remain provenance artifacts.
 rejected=[];conf=[];coverage=[]
 for s in stats:
  g=s['gene']
  for suffix,ls in [('excluded',rejected),('conflicts',conf)]:
   p=out/'_per_gene'/f'{g}.{suffix}.csv'
   if p.exists() and p.stat().st_size>1:
    try:ls.append(pd.read_csv(p))
    except pd.errors.EmptyDataError:pass
  for feat,v in s.get('feature_coverage',{}).items():coverage.append({'gene':g,'feature':feat,'n_variants_pre_global_conflicts':s['retained'],'coverage_pre_global_conflicts':v,'present_in_source_header':feat in s['panel_fields_in_header']})
 pd.concat(rejected,ignore_index=True).to_csv(out/'excluded_variants.csv',index=False) if rejected else pd.DataFrame(columns=['gene','variant_key','reason']).to_csv(out/'excluded_variants.csv',index=False)
 pd.concat(conf,ignore_index=True).to_csv(out/'within_gene_label_conflicts.csv',index=False) if conf else pd.DataFrame(columns=['gene','variant_key']).to_csv(out/'within_gene_label_conflicts.csv',index=False)
 pd.DataFrame(coverage).to_csv(out/'feature_coverage_by_gene.csv',index=False)
 pd.DataFrame([{k:v for k,v in s.items() if k in ['gene','status','raw_filter_rows','raw_labeled_rows','raw_feature_columns','retained','retained_by_label','n_conflicts']} for s in stats]).to_csv(out/'gene_counts.csv',index=False)
 (out/'source_hashes_and_gene_details.json').write_text(json.dumps(sorted(stats,key=lambda s:s['gene']),indent=2))
 (out/'errors.json').write_text(json.dumps(errors,indent=2))
 summary={'source':str(src),'scope':'New strict-missense full revision cohort; not the original manuscript experiment','n_eligible_genes':len(genes),'n_processed_genes':len(stats),'n_errors':len(errors),'status_counts':dict(collections.Counter(s['status'] for s in stats)),'n_raw_filter_rows':sum(s.get('raw_filter_rows',0) for s in stats),'n_raw_labeled_rows':sum(s.get('raw_labeled_rows',0) for s in stats),'n_retained_rows':len(d),'n_retained_unique_genomic_alleles':d.variant_key.nunique(),'n_retained_genes':d.gene.nunique(),'retained_by_label':d.label.value_counts().to_dict(),'n_cross_gene_opposing_keys':len(bad),'n_cross_gene_removed_rows':len(opposing),'n_features':len(panel),'feature_columns':panel,'excluded_reasons':dict(sum((collections.Counter(s.get('excluded_reasons',{})) for s in stats),collections.Counter())),'aa_exclusion_categories':dict(sum((collections.Counter(s.get('aa_exclusion_categories',{})) for s in stats),collections.Counter())),'elapsed_seconds':time.time()-t,'rules':{k:pmanifest[k] for k in ['predictor_aggregation','amino_acid_filter','duplicate_policy']},'no_model_preprocessing':'No split, imputation or scaling performed. Metadata are not model features. Fit all numeric preprocessing on root-defined training set only. Split by variant_key to keep duplicate genomic alleles together.'}
 (out/'full_cohort_manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
