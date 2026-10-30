#!/usr/bin/env python3
"""Build a new five-gene revision cohort from untouched raw source tables.
No label-conditioned fill, scaling, split, or model fitting is performed.
"""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
import pandas as pd

GENES=['BRCA1','BRCA2','DMD','PKD1','NF1']
AA20=set('ACDEFGHIKLMNPQRSTVWY')
LABELS={'Pathogenic':1,'Likely_pathogenic':1,'Pathogenic/Likely_pathogenic':1,
        'Benign':0,'Likely_benign':0,'Benign/Likely_benign':0}
INFO={'#chr','pos(1-based)','ref','alt','hg19_chr','hg19_pos(1-based)','hg18_chr','hg18_pos(1-based)','clinvar_clnsig','clinvar_Orphanet_id','clinvar_id','clinvar_OMIM_id'}

def chr_clean(x):
 return str(x).removeprefix('chr').removesuffix('.0')
def variant_key(row):
 try:return chr_clean(row['hg19_chr'])+':'+str(int(float(row['hg19_pos(1-based)'])))+':'+str(row['ref']).upper()+'>'+str(row['alt']).upper()
 except (TypeError,ValueError):return None

def numeric_aggregate(x):
 """Mean of all finite numeric tokens across transcript entries; missing remains NaN."""
 vals=[]
 for token in re.split(r'[;|:]',str(x)):
  try:
   f=float(token.strip())
   if np.isfinite(f):vals.append(f)
  except ValueError:pass
 return float(np.mean(vals)) if vals else np.nan

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='/home/bio1/workdata/GeneMutation2/train_all');ap.add_argument('--output',required=True);ap.add_argument('--described_features',required=True);args=ap.parse_args()
 src=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 desc=json.loads(Path(args.described_features).read_text())['feature_list']
 desc=[c.replace('M.CAP','M-CAP').replace('LIST.S2','LIST-S2') for c in desc if c!='Reliability_index']
 raw={};cols_by_gene={};sources={};counts={};rejected=[];conflicts=[];all_rows=[];bed_manifest={}
 for gene in GENES:
  folder=src/gene;fp=folder/'score'/f'{gene}_filter.txt';cp=folder/'score'/f'{gene}_clnsig.txt';bp=folder/'ref'/f'{gene}_.bed';fa=folder/'ref'/f'{gene}_ref.fa'
  d=pd.read_csv(fp,sep='\t',dtype=str,low_memory=False).fillna('');c=pd.read_csv(cp,sep='\t',dtype=str,low_memory=False).fillna('')
  cols_by_gene[gene]=[k for k in d.columns if k not in INFO]
  sources[gene]={str(p):sha256(p) for p in [fp,cp,bp,fa]}
  for df in [d,c]:df['_key']=df.apply(variant_key,axis=1)
  # Source annotation is evaluated before any predictor aggregation.
  c['_strict_missense']=[a in AA20 and b in AA20 and a!=b for a,b in zip(c.aaref,c.aaalt)]
  c['_binary_label']=c.clinvar_clnsig.map(LABELS)
  groups=c.groupby('_key',sort=False)
  by_key={}
  for key,z in groups:
   ls=set(z['_binary_label'].dropna().astype(int))
   by_key[key]={'aa_ok':bool(z['_strict_missense'].all()),'has_missense':bool(z['_strict_missense'].any()),'labels':ls,
     'aaref':';'.join(sorted(set(z.aaref))), 'aaalt':';'.join(sorted(set(z.aaalt))),
     'ensembl_transcriptids':';'.join(sorted(set(z.get('Ensembl_transcriptid',pd.Series(dtype=str)).astype(str)))),
     'clinvar_ids':';'.join(sorted(set(z.get('clinvar_id',pd.Series(dtype=str)).astype(str)))),'n_annotation_rows':len(z)}
  bed=bp.read_text().strip().split('\t');chrom=chr_clean(bed[0]);start,end=map(int,bed[1:3]);seq=''.join(l.strip() for l in fa.read_text().splitlines() if not l.startswith('>')).upper()
  bed_manifest[gene]={'chr':chrom,'bed_start_0based':start,'bed_end_exclusive':end,'sequence_length':len(seq),'bed_length':end-start,'fasta_path':str(fa),'bed_path':str(bp),'fasta_matches_bed_length':len(seq)==end-start}
  assert len(seq)==end-start,(gene,'FASTA/BED length mismatch')
  d['_label']=d.clinvar_clnsig.map(LABELS);counts[gene]={'raw_filter_rows':len(d),'raw_labeled_rows':int(d['_label'].notna().sum()),'raw_feature_columns':len(cols_by_gene[gene])}
  for key,z in d[d['_label'].notna()].groupby('_key',sort=False):
   first=z.iloc[0];ls=set(z['_label'].astype(int));a=by_key.get(key)
   reason=None
   if a is None:reason='no_amino_acid_annotation'
   elif len(ls)>1 or len(a['labels'])>1 or (a['labels'] and a['labels']!=ls):reason='opposing_binary_labels'
   elif not a['aa_ok']:reason='non_missense_or_mixed_amino_acid_consequence'
   elif str(first['ref']).upper() not in 'ACGT' or len(str(first['ref']))!=1 or str(first['alt']).upper() not in 'ACGT' or len(str(first['alt']))!=1 or first['ref']==first['alt']:reason='not_single_nucleotide_substitution'
   if reason:
    rejected.append({'gene':gene,'variant_key':key,'reason':reason,'n_filter_rows':len(z)})
    if reason=='opposing_binary_labels':conflicts.append({'gene':gene,'variant_key':key,'filter_labels':str(sorted(ls)),'annotation_labels':str(sorted(a['labels']))})
    continue
   pos=int(float(first['hg19_pos(1-based)']));ref=str(first['ref']).upper();alt=str(first['alt']).upper();idx=pos-(start+1)
   if chr_clean(first['hg19_chr'])!=chrom or not (0<=idx<len(seq)):reason='outside_fasta_interval'
   elif seq[idx]!=ref:reason='reference_base_mismatch'
   if reason:
    rejected.append({'gene':gene,'variant_key':key,'reason':reason,'n_filter_rows':len(z),'fasta_base':seq[idx] if 0<=idx<len(seq) else ''});continue
   row={'variant_id':gene+':'+key,'variant_key':key,'gene':gene,'chr':chrom,'pos':pos,'ref':ref,'alt':alt,'label':int(next(iter(ls))),
    'clinvar_clnsig':';'.join(sorted(set(z.clinvar_clnsig))), 'aaref':a['aaref'],'aaalt':a['aaalt'],'ensembl_transcriptids':a['ensembl_transcriptids'],'clinvar_ids':a['clinvar_ids'],
    'n_filter_rows':len(z),'n_annotation_rows':a['n_annotation_rows'],'sequence_index_0based':idx,'reference_match':True}
   for col in cols_by_gene[gene]:
    vals=[numeric_aggregate(v) for v in z[col]];vals=[v for v in vals if np.isfinite(v)];row[col]=float(np.mean(vals)) if vals else np.nan
   all_rows.append(row)
  print(gene,counts[gene],flush=True)
 all_df=pd.DataFrame(all_rows)
 # A genomic allele mapped to more than one gene must have one consistent binary label.
 bad=all_df.groupby('variant_key').label.nunique();bad=set(bad[bad>1].index)
 if bad:
  for key in sorted(bad):conflicts.append({'gene':'CROSS_GENE','variant_key':key,'filter_labels':'0;1','annotation_labels':'0;1'})
  all_df=all_df[~all_df.variant_key.isin(bad)].copy()
 union=sorted(set().union(*(set(v) for v in cols_by_gene.values())))
 inter=sorted(set.intersection(*(set(v) for v in cols_by_gene.values())))
 panel=[c for c in desc if c in union]
 metadata=[c for c in all_df.columns if c not in union]
 for c in union:
  if c not in all_df:all_df[c]=np.nan
 # Keep a raw union table and a separate recommended fixed panel. No imputing or scaling.
 all_df=all_df[metadata+union].sort_values(['gene','pos','ref','alt']).reset_index(drop=True)
 all_df.to_parquet(out/'five_gene_raw_union.parquet',index=False)
 all_df.to_csv(out/'five_gene_raw_union.csv',index=False)
 all_df[metadata+panel].to_parquet(out/'five_gene_fixed_panel.parquet',index=False)
 all_df[metadata+panel].to_csv(out/'five_gene_fixed_panel.csv',index=False)
 pd.DataFrame(rejected).to_csv(out/'excluded_variants.csv',index=False)
 pd.DataFrame(conflicts,columns=['gene','variant_key','filter_labels','annotation_labels']).to_csv(out/'label_conflicts.csv',index=False)
 coverage=[]
 for gene,z in all_df.groupby('gene'):
  for c in panel:coverage.append({'gene':gene,'feature':c,'n_variants':len(z),'n_observed':int(z[c].notna().sum()),'coverage':float(z[c].notna().mean()),'present_in_original_filtered_header':c in cols_by_gene[gene]})
 pd.DataFrame(coverage).to_csv(out/'feature_coverage_by_gene.csv',index=False)
 manifest={'source':'GeneMutation2/train_all raw _filter.txt with amino-acid annotations joined from untouched _clnsig.txt','scope':'New five-gene revision cohort; not a reconstruction of the original fixed-56 experiment','genes':GENES,'counts_before_filtering':counts,'n_retained':len(all_df),'retained_by_gene_label':all_df.groupby(['gene','label']).size().reset_index(name='n').to_dict('records'),'exclusion_reasons':pd.Series([r['reason'] for r in rejected]).value_counts().to_dict(),'n_label_conflicts':len(conflicts),'raw_union_n':len(union),'raw_union_columns':union,'raw_intersection_n':len(inter),'raw_intersection_columns':inter,'described_list_without_Reliability_n':len(desc),'recommended_fixed_panel_n':len(panel),'recommended_fixed_panel':panel,'described_columns_absent_from_all_five':list(c for c in desc if c not in union),'predictor_aggregation':'Mean of finite numeric tokens split on semicolon, pipe or colon; then mean across duplicate genomic rows. No label-conditioned imputation; unavailable values remain NaN. No scaling. No split.','amino_acid_filter':'All source annotation rows for a genomic allele must have distinct aaref and aaalt each in the canonical 20 amino acids. Mixed missense/nonmissense consequences are conservatively excluded. No canonical transcript is inferred.','duplicate_policy':'Deduplicate genomic chr/pos/ref/alt within each gene; drop any opposing P/LP and B/LB annotations within or across genes. Preserve variant_key for root to split by genomic allele.','reference_validation':bed_manifest,'source_sha256':sources}
 (out/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_sha256','raw_union_columns','reference_validation','counts_before_filtering']},indent=2),flush=True)
if __name__=='__main__':main()
