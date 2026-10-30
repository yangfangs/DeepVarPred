#!/usr/bin/env python3
"""Prepare current BRCA1 VUS inputs; no fitting, imputation, scaling or GPU work."""
from pathlib import Path
import importlib.util,json,hashlib,re
import numpy as np
import pandas as pd

ROOT=Path('/home/bio1/workdata/GeneMutation2')
REV=ROOT/'Revision_2026-10-26'
OUT=REV/'paired_analysis/current_BRCA1_VUS_input'
OUT.mkdir(parents=True,exist_ok=True)
HELPER=REV/'retraining/input/prep_five_gene_cohort.py'
spec=importlib.util.spec_from_file_location('frozen_cohort_helpers',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
manifest_path=REV/'retraining/full_input/full_cohort_manifest.json'
manifest=json.loads(manifest_path.read_text())
features=manifest['feature_columns']
assert len(features)==46 and len(set(features))==46
five_manifest_path=REV/'retraining/input/cohort_manifest.json'
assert features==json.loads(five_manifest_path.read_text())['recommended_fixed_panel']
gene='BRCA1';base=ROOT/'train_all'/gene
raw_path=base/'score/BRCA1_filter.txt'
ann_path=base/'score/BRCA1_clnsig.txt'
bed_path=base/'ref/BRCA1_.bed';fasta_path=base/'ref/BRCA1_ref.fa'
binary_path=REV/'retraining/full_input/full_cohort.parquet'
inputs=[raw_path,ann_path,bed_path,fasta_path,HELPER,manifest_path,five_manifest_path,binary_path]
sha_before={str(p):helper.sha256(p) for p in inputs}
d=pd.read_csv(raw_path,sep='\t',dtype=str,low_memory=False).fillna('')
c=pd.read_csv(ann_path,sep='\t',dtype=str,low_memory=False).fillna('')
for z in [d,c]:z['_key']=z.apply(helper.variant_key,axis=1)
c['_aa_ok']=[a in helper.AA20 and b in helper.AA20 and a!=b for a,b in zip(c.aaref,c.aaalt)]
c['_binary_label']=c.clinvar_clnsig.map(helper.LABELS)
d['_binary_label']=d.clinvar_clnsig.map(helper.LABELS)
anno={}
for key,z in c.groupby('_key',sort=False):
 anno[key]={'aa_ok':bool(z._aa_ok.all()),'labels':set(z._binary_label.dropna().astype(int)),
  'all_clinvar_labels':';'.join(sorted(set(z.clinvar_clnsig))),
  'aaref':';'.join(sorted(set(z.aaref))),'aaalt':';'.join(sorted(set(z.aaalt))),
  'ensembl_transcriptids':';'.join(sorted(set(z.get('Ensembl_transcriptid',pd.Series(dtype=str)).astype(str)))),
  'clinvar_ids':';'.join(sorted(set(z.get('clinvar_id',pd.Series(dtype=str)).astype(str)))),
  'n_annotation_rows':len(z)}
raw_binary_keys=set(d.loc[d._binary_label.notna(),'_key'])
all_binary=pd.read_parquet(binary_path,columns=['variant_key','label'])
binary_keys=set(all_binary.variant_key)
bed=bed_path.read_text().strip().split('\t');chrom=helper.chr_clean(bed[0]);start,end=map(int,bed[1:3])
sequence=''.join(x.strip() for x in fasta_path.read_text().splitlines() if not x.startswith('>')).upper()
assert len(sequence)==end-start
vu=d[d.clinvar_clnsig.eq('Uncertain_significance')].copy()
rows=[];excluded=[]
for key,z in vu.groupby('_key',sort=False,dropna=False):
 first=z.iloc[0];a=anno.get(key);reason=None
 try:
  chr_=helper.chr_clean(first.hg19_chr);pos=int(float(first['hg19_pos(1-based)']));ref=str(first.ref).upper();alt=str(first.alt).upper()
 except (ValueError,TypeError):reason='invalid_genomic_key'
 if not reason and (key is None or chr_ not in [str(i) for i in range(1,23)]+['X','Y','M','MT'] or pos<=0):reason='invalid_genomic_key'
 if not reason and a is None:reason='no_amino_acid_annotation'
 if not reason and (key in raw_binary_keys or a['labels']):reason='binary_annotation_for_same_genomic_key'
 if not reason and key in binary_keys:reason='overlap_with_frozen_full_binary_cohort'
 if not reason and not a['aa_ok']:reason='non_missense_or_mixed_amino_acid_consequence'
 if not reason and (len(ref)!=1 or len(alt)!=1 or ref not in 'ACGT' or alt not in 'ACGT' or ref==alt):reason='not_single_nucleotide_substitution'
 idx=pos-(start+1) if not reason else None
 if not reason and (chr_!=chrom or not 0<=idx<len(sequence)):reason='outside_fasta_interval'
 if not reason and sequence[idx]!=ref:reason='reference_base_mismatch'
 if reason:
  excluded.append({'gene':gene,'variant_key':key,'reason':reason,'n_filter_rows':len(z),'source_clinvar_labels':a['all_clinvar_labels'] if a else '', 'fasta_base':sequence[idx] if idx is not None and 0<=idx<len(sequence) else ''})
  continue
 row={'variant_id':gene+':'+key,'variant_key':key,'gene':gene,'chr':chrom,'pos':pos,'ref':ref,'alt':alt,
  'label':np.nan,'clinvar_clnsig':'Uncertain_significance','all_source_clinvar_labels':a['all_clinvar_labels'],
  'aaref':a['aaref'],'aaalt':a['aaalt'],'ensembl_transcriptids':a['ensembl_transcriptids'],'clinvar_ids':a['clinvar_ids'],
  'n_filter_rows':len(z),'n_annotation_rows':a['n_annotation_rows'],'sequence_index_0based':idx,
  'reference_match':True,'binary_cohort_overlap':False,'genome_build':'GRCh37/hg19'}
 for col in features:
  if col not in z:row[col]=np.nan;continue
  vals=[helper.numeric_aggregate(v) for v in z[col]];vals=[v for v in vals if np.isfinite(v)]
  row[col]=float(np.mean(vals)) if vals else np.nan
 rows.append(row)
result=pd.DataFrame(rows).sort_values(['gene','pos','ref','alt']).reset_index(drop=True)
assert result.variant_key.is_unique
assert not set(result.variant_key)&binary_keys
assert result.reference_match.all()
assert result.label.isna().all()
assert result[features].shape[1]==46
metadata=[x for x in result.columns if x not in features]
result=result[metadata+features]
result.to_parquet(OUT/'current_BRCA1_strict_VUS_fixed46.parquet',index=False)
result.to_csv(OUT/'current_BRCA1_strict_VUS_fixed46.csv',index=False)
coverage=pd.DataFrame([{'feature':f,'source_column_present':f in d.columns,'n_variants':len(result),'n_observed':int(result[f].notna().sum()),'n_missing':int(result[f].isna().sum()),'observed_fraction':float(result[f].notna().mean())} for f in features])
coverage.to_csv(OUT/'feature_coverage.csv',index=False)
pd.DataFrame(excluded,columns=['gene','variant_key','reason','n_filter_rows','source_clinvar_labels','fasta_base']).to_csv(OUT/'excluded_variants.csv',index=False)
source_manifest=[]
for p in inputs:
 after=helper.sha256(p);assert after==sha_before[str(p)],str(p)+' changed during preparation'
 source_manifest.append({'path':str(p),'bytes':p.stat().st_size,'mtime_utc':pd.Timestamp(p.stat().st_mtime,unit='s',tz='UTC').isoformat(),'sha256':after})
pd.DataFrame(source_manifest).to_csv(OUT/'source_manifest.csv',index=False)
summary={'date':'2026-10-26','scope':'Current GeneMutation2 BRCA1 strict VUS inference inputs; separate from historical 2583-VUS analysis; not a training or evaluation set with known binary outcomes',
 'gene':gene,'raw_filter_rows':len(d),'raw_annotation_rows':len(c),'raw_vus_rows':len(vu),'raw_vus_unique_keys':int(vu._key.nunique()),
 'retained_unique_vus':len(result),'retained_vus_from_duplicate_source_rows':int(result.n_filter_rows.gt(1).sum()),
 'excluded_keys':len(excluded),'excluded_source_vus_rows':int(sum(x['n_filter_rows'] for x in excluded)),
 'exclusions_by_reason_keys':pd.Series([x['reason'] for x in excluded],dtype=str).value_counts().to_dict(),
 'exclusions_by_reason_rows':pd.DataFrame(excluded).groupby('reason').n_filter_rows.sum().to_dict() if excluded else {},
 'feature_count':len(features),'feature_columns':features,'features_absent_from_raw_filter':[f for f in features if f not in d.columns],
 'raw_aggregation':'Imported unchanged numeric_aggregate from prep_five_gene_cohort.py: mean finite tokens split on semicolon, pipe, colon, then mean over duplicate genomic rows. No cross-variant fill, no scaling, no train/val/test assignment.',
 'aa_rule':'All source _clnsig annotation rows for the genomic key must encode distinct aaref/aaalt each among the canonical 20 amino acids. No canonical/MANE transcript is chosen. Transcript IDs are preserved as metadata.',
 'label_rule':'Source _filter record must be Uncertain_significance. Exclude a key if ANY raw-filter or annotation record assigns P/LP or B/LB; also exclude overlap with the frozen full binary cohort across all genes. Other nonbinary labels are preserved in all_source_clinvar_labels.',
 'n_retained_with_non_VUS_nonbinary_annotation':int(result.all_source_clinvar_labels.ne('Uncertain_significance').sum()),
 'fasta_reference_validation':{'chromosome':chrom,'bed_start_0based':start,'bed_end_exclusive':end,'sequence_length':len(sequence),'bed_length':end-start,'all_retained_ref_match':bool(result.reference_match.all()),'variant_offset':'one_based_position - (BED_start + 1)'},
 'binary_label_column':'All NaN by design; VUS are never assigned pathogenic/benign ground truth.',
 'binary_cohort_overlap_n':len(set(result.variant_key)&binary_keys),'recommended_inference':'Use only the 46 feature columns in the exact stored order and the trained primary model train-only imputer/scaler; never fit them on these VUS. Metadata and missing binary labels are not predictor features.',
 'original_sources_unchanged':True,'helper_sha256':sha_before[str(HELPER)],'full_cohort_manifest_sha256':sha_before[str(manifest_path)],'source_snapshot_note':'Source file modification dates and exact content checksums are recorded in source_manifest.csv; they are not inferred database release dates.'}
(OUT/'input_manifest.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
