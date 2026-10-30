#!/usr/bin/env python3
"""CPU-only summary of NEW matched five-gene fits; no historical exports read."""
from pathlib import Path
import argparse,hashlib,json,itertools
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,f1_score
SEEDS=[42,123,2026]
CONDITIONS=['full','score_only','sequence_only','full_normalized','full_native_scaled','permuted_cyclic','permuted_reverse','context_change','context_change_native_scaled','reduced','cadd','cadd_score_only','plus_alphamissense']
TRAINED=CONDITIONS+['logistic_regression','xgboost']
GENES=['BRCA1','BRCA2','DMD','PKD1','NF1']
METRICS=['AUROC','AUPRC','accuracy','balanced_accuracy','F1','macro_AUROC']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jwrite(p,obj):Path(p).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def metric(y,s):
 p=s>.5
 return dict(AUROC=roc_auc_score(y,s),AUPRC=average_precision_score(y,s),accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),F1=f1_score(y,p,zero_division=0))
def boot_indices(y,rng,B):
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
 return np.c_[rng.choice(pos,(B,len(pos)),replace=True),rng.choice(neg,(B,len(neg)),replace=True)]
def boot_auc(y,s,ix):
 labels=y[ix[0]];assert np.all(y[ix]==labels)
 np_=int(labels.sum());nn=len(labels)-np_
 ranks=rankdata(s[ix],method='average',axis=1)
 return ((ranks*labels).sum(axis=1)-np_*(np_+1)/2)/(np_*nn)
def ci(a):return list(np.quantile(a,[.025,.975]))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--cohort',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--bootstraps',type=int,default=2000);args=ap.parse_args()
 ip=Path(args.input_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 split=pd.read_csv(ip/'split_manifest.csv');raw=pd.read_parquet(args.cohort);assert raw.variant_key.is_unique and split.variant_key.is_unique
 full=split.merge(raw[['variant_key','gene','label']],on='variant_key',suffixes=('','_source'),validate='one_to_one');assert (full.gene==full.gene_source).all() and (full.label==full.label_source).all()
 test=full[full.split.eq('test')][['variant_id','variant_key','gene','label']].sort_values('variant_key').reset_index(drop=True);assert len(test)==445 and test.label.sum()==170
 y=test.label.to_numpy(dtype=int);genes=test.gene.to_numpy();keys=test.variant_key.tolist();gene_ix={g:np.flatnonzero(genes==g) for g in GENES}
 assert all(set(y[ix])=={0,1} for ix in gene_ix.values())
 train=full[full.split.eq('train')];rates=train.groupby('gene').label.mean();prior=test.gene.map(rates).to_numpy();assert np.isfinite(prior).all()
 priorpred=test.copy();priorpred['prob_pathogenic']=prior;priorpred.to_csv(out/'gene_prior_only_predictions.csv',index=False)
 rates.rename('train_P_fraction').to_frame().join(train.groupby('gene').label.agg(n_train='size',n_train_P='sum')).to_csv(out/'gene_prior_training_rates.csv')
 predictions={};alignment=[];provenance=[]
 for cond in TRAINED:
  for seed in SEEDS:
   p=ip/f'{cond}_seed{seed}_predictions.csv';assert p.exists(),str(p)
   d=pd.read_csv(p);assert d.variant_key.is_unique and len(d)==len(test) and set(d.variant_key)==set(keys)
   d=d.set_index('variant_key').loc[keys].reset_index()
   for c in ['variant_id','gene','label']:assert d[c].tolist()==test[c].tolist(),(cond,seed,c)
   s=d.prob_pathogenic.to_numpy(dtype=float);assert np.isfinite(s).all() and np.all((s>=0)&(s<=1))
   predictions[(cond,seed)]=s;alignment.append(dict(condition=cond,seed=seed,n=len(d),keys_match=True,ID_gene_labels_match=True,finite_probability=True))
   provenance.append(dict(file=str(p),sha256=sha(p)))
 predictions[('gene_prior_only',42)]=prior
 rng=np.random.default_rng(913);boot=boot_indices(y,rng,args.bootstraps)
 gene_boot={g:boot_indices(y[ix],rng,args.bootstraps) for g,ix in gene_ix.items()}
 rows=[];per=[];boots={};macroboots={}
 for (cond,seed),s in predictions.items():
  r=dict(condition=cond,seed=seed,n=len(test),P=int(y.sum()),B=int(len(y)-y.sum()),**metric(y,s));ga=[]
  for g,ix in gene_ix.items():
   z=dict(condition=cond,seed=seed,gene=g,n=len(ix),P=int(y[ix].sum()),B=int(len(ix)-y[ix].sum()),**metric(y[ix],s[ix]));ga.append(z['AUROC'])
   if seed==42:z['AUROC_CI_low'],z['AUROC_CI_high']=ci(boot_auc(y[ix],s[ix],gene_boot[g]))
   per.append(z)
  r['macro_AUROC']=float(np.mean(ga));rows.append(r)
  if seed==42:
   boots[cond]=boot_auc(y,s,boot)
   macroboots[cond]=np.mean([boot_auc(y[ix],s[ix],gene_boot[g]) for g,ix in gene_ix.items()],axis=0)
 detail=pd.DataFrame(rows);pg=pd.DataFrame(per);detail.to_csv(out/'condition_seed_metrics.csv',index=False);pg.to_csv(out/'per_gene_seed_metrics.csv',index=False)
 summaries=[]
 for cond,z in detail.groupby('condition',sort=False):
  r=dict(condition=cond,n_seeds=len(z),n=445,P=170,B=275)
  for m in METRICS:r[m+'_mean']=float(z[m].mean());r[m+'_SD']=float(z[m].std(ddof=1)) if len(z)>1 else 0.
  for m in ['AUROC','macro_AUROC']:
   r[m+'_seed42']=float(z.loc[z.seed.eq(42),m].iloc[0]);r[m+'_seed42_CI_low'],r[m+'_seed42_CI_high']=ci(boots[cond] if m=='AUROC' else macroboots[cond])
  summaries.append(r)
 summary=pd.DataFrame(summaries);summary.to_csv(out/'condition_summary.csv',index=False)
 # Untouched native CADD is distinct from a CADD-only CNN using gene-size geometry.
 cadd=raw.set_index('variant_key').loc[keys,'CADD_phred'].to_numpy(dtype=float);assert np.isfinite(cadd).all()
 native=[dict(scope='pooled',n=len(y),P=int(y.sum()),AUROC=float(roc_auc_score(y,cadd)),AUPRC=float(average_precision_score(y,cadd)),AUROC_CI_low=ci(boot_auc(y,cadd,boot))[0],AUROC_CI_high=ci(boot_auc(y,cadd,boot))[1])]
 for g,ix in gene_ix.items():native.append(dict(scope=g,n=len(ix),P=int(y[ix].sum()),AUROC=float(roc_auc_score(y[ix],cadd[ix])),AUPRC=float(average_precision_score(y[ix],cadd[ix]))))
 pd.DataFrame(native).to_csv(out/'native_CADD_phred_test_metrics.csv',index=False)

 gps=pg.groupby(['condition','gene'],sort=False).agg(n=('n','first'),P=('P','first'),B=('B','first'),n_seeds=('seed','size'),AUROC_mean=('AUROC','mean'),AUROC_SD=('AUROC','std'),AUPRC_mean=('AUPRC','mean'),AUPRC_SD=('AUPRC','std')).reset_index();gps[['AUROC_SD','AUPRC_SD']]=gps[['AUROC_SD','AUPRC_SD']].fillna(0);gps.to_csv(out/'per_gene_summary.csv',index=False)
 contrasts=[('full','score_only'),('sequence_only','gene_prior_only'),('full','reduced'),('full','cadd'),('full','cadd_score_only'),('cadd','cadd_score_only'),('plus_alphamissense','full'),('full_normalized','full'),('full_native_scaled','full'),('context_change','full'),('context_change_native_scaled','full_native_scaled'),('permuted_cyclic','full'),('permuted_reverse','full'),('logistic_regression','full'),('xgboost','full')]
 paired=[]
 for a,b in contrasts:
  ds=[];dms=[]
  for seed in SEEDS:
   aa=detail[(detail.condition==a)&(detail.seed==seed)].iloc[0]
   zz=detail[(detail.condition==b)&(detail.seed==(42 if b=='gene_prior_only' else seed))].iloc[0]
   ds.append(aa.AUROC-zz.AUROC);dms.append(aa.macro_AUROC-zz.macro_AUROC)
  rr=dict(condition=a,reference=b,delta_AUROC_mean=float(np.mean(ds)),delta_AUROC_SD=float(np.std(ds,ddof=1)),delta_macro_AUROC_mean=float(np.mean(dms)),delta_macro_AUROC_SD=float(np.std(dms,ddof=1)))
  rr['delta_AUROC_seed42']=float(detail[(detail.condition==a)&(detail.seed==42)].AUROC.iloc[0]-detail[(detail.condition==b)&(detail.seed==42)].AUROC.iloc[0]);rr['delta_AUROC_seed42_CI_low'],rr['delta_AUROC_seed42_CI_high']=ci(boots[a]-boots[b]);rr['delta_macro_AUROC_seed42_CI_low'],rr['delta_macro_AUROC_seed42_CI_high']=ci(macroboots[a]-macroboots[b]);paired.append(rr)
 pd.DataFrame(paired).to_csv(out/'paired_model_contrasts.csv',index=False);pd.DataFrame(alignment).to_csv(out/'paired_test_key_alignment.csv',index=False)
 # Recompute each saved CNN metrics JSON rather than treating summaries as evidence.
 checks=[]
 for cond in CONDITIONS:
  for seed in SEEDS:
   p=ip/f'{cond}_seed{seed}_metrics.json';old=json.loads(p.read_text());new=detail[(detail.condition==cond)&(detail.seed==seed)].iloc[0]
   err=max(abs(float(old[k])-float(new[k])) for k in ['AUROC','AUPRC','accuracy','balanced_accuracy','F1']);assert err<1e-10
   checks.append(dict(condition=cond,seed=seed,max_absolute_metric_difference=err));provenance.append(dict(file=str(p),sha256=sha(p)))
 pd.DataFrame(checks).to_csv(out/'saved_metrics_recomputed_check.csv',index=False)
 # Histories retain every seed's selected epoch; do not choose a favorable seed.
 hist=[]
 for cond in CONDITIONS:
  for seed in SEEDS:
   h=pd.read_csv(ip/f'{cond}_seed{seed}_history.csv');best=h.sort_values(['validation_AUROC','epoch'],ascending=[False,True]).iloc[0];m=json.loads((ip/f'{cond}_seed{seed}_metrics.json').read_text());assert int(best.epoch)==int(m['best_epoch'])
   hist.append(dict(condition=cond,seed=seed,best_epoch=int(best.epoch),best_validation_AUROC=float(best.validation_AUROC),n_epochs=len(h),runtime_seconds=m['runtime_seconds']))
 pd.DataFrame(hist).to_csv(out/'training_selection_by_seed.csv',index=False)
 # Encoding audits describe all 2224 cohort examples, not test-only examples.
 enc=[]
 for tag in ['base','native_scaled','change','change_native_scaled']:
  a=json.loads((ip/f'encoding_audit_{tag}.json').read_text());cache=np.load(ip/f'images_{tag}.npy',mmap_mode='r') if (ip/f'images_{tag}.npy').exists() else None
  for g in a['genes']:
   row=dict(encoding=tag,**g)
   if cache is not None and tag.startswith('change'):
    ix=raw.index[raw.gene.eq(g['gene'])].to_numpy();assert len(ix)==g['n'];zero=0
    for st in range(0,len(ix),64):zero+=int(np.count_nonzero(np.all(cache[ix[st:st+64],2]==0,axis=(1,2))))
    row['all_zero_change_mask_images']=zero;row['zero_change_mask_fraction']=zero/g['n']
   enc.append(row)
 pd.DataFrame(enc).to_csv(out/'encoding_quantization_by_gene.csv',index=False)
 corr=pd.read_csv(ip/'feature_pairwise_correlations.csv');corr.to_csv(out/'feature_pairwise_correlations.csv',index=False)
 # Readable submission supplement: point estimates/3seed SD and conditional seed42 CIs.
 pretty=[]
 for _,r in summary.iterrows():
  pretty.append(dict(Condition=r.condition,Test_N=445,P_LP=170,B_LB=275,Seeds=int(r.n_seeds),AUROC_mean_SD=f'{r.AUROC_mean:.4f} +/- {r.AUROC_SD:.4f}',AUROC_seed42_95CI=f'{r.AUROC_seed42:.4f} ({r.AUROC_seed42_CI_low:.4f}-{r.AUROC_seed42_CI_high:.4f})',Macro_AUROC_mean_SD=f'{r.macro_AUROC_mean:.4f} +/- {r.macro_AUROC_SD:.4f}',AUPRC_mean_SD=f'{r.AUPRC_mean:.4f} +/- {r.AUPRC_SD:.4f}',Accuracy_mean_SD=f'{r.accuracy_mean:.4f} +/- {r.accuracy_SD:.4f}'))
 pd.DataFrame(pretty).to_csv(out/'Supplementary_Table_diagnostic_controls.csv',index=False)
 jwrite(out/'analysis_manifest.json',dict(new_experiments_only=True,n_cohort=len(raw),n_train=int((full.split=='train').sum()),n_validation=int((full.split=='validation').sum()),n_test=445,P_test=170,B_test=275,conditional_genes=GENES,n_CNN_conditions=13,seeds=SEEDS,n_matched_models=len(predictions)-1,all_prediction_keys_individually_aligned=True,bootstrap=dict(repeats=args.bootstraps,seed=913,pooled='Case/class-stratified bootstrap of heldout variants for fixed seed42 predictions; fixed five observed genes, not gene generalization or refitting uncertainty',macro='Within each fixed gene, case/class-stratified bootstrap; unweighted mean of five gene AUROCs',paired='Same resampled variant indices for each contrasted model'),SD='Sample SD across all three training seeds; deterministic gene_prior_only has one fitted rate vector and no seed variation',score_only_geometry='Zero sequence channels still retain gene-size-specific native square and resize geometry; neither score-only nor CADD-only CNN is mathematically a pure tabular/single-variable function',input_provenance=provenance,script_sha256=sha(__file__),correlation_pairs_abs_rho_ge_095=int(((corr.Spearman.abs()>=.95)&(corr.n_pairwise>=20)).sum())))
 print(summary[['condition','n_seeds','AUROC_mean','AUROC_SD','macro_AUROC_mean','macro_AUROC_SD']].to_string(index=False),flush=True)
if __name__=='__main__':main()
