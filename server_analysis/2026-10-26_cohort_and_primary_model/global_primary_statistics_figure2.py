#!/usr/bin/env python3
"""CPU statistics for the completed, frozen primary run; Python-only Figure 2.
Contract: quantify held-out-gene discrimination, gene case-study variability, and
matched native-score comparisons, without presupposing superiority.
Primary panels: a ROC, b PR. Supporting: c five prespecified genes, d canonical
CNN-native paired delta AUROC. 180 x 230 mm, editable SVG + PDF + 300dpi PNG.
No historical score files, model fitting, CUDA, or input mutations.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
SEED=20261026
MODELS=['CNN','Logistic regression','XGBoost']
PRED_FILES={'CNN':'cnn_full_seed42_predictions.csv','Logistic regression':'logistic_regression_seed42_predictions.csv','XGBoost':'xgboost_seed42_predictions.csv'}
FIVE=['BRCA1','BRCA2','DMD','PKD1','NF1']
CANONICAL=['SIFT_score','SIFT4G_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','MutationTaster_score','MutationAssessor_score','PROVEAN_converted_rankscore','VEST4_score','MetaSVM_rankscore','MetaLR_score','MetaRNN_score','M-CAP_score','REVEL_score','MutPred_score','MVP_score','MPC_score','PrimateAI_score','DEOGEN2_score','BayesDel_addAF_score','BayesDel_noAF_rankscore','ClinPred_score','LIST-S2_score','CADD_phred','DANN_score']
LOWER={'SIFT_score','SIFT4G_score'}
ALIASES=['SIFT','SIFT4G','PolyPhen2 HDIV','PolyPhen2 HVAR','MutationTaster','MutationAssessor','PROVEAN (rank)','VEST4','MetaSVM (rank)','MetaLR','MetaRNN','M-CAP','REVEL','MutPred','MVP','MPC','PrimateAI','DEOGEN2','BayesDel (+AF)','BayesDel (noAF rank)','ClinPred','LIST-S2','CADD (PHRED)','DANN']
COLORS={'CNN':'#0072B2','Logistic regression':'#7A8899','XGBoost':'#35BFC6'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,o):
 Path(p).write_text(json.dumps(o,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def ci(a):
 a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
 return [float(x) for x in np.quantile(a,[.025,.975])] if len(a) else [None,None]
def ordered(y,s):
 s=np.asarray(s,dtype=float);y=np.asarray(y,dtype=int);ix=np.argsort(s,kind='mergesort');ss=s[ix]
 starts=np.r_[0,1+np.flatnonzero(ss[1:]!=ss[:-1])]
 return ix,y[ix],starts

def weighted_metrics(prepared,w):
 ix,y,starts=prepared;ww=np.asarray(w)[ix]
 p=np.add.reduceat(ww*y,starts);n=np.add.reduceat(ww*(1-y),starts);P=p.sum();N=n.sum()
 if P<=0 or N<=0:return np.nan,np.nan
 auc=np.sum(p*(np.cumsum(n)-.5*n))/(P*N)
 pd=p[::-1];nd=n[::-1];pc=np.cumsum(pd);nc=np.cumsum(nd)
 den=pc+nc;precision=np.divide(pc,den,out=np.zeros_like(pc,dtype=float),where=den>0)
 ap=np.sum(pd*precision)/P
 return float(auc),float(ap)

def check_weighted_algorithm():
 from sklearn.metrics import roc_auc_score,average_precision_score
 r=np.random.default_rng(72)
 for rep in range(10):
  y=np.r_[np.zeros(13,dtype=int),np.ones(17,dtype=int)];s=np.round(r.uniform(size=30),1);w=r.integers(0,5,size=30)
  a,b=weighted_metrics(ordered(y,s),w)
  assert abs(a-roc_auc_score(y,s,sample_weight=w))<1e-12
  assert abs(b-average_precision_score(y,s,sample_weight=w))<1e-12

def compute(ip,out,B):
 from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,accuracy_score,balanced_accuracy_score,f1_score
 started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
 assert (ip/'training_complete.json').exists(),'Final training_complete.json required before any analysis'
 done=json.loads((ip/'training_complete.json').read_text());assert done['complete'] and done['seed']==42
 prep=json.loads((ip/'preparation_complete.json').read_text());pre=json.loads((ip/'preprocessing.json').read_text());cfg=json.loads((ip/'training_config.json').read_text())
 assert cfg['seed']==42 and done['n_test']==29364
 for name,h in done['artifact_sha256'].items():assert sha(ip/name)==h,name
 assert sha(ip/'analysis_input.parquet')==prep['artifact_sha256']['analysis_input.parquet']
 df=pd.read_parquet(ip/'analysis_input.parquet');test=df[df.split.eq('test')].copy().reset_index(drop=True);assert len(test)==29364 and test.gene.nunique()==737 and test.variant_key.is_unique
 assert int(test.label.sum())==11374 and int((test.label==0).sum())==17990
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  assert not set(df[df.split.eq(a)].gene)&set(df[df.split.eq(b)].gene)
  assert not set(df[df.split.eq(a)].variant_key)&set(df[df.split.eq(b)].variant_key)
 features=pre['feature_columns'];assert len(features)==46 and set(CANONICAL).issubset(features)
 y=test.label.to_numpy(dtype=int);scores={};sources={};alignment=[];aligned=test[['variant_id','variant_key','gene','label']].copy()
 for model,file in PRED_FILES.items():
  p=ip/file;pr=pd.read_csv(p);assert len(pr)==len(test) and pr.variant_key.is_unique and set(pr.variant_key)==set(test.variant_key)
  q=pr.set_index('variant_key').loc[test.variant_key]
  for key in ['variant_id','gene','label']:assert q[key].astype(str).tolist()==test[key].astype(str).tolist(),(model,key)
  s=q.prob_pathogenic.to_numpy(dtype=float);assert np.isfinite(s).all() and (s>=0).all() and (s<=1).all()
  scores[model]=s;aligned[model]=s;sources[file]=sha(p);alignment.append(dict(model=model,n=len(pr),P=int(pr.label.sum()),same_test_keys_labels=True,source_SHA256=sources[file]))
 aligned.to_csv(out/'aligned_primary_test_predictions.csv',index=False);pd.DataFrame(alignment).to_csv(out/'prediction_alignment_audit.csv',index=False)
 check_weighted_algorithm()
 genes=np.array(sorted(test.gene.unique()),dtype=str);gene_index=pd.Categorical(test.gene,categories=genes).codes;G=len(genes)
 rng=np.random.default_rng(SEED);draws=np.array([np.bincount(rng.integers(0,G,size=G),minlength=G) for _ in range(B)],dtype=np.uint16)
 np.savez_compressed(out/'heldout_gene_bootstrap_draws.npz',gene_order=genes,multiplicities=draws)
 preparations={m:ordered(y,scores[m]) for m in MODELS};boot={m:np.empty((B,2)) for m in MODELS}
 for b,counts in enumerate(draws):
  w=counts[gene_index]
  for m in MODELS:boot[m][b]=weighted_metrics(preparations[m],w)
 primary=[];model_summary={}
 for model in MODELS:
  s=scores[model];a=float(roc_auc_score(y,s));ap=float(average_precision_score(y,s));ac=ci(boot[model][:,0]);pc=ci(boot[model][:,1]);valid=int(np.isfinite(boot[model][:,0]).sum())
  row=dict(model=model,n=len(y),P=int(y.sum()),B=int(len(y)-y.sum()),genes=G,training_seed=42,AUROC=a,AUROC_CI_lower=ac[0],AUROC_CI_upper=ac[1],AUPRC=ap,AUPRC_CI_lower=pc[0],AUPRC_CI_upper=pc[1],accuracy=float(accuracy_score(y,s>.5)),balanced_accuracy=float(balanced_accuracy_score(y,s>.5)),F1=float(f1_score(y,s>.5)),bootstrap_repeats=B,valid_bootstrap_repeats=valid,bootstrap_unit='held-out gene cluster')
  primary.append(row);model_summary[model]=row
 pd.DataFrame(primary).to_csv(out/'primary_models_gene_cluster_CI.csv',index=False)
 contrasts=[]
 for a,b in [('CNN','Logistic regression'),('CNN','XGBoost'),('Logistic regression','XGBoost')]:
  for j,metric in enumerate(['AUROC','AUPRC']):
   d=boot[a][:,j]-boot[b][:,j];interval=ci(d);contrasts.append(dict(model_A=a,model_B=b,metric=metric,delta=model_summary[a][metric]-model_summary[b][metric],CI_lower=interval[0],CI_upper=interval[1],bootstrap_repeats=B,unit='paired held-out gene clusters'))
 pd.DataFrame(contrasts).to_csv(out/'primary_model_paired_differences.csv',index=False)
 curves=[];prcurves=[]
 for m in MODELS:
  fpr,tpr,thr=roc_curve(y,scores[m]);curves.extend(dict(model=m,FPR=x,TPR=z) for x,z in zip(fpr,tpr))
  precision,recall,thr=precision_recall_curve(y,scores[m],drop_intermediate=True);prcurves.extend(dict(model=m,recall=x,precision=z) for x,z in zip(recall,precision))
 pd.DataFrame(curves).to_csv(out/'primary_ROC_coordinates.csv',index=False);pd.DataFrame(prcurves).to_csv(out/'primary_PR_coordinates.csv',index=False)
 # The fixed five genes were wholly absent from fitting. Their intervals are
 # case-stratified within each gene, and distinct from the global gene bootstrap.
 per_gene=[];five_summary={}
 for gi,gene in enumerate(FIVE):
  mask=test.gene.eq(gene).to_numpy();yy=y[mask];pos=np.flatnonzero(yy==1);neg=np.flatnonzero(yy==0);n=len(yy);both=len(pos)>0 and len(neg)>0
  pp={m:ordered(yy,scores[m][mask]) for m in MODELS};gb={m:np.empty((B,2)) for m in MODELS};grng=np.random.default_rng(SEED+100000+gi)
  if both:
   for b in range(B):
    chosen=np.r_[grng.choice(pos,len(pos),replace=True),grng.choice(neg,len(neg),replace=True)];w=np.bincount(chosen,minlength=n)
    for m in MODELS:gb[m][b]=weighted_metrics(pp[m],w)
  for m in MODELS:
   ac=ci(gb[m][:,0]) if both else [None,None];pc=ci(gb[m][:,1]) if both else [None,None]
   row=dict(gene=gene,model=m,n=n,P=len(pos),B=len(neg),AUROC=float(roc_auc_score(yy,scores[m][mask])) if both else None,AUROC_CI_lower=ac[0],AUROC_CI_upper=ac[1],AUPRC=float(average_precision_score(yy,scores[m][mask])) if len(pos) else None,AUPRC_CI_lower=pc[0],AUPRC_CI_upper=pc[1],bootstrap_repeats=B if both else 0,bootstrap_seed=SEED+100000+gi,bootstrap_unit='cases stratified by binary label within the fixed gene')
   per_gene.append(row);five_summary.setdefault(gene,{})[m]=row
 pd.DataFrame(per_gene).to_csv(out/'five_heldout_genes_metrics_CI.csv',index=False)
 # Descriptive per-gene metrics include single-class genes as non-estimable AUC.
 allgene=[]
 for g,ix in test.groupby('gene').groups.items():
  ix=np.asarray(ix);yy=y[ix];both=len(np.unique(yy))==2
  for m in MODELS:allgene.append(dict(gene=g,model=m,n=len(ix),P=int(yy.sum()),B=int(len(ix)-yy.sum()),AUROC=float(roc_auc_score(yy,scores[m][ix])) if both else None,AUPRC=float(average_precision_score(yy,scores[m][ix])) if np.any(yy==1) else None))
 pd.DataFrame(allgene).to_csv(out/'all_heldout_genes_descriptive_metrics.csv',index=False)
 # Native fields are never imputed or flipped based on observed performance.
 # Raw AUC remains available; only prespecified SIFT/SIFT4G get sign reversal.
 native=[]
 for feature in features:
  raw=pd.to_numeric(test[feature],errors='coerce').to_numpy(dtype=float);ok=np.isfinite(raw);yy=y[ok];both=len(np.unique(yy))==2;direction='lower' if feature in LOWER else 'higher';oriented=-raw[ok] if direction=='lower' else raw[ok]
  row=dict(feature=feature,canonical=feature in CANONICAL,n_pairwise=int(ok.sum()),P=int(yy.sum()),B=int(len(yy)-yy.sum()),genes_with_finite_score=int(test.loc[ok,'gene'].nunique()),bootstrap_population_genes=G,native_direction=direction,native_AUROC_raw=float(roc_auc_score(yy,raw[ok])) if both else None,native_AUROC_direction_corrected=float(roc_auc_score(yy,oriented)) if both else None)
  for m in MODELS:row[m+'_AUROC_same_variants']=float(roc_auc_score(yy,scores[m][ok])) if both else None
  if both and feature in CANONICAL:
   nprep=ordered(yy,oriented);cprep=ordered(yy,scores['CNN'][ok]);nb=np.empty(B);cb=np.empty(B);idx=gene_index[ok]
   for b,counts in enumerate(draws):
    w=counts[idx];nb[b]=weighted_metrics(nprep,w)[0];cb[b]=weighted_metrics(cprep,w)[0]
   nc=ci(nb);cc=ci(cb);dc=ci(cb-nb);row.update(native_AUROC_CI_lower=nc[0],native_AUROC_CI_upper=nc[1],CNN_AUROC_CI_lower=cc[0],CNN_AUROC_CI_upper=cc[1],CNN_minus_native_AUROC=row['CNN_AUROC_same_variants']-row['native_AUROC_direction_corrected'],delta_AUROC_CI_lower=dc[0],delta_AUROC_CI_upper=dc[1],bootstrap_repeats=B,valid_bootstrap_repeats=int(np.isfinite(cb-nb).sum()))
   print(json.dumps(dict(stage='canonical_pairwise',feature=feature,n=row['n_pairwise'])),flush=True)
  native.append(row)
 pd.DataFrame(native).to_csv(out/'native_predictors_pairwise_with_cluster_CI.csv',index=False)
 # Recheck frozen runner's independent pairwise point table.
 old=pd.read_csv(ip/'pairwise_native_predictor_comparison.csv').set_index('feature');new=pd.DataFrame(native).set_index('feature')
 for f in features:
  assert int(old.loc[f,'n_pairwise'])==int(new.loc[f,'n_pairwise'])
  for k in ['native_AUROC_raw','native_AUROC_direction_corrected','CNN_AUROC_same_variants']:
   assert abs(old.loc[f,k]-new.loc[f,k])<1e-12,(f,k)
  for oldk,newk in [('logistic_regression_AUROC_same_variants','Logistic regression_AUROC_same_variants'),('xgboost_AUROC_same_variants','XGBoost_AUROC_same_variants')]:assert abs(old.loc[f,oldk]-new.loc[f,newk])<1e-12,(f,oldk)
 policy=dict(repeats=B,confidence=.95,interval='percentile',seed=SEED,global_unit='737 held-out genes resampled with replacement; all variants of sampled genes retained with cluster multiplicity',native_pairing='same gene multiplicities as primary bootstrap, then each native field finite-score mask; CNN and native evaluated on identical variants and weights',five_gene_unit='cases resampled within each label, within each fixed gene',training_seed=42,training_seed_uncertainty_included=False,native_direction_policy='negative sign for SIFT_score/SIFT4G_score only, fixed before viewing results; raw AUC retained for all 46',canonical_order=CANONICAL)
 dump(out/'bootstrap_policy.json',policy)
 tokens={'GLOBAL_AUC':f"{model_summary['CNN']['AUROC']:.4f}",'GLOBAL_AUC_CI':f"{model_summary['CNN']['AUROC_CI_lower']:.4f}-{model_summary['CNN']['AUROC_CI_upper']:.4f}",'GLOBAL_AUPRC':f"{model_summary['CNN']['AUPRC']:.4f}",'GLOBAL_LR_AUC':f"{model_summary['Logistic regression']['AUROC']:.4f}",'GLOBAL_XGB_AUC':f"{model_summary['XGBoost']['AUROC']:.4f}",'GLOBAL_CI_METHOD':f'95% percentile intervals from {B:,} bootstrap resamples of the 737 held-out gene clusters; fixed training seed42, excluding training-seed uncertainty','FIGURE2_FINAL_PANEL_MAP':'a, primary ROC; b, primary precision-recall; c, AUROC with case-stratified intervals for five wholly held-out genes, with N/P/B; d, canonical native predictor comparisons on method-specific finite scores, showing paired CNN-minus-native AUROC intervals','PRIMARY_PAIRWISE_TABLE':'native_predictors_pairwise_with_cluster_CI.csv','PRIMARY_PER_GENE_TABLE':'five_heldout_genes_metrics_CI.csv and all_heldout_genes_descriptive_metrics.csv'}
 for g in FIVE:tokens['GLOBAL_'+g+'_AUC']=f"{five_summary[g]['CNN']['AUROC']:.4f}"
 summary=dict(scope='New fixed46 strict-missense gene-heldout evaluation; no historical model outputs',n_test=len(y),P=int(y.sum()),B=int(len(y)-y.sum()),heldout_genes=G,training_seed=42,bootstrap_policy=policy,models=model_summary,paired_primary_differences=contrasts,five_genes=five_summary,tokens=tokens,best_epoch=cfg.get('best_epoch',json.loads((ip/'cnn_full_seed42_metrics.json').read_text())['best_epoch']),best_validation_AUROC=json.loads((ip/'cnn_full_seed42_metrics.json').read_text())['best_validation_AUROC'],alignment_all_passed=True,native_points_match_frozen_runner=True,source_paths={f:str(ip/f) for f in PRED_FILES.values()},source_prediction_sha256=sources,training_complete_sha256=sha(ip/'training_complete.json'),statistics_script_sha256=sha(__file__),runtime_seconds=time.perf_counter()-started)
 dump(out/'manuscript_summary.json',summary)
 (out/'README_statistics.md').write_text(f'''# Primary statistics\n\n{len(y):,} alleles from 737 wholly held-out genes; P={int(y.sum()):,}, B={int(len(y)-y.sum()):,}. All three models were fitted only under training seed42. Primary and native-paired intervals are percentile 95% intervals from {B:,} resamples of the 737 gene clusters (seed {SEED}). Entire genes are resampled, retaining all their eligible variants. These are test-gene uncertainty intervals, not training-seed variability. Five-gene intervals instead resample cases separately by label within each gene. Native scores are unmodified finite source values, never imputed; each method has its own explicit N/P/B and all compared models are recomputed on that same mask. No direction is selected from observed performance.\n\nSee manuscript_summary.json for exact values and token suggestions; native_predictors_pairwise_with_cluster_CI.csv for fields, directions, Ns and paired deltas. Source-data predictions and bootstrap multiplicities are included for reproduction. The preparation inputs, original model results and raw data were not modified.\n''')
 dump(out/'statistics_complete.json',dict(complete=True,bootstrap_repeats=B,seed=SEED,n_test=len(y),genes=G,runtime_seconds=time.perf_counter()-started))
 print('STATISTICS_COMPLETE',out,flush=True)

def plot(out):
 import matplotlib as mpl
 mpl.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.3,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'xtick.direction':'out','ytick.direction':'out','xtick.major.width':.7,'ytick.major.width':.7,'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.frameon':False,'savefig.dpi':300})
 assert json.loads((out/'statistics_complete.json').read_text())['complete']
 summary=json.loads((out/'manuscript_summary.json').read_text());primary=pd.read_csv(out/'primary_models_gene_cluster_CI.csv').set_index('model');roc=pd.read_csv(out/'primary_ROC_coordinates.csv');pr=pd.read_csv(out/'primary_PR_coordinates.csv');fg=pd.read_csv(out/'five_heldout_genes_metrics_CI.csv');native=pd.read_csv(out/'native_predictors_pairwise_with_cluster_CI.csv').set_index('feature').loc[CANONICAL]
 fig=plt.figure(figsize=(180/25.4,230/25.4),facecolor='white');a=fig.add_axes([.105,.725,.37,.215]);b=fig.add_axes([.595,.725,.37,.215]);c=fig.add_axes([.15,.505,.54,.145]);ct=fig.add_axes([.745,.505,.22,.145]);d=fig.add_axes([.525,.08,.435,.345]);dt=fig.add_axes([.10,.08,.385,.345]);style={'CNN':'-','Logistic regression':'--','XGBoost':'-.'}
 for m in MODELS:
  z=roc[roc.model.eq(m)];r=primary.loc[m];a.plot(z.FPR,z.TPR,color=COLORS[m],ls=style[m],lw=1.3,label=f"{m}: {r.AUROC:.3f} [{r.AUROC_CI_lower:.3f}, {r.AUROC_CI_upper:.3f}]")
  z=pr[pr.model.eq(m)];b.plot(z.recall,z.precision,color=COLORS[m],ls=style[m],lw=1.3,label=f"{m}: {r.AUPRC:.3f} [{r.AUPRC_CI_lower:.3f}, {r.AUPRC_CI_upper:.3f}]")
 a.plot([0,1],[0,1],color='#B5B5B5',ls=':',lw=.8,zorder=0);a.set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate',ylabel='True-positive rate');a.legend(loc='lower right',fontsize=5.7,handlelength=1.5,labelspacing=.45)
 prevalence=summary['P']/summary['n_test'];b.axhline(prevalence,color='#B5B5B5',ls=':',lw=.8);b.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision');b.legend(loc='lower left',fontsize=5.7,handlelength=1.5,labelspacing=.45)
 for ax,title in [(a,'Primary ROC (AUROC with gene-cluster CI)'),(b,'Primary precision-recall (AP with gene-cluster CI)')]:ax.set_title(title,loc='left',pad=7)
 offsets={'CNN':.20,'Logistic regression':0,'XGBoost':-.20};markers={'CNN':'o','Logistic regression':'s','XGBoost':'^'}
 for i,g in enumerate(FIVE):
  z=fg[fg.gene.eq(g)]
  for m in MODELS:
   r=z[z.model.eq(m)].iloc[0];yy=i+offsets[m];c.hlines(yy,r.AUROC_CI_lower,r.AUROC_CI_upper,color=COLORS[m],linewidth=.8);c.plot(r.AUROC,yy,markers[m],ms=3,color=COLORS[m])
 ct.set(xlim=(0,1),ylim=(-.55,4.55));ct.invert_yaxis();ct.axis('off');ct.text(.02,1.08,'N       P       B',transform=ct.transAxes,fontsize=6.7,fontweight='bold')
 for i,g in enumerate(FIVE):
  r=fg[(fg.gene.eq(g))&(fg.model.eq('CNN'))].iloc[0];ct.text(.02,i,f'{int(r.n):3d}     {int(r.P):3d}     {int(r.B):3d}',va='center',fontfamily='DejaVu Sans Mono',fontsize=6.5)
 c.set(yticks=np.arange(5),yticklabels=FIVE,ylim=(-.55,4.55),xlim=(max(0,min(.45,float(fg.AUROC_CI_lower.min()))-.04),1.005),xlabel='AUROC (within-gene case-stratified 95% CI)');c.invert_yaxis();c.axvline(.5,color='#C5C5C5',ls=':',lw=.8);c.tick_params(axis='y',length=0)
 n=len(CANONICAL);dt.set(xlim=(0,1),ylim=(-.6,n-.4));dt.invert_yaxis();dt.axis('off');dt.text(0,1.025,'Native predictor',transform=dt.transAxes,fontweight='bold',fontsize=6.7);dt.text(.69,1.025,'N',transform=dt.transAxes,fontweight='bold',ha='right',fontsize=6.7);dt.text(.985,1.025,'AUROC',transform=dt.transAxes,fontweight='bold',ha='right',fontsize=6.7)
 for i,(f,label) in enumerate(zip(CANONICAL,ALIASES)):
  r=native.loc[f];dt.text(0,i,label,va='center',fontsize=6.3);dt.text(.69,i,f'{int(r.n_pairwise):,}',ha='right',va='center',fontsize=6.1);dt.text(.985,i,f'{r.native_AUROC_direction_corrected:.3f}',ha='right',va='center',fontsize=6.3)
  if np.isfinite(r.CNN_minus_native_AUROC):
   d.hlines(i,r.delta_AUROC_CI_lower,r.delta_AUROC_CI_upper,color=COLORS['CNN'],linewidth=.8);d.plot(r.CNN_minus_native_AUROC,i,'o',ms=2.8,color=COLORS['CNN'])
 lo=min(-.005,float(native.delta_AUROC_CI_lower.min()));hi=max(.005,float(native.delta_AUROC_CI_upper.max()));margin=(hi-lo)*.08;d.set(xlim=(max(-1,lo-margin),min(1,hi+margin)),ylim=(-.6,n-.4),yticks=[],xlabel='AUROC difference (CNN - native predictor)');d.invert_yaxis();d.axvline(0,color='#999999',ls='--',lw=.8);d.spines['left'].set_visible(False)
 fig.text(.03,.962,'a',fontweight='bold',fontsize=10);fig.text(.52,.962,'b',fontweight='bold',fontsize=10);fig.text(.03,.680,'c',fontweight='bold',fontsize=10);fig.text(.10,.681,'Five prespecified genes wholly held out from fitting',fontsize=7.5,fontweight='bold');fig.text(.03,.452,'d',fontweight='bold',fontsize=10);fig.text(.10,.453,'Canonical native predictors on matched finite-score variants',fontsize=7.5,fontweight='bold')
 legend=[Line2D([0],[0],color=COLORS[m],marker=markers[m],ls='none',label=m,markersize=3.5) for m in MODELS];fig.legend(handles=legend,loc='center',bbox_to_anchor=(.56,.477),ncol=3,fontsize=6.5,handlelength=1.0,columnspacing=1.8)
 fig.text(.105,.995,f"New fixed-46 gene-heldout evaluation: {summary['n_test']:,} alleles, {summary['heldout_genes']} test genes",fontsize=8,va='top')
 fig.text(.10,.034,f"Primary/native CIs: {summary['bootstrap_policy']['repeats']:,} resamples of 737 held-out gene clusters. All model fits: seed 42.",fontsize=6)
 fig.text(.10,.020,'Gene CIs use label-stratified case resampling. Intervals exclude training-seed uncertainty. Native field identities are in source tables.',fontsize=5.8)
 for ext in ['svg','pdf','png']:fig.savefig(out/('Figure2_global_primary.'+ext),dpi=300,facecolor='white')
 plt.close(fig)
 (out/'Figure2_caption.txt').write_text(f'''Figure 2. New fixed-46 gene-heldout evaluation. (a) ROC and (b) precision-recall curves for the newly trained CNN, logistic regression and XGBoost, all evaluated on the same 29,364 held-out alleles from 737 genes (P/LP 11,374; B/LB 17,990). AUROC and average precision (AP) intervals are percentile 95% intervals from {summary['bootstrap_policy']['repeats']:,} gene-cluster bootstrap resamples. (c) AUROC and case-stratified percentile 95% intervals for five prespecified genes excluded entirely from training and validation; N/P/B are shown. (d) The prespecified 24 representative native fields retain their finite source values and fixed directions. For each field, native AUROC and N refer to that field's eligible test variants; the CNN-minus-native AUROC interval uses paired resampling of the 737 held-out gene clusters and evaluates both scores on exactly the same eligible variants. Source tables also provide same-N logistic-regression and XGBoost results, class counts and all 46 field identities. Positive deltas favor the CNN; negative deltas favor the native predictor. No direction or field order was selected from observed performance. All primary fits used one training seed (42); these intervals do not quantify training-seed uncertainty.\n''')
 svg=(out/'Figure2_global_primary.svg').read_text();assert '<text' in svg
 dump(out/'figure_export_audit.json',dict(width_mm=180,height_mm=230,png_dpi=300,svg_text_nodes=svg.count('<text'),backend='Python matplotlib',fonts=['Arial','DejaVu Sans','Liberation Sans'],panels=['a ROC','b PR','c five-gene CI/N/P/B','d canonical paired delta'],source='only completed primary run; no invented or legacy results'))
 print('FIGURE2_EXPORTED',out,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir');p.add_argument('--output-dir',required=True);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--compute-only',action='store_true');p.add_argument('--plot-only',action='store_true');p.add_argument('--wait-for-completion',action='store_true');a=p.parse_args();out=Path(a.output_dir)
 if not a.plot_only:
  ip=Path(a.input_dir)
  if a.wait_for_completion:
   while not (ip/'training_complete.json').exists():time.sleep(10)
  compute(ip,out,a.bootstrap)
 if not a.compute_only:plot(out)
if __name__=='__main__':main()
