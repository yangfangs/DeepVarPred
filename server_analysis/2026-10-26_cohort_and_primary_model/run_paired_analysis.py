#!/usr/bin/env python3
"""Read-only retrospective paired-score analyses. No fitting or source mutation."""
from pathlib import Path
import ast, csv, hashlib, json, re, warnings
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path('/home/bio1/workdata/GeneMutation2')
OUT=ROOT/'Revision_2026-10-26/paired_analysis'
OUT.mkdir(parents=True, exist_ok=True)
OLD=Path('/home/bio1/workdata/GeneMutation/train_1/BRCA1/pred/pred_all_score.csv')
SOURCE=Path('/home/bio1/PycharmProjects/SNPdp')
AA=set('ACDEFGHIKLMNPQRSTVWY')
manifest=[]
def record(p):
    p=Path(p)
    if p.exists() and p.is_file():
        manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def save(name,df):df.to_csv(OUT/name,index=False)
def group_label(x):
    x=str(x).strip().lower().replace(' ','_')
    if x in {'pathogenic','likely_pathogenic','pathogenic/likely_pathogenic'}:return 'P/LP'
    if x in {'benign','likely_benign','benign/likely_benign'}:return 'B/LB'
    if x=='uncertain_significance':return 'VUS'
    if 'conflict' in x:return 'Conflicting'
    return 'Other_or_unannotated'
def keys(df):
    out=df.copy()
    chrcol=next((x for x in ['hg19_chr','chr'] if x in out),None)
    poscol=next((x for x in ['hg19_pos(1-based)','hg19_pos'] if x in out),None)
    if chrcol is None or poscol is None:raise ValueError('No explicit hg19 key')
    ch=out[chrcol].astype(str).str.replace(r'^chr','',regex=True).str.replace(r'\.0$','',regex=True).str.upper()
    pos=pd.to_numeric(out[poscol],errors='coerce')
    ref=out['ref'].astype(str).str.upper();alt=out['alt'].astype(str).str.upper()
    valid=ch.isin([str(i) for i in range(1,23)]+['X','Y','MT','M']) & pos.notna() & (pos>0) & (pos==np.floor(pos)) & ref.isin(list('ACGT')) & alt.isin(list('ACGT')) & (ref!=alt)
    out=out.loc[valid].copy()
    out['key']=ch[valid]+':'+pos[valid].astype('int64').astype(str)+':'+ref[valid]+'>'+alt[valid]
    out['position_allele_key']=pos[valid].astype('int64').astype(str)+':'+ref[valid]+'>'+alt[valid]
    return out

def corr(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    m=np.isfinite(x)&np.isfinite(y);x=x[m];y=y[m]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return {'n':len(x),'pearson_r':np.nan,'pearson_p':np.nan,'spearman_rho':np.nan,'spearman_p':np.nan}
    a=stats.pearsonr(x,y);b=stats.spearmanr(x,y)
    return {'n':len(x),'pearson_r':float(a.statistic),'pearson_p':float(a.pvalue),'spearman_rho':float(b.statistic),'spearman_p':float(b.pvalue)}

def numeric_parts(value):
    vals=[]
    for token in str(value).split(';'):
        try:
            v=float(token)
            if np.isfinite(v):vals.append(v)
        except ValueError:pass
    return vals
def consistent_scalar(value):
    vals=numeric_parts(value)
    return vals[0] if vals and max(vals)-min(vals)<=1e-12 else np.nan
def nonmissing_transcript_mean(value):
    vals=numeric_parts(value)
    return float(np.mean(vals)) if vals else np.nan

# Original six-panel scatter source, exactly identified by R_script2/plot_cor_each.R.
record(OLD);record(SOURCE/'R_script2/plot_cor_each.R')
old=pd.read_csv(OLD,dtype=str,keep_default_na=False)
old['label_group']=old['clinvar_clnsig'].map(group_label)
old['DVP']=pd.to_numeric(old['DeepVarPred_score'],errors='coerce')
old=keys(old)
save('original_figure4_label_composition.csv',old.groupby(['clinvar_clnsig','label_group'],dropna=False).size().reset_index(name='n_rows'))
methods={'CADD_raw':1,'REVEL_score':1,'MVP_score':1,'MutPred_score':1,'PROVEAN_converted_rankscore':1,'SIFT_score':-1}
corrows=[];plotdata={}
for method,direction in methods.items():
    # R as.numeric + round(4); SIFT plot uses 1 - score after rounding.
    raw=pd.to_numeric(old[method],errors='coerce').round(4)
    oriented=1-raw if direction==-1 else raw
    plotdata[method]=oriented
    for group in ['All','P/LP','B/LB','VUS','Conflicting','Other_or_unannotated']:
        sub=old if group=='All' else old.loc[old['label_group']==group]
        z=oriented.loc[sub.index];d=sub['DVP']
        cr=corr(z,d)
        corrows.append({'method':method,'orientation':'1 - SIFT_score' if direction==-1 else 'native score','label_group':group,'n_group_rows':len(sub),'n_finite_dvp':int(d.notna().sum()),'n_finite_comparator':int(z.notna().sum()),'n_excluded_missing_pair':int((~(np.isfinite(z)&np.isfinite(d))).sum()),**cr})
cor_df=pd.DataFrame(corrows);save('figure4_label_stratified_correlations.csv',cor_df)
oldout=old[['key','clinvar_clnsig','label_group','DVP']].copy()
for m in methods:oldout[m]=pd.to_numeric(old[m],errors='coerce')
save('figure4_per_variant_scores.csv',oldout)
expected={'CADD_raw':.79,'REVEL_score':.69,'MVP_score':.61,'MutPred_score':.71,'PROVEAN_converted_rankscore':.56,'SIFT_score':.38}
validation=[]
for m,v in expected.items():
    row=cor_df[(cor_df.method==m)&(cor_df.label_group=='All')].iloc[0]
    validation.append({'method':m,'figure_displayed_r':v,'recomputed_r':row.pearson_r,'n':int(row.n),'matches_displayed_rounding':bool(round(row.pearson_r,2)==v)})
save('figure4_original_coefficient_reproduction.csv',pd.DataFrame(validation))

# VUS follow-up uses the archived all-class prediction file from the same gene/run
# directory plus UNIMPUTED predictor fields. Duplicated genomic keys are excluded,
# not averaged, because archived duplicate rows can carry different predictions.
predpath=OLD.parent/'pred_score.csv'
rawpath=OLD.parents[1]/'score/BRCA1_filter.txt'
record(predpath);record(rawpath)
pred=pd.read_csv(predpath,dtype=str,keep_default_na=False).rename(columns={'h19':'hg19_pos(1-based)','score':'DVP'})
pred['hg19_chr']='17'
pred=keys(pred)
pred['DVP']=pd.to_numeric(pred.DVP,errors='coerce')
pred['pred_label_group']=pred.cls.map({'p':'P/LP','b':'B/LB','u':'VUS','c':'Conflicting'})
raw=pd.read_csv(rawpath,sep='\t',dtype=str,keep_default_na=False)
raw=keys(raw);raw['label_group']=raw.clinvar_clnsig.map(group_label)
raw_dup=raw.key.duplicated(keep=False);pred_dup=pred.key.duplicated(keep=False)
save('BRCA1_excluded_duplicate_prediction_rows.csv',pred[pred_dup])
save('BRCA1_excluded_duplicate_annotation_rows.csv',raw[raw_dup][['key','clinvar_clnsig','label_group']])
raw_unique=raw[~raw_dup].copy();pred_unique=pred[~pred_dup].copy()
new=raw_unique.merge(pred_unique[['key','DVP','pred_label_group']],on='key',how='left',validate='one_to_one')
new['label_agrees']=new.label_group.eq(new.pred_label_group)
new['DVP']=new.DVP.where(new.label_agrees)
# Verify molecular scope using the original dbNSFP annotation, without multiplying
# rows at transcript-related duplicated genomic keys. Conservative primary set:
# exactly one AA substitution profile and both residues in the canonical 20 AAs.
aa_path=OLD.parents[1]/'score/BRCA1_dbNSFP.txt'
record(aa_path)
db=keys(pd.read_csv(aa_path,sep='\t',dtype=str,keep_default_na=False,usecols=lambda c:c in {'hg19_chr','hg19_pos(1-based)','ref','alt','aaref','aaalt'}))
db['aa_pair']=db.aaref+'>'+db.aaalt
aa_info=db.groupby('key').agg(aa_profiles=('aa_pair',lambda s:';'.join(sorted(set(s)))),n_unique_aa_profiles=('aa_pair','nunique')).reset_index()
aa_info['strict_canonical_missense']=aa_info.n_unique_aa_profiles.eq(1)&aa_info.aa_profiles.map(lambda v:len(v)==3 and v[0] in AA and v[2] in AA and v[0]!=v[2])
original_aa=old[['key','clinvar_clnsig','label_group','DVP']].merge(aa_info,on='key',how='left',validate='many_to_one')
save('original_figure4_AA_scope_audit.csv',original_aa)
save('original_figure4_AA_scope_summary.csv',original_aa.groupby(['label_group','aa_profiles','strict_canonical_missense'],dropna=False).size().reset_index(name='n_original_rows'))
new=new.merge(aa_info,on='key',how='left',validate='one_to_one')
before_aa=len(new);vus_before_aa=int((new.label_group.eq('VUS')&new.DVP.notna()).sum())
save('BRCA1_excluded_nonmissense_or_ambiguous_AA.csv',new[~new.strict_canonical_missense.fillna(False)][['key','clinvar_clnsig','label_group','DVP','aa_profiles','n_unique_aa_profiles','strict_canonical_missense']])
new=new[new.strict_canonical_missense.fillna(False)].copy().reset_index(drop=True)
over=old[['key','DVP']].merge(pred_unique[['key','DVP']],on='key',how='inner',suffixes=('_figure4','_allclasses'))
overlapcheck={'n_original_figure4_rows':len(old),'n_rows_with_unique_allclass_prediction':len(over),'n_matching_scores_tolerance_1e7':int(((over.DVP_figure4-over.DVP_allclasses).abs()<1e-7).sum()),'max_absolute_score_difference':float((over.DVP_figure4-over.DVP_allclasses).abs().max())}
dump('BRCA1_same_run_score_overlap_check.json',overlapcheck)
vusflow={'raw_annotation_rows':len(raw),'prediction_rows':len(pred),'raw_duplicate_key_rows_excluded':int(raw_dup.sum()),'prediction_duplicate_key_rows_excluded':int(pred_dup.sum()),'raw_vus_rows':int(raw.label_group.eq('VUS').sum()),'raw_unique_vus_rows':int(raw_unique.label_group.eq('VUS').sum()),'vus_with_unique_label_matching_prediction_before_aa_filter':vus_before_aa,'n_unique_rows_before_aa_filter':before_aa,'n_strict_canonical_missense_rows':len(new),'vus_with_unique_label_matching_prediction':int((new.label_group.eq('VUS')&new.DVP.notna()).sum()),'original_figure4_strict_canonical_missense_rows':int(original_aa.strict_canonical_missense.fillna(False).sum()),'preprocessing':'No cross-variant predictor imputation. Primary analysis collapses repeated identical nonmissing transcript values; genuinely different transcript scores are excluded. Secondary sensitivity uses their arithmetic mean. Original R-compatible round(4); SIFT is 1 - native SIFT. Every duplicated genomic key is excluded. Molecular scope requires one unambiguous canonical AA replacement in original dbNSFP.'}
dump('BRCA1_VUS_followup_flow.json',vusflow)
newcorr=[];secondary=[];coverage=[];parsed_fields={};mean_fields={}
for method,direction in methods.items():
    rawscore=new[method].map(consistent_scalar).round(4)
    mean_score=new[method].map(nonmissing_transcript_mean).round(4)
    parsed_fields[method]=rawscore;mean_fields[method]=mean_score
    oriented=1-rawscore if direction==-1 else rawscore
    mean_oriented=1-mean_score if direction==-1 else mean_score
    vu_raw=new.loc[new.label_group.eq('VUS'),method]
    coverage.append({'method':method,'n_vus_after_key_dedup':len(vu_raw),'n_with_any_numeric_value':int(vu_raw.map(lambda v:bool(numeric_parts(v))).sum()),'n_scalar_or_consistent_repeated_values':int(rawscore.loc[vu_raw.index].notna().sum()),'n_distinct_multitranscript_values_excluded_primary':int((mean_score.loc[vu_raw.index].notna()&rawscore.loc[vu_raw.index].isna()).sum()),'n_no_numeric_values':int(mean_score.loc[vu_raw.index].isna().sum())})
    for group in ['All','P/LP','B/LB','VUS','Conflicting','Other_or_unannotated']:
        sub=new if group=='All' else new[new.label_group.eq(group)]
        z=oriented.loc[sub.index];d=sub.DVP
        newcorr.append({'method':method,'orientation':'1 - SIFT_score' if direction==-1 else 'native score','label_group':group,'n_group_rows':len(sub),'n_finite_dvp':int(d.notna().sum()),'n_finite_comparator':int(z.notna().sum()),'n_excluded_missing_pair':int((~(np.isfinite(z)&np.isfinite(d))).sum()),**corr(z,d)})
        zmean=mean_oriented.loc[sub.index]
        secondary.append({'method':method,'orientation':'1 - SIFT_score' if direction==-1 else 'native score','label_group':group,'n_group_rows':len(sub),'aggregation':'arithmetic mean of nonmissing transcript scores; no cross-variant imputation',**corr(zmean,d)})
save('BRCA1_allclass_label_stratified_correlations.csv',pd.DataFrame(newcorr))
save('BRCA1_multitranscript_mean_sensitivity_correlations.csv',pd.DataFrame(secondary))
save('BRCA1_VUS_comparator_coverage.csv',pd.DataFrame(coverage))
newout=new[['key','clinvar_clnsig','label_group','pred_label_group','label_agrees','DVP']].copy()
for m in methods:
    newout[m]=parsed_fields[m]
    newout[m+'_transcript_mean_sensitivity']=mean_fields[m]
save('BRCA1_allclass_per_variant_scores.csv',newout)

# Thresholds are extracted without importing/executing original code.
thresholds=[]
for source in ['somatic_mutation/compare_all_cancer.py','Clingene_new/compare_all_clingen.py']:
    path=SOURCE/source;record(path);txt=path.read_text()
    func=txt[txt.index('def trans_score_to_label'):txt.index('# each gene for mean')]
    current=None
    for line in func.splitlines():
        mo=re.search(r'if method == [\"\']([^\"\']+)',line)
        if mo:current=mo.group(1)
        mo=re.search(r'1 if label\s*(>=|<=|>|<)\s*(-?[0-9.]+)',line)
        if mo and not line.lstrip().startswith('#'):
            threshold=float(mo.group(2));pre='1 - native score' if current in {'SIFT_score','LRT_score','FATHMM_score'} else 'native score'
            native_op='<= ' + str(1-threshold) if pre!='native score' else mo.group(1)+' '+str(threshold)
            thresholds.append({'source_script':source,'method':current,'comparison_on_processed_score':mo.group(1),'processed_cutoff':threshold,'preprocessing':pre,'equivalent_native_decision':native_op,'interpretation':'historical code rule; not a newly optimized or necessarily literature-recommended cutoff'})
save('historical_comparator_thresholds.csv',pd.DataFrame(thresholds))
record(ROOT/'compare/software_score_threshold.txt')

profiles=[]
def load_export(path,tag):
    record(path);df=pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False)
    info={'export':tag,'source_rows':len(df),'columns':list(df.columns)}
    df=keys(df);info['valid_hg19_snv_rows']=len(df)
    df['score']=pd.to_numeric(df['score'],errors='coerce')
    info['invalid_or_out_of_range_score_rows']=int((~np.isfinite(df.score)|~df.score.between(0,1)).sum())
    df=df[np.isfinite(df.score)&df.score.between(0,1)].copy()
    aa=df['aaref'].isin(AA)&df['aaalt'].isin(AA)&df.aaref.ne(df.aaalt)
    df['canonical_missense']=aa
    info['canonical_missense_rows']=int(aa.sum())
    info['noncanonical_aa_rows']=int((~aa).sum())
    info['aaalt_counts']=df.aaalt.value_counts().to_dict()
    st=df.groupby('key').score.agg(['min','max','count'])
    conflicts=st.index[(st['max']-st['min']).abs()>1e-8]
    info['duplicate_key_rows']=int(df.duplicated('key',keep=False).sum());info['unique_keys']=int(df.key.nunique());info['conflicting_score_keys_excluded']=len(conflicts)
    info['duplicate_rows_removed_after_conflict_exclusion']=int(df[~df.key.isin(conflicts)].duplicated('key').sum())
    # A key is considered missense if at least one retained transcript annotation is canonical;
    # all duplicate scores must agree. Retain aa profile for transparent cross-export checks.
    miss=df.groupby('key').canonical_missense.any()
    df=df[~df.key.isin(conflicts)].sort_values('canonical_missense',ascending=False).drop_duplicates('key')
    df['canonical_missense']=df.key.map(miss)
    info['retained_unique_keys']=len(df);profiles.append(info)
    return df[['key','position_allele_key','aaref','aaalt','score','canonical_missense']].rename(columns={c:f'{c}_{tag}' for c in ['aaref','aaalt','score','canonical_missense']})
germ=load_export(ROOT/'DeepVarPred_score_all_hg19.tsv','germline')
som=load_export(ROOT/'DeepVarPred_score_somatic_hg19.tsv','somatic')
paired=germ.merge(som,on=['key','position_allele_key'],how='inner',validate='one_to_one')
paired['both_canonical_missense']=paired.canonical_missense_germline & paired.canonical_missense_somatic
paired['same_aa_annotation']=paired.aaref_germline.eq(paired.aaref_somatic)&paired.aaalt_germline.eq(paired.aaalt_somatic)
paired['somatic_minus_germline']=paired.score_somatic-paired.score_germline
paired['tie_at_0p5']=paired.score_germline.eq(.5)|paired.score_somatic.eq(.5)
paired['germline_call_gt0p5']=paired.score_germline.gt(.5).astype(int)
paired['somatic_call_gt0p5']=paired.score_somatic.gt(.5).astype(int)
paired['discordant_gt0p5']=paired.germline_call_gt0p5.ne(paired.somatic_call_gt0p5)

# Annotate with the historical somatic collection, not the later pooled tabular split.
annotations=[]
for gd in sorted((ROOT/'train_somatic').iterdir()):
    if not gd.is_dir():continue
    f=gd/'score'/f'{gd.name}_filter_mean_all_somatic.txt'
    if not f.exists():continue
    try:
        df=pd.read_csv(f,sep='\t',dtype=str,keep_default_na=False,usecols=lambda c:c in {'hg19_chr','hg19_pos(1-based)','ref','alt','clinvar_clnsig'})
        df=keys(df);df=df[df.key.isin(set(paired.key))]
        if len(df):
            df['gene']=gd.name;df['label_group']=df.clinvar_clnsig.map(group_label)
            annotations.append(df[['key','gene','clinvar_clnsig','label_group']]);record(f)
    except Exception as e:profiles.append({'annotation_file_error':str(f),'error':str(e)})
anno=pd.concat(annotations,ignore_index=True) if annotations else pd.DataFrame(columns=['key','gene','clinvar_clnsig','label_group'])
save('paired_annotation_records.csv',anno)
if len(anno):
    def labels(s):
        vals=set(s)
        if len(vals)==1:return next(iter(vals))
        if {'P/LP','B/LB'}<=vals:return 'Cross_record_conflict'
        return 'Mixed_annotation'
    ann=anno.groupby('key').agg(genes=('gene',lambda s:';'.join(sorted(set(s)))),annotation_labels=('clinvar_clnsig',lambda s:';'.join(sorted(set(s)))),label_group=('label_group',labels)).reset_index()
    paired=paired.merge(ann,on='key',how='left',validate='one_to_one')
else:paired['genes']='';paired['annotation_labels']='';paired['label_group']='Unmapped'
paired['label_group']=paired.label_group.fillna('Unmapped')

# Candidate membership only: pooled PNG filenames omit chromosome and gene.
member=[]
candidates={
 'old_GeneMutation_train_all':Path('/home/bio1/workdata/GeneMutation/train_all/dataset/dataset'),
 'GeneMutation2_train_first':ROOT/'train_first/dataset/dataset',
 'GeneMutation2_train_clingene':ROOT/'train_clingene/dataset/dataset',
 'GeneMutation2_train_somatic':ROOT/'train_somatic/dataset/dataset'}
for tag,base in candidates.items():
    for split in ['train','val','test']:
        folder=base/split;ks=set();nf=0
        if folder.exists():
            for cls in folder.iterdir():
                if not cls.is_dir():continue
                for f in cls.iterdir():
                    if not f.name.endswith('.png'):continue
                    nf+=1;a=f.stem.split('_')
                    if len(a)>=3:
                        try:ks.add(f'{int(a[0])}:{a[1].upper()}>{a[2].upper()}')
                        except ValueError:pass
        col=f'potential_match_{tag}_{split}'
        paired[col]=paired.position_allele_key.isin(ks) if folder.exists() else pd.NA
        member.append({'candidate_collection':tag,'split':split,'path':str(folder),'exists':folder.exists(),'n_png_files':nf,'n_unique_position_ref_alt':len(ks),'paired_matches_without_chr_gene':int(paired.position_allele_key.isin(ks).sum()),'membership_status':'candidate only; filenames omit chr/gene and checkpoint-to-export linkage not frozen'})
save('candidate_training_membership_inventory.csv',pd.DataFrame(member))
save('paired_export_scores.csv',paired)
dump('export_input_profiles.json',profiles)

summary=[]
for subset_name,mask in [('All_shared_export_SNVs',np.ones(len(paired),dtype=bool)),('Shared_canonical_missense',paired.both_canonical_missense),('Shared_canonical_missense_same_aa_annotation',paired.both_canonical_missense&paired.same_aa_annotation)]:
    pool=paired.loc[mask]
    for group in ['All']+sorted(set(pool.label_group)):
        d=pool if group=='All' else pool[pool.label_group==group]
        if d.empty:continue
        delta=d.somatic_minus_germline;no_tie=d.loc[~d.tie_at_0p5]
        summary.append({'subset':subset_name,'label_group':group,**corr(d.score_germline,d.score_somatic),'mean_somatic_minus_germline':float(delta.mean()),'median_somatic_minus_germline':float(delta.median()),'q25_delta':float(delta.quantile(.25)),'q75_delta':float(delta.quantile(.75)),'mean_absolute_delta':float(delta.abs().mean()),'median_absolute_delta':float(delta.abs().median()),'n_delta_positive':int((delta>0).sum()),'n_delta_negative':int((delta<0).sum()),'n_delta_zero':int((delta==0).sum()),'n_equal_0p5_either_export':int(d.tie_at_0p5.sum()),'n_discordant_gt0p5':int(d.discordant_gt0p5.sum()),'discordance_fraction_gt0p5':float(d.discordant_gt0p5.mean()),'n_excluding_0p5':len(no_tie),'n_discordant_excluding_0p5':int(no_tie.discordant_gt0p5.sum()),'discordance_fraction_excluding_0p5':float(no_tie.discordant_gt0p5.mean()) if len(no_tie) else np.nan})
summary=pd.DataFrame(summary);save('paired_export_summary.csv',summary)

# Figures: neutral journal style. Core point = descriptive score agreement, not accuracy.
# VUS panel map: six distinct comparator relationships; paired a=joint score relation,
# b=signed score difference, an orthogonal question. Backend Python throughout.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300})
fig,axes=plt.subplots(2,3,figsize=(7.2,4.7),layout='constrained')
vu=new.label_group.eq('VUS')
display_names={'CADD_raw':'CADD raw','REVEL_score':'REVEL','MVP_score':'MVP','MutPred_score':'MutPred','PROVEAN_converted_rankscore':'PROVEAN rankscore','SIFT_score':'1 - SIFT'}
for i,(ax,m) in enumerate(zip(axes.flat,methods)):
    v=parsed_fields[m]
    x=(1-v if m=='SIFT_score' else v)[vu];y=new.loc[vu,'DVP'];valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
    ax.scatter(x,y,s=5,alpha=.5,color='#3C5488',edgecolors='none',rasterized=False)
    rr=corr(x,y)
    ax.set_title(f'{chr(97+i)}  '+display_names[m],loc='left',fontweight='bold')
    ax.set_xlabel('Comparator score');ax.set_ylabel('DeepVarPred score');ax.set_ylim(-.04,1.04)
    ax.text(.04,.96,f'n = {rr["n"]:,}\nr = {rr["pearson_r"]:.3f}\nrho = {rr["spearman_rho"]:.3f}',transform=ax.transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':1,'pad':1})
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'Figure_R_VUS_only_comparison.{ext}',bbox_inches='tight')
plt.close(fig)
miss=paired[paired.both_canonical_missense]
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
ax=axes[0];im=ax.hexbin(miss.score_germline,miss.score_somatic,gridsize=35,mincnt=1,cmap='Blues',bins='log');ax.plot([0,1],[0,1],color='#777777',linewidth=.7,linestyle='--')
ax.set(xlabel='Germline export score',ylabel='Somatic export score',xlim=(-.02,1.02),ylim=(-.02,1.02));ax.set_title('a  Shared missense variants',loc='left',fontweight='bold')
rr=corr(miss.score_germline,miss.score_somatic);ax.text(.04,.96,f'n = {len(miss):,}\nr = {rr["pearson_r"]:.3f}\nrho = {rr["spearman_rho"]:.3f}',transform=ax.transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':1,'pad':1});fig.colorbar(im,ax=ax,label='Variant count (log scale)')
ax=axes[1];ax.hist(miss.somatic_minus_germline,bins=np.linspace(-1,1,51),color='#3C5488',edgecolor='white',linewidth=.3);ax.axvline(0,color='#777777',linestyle='--',linewidth=.7);ax.set(xlabel='Somatic minus germline score',ylabel='Variant count');ax.set_title('b  Paired score differences',loc='left',fontweight='bold')
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'Figure_R_shared_model_scores.{ext}',bbox_inches='tight')
plt.close(fig)
record(SOURCE/'predict_all/Predict_all_loc.py');record(SOURCE/'predict_all/Predict_all_samatic.py')
save('source_manifest.csv',pd.DataFrame(manifest))
info={'original_figure4_rows':len(old),'original_figure4_label_counts':old.label_group.value_counts().to_dict(),'original_figure4_unique_keys':int(old.key.nunique()),'original_figure4_duplicated_keys':int(old.duplicated('key').sum()),'original_figure4_reproduced_rounded_coefficients':sum(x['matches_displayed_rounding'] for x in validation),'vus_followup_flow':vusflow,'vus_prediction_provenance_overlap':overlapcheck,'germline_retained_unique_keys':len(germ),'somatic_retained_unique_keys':len(som),'shared_unique_snv_keys':len(paired),'shared_canonical_missense_keys':int(paired.both_canonical_missense.sum()),'shared_canonical_same_aa_keys':int((paired.both_canonical_missense&paired.same_aa_annotation).sum()),'export_scores_rounding':'softmax class 1 rounded to 3 decimal places in available export scripts','training_membership':'unverified; candidate pooled PNG filenames omit chromosome/gene; no frozen export-to-checkpoint receipt','environment':{'numpy':np.__version__,'pandas':pd.__version__},'analyses_are_descriptive':'No accuracy, clinical reclassification, or context-specific correctness claim is made from VUS agreement or paired score differences.'}
dump('analysis_summary.json',info)
print(json.dumps(info,indent=2));print(pd.DataFrame(newcorr).query('label_group == "VUS"').to_string(index=False));print(summary[(summary.subset=='Shared_canonical_missense')&(summary.label_group=='All')].to_string(index=False))
