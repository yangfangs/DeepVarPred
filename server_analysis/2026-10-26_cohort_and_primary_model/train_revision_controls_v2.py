"""Matched revision experiments. All writes go to a new revision workspace.

This script trains new models. It does not reuse historical CNN predictions as if
they had been trained on a new split. Original data and weights are read only.
"""
import argparse, copy, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def jwrite(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def metrics(y,s):
    pred=s>.5
    return dict(n=len(y),n_positive=int(sum(y)),n_negative=int(len(y)-sum(y)),
        AUROC=float(roc_auc_score(y,s)) if len(set(y))==2 else None,
        AUPRC=float(average_precision_score(y,s)),accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),F1=float(f1_score(y,pred,zero_division=0)))


def load_fasta(path):
    return ''.join(line.strip().upper() for line in Path(path).read_text().splitlines() if not line.startswith('>'))


def make_split(df,holdout=None):
    ix=np.arange(len(df))
    if holdout:
        test=ix[df.gene.eq(holdout).values]
        pool=ix[~df.gene.eq(holdout).values]
        strata=df.iloc[pool].gene.astype(str)+':'+df.iloc[pool].label.astype(str)
        tr,va=train_test_split(pool,test_size=.25,random_state=123,stratify=strata)
    else:
        strata=df.gene.astype(str)+':'+df.label.astype(str)
        tr,rest=train_test_split(ix,test_size=.4,random_state=123,stratify=strata)
        strata2=df.iloc[rest].gene.astype(str)+':'+df.iloc[rest].label.astype(str)
        va,test=train_test_split(rest,test_size=.5,random_state=123,stratify=strata2)
    assert not (set(df.iloc[tr].variant_key)&set(df.iloc[va].variant_key))
    assert not (set(df.iloc[tr].variant_key)&set(df.iloc[test].variant_key))
    assert not (set(df.iloc[va].variant_key)&set(df.iloc[test].variant_key))
    return np.array(tr),np.array(va),np.array(test)


def build_images(df,scaled,features,refmeta,out,mapping,tag):
    p=out/('images_'+tag+'.npy')
    if p.exists() and (out/('encoding_audit_'+tag+'.json')).exists():return np.load(p,mmap_mode='r')
    partial=out/('images_'+tag+'.partial.npy')
    imgs=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint8,shape=(len(df),3,300,300))
    lut=np.zeros(256,dtype=np.uint8)
    for base,value in mapping.items():lut[ord(base)]=value
    audit=[]
    for gene,gidx in df.groupby('gene',sort=False).groups.items():
        meta=refmeta[gene];seq=load_fasta(meta['fasta_path']);L=len(seq);W=round(math.sqrt(L))+1
        assert L==meta['bed_length']
        codes=lut[np.frombuffer(seq.encode('ascii'),dtype=np.uint8)]
        native=np.zeros(W*W,dtype=np.uint8);native[:L]=codes;native=native.reshape(W,W)
        smallref=np.array(Image.fromarray(native).resize((300,300),Image.Resampling.BILINEAR))
        counts=[];same=0
        for i in gidx:
            row=df.iloc[i];k=int(row.sequence_index_0based)
            assert seq[k]==row.ref and row.ref!=row.alt and row.alt in mapping
            score=np.zeros(W*W,dtype=np.uint8);score[:len(features)]=scaled[i]
            score=np.array(Image.fromarray(score.reshape(W,W)).resize((300,300),Image.Resampling.BILINEAR))
            alt=native.copy();alt.flat[k]=mapping[row.alt]
            smallalt=np.array(Image.fromarray(alt).resize((300,300),Image.Resampling.BILINEAR))
            if tag.startswith('change'):
                delta=np.zeros_like(native);delta.flat[k]=mapping[row.alt]
                smallalt=np.array(Image.fromarray(delta).resize((300,300),Image.Resampling.BILINEAR))
            imgs[i,0]=score;imgs[i,1]=smallref;imgs[i,2]=smallalt
            if not tag.startswith('change'):same += int(np.array_equal(smallref,smallalt))
            counts.append(int(np.count_nonzero(score)))
        audit.append(dict(gene=gene,n=len(gidx),native_width=W,sequence_length=L,
            score_slots=len(features),native_score_slot_fraction=len(features)/(W*W),
            mean_resized_score_nonzero_fraction=np.mean(counts)/90000,
            identical_reference_alternate_images=same if not tag.startswith('change') else None,
            unknown_reference_bases=int(np.sum(codes==0))))
    imgs.flush();del imgs;partial.replace(p)
    jwrite(out/('encoding_audit_'+tag+'.json'),dict(mapping=mapping,genes=audit))
    return np.load(p,mmap_mode='r')


def fit_model(cache,df,tr,va,te,condition,seed,out,feature_scaled,features,reduced,epochs=10):
    name=condition+'_seed'+str(seed);predfile=out/(name+'_predictions.csv');metafile=out/(name+'_metrics.json')
    if predfile.exists() and metafile.exists():return json.loads(metafile.read_text())
    seed_all(seed);model=torchvision.models.resnet34(weights=None,num_classes=2).cuda()
    optimizer=torch.optim.SGD(model.parameters(),lr=.025,momentum=.9)
    scaler=torch.cuda.amp.GradScaler();y=df.label.to_numpy(dtype=np.int64)
    # Interventions are applied to native score values BEFORE resizing. Replacing
    # an already resized pixel would mix neighboring predictor slots.
    condition_cache=cache
    if condition in ['score_reduced','cadd_only','full_plus_alphamissense']:
        raise RuntimeError('Native feature-intervention cache must be selected by caller')
    def tensor(ids):
        x=torch.from_numpy(np.array(condition_cache[ids],copy=True)).to('cuda',non_blocking=True).float().div_(255)
        if condition in ['score_only','cadd_score_only']:x[:,1:]=0
        if condition=='sequence_only':x[:,0]=0
        if condition=='full_normalized':x[:,1:]*=(255/4)
        return x
    def predict(ids):
        model.eval();scores=[]
        with torch.inference_mode():
            for st in range(0,len(ids),64):
                with torch.autocast('cuda',dtype=torch.float16):logits=model(tensor(ids[st:st+64]))
                scores.extend(torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
        return np.asarray(scores,dtype=float)
    nclass=np.bincount(y[tr],minlength=2);weights=1/nclass[y[tr]];weights=weights/weights.sum()
    best_auc=-1.;best_state=None;hist=[];start=time.perf_counter()
    for epoch in range(epochs):
        model.train();r=np.random.default_rng(seed+epoch*10007)
        train_ids=r.choice(tr,size=len(tr),replace=True,p=weights);losses=[]
        for st in range(0,len(train_ids),64):
            ids=train_ids[st:st+64];target=torch.from_numpy(y[ids]).to('cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(tensor(ids)),target)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.detach().cpu()))
        vs=predict(va);auc=roc_auc_score(y[va],vs)
        hist.append(dict(epoch=epoch+1,train_loss=np.mean(losses),validation_AUROC=auc))
        if auc>best_auc+1e-12:
            best_auc=auc;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);test_score=predict(te)
    pred=df.iloc[te][['variant_id','variant_key','gene','label']].copy();pred['prob_pathogenic']=test_score
    pred.to_csv(predfile,index=False);pd.DataFrame(hist).to_csv(out/(name+'_history.csv'),index=False)
    torch.save(best_state,out/(name+'_weights.pth'))
    row=metrics(y[te],test_score);row.update(condition=condition,seed=seed,best_epoch=best_epoch,
        validation_AUROC=best_auc,n_train=len(tr),n_validation=len(va),runtime_seconds=time.perf_counter()-start)
    jwrite(metafile,row)
    per=[]
    for gene,g in pred.groupby('gene'):
        rr=metrics(g.label.values,g.prob_pathogenic.values);rr.update(gene=gene,condition=condition,seed=seed);per.append(rr)
    pd.DataFrame(per).to_csv(out/(name+'_per_gene_metrics.csv'),index=False)
    print(json.dumps(row),flush=True)
    del model,optimizer,best_state;torch.cuda.empty_cache()
    return row


def bootstrap_ci(y,score,seed=713,repeats=500):
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);aucs=[]
    for _ in range(repeats):
        ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        aucs.append(roc_auc_score(y[ix],score[ix]))
    return list(np.quantile(aucs,[.025,.975]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--seeds',default='42,123,2026');ap.add_argument('--holdout')
    ap.add_argument('--conditions',default='full,score_only,sequence_only,full_normalized,reduced,cadd,permuted_cyclic,permuted_reverse,context_change,plus_alphamissense')
    args=ap.parse_args();start=time.perf_counter();torch.set_num_threads(4)
    ip=Path(args.input_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((ip/'cohort_manifest.json').read_text());df=pd.read_parquet(ip/'five_gene_raw_union.parquet').reset_index(drop=True)
    features=meta['recommended_fixed_panel'];assert len(features)==46
    # AlphaMissense is appended in a fixed slot for the additional-score control.
    features=features+['AlphaMissense_score'];tr,va,te=make_split(df,args.holdout)
    split=df[['variant_id','variant_key','gene','label']].copy();split['split']='';split.loc[tr,'split']='train';split.loc[va,'split']='validation';split.loc[te,'split']='test';split.to_csv(out/'split_manifest.csv',index=False)
    X=df[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);imputer.fit(X[tr]);Xi=imputer.transform(X)
    mm=MinMaxScaler(feature_range=(0,255),clip=True);mm.fit(Xi[tr]);scaled=np.rint(mm.transform(Xi)).astype(np.uint8)
    jwrite(out/'preprocessing.json',dict(feature_columns=features,imputation='training-set median; empty training feature=0',
        imputation_values=imputer.statistics_.tolist(),training_minima=mm.data_min_.tolist(),training_maxima=mm.data_max_.tolist(),
        n_train=len(tr),n_validation=len(va),n_test=len(te),split_seed=123,holdout_gene=args.holdout,
        input_sha256=hashlib.sha256((ip/'five_gene_raw_union.parquet').read_bytes()).hexdigest(),
        architecture='torchvision ResNet34 weights=None two logits',optimizer='SGD lr=.025 momentum=.9',
        epochs=args.epochs,batch_size=64,training='inverse-class-frequency sampling with replacement; best validation AUROC checkpoint',
        image='Explicit RGB score/reference/alternate; native gene-interval square, zero padding, PIL bilinear resize300; no OpenCV BGR roundtrip',
        score_scaling='per-feature MinMax on training only, rounded uint8, clipped test/validation to training range',
        probability='softmax(logits)[1]; binary probability>0.5; no clinical calibration claimed',
        versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,pandas=pd.__version__),
        GPU=torch.cuda.get_device_name(0)))
    canonical=['SIFT_score','SIFT4G_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','MutationTaster_score','MutationAssessor_score',
        'PROVEAN_converted_rankscore','VEST4_score','MetaSVM_rankscore','MetaLR_score','MetaRNN_score','M-CAP_score','REVEL_score',
        'MutPred_score','MVP_score','MPC_score','PrimateAI_score','DEOGEN2_score','BayesDel_addAF_score','BayesDel_noAF_rankscore',
        'ClinPred_score','LIST-S2_score','CADD_phred','DANN_score']
    canonical=[c for c in canonical if c in features]
    jwrite(out/'reduced_panel.json',dict(columns=canonical,n=len(canonical),policy='One native score per retained predictor/configuration; CADD PHRED only; positions of removed features are zeroed before resize.'))
    base=scaled.copy();base[:,-1]=0
    conditions=args.conditions.split(',');seeds=[int(s) for s in args.seeds.split(',')];allrows=[]
    for cond in conditions:
        mapping=dict(A=1,T=2,C=3,G=4);s=base.copy();tag=cond
        if cond=='reduced':
            s[:,[i for i,c in enumerate(features) if c not in canonical]]=0
        if cond in ['cadd','cadd_score_only']:
            s[:,[i for i,c in enumerate(features) if c!='CADD_phred']]=0
            tag='cadd'
        if cond=='plus_alphamissense':s=scaled.copy()
        if cond=='permuted_cyclic':mapping=dict(A=2,T=3,C=4,G=1)
        if cond=='permuted_reverse':mapping=dict(A=4,T=3,C=2,G=1)
        if cond=='context_change':tag='change'
        if cond=='full_native_scaled':mapping=dict(A=64,T=128,C=191,G=255);tag='native_scaled'
        if cond=='context_change_native_scaled':mapping=dict(A=64,T=128,C=191,G=255);tag='change_native_scaled'
        if cond in ['full','score_only','sequence_only','full_normalized']:tag='base'
        cache=build_images(df,s,features,meta['reference_validation'],out,mapping,tag)
        for seed in seeds:
            allrows.append(fit_model(cache,df,tr,va,te,cond,seed,out,scaled,features,canonical,args.epochs))
            pd.DataFrame(allrows).to_csv(out/'cnn_metrics.csv',index=False)
        del cache
    # Matched tabular controls on the same train/validation/test rows and only the
    # original 46 fields. These controls do NOT borrow old CNN output files.
    Xt=Xi[:,:46];std=StandardScaler().fit(Xt[tr]);Xs=std.transform(Xt)
    y=df.label.to_numpy(dtype=int);tabs=[]
    for seed in seeds:
        for name in ['logistic_regression','xgboost']:
            if name=='logistic_regression':model=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed).fit(Xs[tr],y[tr]);score=model.predict_proba(Xs[te])[:,1]
            else:
                model=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=1.,
                    scale_pos_weight=float(np.sum(y[tr]==0)/np.sum(y[tr]==1)),random_state=seed,n_jobs=4,eval_metric='logloss',early_stopping_rounds=30)
                model.fit(Xt[tr],y[tr],eval_set=[(Xt[va],y[va])],verbose=False);score=model.predict_proba(Xt[te])[:,1]
            pred=df.iloc[te][['variant_id','variant_key','gene','label']].copy();pred['prob_pathogenic']=score;pred.to_csv(out/(name+'_seed'+str(seed)+'_predictions.csv'),index=False)
            row=metrics(y[te],score);row.update(condition=name,seed=seed,n_train=len(tr),n_validation=len(va));tabs.append(row)
    pd.DataFrame(tabs).to_csv(out/'tabular_metrics.csv',index=False)
    # Resumed runs may add conditions. Gather all completed new fits, retaining
    # previously completed matched controls rather than overwriting their summary.
    allrows=[json.loads(p.read_text()) for p in out.glob('*_metrics.json')]
    combined=pd.concat([pd.DataFrame(allrows),pd.DataFrame(tabs)],ignore_index=True)
    combined.to_csv(out/'all_metrics.csv',index=False)
    means=combined.groupby('condition').agg(AUROC_mean=('AUROC','mean'),AUROC_sd=('AUROC','std'),AUPRC_mean=('AUPRC','mean'),accuracy_mean=('accuracy','mean'),n_test=('n','first'),seeds=('seed','count')).reset_index()
    for i,row in means.iterrows():
        p=out/(row.condition+'_seed42_predictions.csv')
        if p.exists():
            pr=pd.read_csv(p);ci=bootstrap_ci(pr.label.values,pr.prob_pathogenic.values)
            means.loc[i,'seed42_AUROC_CI_lower']=ci[0];means.loc[i,'seed42_AUROC_CI_upper']=ci[1]
    means.to_csv(out/'condition_summary.csv',index=False)
    # Correlation quantifies numerical redundancy without claiming necessity.
    corr=df[features[:46]].corr(method='spearman',min_periods=20);pairs=[]
    for i,a in enumerate(features[:46]):
        for b in features[i+1:46]:
            pairs.append(dict(feature1=a,feature2=b,Spearman=corr.loc[a,b],n_pairwise=int(df[[a,b]].notna().all(axis=1).sum())))
    pd.DataFrame(pairs).to_csv(out/'feature_pairwise_correlations.csv',index=False)
    jwrite(out/'completion.json',dict(complete=True,runtime_seconds=time.perf_counter()-start,
        conditions=sorted({r['condition'] for r in allrows}),seeds=seeds,scope=meta['scope'],n_cohort=len(df),holdout=args.holdout,
        cautions=['full_normalized rescales sequence amplitude AFTER uint8 resize and cannot restore lost variation',
        'full_native_scaled uses nucleotide bytes64/128/191/255 BEFORE resize',
        'context_change uses original low-amplitude alternate-only mask; context_change_native_scaled is the signal-preserving alternative',
        'cadd retains sequence context; cadd_score_only zeros sequence channels',
        'within-gene stratified pooled AUC can reflect gene-level label prevalence; per-gene metrics must also be reported']))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
