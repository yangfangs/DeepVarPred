#!/usr/bin/env python3
"""New, prospectively specified gene-disjoint revision experiment.

Original inputs are read only. --prepare-only never initializes a CUDA model.
Historical prediction exports and old checkpoints are never read by this runner.
"""
import argparse, hashlib, json, math, os, random, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score

EXPECTED_INPUT_SHA='246a5507a9c7d8c5150fcb4f27121c02983c2b2a1da760fa3e27b0b864b85018'
FORCED_TEST_GENES=['BRCA1','BRCA2','DMD','PKD1','NF1']
MAPPING={'A':1,'T':2,'C':3,'G':4}
ENCODING='RGB score/reference/alternate; native W=round(sqrt(gene interval length))+1; zero padding; PIL bilinear uint8 300x300; ToTensor /255; no BGR conversion'
LOWER_IS_PATHOGENIC={'SIFT_score','SIFT4G_score'}
CANONICAL=['SIFT_score','SIFT4G_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','MutationTaster_score','MutationAssessor_score','PROVEAN_converted_rankscore','VEST4_score','MetaSVM_rankscore','MetaLR_score','MetaRNN_score','M-CAP_score','REVEL_score','MutPred_score','MVP_score','MPC_score','PrimateAI_score','DEOGEN2_score','BayesDel_addAF_score','BayesDel_noAF_rankscore','ClinPred_score','LIST-S2_score','CADD_phred','DANN_score']


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def jwrite(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    tmp.replace(p)


def metrics(y,s):
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float);assert len(y)==len(s) and np.isfinite(s).all()
    both=len(np.unique(y))==2;pred=s>.5
    return dict(n=len(y),n_positive=int(y.sum()),n_negative=int(len(y)-y.sum()),
        AUROC=float(roc_auc_score(y,s)) if both else None,
        AUPRC=float(average_precision_score(y,s)) if np.any(y==1) else None,
        accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else None,
        F1=float(f1_score(y,pred,zero_division=0)))


def make_split(df):
    assert df.variant_key.is_unique and df.variant_id.is_unique
    genes=np.array(sorted(df.gene.unique()),dtype=object)
    np.random.default_rng(123).shuffle(genes)
    n=len(genes);a=int(.6*n);b=int(.8*n)
    initial={g:('train' if i<a else 'validation' if i<b else 'test') for i,g in enumerate(genes)}
    assignment=initial.copy()
    assert set(FORCED_TEST_GENES).issubset(assignment)
    for gene in FORCED_TEST_GENES:assignment[gene]='test'
    split=df.gene.map(assignment)
    ids={name:np.flatnonzero(split.eq(name).to_numpy()) for name in ['train','validation','test']}
    for i,x in enumerate(ids):
        assert len(ids[x])>0 and set(df.iloc[ids[x]].label)=={0,1}
        for z in list(ids)[i+1:]:
            assert not set(df.iloc[ids[x]].gene)&set(df.iloc[ids[z]].gene)
            assert not set(df.iloc[ids[x]].variant_key)&set(df.iloc[ids[z]].variant_key)
    assert all(split[df.gene.eq(g)].eq('test').all() for g in FORCED_TEST_GENES)
    rows=[]
    for i,gene in enumerate(genes):
        z=df[df.gene.eq(gene)]
        rows.append(dict(gene=gene,shuffle_index=i,initial_split=initial[gene],split=assignment[gene],forced_test=gene in FORCED_TEST_GENES,n=len(z),n_positive=int(z.label.sum()),n_negative=int(len(z)-z.label.sum())))
    return split,ids,pd.DataFrame(rows)


def encode_gene(task):
    gene,ids,positions,refs,alts,scores,meta,cache_path,cache_shape,config_sha,audit_dir=task
    seq=''.join(x.strip().upper() for x in Path(meta['fasta_path']).read_text().splitlines() if not x.startswith('>'))
    L=len(seq);W=round(math.sqrt(L))+1;assert L==meta['bed_length'] and L==meta['bed_end_exclusive']-meta['bed_start_0based']
    lut=np.zeros(256,dtype=np.uint8)
    for b,v in MAPPING.items():lut[ord(b)]=v
    codes=lut[np.frombuffer(seq.encode('ascii'),dtype=np.uint8)]
    native=np.zeros(W*W,dtype=np.uint8);native[:L]=codes;native=native.reshape(W,W)
    smallref=np.array(Image.fromarray(native).resize((300,300),Image.Resampling.BILINEAR))
    cache=np.lib.format.open_memmap(cache_path,mode='r+',dtype=np.uint8,shape=tuple(cache_shape))
    same=0;score_nonzero=0;alt_diff=0
    for j,i in enumerate(ids):
        k=int(positions[j]);assert 0<=k<L and seq[k]==refs[j] and refs[j]!=alts[j] and alts[j] in MAPPING
        sc=np.zeros(W*W,dtype=np.uint8);sc[:scores.shape[1]]=scores[j]
        smallscore=np.array(Image.fromarray(sc.reshape(W,W)).resize((300,300),Image.Resampling.BILINEAR))
        alt=native.copy();alt.flat[k]=MAPPING[alts[j]]
        smallalt=np.array(Image.fromarray(alt).resize((300,300),Image.Resampling.BILINEAR))
        cache[i,0]=smallscore;cache[i,1]=smallref;cache[i,2]=smallalt
        same+=int(np.array_equal(smallref,smallalt));score_nonzero+=int(np.count_nonzero(smallscore));alt_diff+=int(np.count_nonzero(smallref!=smallalt))
    cache.flush();del cache
    result=dict(gene=gene,n=len(ids),native_width=W,sequence_length=L,score_slots=scores.shape[1],native_score_slot_fraction=scores.shape[1]/(W*W),mean_resized_score_nonzero_fraction=score_nonzero/(len(ids)*90000),identical_reference_alternate_images=same,mean_reference_alternate_different_pixels=alt_diff/len(ids),unknown_reference_bases=int(np.sum(codes==0)),cache_config_sha256=config_sha)
    jwrite(Path(audit_dir)/(gene+'.json'),result)
    return result


def build_cache(df,scaled,features,refmeta,out,workers):
    p=out/'images_base.npy';auditfile=out/'encoding_audit_base.json'
    config=dict(input_sha256=EXPECTED_INPUT_SHA,n=len(df),variant_order_sha256=hashlib.sha256('\n'.join(df.variant_id).encode()).hexdigest(),feature_columns=features,mapping=MAPPING,encoding=ENCODING,scaled_sha256=hashlib.sha256(scaled.tobytes()).hexdigest())
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    shape=(len(df),3,300,300)
    if p.exists() and auditfile.exists():
        audit=json.loads(auditfile.read_text());assert audit['cache_config_sha256']==config_sha
        cache=np.load(p,mmap_mode='r');assert cache.shape==shape and cache.dtype==np.uint8
        return audit
    partial=out/'images_base.partial.npy';ad=out/'_encoding_gene_audits';ad.mkdir(exist_ok=True)
    # A partial cache may resume only when its separately frozen config matches.
    configpath=out/'cache_config.json';resume=partial.exists() and configpath.exists() and json.loads(configpath.read_text())==dict(config,cache_config_sha256=config_sha)
    if not resume:
        cache=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint8,shape=shape);cache.flush();del cache
        for old in ad.glob('*.json'):old.unlink()
        jwrite(configpath,dict(config,cache_config_sha256=config_sha))
    audits=[];tasks=[]
    for gene,gidx in df.groupby('gene',sort=False).groups.items():
        marker=ad/(gene+'.json')
        if resume and marker.exists():
            r=json.loads(marker.read_text());assert r['cache_config_sha256']==config_sha;audits.append(r);continue
        ix=np.array(gidx,dtype=int);g=df.iloc[ix]
        tasks.append((gene,ix,g.sequence_index_0based.to_numpy(),g.ref.to_numpy(),g.alt.to_numpy(),scaled[ix],refmeta[gene],str(partial),shape,config_sha,str(ad)))
    started=time.perf_counter();done_rows=sum(r['n'] for r in audits)
    print(json.dumps(dict(stage='cache_start',shape=shape,bytes=int(np.prod(shape)),workers=workers,resumed_genes=len(audits),genes_to_encode=len(tasks))),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(encode_gene,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();audits.append(r);done_rows+=r['n']
            if i%100==0 or i==len(tasks):print(json.dumps(dict(stage='cache_progress',genes_done=len(audits),genes_total=df.gene.nunique(),rows_done=done_rows,seconds=round(time.perf_counter()-started,1))),flush=True)
    assert len(audits)==df.gene.nunique() and sum(r['n'] for r in audits)==len(df)
    partial.replace(p)
    audit=dict(cache_config_sha256=config_sha,encoding=ENCODING,mapping=MAPPING,genes=sorted(audits,key=lambda x:x['gene']),identical_reference_alternate_images=sum(x['identical_reference_alternate_images'] for x in audits),n_variants=len(df),mean_score_nonzero_fraction=sum(x['mean_resized_score_nonzero_fraction']*x['n'] for x in audits)/len(df),runtime_seconds=time.perf_counter()-started)
    jwrite(auditfile,audit);return audit


def prepare(ip,out,workers):
    started=time.perf_counter();source=ip/'full_cohort.parquet';assert sha256(source)==EXPECTED_INPUT_SHA,'Input cohort SHA mismatch'
    meta=json.loads((ip/'full_cohort_manifest.json').read_text());features=meta['feature_columns']
    assert len(features)==46 and len(set(features))==46
    assert not any(c.startswith(('clinvar_','gnomAD','ALFA','TOPMed','1000G')) for c in features)
    raw=pd.read_parquet(source);duplicate=raw.variant_key.duplicated(keep=False)
    raw[duplicate].to_csv(out/'excluded_all_duplicate_variant_keys.csv',index=False)
    df=raw[~duplicate].sort_values(['gene','pos','ref','alt']).reset_index(drop=True)
    assert df.variant_key.is_unique and df.reference_match.all()
    split,ids,gs=make_split(df);df['split']=split
    df[['variant_id','variant_key','gene','label','split']].to_csv(out/'split_manifest.csv',index=False)
    gs.to_csv(out/'gene_split_manifest.csv',index=False);df.to_parquet(out/'analysis_input.parquet',index=False)
    refdetails=json.loads((ip/'source_hashes_and_gene_details.json').read_text());refmeta={r['gene']:r['reference'] for r in refdetails if 'reference' in r}
    assert set(df.gene).issubset(refmeta)
    X=df[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X[ids['train']]);Xi=imp.transform(X)
    mm=MinMaxScaler(feature_range=(0,255),clip=True).fit(Xi[ids['train']]);scaled=np.rint(mm.transform(Xi)).astype(np.uint8)
    assert Xi.shape==(len(df),46) and np.isfinite(Xi).all()
    np.save(out/'features_imputed.npy',Xi);np.save(out/'features_scaled_uint8.npy',scaled)
    import joblib
    joblib.dump(dict(imputer=imp,scaler=mm,feature_columns=features),out/'preprocessing.joblib')
    counts={s:dict(n=len(ix),genes=int(df.iloc[ix].gene.nunique()),P=int(df.iloc[ix].label.sum()),B=int(len(ix)-df.iloc[ix].label.sum())) for s,ix in ids.items()}
    preprocessing=dict(input_sha256=EXPECTED_INPUT_SHA,feature_columns=features,n_features=46,imputation='Training-set median; all-missing training feature=0',imputation_values=imp.statistics_.tolist(),training_minima=mm.data_min_.tolist(),training_maxima=mm.data_max_.tolist(),all_missing_training_features=[features[i] for i in np.flatnonzero(np.isnan(X[ids['train']]).all(axis=0))],scaling='Per-feature training-only MinMax[0,255], validation/test clipped, rounded uint8; no direction inversion',split_rule='Sorted gene names, default_rng(123) shuffle; indices <int(.6N) train, <int(.8N) validation, remainder test; five prespecified reviewer genes moved to test',forced_test_genes=FORCED_TEST_GENES,duplicate_rule='All records of every duplicated variant_key excluded globally, no preferred gene chosen',excluded_duplicate_rows=int(duplicate.sum()),excluded_duplicate_keys=int(raw.loc[duplicate,'variant_key'].nunique()),counts=counts,encoding=ENCODING,source_feature_availability='Inherited old per-gene >30% missingness column filtering; remaining unavailable panel values are NaN before train-only imputation',no_historical_scores=True)
    jwrite(out/'preprocessing.json',preprocessing)
    audit=build_cache(df,scaled,features,refmeta,out,workers)
    # Hash full cache once to freeze the tensor used by later GPU training.
    hashes={n:sha256(out/n) for n in ['analysis_input.parquet','split_manifest.csv','gene_split_manifest.csv','preprocessing.json','features_imputed.npy','features_scaled_uint8.npy','images_base.npy']}
    jwrite(out/'preparation_complete.json',dict(complete=True,input_sha256=EXPECTED_INPUT_SHA,script_sha256=sha256(__file__),n=len(df),n_genes=df.gene.nunique(),counts=counts,excluded_duplicate_rows=int(duplicate.sum()),excluded_duplicate_keys=int(raw.loc[duplicate,'variant_key'].nunique()),all_split_intersections_empty=True,artifact_sha256=hashes,cache_config_sha256=audit['cache_config_sha256'],runtime_seconds=time.perf_counter()-started))
    print(json.dumps(dict(stage='PREPARE_COMPLETE',counts=counts,runtime_seconds=time.perf_counter()-started)),flush=True)


def fit_cnn(cache,df,ids,out,epochs,seed):
    import torch, torchvision
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available(),'--train requires CUDA; preparation remains CPU-only'
    model=torchvision.models.resnet34(weights=None,num_classes=2).cuda();opt=torch.optim.SGD(model.parameters(),lr=.025,momentum=.9);amp=torch.cuda.amp.GradScaler()
    y=df.label.to_numpy(dtype=np.int64);tr,va,te=[ids[s] for s in ['train','validation','test']]
    def tensor(ix):return torch.from_numpy(np.array(cache[ix],copy=True)).to('cuda',non_blocking=True).float().div_(255)
    def predict(ix):
        model.eval();scores=[]
        with torch.inference_mode():
            for st in range(0,len(ix),64):
                with torch.autocast('cuda',dtype=torch.float16):logits=model(tensor(ix[st:st+64]))
                scores.extend(torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
        return np.asarray(scores,dtype=float)
    nclass=np.bincount(y[tr],minlength=2);p=(1/nclass[y[tr]]);p=p/p.sum();hist=[];best=-1;best_epoch=None;best_state=None;started=time.perf_counter()
    for epoch in range(epochs):
        epstart=time.perf_counter();model.train();r=np.random.default_rng(seed+epoch*10007);sample=r.choice(tr,size=len(tr),replace=True,p=p);loss_sum=0.;seen=0
        for st in range(0,len(sample),64):
            ix=sample[st:st+64];target=torch.from_numpy(y[ix]).to('cuda');opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(tensor(ix)),target)
            amp.scale(loss).backward();amp.step(opt);amp.update();loss_sum+=float(loss.detach().cpu())*len(ix);seen+=len(ix)
            if (st//64+1)%100==0:print(json.dumps(dict(stage='train_progress',epoch=epoch+1,examples=seen,n_train=len(tr),seconds=round(time.perf_counter()-epstart,1))),flush=True)
        vs=predict(va);auc=float(roc_auc_score(y[va],vs));row=dict(epoch=epoch+1,train_loss=loss_sum/seen,validation_AUROC=auc,epoch_seconds=time.perf_counter()-epstart);hist.append(row)
        if auc>best+1e-12:
            best=auc;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,out/'cnn_full_seed42_best_weights.partial.pth');(out/'cnn_full_seed42_best_weights.partial.pth').replace(out/'cnn_full_seed42_best_weights.pth')
        pd.DataFrame(hist).to_csv(out/'cnn_full_seed42_history.csv',index=False);print(json.dumps(dict(stage='epoch_complete',**row,best_epoch=best_epoch)),flush=True)
    model.load_state_dict(best_state);score=predict(te)
    pred=df.iloc[te][['variant_id','variant_key','gene','label']].copy();pred['prob_pathogenic']=score;pred.to_csv(out/'cnn_full_seed42_predictions.csv',index=False)
    result=dict(**metrics(y[te],score),condition='cnn_full',seed=seed,n_train=len(tr),n_validation=len(va),best_epoch=best_epoch,best_validation_AUROC=best,runtime_seconds=time.perf_counter()-started)
    jwrite(out/'cnn_full_seed42_metrics.json',result)
    per=[]
    for gene,g in pred.groupby('gene'):
        per.append(dict(gene=gene,**metrics(g.label,g.prob_pathogenic)))
    pd.DataFrame(per).to_csv(out/'cnn_full_seed42_per_gene_metrics.csv',index=False)
    del model,opt,best_state;torch.cuda.empty_cache();return pred,result


def fit_tabular(df,ids,out,seed):
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    import joblib
    X=np.load(out/'features_imputed.npy',mmap_mode='r');y=df.label.to_numpy(dtype=int);tr,va,te=[ids[s] for s in ['train','validation','test']]
    std=StandardScaler().fit(X[tr]);lr=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed).fit(std.transform(X[tr]),y[tr]);joblib.dump(dict(model=lr,standard_scaler=std),out/'logistic_regression_seed42.joblib')
    xgb=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=1.,scale_pos_weight=float(np.sum(y[tr]==0)/np.sum(y[tr]==1)),random_state=seed,n_jobs=4,eval_metric='auc',early_stopping_rounds=30,tree_method='hist')
    xgb.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],verbose=False);xgb.save_model(out/'xgboost_seed42.json')
    scores={'logistic_regression':lr.predict_proba(std.transform(X[te]))[:,1],'xgboost':xgb.predict_proba(X[te])[:,1]};rows=[]
    for name,score in scores.items():
        pred=df.iloc[te][['variant_id','variant_key','gene','label']].copy();pred['prob_pathogenic']=score;pred.to_csv(out/(name+'_seed42_predictions.csv'),index=False)
        row=dict(**metrics(y[te],score),condition=name,seed=seed,n_train=len(tr),n_validation=len(va))
        if name=='xgboost':row.update(best_iteration=int(xgb.best_iteration),validation_metric='AUROC',n_estimators_max=300,max_depth=4,learning_rate=.05,early_stopping_rounds=30)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'tabular_metrics.csv',index=False);return scores,rows


def pairwise_native_comparison(df,te,features,cnn,tabular,out):
    # All native scores remain untouched: only finite raw test scores are eligible.
    # Report both raw-score AUROC and explicit known lower-is-damaging direction.
    test=df.iloc[te];y=test.label.to_numpy(dtype=int);cnnscore=cnn.prob_pathogenic.to_numpy();assert test.variant_id.tolist()==cnn.variant_id.tolist();rows=[]
    for feature in features:
        raw=pd.to_numeric(test[feature],errors='coerce').to_numpy(dtype=float);ok=np.isfinite(raw);both=len(np.unique(y[ok]))==2
        direction='lower' if feature in LOWER_IS_PATHOGENIC else 'higher';oriented=-raw[ok] if direction=='lower' else raw[ok]
        row=dict(feature=feature,is_canonical_representative=feature in CANONICAL,n_pairwise=int(ok.sum()),n_positive=int(y[ok].sum()),n_negative=int(ok.sum()-y[ok].sum()),native_direction=direction,native_AUROC_raw=float(roc_auc_score(y[ok],raw[ok])) if both else None,native_AUROC_direction_corrected=float(roc_auc_score(y[ok],oriented)) if both else None,CNN_AUROC_same_variants=float(roc_auc_score(y[ok],cnnscore[ok])) if both else None)
        for name,s in tabular.items():row[name+'_AUROC_same_variants']=float(roc_auc_score(y[ok],s[ok])) if both else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'pairwise_native_predictor_comparison.csv',index=False)
    jwrite(out/'pairwise_comparison_policy.json',dict(policy='Each native predictor uses only its own finite raw scores in the locked test set. CNN/LR/XGB AUROC are recomputed on exactly those same variants. Different predictors can have different N; no overall same-N claim.',directions='Original raw AUROC always retained; only SIFT_score/SIFT4G_score additionally use sign reversal for known lower-is-damaging native direction; training inputs were never reversed.'))


def train(ip,out,epochs,seed):
    import torch,torchvision,sklearn,xgboost
    assert seed==42,'The prespecified global run uses seed42'
    done=json.loads((out/'preparation_complete.json').read_text());assert done['complete'] and done['input_sha256']==EXPECTED_INPUT_SHA and sha256(ip/'full_cohort.parquet')==EXPECTED_INPUT_SHA
    # Verify all small fitted/split inputs; full tensor SHA was frozen in preparation.
    for name in ['analysis_input.parquet','split_manifest.csv','gene_split_manifest.csv','preprocessing.json','features_imputed.npy','features_scaled_uint8.npy']:assert sha256(out/name)==done['artifact_sha256'][name],name
    assert sha256(out/'images_base.npy')==done['artifact_sha256']['images_base.npy'],'Image tensor SHA mismatch'
    df=pd.read_parquet(out/'analysis_input.parquet');features=json.loads((out/'preprocessing.json').read_text())['feature_columns'];assert len(features)==46
    ids={s:np.flatnonzero(df.split.eq(s).to_numpy()) for s in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(df.iloc[ids[a]].gene)&set(df.iloc[ids[b]].gene) and not set(df.iloc[ids[a]].variant_key)&set(df.iloc[ids[b]].variant_key)
    cache=np.load(out/'images_base.npy',mmap_mode='r');assert cache.shape==(len(df),3,300,300) and cache.dtype==np.uint8
    jwrite(out/'training_config.json',dict(seed=seed,epochs=epochs,batch_size=64,architecture='torchvision ResNet34 weights=None two logits',precision='CUDA AMP float16',optimizer='SGD lr=.025 momentum=.9',sampling='inverse-class-frequency sampling with replacement; len(train) draws per epoch',checkpoint_rule='highest validation AUROC, earliest epoch on exact tie',probability='continuous softmax(logits.float())[1]; threshold>0.5 descriptive metrics',new_predictions_only=True,script_sha256=sha256(__file__),preparation_manifest_sha256=sha256(out/'preparation_complete.json'),versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__),GPU=torch.cuda.get_device_name(0)))
    pred,cnnrow=fit_cnn(cache,df,ids,out,epochs,seed);tabs,tabrows=fit_tabular(df,ids,out,seed);pairwise_native_comparison(df,ids['test'],features,pred,tabs,out)
    pd.DataFrame([cnnrow]+tabrows).to_csv(out/'all_metrics.csv',index=False)
    jwrite(out/'training_complete.json',dict(complete=True,epochs=epochs,seed=seed,n_cohort=len(df),n_test=len(ids['test']),scope='New fixed46 strict-missense gene-heldout revision experiment; not historical model reproduction',artifact_sha256={n:sha256(out/n) for n in ['cnn_full_seed42_predictions.csv','cnn_full_seed42_metrics.json','tabular_metrics.csv','pairwise_native_predictor_comparison.csv','all_metrics.csv']}))
    print('TRAIN_COMPLETE',out,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--seed',type=int,default=42)
    mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--train',action='store_true');args=ap.parse_args()
    ip=Path(args.input_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if args.prepare_only:prepare(ip,out,args.workers)
    else:train(ip,out,args.epochs,args.seed)

if __name__=='__main__':main()
