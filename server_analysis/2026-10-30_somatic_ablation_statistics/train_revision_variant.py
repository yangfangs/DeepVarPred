#!/usr/bin/env python3
"""Revision runner (2026-10-30): gene-disjoint DeepVarPred variants.

Same encoding, preprocessing and training protocol as
Revision_2026-10-26/retraining/global_gene_holdout/train_global_gene_holdout.py
(fixed score panel; training-only median imputation and MinMax scaling; native gene
interval packed row-wise, bilinear resize to 300x300; ResNet34 from random init;
SGD lr .025 momentum .9; inverse-class-frequency sampling; 10 epochs; best validation
AUROC checkpoint), generalised to (i) other cohorts (somatic), (ii) other score panels
and (iii) channel ablations that zero whole input channels at tensor level.
"""
import argparse, hashlib, json, math, random, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score

MAPPING = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
LOWER_IS_PATHOGENIC = {'SIFT_score', 'SIFT4G_score', 'ESM1b_score'}
CHANNELS = {'full': (0, 1, 2), 'score_only': (0,), 'sequence_only': (1, 2)}


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def jwrite(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    tmp.replace(p)


def metrics(y, s):
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float); both = len(np.unique(y)) == 2; pred = s > .5
    return dict(n=len(y), n_positive=int(y.sum()), n_negative=int(len(y) - y.sum()),
                AUROC=float(roc_auc_score(y, s)) if both else None,
                AUPRC=float(average_precision_score(y, s)) if np.any(y == 1) else None,
                accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)) if both else None,
                F1=float(f1_score(y, pred, zero_division=0)))


def assign_split(df, split_manifest):
    """Genes keep their germline partition; genes absent from the germline cohort are shuffled (rng 123) 60/20/20."""
    known = pd.read_csv(split_manifest).set_index('gene').split.to_dict()
    other = np.array(sorted(set(df.gene) - set(known)), dtype=object)
    np.random.default_rng(123).shuffle(other)
    n = len(other); a = int(.6 * n); b = int(.8 * n)
    extra = {g: ('train' if i < a else 'validation' if i < b else 'test') for i, g in enumerate(other)}
    assign = {**{g: known[g] for g in set(df.gene) & set(known)}, **extra}
    rows = [dict(gene=g, split=assign[g], source='germline_partition' if g in known else 'new_random_partition',
                 n=int((df.gene == g).sum()), n_positive=int(df.loc[df.gene == g, 'label'].sum())) for g in sorted(assign)]
    return df.gene.map(assign), pd.DataFrame(rows)


def encode_gene(task):
    gene, ids, positions, refs, alts, scores, meta, cache_path, cache_shape = task
    seq = ''.join(x.strip().upper() for x in Path(meta['fasta_path']).read_text().splitlines() if not x.startswith('>'))
    L = len(seq); W = round(math.sqrt(L)) + 1
    assert L == meta['bed_end_exclusive'] - meta['bed_start_0based']
    lut = np.zeros(256, dtype=np.uint8)
    for b, v in MAPPING.items(): lut[ord(b)] = v
    native = np.zeros(W * W, dtype=np.uint8); native[:L] = lut[np.frombuffer(seq.encode('ascii'), dtype=np.uint8)]; native = native.reshape(W, W)
    smallref = np.array(Image.fromarray(native).resize((300, 300), Image.Resampling.BILINEAR))
    cache = np.lib.format.open_memmap(cache_path, mode='r+', dtype=np.uint8, shape=tuple(cache_shape))
    same = 0
    for j, i in enumerate(ids):
        k = int(positions[j]); assert 0 <= k < L and seq[k] == refs[j] and refs[j] != alts[j]
        sc = np.zeros(W * W, dtype=np.uint8); sc[:scores.shape[1]] = scores[j]
        alt = native.copy(); alt.flat[k] = MAPPING[alts[j]]
        cache[i, 0] = np.array(Image.fromarray(sc.reshape(W, W)).resize((300, 300), Image.Resampling.BILINEAR))
        cache[i, 1] = smallref
        cache[i, 2] = np.array(Image.fromarray(alt).resize((300, 300), Image.Resampling.BILINEAR))
        same += int(np.array_equal(cache[i, 1], cache[i, 2]))
    cache.flush(); del cache
    return dict(gene=gene, n=len(ids), native_width=W, sequence_length=L, score_slots=int(scores.shape[1]), identical_reference_alternate_images=same)


def prepare(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); t = time.perf_counter()
    df = pd.read_parquet(args.cohort)
    if args.extra_features:
        extra = pd.read_parquet(args.extra_features)
        df = df.merge(extra, on='variant_id', how='left', validate='one_to_one')
    features = [c.strip() for c in args.features.split(',')] if not args.features.endswith('.json') else json.loads(Path(args.features).read_text())['feature_columns']
    assert all(f in df.columns for f in features), [f for f in features if f not in df.columns]
    if 'split' not in df.columns:
        dup = df.variant_key.duplicated(keep=False); df = df[~dup].sort_values(['gene', 'pos', 'ref', 'alt']).reset_index(drop=True)
        df['split'], gs = assign_split(df, args.split_manifest); gs.to_csv(out / 'gene_split_manifest.csv', index=False)
    ids = {s: np.flatnonzero(df.split.eq(s).to_numpy()) for s in ['train', 'validation', 'test']}
    for i, a in enumerate(ids):
        assert set(df.iloc[ids[a]].label) == {0, 1}, a
        for b in list(ids)[i + 1:]:
            assert not set(df.iloc[ids[a]].gene) & set(df.iloc[ids[b]].gene)
            assert not set(df.iloc[ids[a]].variant_key) & set(df.iloc[ids[b]].variant_key)
    df.to_parquet(out / 'analysis_input.parquet', index=False)
    X = df[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X[ids['train']]); Xi = imp.transform(X)
    mm = MinMaxScaler(feature_range=(0, 255), clip=True).fit(Xi[ids['train']]); scaled = np.rint(mm.transform(Xi)).astype(np.uint8)
    np.save(out / 'features_imputed.npy', Xi); np.save(out / 'features_scaled_uint8.npy', scaled)
    counts = {s: dict(n=len(ix), genes=int(df.iloc[ix].gene.nunique()), P=int(df.iloc[ix].label.sum()), B=int(len(ix) - df.iloc[ix].label.sum())) for s, ix in ids.items()}
    jwrite(out / 'preprocessing.json', dict(cohort=args.cohort, cohort_sha256=sha256(args.cohort), feature_columns=features, n_features=len(features),
           imputation='training-set median', imputation_values=imp.statistics_.tolist(), training_minima=mm.data_min_.tolist(),
           training_maxima=mm.data_max_.tolist(), counts=counts, split_manifest=args.split_manifest))
    refmeta = {r['gene']: r['reference'] for r in json.loads(Path(args.refmeta).read_text()) if 'reference' in r}
    shape = (len(df), 3, 300, 300); partial = out / 'images.partial.npy'
    cache = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8, shape=shape); cache.flush(); del cache
    tasks = []
    for gene, gidx in df.groupby('gene', sort=False).groups.items():
        ix = np.array(gidx, dtype=int); g = df.iloc[ix]
        tasks.append((gene, ix, g.sequence_index_0based.to_numpy(), g.ref.to_numpy(), g.alt.to_numpy(), scaled[ix], refmeta[gene], str(partial), shape))
    audits = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, f in enumerate(as_completed([pool.submit(encode_gene, x) for x in tasks]), 1):
            audits.append(f.result())
            if i % 200 == 0 or i == len(tasks): print(json.dumps(dict(stage='cache', genes=i, total=len(tasks), s=round(time.perf_counter() - t, 1))), flush=True)
    partial.replace(out / 'images.npy')
    pd.DataFrame(audits).to_csv(out / 'encoding_audit.csv', index=False)
    jwrite(out / 'preparation_complete.json', dict(complete=True, counts=counts, n=len(df), n_genes=int(df.gene.nunique()), features=features,
           identical_reference_alternate_images=int(sum(a['identical_reference_alternate_images'] for a in audits)), seconds=time.perf_counter() - t))
    print('PREPARE_COMPLETE', json.dumps(counts), flush=True)


def fit_cnn(cache, df, ids, out, tag, channels, epochs, seed):
    import torch, torchvision
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = torchvision.models.resnet34(weights=None, num_classes=2).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=.025, momentum=.9); amp = torch.cuda.amp.GradScaler()
    keep = torch.zeros(1, 3, 1, 1, device='cuda'); keep[0, list(CHANNELS[channels])] = 1.
    y = df.label.to_numpy(dtype=np.int64); tr, va, te = [ids[s] for s in ['train', 'validation', 'test']]

    def tensor(ix):
        return torch.from_numpy(np.array(cache[ix], copy=True)).to('cuda', non_blocking=True).float().div_(255).mul_(keep)

    def predict(ix):
        model.eval(); scores = []
        with torch.inference_mode():
            for st in range(0, len(ix), 64):
                with torch.autocast('cuda', dtype=torch.float16): logits = model(tensor(ix[st:st + 64]))
                scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        return np.asarray(scores, dtype=float)

    nclass = np.bincount(y[tr], minlength=2); p = 1 / nclass[y[tr]]; p = p / p.sum()
    hist = []; best = -1; best_epoch = None; best_state = None; started = time.perf_counter()
    for epoch in range(epochs):
        e0 = time.perf_counter(); model.train(); r = np.random.default_rng(seed + epoch * 10007)
        sample = r.choice(tr, size=len(tr), replace=True, p=p); loss_sum = 0.; seen = 0
        for st in range(0, len(sample), 64):
            ix = sample[st:st + 64]; target = torch.from_numpy(y[ix]).to('cuda'); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16): loss = torch.nn.functional.cross_entropy(model(tensor(ix)), target)
            amp.scale(loss).backward(); amp.step(opt); amp.update(); loss_sum += float(loss.detach().cpu()) * len(ix); seen += len(ix)
        vs = predict(va); auc = float(roc_auc_score(y[va], vs))
        row = dict(epoch=epoch + 1, train_loss=loss_sum / seen, validation_AUROC=auc, epoch_seconds=time.perf_counter() - e0); hist.append(row)
        if auc > best + 1e-12:
            best = auc; best_epoch = epoch + 1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out / f'{tag}_best_weights.pth')
        pd.DataFrame(hist).to_csv(out / f'{tag}_history.csv', index=False); print(json.dumps(dict(stage='epoch', tag=tag, **row, best_epoch=best_epoch)), flush=True)
    model.load_state_dict(best_state); score = predict(te)
    pred = df.iloc[te][['variant_id', 'variant_key', 'gene', 'label']].copy(); pred['prob_pathogenic'] = score
    pred.to_csv(out / f'{tag}_predictions.csv', index=False)
    res = dict(**metrics(y[te], score), condition=tag, channels=channels, seed=seed, n_train=len(tr), n_validation=len(va), best_epoch=best_epoch,
               best_validation_AUROC=best, runtime_seconds=time.perf_counter() - started)
    jwrite(out / f'{tag}_metrics.json', res)
    pd.DataFrame([dict(gene=g, **metrics(z.label, z.prob_pathogenic)) for g, z in pred.groupby('gene')]).to_csv(out / f'{tag}_per_gene_metrics.csv', index=False)
    del model, opt, best_state; torch.cuda.empty_cache(); return pred, res


def fit_tabular(prep, df, ids, out, tag, seed):
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    X = np.load(prep / 'features_imputed.npy', mmap_mode='r'); y = df.label.to_numpy(dtype=int); tr, va, te = [ids[s] for s in ['train', 'validation', 'test']]
    std = StandardScaler().fit(X[tr]); lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=seed).fit(std.transform(X[tr]), y[tr])
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=1., scale_pos_weight=float(np.sum(y[tr] == 0) / np.sum(y[tr] == 1)),
                        random_state=seed, n_jobs=4, eval_metric='auc', early_stopping_rounds=30, tree_method='hist')
    xgb.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    scores = {'logistic_regression': lr.predict_proba(std.transform(X[te]))[:, 1], 'xgboost': xgb.predict_proba(X[te])[:, 1]}; rows = []
    for name, s in scores.items():
        pred = df.iloc[te][['variant_id', 'variant_key', 'gene', 'label']].copy(); pred['prob_pathogenic'] = s
        pred.to_csv(out / f'{tag}_{name}_predictions.csv', index=False)
        rows.append(dict(**metrics(y[te], s), condition=f'{tag}_{name}', seed=seed, n_train=len(tr), n_validation=len(va)))
    return scores, rows


def native(df, te, features, cnn, tabs, out, tag):
    test = df.iloc[te]; y = test.label.to_numpy(dtype=int); c = cnn.prob_pathogenic.to_numpy(); assert test.variant_id.tolist() == cnn.variant_id.tolist(); rows = []
    for f in features:
        raw = pd.to_numeric(test[f], errors='coerce').to_numpy(dtype=float); ok = np.isfinite(raw)
        if ok.sum() == 0 or len(np.unique(y[ok])) < 2: continue
        o = -raw[ok] if f in LOWER_IS_PATHOGENIC else raw[ok]
        row = dict(feature=f, n=int(ok.sum()), P=int(y[ok].sum()), B=int(ok.sum() - y[ok].sum()), direction='lower' if f in LOWER_IS_PATHOGENIC else 'higher',
                   native_AUROC=float(roc_auc_score(y[ok], o)), CNN_AUROC_same_variants=float(roc_auc_score(y[ok], c[ok])))
        for k, s in tabs.items(): row[k + '_AUROC_same_variants'] = float(roc_auc_score(y[ok], s[ok]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f'{tag}_native_comparison.csv', index=False)


def train(args):
    import torch
    prep = Path(args.prepared_dir); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(prep / 'analysis_input.parquet'); features = json.loads((prep / 'preprocessing.json').read_text())['feature_columns']
    ids = {s: np.flatnonzero(df.split.eq(s).to_numpy()) for s in ['train', 'validation', 'test']}
    cache_file = prep / ('images.npy' if (prep / 'images.npy').exists() else 'images_base.npy')
    cache = np.load(cache_file, mmap_mode='r'); assert cache.shape == (len(df), 3, 300, 300)
    jwrite(out / f'{args.tag}_config.json', dict(prepared_dir=str(prep), cache=str(cache_file), channels=args.channels, features=features, seed=args.seed, epochs=args.epochs,
           architecture='torchvision ResNet34 weights=None, 2 logits', optimizer='SGD lr .025 momentum .9, batch 64, CUDA AMP', sampling='inverse class frequency',
           checkpoint='best validation AUROC', script_sha256=sha256(__file__), torch=torch.__version__, GPU=torch.cuda.get_device_name(0)))
    pred, res = fit_cnn(cache, df, ids, out, args.tag, args.channels, args.epochs, args.seed); rows = [res]
    tabs = {}
    if args.tabular:
        tabs, trows = fit_tabular(prep, df, ids, out, args.tag, args.seed); rows += trows
    native(df, ids['test'], features, pred, tabs, out, args.tag)
    pd.DataFrame(rows).to_csv(out / f'{args.tag}_all_metrics.csv', index=False)
    print('TRAIN_COMPLETE', args.tag, json.dumps({r['condition']: round(r['AUROC'], 4) for r in rows}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort'); ap.add_argument('--extra-features'); ap.add_argument('--features'); ap.add_argument('--split-manifest'); ap.add_argument('--refmeta')
    ap.add_argument('--prepared-dir'); ap.add_argument('--output-dir', required=True); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tag', default='cnn_full'); ap.add_argument('--channels', default='full', choices=list(CHANNELS)); ap.add_argument('--tabular', action='store_true')
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--seed', type=int, default=42)
    m = ap.add_mutually_exclusive_group(required=True); m.add_argument('--prepare', action='store_true'); m.add_argument('--train', action='store_true')
    a = ap.parse_args()
    prepare(a) if a.prepare else train(a)


if __name__ == '__main__':
    main()
