#!/usr/bin/env python3
"""Revision R2 (2026-11-17): score-channel layout sensitivity analysis (Reviewer 1, round 2, comment 2).

Identical data, gene split, preprocessing, network and training protocol as the primary gene-disjoint
germline model (Revision_2026-10-26/retraining/global_gene_holdout; runner
Revision_2026-10-30/scripts/train_revision_variant.py). Only the red (score) channel differs:

  resized      original encoding: 46 scaled scores in the first 46 positions of the native W x W
               gene matrix, bilinear resize to 300 x 300 (read from the cached images)
  fixed_row    the 46 scaled scores written unresized into the first 46 pixels of row 0 of the
               300 x 300 image (all other red pixels zero), i.e. identical for every gene length
  fixed_tiled  each of the 46 scores fills an 8 x 8 pixel cell in a fixed 48 x 64 block at the
               top-left corner (8 cells per row), all other red pixels zero

The green/blue (sequence) channels are the cached, resized gene-interval channels in all layouts.
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/bio1/workdata/GeneMutation2/Revision_2026-10-30/scripts')
from train_revision_variant import metrics, jwrite, sha256  # noqa: E402

CELL, PER_ROW = 8, 8


def score_plane(scaled, layout):
    """uint8 (n, 300, 300) red channel for the fixed layouts."""
    n, k = scaled.shape; red = np.zeros((n, 300, 300), dtype=np.uint8)
    if layout == 'fixed_row':
        red[:, 0, :k] = scaled
    else:
        for j in range(k):
            r, c = divmod(j, PER_ROW)
            red[:, r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = scaled[:, j, None, None]
    return red


def fit(cache, scaled, df, ids, out, tag, layout, epochs, seed):
    import torch, torchvision
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = torchvision.models.resnet34(weights=None, num_classes=2).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=.025, momentum=.9); amp = torch.cuda.amp.GradScaler()
    y = df.label.to_numpy(dtype=np.int64); tr, va, te = [ids[s] for s in ['train', 'validation', 'test']]

    def tensor(ix):
        x = np.array(cache[ix], copy=True)
        if layout != 'resized': x[:, 0] = score_plane(scaled[ix], layout)
        return torch.from_numpy(x).to('cuda', non_blocking=True).float().div_(255)

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
    res = dict(**metrics(y[te], score), condition=tag, score_layout=layout, seed=seed, n_train=len(tr), n_validation=len(va), best_epoch=best_epoch,
               best_validation_AUROC=best, runtime_seconds=time.perf_counter() - started)
    jwrite(out / f'{tag}_metrics.json', res)
    pd.DataFrame([dict(gene=g, **metrics(z.label, z.prob_pathogenic)) for g, z in pred.groupby('gene')]).to_csv(out / f'{tag}_per_gene_metrics.csv', index=False)
    print('TRAIN_COMPLETE', tag, round(res['AUROC'], 4), flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-dir', required=True); ap.add_argument('--output-dir', required=True)
    ap.add_argument('--layout', required=True, choices=['resized', 'fixed_row', 'fixed_tiled'])
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=10)
    a = ap.parse_args()
    prep = Path(a.prepared_dir); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); tag = f'{a.layout}_seed{a.seed}'
    df = pd.read_parquet(prep / 'analysis_input.parquet'); features = json.loads((prep / 'preprocessing.json').read_text())['feature_columns']
    ids = {s: np.flatnonzero(df.split.eq(s).to_numpy()) for s in ['train', 'validation', 'test']}
    cache_file = prep / ('images.npy' if (prep / 'images.npy').exists() else 'images_base.npy')
    cache = np.load(cache_file, mmap_mode='r'); assert cache.shape == (len(df), 3, 300, 300)
    scaled = np.load(prep / 'features_scaled_uint8.npy'); assert scaled.shape == (len(df), len(features))
    jwrite(out / f'{tag}_config.json', dict(prepared_dir=str(prep), cache=str(cache_file), score_layout=a.layout, cell=CELL, cells_per_row=PER_ROW,
           features=features, seed=a.seed, epochs=a.epochs, architecture='torchvision ResNet34 weights=None, 2 logits',
           optimizer='SGD lr .025 momentum .9, batch 64, CUDA AMP', sampling='inverse class frequency', checkpoint='best validation AUROC',
           script_sha256=sha256(__file__), torch=torch.__version__, GPU=torch.cuda.get_device_name(0)))
    fit(cache, scaled, df, ids, out, tag, a.layout, a.epochs, a.seed)


if __name__ == '__main__':
    main()
