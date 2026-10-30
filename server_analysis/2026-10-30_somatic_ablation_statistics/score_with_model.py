#!/usr/bin/env python3
"""Score variants with a trained revision checkpoint, rebuilding images with that model's own
training-set preprocessing (median imputation values and MinMax ranges from its preprocessing.json).
Encoding identical to the training runners (native gene interval, bilinear resize to 300x300)."""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch, torchvision

MAPPING = {'A': 1, 'T': 2, 'C': 3, 'G': 4}


def scale(X, pp):
    imp = np.asarray(pp['imputation_values'], float); lo = np.asarray(pp['training_minima'], float); hi = np.asarray(pp['training_maxima'], float)
    X = np.where(np.isfinite(X), X, imp[None, :])
    rng = hi - lo; rng[rng == 0] = 1.0
    return np.rint(np.clip((X - lo) / rng * 255.0, 0, 255)).astype(np.uint8)


def images_for_gene(meta, idx, refs, alts, scores):
    seq = ''.join(x.strip().upper() for x in Path(meta['fasta_path']).read_text().splitlines() if not x.startswith('>'))
    L = len(seq); Wd = round(math.sqrt(L)) + 1
    lut = np.zeros(256, dtype=np.uint8)
    for b, v in MAPPING.items(): lut[ord(b)] = v
    native = np.zeros(Wd * Wd, dtype=np.uint8); native[:L] = lut[np.frombuffer(seq.encode('ascii'), dtype=np.uint8)]; native = native.reshape(Wd, Wd)
    smallref = np.array(Image.fromarray(native).resize((300, 300), Image.Resampling.BILINEAR))
    out = np.zeros((len(idx), 3, 300, 300), dtype=np.uint8)
    for j, k in enumerate(idx):
        k = int(k); assert seq[k] == refs[j], (meta['fasta_path'], k, seq[k], refs[j])
        sc = np.zeros(Wd * Wd, dtype=np.uint8); sc[:scores.shape[1]] = scores[j]
        alt = native.copy(); alt.flat[k] = MAPPING[alts[j]]
        out[j, 0] = np.array(Image.fromarray(sc.reshape(Wd, Wd)).resize((300, 300), Image.Resampling.BILINEAR))
        out[j, 1] = smallref
        out[j, 2] = np.array(Image.fromarray(alt).resize((300, 300), Image.Resampling.BILINEAR))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', required=True, help='parquet/csv with variant_id, gene, ref, alt, sequence_index_0based and score columns')
    ap.add_argument('--preprocessing', required=True); ap.add_argument('--refmeta', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    df = pd.read_parquet(a.variants) if a.variants.endswith('.parquet') else pd.read_csv(a.variants)
    pp = json.loads(Path(a.preprocessing).read_text()); feats = pp['feature_columns']
    for f in feats:
        if f not in df.columns: df[f] = np.nan
    X = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(float)
    S = scale(X, pp)
    refmeta = {r['gene']: r['reference'] for r in json.loads(Path(a.refmeta).read_text()) if 'reference' in r}
    model = torchvision.models.resnet34(weights=None, num_classes=2)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu')); model.to(a.device).eval()
    prob = np.full(len(df), np.nan)
    for gene, gidx in df.groupby('gene', sort=False).groups.items():
        ix = np.array(gidx, dtype=int); g = df.iloc[ix]
        imgs = images_for_gene(refmeta[gene], g.sequence_index_0based.to_numpy(), g.ref.to_numpy(), g.alt.to_numpy(), S[ix])
        with torch.inference_mode():
            for st in range(0, len(ix), 64):
                x = torch.from_numpy(imgs[st:st + 64]).to(a.device).float().div_(255)
                with torch.autocast(a.device, dtype=torch.float16, enabled=a.device == 'cuda'):
                    lo = model(x)
                prob[ix[st:st + 64]] = torch.softmax(lo.float(), 1)[:, 1].cpu().numpy()
    out = df[[c for c in ['variant_id', 'variant_key', 'gene', 'label', 'clinvar_class', 'split'] if c in df.columns]].copy(); out['prob_pathogenic'] = prob
    out.to_csv(a.output, index=False); print('scored', len(out), a.output)


if __name__ == '__main__':
    main()
