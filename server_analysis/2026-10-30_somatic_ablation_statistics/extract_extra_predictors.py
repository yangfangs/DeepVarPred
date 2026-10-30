#!/usr/bin/env python3
"""Attach AlphaMissense and ESM1b scores (dbNSFP v5.0a, untouched *_clnsig.txt tables) to a cohort.

Values are the mean of finite per-transcript tokens; repeated rows for the same hg19 allele are averaged.
No imputation. Output: parquet with variant_id plus the requested score columns.
"""
import argparse, re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ['AlphaMissense_score', 'ESM1b_score']


def numeric(x):
    vals = []
    for t in re.split(r'[;|:]', str(x)):
        try:
            v = float(t.strip())
            if np.isfinite(v): vals.append(v)
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan


def one(task):
    gene, src, keys = task
    p = Path(src) / gene / 'score' / f'{gene}_clnsig.txt'
    head = pd.read_csv(p, sep='\t', nrows=0).columns
    use = ['hg19_chr', 'hg19_pos(1-based)', 'ref', 'alt'] + [c for c in COLS if c in head]
    d = pd.read_csv(p, sep='\t', usecols=use, dtype=str, low_memory=False).fillna('')
    pos = pd.to_numeric(d['hg19_pos(1-based)'], errors='coerce'); ok = pos.notna()
    d = d[ok].copy()
    d['variant_key'] = d.hg19_chr.str.removeprefix('chr') + ':' + pos[ok].astype('int64').astype(str) + ':' + d.ref.str.upper() + '>' + d.alt.str.upper()
    d = d[d.variant_key.isin(keys)]
    out = pd.DataFrame({'variant_key': sorted(keys)})
    for c in COLS:
        if c in d.columns and len(d):
            m = d.assign(_v=d[c].map(numeric)).groupby('variant_key')._v.mean()
            out[c] = out.variant_key.map(m)
        else:
            out[c] = np.nan
    out['variant_id'] = gene + ':' + out.variant_key
    return out[['variant_id'] + COLS]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', required=True); ap.add_argument('--source', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
    df = pd.read_parquet(a.cohort, columns=['variant_id', 'variant_key', 'gene'])
    tasks = [(g, a.source, set(z.variant_key)) for g, z in df.groupby('gene')]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = pd.concat(list(pool.map(one, tasks, chunksize=8)), ignore_index=True)
    res = res[res.variant_id.isin(set(df.variant_id))].drop_duplicates('variant_id')
    assert len(res) == len(df), (len(res), len(df))
    res.to_parquet(a.output, index=False)
    print({c: round(float(res[c].notna().mean()), 4) for c in COLS}, len(res))


if __name__ == '__main__':
    main()
