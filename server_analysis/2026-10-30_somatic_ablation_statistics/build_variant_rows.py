#!/usr/bin/env python3
"""Build model-input rows (fixed 46-field panel, raw values, no imputation) for selected variants of one gene,
using the same aggregation rules as the corrected cohorts. Selection by cDNA change (HGVSc) or by hg19 key."""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

AA20 = set('ACDEFGHIKLMNPQRSTVWY')


def numeric(x):
    vals = []
    for t in re.split(r'[;|:]', str(x)):
        try:
            v = float(t.strip())
            if np.isfinite(v): vals.append(v)
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', default='/home/bio1/workdata/GeneMutation2/train_all'); ap.add_argument('--gene', required=True)
    ap.add_argument('--hgvsc', default=''); ap.add_argument('--panel', default='/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/retraining/full_input/full_cohort_manifest.json')
    ap.add_argument('--output', required=True); a = ap.parse_args()
    panel = json.loads(Path(a.panel).read_text())['feature_columns']
    g = a.gene; src = Path(a.source) / g
    c = pd.read_csv(src / 'score' / f'{g}_clnsig.txt', sep='\t', dtype=str, low_memory=False).fillna('')
    f = pd.read_csv(src / 'score' / f'{g}_filter.txt', sep='\t', dtype=str, low_memory=False).fillna('')
    def key(d): return d.hg19_chr.str.removeprefix('chr') + ':' + d['hg19_pos(1-based)'].astype(str) + ':' + d.ref.str.upper() + '>' + d.alt.str.upper()
    c['_key'] = key(c); f['_key'] = key(f)
    hcols = [x for x in c.columns if x.startswith('HGVSc')]
    bed = (src / 'ref' / f'{g}_.bed').read_text().split('\t'); start = int(bed[1])
    rows = []
    for h in [x.strip() for x in a.hgvsc.split(',') if x.strip()]:
        pat = re.compile(r'(^|[;:|])' + re.escape(h) + r'($|[;|])')
        hit = c[c[hcols].apply(lambda r: any(pat.search(v) for v in r), axis=1)]
        keys = sorted(set(hit._key))
        assert len(keys) == 1, (h, keys)
        k = keys[0]; z = f[f._key.eq(k)]; za = c[c._key.eq(k)]
        aa_ok = all(x in AA20 and y in AA20 and x != y for x, y in zip(za.aaref, za.aaalt))
        pos = int(k.split(':')[1]); ref, alt = k.split(':')[2].split('>')
        row = dict(variant_id=f'{g}:{k}', variant_key=k, gene=g, hgvsc=h, ref=ref, alt=alt, sequence_index_0based=pos - (start + 1),
                   clinvar_clnsig=';'.join(sorted(set(za.clinvar_clnsig))), aaref=';'.join(sorted(set(za.aaref))), aaalt=';'.join(sorted(set(za.aaalt))),
                   strict_missense=aa_ok, aapos=';'.join(sorted(set(za.get('aapos', pd.Series(dtype=str))))), n_filter_rows=len(z))
        for col in panel:
            vals = [numeric(v) for v in z[col]] if col in z.columns else []
            vals = [v for v in vals if np.isfinite(v)]; row[col] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.output, index=False); print(pd.DataFrame(rows)[['hgvsc', 'variant_key', 'clinvar_clnsig', 'aaref', 'aaalt', 'aapos', 'strict_missense', 'n_filter_rows']])


if __name__ == '__main__':
    main()
