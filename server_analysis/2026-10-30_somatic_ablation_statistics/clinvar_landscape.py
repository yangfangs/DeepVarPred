#!/usr/bin/env python3
"""Strict-missense ClinVar landscape (dbNSFP v5.0a; ClinVar 2024-09-09) across the genes analysed."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
AA20 = set('ACDEFGHIKLMNPQRSTVWY')
SRC = Path('/home/bio1/workdata/GeneMutation2/train_all')
def cls(v):
    v = str(v)
    if v in ('Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'): return 'P/LP'
    if v in ('Benign', 'Likely_benign', 'Benign/Likely_benign'): return 'B/LB'
    if v == 'Uncertain_significance': return 'VUS'
    if v.startswith('Conflicting'): return 'Conflicting'
    if v in ('.', '', 'nan'): return None
    return 'Other'
def one(g):
    p = SRC / g / 'score' / f'{g}_dbNSFP.txt'
    if not p.exists() or p.stat().st_size == 0: return None
    try:
        d = pd.read_csv(p, sep='\t', usecols=['hg19_chr', 'hg19_pos(1-based)', 'ref', 'alt', 'aaref', 'aaalt', 'clinvar_clnsig'], dtype=str, low_memory=False).fillna('')
    except (pd.errors.EmptyDataError, ValueError):
        return None
    d = d[d.clinvar_clnsig.ne('.') & d.clinvar_clnsig.ne('') & d['hg19_pos(1-based)'].ne('.')]
    if d.empty: return None
    d['key'] = d.hg19_chr + ':' + d['hg19_pos(1-based)'] + ':' + d.ref + '>' + d.alt
    d['ok'] = [a in AA20 and b in AA20 and a != b for a, b in zip(d.aaref, d.aaalt)]
    agg = d.groupby('key').agg(chr=('hg19_chr', 'first'), ok=('ok', 'all'), clnsig=('clinvar_clnsig', 'first')).reset_index()
    agg = agg[agg.ok]; agg['class'] = agg.clnsig.map(cls); agg['gene'] = g
    return agg[agg['class'].notna()][['key', 'chr', 'class', 'gene']]
if __name__ == '__main__':
    genes = sorted(x.name for x in SRC.iterdir() if x.is_dir())
    with ProcessPoolExecutor(8) as ex: res = [r for r in ex.map(one, genes, chunksize=16) if r is not None]
    a = pd.concat(res, ignore_index=True)
    out = Path('/home/bio1/workdata/GeneMutation2/Revision_2026-10-30/analysis/landscape'); out.mkdir(parents=True, exist_ok=True)
    a[a.gene.eq('BRCA1')]['class'].value_counts().to_csv(out / 'BRCA1_strict_missense_clinvar_classes.csv')
    u = a.drop_duplicates('key')
    u.groupby(['chr', 'class']).size().unstack(fill_value=0).to_csv(out / 'strict_missense_clinvar_by_chromosome.csv')
    print(len(u), u['class'].value_counts().to_dict(), a.gene.nunique())
