#!/usr/bin/env python3
"""Corrected somatic cohort for the DeepVarPred revision (2026-10-30).

Positives: strict-missense SNVs recorded in COSMIC Mutant Census v101 (GRCh37) as
"Confirmed somatic variant" in >=1 sample and NOT classified B/LB in ClinVar.
Negatives: strict-missense ClinVar B/LB SNVs in the same cancer genes that have NO
COSMIC record of any somatic status.
Variants that are both COSMIC-recorded and ClinVar B/LB (contradictory labels in the
original design), or recorded in COSMIC only with non-confirmed status, are excluded.

Same predictor rules as the corrected germline cohort: fixed 46-field panel read from
the untouched *_filter.txt tables, mean of finite tokens, no imputation, no scaling.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, collections, hashlib, json, re, time, traceback
import numpy as np
import pandas as pd

AA20 = set('ACDEFGHIKLMNPQRSTVWY')
CLASS = {'Pathogenic': 'P/LP', 'Likely_pathogenic': 'P/LP', 'Pathogenic/Likely_pathogenic': 'P/LP',
         'Benign': 'B/LB', 'Likely_benign': 'B/LB', 'Benign/Likely_benign': 'B/LB',
         'Uncertain_significance': 'VUS',
         'Conflicting_interpretations_of_pathogenicity': 'Conflicting',
         'Conflicting_classifications_of_pathogenicity': 'Conflicting'}
COSMIC = '/home/bio1/workdata/GeneMutation2/cancer/Cosmic_MutantCensus_Tsv_v101_GRCh37/Cosmic_MutantCensus_v101_GRCh37.tsv'
META = ['variant_id', 'variant_key', 'gene', 'chr', 'pos', 'ref', 'alt', 'label', 'clinvar_class', 'clinvar_clnsig',
        'aaref', 'aaalt', 'ensembl_transcriptids', 'cosmic_confirmed_samples', 'cosmic_any_samples',
        'n_filter_rows', 'sequence_index_0based', 'reference_match']


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for ch in iter(lambda: f.read(1048576), b''):
            h.update(ch)
    return h.hexdigest()


def chr_clean(x):
    x = str(x).removeprefix('chr').removesuffix('.0')
    return {'23': 'X', '24': 'Y'}.get(x, x)


def add_key(df):
    pos = pd.to_numeric(df['hg19_pos(1-based)'], errors='coerce')
    valid = pos.notna() & (pos == pos.round())
    res = pd.Series(None, index=df.index, dtype=object)
    res.loc[valid] = (df.loc[valid, 'hg19_chr'].map(chr_clean) + ':' + pos.loc[valid].astype('int64').astype(str) + ':'
                      + df.loc[valid, 'ref'].str.upper() + '>' + df.loc[valid, 'alt'].str.upper())
    df['_key'] = res
    return df


def numeric(x):
    vals = []
    for token in re.split(r'[;|:]', str(x)):
        try:
            v = float(token.strip())
            if np.isfinite(v):
                vals.append(v)
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan


def load_cosmic():
    cols = ['GENE_SYMBOL', 'COSMIC_SAMPLE_ID', 'CHROMOSOME', 'GENOME_START', 'GENOMIC_WT_ALLELE', 'GENOMIC_MUT_ALLELE',
            'MUTATION_SOMATIC_STATUS']
    c = pd.read_csv(COSMIC, sep='\t', usecols=cols, dtype=str, low_memory=False).fillna('')
    snv = c.GENOMIC_WT_ALLELE.isin(list('ACGT')) & c.GENOMIC_MUT_ALLELE.isin(list('ACGT')) & c.CHROMOSOME.ne('') & c.GENOME_START.ne('')
    c = c[snv].copy()
    c['_key'] = c.CHROMOSOME.map(chr_clean) + ':' + c.GENOME_START.astype(int).astype(str) + ':' + c.GENOMIC_WT_ALLELE + '>' + c.GENOMIC_MUT_ALLELE
    conf = c[c.MUTATION_SOMATIC_STATUS.eq('Confirmed somatic variant')].groupby('_key').COSMIC_SAMPLE_ID.nunique()
    anyc = c.groupby('_key').COSMIC_SAMPLE_ID.nunique()
    return conf.to_dict(), anyc.to_dict(), dict(n_rows=len(c), n_keys=int(anyc.size), n_confirmed_keys=int(conf.size))


_CONF, _ANY = {}, {}


def init_worker(conf, anyc):
    global _CONF, _ANY
    _CONF, _ANY = conf, anyc


def worker(task):
    gene, srcstr, outstr, panel = task
    conf, anyc = _CONF, _ANY
    src = Path(srcstr) / gene
    od = Path(outstr) / '_per_gene'
    od.mkdir(exist_ok=True)
    fp = src / 'score' / f'{gene}_filter.txt'; cp = src / 'score' / f'{gene}_clnsig.txt'
    bp = src / 'ref' / f'{gene}_.bed'; fa = src / 'ref' / f'{gene}_ref.fa'
    stats = {'gene': gene, 'source_sha256': {str(p): sha256(p) for p in [fp, cp, bp, fa]}}
    head = pd.read_csv(fp, sep='\t', nrows=0).columns.tolist()
    present = [c for c in panel if c in head]
    if 'hg19_pos(1-based)' not in head:
        stats.update(status='missing_hg19_columns', retained=0)
        (od / f'{gene}.json').write_text(json.dumps(stats)); return stats
    cols = ['hg19_chr', 'hg19_pos(1-based)', 'ref', 'alt', 'clinvar_clnsig'] + present
    d = pd.read_csv(fp, sep='\t', usecols=cols, dtype=str, low_memory=False).fillna('')
    stats['raw_filter_rows'] = len(d)
    d['_class'] = d.clinvar_clnsig.map(CLASS)
    d = d[d._class.notna()].copy()
    add_key(d)
    keys = set(d._key.dropna())
    cc = [c for c in ['hg19_chr', 'hg19_pos(1-based)', 'ref', 'alt', 'aaref', 'aaalt', 'Ensembl_transcriptid']
          if c in pd.read_csv(cp, sep='\t', nrows=0).columns]
    c = pd.read_csv(cp, sep='\t', usecols=cc, dtype=str, low_memory=False).fillna('')
    add_key(c); c = c[c._key.isin(keys)].copy()
    c['_ok'] = [a in AA20 and b in AA20 and a != b for a, b in zip(c.aaref, c.aaalt)]
    anno = {k: dict(aa_ok=bool(z._ok.all()), aaref=';'.join(sorted(set(z.aaref))), aaalt=';'.join(sorted(set(z.aaalt))),
                    tx=';'.join(sorted(set(z.get('Ensembl_transcriptid', pd.Series(dtype=str)).astype(str)))))
            for k, z in c.groupby('_key', sort=False)}
    bed = bp.read_text().strip().split('\t'); chrom = chr_clean(bed[0]); start, end = map(int, bed[1:3])
    seq = ''.join(l.strip() for l in fa.read_text().splitlines() if not l.startswith('>')).upper()
    if len(seq) != end - start:
        stats.update(status='FASTA_BED_length_mismatch', retained=0)
        (od / f'{gene}.json').write_text(json.dumps(stats)); return stats
    stats['reference'] = dict(chr=chrom, bed_start_0based=start, bed_end_exclusive=end, sequence_length=len(seq),
                              bed_length=end - start, fasta_path=str(fa), bed_path=str(bp))
    rows, rejected = [], []
    for key, z in d[d._key.notna()].groupby('_key', sort=False):
        first = z.iloc[0]; classes = set(z._class); a = anno.get(key); reason = None
        nconf = int(conf.get(key, 0)); nany = int(anyc.get(key, 0))
        if a is None: reason = 'no_amino_acid_annotation'
        elif not a['aa_ok']: reason = 'non_missense_or_mixed_amino_acid_consequence'
        elif len(classes) > 1: reason = 'multiple_clinvar_classes'
        elif len(first.ref) != 1 or len(first.alt) != 1 or first.ref.upper() not in 'ACGT' or first.alt.upper() not in 'ACGT':
            reason = 'not_single_nucleotide_substitution'
        cls = next(iter(classes))
        label = None
        if reason is None:
            if cls == 'B/LB':
                if nany == 0: label = 0
                else: reason = 'cosmic_recorded_but_clinvar_benign'
            else:
                if nconf >= 1: label = 1
                elif nany >= 1: reason = 'cosmic_recorded_without_confirmed_somatic_status'
                else: reason = 'clinvar_nonbenign_without_cosmic_record'
        if reason is None:
            pos = int(float(first['hg19_pos(1-based)'])); ref = first.ref.upper(); alt = first.alt.upper(); idx = pos - (start + 1)
            if chr_clean(first.hg19_chr) != chrom or not 0 <= idx < len(seq): reason = 'outside_fasta_interval'
            elif seq[idx] != ref: reason = 'reference_base_mismatch'
        if reason:
            rejected.append(dict(gene=gene, variant_key=key, reason=reason, clinvar_class=cls, cosmic_confirmed_samples=nconf, cosmic_any_samples=nany))
            continue
        row = dict(variant_id=gene + ':' + key, variant_key=key, gene=gene, chr=chrom, pos=pos, ref=ref, alt=alt, label=label,
                   clinvar_class=cls, clinvar_clnsig=';'.join(sorted(set(z.clinvar_clnsig))), aaref=a['aaref'], aaalt=a['aaalt'],
                   ensembl_transcriptids=a['tx'], cosmic_confirmed_samples=nconf, cosmic_any_samples=nany, n_filter_rows=len(z),
                   sequence_index_0based=idx, reference_match=True)
        for col in panel:
            if col not in present: row[col] = np.nan; continue
            vals = [numeric(v) for v in z[col]]; vals = [v for v in vals if np.isfinite(v)]
            row[col] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    df = pd.DataFrame(rows, columns=META + panel)
    df.to_parquet(od / f'{gene}.parquet', index=False)
    pd.DataFrame(rejected).to_csv(od / f'{gene}.excluded.csv', index=False)
    stats.update(status='ok', retained=len(df), retained_by_label={int(k): int(v) for k, v in df.label.value_counts().items()},
                 excluded_reasons=dict(collections.Counter(x['reason'] for x in rejected)),
                 positive_clinvar_class=dict(collections.Counter(df.loc[df.label.eq(1), 'clinvar_class'])))
    (od / f'{gene}.json').write_text(json.dumps(stats)); return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/home/bio1/workdata/GeneMutation2/train_somatic')
    ap.add_argument('--output', required=True)
    ap.add_argument('--germline_manifest', default='/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/retraining/full_input/full_cohort_manifest.json')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); src = Path(args.source)
    panel = json.loads(Path(args.germline_manifest).read_text())['feature_columns']; assert len(panel) == 46
    t = time.time()
    conf, anyc, cstats = load_cosmic()
    print('cosmic', cstats, round(time.time() - t, 1), flush=True)
    genes = [g.name for g in sorted(src.iterdir()) if g.is_dir() and all(p.exists() for p in [
        g / 'score' / f'{g.name}_filter.txt', g / 'score' / f'{g.name}_clnsig.txt', g / 'ref' / f'{g.name}_ref.fa', g / 'ref' / f'{g.name}_.bed'])]
    print('eligible_genes', len(genes), flush=True)
    stats, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(conf, anyc)) as pool:
        futs = {pool.submit(worker, (g, str(src), str(out), panel)): g for g in genes}
        for i, f in enumerate(as_completed(futs), 1):
            try: stats.append(f.result())
            except Exception as e: errors.append(dict(gene=futs[f], error=repr(e), traceback=traceback.format_exc()))
            if i % 100 == 0 or i == len(futs): print('completed', i, len(futs), 'errors', len(errors), round(time.time() - t, 1), flush=True)
    d = pd.concat([pd.read_parquet(out / '_per_gene' / f"{s['gene']}.parquet") for s in stats if s.get('retained', 0) > 0], ignore_index=True)
    # A genomic allele annotated to several genes is kept only if it has one label; then every duplicated key is removed
    # (as in the germline runner) so that no allele can fall into two partitions.
    nlab = d.groupby('variant_key').label.nunique(); opposing = set(nlab[nlab > 1].index)
    dup = d.variant_key.duplicated(keep=False)
    d[d.variant_key.isin(opposing) | dup].to_csv(out / 'excluded_cross_gene_duplicates.csv', index=False)
    d = d[~(d.variant_key.isin(opposing) | dup)].sort_values(['gene', 'pos', 'ref', 'alt']).reset_index(drop=True)
    d.to_parquet(out / 'somatic_cohort.parquet', index=False); d.to_csv(out / 'somatic_cohort.csv', index=False)
    rej = [pd.read_csv(out / '_per_gene' / f"{s['gene']}.excluded.csv") for s in stats
           if (out / '_per_gene' / f"{s['gene']}.excluded.csv").exists() and (out / '_per_gene' / f"{s['gene']}.excluded.csv").stat().st_size > 1]
    rej = pd.concat(rej, ignore_index=True) if rej else pd.DataFrame()
    rej.to_csv(out / 'excluded_variants.csv', index=False)
    (out / 'source_hashes_and_gene_details.json').write_text(json.dumps(sorted(stats, key=lambda s: s['gene']), indent=1))
    (out / 'errors.json').write_text(json.dumps(errors, indent=1))
    summary = dict(scope='Corrected strict-missense somatic cohort (revision 2026-10-30)', cosmic=cstats, cosmic_file=COSMIC,
                   cosmic_sha256=sha256(COSMIC), source=str(src), n_eligible_genes=len(genes), n_errors=len(errors),
                   n_retained=len(d), n_genes=int(d.gene.nunique()), retained_by_label={int(k): int(v) for k, v in d.label.value_counts().items()},
                   positive_clinvar_class={k: int(v) for k, v in d[d.label.eq(1)].clinvar_class.value_counts().items()},
                   positive_recurrence={'>=2_confirmed_samples': int((d[d.label.eq(1)].cosmic_confirmed_samples >= 2).sum()),
                                        '>=3_confirmed_samples': int((d[d.label.eq(1)].cosmic_confirmed_samples >= 3).sum())},
                   excluded_reasons={k: int(v) for k, v in rej.reason.value_counts().items()} if len(rej) else {},
                   feature_columns=panel, elapsed_seconds=time.time() - t,
                   rules=dict(positive='strict missense; COSMIC Confirmed somatic variant in >=1 sample; ClinVar class not B/LB',
                              negative='strict missense; ClinVar B/LB; no COSMIC record of any status',
                              excluded='COSMIC-recorded ClinVar B/LB; COSMIC-recorded without confirmed status; ClinVar non-benign without COSMIC record',
                              predictors='fixed 46-field panel; finite-token mean; duplicate-row mean; no imputation; no scaling'))
    (out / 'somatic_cohort_manifest.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'feature_columns'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
