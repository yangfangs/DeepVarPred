#!/usr/bin/env python3
"""Revision R2 statistics (2026-11-17), Reviewer 1 round-2 comments.

Q1  DeepVarPred (CNN) versus logistic regression / XGBoost trained on the same 46 scores: paired AUROC
    differences, probability-averaged combinations, strata by number of missing predictor values and by
    gene interval length, per-gene AUROC and Brier score (germline and somatic gene-disjoint test sets).
Q2  Score-channel layout sensitivity (resized versus fixed_row / fixed_tiled; three seeds each): per-seed
    AUROC, seed-averaged predictions with gene-cluster CI, paired differences, five held-out genes and
    gene-length strata.
Q3  Seed-averaged layout models versus each native predictor on identical variants (germline).
Same gene-cluster bootstrap as Revision_2026-10-30/scripts/revision_stats.py (1,000 resamples, seed 20261030).
Writes Revision_2026-11-17/analysis/*.csv.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/bio1/workdata/GeneMutation2/Revision_2026-10-30/scripts')
from revision_stats import boot_index, auc_w, ci  # noqa: E402

R30 = Path('/home/bio1/workdata/GeneMutation2/Revision_2026-10-30')
P26 = Path('/home/bio1/workdata/GeneMutation2/Revision_2026-10-26')
G = P26 / 'retraining/global_gene_holdout'
R = Path('/home/bio1/workdata/GeneMutation2/Revision_2026-11-17'); OUT = R / 'analysis'; OUT.mkdir(parents=True, exist_ok=True)
FIVE = ['BRCA1', 'BRCA2', 'DMD', 'PKD1', 'NF1']
MISS = [(0, 0, '0'), (1, 4, '1-4'), (5, 14, '5-14'), (15, 46, '>=15')]
LEN = [(0, 2e4, '<20 kb'), (2e4, 1e5, '20-100 kb'), (1e5, 5e5, '100-500 kb'), (5e5, 1e10, '>500 kb')]


def auc_ci(y, s, W):
    return roc_auc_score(y, s), *ci([auc_w(y, s, w) for w in W])


def diff_ci(y, a, b, W):
    return roc_auc_score(y, a) - roc_auc_score(y, b), *ci([auc_w(y, a, w) - auc_w(y, b, w) for w in W])


def test_frame(prep, preds_csv):
    feats = json.loads((prep / 'preprocessing.json').read_text())['feature_columns']
    d = pd.read_parquet(prep / 'analysis_input.parquet'); t = d[d.split == 'test'].reset_index(drop=True)
    t['n_missing'] = t[feats].apply(pd.to_numeric, errors='coerce').isna().sum(1).to_numpy()
    det = {}
    for src in [R30 / 'somatic/input/source_hashes_and_gene_details.json', P26 / 'retraining/full_input/source_hashes_and_gene_details.json']:
        det.update({r['gene']: r['reference'] for r in json.loads(src.read_text()) if 'reference' in r})
    t['gene_length'] = t.gene.map(lambda g: det[g]['sequence_length'] if g in det else np.nan)
    p = pd.read_csv(preds_csv).set_index('variant_id').reindex(t.variant_id)
    return t, p


def q1():
    rows, strata, genes_out = [], [], []
    for cohort, prep, pcsv in [('germline', G, R30 / 'analysis/stats/germline_test_predictions_all_models.csv'),
                               ('somatic', R30 / 'somatic/prepared', R30 / 'analysis/stats/somatic_test_predictions_all_models.csv')]:
        t, p = test_frame(prep, pcsv); y = t.label.to_numpy(int); genes = t.gene.to_numpy(); W = boot_index(genes)
        S = {k: p[k].to_numpy(float) for k in ['DeepVarPred', 'DeepVarPred score-only', 'Logistic regression', 'XGBoost']}
        assert all(np.isfinite(v).all() for v in S.values())
        S['Mean of DeepVarPred and XGBoost'] = (S['DeepVarPred'] + S['XGBoost']) / 2
        S['Mean of DeepVarPred and LR'] = (S['DeepVarPred'] + S['Logistic regression']) / 2
        S['Mean of LR and XGBoost'] = (S['Logistic regression'] + S['XGBoost']) / 2
        for k, s in S.items():
            a, lo, hi = auc_ci(y, s, W); d, dlo, dhi = diff_ci(y, s, S['XGBoost'], W) if k != 'XGBoost' else (0, 0, 0)
            rows.append(dict(cohort=cohort, model=k, n=len(y), genes=len(set(genes)), AUROC=a, AUROC_lo=lo, AUROC_hi=hi,
                             delta_vs_XGBoost=d, delta_lo=dlo, delta_hi=dhi, Brier=float(np.mean((s - y) ** 2)), accuracy_at_0_5=float(np.mean((s > .5) == y))))
        for col, bins, name in [('n_missing', MISS, 'missing predictor values'), ('gene_length', LEN, 'gene interval length')]:
            for lo_, hi_, lab in bins:
                m = ((t[col] >= lo_) & (t[col] <= hi_) if col == 'n_missing' else (t[col] >= lo_) & (t[col] < hi_)).to_numpy()
                if m.sum() == 0 or len(np.unique(y[m])) < 2: continue
                Wm = boot_index(genes[m]); r = dict(cohort=cohort, stratifier=name, stratum=lab, n=int(m.sum()), P=int(y[m].sum()), genes=len(set(genes[m])))
                for k in ['DeepVarPred', 'DeepVarPred score-only', 'Logistic regression', 'XGBoost']:
                    r[k], r[k + '_lo'], r[k + '_hi'] = auc_ci(y[m], S[k][m], Wm)
                r['DeepVarPred_minus_XGBoost'], r['diff_lo'], r['diff_hi'] = diff_ci(y[m], S['DeepVarPred'][m], S['XGBoost'][m], Wm)
                strata.append(r)
        per = []
        for g in sorted(set(genes)):
            m = genes == g
            if len(np.unique(y[m])) < 2: continue
            per.append(dict(cohort=cohort, gene=g, n=int(m.sum()), P=int(y[m].sum()), **{k: roc_auc_score(y[m], S[k][m]) for k in ['DeepVarPred', 'DeepVarPred score-only', 'Logistic regression', 'XGBoost']}))
        per = pd.DataFrame(per); genes_out.append(per)
        print(cohort, 'genes with both classes', len(per), 'CNN > XGB', int((per.DeepVarPred > per.XGBoost).sum()), 'CNN < XGB', int((per.DeepVarPred < per.XGBoost).sum()),
              'macro', per[['DeepVarPred', 'Logistic regression', 'XGBoost']].mean().round(4).to_dict(), flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'q1_cnn_vs_tabular.csv', index=False); pd.DataFrame(strata).to_csv(OUT / 'q1_strata.csv', index=False)
    pd.concat(genes_out).to_csv(OUT / 'q1_per_gene_auroc.csv', index=False)
    print(pd.DataFrame(rows)[['cohort', 'model', 'AUROC', 'AUROC_lo', 'AUROC_hi', 'delta_vs_XGBoost', 'delta_lo', 'delta_hi', 'Brier']].round(4).to_string(), flush=True)


def q2(cohort='germline'):
    if cohort == 'germline':
        prep, pcsv, base, ldir, tag = G, R30 / 'analysis/stats/germline_test_predictions_all_models.csv', G / 'cnn_full_seed42_predictions.csv', R / 'layout', 'q2'
    else:
        prep, pcsv, base, ldir, tag = (R30 / 'somatic/prepared', R30 / 'analysis/stats/somatic_test_predictions_all_models.csv',
                                       R30 / 'somatic/results/somatic_cnn_full_predictions.csv', R / 'layout_somatic', 'q2_somatic')
    t, p = test_frame(prep, pcsv); y = t.label.to_numpy(int); genes = t.gene.to_numpy(); W = boot_index(genes)
    ref = {'XGBoost': p['XGBoost'].to_numpy(float), 'Logistic regression': p['Logistic regression'].to_numpy(float)}
    runs = {('resized', 42): base}
    for f in sorted(ldir.glob('*_seed*_predictions.csv')):
        lay, seed = f.name.replace('_predictions.csv', '').rsplit('_seed', 1); runs[(lay, int(seed))] = f
    S = {}
    for (lay, seed), f in runs.items():
        z = pd.read_csv(f).set_index('variant_id').reindex(t.variant_id); assert z.prob_pathogenic.notna().all(), f; S[(lay, seed)] = z.prob_pathogenic.to_numpy(float)
    five_genes = [g for g in FIVE if (genes == g).any()]
    per_seed, summary, five = [], [], []
    for (lay, seed), s in sorted(S.items()):
        r = dict(layout=lay, seed=seed, AUROC=roc_auc_score(y, s))
        for g in five_genes: m = genes == g; r[g] = roc_auc_score(y[m], s[m])
        for lo_, hi_, lab in LEN:
            m = ((t.gene_length >= lo_) & (t.gene_length < hi_)).to_numpy()
            r[lab] = roc_auc_score(y[m], s[m]) if len(np.unique(y[m])) == 2 else np.nan
        per_seed.append(r)
    per_seed = pd.DataFrame(per_seed); per_seed.to_csv(OUT / f'{tag}_layout_per_seed.csv', index=False)
    print(per_seed.round(4).to_string(), flush=True)
    avg = {lay: np.mean([s for (l, _), s in S.items() if l == lay], axis=0) for lay in sorted({l for l, _ in S})}
    nseeds = {lay: sum(1 for l, _ in S if l == lay) for lay in avg}
    for lay, s in {**avg, **ref}.items():
        a, lo, hi = auc_ci(y, s, W); row = dict(model=lay, seeds=nseeds.get(lay, 1), AUROC_seed_averaged=a, lo=lo, hi=hi)
        if lay in avg:
            ps = per_seed[per_seed.layout == lay].AUROC; row.update(AUROC_mean_over_seeds=ps.mean(), AUROC_sd_over_seeds=ps.std(ddof=1) if len(ps) > 1 else np.nan)
        if lay != 'resized' and 'resized' in avg: row['delta_vs_resized'], row['delta_lo'], row['delta_hi'] = diff_ci(y, s, avg['resized'], W)
        row['delta_vs_XGBoost'], row['dx_lo'], row['dx_hi'] = diff_ci(y, s, ref['XGBoost'], W)
        long_ = (t.gene_length >= 5e5).to_numpy()
        if long_.sum() and len(np.unique(y[long_])) == 2:
            Wl = boot_index(genes[long_]); row['>500 kb'], row['>500 kb_lo'], row['>500 kb_hi'] = auc_ci(y[long_], s[long_], Wl)
            if lay != 'resized' and 'resized' in avg: row['>500 kb_delta'], row['>500 kb_dlo'], row['>500 kb_dhi'] = diff_ci(y[long_], s[long_], avg['resized'][long_], Wl)
        summary.append(row)
        for g in five_genes:  # single gene: variant-level bootstrap within the gene
            m = np.flatnonzero(genes == g); rng = np.random.default_rng(20261117); bs = []
            for _ in range(1000):
                ix = rng.choice(m, len(m), replace=True)
                if len(np.unique(y[ix])) == 2: bs.append(roc_auc_score(y[ix], s[ix]))
            five.append(dict(model=lay, gene=g, n=len(m), P=int(y[m].sum()), AUROC=roc_auc_score(y[m], s[m]), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    pd.DataFrame(summary).to_csv(OUT / f'{tag}_layout_summary.csv', index=False)
    print(pd.DataFrame(summary).round(4).to_string(), flush=True)
    if five:
        pd.DataFrame(five).to_csv(OUT / f'{tag}_layout_five_genes.csv', index=False)
        print(pd.DataFrame(five).pivot(index='gene', columns='model', values='AUROC').round(3).to_string(), flush=True)


def q3():
    """Seed-averaged layout models versus each native predictor on its own finite-score variants (same method as
    revision_stats.native_block, gene-cluster bootstrap)."""
    from revision_stats import native_block, CANON
    gi = pd.read_parquet(G / 'analysis_input.parquet'); test = gi[gi.split.eq('test')].reset_index(drop=True)
    ex = R30 / 'germline_extra/extra_AM_ESM1b.parquet'
    if ex.exists(): test = test.merge(pd.read_parquet(ex), on='variant_id', how='left')
    y = test.label.to_numpy(int); genes = test.gene.to_numpy(); W = boot_index(genes)
    rows = []
    for lay in ['resized', 'fixed_row', 'fixed_tiled']:
        fs = sorted((R / 'layout').glob(f'{lay}_seed*_predictions.csv'))
        if lay == 'resized': fs = [G / 'cnn_full_seed42_predictions.csv'] + fs
        s = np.mean([pd.read_csv(f).set_index('variant_id').reindex(test.variant_id).prob_pathogenic.to_numpy(float) for f in fs], axis=0)
        pr, _ = native_block('germline gene-disjoint test', test, y, genes, s, {}, W, CANON + ['AlphaMissense_score', 'ESM1b_score'])
        for r in pr: r.update(layout=lay, seeds=len(fs))
        rows += pr
    df = pd.DataFrame(rows); df.to_csv(OUT / 'q3_layout_vs_native_predictors.csv', index=False)
    print(df[['layout', 'feature', 'n', 'AUROC_A', 'AUROC_B', 'delta', 'delta_lo', 'delta_hi']].round(4).to_string(), flush=True)


if __name__ == '__main__':
    {'q1': q1, 'q2': q2, 'q2s': lambda: q2('somatic'), 'q3': q3}[sys.argv[1]]()
