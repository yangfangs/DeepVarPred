#!/usr/bin/env python3
"""Revision statistics (2026-10-30). Reads only revision outputs; writes analysis/stats/*.csv|json.
Gene-cluster bootstrap: 1,000 resamples of held-out genes (seed 20261030); all variants of a sampled gene
are retained with its multiplicity; paired comparisons use identical resamples and variant subsets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr

R = Path('/home/bio1/workdata/GeneMutation2/Revision_2026-10-30')
G = Path('/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/retraining/global_gene_holdout')
P26 = Path('/home/bio1/workdata/GeneMutation2/Revision_2026-10-26')
OUT = R / 'analysis' / 'stats'; OUT.mkdir(parents=True, exist_ok=True)
B = 1000; SEED = 20261030
CANON = ['SIFT_score', 'SIFT4G_score', 'Polyphen2_HDIV_score', 'Polyphen2_HVAR_score', 'MutationTaster_score', 'MutationAssessor_score',
         'PROVEAN_converted_rankscore', 'VEST4_score', 'MetaSVM_rankscore', 'MetaLR_score', 'MetaRNN_score', 'M-CAP_score', 'REVEL_score',
         'MutPred_score', 'MVP_score', 'MPC_score', 'PrimateAI_score', 'DEOGEN2_score', 'BayesDel_addAF_score', 'BayesDel_noAF_rankscore',
         'ClinPred_score', 'LIST-S2_score', 'CADD_phred', 'DANN_score']
LOWER = {'SIFT_score', 'SIFT4G_score', 'ESM1b_score'}
# Published / dbNSFP-documented binary cut-offs on the native field (">= cut" = damaging unless LOWER)
CUT = {'SIFT_score': 0.05, 'SIFT4G_score': 0.05, 'Polyphen2_HDIV_score': 0.5, 'Polyphen2_HVAR_score': 0.5, 'MutationTaster_score': 0.5,
       'MutationAssessor_score': 1.935, 'PROVEAN_converted_rankscore': 0.54382, 'VEST4_score': 0.5, 'MetaSVM_rankscore': 0.82257,
       'MetaLR_score': 0.5, 'MetaRNN_score': 0.5, 'M-CAP_score': 0.025, 'REVEL_score': 0.5, 'MutPred_score': 0.5, 'MVP_score': 0.7,
       'MPC_score': 2.0, 'PrimateAI_score': 0.803, 'DEOGEN2_score': 0.5, 'BayesDel_addAF_score': 0.0692655, 'ClinPred_score': 0.5,
       'LIST-S2_score': 0.85, 'CADD_phred': 20.0, 'DANN_score': None, 'BayesDel_noAF_rankscore': None, 'AlphaMissense_score': 0.564, 'ESM1b_score': None}
CUT_SOURCE = {'SIFT_score': 'dbNSFP v5.0a (SIFT_pred: score < 0.05 damaging)', 'SIFT4G_score': 'dbNSFP v5.0a (SIFT4G_pred: score < 0.05)',
              'Polyphen2_HDIV_score': 'dbNSFP v5.0a binary cut-off 0.5', 'Polyphen2_HVAR_score': 'dbNSFP v5.0a binary cut-off 0.5',
              'MutationTaster_score': 'dbNSFP v5.0a (D/N cut-off 0.5)', 'MutationAssessor_score': 'dbNSFP v5.0a (medium/high impact >= 1.935)',
              'PROVEAN_converted_rankscore': 'dbNSFP v5.0a (rankscore >= 0.54382, equivalent to PROVEAN <= -2.5)', 'VEST4_score': 'conventional 0.5 (no author-recommended cut-off)',
              'MetaSVM_rankscore': 'dbNSFP v5.0a (rankscore >= 0.82257, equivalent to MetaSVM score > 0)', 'MetaLR_score': 'dbNSFP v5.0a (0.5)',
              'MetaRNN_score': 'dbNSFP v5.0a (0.5)', 'M-CAP_score': 'authors (0.025)', 'REVEL_score': 'authors (0.5)', 'MutPred_score': 'conventional 0.5',
              'MVP_score': 'authors (0.7, constrained genes)', 'MPC_score': 'authors (2.0)', 'PrimateAI_score': 'authors (0.803)', 'DEOGEN2_score': 'authors (0.5)',
              'BayesDel_addAF_score': 'authors (0.0692655)', 'ClinPred_score': 'authors (0.5)', 'LIST-S2_score': 'authors (0.85)', 'CADD_phred': 'conventional PHRED 20',
              'DANN_score': 'no recommended cut-off; AUROC only', 'BayesDel_noAF_rankscore': 'cut-off defined on raw score only; AUROC only',
              'AlphaMissense_score': 'authors (likely pathogenic > 0.564)', 'ESM1b_score': 'authors recommend no cut-off; AUROC only'}


def boot_index(genes):
    ug = np.array(sorted(set(genes))); gid = pd.Index(ug).get_indexer(genes)
    rng = np.random.default_rng(SEED)
    return [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gid].astype(float) for _ in range(B)]


def auc_w(y, s, w):
    m = w > 0
    if len(np.unique(y[m])) < 2: return np.nan
    return roc_auc_score(y[m], s[m], sample_weight=w[m])


def ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)]); return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def model_table(name, y, genes, scores, W):
    rows = []
    for k, s in scores.items():
        a = roc_auc_score(y, s); ap = average_precision_score(y, s)
        bs = [auc_w(y, s, w) for w in W]; lo, hi = ci(bs)
        rows.append(dict(cohort=name, model=k, n=len(y), P=int(y.sum()), B=int(len(y) - y.sum()), genes=len(set(genes)), AUROC=a, AUROC_lo=lo, AUROC_hi=hi, AUPRC=ap))
    return rows


def paired(name, y, s_ref, s_other, W, label_ref, label_other, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    d0 = roc_auc_score(y[m], s_ref[m]) - roc_auc_score(y[m], s_other[m])
    ds = [auc_w(y[m], s_ref[m], w[m]) - auc_w(y[m], s_other[m], w[m]) for w in W]; lo, hi = ci(ds)
    return dict(cohort=name, A=label_ref, B=label_other, n=int(m.sum()), AUROC_A=roc_auc_score(y[m], s_ref[m]), AUROC_B=roc_auc_score(y[m], s_other[m]), delta=d0, delta_lo=lo, delta_hi=hi)


def binary(y, s, cut, lower=False):
    pred = s <= cut if lower else s >= cut
    tp = np.sum(pred & (y == 1)); tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(sensitivity=sens, specificity=spec, accuracy=(tp + tn) / len(y), balanced_accuracy=(sens + spec) / 2, F1=2 * tp / (2 * tp + fp + fn))


def native_block(name, df, y, genes, cnn, extra, W, feats):
    pair_rows, thr_rows = [], []
    for f in feats:
        if f not in df.columns: continue
        raw = pd.to_numeric(df[f], errors='coerce').to_numpy(float); ok = np.isfinite(raw)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2: continue
        o = -raw if f in LOWER else raw
        r = paired(name, y, cnn, np.where(ok, o, 0), W, 'DeepVarPred', f, ok); r['feature'] = f
        for k, s in extra.items(): r[k + '_AUROC_same_variants'] = roc_auc_score(y[ok], s[ok])
        pair_rows.append(r)
        cut = CUT.get(f)
        row = dict(cohort=name, method=f, threshold=('<= ' if f in LOWER else '>= ') + str(cut) if cut is not None else 'none', threshold_source=CUT_SOURCE.get(f, ''), n=int(ok.sum()))
        if cut is not None: row.update(binary(y[ok], raw[ok], cut, f in LOWER))
        thr_rows.append(row)
    return pair_rows, thr_rows


def main():
    summary = {}
    # ---------------- germline primary + extra conditions
    gi = pd.read_parquet(G / 'analysis_input.parquet'); test = gi[gi.split.eq('test')].reset_index(drop=True)
    preds = {'DeepVarPred': G / 'cnn_full_seed42_predictions.csv', 'Logistic regression': G / 'logistic_regression_seed42_predictions.csv', 'XGBoost': G / 'xgboost_seed42_predictions.csv',
             'DeepVarPred score-only': R / 'germline_extra/channel_ablation/germline_score_only_predictions.csv',
             'DeepVarPred sequence-only': R / 'germline_extra/channel_ablation/germline_sequence_only_predictions.csv',
             'DeepVarPred reduced (24 scores)': R / 'germline_extra/reduced24/germline_reduced24_predictions.csv',
             'LR reduced (24 scores)': R / 'germline_extra/reduced24/germline_reduced24_logistic_regression_predictions.csv',
             'XGBoost reduced (24 scores)': R / 'germline_extra/reduced24/germline_reduced24_xgboost_predictions.csv',
             'DeepVarPred + AlphaMissense + ESM1b': R / 'germline_extra/plus2/germline_plus_AM_ESM1b_predictions.csv',
             'LR + AlphaMissense + ESM1b': R / 'germline_extra/plus2/germline_plus_AM_ESM1b_logistic_regression_predictions.csv',
             'XGBoost + AlphaMissense + ESM1b': R / 'germline_extra/plus2/germline_plus_AM_ESM1b_xgboost_predictions.csv'}
    S = {}
    for k, p in preds.items():
        if p.exists():
            z = pd.read_csv(p).set_index('variant_id').reindex(test.variant_id); assert z.prob_pathogenic.notna().all(), k; S[k] = z.prob_pathogenic.to_numpy()
    y = test.label.to_numpy(int); genes = test.gene.to_numpy(); W = boot_index(genes)
    rows = model_table('germline gene-disjoint test', y, genes, S, W)
    prows = [paired('germline gene-disjoint test', y, S['DeepVarPred'], S[k], W, 'DeepVarPred', k) for k in S if k != 'DeepVarPred']
    # extra predictors (AlphaMissense / ESM1b) native AUROC
    ex = R / 'germline_extra/extra_AM_ESM1b.parquet'
    tt = test.copy()
    if ex.exists(): tt = tt.merge(pd.read_parquet(ex), on='variant_id', how='left')
    nat, thr = native_block('germline gene-disjoint test', tt, y, genes, S['DeepVarPred'], {k: S[k] for k in ['Logistic regression', 'XGBoost']}, W,
                            CANON + ['AlphaMissense_score', 'ESM1b_score'])
    thr.append(dict(cohort='germline gene-disjoint test', method='DeepVarPred', threshold='> 0.5 (class-1 softmax probability; argmax)', threshold_source='this study', n=len(y), **binary(y, S['DeepVarPred'], 0.5 + 1e-12)))
    for k in ['Logistic regression', 'XGBoost']:
        thr.append(dict(cohort='germline gene-disjoint test', method=k, threshold='> 0.5', threshold_source='this study', n=len(y), **binary(y, S[k], 0.5 + 1e-12)))
    # ---------- gene size / interval length strata (R1 Q4, R2 Q1)
    det = {r['gene']: r['reference'] for r in json.loads((P26 / 'retraining/full_input/source_hashes_and_gene_details.json').read_text()) if 'reference' in r}
    full_counts = gi.groupby('gene').size()
    test['gene_n'] = test.gene.map(full_counts); test['gene_len'] = test.gene.map(lambda g: det[g]['sequence_length'])
    strata = []
    for col, bins, labels in [('gene_n', [0, 10, 30, 100, 10 ** 9], ['1-9', '10-29', '30-99', '>=100']),
                              ('gene_len', [0, 2e4, 1e5, 5e5, 1e10], ['<20 kb', '20-100 kb', '100-500 kb', '>500 kb'])]:
        cat = pd.cut(test[col], bins=bins, labels=labels, right=False)
        for lab in labels:
            m = (cat == lab).to_numpy()
            if m.sum() == 0: continue
            Wm = boot_index(genes[m]); r = dict(stratifier='labelled variants per gene' if col == 'gene_n' else 'gene interval length', stratum=lab, n=int(m.sum()), genes=int(len(set(genes[m]))), P=int(y[m].sum()))
            for k in ['DeepVarPred', 'Logistic regression', 'XGBoost']:
                a = roc_auc_score(y[m], S[k][m]); lo, hi = ci([auc_w(y[m], S[k][m], w) for w in Wm]); r[k] = a; r[k + '_lo'] = lo; r[k + '_hi'] = hi
            cad = test.CADD_phred.to_numpy(float)[m]; r['CADD_phred'] = roc_auc_score(y[m], cad)
            # macro AUROC over genes with >=1 per class
            per = [roc_auc_score(y[m][genes[m] == g], S['DeepVarPred'][m][genes[m] == g]) for g in sorted(set(genes[m])) if len(np.unique(y[m][genes[m] == g])) == 2]
            r['DeepVarPred_macro_AUROC'] = float(np.mean(per)) if per else np.nan; r['genes_with_both_classes'] = len(per)
            strata.append(r)
    pd.DataFrame(strata).to_csv(OUT / 'germline_strata.csv', index=False)
    # ---------- correlations (R1 Q7, R2 Q5)
    def corr_table(df, scores, label):
        cols = {'DeepVarPred': scores, 'CADD': df.CADD_raw.to_numpy(float), 'REVEL': df.REVEL_score.to_numpy(float), 'MVP': df.MVP_score.to_numpy(float),
                'MutPred': df.MutPred_score.to_numpy(float), 'PROVEAN': df.PROVEAN_converted_rankscore.to_numpy(float), 'SIFT (1-score)': 1 - df.SIFT_score.to_numpy(float),
                'ClinPred': df.ClinPred_score.to_numpy(float), 'MetaRNN': df.MetaRNN_score.to_numpy(float), 'PolyPhen2 HVAR': df.Polyphen2_HVAR_score.to_numpy(float)}
        if 'AlphaMissense_score' in df: cols['AlphaMissense'] = df.AlphaMissense_score.to_numpy(float)
        names = list(cols); out = []
        for i, a in enumerate(names):
            for b in names:
                ok = np.isfinite(cols[a]) & np.isfinite(cols[b])
                out.append(dict(set=label, A=a, B=b, n=int(ok.sum()), spearman=spearmanr(cols[a][ok], cols[b][ok])[0] if ok.sum() > 10 and a != b else (1.0 if a == b else np.nan),
                                pearson=pearsonr(cols[a][ok], cols[b][ok])[0] if ok.sum() > 10 and a != b else (1.0 if a == b else np.nan)))
        return out
    corr = corr_table(tt, S['DeepVarPred'], 'germline test, all (P/LP + B/LB)')
    b1 = tt.gene.eq('BRCA1').to_numpy(); corr += corr_table(tt[b1], S['DeepVarPred'][b1], 'BRCA1 held-out P/LP + B/LB')
    vus = pd.read_csv(P26 / 'paired_analysis/current_BRCA1_VUS_input/current_BRCA1_strict_VUS_fixed46.csv')
    vp = pd.read_csv(P26 / 'paired_analysis/current_BRCA1_VUS_input/primary_VUS_new_figure/current_primary_BRCA1_VUS_predictions.csv')
    vus = vus.merge(vp[['variant_id', 'prob_pathogenic']], on='variant_id')
    am = R / 'analysis/stats/BRCA1_VUS_AM_ESM1b.parquet'
    if am.exists(): vus = vus.merge(pd.read_parquet(am), on='variant_id', how='left')
    corr += corr_table(vus, vus.prob_pathogenic.to_numpy(), 'BRCA1 VUS')
    pd.DataFrame(corr).to_csv(OUT / 'correlations.csv', index=False)
    vus[[c for c in ['variant_id', 'variant_key', 'pos', 'aaref', 'aaalt', 'prob_pathogenic', 'AlphaMissense_score', 'ESM1b_score', 'CADD_raw', 'REVEL_score', 'MVP_score', 'MutPred_score', 'PROVEAN_converted_rankscore', 'SIFT_score'] if c in vus.columns]].to_csv(OUT / 'BRCA1_VUS_scores.csv', index=False)
    tt[b1][['variant_id', 'variant_key', 'pos', 'label', 'CADD_raw', 'REVEL_score', 'MVP_score', 'MutPred_score', 'PROVEAN_converted_rankscore', 'SIFT_score']].assign(prob_pathogenic=S['DeepVarPred'][b1]).to_csv(OUT / 'BRCA1_labelled_scores.csv', index=False)
    # ---------------- somatic
    si = pd.read_parquet(R / 'somatic/prepared/analysis_input.parquet'); st = si[si.split.eq('test')].reset_index(drop=True)
    sp = {'DeepVarPred': R / 'somatic/results/somatic_cnn_full_predictions.csv', 'Logistic regression': R / 'somatic/results/somatic_cnn_full_logistic_regression_predictions.csv',
          'XGBoost': R / 'somatic/results/somatic_cnn_full_xgboost_predictions.csv', 'DeepVarPred score-only': R / 'somatic/results/somatic_score_only_predictions.csv',
          'DeepVarPred sequence-only': R / 'somatic/results/somatic_sequence_only_predictions.csv', 'Germline DeepVarPred model': R / 'analysis/cross/somatic_test_scored_by_germline_model.csv'}
    SS = {}
    for k, p in sp.items():
        if p.exists(): z = pd.read_csv(p).set_index('variant_id').reindex(st.variant_id); SS[k] = z.prob_pathogenic.to_numpy()
    ys = st.label.to_numpy(int); gs = st.gene.to_numpy(); Ws = boot_index(gs)
    rows += model_table('somatic gene-disjoint test', ys, gs, SS, Ws)
    prows += [paired('somatic gene-disjoint test', ys, SS['DeepVarPred'], SS[k], Ws, 'DeepVarPred', k) for k in SS if k != 'DeepVarPred']
    n2, t2 = native_block('somatic gene-disjoint test', st, ys, gs, SS['DeepVarPred'], {k: SS[k] for k in ['Logistic regression', 'XGBoost']}, Ws, CANON)
    nat += n2; thr += t2
    for k in ['DeepVarPred', 'Logistic regression', 'XGBoost']:
        thr.append(dict(cohort='somatic gene-disjoint test', method=k, threshold='> 0.5', threshold_source='this study', n=len(ys), **binary(ys, SS[k], 0.5 + 1e-12)))
    # sensitivity: alternative positive definitions vs all negatives
    sens = []
    for lab, m in [('all COSMIC-confirmed positives', np.ones(len(st), bool)), ('recurrent positives (>=3 confirmed samples)', (st.label.eq(0) | (st.cosmic_confirmed_samples >= 3)).to_numpy()),
                   ('ClinVar P/LP positives only', (st.label.eq(0) | st.clinvar_class.eq('P/LP')).to_numpy()), ('ClinVar VUS positives only', (st.label.eq(0) | st.clinvar_class.eq('VUS')).to_numpy())]:
        r = dict(subset=lab, n=int(m.sum()), P=int(ys[m].sum()), B=int((ys[m] == 0).sum()))
        for k in ['DeepVarPred', 'Logistic regression', 'XGBoost', 'Germline DeepVarPred model']:
            if k in SS: r[k] = roc_auc_score(ys[m], SS[k][m])
        for f in ['CADD_phred', 'REVEL_score', 'ClinPred_score', 'MetaRNN_score']:
            v = pd.to_numeric(st[f], errors='coerce').to_numpy(float); ok = m & np.isfinite(v); r[f] = roc_auc_score(ys[ok], v[ok])
        sens.append(r)
    pd.DataFrame(sens).to_csv(OUT / 'somatic_positive_definition_sensitivity.csv', index=False)
    # cross-application on germline test in cancer genes
    gc = pd.read_csv(R / 'analysis/cross/germline_test_scored_by_somatic_model.csv')
    gp = pd.read_csv(G / 'cnn_full_seed42_predictions.csv').set_index('variant_id').prob_pathogenic
    gc['germline_model'] = gc.variant_id.map(gp)
    yc = gc.label.to_numpy(int); Wc = boot_index(gc.gene.to_numpy())
    cross = [dict(test_set='somatic test (COSMIC-confirmed vs ClinVar B/LB)', n=len(ys), somatic_model=roc_auc_score(ys, SS['DeepVarPred']), germline_model=roc_auc_score(ys, SS['Germline DeepVarPred model'])),
             dict(test_set='germline test, cancer genes (ClinVar P/LP vs B/LB)', n=len(yc), somatic_model=roc_auc_score(yc, gc.prob_pathogenic), germline_model=roc_auc_score(yc, gc.germline_model))]
    prows.append(paired('germline test in cancer genes', yc, gc.germline_model.to_numpy(), gc.prob_pathogenic.to_numpy(), Wc, 'Germline DeepVarPred model', 'Somatic DeepVarPred model'))
    cc = []
    for cls in ['B/LB', 'P/LP', 'VUS', 'all']:
        m = np.ones(len(st), bool) if cls == 'all' else st.clinvar_class.eq(cls).to_numpy()
        a, b = SS['DeepVarPred'][m], SS['Germline DeepVarPred model'][m]
        cc.append(dict(clinvar_class=cls, n=int(m.sum()), mean_somatic=a.mean(), mean_germline=b.mean(), median_diff=float(np.median(a - b)), pearson=pearsonr(a, b)[0], spearman=spearmanr(a, b)[0],
                       discordant_calls_at_0p5=float(np.mean((a > .5) != (b > .5)))))
    pd.DataFrame(cross).to_csv(OUT / 'cross_application.csv', index=False); pd.DataFrame(cc).to_csv(OUT / 'cross_model_agreement_by_class.csv', index=False)
    st[['variant_id', 'gene', 'label', 'clinvar_class', 'cosmic_confirmed_samples']].assign(somatic_model=SS['DeepVarPred'], germline_model=SS['Germline DeepVarPred model']).to_csv(OUT / 'somatic_test_two_model_scores.csv', index=False)
    # ---------------- write
    pd.DataFrame(rows).to_csv(OUT / 'model_auroc_with_gene_cluster_CI.csv', index=False)
    pd.DataFrame(prows).to_csv(OUT / 'paired_model_differences.csv', index=False)
    pd.DataFrame(nat).to_csv(OUT / 'native_predictor_paired_comparisons.csv', index=False)
    pd.DataFrame(thr).to_csv(OUT / 'threshold_dependent_metrics.csv', index=False)
    # prediction exports for figures
    test[['variant_id', 'gene', 'label']].assign(**{k: S[k] for k in S}).to_csv(OUT / 'germline_test_predictions_all_models.csv', index=False)
    st[['variant_id', 'gene', 'label', 'clinvar_class', 'cosmic_confirmed_samples', 'CADD_phred', 'REVEL_score', 'ClinPred_score', 'MetaRNN_score', 'MVP_score', 'MetaSVM_rankscore']].assign(**{k: SS[k] for k in SS}).to_csv(OUT / 'somatic_test_predictions_all_models.csv', index=False)
    print('done', OUT)


if __name__ == '__main__':
    main()
