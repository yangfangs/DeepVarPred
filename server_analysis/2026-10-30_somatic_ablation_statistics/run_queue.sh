#!/usr/bin/env bash
# Revision 2026-10-30 GPU/CPU queue. Outputs only under Revision_2026-10-30.
set -euo pipefail
R=/home/bio1/workdata/GeneMutation2/Revision_2026-10-30
G=/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/retraining/global_gene_holdout
FI=/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/retraining/full_input
export PYTHONPATH=/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/runtime OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/usr/bin/python3; S=$R/scripts/train_revision_variant.py
REDUCED=SIFT_score,SIFT4G_score,Polyphen2_HDIV_score,Polyphen2_HVAR_score,MutationTaster_score,MutationAssessor_score,PROVEAN_converted_rankscore,VEST4_score,MetaSVM_rankscore,MetaLR_score,MetaRNN_score,M-CAP_score,REVEL_score,MutPred_score,MVP_score,MPC_score,PrimateAI_score,DEOGEN2_score,BayesDel_addAF_score,BayesDel_noAF_rankscore,ClinPred_score,LIST-S2_score,CADD_phred,DANN_score
step(){ echo "$(date '+%F %T') $*"; }

# CPU side-queue: germline reduced-panel and +AlphaMissense/ESM1b caches (same variants and gene split as primary)
(
  step cpu: reduced24 prepare
  $PY -u $S --prepare --cohort $G/analysis_input.parquet --features $REDUCED --refmeta $FI/source_hashes_and_gene_details.json --output-dir $R/germline_extra/reduced24_prepared > $R/germline_extra/reduced24_prepare.log 2>&1
  step cpu: extract AlphaMissense/ESM1b
  $PY -u $R/scripts/extract_extra_predictors.py --cohort $G/analysis_input.parquet --source /home/bio1/workdata/GeneMutation2/train_all --output $R/germline_extra/extra_AM_ESM1b.parquet > $R/germline_extra/extract_extra.log 2>&1
  step cpu: plus2 prepare
  $PY -u $S --prepare --cohort $G/analysis_input.parquet --extra-features $R/germline_extra/extra_AM_ESM1b.parquet --features $R/germline_extra/features48.json --refmeta $FI/source_hashes_and_gene_details.json --output-dir $R/germline_extra/plus2_prepared > $R/germline_extra/plus2_prepare.log 2>&1
  step cpu: done
) > $R/cpu_queue.log 2>&1 &
CPU=$!

step somatic prepare
$PY -u $S --prepare --cohort $R/somatic/input/somatic_cohort.parquet --features $FI/full_cohort_manifest.json --split-manifest $G/gene_split_manifest.csv --refmeta $R/somatic/input/source_hashes_and_gene_details.json --output-dir $R/somatic/prepared > $R/somatic/prepare.log 2>&1
step somatic train full
$PY -u $S --train --prepared-dir $R/somatic/prepared --output-dir $R/somatic/results --tag somatic_cnn_full --channels full --tabular > $R/somatic/train_full.log 2>&1
step germline score_only
$PY -u $S --train --prepared-dir $G --output-dir $R/germline_extra/channel_ablation --tag germline_score_only --channels score_only > $R/germline_extra/score_only.log 2>&1
step germline sequence_only
$PY -u $S --train --prepared-dir $G --output-dir $R/germline_extra/channel_ablation --tag germline_sequence_only --channels sequence_only > $R/germline_extra/sequence_only.log 2>&1
step somatic score_only / sequence_only
$PY -u $S --train --prepared-dir $R/somatic/prepared --output-dir $R/somatic/results --tag somatic_score_only --channels score_only > $R/somatic/train_score_only.log 2>&1
$PY -u $S --train --prepared-dir $R/somatic/prepared --output-dir $R/somatic/results --tag somatic_sequence_only --channels sequence_only > $R/somatic/train_sequence_only.log 2>&1
wait $CPU
step germline reduced24 train
$PY -u $S --train --prepared-dir $R/germline_extra/reduced24_prepared --output-dir $R/germline_extra/reduced24 --tag germline_reduced24 --channels full --tabular > $R/germline_extra/reduced24_train.log 2>&1
step germline plus2 train
$PY -u $S --train --prepared-dir $R/germline_extra/plus2_prepared --output-dir $R/germline_extra/plus2 --tag germline_plus_AM_ESM1b --channels full --tabular > $R/germline_extra/plus2_train.log 2>&1
step QUEUE COMPLETE
touch $R/queue_complete
