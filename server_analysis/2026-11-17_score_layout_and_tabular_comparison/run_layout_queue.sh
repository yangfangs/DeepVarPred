#!/usr/bin/env bash
# Revision R2 (2026-11-17) GPU queue: score-layout sensitivity (germline gene-disjoint split).
# Outputs only under Revision_2026-11-17; primary model files are read-only.
set -euo pipefail
R=/home/bio1/workdata/GeneMutation2/Revision_2026-11-17
G=/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/retraining/global_gene_holdout
export PYTHONPATH=/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/runtime OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/usr/bin/python3; S=$R/scripts/train_score_layout.py; O=$R/layout
step(){ echo "$(date '+%F %T') $*"; }

step prewarm image cache
cat $G/images_base.npy > /dev/null
for run in "fixed_row 42" "resized 43" "fixed_row 43" "resized 44" "fixed_row 44" "fixed_tiled 42" "fixed_tiled 43" "fixed_tiled 44"; do
  set -- $run
  step train $1 seed $2
  $PY -u $S --prepared-dir $G --output-dir $O --layout $1 --seed $2 > $O/$1_seed$2.log 2>&1
done
step QUEUE COMPLETE
touch $R/layout_queue_complete
