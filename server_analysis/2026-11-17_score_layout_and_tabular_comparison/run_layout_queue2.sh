#!/usr/bin/env bash
# Revision R2 (2026-11-17) GPU queue, reordered: continues after the running fixed_row seed-42 job.
# Germline core seeds first, then somatic fixed_row/resized seeds, then the tiled layout.
set -euo pipefail
R=/home/bio1/workdata/GeneMutation2/Revision_2026-11-17
G=/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/retraining/global_gene_holdout
SP=/home/bio1/workdata/GeneMutation2/Revision_2026-10-30/somatic/prepared
export PYTHONPATH=/home/bio1/workdata/GeneMutation2/Revision_2026-10-26/runtime OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/usr/bin/python3; S=$R/scripts/train_score_layout.py
step(){ echo "$(date '+%F %T') $*"; }
run(){ step train $1 $2 seed $3; mkdir -p $R/$1; $PY -u $S --prepared-dir $4 --output-dir $R/$1 --layout $2 --seed $3 > $R/$1/$2_seed$3.log 2>&1; }
while pgrep -f "train_score_layout.py.*fixed_row --seed 42" > /dev/null; do sleep 30; done
run layout resized 43 $G; run layout fixed_row 43 $G; run layout resized 44 $G; run layout fixed_row 44 $G
touch $R/layout_core_complete
step prewarm somatic cache; cat $SP/images.npy > /dev/null
run layout_somatic fixed_row 42 $SP; run layout_somatic resized 43 $SP; run layout_somatic fixed_row 43 $SP; run layout_somatic resized 44 $SP; run layout_somatic fixed_row 44 $SP
touch $R/layout_somatic_complete
run layout fixed_tiled 42 $G; run layout fixed_tiled 43 $G; run layout fixed_tiled 44 $G
step QUEUE COMPLETE; touch $R/layout_queue_complete
