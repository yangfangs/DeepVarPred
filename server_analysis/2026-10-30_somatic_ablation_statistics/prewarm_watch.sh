#!/usr/bin/env bash
R=/home/bio1/workdata/GeneMutation2/Revision_2026-10-30
until grep -q "somatic score_only" $R/queue.log; do sleep 20; done
cat $R/germline_extra/reduced24_prepared/images.npy > /dev/null
cat $R/germline_extra/plus2_prepared/images.npy > /dev/null
echo prewarm done $(date) > $R/prewarm.done
