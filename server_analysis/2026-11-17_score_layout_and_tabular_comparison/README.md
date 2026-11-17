# Second revision (2026-11-17): score-channel layout and comparison with tabular classifiers

Scripts for the analyses added in the second revision. They read the prepared cohorts and model outputs
produced by `../2026-10-26_cohort_and_primary_model` and `../2026-10-30_somatic_ablation_statistics`.

| File | Purpose |
|---|---|
| `train_score_layout.py` | Retrains the CNN with the score channel in the original layout (`resized`), in the first 46 pixels of the first row without resizing (`fixed_row`), or as 8 × 8 pixel cells in a 48 × 64 pixel block (`fixed_tiled`). Sequence channels, partitions, preprocessing and training protocol are unchanged. |
| `run_layout_queue.sh`, `run_layout_queue2.sh` | GPU job queues used for the runs (germline seeds 42–44 for each layout, somatic `resized`/`fixed_row`). |
| `r2_stats.py` | Statistics with the gene-cluster bootstrap of `revision_stats.py`: `q1` CNN versus logistic regression and XGBoost (combinations, strata, per-gene AUROC); `q2`/`q2s` layout sensitivity (germline/somatic); `q3` layouts versus each individual predictor on identical variants. |

Example:

```bash
python train_score_layout.py --prepared-dir <prepared cohort> --output-dir layout --layout fixed_tiled --seed 42
python r2_stats.py q2
```
