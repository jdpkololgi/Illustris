# Abacus SBI Debugging Strategy

## Current Alignment Audit (Completed)


| Check                                               | Scope                       | Result                                                 | Interpretation                                             |
| --------------------------------------------------- | --------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| Partition `x` vs canonical GNN `x[global_node_ids]` | 18 sampled partitions       | Max abs diff `0.000e+00`                               | Feature row-order alignment is correct.                    |
| Duplicate-node consistency for `x`                  | 720k sampled overlap checks | `0` conflicts                                          | Same galaxy ID has consistent features across partitions.  |
| Duplicate-node consistency for `targets`            | 720k sampled overlap checks | `0` conflicts                                          | Same galaxy ID has consistent targets across partitions.   |
| Core-node split integrity                           | All partitions              | `0` within-split duplicates, `0` cross-split conflicts | Core-node train/val/test assignment is clean and disjoint. |
| Core coverage                                       | All partitions              | Unique assigned core nodes = `22,981,777`              | Core partitioning covers all nodes exactly once.           |


## Priority Experiment Matrix


| Priority | Experiment                                           | Hypothesis Tested                           | How to Run (high level)                           | Pass Criterion                                       | Fail Signal                                                   |
| -------- | ---------------------------------------------------- | ------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| P0       | Tiny overfit test                                    | Pipeline can represent labels at all        | Train on 1-2 partitions only for many epochs      | Near-zero train NLL and strong pred-vs-true diagonal | Persistent mean-collapse implies implementation/objective bug |
| P0       | Partition vs non-partition A/B on same tiny subgraph | Partitioning path is not degrading learning | Build a tiny induced subgraph; run both trainers  | Comparable fit quality                               | Non-partition succeeds but partition fails                    |
| P1       | Target transform/scaling ablation                    | Transform/scaling may induce compression    | Compare transformed targets vs raw-scaled targets | One mode materially improves slope/tail fit          | Both modes collapse similarly                                 |
| P1       | Hardness-stratified diagnostics                      | Model fails mainly in dense/rare regimes    | Evaluate by degree/density quantiles              | Similar calibration/performance across bins          | Strong degradation in high-degree/high-density bins           |
| P1       | Label-shuffle negative control                       | Current signal exceeds random baseline      | Shuffle targets and retrain short run             | Shuffled model much worse than real-label model      | Similar metrics indicate no meaningful learning               |
| P2       | Capacity/schedule sweep (minimal grid)               | Underfitting due to model/optimization      | Increase latent/flow width + lower-LR tail        | Better pred-vs-true slope + tails                    | No improvement with added capacity/training                   |
| P2       | Smaller-volume Abacus (redshift cut)                 | Large heterogeneity hurts optimization      | Rebuild subset and rerun same pipeline            | Better learning on subset then degrades with scale   | Same collapse even on easier subset                           |
| P3       | Simple baseline (MLP/XGBoost on node features)       | Graph path may be bottleneck                | Train tabular baseline on same splits             | Baseline competitive or better => graph setup issue  | Baseline also poor => labels/features may be weak/noisy       |


## Recommended 1-Day Execution Order


| Step | Action                                          | Why first                                                        |
| ---- | ----------------------------------------------- | ---------------------------------------------------------------- |
| 1    | Tiny overfit test (P0)                          | Fastest hard check for fundamental training correctness.         |
| 2    | Partition vs non-partition A/B (P0)             | Isolates partition pipeline risk immediately.                    |
| 3    | Transform/scaling ablation + hardness bins (P1) | Identifies compression source and regime-specific failures.      |
| 4    | Short capacity/schedule sweep (P2)              | Tests whether this is mostly underfitting/optimization.          |
| 5    | Reduced-volume Abacus run (P2)                  | Confirms scale/heterogeneity effects before expensive full runs. |


## Step 1 Launch (Prepared)


| Item           | Value                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------- |
| Slurm script   | `workflows/sbi/submit_sbi_overfit_tiny.slurm`                                               |
| Default setup  | 1 GPU, 2 train partitions, 2 val partitions, 300 epochs, dropout=0, weight_decay=0          |
| Submit command | `sbatch /global/homes/d/dkololgi/TNG/Illustris/workflows/sbi/submit_sbi_overfit_tiny.slurm` |
| Pass signal    | Train NLL keeps dropping and tiny-subset pred-vs-true is strongly diagonal.                 |
| Fail signal    | Early plateau plus near-constant posterior means around 0 on tiny subset.                   |


## Step 1 Results (Executed)


| Run                              | Config                                                                     | Key result                                                                            | Status                                                                  |
| -------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Tiny overfit (2 partitions)      | `jraph_sbi_flowjax_partitioned.py`, train limit=2, val limit=2, 300 epochs | `train_nll` dropped from ~61.54 to ~2.93 and plateaued; no memorization-level fit     | **Fail**                                                                |
| Strict overfit (1 partition)     | same trainer, train limit=1, val limit=1, 500 epochs                       | `train_nll` fluctuated ~3.1-3.6 late; best `val_nll` ~3.03                            | **Fail**                                                                |
| Micro overfit (4096-node subset) | same trainer, 1000 epochs                                                  | better (`train_nll` min ~1.92, best `val_nll` ~2.02) but still not clear memorization | **Borderline fail**                                                     |
| Deterministic regression micro   | `jraph_regression_partitioned.py`, same micro manifest                     | `train_mse` ~0.834 (with passes) and ~0.936 (`num_passes=0`)                          | **Underfitting**                                                        |
| Tabular micro baselines          | `debug_micro_mlp_baseline.py`                                              | Mean: 0.951 MSE, Linear: 0.935 MSE, MLP: 0.797 MSE                                    | Indicates some signal, but current graph/SBI path not extracting enough |


## Current Diagnosis

Current evidence does **not** support a simple row-index alignment bug.  
The more likely issue is **representation/objective mismatch**:

- **Representation issue:** current node/edge features and message-passing setup are not encoding enough information for the target map (especially tails/rare regimes).
- **Objective issue:** the training objective (flow NLL, and even current regression setup) tends to favor conservative central predictions and does not strongly force tail fidelity on this data regime.

## Immediate Fix Plan (Ranked)


| Priority | Action                                                                                                                                          | Why                                                                      | Success criterion                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| P0       | Run deterministic regression with richer capacity sweep (`num_passes` in {1,2,4}, `latent_size` in {64,128,256}, `dropout=0`, `weight_decay=0`) | Identify whether model capacity/message passing can overfit micro subset | At least one config drives micro train MSE well below tabular MLP baseline (~0.80) |
| P0       | Add explicit target weighting for tails (e.g., weight by per-dim z-score magnitude) in regression objective                                     | Penalize collapse-to-mean behavior                                       | Improved tail-bin error and stronger pred-vs-true slope at extremes                |
| P1       | Train on raw eigenvalues vs transformed targets (same micro setup)                                                                              | Check if transform is harming optimization geometry                      | One target mode clearly improves micro overfit and tail fidelity                   |
| P1       | Add/restore physically informative node features (true xyz and/or local geometric summaries) into the learning path                             | Current 7 feature set may be insufficient for full target variation      | Significant MSE drop over existing feature set under same model/hparams            |
| P2       | Reduce partition context complexity (fewer halo hops for debug) and compare                                                                     | Isolate oversmoothing/noise from large context                           | Better fit with reduced context indicates message-passing dilution                 |
| P2       | Then rerun SBI flow only after regression passes micro-overfit gate                                                                             | Avoid tuning flow objective on a weak encoder                            | SBI no longer collapses to central posterior on micro set                          |


