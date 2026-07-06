# FlowJAX SBI Workflows

This directory contains conditional density estimation workflows for cosmic web
targets. The models combine a graph neural network encoder with FlowJAX
normalizing flows to estimate posteriors over T-Web eigenvalue targets from
galaxy graph observables.

## Which Trainer To Use

| Use case | Entrypoint | Notes |
| --- | --- | --- |
| TNG/full-graph cache | `jraph_sbi_flowjax.py` | Loads one SBI cache in memory and trains/evaluates the baseline FlowJAX NPE model. |
| Abacus wedge-subvolume cache | `jraph_sbi_flowjax.py` | Current Abacus-scale path: run one RA/Dec/z wedge graph at a time using a cache built under `workflows/abacus_tweb/`. |
| Posterior plots, full graph/wedge | `plot_flowjax_posteriors.py` | Uses saved model outputs from the full-graph trainer. |
| Abacus partition artifacts | `jraph_sbi_flowjax_partitioned.py` | Legacy partitioned experiment; keep for audit/debugging, not new production runs. |
| Posterior plots, partitioned | `plot_flowjax_posteriors_partitioned.py` | Legacy diagnostics for partitioned checkpoints. |
| Data-parallel benchmark | `benchmark_partition_data_parallel.py` | Measures legacy partition collation/loading behavior. |
| Tensor sharding prototype | `prototype_tensor_sharding_fullgraph.py` | Experimental full-graph sharding prototype. |
| Two-stage prototype | `experimental/jraph_sbi_two_stage.py` | Optional experimental path, not the primary Abacus run. |

## Experimental G4 Gate Scripts

The G4 scripts are research diagnostics for the Abacus wedge SBI program, not
production trainers. They test whether raw geometry, graph construction, dynamic
candidate selection, or equivariance explains the current T-Web eigenvalue
performance. The durable scientific plan is `docs/plan_g4_proper_equivariant_tensor.md`;
the chronological run record is `SCIENCE_LOG.md`.

| Gate / runner | Entrypoint | Purpose |
| --- | --- | --- |
| G4-SMOKE / P1a controls | `gate_g4_egnn_smoke.py` | Non-equivariant geometric message passing. With `--positions-only --build-radius-mpc 14.78`, this is run D: a point-attention, positions-only radius-graph control. With `--gnn-arrays`, it can also consume prebuilt edge sets such as the union graph. |
| P1b steerable tensor test | `gate_g4_p1b_segnn.py` | SEGNN-style e3nn model with invariant-logit attention and an internal `1x0e+1x2e` symmetric-tensor head. Tier-A supervision is still on the existing sorted eigenvalues; tensor targets/eigenvectors remain a gated Tier-B idea. |
| P1a-iii dynamic graph | `gate_g4_p1e_dgcnn_attn.py` | Attentional DGCNN that recomputes kNN in learned feature space. `--curated-features` gives run F; `--knn-radius-cap` constrains learned candidate selection to a physical envelope for capped follow-up ablations. |
| Wave-1 tmux chain | `run_g4_chain.sh` | Login-node tmux orchestrator for B/C/D/E. It is idempotent by result-file existence and fresh-log liveness checks. |
| Wave-2 tmux chain | `run_g4_wave2_chain.sh` | Login-node tmux orchestrator currently prioritising F/G before D seed replicates. It runs at most one new launcher per pass under the interactive QOS constraints. |

Current interpretation, as of the 2026-07-04 science log: production remains the
G3 GraphNet+NPE union-graph path. G4 scripts are for attribution and diagnostics:
union connectivity appears to act as a discrete support for the nonlocal tidal
operator, while uncapped feature-space dynamic graphs hurt the positions-only
control in the first wave. Do not treat the G4 runners as a replacement for
`jraph_sbi_flowjax.py` unless a later science-log entry promotes them.

## Current Abacus Wedge Inputs

Build wedge caches in `workflows/abacus_tweb/`:

```text
subset_abacus_graph_wedge_for_sbi.py
  -> subset_cugraph_metrics_for_wedge.py
  -> build_abacus_sbi_cache.py
  -> jraph_sbi_flowjax.py
```

The full-graph trainer currently resolves its input through
`shared/tng_pipeline_paths.py`. For a wedge run, place or symlink the desired
cache under `TNG_SBI_CACHE_DIR` using the expected transformed-cache name:

```bash
export TNG_SBI_CACHE_DIR="/path/to/wedge_sbi_cache_dir"
python workflows/sbi/jraph_sbi_flowjax.py --epochs 1000 --output_dir "/path/to/out"
```

By default the expected cache file is:

```text
$TNG_SBI_CACHE_DIR/processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl
```

Use `--no_transformed_eig` only for explicit raw-eigenvalue ablations; that mode
expects the `_raw_eig.pkl` cache name instead.

## Target Convention

The current SBI target is the ordered softplus-increment representation from
`shared/eigenvalue_transformations.py`: `lambda1` is the anchor and the
`lambda2 - lambda1` and `lambda3 - lambda2` gaps are encoded as non-negative
increments. The model trains and samples in increment space; conversion back to
physical `(lambda1, lambda2, lambda3)` is for evaluation and plotting. Do not
replace this with an unconstrained direct three-eigenvalue head unless the run is
an intentional ablation.

## Launchers

There is no tracked production `submit_sbi_flowjax.slurm` for the current Abacus
wedge path. The wedge NPE workflow is run directly today inside an appropriate
GPU allocation:

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
```

Tracked SLURM scripts in this directory are mostly legacy partition diagnostics
or experiments:

| SLURM script | Purpose |
| --- | --- |
| `submit_sbi_partitioned_data_parallel.slurm` | Legacy single-node, four-GPU partitioned SBI training. |
| `submit_sbi_partitioned_data_parallel_multinode.slurm` | Legacy multi-node partitioned SBI training with JAX distributed initialization. |
| `submit_sbi_overfit_tiny.slurm` | Tiny overfit diagnostic for partitioned SBI. |
| `submit_partition_data_parallel_benchmark.slurm` | Partition loading and data-parallel benchmark. |
| `submit_plot_flowjax_posteriors_partitioned.slurm` | Posterior plotting for partitioned checkpoints. |
| `submit_sbi_stageB_4node.slurm` | Stage-B multi-node experiment. |
| `submit_tensor_sharding_prototype.slurm` | Experimental tensor-sharding prototype. |

## Legacy Partition Notes

The partitioned trainer requires:

- `--partition-manifest`: path to `partition_manifest.json` generated by
  `workflows/abacus_tweb/build_abacus_partition_batches.py`.
- `--sbi-cache-path`: original SBI cache pickle generated by
  `workflows/abacus_tweb/build_abacus_sbi_cache.py`; this carries scaler and
  raw-eigenvalue metadata.
- `--output-dir`: model checkpoints, logs, and metrics output directory.

Partition semantics are documented in
`workflows/abacus_tweb/PARTITION_ARTIFACT_SCHEMA.md`. `ABACUS_SBI_DEBUG_STRATEGY.md`
records the row-order checks and overfit diagnostics that motivated moving away
from this path for current Abacus SBI work.
