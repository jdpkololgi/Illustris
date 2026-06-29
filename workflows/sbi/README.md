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
| Posterior plots, full graph/wedge | `plot_flowjax_posteriors.py` | Uses saved model outputs from the full-graph trainer; also computes posterior T-Web class probabilities. |
| Abacus partition artifacts | `jraph_sbi_flowjax_partitioned.py` | Legacy partitioned experiment; keep for audit/debugging, not new production runs. |
| Posterior plots, partitioned | `plot_flowjax_posteriors_partitioned.py` | Legacy diagnostics for partitioned checkpoints. |
| Data-parallel benchmark | `benchmark_partition_data_parallel.py` | Measures legacy partition collation/loading behavior. |
| Tensor sharding prototype | `prototype_tensor_sharding_fullgraph.py` | Experimental full-graph sharding prototype. |
| Two-stage prototype | `experimental/jraph_sbi_two_stage.py` | Optional experimental path, not the primary Abacus run. |

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
cache under `TNG_SBI_CACHE_DIR` using the filename suffix that matches the
target mode:

```bash
export TNG_SBI_CACHE_DIR="/path/to/wedge_sbi_cache_dir"
python workflows/sbi/jraph_sbi_flowjax.py \
  --increment_mode linear \
  --epochs 1000 \
  --checkpoint_every 250 \
  --output_dir "/path/to/out"
```

The trainer selects a cache suffix from the target parameterisation:

| `jraph_sbi_flowjax.py` mode | Cache suffix | Typical cache-builder flag |
| --- | --- | --- |
| default / `--increment_mode softplus` | `_transformed_eig.pkl` | default transformed increments |
| `--increment_mode linear` | `_linear_eig.pkl` | `--linear-increments --three-targets-only` |
| `--increment_mode raw` or legacy `--no_transformed_eig` | `_raw_eig.pkl` | `--no-transformed-eig` |

For example, a linear-increment wedge run expects:

```text
$TNG_SBI_CACHE_DIR/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl
```

## Target Convention

The trainer and plotting utilities use the target mode recorded in the model
metadata, or the explicit `--increment_mode` argument during training:

- `softplus`: ordered softplus increments from
  `shared/eigenvalue_transformations.py`. This is the default and enforces
  `lambda1 <= lambda2 <= lambda3` when samples are converted back to physical
  eigenvalues.
- `linear`: plain increments
  `(v1=lambda1, v2=lambda2-lambda1, v3=lambda3-lambda2)`. This avoids the
  inverse-softplus tail and is the current explicit Abacus wedge NPE mode used
  for recent DESI-transfer studies, but it does not guarantee positive sampled
  gaps.
- `raw`: direct scaled eigenvalues. Keep this for controlled ablations.

All non-raw modes train and sample in increment space. Convert samples back to
physical `(lambda1, lambda2, lambda3)` only for evaluation, plotting, and T-Web
class probabilities.

## Checkpointing And Diagnostics

`jraph_sbi_flowjax.py` writes resumable checkpoints atomically to
`flowjax_sbi_checkpoint_seed_<seed>.pkl` in the output directory every
`--checkpoint_every` epochs. Use `--resume` to pick up that seed-specific file
from `--output_dir`, or `--resume_from /path/to/checkpoint.pkl` to name the
checkpoint explicitly. When resuming, the script restores GNN parameters, flow
arrays, optimizer state, RNG state, best-validation state, and logs.

Test-set posterior diagnostics draw `--test_posterior_samples` samples per test
node in chunks of `--test_eval_chunk_size`, then report both single-sample and
posterior-mean metrics in raw eigenvalue space. `plot_flowjax_posteriors.py`
reuses the saved `increment_mode` metadata, produces raw/transformed posterior
plots, SBC/TARP-style calibration plots, and posterior T-Web class probabilities
using `--lambda_th` (default `0.2`, matching the CACTUS/Abacus `CWEB` labels).

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
