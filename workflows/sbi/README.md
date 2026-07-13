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
| Field-level/F-tier gates | `gate_t4_graph_field_poisson.py`, `gate_ftier_v2.py`, `gate_f3_generative_ftier.py`, `flow_ftier_head.py` | Research diagnostics for graph-to-field-to-Poisson point estimates and posterior calibration; not the current production VAC trainer. |

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

## Field-Level And F-Tier Diagnostics

The field-level scripts test the graph -> density grid -> fixed FFT tidal
operator -> eigenvalue route described in `docs/plan_field_level_multimodal.md`.
They are useful for reproducing current F-tier and calibration gates, but the
current shippable calibrated VAC headline remains G3 + FMPE + validation-set
tempering for `P(lambda1 > lambda_th)`.

| Script | Purpose | Runtime constraints |
| --- | --- | --- |
| `gate_t2_cnn_counts.py` | CNN-on-voxel-counts point-estimate control. | PyTorch GPU for real runs. |
| `gate_t4_graph_field_poisson.py` | Original F1 graph -> field -> Poisson gate using CIC scatter, a 3-D U-Net, and the fixed FFT physics layer. | Requires CUDA; `--smoke` is a short GPU-bound shape/backward check, not a CPU fallback. |
| `gate_ftier_v2.py` | Upgraded F-tier point-estimate gate: union graph, attention aggregation, edge attributes, TSC scatter, optional survey mask, and U-Net/FNO decoder. | Requires CUDA unless `--smoke`; `--save-cond` writes conditioning arrays for `flow_ftier_head.py`. |
| `gate_f3_generative_ftier.py` | Stochastic F-tier posterior experiment trained with the energy score; supports FiLM/concat/diffusion latent modes and `--log-density` ablations. | Requires CUDA unless `--smoke`; use for research diagnostics, not production calibration. |
| `flow_ftier_head.py` | CPU-only MLE posterior head on saved F-tier physics point estimates; compares NPSE, FMPE, and MAF and reports R2, SBC, coverage, and trace diagnostics. | Reads `gate_ftier_v2.py --save-cond` output; FMPE/NPSE sampling can be slow, so `--n-eval` defaults to a test subsample. |

Minimal v2-to-flow reproduction pattern:

```bash
python workflows/sbi/gate_ftier_v2.py \
  --cache "/path/to/processed_jraph_data_mc1e+09_v2_scaled_3_transformed_eig.pkl" \
  --points-xyz "/path/to/path1_wedge_points_xyz.npy" \
  --gnn-arrays "/path/to/union_graph_gnn_arrays.npz" \
  --scatter tsc \
  --decoder unet \
  --out-file "/path/to/ftier_v2.txt" \
  --save-cond "/path/to/ftier_cond.npz"

python workflows/sbi/flow_ftier_head.py \
  --cond-npz "/path/to/ftier_cond.npz" \
  --out-file "/path/to/flow_result.txt" \
  --samples-npz "/path/to/flow_samples.npz"
```

When running these scripts on a GPU process that unpickles JAX/Jraph cache
objects, set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

Without those variables, importing or unpickling JAX-backed cache objects can
reserve a large fraction of GPU memory before PyTorch training starts.

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
