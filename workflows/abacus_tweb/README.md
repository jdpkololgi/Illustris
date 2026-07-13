# Abacus T-Web And Graph Workflow

This directory contains the Abacus side of the cosmic web pipeline: build or
load T-Web eigenvalue fields, attach those labels to DESI/Abacus CutSky mock
galaxies, construct graph topology, compute graph features, and export
SBI-ready caches or staged-mock products.

## Canonical Pipeline

```text
Abacus particle/halo products
  -> slabwise T-Web grids
  -> host-halo linked CutSky FITS with CWEB/LAMBDA targets
  -> alpha or Delaunay graph artifacts
  -> graph node/edge features
  -> RA/Dec/z wedge subgraph and aligned wedge targets
  -> wedge graph-feature metadata
  -> SBI cache pickle for NPE training
```

## Entry Points

| Stage | Scripts |
| --- | --- |
| T-Web field generation | `abacus_cactus_tweb.py`, `abacus_cactus_tweb_fullgrid_mpi.py`, `submit_abacus_tweb_cpu.slurm` |
| CutSky annotation | `annotate_cutsky_with_tweb_eigs.py` |
| Graph construction | `build_abacus_graph.py`, `submit_abacus_graph_cpu.slurm` |
| Graph features | `abacus_graph_features.py`, `abacus_graph_features_cugraph.py`, `submit_abacus_graph_features_cpu.slurm`, `submit_abacus_graph_features_cugraph.slurm` |
| Selection-function diagnostics | `s0_selection_atlas.py` |
| Wedge subgraphs for SBI | `subset_abacus_graph_wedge_for_sbi.py`, `subset_cugraph_metrics_for_wedge.py` |
| SBI cache | `build_abacus_sbi_cache.py`, `build_staged_mock_wedge_sbi_cache.py` |
| Legacy partition batches | `build_abacus_partition_batches.py`, `submit_build_partitions_adaptive.slurm`, `PARTITION_ARTIFACT_SCHEMA.md` |
| Validation / audits | `validate_unique_halo_eigs_fits_vs_slabs.py`, `validate_cutsky_eigs_boxindex_vs_halo_xcom.py`, `diagnose_cutsky_tweb_alignment.py`, `audit_abacus_leakage_alignment.py`, `ABACUS_TWEB_AUDIT_FINDINGS.md` |
| Staged mocks / fiberassign | `build_staged_mock_wedge_variants.py`, `build_staged_mock_wedge_truth_npz.py`, `build_staged_mock_wedge_sbi_cache.py`, `write_fiberassign_mock_science_fits.py`, `write_stage3_postcollision_science_fits.py`, `join_cutsky_eigs_to_fiberassign_catalog.py` |

## Scientific And Alignment Constraints

- `annotate_cutsky_with_tweb_eigs.py` assigns labels through host-halo linkage:
  `(FILE_NUM, HALO_INDEX)` -> Abacus halo-info box-frame position -> T-Web voxel.
  This is preferred over assigning labels by sky-coordinate inversion or modulo
  wrapping.
- `build_abacus_graph.py` catalog mode applies DESI BGS mock selection
  `(IN_Y1 | IN_Y5)` and `R_MAG_APP < 19.5`, excludes `BOX_INDEX == -1` by
  default, and uses observed redshift `Z` to preserve RSD-like effects.
- Graph construction splits north/south Galactic hemispheres before building
  Gudhi alpha complexes. This avoids long artificial edges across the Galactic
  plane in CutSky geometry.
- `build_abacus_sbi_cache.py` repeats the Y1/Y5 and invalid `BOX_INDEX` filters
  by default so graph rows and target rows stay aligned.

## Graph Artifacts

`build_abacus_graph.py` writes a metadata JSON plus arrays such as:

- `<prefix>_points.npy`
- `<prefix>_points_xyz.npy`
- `<prefix>_edges_combined_idx.npy`
- `<prefix>_tetrahedra_idx.npy`
- `<prefix>_tetrahedra_volumes.npy`

Feature builders consume those artifacts and write either CPU feature tables or
cuGraph GNN arrays. The cuGraph path writes metadata consumed by
`build_abacus_sbi_cache.py`.

## S-Track Selection Atlas

`s0_selection_atlas.py` is the S0 diagnostic for the full BGS redshift range.
It compares DESI DR2 and the sentinelfix mock inside the same RA/Dec wedge,
fits smooth per-dataset `n~(z)` conditioning curves, and records broad-shell
structural stress tests for both graph and grid encoders:

- mock/DESI count ratio versus redshift;
- mean spacing and nearest-neighbour distance;
- radius-10 Mpc/h union-graph degree statistics;
- 5 and 6 Mpc voxel occupancy for CNN/F-tier grid inputs.

Typical CPU run:

```bash
python workflows/abacus_tweb/s0_selection_atlas.py \
  --ra 120 160 \
  --dec 14.5 30.6 \
  --zmin 0.03 \
  --zmax 0.62 \
  --out-dir "/pscratch/sd/d/dkololgi/abacus/s0_selection_atlas"
```

The script writes `s0_atlas.json` plus `s0_selection_atlas.png`. The spline
arrays in the JSON are the S3 `n~(z)` conditioning inputs; do not substitute raw
redshift for this covariate unless running an explicit ablation.

## Wedge SBI Cache

`build_abacus_sbi_cache.py` produces a pickle with a `jraph.GraphsTuple`,
regression targets, split masks, scalers, raw eigenvalues, and optional CWEB
classification labels. Current Abacus-scale SBI uses one cache per survey-space
wedge rather than graph partitions.

The wedge path is:

```text
full graph artifacts + annotated CutSky FITS
  -> subset_abacus_graph_wedge_for_sbi.py
  -> subset_cugraph_metrics_for_wedge.py
  -> build_abacus_sbi_cache.py
  -> workflows/sbi/jraph_sbi_flowjax.py
```

Wedge constraints:

- Wedge selection is in survey coordinates (`RA`, `DEC`, observed `Z`) after
  reproducing the graph-build row mask.
- `<prefix>_global_node_ids.npy` maps wedge-local nodes back to the full graph,
  so cuGraph node features are copied as `x_full[global_node_ids]`.
- Wedge edges are induced from the parent graph. Missing parent edge features
  fail by default; `--recompute-missing-edge-lengths` is a diagnostic fallback,
  not a production mode.
- When the target file is the compact `<prefix>_wedge_targets.fits`, call
  `build_abacus_sbi_cache.py` with
  `--no-apply-y1y5-filter --no-exclude-invalid-box-index`; the rows have already
  been filtered and ordered by the wedge builder.
- `build_abacus_sbi_cache.py` trains on ordered softplus eigenvalue increments by
  default. Use `--no-transformed-eig` only for explicit raw-eigenvalue ablations.

`build_abacus_partition_batches.py` and `PARTITION_ARTIFACT_SCHEMA.md` document
the older partitioned FlowJAX experiment. Keep them for audit/debugging, but do
not start new Abacus SBI runs from partition artifacts.

## Operational Notes

- Graph construction and partition building are CPU-heavy and should run inside
  SLURM allocations.
- cuGraph feature extraction uses a RAPIDS environment, not necessarily
  `cosmic_env`.
- Large all-points alpha/Delaunay construction can require hundreds of GB of
  RAM; avoid login nodes.
- For launch commands, environment overrides, and common pitfalls, see
  `RUNBOOK.md`.
