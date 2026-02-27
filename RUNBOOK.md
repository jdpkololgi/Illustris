# TNG/Illustris Runbook

This runbook lists validated workflow entrypoints and launch commands.

## Environment setup

```bash
source ~/.bashrc
conda activate cosmic_env
desienv
```

## Canonical workflows

### Abacus slab + MPI T-Web (CPU)

Batch launch:

```bash
sbatch workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm
```

Direct entrypoints:

```bash
python workflows/abacus_tweb/abacus_cactus_tweb.py
python workflows/abacus_tweb/annotate_cutsky_with_tweb.py
python workflows/abacus_tweb/abacus_process_particles2.py --show-workflow
```

### Abacus mock graph construction (Gudhi; CPU)

This step builds graph artifacts from Abacus mock galaxies using observed
`RA/DEC/Z` (default `Z`, not `Z_COSMO`) and writes edge/tetrahedra arrays for
downstream feature extraction.

`build_abacus_graph.py` now enforces a CPU SLURM allocation at runtime; it will
fail fast on login/GPU allocations.

Example CPU interactive allocation:

```bash
salloc --constraint=cpu --nodes=1 --ntasks=1 --time=02:00:00
```

Inspect CLI options:

```bash
python workflows/abacus_tweb/build_abacus_graph.py --help
```

Build full Delaunay-equivalent graph artifacts from mock catalog:

```bash
python workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits" \
  --mode delaunay \
  --output-dir "/pscratch/sd/d/dkololgi/abacus" \
  --output-prefix abacus_delaunay
```

Build alpha-pruned graph artifacts (explicit alpha_sq):

```bash
python workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits" \
  --mode alpha \
  --alpha-sq 50.0 \
  --output-dir "/pscratch/sd/d/dkololgi/abacus" \
  --output-prefix abacus_alpha
```

Build alpha-pruned graph artifacts with automatic Illustris-style alpha estimate
(`alpha = 1.5 * n^(-1/3)`, then `alpha_sq = alpha^2`):

```bash
python workflows/abacus_tweb/build_abacus_graph.py \
  --catalog-path "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits" \
  --mode alpha \
  --boxsize-mpc 2000.0 \
  --output-dir "/pscratch/sd/d/dkololgi/abacus" \
  --output-prefix abacus_alpha
```

The builder writes:

- `<prefix>_edges_combined_idx.npy`
- `<prefix>_tetrahedra_idx.npy`
- `<prefix>_tetrahedra_volumes.npy`

Memory guidance (all-points, ~23M galaxies):

- Full-run Gudhi alpha/Delaunay construction is very memory heavy.
- Plan for at least ~512 GB RAM; ~0.8-1.2 TB is safer for all-points alpha runs.
- Login-node execution is expected to be killed by OOM.

### Jraph training baseline (GPU)

Batch launch:

```bash
sbatch workflows/jraph/submit_jraph.slurm
```

Other launchers:

```bash
sbatch workflows/jraph/debug_jraph.slurm
sbatch workflows/jraph/submit_tuning.slurm
sbatch workflows/jraph/train_ensemble.slurm
```

Direct:

```bash
python workflows/jraph/jraph_pipeline.py --help
```

### SBI FlowJAX canonical path (GPU)

Batch launch:

```bash
sbatch workflows/sbi/submit_sbi_flowjax.slurm
```

Direct:

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
```

### Optional experimental SBI two-stage path (GPU)

Batch launch:

```bash
sbatch workflows/sbi/experimental/run_sbi_two_stage.slurm
```

Direct:

```bash
python workflows/sbi/experimental/jraph_sbi_two_stage.py --help
```

### GCN paper workflow (GPU)

Batch launch:

```bash
sbatch workflows/gcn_paper/submit_gcn.slurm
```

Direct:

```bash
python workflows/gcn_paper/gcn_pipeline.py --help
python workflows/gcn_paper/gcn_pipeline_postprocess.py --help
python workflows/gcn_paper/postprocessing.py --help
```

## Compatibility wrappers (temporary)

Root-level script names still forward to canonical or legacy modules during migration.
Prefer canonical `workflows/...` paths for all new scripts and docs.

Examples:

```bash
python jraph_pipeline.py --help
python jraph_sbi_flowjax.py --help
python jraph_sbi_pipeline.py --help
```

## Notes

- Active workflows and statuses: `ACTIVE_WORKFLOWS.md`
- Migration docs: `docs/migration/`
- Legacy scripts for reference: `legacy/`
