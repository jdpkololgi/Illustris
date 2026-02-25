# Active Workflow Index

This file is the Phase 0 quick reference for what to run in this repository now.
For pscratch organization and migration env vars, see `/global/homes/d/dkololgi/PSCRATCH_LAYOUT.md`.

## Canonical Layout (Migration Target)

- `workflows/abacus_tweb/` for Abacus slab/T-Web annotation pipeline
- `workflows/jraph/` for Jraph training, tuning, and diagnostics
- `workflows/sbi/` for primary SBI FlowJAX path
- `workflows/sbi/experimental/` for alternate two-stage SBI variants
- `shared/` for reusable model/transformation/config modules
- `legacy/` for deprecated scripts kept for reference only
- `scripts/` for temporary compatibility wrappers during migration

## Active

- Abacus slab + MPI T-Web:
  - `workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm` (canonical)
  - `workflows/abacus_tweb/abacus_cactus_tweb.py` (canonical)
  - `workflows/abacus_tweb/annotate_cutsky_with_tweb.py` (canonical)
- Jraph training baseline:
  - `workflows/jraph/jraph_pipeline.py` (canonical)
  - `workflows/jraph/submit_jraph.slurm` (canonical)
- SBI FlowJAX path:
  - `workflows/sbi/jraph_sbi_flowjax.py` (canonical)
  - `workflows/sbi/submit_sbi_flowjax.slurm` (canonical)
- Utility:
  - `workflows/abacus_tweb/abacus_catalog.py` (canonical)

## Experimental

- `workflows/sbi/experimental/jraph_sbi_two_stage.py`
- `workflows/sbi/experimental/jraph_sbi_flowjax_two_stage.py`
- `local-subgraph-pipeline/*`

## Legacy

- `legacy/abacus_process_particles.py` (use `workflows/abacus_tweb/abacus_process_particles2.py`)
- `workflows/sbi/experimental/jraph_sbi_pipeline.py` (legacy SBI path)

## Compatibility

- Root-level script names are temporary wrappers and remain available during migration.
- Wrapper deprecation schedule is tracked in `docs/migration/WORKFLOW_REORG.md`.
