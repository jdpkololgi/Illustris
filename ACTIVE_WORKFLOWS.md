# Active Workflow Index

This file is the Phase 0 quick reference for what to run in this repository now.
For pscratch organization and migration env vars, see `/global/homes/d/dkololgi/PSCRATCH_LAYOUT.md`.

## Canonical Layout (Migration Target)

- `workflows/abacus_tweb/` for Abacus slab/T-Web annotation pipeline
- `workflows/jraph/` for Jraph training, tuning, and diagnostics
- `workflows/sbi/` for primary SBI FlowJAX path
- `workflows/sbi/experimental/` for the single optional two-stage SBI variant
- `workflows/gcn_paper/` for the paper-critical legacy GCN workflow
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
- GCN paper workflow (active for paper reproduction):
  - `workflows/gcn_paper/gcn_pipeline.py` (canonical)
  - `workflows/gcn_paper/gcn_pipeline_postprocess.py` (canonical)
  - `workflows/gcn_paper/postprocessing.py` (canonical)
  - `workflows/gcn_paper/submit_gcn.slurm` (canonical)
- Utility:
  - `workflows/abacus_tweb/abacus_catalog.py` (canonical)

## Experimental

- `workflows/sbi/experimental/jraph_sbi_two_stage.py`
- `local-subgraph-pipeline/*`

## Legacy

- `legacy/abacus_process_particles.py` (use `workflows/abacus_tweb/abacus_process_particles2.py`)
- `legacy/sbi/jraph_sbi_pipeline.py` (legacy SBI path; use `workflows/sbi/jraph_sbi_flowjax.py`)
- `legacy/sbi/jraph_sbi_flowjax_two_stage.py` (retired overlap; optional two-stage path is `workflows/sbi/experimental/jraph_sbi_two_stage.py`)

## Compatibility

- Root-level script names are temporary wrappers and remain available during migration.
- Wrapper deprecation schedule is tracked in `docs/migration/WORKFLOW_REORG.md`.
