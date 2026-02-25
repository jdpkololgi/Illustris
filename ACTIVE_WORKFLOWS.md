# Active Workflow Index

This file is the Phase 0 quick reference for what to run in this repository now.

## Active

- Abacus slab + MPI T-Web:
  - `submit_abacus_tweb_cpu.slurm`
  - `abacus_cactus_tweb.py`
  - `annotate_cutsky_with_tweb.py`
- Jraph training baseline:
  - `jraph_pipeline.py`
  - `submit_jraph.slurm`
- SBI FlowJAX path:
  - `jraph_sbi_flowjax.py`
  - `submit_sbi_flowjax.slurm`
- Utility:
  - `abacus_catalog.py`

## Experimental

- `jraph_sbi_two_stage.py`
- `jraph_sbi_flowjax_two_stage.py`
- `local-subgraph-pipeline/*`

## Legacy

- `abacus_process_particles.py` (use `abacus_process_particles2.py` instead)
- `jraph_sbi_pipeline.py` (legacy SBI path)
