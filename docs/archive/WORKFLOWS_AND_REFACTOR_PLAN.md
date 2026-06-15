# TNG/Illustris Workflows and Refactor Plan

> **HISTORICAL (superseded).** This plan reflects the *pre-reorg* state (root-level
> scripts, names like `annotate_cutsky_with_tweb.py`) and a staged refactor that is
> now substantially done — code lives under `workflows/` + `shared/`. Kept for
> context only. For the current state use `ACTIVE_WORKFLOWS.md`, `RUNBOOK.md`, and
> `CLAUDE.md`.

This document defines the current workflows in this repository, what is active vs legacy, and a staged refactor plan to reduce redundancy and path/config drift.

## 1) Workflow map (current state)

### Workflow A: Abacus -> density slabs -> MPI T-Web -> CutSky annotation (**active**)

**Primary entrypoints**
- `submit_abacus_tweb_cpu.slurm`
- `abacus_cactus_tweb.py`
- `annotate_cutsky_with_tweb.py`

**Pipeline**
1. Build density slabs from Abacus mocks (`abacus_process_particles2.py` routines).
2. Run MPI T-Web per slab (`abacus_cactus_tweb.py` imports `run_tweb_memory_optimized` from `abacus_process_particles2.py`).
3. Append `CWEB`, `LAMBDA1`, `LAMBDA2`, `LAMBDA3` to CutSky FITS (`annotate_cutsky_with_tweb.py`).

**Key dependencies**
- `abacus_cactus_tweb.py` -> `abacus_process_particles2.py`
- `annotate_cutsky_with_tweb.py` -> outputs of `abacus_cactus_tweb.py`
- `submit_abacus_tweb_cpu.slurm` -> expects slab file count to equal MPI rank count

**Risks**
- Hardcoded path coupling to `/pscratch/...` and `/global/cfs/...`.
- Docstring drift in `abacus_cactus_tweb.py` still references old particle script name.

---

### Workflow B: PyTorch graph classification pipeline (**active, older stack**)

**Primary entrypoints**
- `submit_gcn.slurm`
- `gcn_pipeline.py`
- `gcn_pipeline_postprocess.py` (post-run analysis)

**Pipeline**
1. Build/load graph features via `Network_stats.py`.
2. Train GCN/GAT model.
3. Save model outputs and run postprocessing/plots.

**Key dependencies**
- `gcn_pipeline.py` -> `Network_stats.py`, `gnn_models.py`, `utils.py`
- `Network_stats.py` -> `Utilities.py`

---

### Workflow C: Jraph baseline classification/regression (**active-ish, mixed with legacy flags**) 

**Primary entrypoints**
- `submit_jraph.slurm`
- `debug_jraph.slurm`
- `jraph_pipeline.py`
- `plot_jraph_logs.py`

**Pipeline**
1. Load cached Jraph data or generate from `Network_stats.network`.
2. Train/evaluate Jraph model.
3. Save artifacts and diagnostics.

**Key dependencies**
- `jraph_pipeline.py` -> `Network_stats.py`, `graph_net_models.py`, `eigenvalue_transformations.py`
- `hyperparameter_tuning.py` and `verify_ensemble.py` call into `jraph_pipeline.py` APIs

**Risks**
- API drift in helper scripts that assume older `load_data()` signatures.

---

### Workflow D: SBI with FlowJAX + GNN encoder (**primary SBI path; active**)

**Primary entrypoints**
- `submit_sbi_flowjax.slurm`
- `run_jraph_sbi_flowjax.sh`
- `jraph_sbi_flowjax.py`
- `plot_flowjax_posteriors.py`

**Pipeline**
1. Load cached transformed/raw eigenvalue targets.
2. Build Haiku GNN encoder.
3. Train FlowJAX conditional flow.
4. Save posterior artifacts and diagnostics.

**Key dependencies**
- `jraph_sbi_flowjax.py` -> `graph_net_models.py`, `eigenvalue_transformations.py`

---

### Workflow E: Two-stage SBI variants (**experimental / legacy mix**)

**Primary entrypoints**
- `run_sbi_two_stage.slurm`
- `jraph_sbi_two_stage.py`
- `jraph_sbi_flowjax_two_stage.py`
- `jraph_sbi_pipeline.py` (legacy SBI path)

**Recommendation**
- Keep only one two-stage implementation if still needed.
- Mark the rest clearly as `legacy/experimental` to prevent accidental use.

---

### Workflow F: Local subgraph pipeline (**active experimental**) 

**Primary entrypoints**
- `local-subgraph-pipeline/train_flowjax_subgraphs.py`
- `local-subgraph-pipeline/eval_local_subgraph.py`

**Pipeline**
1. Build/load local ego-subgraphs.
2. Train integrated local graph + flow models.
3. Evaluate held-out center nodes.

---

## 2) Redundancy and deprecation decisions

### Immediate deprecation target
- `abacus_process_particles.py` -> **deprecate now** (retained only for reference).
- `abacus_process_particles2.py` -> **canonical** particle/T-Web prep implementation.

### Why
- Active MPI script imports only `abacus_process_particles2.py`.
- `abacus_process_particles.py` has stale/incomplete behavior and diverges from active path.

### Deprecation action
1. Add a header comment in `abacus_process_particles.py` stating it is legacy and not used in production.
2. Stop referencing it in docs/comments/scripts.
3. Optional: move to `legacy/abacus_process_particles.py` after one release cycle.

---

## 3) Script dependency map (high level)

- `abacus_cactus_tweb.py` -> `abacus_process_particles2.py`
- `annotate_cutsky_with_tweb.py` -> `abacus_cactus_tweb.py` outputs
- `gcn_pipeline.py` -> `Network_stats.py` -> `Utilities.py`
- `jraph_pipeline.py` -> `Network_stats.py`, `graph_net_models.py`, `eigenvalue_transformations.py`
- `jraph_sbi_flowjax.py` -> `graph_net_models.py`, `eigenvalue_transformations.py`
- `jraph_sbi_two_stage.py` -> `graph_net_models.py`, cached Jraph outputs
- `hyperparameter_tuning.py` -> `jraph_pipeline.py`
- `verify_ensemble.py` -> `jraph_pipeline.py`

Cross-repo:
- `GraphWeb_DESI/graph_catalog.py` -> `TNG/Illustris/Network_stats.py`, `TNG/Illustris/Utilities.py`

---

## 4) Refactor plan (surgical, low-risk)

## Phase 0 (1-2 days): freeze interfaces and label status
- Create a top-level table of active entrypoints and expected outputs.
- Add `ACTIVE`, `EXPERIMENTAL`, `LEGACY` labels to scripts in headers.
- Flag `abacus_process_particles.py` as legacy.

## Phase 1 (2-3 days): centralize paths/config without logic changes
- Add `config_paths.py` with defaults matching current behavior:
  - `CACHE_ROOT`
  - `OUTPUT_ROOT`
  - `ABACUS_SLAB_DIR`
  - `ABACUS_TWEB_OUTPUT_DIR`
  - `DESI_CFS_ROOT`
- Override via environment variables.
- Replace hardcoded `/pscratch/...` and `/global/cfs/...` literals in active entry scripts first.

## Phase 2 (3-5 days): isolate I/O from compute
- For each active heavy script, split into:
  - path resolution/config
  - I/O/cache functions
  - compute functions
  - CLI wrapper (`main`)
- Initial target files:
  - `abacus_process_particles2.py`
  - `abacus_cactus_tweb.py`
  - `annotate_cutsky_with_tweb.py`
  - `jraph_pipeline.py`
  - `jraph_sbi_flowjax.py`

## Phase 3 (2-4 days): remove overlapping SBI variants
- Select one canonical SBI training path and one optional experimental path.
- Move stale alternatives to `legacy/` and update SLURM scripts accordingly.

## Phase 4 (2-3 days): add smoke tests and schema checks
- Add lightweight tests for:
  - import + argument parse for active entrypoints
  - cache key/schema validation
  - transformed vs raw eigenvalue compatibility

---

## 5) Priority bug/doc cleanup list

1. Fix stale `load_cutsky_mock(...)` call in `abacus_process_particles2.py` main block.
2. Update `abacus_cactus_tweb.py` docstring reference to `abacus_process_particles2.py`.
3. Update stale usage/header notes in `jraph_sbi_flowjax_two_stage.py`.
4. Verify and update helper scripts (`hyperparameter_tuning.py`, `verify_ensemble.py`) for current `jraph_pipeline.py` interfaces.

---

## 6) Suggested target structure (incremental)

Keep current files runnable during migration, but converge toward:

- `pipelines/` (workflow orchestration)
- `core/` (pure compute/model transforms)
- `io/` (catalog/cache loading/saving)
- `configs/` (path + run config)
- `scripts/` (thin CLI/SLURM-facing entrypoints)
- `legacy/` (deprecated scripts)

This can be done gradually by introducing wrappers first, then moving internals.
