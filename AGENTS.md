# AGENTS.md

This file is the canonical agent job description for this repository (`CLAUDE.md` and `GEMINI.md` are symlinks to this file).

## Deliver the requested result

- Keep the user's named deliverable active across turns and days until it is delivered,
  paused, or replaced. Treat status questions and corrections as part of that task.
  After an interruption, recover the last explicit request and outstanding work from
  the chat or existing handoff. Ask one focused question if the objective is still unclear.
- Keep routine planning, progress and summaries in the chat. Create a standalone
  Markdown plan, checklist or report only when the user requests it or an existing
  workflow requires it. Record durable scientific decisions once in `SCIENCE_LOG.md`
  and link the original evidence. Put Slurm submissions, `PENDING` states, tmux sessions,
  syntax checks, and routine run receipts in `RUN_LOG.md` (not `SCIENCE_LOG.md`).
- Use existing outputs first. Investigate an anomaly only as far as it could affect
  the requested result. Start with the cheapest check that distinguishes the likely
  explanations. If the anomaly is incidental, return to the deliverable. If it blocks
  a trustworthy answer, report the evidence and the specific decision or input needed.
- Keep existing NERSC and Slurm approvals in force. Return the result in chat with
  the figure, table or evidence path and any material limitation. Stop at delivery.

## Scientific claim & limitation guardrails

- **5-column Claim Key (mandatory for comparisons):** Every cross-model or cross-run
  headline number in chat, `CONTEXT.md`, or `SCIENCE_LOG.md` must carry its exact
  `metric | split_type | phase_set | z_range | artifact_id`. Never compare numbers
  whose `split_type`, `phase_set`, `z_range`, or row-set hash do not match.
- **10-row raw-data trace before accepting any limitation:** Before declaring a
  redshift range, catalogue subset, or method an "upstream/physical limitation" or
  proposing to exclude it, inspect at least 10 raw source rows (coordinates, IDs,
  masses/labels) against the primary catalogue/paper and show the concrete trace to
  the user.

## Documentation & state segregation

- `CONTEXT.md` — **overwritten live state dashboard (≤ 8 KB / ~150 lines).** Read
  first at session start (`get-up-to-speed` skill). Never append dated logs here;
  overwrite in place on `save-state`.
- `SCIENCE_LOG.md` — **append-only science log** (completed scientific results,
  negative results, bug discoveries, methodological decisions only). Older history
  is archived in `docs/log_archive/`.
- `RUN_LOG.md` — **append-only operational/Slurm ledger** (submissions, queue state,
  tmux owners, wrapper checks, SHA-256 receipts).
- `README.md`, `ACTIVE_WORKFLOWS.md`, `RUNBOOK.md` — static orientation, canonical
  entrypoints, and Perlmutter recipes.

## Project Overview

This repository implements machine learning pipelines for inferring cosmic web
structure from galaxy observables in IllustrisTNG and Abacus mock catalogs. The
main targets are local density-Hessian eigenvalues and derived T-Web classes
for voids, walls, filaments, and clusters.

Start with:

- `CONTEXT.md` for current live state, active deliverables, Claim & Provenance Table, and next steps — **read this first** (or invoke `get-up-to-speed`).
- `SCIENCE_LOG.md` for recent scientific decisions and completed results (if a recent scientific entry conflicts with `CONTEXT.md`, follow `SCIENCE_LOG.md` and refresh `CONTEXT.md`).
- `RUN_LOG.md` for recent Slurm submissions, queue telemetry, and operational receipts.
- `README.md`, `ACTIVE_WORKFLOWS.md`, `RUNBOOK.md` for orientation, canonical entrypoints, and Perlmutter commands.

## Running Jobs On NERSC Perlmutter

Production workflows generally run through SLURM. Always activate the expected
Python environment before running tests, help commands, workflow scripts, or
interactive diagnostics:

- Use `cosmic_env` for all normal repository work: T-Web annotation, graph
  construction/subsetting, cache building, Jraph/SBI training, GCN workflows,
  plotting, tests, and documentation validation.
- Use the RAPIDS/cuGraph `rapids-gnn` environment whenever calculating graph
  metrics/features. This includes `abacus_graph_features_cugraph.py`,
  `abacus_graph_features.py`, and any new graph-metric recomputation scripts.
  The default path is controlled by `ABACUS_RAPIDS_ENV_PATH` and currently
  falls back to `/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn`.

### JAX/Jraph Regression Pipeline

```bash
sbatch workflows/jraph/submit_jraph.slurm
python workflows/jraph/jraph_pipeline.py --prediction_mode regression --epochs 10000
```

### PyTorch GCN Paper Pipeline

```bash
sbatch workflows/gcn_paper/submit_gcn.slurm
python workflows/gcn_paper/gcn_pipeline.py --help
```

### TNG / Full-Graph SBI FlowJAX

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
```

### Abacus SBI — wedge NPE (current) / partitioned FlowJAX (legacy)

The current Abacus-scale SBI path is **NPE on wedge subvolumes** (one graph per
RA/Dec/z wedge). Per `SCIENCE_LOG.md` this runs interactively today; a production
`sbatch` submit script is still an open thread.

The **partitioned / graph-partitioned FlowJAX** path below is **legacy and
abandoned** — kept for reference only; do not start new runs from it:

```bash
sbatch workflows/abacus_tweb/submit_build_partitions_adaptive.slurm
sbatch workflows/sbi/submit_sbi_partitioned_data_parallel.slurm
```

### Key Environment Setup

Default environment:

```bash
source ~/.bashrc
conda activate cosmic_env
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
```

Graph-metric environment:

```bash
source ~/.bashrc
unset PYTHONPATH PYTHONHOME LD_PRELOAD
source /global/homes/d/dkololgi/miniforge3/bin/activate "${ABACUS_RAPIDS_ENV_PATH:-/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn}"
```

## Architecture

### Main Pipeline Approaches

1. **Abacus T-Web and mock graph pipeline** (`workflows/abacus_tweb/`): builds
   slabwise T-Web outputs, annotates DESI/Abacus CutSky mocks at each galaxy's RA/DEC/Z_COSMO real-space
   position (all populations; independent checks recorded separately), constructs alpha/Delaunay graph artifacts, computes graph features,
   and builds SBI caches (wedge subvolumes are current; partitioned caches are
   legacy).
2. **Regression** (`workflows/jraph/jraph_pipeline.py`): JAX/Jraph
   GraphNetwork predicting eigenvalue (ordered softplus increments).
3. **SBI** (`workflows/sbi/`): GNN encoder plus a normalizing-flow posterior
   (NPE). The current Abacus-scale path is **NPE on wedge subvolumes**; the
   partitioned `jraph_sbi_flowjax_partitioned.py` path is legacy/abandoned.
4. **Classification** (`workflows/gcn_paper/gcn_pipeline.py`): PyTorch/Torch
   Geometric GCN/GAT workflow for 4-class T-Web classification.

### Key Modules

| Module | Purpose |
| --- | --- |
| `shared/graph_net_models.py` | JAX GraphNetwork and encoder helpers. |
| `shared/eigenvalue_transformations.py` | Target transforms. **Canonical = ordered softplus eigenvalue increments** (λ₁ + cumulative softplus → λ₁ ≤ λ₂ ≤ λ₃); invert to (λ₁,λ₂,λ₃) for eval only. Shape-param/invariant converters deprecated as ML targets. |
| `shared/config_paths.py` | Environment-variable driven Perlmutter and scratch paths. |
| `shared/tng_pipeline_paths.py` | TNG/Jraph/SBI cache and output path resolution. |
| `shared/resource_requirements.py` | Runtime guards for CPU/GPU SLURM allocations. |
| `shared/sbi_cache_schema.py` | Cache-schema helpers used by tests and SBI paths. |
| `workflows/gcn_paper/gnn_models.py` | PyTorch GCN/GAT model definitions. |
| `workflows/gcn_paper/Utilities.py` | TNG data loading and graph construction for the paper workflow. |
| `workflows/gcn_paper/Network_stats.py` | Graph feature extraction and T-Web classification utilities. |

### Physics Targets

**Canonical target representation.** Models are trained on the tidal-tensor
*eigenvalues* (λ₁ ≤ λ₂ ≤ λ₃), parameterised as **ordered softplus increments**:
predict λ₁ directly, then λ₂ = λ₁ + softplus(·), λ₃ = λ₂ + softplus(·). This
enforces λ₁ ≤ λ₂ ≤ λ₃ by construction and is the canonical head for both the
regression and SBI-flow stacks. The policy and converters live in
`shared/eigenvalue_transformations.py` (`eigenvalues_to_increments` /
`increments_to_eigenvalues`).

- Do **not** train a direct 3-output (λ₁, λ₂, λ₃) head — it reintroduces
  ordering violations. Use the increment parameterisation.
- The network trains and predicts in increment space; the inverse map to
  physical (λ₁, λ₂, λ₃) is applied only at evaluation/plotting time, never as
  the training target.
- Shape-parameter (I₁, e, p) and invariant (I₁, I₂, I₃) representations are
  **deprecated** as ML targets — their distributions are pathological. The
  `--use_shape_params` flag and the shape/invariant converters are retained for
  legacy caches only; do not use them for new runs.

For Abacus SBI caches, inspect `build_abacus_sbi_cache.py --help` and the
generated cache metadata to confirm the target parameterisation that was
written.

### Data Flow

1. Load IllustrisTNG subhalos or Abacus/DESI CutSky mock galaxies.
2. Assign or load T-Web Hessian eigenvalues.
3. Construct graph topology via Delaunay, MST, alpha-complex, or wedge
   subvolumes depending on workflow (partitioned subgraphs are legacy).
4. Extract node and edge features.
5. Train regression, classification, or conditional density models.

### Caching

Canonical cache and output roots are resolved by `shared/config_paths.py`.
Override with `TNG_CANONICAL_CACHE_ROOT`, `TNG_CANONICAL_OUTPUT_ROOT`,
`TNG_JRAPH_CACHE_DIR`, `TNG_SBI_CACHE_DIR`, and related variables rather than
hard-coding new scratch paths.

## Local Subgraph Pipeline

The `local-subgraph-pipeline/` directory contains an independent inductive pilot
that trains on batched ego-graphs extracted from cached TNG graph data. It is
separate from the transductive full-graph Jraph/SBI paths.

## Testing

For lightweight local validation, run:

```bash
python -m unittest discover -s tests/phase4
```

These tests do not replace Perlmutter-scale scientific validation, but they are
useful for catching broken entrypoints, cache-schema drift, and eigenvalue
transformation regressions.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

**Sibling repo (cross-project):** `~/GraphWeb_DESI` — DESI BGS observed-galaxy graphs and inference (GAT + Jraph wedge). It imports trained models and shared modules from this repo (`shared/graph_net_models.py`, GAT checkpoints, Abacus-trained Jraph weights).

Rules:
- MANDATORY: Before using Read, Grep, Glob, or Bash to explore the codebase, run graphify first.
- Environment note (local vs NERSC): On local macOS, `graphify` (`~/.local/bin/graphify`) is on `PATH`. On NERSC Perlmutter, `graphify` is installed inside `cosmic_env` (and `rapids-gnn`), not on the default non-interactive `PATH` — invoke `/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/graphify` directly or activate `cosmic_env` first (`source ~/.bashrc && conda activate cosmic_env && graphify ...`).
- For codebase questions, run `graphify query "<question>"`. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost); then refresh the global with `graphify global add graphify-out/graph.json --as Illustris`.
- Only use Read/Grep/Glob directly when graphify has already oriented you and you need to modify or debug specific lines.
- This rule applies to subagents too — include it in every subagent prompt involving code exploration.

**Cross-repo questions** (Illustris ↔ GraphWeb_DESI, DESI inference dependencies, shared models/paths): use the global graph:
- `graphify query "<question>" --graph ~/.graphify/global-graph.json`
- `graphify path "<A>" "<B>" --graph ~/.graphify/global-graph.json`
- `graphify explain "<concept>" --graph ~/.graphify/global-graph.json`
