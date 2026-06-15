# SCIENCE_LOG.md — shared brain: Claude Desktop (science) ⇄ Claude Code (NERSC)

The boundary object that keeps **science discussions** (Claude Desktop, local Mac)
and **agent work** (Claude Code, NERSC) aware of each other. It rides git, so it
reaches every machine.

- NERSC path: `~/TNG/Illustris/SCIENCE_LOG.md`
- Mac path:   `~/Developer/Illustris/SCIENCE_LOG.md`
- Syncs through GitHub (`jdpkololgi/Illustris`).

## How to use this file (both assistants)

1. **Read it at the start of a session** to load current direction before doing
   anything substantive.
2. **Append a short entry** (newest at the top of the Log) when a decision is
   made, a result lands, or direction changes. Don't rewrite history; add.
3. **It's only current after `git pull`.** Pull before reading/editing; commit +
   push after writing. (Stage just this file: `git add SCIENCE_LOG.md`.)
4. Keep entries terse and skimmable. Prune "Open threads" as items close.

Tags: `[science]` = decisions/hypotheses/conclusions from Desktop discussions.
`[code]` = what was run/changed/found, blockers, next actions from NERSC.

Entry shape:

```
### YYYY-MM-DD — [science|code] short title
- What: ...
- Why / decision: ...
- Next: ...
- Refs: files, commits, run dirs
```

## Open threads / current focus

- **NPE on Abacus wedge subvolumes** is the path forward for SBI. Graph-partitioned
  FlowJAX is legacy ("a nightmare") and to be retired when convenient.
- **GraphWeb production VAC** needs a real `sbatch` submit script — the DESI
  graph→features→inference chain currently only runs interactively.
- Jraph regression on wedge/cube caches is active and fine.

## Log (newest first)

### 2026-06-15 — [code] Docs aligned to wedge-NPE Abacus path
- What: Updated top-level and workflow docs so Abacus-scale SBI points to
  RA/Dec/z wedge subvolume caches with `jraph_sbi_flowjax.py`, while partitioned
  FlowJAX is marked legacy/reference.
- Why / decision: Several public docs still described partitioned Abacus FlowJAX
  as active, contradicting the current wedge-subvolume SBI direction.
- Next: Add a production `sbatch` launcher for wedge NPE when the interactive
  recipe stabilizes.
- Refs: `README.md`, `ACTIVE_WORKFLOWS.md`, `RUNBOOK.md`,
  `workflows/abacus_tweb/README.md`, `workflows/sbi/README.md`.

### 2026-06-15 — [science] Target representation: ordered increments are canonical, not "raw eigenvalues"
- What: Cleared a doc ambiguity that conflated two independent axes. (1) Target
  *quantity*: eigenvalues beat shape-param (I₁,e,p) and invariant (I₁,I₂,I₃)
  reps, which are pathological for ML. (2) Target *parameterisation*: eigenvalues
  are trained as ordered softplus increments (λ₁ anchor + two non-negative
  increments), enforcing λ₁≤λ₂≤λ₃ by construction. The net trains/emits in
  increment space; inversion to (λ₁,λ₂,λ₃) is presentation/eval-only.
- Why / decision: "raw eigenvalues preferred" (true only on axis 1) could lead an
  agent to collapse the softplus head to a direct 3-output regressor and silently
  reintroduce ordering violations. Canonical policy now documented in code + docs.
- Next: retire the shape-param path properly — remove `--use_shape_params` and the
  dead `compute_shape_param_statistics` (builds `stats`, returns `None`) from
  `jraph_pipeline.py` + `eigenvalue_transformations.py` once confirmed unused.
- Refs: `shared/eigenvalue_transformations.py` (new module docstring + dep
  banners), `CLAUDE.md` (Physics Targets, module table, regression cmd).

### 2026-06-15 — [science] Cambridge SBI-GalEv talk: framing + reusable assets
- What: Talk narrative = open on environment→galaxy-evolution motivation, then
  frame the whole pipeline as amortised per-galaxy posterior inference of T-Web
  eigenvalues with TARP calibration as the headline. Regression = validated
  encoder/forward-model backbone; NPE + flows = frontier result. Reviewed prior
  poster (classification-era) and FLATS deck: deck already has FlowJAX
  posterior-comparison + TARP-coverage figures (from the retired partitioned
  path) — reuse as visual template, regenerate from wedge-NPE.
- Why / decision: venue is SBI-first, so calibration/posterior must lead, not
  cartography; self-diagnosed weak spot is astro motivation, so first ~3 min
  carry the science case. Slot Thu 25 Jun ~11:30, 15-min block (confirm 12+3 vs
  15+5 with organisers).
- Next: (1) NPE on one Abacus wedge + TARP plot = critical path; (2)
  R²-vs-smoothing-scale plot; (3) polish 3-way λ hists + class-fraction bar + 3D
  wedge render in deck palette (Eurostile, #FF006E/#3A86FF). Stretch: ξ(r_p,π)
  forward-model check, 2nd wedge for cosmic-variance scatter. Methods-slide
  architecture diagram drafted (draw.io).
- Refs: FLATS deck; `workflows/jraph/`, wedge caches; visualization notebook.

### 2026-06-14 — [code] Cross-tool infra + this log
- What: Set up `~/.claude/CLAUDE.md` as cross-repo canon (3-location map, conda
  envs, proven Perlmutter `salloc`/`srun` recipes; production = `sbatch`). Fixed
  the `nersc` skill's `qos.md` (no `gpu_`-prefixed QOS). Added two Claude Code
  skills: `desi-bgs-graph` and `jraph-eval`. Created this SCIENCE_LOG.
- Next: stand up the Desktop⇄NERSC bridge (filesystem MCP + git) so science
  decisions land here automatically.
- Refs: `~/.claude/CLAUDE.md`, `~/.claude/skills/{desi-bgs-graph,jraph-eval}`.

### 2026-06-14 — [science] SBI direction: wedges, not partitions
- What: Abacus-scale inference works via **wedge subvolumes** (graph per RA/Dec/z
  wedge); the graph-partitioning / partitioned-FlowJAX path is abandoned.
- Why / decision: graph partitioning was a nightmare; wedge training has worked
  over the last ~2 months. Future = NPE on the wedge subvolumes.
- Next: when convenient, retire `workflows/sbi/*partitioned*` + partition builders.
- Refs: `workflows/sbi/`, see also Claude Code memory `project-abacus-wedge-npe-direction`.
