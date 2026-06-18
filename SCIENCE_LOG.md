# SCIENCE_LOG.md — shared brain: Claude Desktop (science) ⇄ Claude Code (NERSC)

### 2026-06-16 — [code] graphify knowledge graph setup complete (Mac + NERSC)
- What: Installed graphify across Mac (`~/Developer/Illustris`, `~/Developer/GraphWeb_DESI`) and NERSC (`~/TNG/Illustris`, `~/GraphWeb_DESI`). Per-repo graphs built on both machines. Global cross-repo graph built at `~/.graphify/global-graph.json` (tags: `Illustris`, `GraphWeb_DESI`) on both machines. Claude Code PreToolUse hooks (grep/Read/Glob interception) and Cursor `.cursor/rules/graphify.mdc` updated in both repos to reference global graph for cross-repo queries. `graphify-out/` gitignored; `CLAUDE.md` + `.claude/settings.json` + `.cursor/rules/graphify.mdc` travel via git.
- Why: LLMs (Claude Code + Cursor) now consult a knowledge graph before grep/file reads, reducing token cost and surfacing cross-file dependency edges. Global graph captures the Illustris→GraphWeb_DESI model/module dependency that per-repo graphs can't see.
- Next: after any significant code change, run `graphify update .` in the repo then `graphify global add graphify-out/graph.json --as <tag>` to keep global current. Mac global uses `~/Developer/` paths; NERSC uses `~/TNG/Illustris/` paths.

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

### 2026-06-18 — [science] Posterior validation plan beyond TARP: class probabilities, skewer check, boundary degradation, closure tests
- What: TARP only certifies statistical self-consistency, not physical
  correctness on DESI where no per-galaxy truth exists. Defined four concrete,
  implementable validation steps for wedge-NPE q(λ|X) output:
  (1) **Class probabilities from samples.** Per galaxy, draw S posterior samples
  in increment space, invert to physical (λ₁,λ₂,λ₃) (ordering guaranteed by
  construction), threshold at λ_th, count crossings n=Σ1[λ_k>λ_th] (n∈{0,1,2,3}
  by ordering): P(void)=mean(n=0), P(wall)=mean(n=1), P(filament)=mean(n=2),
  P(cluster)=mean(n=3). Cross-check analytically via marginal CDFs: P(λ_k>λ_th)
  from each marginal. **NB ordering is ASCENDING λ₁≤λ₂≤λ₃ (CACTUS-native: eig1
  smallest, eig3 largest — verified end-to-end, reproduces CWEB exactly at
  λ_th=0.2), so the decomposition is P(void)=1-P(λ₃>λ_th),
  P(wall)=P(λ₃>λ_th)-P(λ₂>λ_th), P(filament)=P(λ₂>λ_th)-P(λ₁>λ_th),
  P(cluster)=P(λ₁>λ_th)** — the *smallest* eigenvalue λ₁ exceeding λ_th ⇒ cluster
  (all exceed). (An earlier draft of this entry flipped λ₁↔λ₃, matching a
  descending convention the codebase does not use.) Sample-based and analytic
  versions must agree to MC error — mismatch indicates an inversion or sampling
  bug. **λ_th must equal the CWEB-labeling threshold (Abacus = 0.2, the CACTUS
  default), not 0.0.** Implemented in
  `shared/eigenvalue_transformations.py::posterior_to_classprobs`.
  (2) **Reliability diagram for class probabilities** (discrete-output
  complement to TARP): on Abacus (CACTUS ground truth available), bin galaxies
  by predicted P(filament) (and other classes), plot empirical true-class
  fraction per bin vs predicted probability. Diagonal = calibrated; deviation
  flags over/under-confidence, expected to concentrate at wall/filament
  boundary per RASTI confusion matrix.
  (3) **Posterior-width vs distance-to-boundary.** Compute per-galaxy posterior
  width (e.g. mean marginal σ across λ₁,λ₂,λ₃, or posterior entropy) vs distance
  to wedge/survey edge (RA/Dec/z footprint boundary, alpha-complex hemisphere
  split) and vs distance to T-web class boundary (λ near λ_th). Expect
  monotonic widening near both. On Abacus (truth known) also plot width vs
  true prediction error to confirm widening tracks real degradation, not just
  graph truncation artifacts; confirm DESI shows the same *shape* of
  width-vs-edge-distance curve as Abacus.
  (4) **Property–environment closure test (DESI-only, no truth needed).** Bin
  real DESI BGS galaxies by inferred environment (E[trace(λ)] or dominant class
  probability) and check recovery of known relations: quenched fraction /
  colour / sSFR rising toward filament→cluster, morphology–density trend.
  Recovering established astrophysical trends from inferred-only environment is
  evidence the posteriors carry real physical information.
  Also scoped (5) cross-check against independent structure finder (DisPerSE
  filaments, or BORG T-web per existing on-the-horizon validation target):
  expect bulk agreement, divergence concentrated at class boundaries as the
  reassuring signature, not a failure mode.
  (6) **Animated line-of-sight skewer plot.** Pick a 1D skewer through a wedge
  (Abacus or DESI Loa) crossing void→wall→filament→cluster. At each position
  along the skewer: render the three eigenvalue marginal posterior densities
  (KDE over per-position-bin galaxy samples, not Gaussian — Gaussian only used
  for the Desktop mockup) with λ_th marked, a posterior-width ribbon, and the
  class-probability bar from (1)/(2)'s class-prob math. Animate by sweeping
  position; render as a saved video/gif, not just an interactive widget, so it
  drops into the talk deck. On Abacus, overlay the true CACTUS (λ₁,λ₂,λ₃) as
  moving markers on the posterior panel — credible bands containing truth
  through the transition, widening exactly where truth nears λ_th, is the
  core "physically sensible" evidence this whole validation plan is for.
- Why / decision: visualizing 3 correlated posteriors per galaxy across an
  entire wedge does not scale; need scalar/derived functionals overlaid on
  geometry instead of raw posterior dumps, and need validation that is
  meaningful on DESI specifically (no per-galaxy truth there), not just on
  Abacus. (1)+(2) give a calibration check at the class level; (3) gives a
  boundary-physics sense-check transferable Abacus→DESI; (4) gives a truth-free
  closure test directly on real BGS data. (6) is the figure that actually
  conveys physicality to a human audience — the others are diagnostic plots,
  this is the one that shows the posteriors tracking real structure.
- Next: implement (1) as a `posterior_to_classprobs(samples, lambda_th)`
  utility in `shared/eigenvalue_transformations.py` (sample-count path +
  analytic-CDF path, with an assert/warning if they diverge beyond MC
  tolerance); implement (2) reliability diagram + (3) width-vs-boundary-distance
  + (6) skewer animation in `workflows/visualization/` using
  `shared/plot_style.py` (COSMIC_WEB_COLORS for class bins, matplotlib
  FuncAnimation or equivalent for (6)); run (3) and (6) on one Abacus wedge
  first (truth overlay available), then the matching DESI Loa wedge, compare;
  (4) needs BGS property table joined to wedge-NPE output by target ID — check
  join keys exist in current DESI Loa wedge cache before scripting. (5)
  BORG/DisPerSE cross-check stays "on the horizon" pending BORG catalogue
  access — not in this sprint.
- Refs: `shared/eigenvalue_transformations.py` (ordered softplus increment
  policy — class-prob math must operate in this space then invert), CACTUS
  labels (Abacus truth), DESI BGS LSS catalogues at
  `/global/cfs/cdirs/desi` (property–environment closure), RASTI confusion
  matrix (wall/filament boundary expectation), `PLOT_STYLE_GUIDE.md`. Desktop
  mockup of (6) (Gaussian-approximation, interactive, not the real KDE version)
  built this session for layout reference.

### 2026-06-17 — [science] Plot style guide finalized: true-black theme, cosmic-web class colors, IBM Plex Sans
- What: Defined canonical matplotlib style for all GraphWeb_DESI plots: true-black
  (#000000) background, off-white (#F2F2F2) text/ticks; confirmed FLATS-deck
  cosmic-web class colors (Void #A1FCDD, Wall #4E84F7, Filament #EB336F, Cluster
  #F5C144) as a categorical-only palette, kept separate from the FLATS general
  accent colors (magenta #FF006E, blue #3A86FF, red #D62828); replaced Eurostile
  with IBM Plex Sans as the single font everywhere (Eurostile has no usable Greek
  glyphs, which matters for λ-notation); mathtext.fontset='dejavusans' handles
  λ-subscripts in labels regardless of body font. Wrote PLOT_STYLE_GUIDE.md +
  shared/plot_style.py (apply_style(), finalize_axes(), COSMIC_WEB_COLORS,
  CLASS_ORDER, ACCENT_COLORS) in GraphWeb_DESI.
- Why / decision: Every plot in the pipeline (regression diagnostics, NPE
  posteriors, TARP coverage, DESI BGS results) needs one consistent visual
  identity before Cambridge SBI-GalEv talk figures are regenerated from
  wedge-NPE output; relying on Eurostile directly would have broken λ-notation
  rendering.
- Next: Download IBM Plex Sans .ttf files into GraphWeb_DESI/assets/fonts/
  (command in PLOT_STYLE_GUIDE.md §2.3) on both Mac and NERSC; migrate existing
  plotting scripts/notebooks (workflows/visualization/*) to
  `from shared.plot_style import apply_style, finalize_axes`; consider mirroring
  plot_style.py into Illustris if TNG-side plots need the same theme.
- Refs: GraphWeb_DESI/PLOT_STYLE_GUIDE.md, GraphWeb_DESI/shared/plot_style.py,
  FLATS deck slide 9 (legend swatches).

### 2026-06-16 — [code] graphify global graph + cross-repo Cursor rules
- What: Merged `Illustris` and `GraphWeb_DESI` into `~/.graphify/global-graph.json`;
  updated `.cursor/rules/graphify.mdc` in both repos to name the sibling repo and
  require `--graph ~/.graphify/global-graph.json` for cross-repo queries.
- Why / decision: Per-repo `graphify query` and grep do not surface Illustris↔DESI
  dependencies; global graph does (e.g. `graph_net_models` → Jraph wedge inference).
- Next: After substantive code changes, `graphify update .` then
  `graphify global add graphify-out/graph.json --as <tag>`.
- Refs: `Illustris/.cursor/rules/graphify.mdc`, `GraphWeb_DESI/.cursor/rules/graphify.mdc`

### 2026-06-15 — [code] Environment policy clarified for agents
- What: Updated agent/runbook guidance: activate `cosmic_env` for normal work and
  `rapids-gnn` whenever calculating graph metrics/features.
- Why / decision: Phase 4 validation on an unprepared Cloud image failed on
  missing scientific dependencies; graph metrics also require the RAPIDS/cuGraph
  stack rather than the default environment.
- Next: Keep workflow launchers aligned with this policy as new graph-metric or
  wedge-NPE scripts are added.
- Refs: `CLAUDE.md`, `RUNBOOK.md`.

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
