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

### 2026-06-20 — [code] Route A: scale-invariant features fix — best cluster recovery (56%), no in-domain cost
- What: Implemented `--scale-invariant-features` in `build_abacus_sbi_cache.py` +
  `GraphWeb_DESI` inference (`abacus_gnn_parity.py`, `infer_desi_wedge_flowjax.py`):
  per-graph-median normalise the scale-carrying node (Degree/Density/NeighDensity/
  I_eig) and edge (edge_length) features into dimensionless **contrasts** before the
  box-cox/log, leaving density_contrast/directions/Clustering untouched. Rebuilt the
  linear cache (SI), retrained the linear NPE (4×A100), re-ran DESI inference with
  matched SI normalisation.
- Result (DESI cluster fraction, λ_th=0.2; truth 0.058):
  baseline 0.027 → edge-domain-adapt 0.034 (22%) → full node+edge adapt 0.040 (40%)
  → **SI retrain 0.046 (56% of the gap closed — best)**. Abacus-side UNCHANGED
  (Test NLL 1.14, posterior-mean R² 0.826, ordering-viol ~0) → the scale-invariance
  has ~zero in-domain cost, confirming the earlier prediction. Abacus TARP is the
  TIGHTEST of all runs (hugs the diagonal, dev <0.02) and SBC clean — scale-invariance
  preserved (slightly improved) calibration. So SI is the unambiguous production model:
  best Abacus calibration + best DESI transfer, no downside.
- Caveat (honest): not a clean across-the-board win — SI recovers clusters by
  REDISTRIBUTING: void 0.254→0.226 (truth 0.27, worse) and filament 0.273→0.295
  (truth 0.26, worse); wall 0.447→0.433 (toward 0.41, better). Total deviation from
  truth ~flat; it trades a little void/filament accuracy for much better cluster+wall.
  Favourable for a study where clusters are the rare science-critical class.
- Takeaway: scale-invariant features are the recommended DURABLE fix for the
  graph-scale/N(z) transfer shift (learned, not a post-hoc hack). Closes ~56% of the
  cluster deficit; the remaining ~44% needs Route B (mock densification at high z —
  the mock under-produces high-z galaxies; `--equal_data_dens` downsamples the wrong
  way). For Cambridge: present SI as the principled fix with the void/filament caveat.
- Refs: cache `path1_flowjax_3d_lineareig_si`; run `path1_wedge_flowjax_3d_Bcorrected_linear_si`;
  DESI `desi_wedge_flowjax_linear_si`; commits Illustris build-cache + GraphWeb parity/infer.

### 2026-06-19 — [code] DESI cluster-suppression diagnosis: not under-density, not FoG; a training-coverage shift
- What: Characterised why the wedge NPE (and the Jraph regression) under-predict
  **clusters** on real DESI (cluster fraction ~0.027 vs Abacus truth ~0.058 at
  λ_th=0.2). Built a diagnostic suite under
  `/pscratch/.../flowjax_inference_outputs/desi_wedge_flowjax_linear/`:
  eigenvalue corner (clean distributions) + environment corner; λ_th sweep
  (static `lambda_th_sweep.png` + animation `lambda_th_sweep_animation.html`);
  FoG magnitude test (`fog_anisotropy.png`) and the proper LOS-direction FoG test
  (`fog_los_alignment.png`). New scripts in `GraphWeb_DESI/workflows/sbi_inference/`.
- Findings (what the deficit is NOT):
  1. **Not under-density.** Real DESI cluster cores are *denser* than the mock
     (Density p99.9 0.18 vs 0.10; ~2× more galaxies above the Abacus 99th pct).
     Rules out fibre-incompleteness erasing cluster density.
  2. **Not Fingers-of-God.** Recomputed local inertia *eigenvectors*: DESI major
     axes are NOT more line-of-sight aligned than Abacus (median |cosθ| 0.421 vs
     0.456; both slightly transverse from wedge geometry). The earlier anisotropy
     *magnitude* excess (esp. void/wall-inferred) is transverse/generic, not RSD
     elongation. So FoG is ruled out as the differential cause.
  3. **Not training-coverage / extrapolation.** Only ~5% of DESI dense
     cluster-candidates (Density feature) exceed the Abacus train p99.9, ~0.01% of
     all DESI galaxies exceed the train max — the bulk is in-distribution
     (`training_coverage.png`). Too small to explain a halved cluster fraction.
- Findings (what it IS): a genuine **high-λ tail suppression** — persists at every
  threshold and *worsens* with λ_th (DESI/Abacus cluster ratio 0.65→0.45→0.30 at
  λ_th 0.1/0.2/0.3). The richest single cluster IS recovered (skewer P(cluster)=0.86
  in a thin 0.6° beam), so the model isn't blind to clusters — the bulk of moderate
  clusters degrade.
  4. **Not dense-structure shape either.** Oaxaca decomposition of the dense
     inferred-cluster-rate gap (Abacus 0.156 vs DESI 0.082, `shape_misclassification.py`):
     only **8%** is explained by DESI being more elongated; **92% is residual** —
     at MATCHED density AND matched anisotropy, DESI dense galaxies are classified
     as cluster ~half as often as Abacus ones. So the earlier shape hypothesis is
     refuted too.
- Bottom line: **every marginal node-feature explanation is falsified** (density,
  FoG, extrapolation, shape). The deficit is a *fixed-density-fixed-shape residual*
  → a joint / graph-level domain shift in the GNN encoder's feature→λ map (shared by
  regression+NPE, so it lives in the encoder, not the flow). Honest open question,
  not solved. Most concrete remaining lever to test: the **N(z) / number-density
  mismatch** (DESI ~12% more galaxies here, different N(z)) changes the graph EDGE
  SCALE (typical edge lengths) — which would shift GNN output even at fixed node
  features and is consistent with a graph-level, not node-level, residual.
- Edge-scale test (`edge_scale.png`) CONFIRMS a graph-level shift: DESI graph is 12%
  denser, edges 7% shorter, scaled log(edge_length) sits −0.11σ off the Abacus N(0,1)
  — the GNN sees a different graph scale even at fixed node features. Modest but the
  only lever that survives falsification.
- **Phase-0 cheap fix (inference-time domain correction, no retrain):** re-standardising
  DESI scaled EDGE features to training N(0,1) (`--edge-domain-adapt`) recovers ~22% of
  the cluster gap (0.027→0.034) and moves ALL classes toward truth — clean. Adding
  NODE-feature re-standardisation (`--node-domain-adapt`) reaches ~40% (cluster→0.040)
  but over-corrects (void 0.21<0.27, filament 0.30>0.26). So ~40% of the deficit is
  correctable input-domain shift; ~60% is a deeper feature→λ / connectivity residual.
  For the talk: use the clean edge-adapt; durable fix = scale-invariant edge features +
  retrain (Route A), graceful vs the blunt post-hoc rescale.
- Implication: expanding RA/Dec won't fix it. Levers, in order of evidence: (a) match
  the mock **N(z)/number density** to real DESI so the graph scale matches — the
  mock HAS the knobs (`upstream_mkCat_SecondGen_amtl.py --equal_data_dens y`,
  `upstream_prepare_mocks_Y3_bright.py --downsampling y`), both DEFAULT OFF in path1,
  which is why N(z) currently mismatches; (b) full **domain adaptation** (align joint
  feature/graph distribution, not marginals); (c) revisit the **HOD**
  (cluster richness/morphology); (d) pragmatic λ_th=0.1 (0.065 vs truth 0.10). Frame
  for Cambridge as a robust, RSD-NOT cluster systematic *localized* (not
  under-density/FoG/shape) to a graph-scale transfer effect — honest open problem.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{plot_eigenvalue_corner,plot_lambda_th_sweep,plot_fog_anisotropy*,plot_fog_los_alignment}.py`;
  diagnostic figures in the linear DESI run dir. See [[project-path1-sentinel-z-bug]].

### 2026-06-19 — [code] DESI transfer: linear NPE runs on real DESI LOA wedge (conference key result)
- What: Ran the linear-increment FlowJAX NPE on the real DESI LOA wedge (112,755
  bright BGS galaxies, same RA120–160/Dec14.5–30.6/z0.2–0.3 footprint). New
  scripts in `GraphWeb_DESI/workflows/sbi_inference/`:
  `infer_desi_wedge_flowjax.py` (GPU forward) + `plot_desi_wedge_flowjax.py`
  (themed figures). Mirrors the regression inference's preprocessing (node box-cox
  from cache `node_feature_scaler`; edge bidirectional+log+StandardScaler via
  `shared/abacus_gnn_parity.py`) but swaps the model for the GNN encoder + flow
  (128 posterior samples/galaxy → λ posteriors → class probs). Reuses
  `load_flowjax_model`/`create_gnn_and_flow`/`batched_sample_posterior`/
  `samples_to_raw_eigenvalues`/`posterior_to_classprobs`.
- Result: **transfer works, no collapse.** DESI NPE class fractions
  {void .254, wall .447, fil .273, clu .027} sit right between Abacus truth
  {.27/.41/.26/.06} and regression DESI {.246/.462/.266/.026} — wall-dominated,
  cluster-rare. Cluster .027 matches regression (.026), both below Abacus truth
  (.06) = the documented DESI cluster-tail transfer gap.
- NPE-specific findings the regression can't give: (a) per-galaxy posterior WIDTH
  sky map shows coherent structure tracking the cosmic web + survey-footprint
  gaps; (b) width rises near the survey edge (graph-truncation) AND rises with
  eigenvalue extremity (tail distance), NOT with class-boundary ambiguity;
  (c) ordering-violation rate is **4.7% on DESI vs 0.8% on Abacus** — a clean
  quantification of the domain shift (flow less certain on out-of-distribution
  real data; count-based class fracs stay robust). Fixable later with a post-hoc
  sort of posterior samples.
- CRITICAL parity guard: the edge scaler MUST be fit on the **path1 fiberassign**
  wedge npz (the model's training graph), NOT the regression wedge the old script
  uses — both have ~100k nodes so only a constants-assert (mean≈[2.15,0], scale≈
  [0.70,1.77]) catches a mix-up. Hard-asserted in the script.
- Outputs: `/pscratch/.../graphweb_desi/flowjax_inference_outputs/desi_wedge_flowjax_linear/`
  (preds npz + summary.json + 5 themed figures). TARP/SBC stay Abacus-side (no DESI
  truth). Stretch (post-conf): property–environment closure (needs TARGETID→
  fastspecfit join), skewer animation. See [[project-path1-wedge-npe-run]].
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{infer,plot}_desi_wedge_flowjax.py`,
  reused DESI wedge `desi_wedge_expanded_..._from_fullgraph/`.

### 2026-06-19 — [code] Wedge NPE parameterisation study: linear increments win; mock ruled out
- What: Ran the 3-d FlowJAX NPE on the path1 fiberassign wedge (z 0.2–0.3,
  100,935 nodes) under all three eigenvalue parameterisations, 4×A100, 7000
  epochs, seed 42, identical arch/reg/split. Result (Test NLL / posterior-mean
  R² / ordering-violation rate / TARP):
  | param | NLL | R² | viol. | TARP |
  |---|---|---|---|---|
  | softplus increments | +2.25 | 0.74 | 0% | tight |
  | **linear increments** | **+1.13** | **0.82** | **0.8%** | **tightest** |
  | raw eigenvalues | −0.07 | 0.85 | 2.5% | over-confident |
  Softplus's +2 NLL floor was the inverse-softplus heavy tail (init NLL 215 vs
  ~8–10 for linear/raw). Raw gets the lowest NLL but is *over-sharpened*
  (over-confident TARP, 2.5% out-of-order samples). **Linear increments
  Pareto-dominate softplus** (better NLL, R², calibration; negligible 0.8%
  violations) and are the best-calibrated overall — and match the 15-d wedge
  regression's own parameterisation. **Headline choice = linear** for the
  Cambridge talk (calibration-first venue). Hyperparameter tuning (dropout 0→0,
  wd 0.08→0.01) made no difference — not the lever.
- Mock audit cross-ref (separate session "Path1 mock generation audit"): found
  the spectro-injection resurrects fibre-unobserved targets as sentinel z≈0.59
  (2.08M rows, 21.8% of mock_bgs_maglim). **Our wedge is 100% clean** (Z∈[0.200,
  0.300], 0.00% at 0.59 — verified). So mock generation is **ruled out** as the
  NLL-floor cause; the floor is the wedge's genuine information content (sparse
  fiber-incomplete DESI-like selection). The sentinel flaw is a **z-expansion
  blocker**: expand in RA/Dec (or fix the injection first), never in z, until
  the injection is corrected.
- Code (uncommitted→committed this entry): added `--linear-increments` to
  `build_abacus_sbi_cache.py`; `eigenvalues_to_linear_increments` /
  `linear_increments_to_eigenvalues` / `resolve_increment_mode` and a 3-mode
  `samples_to_raw_eigenvalues` in `shared/eigenvalue_transformations.py`;
  `--increment_mode {softplus,linear,raw}` threaded through
  `jraph_sbi_flowjax.py` + `resolve_sbi_paths` (suffix `_linear_eig`) + the saved
  model + `plot_flowjax_posteriors.py`; vectorised the eval (chunked vmap, TARP +
  SBC + class-prob); checkpoint/resume.
- Next: (1) regenerate TARP + class-fraction figures from the linear run in the
  plot-style-guide theme for the deck; (2) LOCK the wedge — do not expand before
  the talk (multi-day rebuild, not needed; calibration is the headline);
  (3) post-conf: fix sentinel injection, then RA/Dec wedge expansion for tighter
  constraints. See [[project-path1-wedge-npe-run]], [[project-path1-sentinel-z-bug]].
- Refs: runs `path1_wedge_flowjax_3d_{testA_reg,testB_raweig,Bcorrected_linear}`
  under `/pscratch/.../abacus/sbi_runs/`; caches `path1_flowjax_3d{,_raweig,_lineareig}`.

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
