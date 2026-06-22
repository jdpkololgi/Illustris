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

### 2026-06-22 — [science] Talk script finalized v2.1: all figures locked, deck-build is the only remaining task
- What: Recovered the interrupted talk-prep thread and confirmed against `Plots/` that all
  6 previously-"STILL NEEDED" figures (slide 3 idealised skewer, slide 6 three-way TARP,
  slide 7 Abacus TARP+SBC, slide 9 edge_scale + cluster recovery bars) now exist, matching
  the NERSC 06-22 "6/6 complete" entries below. Updated
  `~/Developer/Working Files/SBI-Galev-2026/cambridge_sbi_galev_2026_script.md` to v2.1:
  every slide's Visual line now points at a FINAL figure filename instead of a placeholder;
  numbers reconciled to the actual figures (slide 6 max|ECP−α| softplus/linear/raw =
  0.012/0.029/0.082; slide 9 recovery sequence 0.027→0.034→0.040→0.046 vs truth 0.058,
  ~60% of gap closed — corrected from the earlier "56%/two-point" figure). All 4 OPEN
  DECISIONS and the STILL-NEEDED list marked resolved/closed in the script.
- Why / decision: with content locked, the only remaining work before Thursday is
  mechanical — building the actual Keynote deck (`SBI-Galev-2026.key`, currently still a
  33-slide copy of the classification-era FLATS talk) from the v2.1 script. Confirmed via
  keynote-mcp the file is open at 33 slides; identified reuse candidates by content audit:
  slide 1 (title layout), slide 3 "The Cosmic Web" (T-web explainer layout), slide 8
  "Combining graphs and machine learning" (pipeline-stage layout) — consistent with the
  06-15/06-21 plan to inherit theme + 3 layouts and rebuild everything else.
- Next: confirm rebuild plan with Dakshesh (strip/repurpose old slides vs keep-and-renumber),
  then build the 11 main + ~8 backup slides via keynote-mcp using the v2.1 script + `Plots/`
  images; convert the two HTML animations (`skewer_idealised.html`,
  `skewer_posterior_animation_real.html`) to mp4/gif for embedding (no live HTML in-talk);
  timer dry-run once built (OPEN DECISIONS #4, still pending).
- Refs: `cambridge_sbi_galev_2026_script.md` v2.1; `SBI-Galev-2026.key`; `Plots/` (all);
  see 06-21/06-22 [code] entries below for figure provenance.

### 2026-06-22 — [code] Dropped three-way TARP; SBC + TARP (with error band) for production linear only
- Decision (user): stop framing results as a parameterisation comparison. Removed
  `three_way_tarp.png` and the band-less copied `abacus_tarp_coverage.png` /
  `abacus_sbc_calibration.png` from the SI folder + canonical mirror.
- NEW `TNG/Illustris/workflows/sbi/plot_calibration_linear.py` (GPU; reuses
  plot_flowjax_posteriors' loaders): SBC + TARP for the **production linear-increment SI
  model**. Outputs `tarp_linear.png` — TARP with a **bootstrap 1σ error band** (N=3000 test
  points, max |ECP−α| = 0.015, i.e. at the MC noise floor √(0.25/N)≈0.009 → consistent with
  perfect calibration) — and `sbc_linear.png` — rank histograms for λ₁/λ₂/λ₃ with the
  expected ±2σ band (flat within band; mild λ₁-upper / λ₂-lower edge structure).
- WHY drop the comparison: the three-way TARP's softplus(0.012) < linear(0.029) ordering sat
  inside the 500-point MC noise floor (per-bin SE≈0.022), so it was NOT a real calibration
  ranking — and claiming "linear best-calibrated" would be a hostage to fortune. Linear's
  case rests on Pareto-dominance (NLL +1.13 vs +2.25, R² 0.82 vs 0.74, ordering-viol 0.8%),
  not "tightest TARP". (Supersedes/corrects the 2026-06-19 "linear tightest TARP" note.)
- `FIGURE_GUIDE.md` updated to the new pair; `plot_tarp_threeway.py` now unused.
- Uncommitted (await go-ahead): + `plot_calibration_linear.py` (Illustris).
- Refs: SI run dir `tarp_linear.png`, `sbc_linear.png`.

### 2026-06-22 — [code] Skewer HTML → GIF for Keynote (NERSC ffmpeg has no h264)
- `render_skewer_video.py` NEW (`GraphWeb_DESI/workflows/sbi_inference/`): extracts the
  embedded JSON payload from a skewer HTML (`var D={…}`) and re-renders the same 3 panels
  (galaxy strip / three λ posteriors + λ_th / class-prob bar) natively with matplotlib +
  shared.plot_style → mp4 or gif. No browser/headless-Chromium needed.
- WHY gif not mp4: the NERSC system ffmpeg (`/usr/bin/ffmpeg`) is built WITHOUT libx264
  (encoders present: only gif, libvpx-vp9; h264/mpeg4 decoders disabled), and
  imageio-ffmpeg isn't installed — so no H.264. Keynote plays + loops GIFs natively, so
  produced GIFs: `skewer_idealised.gif` (3.0 MB), `skewer_real.gif` (0.97 MB) in the SI
  folder + canonical. The renderer auto-falls-back mp4→gif on encode failure.
- For a true H.264 mp4: convert the gif on the Mac (its ffmpeg has libx264):
  `ffmpeg -i x.gif -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" x.mp4`.
- Also fixed `sync_figures_to_canonical.sh` to include *.gif/*.mp4 (were excluded → the
  gifs weren't reaching the canonical dir).
- Uncommitted (await go-ahead): + `render_skewer_video.py` + sync-script edit (GraphWeb).
- Refs: SI run dir `skewer_{idealised,real}.gif`.

### 2026-06-22 — [code] Idealised skewer (slide 3) + per-figure interpretation guide; STILL NEEDED 6/6
- Slide 3 — `skewer_idealised.html` NEW (`GraphWeb_DESI/workflows/sbi_inference/build_skewer_idealised.py`):
  the discrete→continuous bookend. IMPORTS the real skewer's HTML template so layout + the
  standardised style (true-black / IBM Plex Sans / COSMIC_WEB_COLORS / λ_th=0.2) are identical
  — fixes the gap that the Desktop mockup had — but is driven by CLEAN synthetic structure:
  one density peak the sightline crosses, λ₃/λ₂/λ₁ cross λ_th at d≈0.33/0.60/0.86 so the class
  bar morphs void→wall→filament→cluster→…→void (peak P(cluster)=0.78). Slide 10 = the same
  layout on real posteriors. **STILL-NEEDED list now complete (6/6).**
- `FIGURE_GUIDE.md` written into the SI run dir: per-figure interpretation + error-bar
  provenance + variable definitions, esp. closure — inferred env = hard_class (argmax of the
  λ>λ_th class probs) / trace_lambda (E[Σλ] ∝ density); properties = LOGMSTAR, SFR,
  log sSFR=log10(SFR)−LOGMSTAR, quenched≡log sSFR<−11, g−r=ABSMAG01_SDSS_G−R from loa
  FastSpecFit; CIs = Wilson (fractions) / bootstrap 16-84 (medians) / TARP bootstrap bands /
  SBC rank uniformity.
- Uncommitted (await go-ahead): + `build_skewer_idealised.py` (GraphWeb).
- Refs: SI run dir `skewer_idealised.html`, `FIGURE_GUIDE.md` (+ canonical mirror).

### 2026-06-22 — [code] Slide 6 three-way TARP done; label fixes; "STILL NEEDED" now 5/6
- Slide 6 — `three_way_tarp.png` NEW (`TNG/Illustris/workflows/sbi/plot_tarp_threeway.py`,
  GPU): recomputes TARP coverage for the 3 parameterisation models on their own test
  splits and overlays them (bootstrap 1σ bands), reusing plot_flowjax_posteriors' eval
  machinery. Result = the slide-6 message exactly: max|ECP−α| **softplus 0.012, linear
  0.029, raw 0.082** — raw eigenvalues win NLL but are clearly OVER-confident (curve bows
  above the diagonal); softplus/linear hug it. (One JAX gotcha: flowjax needs new-style
  typed keys — `jax.random.key`, not `PRNGKey`.) Written to the SI folder + synced.
- Fixes from user review of yesterday's batch: (a) class-fractions legend now consistent
  "[domain] NPE" → `Abacus NPE` / `DESI NPE` (was "NPE DESI"); (b) recovery-bars 4th stage
  relabelled `Per-graph norm. (production)` (dropped "SI" — the fix is per-graph
  domain-relative normalisation, per the talk-v2 correction, not "scale-invariant").
- STILL NEEDED now just (d) slide-3 idealised fake-data skewer (generative; matches the
  real-skewer panel layout or reuse the 06-18 Desktop mockup) — user's call.
- Uncommitted (await go-ahead): `plot_tarp_threeway.py` (Illustris);
  `plot_desi_wedge_flowjax.py` + `plot_cluster_recovery_bars.py` (GraphWeb_DESI).
- Refs: SI run dir figures + canonical `figures/desi_wedge_flowjax_linear_si/`.

### 2026-06-21 — [code] Slide-figure batch: 4 of 6 "STILL NEEDED" plots done into the SI folder
- What: Actioned the STILL-NEEDED list from the plot→slide entry below. All new/updated
  figures are themed (`shared.plot_style`) and live in the SI run dir
  `/pscratch/.../flowjax_inference_outputs/desi_wedge_flowjax_linear_si/` (also synced to
  canonical `figures/desi_wedge_flowjax_linear_si/`). DONE:
  * Slide 8 — `class_fractions_comparison.png` REGENERATED without the "Regression DESI"
    bars (added `--no-regression` + `--only-class-fractions` to `plot_desi_wedge_flowjax.py`;
    generic bar spacing). Now 3 shades: Abacus truth / Abacus NPE / NPE DESI. Cluster group
    reads 0.06 / 0.05 / 0.05 → on-message "transfer lands near truth incl. rare clusters".
  * Slide 9 — `cluster_recovery_bars.png` NEW (`plot_cluster_recovery_bars.py`): reads the
    four DESI variant summaries live → cluster fraction 0.027→0.034→0.040→0.046 vs Abacus
    truth 0.058, annotated "60% of the gap closed". Gold bars, dashed truth line.
  * Slide 9 — `edge_scale.png` copied in from the baseline linear run (graph-level, identical
    across runs; already themed).
  * Slide 7 — Abacus calibration copied in as `abacus_tarp_coverage.png` +
    `abacus_sbc_calibration.png` (the themed 06-21 SI-run plots; provenance-renamed).
- NOT done (need heavier work, flagged for user decision):
  * Slide 6 three-way TARP (softplus/linear/raw overlay): the per-run npz saves only class
    probs (no posterior samples / no true θ), and `plot_tarp_coverage` computes ecp/alpha
    inline without saving them → a single overlay needs a GPU re-eval of all 3 models.
    User pre-authorised the table fallback; GPU overlay on request.
  * Slide 3 idealised fake-data skewer: generative; `build_skewer_animation.py` is the
    real-data version (panel layout = galaxy strip + width ribbon + class strip + 3 λ
    posterior KDEs + class-prob bar). A synthetic twin can be built to match, or reuse the
    06-18 Desktop mockup — user's call.
- Code: `plot_desi_wedge_flowjax.py` (2 new flags) + new `plot_cluster_recovery_bars.py`
  are UNCOMMITTED in GraphWeb_DESI pending user go-ahead.
- Refs: SI run dir figures; `GraphWeb_DESI/workflows/sbi_inference/{plot_desi_wedge_flowjax,
  plot_cluster_recovery_bars}.py`; this batch maps to the plot→slide entry directly below.

### 2026-06-21 — [science] Plot→slide mapping decided; graph-construction question closed
- What: Audited the conference folder `~/Developer/Working Files/SBI-Galev-2026/`
  (deck .key + 17 figures/animations) and assigned heroes per slide; full mapping
  appended to the script (§PLOT → SLIDE MAPPING v2.1). Curated, not exhaustive
  (we deliberately do not use every plot).
- Decisions:
  * Slide 8 hero = `class_fractions_comparison.png`. KEY REALISATION: this is the
    SI-corrected production run — NPE-DESI cluster fraction is already 0.05 (truth
    0.06), all four classes land near truth. So the transfer result and the cluster
    recovery are the SAME figure; slide 8 = "transfer works incl. rare clusters,"
    and slide 9 becomes the under-the-hood "this didn't come for free" methods
    slide. Reinforces the demotion of cluster-fraction from headline to support.
    ACTION: regenerate this plot WITHOUT the "Regression DESI" bars (off-message
    for an SBI room; clutters the legend) — keep 3 shades (Abacus truth/Abacus
    NPE/NPE DESI).
  * Slide 10 = `closure/closure_categorical.png` (hero: f_quench/colour/sSFR rise
    void→cluster) + `closure_mass_control.png` (survives at fixed M*) + the real
    skewer animation. Strongest part of the deck: validation = first science.
  * Slide 10 animation: `skewer_posterior_animation_real.html` → record to mp4/gif
    and embed in Keynote (no live HTML in a 12-min talk). Bookend to slide 3.
  * Backup/Q&A only: eigenvalue_corner, closure_continuous_{trace,p_cluster},
    posterior_width_sky_map, class_sky_map, width_vs_boundary (counterintuitive:
    width tracks tail extremity not class ambiguity — needs 30s, keep off main deck),
    embeddings (umap/pca/3d), lambda_th_sweep, posterior_width_3d.
- STILL NEEDED (not in folder — sync from NERSC canonical or generate): (a) **Slide 7
  TARP coverage plot + SBC rank histogram (Abacus)** — CRITICAL, the calibration
  headline currently has no figure; sync from the SI run via
  `scripts/sync_figures_to_canonical.sh`. (b) Slide 6 three-way TARP curves
  (softplus/linear/raw) to make "raw wins NLL, fails calibration" visual (table
  alone is acceptable fallback). (c) Slide 9 `edge_scale.png` + cluster-fraction
  recovery-sequence bars (else slide 9 is text-only). (d) Slide 3 idealised
  fake-data skewer — generate or reuse the 06-18 Desktop mockup; must match the
  real-skewer panel layout for the bookend.
- CLOSED: graph construction is **Delaunay on BOTH** the Abacus training wedge and
  the DESI inference wedge (user-confirmed; alpha-complex was tried earlier and
  worked worse, abandoned). => NO train/inference graph mismatch; removed that as a
  slide-9 open question. The 2026-06-21 v2 entry's open-item (1) is hereby resolved.
  Pipeline slide states "Delaunay" cleanly; slide-9 gap is purely the n(z)/edge-scale
  story.
- Closure test: confirmed done and strong per the 2026-06-21 [code] entry (99.7%
  TARGETID→FastSpecFit join; every trend monotonic void→cluster; survives mass
  control). Skewer animations exist; user may iterate.
- Deck file: still a copy of the old FLATS talk inside the SBI-Galev-2026 folder;
  it's the working base for building the new slides.
- Next: sync the four missing-figure sets from NERSC; generate the idealised skewer;
  regen class-fractions without the regression bars; then build slides on the FLATS
  copy. Timer dry-run after.
- Refs: `~/Developer/Working Files/SBI-Galev-2026/` (deck + figures);
  `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (§PLOT→SLIDE v2.1);
  2026-06-21 [code] closure-test entry; 2026-06-19/06-20 [code] diagnostic/SI runs.

### 2026-06-21 — [code] Property–environment closure test runs on DESI LOA wedge (slide 10)
- What: Built the TARGETID→FastSpecFit join + closure-test plots for slide 10's
  truth-free validation (validation item (4) from 2026-06-18). Two new scripts in
  `GraphWeb_DESI/workflows/sbi_inference/`:
  `build_desi_wedge_property_join.py` (recovers TARGETID from preds
  `global_node_id` → source BGS catalog row index; de-dups the 1,252
  hemisphere-split duplicate targets by averaging their posteriors; joins loa
  FastSpecFit SPECPHOT by TARGETID → LOGMSTAR/SFR/Dn4000/ABSMAG g,r; writes
  `desi_wedge_env_props.parquet` + `join_report.json`) and
  `plot_property_environment_closure.py` (themed categorical + continuous +
  mass-control figures).
- Join quality: loa-vs-loa (same release — no Y1/Y3 mismatch), FastSpecFit dir
  `/global/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs/` (already the
  `config_paths.DESI_FASTSPEC_CATALOGS_DIR` default). **99.7% coverage**
  (111,171/111,503 unique targets); RA/Dec cross-check 0.027″ → join verified.
  Used the SI production run preds (`desi_wedge_flowjax_linear_si`).
- RESULT — every known trend recovered, monotonic void→wall→filament→cluster:
  f_quench 0.743→0.777→0.815→0.835; median (g−r)₀.₁ 0.782→0.827→0.870→0.908;
  median log sSFR −11.51→−11.63→−11.77→−11.96. Continuous E[Σλ] (tidal-tensor
  trace ∝ density): Spearman ρ = +0.076 (quench), +0.138 (g−r), −0.090 (sSFR),
  all correct sign at N=111k. **Mass control: trend SURVIVES at fixed M*** —
  within every logM* tertile f_quench rises void→cluster (high-mass tertile
  0.855→0.882→0.906→0.924), so it is not merely the mass–environment relation
  (median logM* itself rises only mildly, 10.60→10.71). Overall f_quench 0.78
  (massive BGS-bright sample; the TREND is the result).
- Caveats (carry to slide): inferred environment derives from galaxy positions
  so a density→quenching trend is partly expected — the point is the inferred
  T-web environment carries real physical signal AND survives mass control; loa
  fibre incompleteness is mildly env-dependent (biases absolute fractions, not
  the trend); quenched ≡ log sSFR < −11 (the colour panel is the independent
  colour-cut view).
- Figures (deck theme) synced to canonical
  `/pscratch/.../graphweb_desi/figures/desi_wedge_flowjax_linear_si_closure/`:
  `closure_categorical.png`, `closure_continuous_{trace,p_cluster}.png`,
  `closure_mass_control.png`. Slide 10 now has a real figure, not text-only.
- Next: drop the categorical (headline) + mass_control (defensible) figures into
  slide 10; optionally rerun on the baseline `desi_wedge_flowjax_linear` preds to
  confirm the trend is model-robust. The build script is `--run`/`--preds`-flagged
  so any variant rejoins in ~1 min.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{build_desi_wedge_property_join,
  plot_property_environment_closure}.py`; run dir
  `flowjax_inference_outputs/desi_wedge_flowjax_linear_si/`. See
  [[project-path1-wedge-npe-run]], 2026-06-18 validation-plan entry.

### 2026-06-21 — [science] Cambridge talk v2: narrative spine + factual corrections
- What: Reworked the talk around a single throughline question ("what cosmic-web
  environment is each galaxy in DESI, and how sure are we?") asked slide 1,
  answered slide 11, with a SKEWER BOOKEND (idealised fake-data skewer early as
  the discrete→continuous intuition pump; real DESI posterior skewer animation
  late as the payoff). Continuous-over-discrete justified with references: λ_th
  is arbitrary (Hahn 2007 used 0, Forero-Romero 2009 used 0.44, chosen for visual
  agreement); magnitude is the physics (tidal-torque theory couples spin to the
  tidal TENSOR — Codis 2015, Peebles 1969/Doroshkevich 1970); hard labels carry
  no uncertainty; and a continuous posterior SUBSUMES the discrete picture
  (threshold posterior samples → calibrated class probs at any λ_th with error
  bars). Back half reframed: the sim→real calibration-survival story is the
  differentiator for this SBI room, and the cluster-deficit / falsification work
  is demoted from headline to ONE supporting-evidence slide ("transfer is
  trustworthy and diagnosable"). Parameterisation slide reframed from a 3-way
  horse race to the methods-culture point "lower NLL ≠ a better posterior — why
  we calibration-test" (raw eigenvalues win NLL but fail TARP). Script v2 written
  to `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (supersedes v1).
- Corrections logged (these override earlier notes/memory):
  1. CORAL is NOT implemented (confirmed against 06-19/06-20 [code] entries — the
     adopted fix is per-graph feature normalisation, not CORAL). CORAL removed
     from the pipeline slide; appears only as a one-line future-work item.
  2. The Abacus wedge graph is built with DELAUNAY triangulation, NOT alpha
     complex. (Alpha complex was the recommendation for REAL survey data with
     gaps; the training wedge uses Delaunay.) Pipeline slide + diagram corrected.
  3. The slide-9 fix is "per-graph (domain-relative) feature normalisation"
     (normalise scale-carrying features relative to each graph's own mean so the
     GNN sees RELATIVE features and the mock-vs-DESI offset cancels) — NOT
     "scale-invariant features." Reworded everywhere to avoid confusing a physics
     audience and to describe what was actually done. (NB the 2026-06-20 [code]
     entry's `--scale-invariant-features` flag implements exactly this per-graph
     mean-relative normalisation; the talk wording is corrected, the code name
     is unchanged.)
- Emphasis added: n(z) differs mock-vs-DESI AND varies ACROSS the wedge, so the
  graph-scale shift is spatially varying — which is the physical reason a single
  global rescale fails and a per-graph relative normalisation is the right tool.
- Resolves the merge-conflict flagged earlier today: falsification walkthrough
  goes to BACKUP (aligns with the NERSC-side 12+3 entry's "appendix-only"
  conclusion), but for a story reason (supporting evidence, not a result), not
  merely time.
- References verified by web search (drop-in list in the script's §REFERENCES):
  Dressler 1980, Peng 2010, Kraljić 2018, Codis 2015, Peebles 1969/Doroshkevich
  1970; Hahn 2007, Forero-Romero 2009; Papamakarios & Murray 2016, Greenberg 2019,
  Cranmer/Brehmer/Louppe 2020, Papamakarios 2021; Talts 2018 (SBC), Lemos 2023 (TARP).
- Open items for next pass: (1) CONFIRM DESI-side graph construction — if it's
  alpha-complex while training is Delaunay, that train/inference graph mismatch is
  a second on-story contributor to the slide-9 domain gap and should be surfaced;
  (2) skewer animation + closure-test plot readiness by Thu drive slide 10's final
  form; (3) timer dry-run — slides 3/5/8/9 are the ~80–90s long ones, slide 9 the
  likely cut.
- Refs: `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (v2);
  2026-06-19 (DESI transfer + cluster-deficit diagnosis), 2026-06-20 (per-graph
  normalisation fix) [code] entries; conference site sbi-galev.github.io/2026.

### 2026-06-21 — [science] Cambridge talk: narrative finalized, format confirmed, deck strategy decided
- What: Format confirmed as **12-min talk + 3-min Q&A** (resolves the
  12+3 vs 15+5 open question from 2026-06-15). Finalized an 11-slide
  narrative built around the post-2026-06-15 results: brief T-web/cosmic-web
  motivation (1 slide) → classifier→posterior pivot with RASTI 2025 as
  single backstory slide → pipeline diagram (alpha-complex graph + GNN
  encoder + flow) → target-parameterisation result (linear increments
  Pareto-dominate softplus/raw — headline design choice) → TARP calibration
  on Abacus (credibility slide) → zero-shot DESI transfer result (class
  fractions land near truth, ordering-violation rate 0.8%→4.7% as the
  domain-shift fingerprint) → falsification walkthrough (ruled out
  under-density/FoG/extrapolation/shape, landed on graph-level edge-scale
  shift) → scale-invariant-feature fix (56% of cluster gap closed, zero
  in-domain cost, honest redistribution caveat) → truth-free validation +
  outlook (closure tests, skewer animation if ready, CORAL as next lever)
  → take-home. Full script with per-slide timing and visual notes written
  to `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md`.
- Why / decision: audience (SBI-GalEv 2026, Cambridge) is SBI-fluent but not
  T-web/DESI-fluent, so intro stays to ~3 sentences on cosmic-web physics
  and zero time explaining NPE/flows. Decided to sell the back half (slides
  7–9) as the differentiator for this specific room — a fully worked,
  falsifiable sim-to-real domain-shift diagnosis on an amortised NPE — since
  the conference's own theory stream lists domain adaptation, transfer
  learning, and calibration/convergence testing verbatim. Classification
  accuracy demoted to backstory only, consistent with the 2026-06-15
  framing decision but now made concrete in slide-by-slide form.
- Decision on deck file: do NOT edit `FLATs.key` in place. It's the
  classification-era (RASTI 2025) poster talk — visual theme (true-black,
  IBM Plex Sans, COSMIC_WEB_COLORS, FLATS accent palette) and several
  layout templates (title slide, T-web explainer, pipeline-stage slide) are
  worth inheriting, but essentially all result slides are from the retired
  4-class/partitioned-FlowJAX path and need rebuilding, not reuse. Plan:
  duplicate `FLATs.key` to a new file, strip outdated result slides, rebuild
  from the script above. Noted blocker: `FLATs.key` is 199MB and timed out
  via keynote-mcp AppleScript (`open_presentation`) — likely has heavy
  embedded assets from earlier iterations; worth auditing/trimming embeds
  before the new file becomes the working copy, to keep keynote-mcp
  reliable for the rest of the week.
- Next: (1) confirm whether the skewer animation (open thread since
  2026-06-18) lands before Thursday — it's the strongest single visual if
  ready, otherwise slide 10 stays closure-test text only; (2) duplicate
  FLATs.key → new working file, audit embedded asset size; (3) build slides
  from the script, regenerating figures already produced by the NERSC
  pipeline (TARP, class-fraction bars, edge_scale.png, lambda_th_sweep.png,
  fog_los_alignment.png) into the finalized PLOT_STYLE_GUIDE theme; (4)
  dry-run with a timer — slide 8 (falsification walkthrough) is the densest
  for its 90s budget and is the most likely cut candidate.
- **CONFLICTS WITH NEXT ENTRY:** this plan puts the falsification
  walkthrough on main-deck slide 8; the next entry (independent NERSC-side
  session) concludes it should be appendix-only. Unresolved — settle before
  building that slide.
- Refs: `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md`,
  `FLATs.key`, conference site (sbi-galev.github.io/2026), 2026-06-15 talk
  framing entry, 2026-06-18 validation plan entry, 2026-06-19/20 wedge-NPE
  + cluster-suppression + scale-invariant-fix entries.

### 2026-06-21 — [science] Cambridge SBI-GalEv talk slot confirmed: 12 min + 3 Q&A
- What: Organisers confirmed the talk slot is 12 minutes + 3 minutes for questions
  (resolves the open "12+3 vs 15+5" question from the 2026-06-15 framing entry).
- Why / decision: Tighter than the 15+5 option assumed in earlier figure planning —
  trim to the minimum figure set that carries the SBI-first narrative (calibration
  headline, DESI transfer result, SI fix) rather than including secondary
  diagnostics (FoG/shape/training-coverage falsification plots are appendix-only,
  not main-deck).
- Refs: FLATS deck; see 2026-06-15 "Cambridge SBI-GalEv talk" entry.

### 2026-06-21 — [code] Plot visual identity + centralised figure output fixed
- What: `workflows/sbi/plot_flowjax_posteriors.py` (Abacus-side TARP/calibration/
  posterior/class-fraction plots) had zero theming — plain matplotlib defaults,
  inconsistent with the GraphWeb_DESI true-black/IBM Plex Sans theme used on all
  DESI-side figures. Wired in `shared.plot_style.apply_style()` +
  `ACCENT_COLORS`/`COSMIC_WEB_COLORS` (the module was already mirrored into
  `TNG/Illustris/shared/`, just never imported by this script). Also added
  `scripts/sync_figures_to_canonical.sh` in both repos: rsyncs a run's finished
  figures into the existing-but-underused `CANONICAL_FIGURE_ROOT` /
  `GRAPHWEB_CANONICAL_FIGURE_DIR` (`/pscratch/.../{tng_illustris,graphweb_desi}/figures/`)
  under a run-named subfolder, so figures are centralised for picking per-conference
  without disturbing per-run experiment directories.
- Why / decision: per-run output dirs are right for active experimentation (compare
  testA/testB/Bcorrected/SI side by side) but meant the canonical figure roots,
  despite already being the documented default in `plot_flowjax_posteriors.py`,
  were being overridden by every launcher and sat empty/stale in practice.
- Next: regenerate `path1_wedge_flowjax_3d_Bcorrected_linear_si/plots/` with the
  fixed theme and sync both Abacus- and DESI-side SI figures into the canonical
  dirs (in progress).
- Refs: `shared/plot_style.py`, `scripts/sync_figures_to_canonical.sh` (both repos),
  `workflows/sbi/plot_flowjax_posteriors.py`.

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
