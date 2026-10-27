# FMSU Tokyo: NERSC figure and media production brief

Draft specification, 10 October 2026. Companion: [timed talk and 37-slide audit](fmsu_tokyo_talk_plan_20261010.md).

This is a **handoff specification, not a submitted compute request**. No NERSC task has been messaged or launched by the presentation review. Produce the static evidence first. Movies are optional and must not delay the main deck.

## Scope and authority

Read the latest `SCIENCE_LOG.md` first. Then consult the experiment/product receipts below. Use Graphify before source exploration, `cosmic_env` for normal scientific work and an allocated compute node for data-heavy work. Preserve the existing resource approvals and exposure ledger; this brief does not authorize new fits, blind-phase opening, target changes, posterior correction or additional campaign budgets. Graph-metric recomputation is unnecessary for this talk.

Prefer **re-rendering already released diagnostic arrays, saved inference outputs and existing figure data**. For the original blind result, use the existing authorized evaluation summaries/curves, not a fresh exploratory opening of ph001 truth. If an asset cannot be built from an allowed source, mark it blocked and report the exact missing artifact; do not silently substitute an older model.

The latest logged full-cap campaign is still generating its matched baseline. Do not interrupt it, reuse its allocations without permission, or open sealed ph018/ph019 for presentation visuals. Do not imply a new Gate-2 or field-calibration pass. NERSC must refresh this status when this brief is actually executed.

## Fixed visual system

Keep the existing deck's dark visual identity, but make every scientific figure projection-ready. These hex values are **a proposed standardized replacement palette**, not a claim that the old deck used these exact RGB values.

| Element | Standard |
| --- | --- |
| Canvas | 16:9; full-slide 1920×1080 px, solid black `#000000`. Plot background also black; no white rectangular journal-figure inserts. |
| Typography | DejaVu Sans for reproducible NERSC plots; matching sans-serif in Keynote. White text `#F5F5F5`, secondary `#C8CDD4`. No glow, italic headlines or text shadows. Embed fonts in vector exports. |
| Class colours | Void `#9CF6D5`; sheet/wall `#5B8FF9`; filament `#E85D9E`; knot `#F2C14E`. Keep these meanings on every figure. Use class names as well as colours. |
| Model comparisons | Direct labels and solid/dashed lines or circles/squares. CG15 solid circles, CG5 dashed squares. Mock truth white; model curve magenta unless this would conflict with class semantics. |
| Continuous scales | Fixed shared viridis for probabilities/positive uncertainty, with explicit colourbar; diverging scale centred at zero only for signed residuals. No rainbow. The dark/low endpoint must remain distinguishable from masked pixels. |
| Null/support | Unsupported values grey crosses or hatching, separate from zero probability. Do not make missing galaxies vanish. |
| Content bounds | At least 80 px side margins, about 70 px top/bottom. Leave the upper 110 px for the Keynote assertion title and a bottom strip for the selection/status line; avoid duplicated titles. |
| Font sizes at final 1080p display | Slide title 56–64 px; axis labels/panel headings 36–42 px; ticks/legend 30–34 px; short status/citation footer ≥26 px. If exporting a 13.33×7.5-inch plot at 144 dpi, these correspond to roughly 28–32/18–21/15–17/13 pt respectively. Judge the **placed output**, not rcParams in isolation. |
| Marks | Lines 3–5 px, reference lines 2–3 px, visible markers 9–13 px, uncertainty bands visible on black. Sparse major ticks; no dense minor grid. Use dashes/shapes as well as colour. |
| Layout | One dominant plot or at most two data panels. The map triptych is the only planned three-panel exception and must pass the 720p legibility check. If it fails, use two reveal states. |

Labels must distinguish Mpc from h⁻¹ Mpc, observed-coordinate display from real-space target, probability density from probability, a posterior interval from a bootstrap interval, and a posterior mean from a posterior draw. Dimensionless eigenvalues should be labelled “λa (dimensionless; normalized tidal tensor)”. Spatial displays need axes or an unambiguous scale bar/orientation key, slice thickness, and coordinate convention. Do not simply rename existing Mpc axes h⁻¹ Mpc: the Loa atlas is explicitly in Planck18 comoving Mpc.

Keep R=7 h⁻¹ Mpc, λth=0.2 and target epoch z=0.2 fixed wherever applicable. Smoothing scale is not voxel size. Field fine/coarse cell widths and target smoothing must be labelled separately.

## Deliverable contract

Suggested new output package: `fmsu_tokyo_20261010/` under an approved output root; do not overwrite existing scientific products. Relative structure:

```text
fmsu_tokyo_20261010/
  README.md
  manifest.json
  theme.json
  static/A01_...png + .pdf
  static/A02_...png + .pdf
  ...
  media/M01_...mp4 + _poster.png
  data/Axx_...csv or .npz
  scripts/...
  qa/contact_sheet.png
  qa/QA.md
```

The manifest must record asset ID, intended slide, source repo commit and dirty diff if relevant, checkpoint/model ID, input paths and SHA256, data selection, cap/phase, geometry, masks, target/coordinate definition, draw count/solver settings, plotted statistic, uncertainty construction, display thinning and seed, common colour limits, output hashes, creation time, scientific status, and allowed claim. Include actual command/environment and any unavailable dependency. Record “not applicable” for schematic fields rather than inventing provenance.

Return both a **plot-only** version for Keynote assembly and a labelled stand-alone review version. Export PNG at ≥1920-pixel full-slide width, preferably 3840 for fine spatial data, and vector PDF for lines/text; rasterize only dense point layers. Preserve units and label sizes after placement. Do not use AI-generated scientific data or image upscaling to repair illegible numerical plots.

## Essential assets

### A01 — What we observe versus what we want to infer

**Slide:** S2. **Priority:** required. **Owner:** NERSC for data render; local assembly for title/annotations.

- Two matched panels in one geometry: simulated R7 matter field and the corresponding observed-success BGS-like tracers. Select one already-authorized development region by recorded geometry, not by appearance or model error. Identify simulation/phase and observational selection.
- Same projected spatial bounds and a fixed stated slice thickness. Retain the actual tracer positions and survey support; do not use a projected slice as if it were the complete 3-D conditioning data.
- Label axes `x [h⁻¹ Mpc]`, `y [h⁻¹ Mpc]` if native Abacus coordinates really are in these units. Colourbar `δ_R` or `log10(1+δ_R)` with the transformation explicit. Galaxy panel needs an explanation of marker meaning, not a density colourbar.
- If a scientifically matched pair is not readily available, provide a labelled **schematic** locally or reuse the credited TNG image as illustration. Do not launch fresh field generation just for this opener.
- Allowed claim: galaxies are a sparse, selected tracer of the hidden field. One example does not itself demonstrate the conditional posterior or irreducible variance.

### A02 — Joint eigenvalue posteriors and derived classes

**Slide:** S3; optional movie M01. **Priority:** required. **Source:** current approved saved P12-A/halo48 draws and their model manifest.

- Two illustrative galaxies, selected by a recorded rule on observations or predicted uncertainty (e.g. predeclared concentration quantiles), without truth-driven cherry-picking. Identify whether mock or real DESI. Real examples have no truth marker.
- For each, show three marginal PDFs or an interval strip with posterior median and a stated central interval. Keep the **joint draws** for computing the four class probabilities. A pair of broad versus narrow examples need not represent a proven monotonic relation with density/support.
- Axes `λ (dimensionless)` / `Posterior density`, vertical λth=0.2; label λ1/λ2/λ3 directly with distinguishable colours/styles. Class bar spans 0–1 with category names and percentage labels large enough to read.
- Include “marginal display; probabilities computed from joint triplet draws”. KDE, if used, is a visualization only; it must not generate the class probabilities or replace saved draws. Record bandwidth, common limits and display truncation.
- Static output must explain the point without animation. Do not request a full corner plot for the main slide.

### A03 — Spatially inductive, phase-separated protocol

**Slide:** S4. **Priority:** required. **Source:** current training/OOF/split manifests and the generalisable-VAC plan.

- One simple region schematic: central scored core, surrounding input context, disjoint held-out region. A second small strip shows training phases → leave-one-phase-out summaries → frozen posterior → untouched confirmation.
- Bind the exact fold/exposure labels from manifests. Keep deployed final-fit inference distinct from training OOF summary generation. Do not suggest that a large number of patches creates new independent universes.
- Optional spatial thumbnail from the actual split geometry; otherwise a clearly labelled schematic is sufficient. No new training/evaluation.
- One takeaway: “Hold out space and phases—not random neighbouring galaxies.” The graph/grid comparison can be mentioned, but do not reproduce a misleading universal method-ranking leaderboard.

### A04 — Actual current inference pipeline

**Slide:** S5. **Priority:** required. **Owner:** local editable diagram, with NERSC metadata verification.

- Verified three-channel input is `counts`, `exposure_apodized`, `log_count_ratio` for the observer golden replay. Display human-readable names, not code tokens.
- Residual U-PATCH compressor → three base eigenvalue summaries → append z, expected-density input ntilde, cap and random-support boundary distance → conditional FMPE → ordered joint triplet draws → class probabilities.
- Put OOF fitting on a training branch; final frozen encoder/halo48 and posterior version on the deployment branch. The seven-summary contract comes from current science-log evidence; confirm it against the delivered manifest before drawing.
- No rendered galaxy-image input, density-reconstruction output or GNN message-passing block unless actually present in that model. The conditional-field project gets its own S9 diagram.
- Export editable SVG/PDF plus PNG. Arrows signify information flow, not gravitational evolution.

### A05 — Calibration with honest model lineage

**Slide:** S6; details B3. **Priority:** required. **Sources:** original `P12A_PH001_BLIND_EVALUATION.json` and its released plots/arrays; halo48 readiness and conditional/golden reports.

- Main figure: nominal credible probability on x (0–1), empirical mock coverage on y (0–1), identity line. Three eigenvalue curves only if legible. Add dependence-aware uncertainty from the registered summaries; do not fabricate bands from the rounded numbers below.
- Clearly separated caption/cards: **original P12-A, blind ph001, 4,897,905 rows, 10/10 registered gates** versus **halo48 successor, already-exposed ph006, 50,000-row diagnostic**. If both fits are plotted, use separate panels/explicit line styles and do not imply a matched-population comparison.
- Reference values for source QA: original C68 .66661/.67075/.66480 and C90 .89115/.89481/.89001. Halo48 natural-weight C68 .69469/.69870/.70975, C90 .91012/.91041/.91765. Conditional-evaluator values differ; retain their separate weighting, sampling and report rather than mixing them into one table.
- If original full curves are unavailable in released summaries, use clearly labelled 68%/90% points with the identity reference and exact receipts. Do not interpolate invented smooth curves. Use current halo48 full diagnostics only under its own label.
- Backup: conditional redshift/support strata, TARP/ranks and proper-score evidence. Original Gaussian-minus-FMPE joint-energy improvement .008999, authoritative-core bootstrap 95% interval [.008740,.009264], 8,199 blocks. Label as absolute score difference, not percent or a comparison against every possible model.
- On-screen footer: “Registered mock/HOD; finite diagnostics. Real-DESI coverage unverified.” Proper calibration alone does not establish informativeness; include the score comparison in notes even if not on the main figure.

### A06 — Current DESI map: location, probability, uncertainty

**Slide:** S7, crop for S1/S10. **Priority:** required. **Source:** original full-survey VAC and `p12a_loa_atlas_20260928/INTEGRITY_AND_VIEWS.json`.

- Reuse the already geometry-selected close-up or full-depth slab. Primary close-up has 6,044 galaxies; full-depth slice 48,798 including 36 unsupported. Keep exact geometry from its receipt; do not pick the most visually filamentary area after viewing probabilities.
- Matched positions/limits across panels: neutral galaxy markers; `P(filament | S, H_fid)`; `max P(class | S, H_fid)` or a clearly defined entropy. Prefer probability over hard-label scatter. Maximum class probability is **posterior concentration**, not accuracy.
- Copy the actual Planck18 comoving **Mpc** axes. Preserve redshift-space distortions. If using the full-depth view, mark z=.35/.45 as domain-warning guides, not physical structures.
- Same 0–1 scale across probability maps. No class balancing/confidence cuts. Unsupported points are grey crosses. Preserve number displayed, geometry and any thinning; ideally no thinning in the close-up.
- Visible status: “DESI Loa — provisional per-galaxy inferences”. Add the full-product count in the slide, not as a caption implying the crop contains millions of points.
- Do not connect independent galaxy posterior draws into smooth tidal fields. Do not call void-class galaxies a void-volume map.

### A07 — Main galaxy-property result

**Slide:** S8. **Priority:** required. **Source:** GraphWeb_DESI `docs/figures/p12a_loa_cigale_20260929/RESULTS.json`, `controlled_lowz_full_sky.png`, saved plot inputs and product receipts.

- Re-render **two** panels: probability-weighted, mass/z-standardized mean log10(sSFR/yr⁻¹) and low-sSFR fraction. Main curves CG15 only; CG5 comparison goes in backup unless both remain clear.
- X axis categories void / sheet / filament / knot, labelled “posterior-probability-weighted environment”. Y labels must make `mean(log sSFR)` distinct from `log(mean sSFR)`. Low-sSFR cut is log10(sSFR/yr⁻¹)<−11; express fraction in percent and contrast in **percentage points**.
- Selection: full available Loa footprint, 0.20≤z<0.30, 2,327,331 usable supported CG15 galaxies before common-cell eligibility. Report the actual retained common-cell sample/effective counts from the numerical receipt; do not imply all 2,327,331 contribute to every standardized cell.
- Use unchanged control cells: dz=.025, CG15 log-mass width .1 dex, 8≤logM<13, all classes effective N≥20, common mass/z reference mixture. Retain probability weighting, not argmax classes.
- Numerical callouts: knot–void **−0.186 dex**, 16–84% sky-bootstrap interval [−.189,−.184]; low-sSFR **+7.13 pp**, interval [7.03,7.22]. Preserve unrounded values in data. Do not recompute the contrast from separately rounded endpoint fractions.
- Error meaning: 64 nside=8 sky-block bootstrap replicates with reference mix fixed; **sampling only**, no SED covariance or model-transfer systematics. Show visible brief caveat and fuller notes.
- Current source image is white-background with three panels and a long footer. Regenerate to this theme; remove mass panel from main and replace “full sky” with “full available footprint”.
- Allowed claim: a modest descriptive association remains after this standardization. Not causal environmental quenching, not true-class recovery, not a calibration target for model tuning.

### A08 — Why coherent-field uncertainty needs its own validation

**Slide:** S9. **Priority:** required concept; numerical panel optional. **Sources:** Stage B 6 October, true-coarse/canvas result in SCIENCE_LOG 8 October, and current whole-field status.

- Default robust deliverable: editable schematic of local marginals versus correlated field draws; regional summary bracket spans several positions. Label as schematic. Show that preserving individual variances does not determine covariance of a regional average.
- Preferred optional numerical inset: the existing exact Gaussian local-canvas control, re-rendered from saved diagnostic arrays. Compare voxel-variance and covariance errors, identifying analytic fixture, domain/window and metric. The log records roughly 1.5–2.8% voxel-variance error but 25.9–26.3% covariance error for the correlated side-3 patch flows. Do not treat different metrics as numerically interchangeable; use separate labelled axes or paired cards.
- Alternative backup numerical figure: corrected-field Stage-B galaxy-weighted paired regional class-fraction joint90 coverage 71.25%/68.75% on ph012/ph013, with the actual rank-based construction and finite-draw convention. Never compare directly to P12's per-galaxy eigenvalue C90 as if it were the same task.
- The referenced `docs/e2e_threshold_canvas_results_20261008.md` was not present in the local checkout at review time; NERSC must locate its actual report/receipts before numerical use. SCIENCE_LOG alone supports the planning statement, not an invented figure.
- Status footer: “Development diagnostics; field posterior not qualified.” Latest available full-cap task completion is engineering evidence only. No new-generation movie is required to make this point.

## Required backup diagnostic asset

### A09 — Domain shift and property-selection limits

**Slides:** B4–B5; a simple inset may replace one A06 panel if desired.

- Panel set 1: current frozen mock versus Loa n(z) or their ratio, with the population selection and normalization explicit. Use existing audited selection counts, not arbitrary unit-area histograms that hide count mismatch. Shade z>.35 and .45–.55 only as the documented warning regions; low z is not marked “validated”.
- Panel set 2: existing CIGALE matching-completeness curves versus z by probability-weighted class. Preserve the conservative duplicate-exclusion policy. Distinguish TARGETID availability from usable uniquely matched properties.
- At z=.50–.55 roughly 46–52% of supported probability weight survives, versus about 97% at .20–.30. These are descriptive source values; render exact bins from existing data, not two invented trend points.
- Source: current VAC release/selection audit and `matching_completeness.png` plus its numerical receipt. Provide separate slides rather than four miniature panels.

## Optional media: at most one in the main talk

### M01 — Posterior skewer / threshold explanation (8–10 seconds)

- Source: A02 saved draws, current model. Fixed galaxy positions; move a clearly labelled selection cursor between three or four preselected galaxies. PDFs and class-probability bars update for that galaxy.
- Title overlay: “Different galaxies; local posteriors—not a field realization”. Do not animate independent samples as cosmological time or physical motion.
- Alternatively build two static A02 states with a simple Keynote dissolve. That is the preferred low-risk fallback.
- Current old skewer GIFs are format inspiration only. Replace tiny labels, verify units and replace the old numerical posterior source.

### M02 — Coherent field draws (optional backup, 8–12 seconds)

- Only use already-saved, source-verified coherent full-cap draws if there is a suitable reviewed visualization. Do not stitch independently sampled tiles and label the result a joint posterior.
- Fixed camera, fixed observed galaxies, fixed volume mask and colour scale, fixed density/isovalue threshold; switch draw identity with a short crossfade. Hold each draw ≥2 seconds. Label “model draw 1/…”, not “time”. Avoid morphology-changing interpolation between draws.
- Visible “Exploratory / joint calibration not established”. State density versus tidal-field quantity; no apparent precision from a beautiful surface plot. Include dimensions, coordinate origin/units, cell width and R7 smoothing separately.
- No new scientific sampling is required by this brief. If the full-cap product/source law changes during the campaign, preserve the version distinction and do not mix pair-local and full-cap draws in one ensemble.

### M03 — DESI map zoom (optional alternative to M01, 6–8 seconds)

- A deterministic zoom from the A06 footprint into the exact selected close-up. Keep scale bar, support flag and provisional label visible. Same data throughout; no changing selection during the camera move unless explicitly annotated.
- This is cartography, not posterior sampling, dynamical evolution or an accuracy demonstration.

### Video specification

- MP4/H.264, yuv420p, 1920×1080, 30 fps, no audio; maximum about 12 seconds. Embed locally in Keynote; no web link dependency. Use constant/slow motion, no spinning camera or flashing points.
- Deliver first/useful-middle/final PNG frames and a designed static poster. A blank first frame is unacceptable—the current slide 23 demonstrates why.
- Motion must not be the only way to read labels or obtain the scientific conclusion. Test start/end frames and playback on the presentation Mac. In PDF, replace video with its useful poster frame.

## Build order and acceptance

1. **Verify sources and status:** bind original blind versus halo48 versus field lineage; identify any absent numerical receipts. No plotting can repair an unverified source.
2. **Produce core evidence first:** A05 calibration, A06 DESI map, A07 property result. These determine what can actually be claimed.
3. **Build explanation:** A02 posterior example, A03 protocol, A04 pipeline, A01 opener.
4. **Complete limitations:** A08 schematic/available diagnostic and A09 backup. Include unqualified status even if no new field result exists.
5. **Optional polish:** at most one main movie, then M02 only if source-verified and useful for questions.

For every figure, the QA report must answer:

- Do plotted values match their immutable receipt? Can the exact source/model be identified?
- Are mock truth, real data, schematic and exploratory model draws unmistakable?
- Are selection/target conventions consistent across compared panels? Any difference must be explicitly labelled.
- Are uncertainty bands defined and appropriate to the dependence structure? No iid galaxy bootstrap used to imply independent universes.
- Are coordinates, units, smoothing, thresholds, axes, colourbars and status legible at **1280×720**? Test placed full-slide mockups, not isolated hi-res figures.
- Are PDF fonts embedded and labels unclipped? Do colours remain distinguishable with grayscale/colour-vision checks and different line styles?
- Are unsupported/masked areas visible? Are identical panels actually using identical IDs, limits and normalizations?
- Does every movie have a meaningful poster and reproducible frame manifest? Does the claim survive a failed playback?

Do not mark the package ready until numerical parity and visual checks are complete. Return a compact contact sheet, asset manifest, blocked items and any discrepancy with these planning values. Report newly available scientific results separately; do not silently upgrade the slide's claim.

## Source map for the NERSC agent

Paths below are verified local source pointers or remote paths **recorded by the source documents**, not a fresh remote existence check.

| Purpose | Authority / starting point |
| --- | --- |
| Latest status | Illustris `SCIENCE_LOG.md`, including appended 29 September full-CIGALE entry as well as top-of-file October updates |
| Protocol and targets | `docs/plan_generalisable_graphweb_vac.md`; historical status overridden by newest log |
| Production and provisional limits | `docs/plan_desi_p12a_vac_20260924.md`; `docs/p12a_halo48_readiness_20260925.md`; `docs/p12a_golden_conditional_results_20260925.md` |
| Original blind result | Existing `blind_predictions/ph001/evaluation/P12A_PH001_BLIND_EVALUATION.json` and associated released plotting outputs; resolve root from original receipts, do not guess |
| Map geometry/data | GraphWeb_DESI `docs/p12a_provisional_release_20260928.md`; `docs/figures/p12a_loa_atlas_20260928/INTEGRITY_AND_VIEWS.json` |
| Original VAC | `/pscratch/sd/d/dkololgi/graphweb_desi/outputs/p12a_loa_full_20260925_v1/DESI_LOA_P12A_HALO48_FULL_SURVEY_VAC.fits`; original SHA256 `cf472cb4e8fb327620629f347115ad26c55a3f985320b293e6753e07f50ebfbc` |
| CIGALE analysis | GraphWeb_DESI `docs/p12a_loa_cigale_20260929.md`; `docs/figures/p12a_loa_cigale_20260929/RESULTS.json`, `PRODUCT.json`, `JOIN.json`, `ARTIFACT_SHA256.json` |
| Enriched full VAC | `/pscratch/sd/d/dkololgi/graphweb_desi/outputs/p12a_loa_cigale_20260929_v1/DESI_LOA_P12A_CIGALE_FULL_VAC.fits`; do not use the intermediate JOIN FITS as the final product |
| Current CIGALE plot sources | GraphWeb_DESI `workflows/p12a_vac/loa_cigale_plots.py`; saved final output root `.../outputs/p12a_loa_cigale_plots_20260929_v3` |
| Field interpretation | `docs/e2e_stage_a_execution_20261006.md`, `docs/e2e_stage_b_execution_20261006.md`, `docs/e2e_stage_c_mechanism_20261006.md`, `docs/e2e_true_coarse_fraction_diagnostic_20261008.md` |
| Avoid unmatched comparison | `docs/e2e_cfm_p12a_comparison_scope_20261009.md` |
| Active whole-field work | `docs/e2e_wholefield_matched_gate2_20261009.md`, `docs/e2e_wholefield_fullcap_regeneration_20261009.md`; verify newest receipts before any status statement |

No external upload, public release, collaboration-storage transfer or NERSC scheduling action is part of this document's creation.
