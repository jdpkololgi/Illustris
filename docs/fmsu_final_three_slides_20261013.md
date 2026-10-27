# Tokyo: three slides after the TARP calibration plot

Prepared 13 October 2026 from the live Keynote sequence through slide 17 and the local `SCIENCE_LOG.md`. Companion to `fmsu_speaker_notes_and_figure_prompts_20261013.md`. This document does not modify the Keynote file or commission NERSC jobs. No live NERSC inspection was performed.

## The ending in one sentence

**We can turn galaxy observations into probabilistic tidal environments, use those probabilities to study galaxy populations, and ultimately extend this to uncertainty in connected structures.**

Use exactly three more slides after the current slide 17. Replace the old tail of the talk; do not append these after another methods section. The existing A03/A04 training diagrams and old classification/graph slides belong in backup.

| Proposed slide | Title | Target time |
|---|---|---:|
| 18 | Probabilistic environments for DESI galaxies | 75–85 s |
| 19 | Connecting tidal environment to galaxy properties | 70–80 s |
| 20 | From a web label to environmental uncertainty | 40–45 s |

Total: about 3:05–3:30 including figure-reading pauses. Aim to finish TARP at about 7:50 and the conclusion by 11:20, leaving 40 seconds of buffer. These are rehearsal targets, not a measurement of your current delivery. The repeated slides in the first part are progressive builds, not separate one-minute topics.

## Small correction to the TARP handoff

The current slide 17 notes repeat the marginal-interval explanation from slide 16. For the TARP panel, distinguish the joint diagnostic:

> We also check the joint distribution of all three eigenvalues, rather than only their individual intervals. This TARP curve is close to the expected diagonal on this mock sample. That supports the joint-posterior behaviour tested here, but it isn't a calibration guarantee for real DESI. So the next question is what this inference gives us when we apply it to the survey.

Keep the current provenance visible: halo48, ph006, 50,000 galaxies, already exposed. Do not describe this as a fresh blind test. The shaded bootstrap maximum-deviation envelope is not a pointwise confidence band. Avoid saying this plot alone proves every conditional posterior is correct.

## Slide 18 — Probabilistic environments for DESI galaxies

### What goes on the slide

**Title:** Probabilistic environments for DESI galaxies

**Main visual:** Two matched views of one DESI region, preferably within `0.20 ≤ z < 0.30`:

- Left: **Observed galaxies** — neutral points.
- Right: **Inferred filament probability** — the same points, coloured continuously by `P(filament)`.

One colour bar, fixed from 0 to 1. Identical coordinates, selection, aspect ratio and point sizes. Use the previously verified region if suitable; no need to show the entire catalogue at once.

**Takeaway:** Each galaxy carries probabilities, not just a label.

**Compact status line:** Provisional DESI application · calibration tested in mocks, not established on data.

Do not add a second equation: slide 15 has already explained how joint eigenvalue draws become class probabilities. Do not include four competing class maps, a separate entropy map, or a rotating 3D movie here. A static matched pair is easier to read in the time available.

### Speaker notes — copy into Keynote

So what does this look like when we apply it to DESI?

On the left are the observed galaxies in one region of the survey. On the right are exactly the same galaxies, but now coloured by their inferred probability of being in a filament, according to our tidal definition.

The useful change is that we haven't forced each object into one class. We retain the distribution over eigenvalues, and the probabilities that follow from it. So a galaxy near an environmental boundary can remain uncertain in the catalogue, rather than acquiring an artificially definite label.

There is an important distinction here. The calibration tests I just showed were in simulations. This is an application to the real survey, where differences between the mock and the data remain a source of uncertainty. These are therefore provisional, model-dependent environmental estimates.

But they already let us ask a scientific question: how do the properties of galaxies differ across these inferred environments?

### Figure-generation prompt — A06 closing version

```text
Produce a reproducible, data-derived figure for slide 18 of the FMSU Tokyo talk. Read SCIENCE_LOG.md first and use the current frozen P12-A Loa per-galaxy posterior catalogue and its verified plotting/provenance receipts. Do not use an older TNG-transfer product or a conditional-field-model product as if it were this catalogue. Do not run inference or change catalogue contents.

Create two side-by-side views of exactly the same supported DESI galaxies in a clearly documented region within 0.20 <= z < 0.30. Prefer the existing verified atlas region if readable. Left panel: neutral grey galaxy markers, titled “Observed galaxies”. Right panel: identical markers coloured by the saved P(filament), titled “Inferred filament probability”. Use a perceptually uniform sequential colour map, visible against black, with a single fixed 0–1 colour bar labelled “P(filament)”. Do not colour by a hard maximum-probability class. Do not invent individual probabilities.

Use one documented projection. If using the existing Planck18 redshift-derived comoving coordinates, label axes in Mpc, not h^-1 Mpc, and explicitly identify observed redshift-space coordinates. For a sky projection use RA and Dec in degrees instead; do not mix conventions. Show the redshift and slice/sky selection. Keep equal aspect ratio where geometrically appropriate. Fix axis limits, marker sizes and any downsampling identically across panels. If downsampling is needed, use a fixed seed independent of inferred environment; record the displayed and eligible counts. Do not interpolate a continuous matter field, join galaxies into filament skeletons, or imply this is a spatially coherent posterior draw.

Exclude unsupported rows from the coloured sample; never assign them probability zero. Record their number in the manifest. If they appear in the neutral panel, explicitly distinguish them; preferably use exactly the same supported rows in both panels.

Match the current deck: black background, white/light-grey axes, matching sans-serif font, restrained blue/mint accents. Preserve the deck's categorical environment colours for other assets, but use a sequential scale for this probability. Reserve space for a separate Keynote title and takeaway. At 1920x1080 final slide size, aim for axis/colour-bar labels >=28 px and ticks >=24 px; verify at 1280x720. No unreadable audit text inside the plotting area.

Deliver A06_DESI_probability_closing.png at >=3840 px width, vector PDF where practical, plotting script, and a JSON/text manifest with catalogue path/hash, model ID, probability field, cuts, coordinate convention, displayed/eligible/unsupported counts and seed. Include a ready-to-paste caption identifying the product as provisional, with real-data calibration unverified. Use actual catalogue data only; no generative-image model for the quantitative figure.
```

### Scientific guardrails

- The known mock–survey mismatch is especially important at higher redshift. A low-redshift illustration is not certification of low-redshift calibration.
- Spatial patterns are an inference product, not independent evidence that the inferred field is correct.
- The model produces local host-linked tidal-target estimates; do not present this as a reconstruction of the matter field at every observed coordinate.
- The catalogue has 5,436,413 rows, of which 5,404,568 are supported in the recorded product. These counts are optional spoken context, not essential slide text; regenerate only after checking the exact source receipt.

## Slide 19 — Connecting tidal environment to galaxy properties

### What goes on the slide

**Title:** Connecting tidal environment to galaxy properties

**Main visual:** One four-category point-and-interval plot of the existing probability-weighted, stellar-mass/redshift-standardized **mean log specific star-formation rate**. Use the saved common-cell estimator, not a new ad hoc mass cut or hard-class assignment.

- Horizontal axis: **Inferred tidal environment** — Void, Wall, Filament, Knot.
- Vertical axis: **Standardized mean log₁₀(sSFR / yr⁻¹)**.
- Small subtitle: **DESI + CIGALE · 0.20 ≤ z < 0.30**.
- One highlighted contrast: **Knot − void: −0.19 dex**.

**Takeaway:** An environmental association remains after mass/redshift standardization.

**Compact qualification:** Probability-weighted association · not a causal or true-class measurement.

Label intervals explicitly: **16–84% spatial-bootstrap intervals; systematics not included**. Put estimator details and the CG15 definition in notes/provenance, not a dense on-slide paragraph. Do not plot all diagnostics or both CIGALE variants.

### Speaker notes — copy into Keynote

One example is the connection between environment and star formation.

Here we've joined the environmental catalogue to CIGALE estimates of stellar mass and star-formation rate. We focus on this lower-redshift interval, and compare the environments using a common stellar-mass and redshift distribution.

We also retain the environmental probabilities: a galaxy contributes according to its probability of belonging to each class, rather than being assigned one hard label.

In this comparison, the standardized mean log specific star-formation rate is about 0.19 dex lower for the inferred knot population than for the inferred void population. So there is a descriptive environmental association even after this mass and redshift standardization.

That isn't a claim that environment caused the difference, or that the galaxy-property trend validates the inferred environments. And the small statistical intervals don't include the uncertainties in the SED modelling or in transferring from mocks to DESI.

The point is that we can now carry probabilistic environmental information into a galaxy-population analysis, instead of beginning with a rigid classification.

### Figure-generation prompt — A07 closing version

```text
Produce one slide-readable summary plot of the current P12-A Loa/CIGALE result, following SCIENCE_LOG.md and the existing full-footprint controlled-analysis receipts. This is a visualization of an established result, not permission to change the estimator or run new inference.

Locate and inspect the verified source for docs/figures/p12a_loa_cigale_20260929/RESULTS.json and controlled_lowz_full_sky.png in GraphWeb_DESI, or the provenance-linked canonical successor. Verify paths rather than assuming they exist on the present host. Use the supported, uniquely matched CG15 sample at 0.20 <= z < 0.30, with the existing common-cell stellar-mass/redshift standardization and probability weights. Preserve the exact common-cell eligibility and reference weighting. Do not substitute hard-class means, compute only a broad mass cut, or invent the intermediate wall/filament points from endpoint contrasts.

Create a single point-and-interval chart: x categories Void, Wall, Filament, Knot; y “Standardized mean log10(sSFR / yr^-1)”. Use the saved category means and corresponding 16–84% spatial-bootstrap intervals. Do not use the endpoint-contrast interval as the error bar on either category. If the saved per-category statistics are missing, recover them through the existing audited analysis workflow without changing its definition; if that cannot be done, report the missing evidence rather than inventing values. An honest fallback is a single knot-minus-void contrast plot, with a zero reference line and the saved contrast interval.

Highlight the verified knot-minus-void contrast: -0.186 dex, rounded on the slide to -0.19 dex. Its recorded 16–84% spatial-bootstrap interval is [-0.189, -0.184] dex. Check these values against the source before plotting; flag any discrepancy rather than silently updating the claim. Do not add a significance star or convert this sampling interval into a total uncertainty. State “16–84% spatial bootstrap; systematics not included”. If intervals are smaller than marker sizes, state that in the caption; do not enlarge them artificially.

Use the full available footprint in the specified redshift slice, not the older small-wedge result. The recorded 2,327,331 usable rows are before common-cell eligibility; report the actual contributing count and effective-weight information from the estimator, not that number as though every row contributes. Record matching exclusions, support cuts, CG15 definition, common-cell count, bootstrap protocol and analysis-source hashes in a manifest. Include the absence of propagated SED/model-transfer systematics in the caption.

Match the existing black slide background, sans-serif typography and exact established four-class colours. Use white/light-grey axes, large labels, generous spacing, no grid clutter. Do not connect the categories with a fitted line or imply an equally spaced continuous environmental coordinate. Keep only this one observable; no mass, SFR, low-sSFR-fraction or second-CIGALE panels. Label the result as a descriptive probability-weighted association, not environmental validation or causal quenching.

Deliver A07_CIGALE_controlled_closing.png at >=3840 px width, vector PDF, plotting script, machine-readable plotted statistics and provenance manifest. Leave the overall slide title/takeaway outside the asset so they remain editable in Keynote. Verify label readability at 1280x720. Generate the chart programmatically from actual results, not with an image generator.
```

### Scientific guardrails

- Probability weighting is the specified descriptive estimator; it is not automatically an unbiased estimate of the true environmental-class population means.
- Mass/redshift standardization does not control all confounding, selection or measurement error.
- “Mean log sSFR” is not the same as “log mean sSFR.” Preserve the plotted estimand.
- The recorded alternative CG5 analysis gives a similar endpoint contrast, but that belongs in questions/backup, not the main slide.
- Use the actual CIGALE catalogue/paper credit from the input provenance. Do not invent an author-year citation from a filename. Keep the detailed receipt in notes.

## Slide 20 — From a web label to environmental uncertainty

### What goes on the slide

**Title:** From a web label to environmental uncertainty

Three short phrases, arranged along a simple left-to-right visual sequence:

**Define the tidal environment → Infer its uncertainty → Use it in galaxy science**

Under the middle phrase, optionally reuse a small crop of the genuine posterior example from slide 15. Under the last phrase, reuse a small DESI map crop. Do not introduce a new quantitative result.

**Main closing line:** Not just “which environment?” — “which environments are supported?”

**Small outlook line:** Next: joint field posteriors for connected structures.

Put your name/contact or an existing project link in a restrained footer. Do not invent a public VAC download link or claim a release. Leave this slide on screen for questions rather than following it with a separate “Thank you” slide.

### Speaker notes — copy into Keynote

So the main message is that environmental inference is more than assigning a cosmic-web label.

We start with a physical definition of the tidal environment. We then infer which eigenvalues, and therefore which environments, are supported by the galaxy information under our model. And we can use those probabilities in galaxy-population studies.

The next step is to move from uncertainty at individual galaxies to uncertainty in a connected field. That requires a spatial joint posterior; separate local distributions don't give us that by themselves.

For now, the question we're answering is not simply, “Which environment is this galaxy in?” It's, “Which environments remain plausible, given what we observe?”

Thank you.

### Figure/layout prompt — optional, no new simulation required

```text
Create an editable SVG closing schematic for the FMSU Tokyo talk, matching the existing black-background Keynote slides. Use three widely spaced groups connected by thin light-grey arrows, with labels “Define the tidal environment”, “Infer its uncertainty”, and “Use it in galaxy science”. Use white text with restrained mint/blue accents and the deck's matching sans-serif font. The arrows mean inference workflow, not temporal gravitational evolution.

If verified assets are supplied, use one small genuine posterior crop and one genuine DESI probability-map crop as visual anchors. Preserve their colours and scientific content. Otherwise use simple vector symbols; do not fabricate a posterior curve, survey map, filament network, or result. Keep any conceptual icons clearly schematic.

Reserve a broad bottom region for an editable Keynote takeaway: “Not just ‘which environment?’ — ‘which environments are supported?’”. Add only a small, visually secondary outlook label: “Next: joint field posteriors for connected structures”. This is future work, not an assertion that the current local model yields joint field samples. Do not show apparently coherent synthetic field realizations as evidence.

Deliver SVG and PNG. Keep the slide title, takeaway and contact footer as native Keynote text where possible. Use no equations, architecture diagram, comparison with other speakers, or additional performance numbers. Check legibility at 1280x720.
```

The schematic is optional: a native Keynote three-part layout is sufficient. Prioritize generating A06 and A07; no new animation/video is needed for this ending.

## If rehearsal runs long

Keep all three slides, but use this approximately 90-second script:

**18:** “Here are the same DESI galaxies, first as observations and then coloured by their inferred filament probability. The product retains ambiguity rather than forcing a label. These are provisional model-dependent estimates: mock calibration is not a guarantee of real-data calibration.”

**19:** “We can use these probabilities to compare galaxy properties. With CIGALE, in this redshift interval and after stellar-mass/redshift standardization, the inferred knot population has a mean log specific star-formation rate about 0.19 dex below the void population. This is a descriptive association, not causal evidence or validation of the inferred environments.”

**20:** “The shift is from a label to a distribution over physically defined environments. That enables probabilistic galaxy science. Joint field inference is the next step for connected structures. The question is: which environments remain plausible, given what we observe?”

## Evidence and handoff checklist

- Current live deck: slides 15–17 already establish posterior interpretation and mock calibration; do not repeat those explanations at length.
- Local `SCIENCE_LOG.md`: 2026-09-29 Loa/CIGALE join and controlled full-footprint result, including -0.186 dex contrast and its limitations; provisional Loa catalogue and support counts in September deployment entries.
- Use `docs/fmsu_tokyo_nersc_assets_20261010.md` for the earlier A06/A07 asset context; these prompts specify the simplified closing versions.
- Scientific status is based on the local log, not a fresh remote audit. NERSC-side generation must resolve current receipts and stop on conflicting provenance.
- Before delivery: verify data/model identities, confirm probability sums/support masks, inspect both figures at slide size, and rehearse from TARP to the final sentence with a timer.
- Two required regenerated figures: matched DESI probability map and controlled CIGALE summary. One optional editable closing schematic. No new training, inference, posterior samples, or field-model claims are required.
