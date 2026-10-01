# Loa field atlas and matched CIGALE comparison

## Scope and authorization

2026-09-30: the user approved compute, including additional compute needed for
both caps, and explicitly requested **both** representative NGC/SGC views and
a full-footprint independently tiled atlas. No training or original VAC changes.
Keep all outputs local; no upload/publication of research data.

Correction to initial orientation: the September 29 science-log entry (appended
near the end, not at the top) supersedes the older wedge-only property source.
Use the full enriched catalogue described in GraphWeb
`docs/p12a_loa_cigale_20260929.md`, not the legacy wedge cache. Its PRODUCT.json
and SHA256 are verified. All 5,436,413 VAC rows are preserved; 5,019,523 unique
consistent CIGALE matches and 4,989,187 usable supported CG15 galaxies. Ambiguous
matches remain missing. Do not resolve duplicate source rows arbitrarily.

## Registered field sampling

- Frozen seed17 EMA: coarse13,312 and fine26,624; same normalizer and NFE128.
- Full atlas: eight draws per tile, using addressed tile/draw/stage seeds.
  This is exploratory low-resolution Monte Carlo, not precise per-galaxy
  environment probabilities. Retain all eight local class assignments.
- Detailed views: sixteen draws each at fixed RA180/Dec10/z.22 NGC and
  RA0/Dec-10/z.22 SGC, snapping the center to the eight-raw-cell lattice.
  The NGC view replays the earlier strict-FP32 draws and recovers their wide
  context; compare stored float32 densities against the previous samples.
- Atlas ownership: two adjacent 16^3 fine cores, with raw stride[64,32,32].
  Assign each original VAC row to exactly one tile with half-open boundaries.
  The audit finds 1,626 NGC and 846 SGC occupied tiles. This covers all VAC
  galaxies across the footprint, not every entirely galaxy-free edge tile.
- Never average overlapping independent draws into a claimed joint field.
  The atlas is a mosaic of independently sampled local conditionals; draw
  indices across tiles do not define a physical joint full-survey realization.
  Render seams/ownership metadata and the qualification visibly.

## Numerical and observation gates

All observation inputs remain target-free; no sealed phases018/019. Raw
coordinates are recomputed from RA/Dec/z in the pinned DESI Mpc/h convention.
Reuse fixed selection and random maps, with nonperiodic missing padding.
Cache raw counts and factor4 response once per cap; local response is rebuilt
at the actual tile coordinates. Both-cap mock channel replay and direct-Loa
replay must pass before inference. Counts must conserve deposited weight.

TF32 is a numerical acceleration only, not a new model. Require original
strict-FP32 density replay, field/coarse RMS numerical errors below1% of
matched-draw spread and <1% tidal-class disagreement at lambda_th=.2 on the
fixed NGC low-z, SGC low-z and SGC high-z checks. The first existing-NGC check
gave rho error0.736% of spread and class disagreement0.073–0.079%.
The double-precision GPU FFT must agree with the unchanged CPU operator to
1e-10; rectangular odd/even and full physical cases pass at ~1e-15.

Compute the tensor from **each paired local and wide draw**, then interpolate
its six components at observed galaxy positions before eigendecomposition.
Check sorted eigenvalues, trace=density contrast, positivity and block-mass
closure. Do not classify density thresholds as tidal environments. Primary
classes count eigenvalues>.2 for the P12-A comparison; retain threshold0 as a
label-definition sensitivity. Finite-context FFT exterior assumptions remain.

## Property estimands and graphics

Use exactly the same galaxies for field/P12-A comparisons: current supported
VAC, unique accepted CIGALE GALAXY match, positive finite mass/SFR, and the
field support requirements. Field eligibility requires interpolated support
>=.5 and both owned-core mean supports>=.25; exclusions stay in the atlas and
are counted, never removed from or written back into the VAC.

Reproduce mass–sSFR distributions, raw/controlled class trends, fixed-mass
relations and matching/support completeness. Use CG15 mass bins0.1dex and
z bins.025, with common control-cell support and the same reference mixture
for both models; >=20 effective rows in every class/model. CG5 sensitivity
uses identical rows and CG15 control cells. Retain the original low-sSFR
proxy log10(sSFR/yr^-1)<-11. Main low-z panel is .20<=z<.30; full-range and
shell panels are explicitly selection-sensitive.

Global intervals, where shown, are matched sky-block bootstrap diagnostics,
not coherent field-posterior intervals. Show detailed-patch draw variation
separately. Record split-half Monte Carlo sensitivity of the eight-draw atlas.
No true-class, causal, physical-evolution or real-data calibration claims.
The known high-z mock mismatch and CIGALE duplicate/SED-selection caveats remain.

The local 3D artifact should provide cap/draw selection, galaxy and density
toggles, true Mpc/h coordinates, visible support/coverage qualifications and
reproducible display-only thinning. Detailed fields may use density surfaces;
the whole atlas may use binned density voxels for browser performance. Mark
display binning/thinning separately from full-resolution analysis.

## Execution and provenance

Preparation and bounded numerical smoke tests use shared-GPU allocation59139104.
The full run follows only after smoke/throughput gates, in frozen source under
ordinary Slurm batch, with one independent worker per GPU and durable per-tile
receipts. Resource scale is chosen from the measured end-to-end tile rate.
Restart accepts only matching source/model/configuration bindings and verified
output hashes. Merging requires exact once-only coverage of all original rows.
No scheduler COMPLETED state alone constitutes scientific qualification.

Official resource reference: https://docs.nersc.gov/jobs/policy/ and
https://docs.nersc.gov/systems/perlmutter/running-jobs/ (checked2026-09-30).
Shared QOS is used for1–2 GPUs; regular for a measured four-worker full-node run.

## Representative results and artifact QA (October 1 UTC)

Production atlas job59140129 is submitted from immutable Illustris238df8f;
currently pending scheduler priority. No complete-atlas result is claimed yet.
Both detailed boxes finished on59139104: sixteen draws each, original NGC
float32 density replay exactly equal to the earlier gallery. NGC/SGC owned-core
rows15,993/13,112; local property figure uses12,074/9,432 identical supported
CG15 galaxies within .20<=z<.30. Raw P12-A weighted mean log sSFR runs
NGC[-10.500,-10.646,-10.850,-10.905] and
SGC[-10.510,-10.618,-10.765,-10.898] across void/sheet/filament/knot.
CFM16-draw median class means are
NGC[-10.525,-10.668,-10.849,-10.914] and
SGC[-10.543,-10.633,-10.752,-10.891]. These local associations are **not**
mass/redshift controlled or global results. The displayed16–84% ranges are
within-patch conditional draw variation, not SED/transfer/systematic errors.

GraphWeb source now includes `workflows/p12a_vac/field_property_environment.py`,
`loa_field_viewer.py` with its HTML template, focused tests and browser checks.
Four estimator identity/support tests and a24,000-row synthetic full8-page
figure-pipeline smoke passed. The fixture was removed and is not evidence.
The production analysis includes full/low-z controlled relations, distributions,
fixed-mass curves, completeness, shell sensitivity, CG5 versusCG15, threshold0
sensitivity and frozen-cell split-half Monte Carlo sensitivity.

The self-contained current viewer is
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_field_atlas_20260930_v1/viewer_details_v4/Loa_3D.html`.
It contains both full-VAC galaxy overviews (100,000/cap display-hash sample) and
both16-draw detailed fields. Detail galaxies are reproducibly thinned to30,000
from310,030NGC/170,611SGC context galaxies; none are thinned for analysis.
Density display uses2x pooling to13.532Mpc/h cells and support>=.5. Explicit
masked marching-cubes surfaces replace Plotly's native isosurface path, which
silently rendered blank with masked values in the tested runtime. Surface
geometry, coordinate extents, draw selection, visibility toggles, desktop/mobile
rendering and no external requests passed Chromium checks; screenshots inspected.
Pinned scikit-image0.25.2/lazy-loader0.4 and browser-test dependencies are isolated
under this Scratch artifact, not installed into project/shared environments.
The atlas-view branch also passed with three real smoke tiles in a clearly
labelled disposable fixture; no complete-atlas claim follows from that test.

Small local results and validation receipts are retained in GraphWeb
`docs/figures/loa_field_properties_20260930/`. Large HTML/fields remain on Scratch.
Full property figures/full-atlas viewer require the production atlas completion
receipt, and a dependent frozen CPU postprocessing job. Review its numerical
results and screenshots before treating those pending products as delivered.

Postprocessing is submitted as CPU59141138, `afterok:59140129`, regular/desi,
one CPU node/32 requested CPUs/30minutes/scratch license. Frozen GraphWebe2c0495
source is under atlasroot`analysis_source_e2c0495`, with five-file SHA256 manifest
checked before analysis. Expected outputs are `property_comparison_v1/` and
`viewer_full_v1/`; dependency-held at handoff. These do not yet exist and are not
delivered results. Browser dependencies were copied from the successful compute
test into bounded atlasroot`browser_runtime/` and `playwright_runtime/`.
