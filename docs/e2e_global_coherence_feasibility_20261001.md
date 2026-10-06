# Loa seam coverage and shared-state feasibility — 2026-10-01

## Decision and scope

The visible gaps do contain galaxies, but those galaxies were not skipped by
inference. The atlas is still a collection of independent local draws, not a
survey-wide joint posterior. A two-window shared-state sampler eliminates
same-location disagreement without retraining. This is an engineering result,
not evidence that the modified sampler has the correct posterior distribution.
No retraining, production replacement, website update, VAC write, or sealed
phase access was performed in this stage.

The user approved one CPU allocation up to 30 minutes and one shared GPU up to
two hours for this first stage. Production retraining remains subject to a
representative smoke test and concrete resource estimate. Simply restarting
the completed local training objective would not impose survey-wide coherence.

## Coverage audit

Code: `workflows/sbi/e2e_loa_seam_audit.py`.
Receipt root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_global_feasibility_20261001_v1`.
Receipt: `coverage/COVERAGE.json`, CPU allocation 59157881 on nid200083.

The audit verified hashes for the plan-bound galaxy table, merged marginal
product, all 2,472 tile receipts and native per-galaxy draw files. TARGETIDs are
unique; every row has exactly one owner. All 5,436,413 galaxies have eight finite
positive density values and valid class probabilities. This is numerical
coverage, not a claim that every galaxy passes the scientific support cuts.

A seam slab is within 6.766 Mpc/h of an ownership face that has an existing
face-neighbour tile. Slabs are unioned across axes, so edge/corner galaxies are
not double-counted. This is a three-dimensional geometric definition of the
potential 13.532 Mpc/h display gap, not a pixel mask of a particular screenshot
or a statement that an isodensity surface must cross every such position.

| Region | Galaxies in seam slabs | With all eight densities and valid probabilities | Field-eligible |
| --- | ---: | ---: | ---: |
| NGC | 1,095,580 | 1,095,580 | 1,011,066 |
| SGC | 444,438 | 444,438 | 389,609 |
| Both | 1,540,018 | 1,540,018 | 1,400,675 |

The remaining 139,343 seam-slab galaxies fail existing field-eligibility cuts;
they are not rows omitted at tile boundaries. Across the full table 423,330
galaxies fail those cuts. Galaxy interpolation uses the full model context
before the atlas saves a cropped ownership region. Separately meshing pooled
owned grids leaves gaps between tile-centre samples. Thus absent rendered
surface, absent inference, and lack of joint coherence are distinct issues.

## Frozen-weight overlap experiment

Code: `workflows/sbi/e2e_global_overlap_probe.py`.
Receipts: `overlap/CONDITIONS.json` and `overlap/PROBE.json`; saved densities
alongside them. GPU allocation 59158363 on nid200320. Conditions use only real
Loa observations, not mock truth. Two adjacent windows per cap were selected
deterministically near the existing representative detail views:

- NGC_004_013_003 and NGC_005_013_003;
- SGC_002_011_003 and SGC_003_011_003.

Frozen seed17 EMA coarse13,312/fine26,624 factors, four draws per variant/cap,
NFE128 per factor/window, existing TF32 sampling setup. The raw-window offset
is 64 cells (216.512 Mpc/h), corresponding to eight wide/coarse and 32 fine
cells. The physical white-noise canvas is shared in both variants; this is
not merely using the same seed for differently positioned local arrays.

1. **Shared initial noise only:** integrate each window independently thereafter.
2. **Shared state every step:** evaluate the two frozen local velocity fields
   on crops of one latent canvas, average velocities on overlap, and perform
   each Heun update on that one canvas. Coarse and fine factors both use this
   rule. Fine block-zero projection and positive mass-preserving decoding
   remain in force. There is no final averaging of independently drawn densities.

| Cap | Initial-noise-only density overlap RMSE | RMSE / local draw SD | Shared-state RMSE |
| --- | ---: | ---: | ---: |
| NGC | 0.051954 | 0.27190 | 0 |
| SGC | 0.060081 | 0.19336 | 0 |

Density uses rho/mean-rho units. All decoded values were positive and finite;
maximum relative coarse-block mass error was below 4.4e-16. Coarse and fine
latent overlap equality also held exactly under shared-state updates. Equality
is enforced by construction; adjacent *different* physical cells are not
required to have equal density.

Each four-draw/two-window shared-state integration took about 7.58 seconds
(both factors, excluding loading and output compression). Peak allocated GPU
tensor memory was 0.742 GiB. These are small-pair inference measurements, not a
survey-wide or training resource estimate. Four draws cannot qualify posterior
coverage, covariance or tails. Only an x-adjacent pair per cap was tested; no
claim is made about full-cap junctions, exterior closure or tiling sensitivity.

The receipts record checkpoint, normalizer, input and script hashes. The
sampler explicitly records `posterior_qualified=false` and
`retraining_performed=false`.

## Why coherence is not yet a qualified posterior

Local networks see different observation windows and independently sampled
wide context in the original atlas. Their learned conditional vector fields
need not be restrictions of one globally conditioned vector field. Averaging
them defines a new probability flow. It guarantees one value per shared voxel
but does not demonstrate correct uncertainty, correlations or conditioning on
the union of observations. Likewise, interpolating a surface across a seam
would not demonstrate any of those properties.

Retraining is therefore **not necessary for algebraic agreement**, but may be
needed for a statistically defensible global model. A qualified route is either
to validate a frozen-weight coupling on exposed mocks or to train/fine-tune a
shared global coarse process and overlapping fine process with a joint target
and objective. Existing weights can be a warm start; a fresh initialization is
not intrinsically required. An overlap-consistency penalty alone is inadequate:
a constant field also agrees everywhere.

Next decision gate (not executed in this first stage):

- Construct observation/truth-aligned overlapping mock unions from the 13
  training phases; use already exposed development phases 012–017 for model
  decisions. Keep 018/019 sealed. Ensure periodic simulation reuse does not
  masquerade as independent long-range sky structure.
- Compare the existing local baseline and frozen shared-state candidate using
  identical observations and physical regions, including sparse/high-z and
  boundary selections. Vary tiling origin and include multi-axis junctions.
- Measure cross-boundary regional mass, spatial covariance/variograms and
  multivariate proper scores alongside marginal/aggregate coverage. Recompute
  the tidal field using a declared common exterior treatment. Do not accept
  visual continuity or same-voxel equality as calibration.
- If frozen coupling fails, smoke-test a globally shared training objective
  and benchmark memory, update time, data construction and evaluation before
  estimating/authorizing production training. The current fixed input-shape
  guards and positional conventions cannot simply be removed and called a
  trained global model.

Known Loa/mock selection mismatch and finite-exterior tidal approximations
remain independent qualification issues. The existing atlas and published
viewer remain unchanged and must still be described as local exploratory draws.

## Focused verification

`tests/phase4/test_e2e_global_overlap.py` checks seam membership only beside
existing neighbours, exact physical overlap addressing at both scales, and
four-cell fine-block alignment. The actual GPU probe additionally checks
positive finite decoding, block-mass conservation and overlap equality.
These tests are implementation checks, not statistical validation.
The focused test exposed an empty-cap indexing edge case in the audit helper;
explicit integer indices fix it. The helper also now fails closed on invalid
draws/probabilities. The coverage receipt binds the original executed version;
neither post-audit robustness change alters the reported both-cap counts.
All three focused tests passed on allocation 59158363 after the fix;
`git diff --check` passed. Graphify refresh could not run because the command
is unavailable; targeted source inspection was used instead.
