# Mock tests of exterior tidal boundary conditions

## Scope and status

This experiment asks how much tidal information is lost when matter outside
the observed footprint is omitted, and whether a small conditional model of
its tidal response is adequate. The four-phase baseline and compact-model
scoring and independent baseline verification are complete. The finer-grid
panel, scoring and independent verification are complete.
The coarse panel is retained as a numerical-sensitivity control. It uses true observed
density, not inferred density, so it isolates a boundary-treatment problem
from galaxy-to-density inference. It does not qualify a DESI tidal posterior.

The pre-result protocol is `configs/e2e_boundary_mock_20261002.json`. The user
approved one two-hour CPU allocation,59202754. The first complete training
phase took255.28seconds, supporting the remaining panel within that bound.
All nine experiment tests and eight existing isolated-operator tests passed.
The allocation's interactive shell auto-logged out after30m15s while a compute
step launched from another shell was active. Slurm marked the allocation
COMPLETED but terminated that step during refinement. This was not a completed
scientific workflow or a two-hour wall-time exhaustion. The user approved a
replacement up to90minutes. Allocation59204439 requests89minutes, keeping the
cumulative allocation bound below two hours. Its steps run in the allocation
shell's foreground. Refinement restarts in fresh `ph016_refined_v2`; the
interrupted directory is preserved without being treated as a valid result.
The replacement refinement completed in441.77seconds. Coarse artifact
verification passed: all64core scores independently recomputed as matrix
norms, original masks exactly replayed, common native-source bindings checked,
and eight direct cuboid sums matched FFT values within1.91e-14.

Before obtaining a refinement result, registered a numerical follow-up in
`configs/e2e_boundary_refinement_followup_20261002.json`: if the ph016 grid
change exceeds10percent of baseline truth error in either cap, repeat the
same full fitting/evaluation panel at raw-cell resolution within the remaining
approved allocation. This changes no masks, geometry, phase split, fit choices
or draw count, and preserves the coarse results as sensitivity evidence.

This gate triggered. In ph016 the change between grids has tensor RMSE0.02222
for NGC and0.02179for SGC, respectively36.13and25.56percent of baseline truth
error. Baseline-to-refined truth RMSE is0.06149to0.05504NGC and0.08525to0.08090
SGC. Thus the grid change is consequential, but is not identical to the net
error reduction. All four phases and the unchanged conditional fit have now
been repeated with3.383Mpc/h sources. `scores/VERIFIED.json` binds the
completed baseline checks and first refinement; finer-grid scores are in
`scores_refined/SCORES.json`.

`scores_refined/VERIFIED.json` passes all64independently recomputed core tensor
scores, unchanged condition masks, native source/reference bindings, periodic
mean identities and eight full-footprint direct sums. Their maximum absolute
FFT discrepancy is2.51e-14. These are implementation checks, not posterior
calibration or continuum-convergence claims.

## Finer-grid results and interpretation

The same64evaluation cores, equal-core weighting, lambda0.2 threshold and
fully supported cells give:

| Treatment | Tensor Frobenius RMSE | Class mismatch percent |
| --- | ---: | ---: |
| Isolated zero exterior | 0.07225 | 4.10 |
| Capwise strict periodic | 0.07156 | 3.83 |
| Capwise periodic plus mean | 0.07156 | 3.83 |
| Oracle constant exterior tensor | 0.06654 | 3.63 |
| Oracle constant plus linear tensor | 0.06110 | 3.21 |
| Conditional constant tensor mean | 0.07226 | 4.08 |
| Conditional constant plus linear mean | 0.07255 | 4.13 |

The compact twelve-mode oracle reduces tensor RMSE by about15percent, but
leaves substantial error, especially in boundary cores. This is an optimistic
representation result, not a predictive result. The fitted conditional means
do not recover that gain. Neither compact model is qualified for production.

The twelve-mode conditional eigenvalue coverage is81.68/85.14/81.56percent
overall against the87.69percent finite-draw reference. Boundary-core coverage
is66.32/72.54/65.98percent, whereas interior coverage is97.03/97.73/97.14percent.
Thus the overall average conceals severe spatial miscalibration: insufficient
spread or missing response modes near boundaries, and excessive spread in
the interior. This shared-covariance, coefficient-only model is inadequate
as a calibrated boundary-response posterior on this panel.

The apparent periodic advantage shrinks substantially upon refinement:
class mismatch changes from5.47versus3.90percent on the coarse grid to
4.10versus3.83percent on the fine grid. Periodic closure is a useful comparator,
but this is not evidence for adopting it as the physical exterior posterior.
Restoring its scalar mean changes little; the missing response is not fixed
by that scalar correction. No claim of continuum convergence follows from
two resolutions, and residual discretization must remain a caveat.

Next discriminating work would separate unrepresented spatial response from
coefficient prediction error, use distance-aware uncertainty, and validate
jointly coherent exterior realizations. This run does not launch those tasks.

## Coarse-grid sensitivity results

These are6.766Mpc/h source-cell results on64evaluation cores, with equal core
weight and two equally weighted phases. Class mismatch is the classification
of the reported mean tensor versus full-box truth at lambda0.2, on fully
supported cells. Oracle rows use evaluation truth and are representation
bounds, not predictions.

| Treatment | Tensor Frobenius RMSE | Class mismatch percent |
| --- | ---: | ---: |
| Isolated zero exterior | 0.07749 | 5.47 |
| Capwise strict periodic | 0.07234 | 3.90 |
| Capwise periodic plus mean | 0.07234 | 3.90 |
| Oracle constant exterior tensor | 0.07224 | 5.05 |
| Oracle constant plus linear tensor | 0.06728 | 4.65 |
| Conditional constant tensor mean | 0.07748 | 5.41 |
| Conditional constant plus linear mean | 0.07776 | 5.43 |

The twelve-mode conditional model has ascending-eigenvalue interval coverage
72.38/81.57/72.80percent against the finite-draw87.69percent reference. Coverage
is58.45/69.95/58.27percent in boundary cores, versus86.30/93.19/87.34percent in
interior cores. These provisional differences expose a calibration concern,
but cannot yet distinguish source-discretization error from exterior-model
error. Do not promote capwise periodic closure, reject all compact exterior
models, or claim calibrated marginalisation from these numbers.

Completed artifacts are the four phase `COMPLETE.json` files and
`scores/SCORES.json`, `scores/MODELS.npz`, `scores/FIT.json`. The interrupted
`ph016_refined` directory is incomplete and must not be used as a refinement
result. It is retained without overwriting its partial source fields. Baseline
direct-sum and saved-array score verification passed for both resolutions.

## Matched data and evaluation regions

Use the complete original mock random-support footprints in both NGC and SGC.
Do not replace them with a local crop around each test region. Start from the
qualified native2048 particle-count grid, apply the established R7 Gaussian
smoothing once, and use the same averaged cubic interpolation as the existing
target products. The baseline source cells are6.766Mpc/h, representing the
average of eight raw3.383Mpc/h samples. Multiply the already-smoothed density
contrast by the original fractional binary support. This is not angular
completeness weighting or a newly apodised mask.

The reference is the existing independent periodic full-box tidal tensor,
averaged onto exactly the same registered core cells. Each core has16cubed
cells and width108.256Mpc/h. Full-cap density must replay every stored core
density before scoring. The native simulation uses its actual2000Mpc/h box;
the observed cap fields sample its periodic continuation in physical survey
coordinates. Mock masks are held fixed within each phase, not replaced by
the observed Loa footprint.

Fit on all256cores per training phase ph007/ph008. Evaluate all32cores per
exposed ph016/ph017: both caps, four redshift shells from0.15to0.55, interior
and boundary strata, and both sides of each registered pair. These are only
two independent evaluation phases. Cores and cells within a phase are
correlated, and these phases are not fresh sealed confirmation. Never access
ph018/ph019 for this experiment.

Primary metrics use cells with support exactly1. Secondary metrics use
support>=0.5. In partially supported cells, the omitted fractional local
source has a trace contribution; it cannot be represented by a trace-free
harmonic exterior correction alone. Keep that secondary limitation explicit.

## Boundary treatments

The isolated treatment uses the verified exact-cuboid free-space kernel on
the complete masked source fields. Add both observed caps at each target in
their original physical coordinates. Padding contains all source-target
displacements; there are no periodic images or patch-mean subtraction.

The strict periodic comparator applies the existing rectangular spectral
operator to the target cap's complete masked source box, omitting k=0. The
other cap's isolated contribution is held fixed, so the comparison isolates
the change in the self-cap treatment. A second periodic comparator restores
the patch mean as an isotropic tensor, mean(delta)/3 on the diagonal. That
addition restores the local Poisson trace; it is not an exterior shear model
or a strictly periodic potential for a nonzero mean source.

Neither capwise periodic treatment is the simulation's actual full-box
boundary condition or a single jointly periodic whole-survey exterior. Their
periods are the registered source bounding boxes, recorded in each cap
receipt. This is a test of a common finite-box FFT closure, not a claim that
periodic cosmological simulations are physically inappropriate.

## Compact exterior response models

Represent the exterior contribution locally using Hessians of harmonic
potentials. Degree2 supplies the five independent components of a constant
symmetric trace-free tidal tensor. Adding degree3 supplies seven more modes,
allowing a spatially linear tensor with the Hessian integrability and zero
trace constraints. These bases have fixed geometric normalization.

First fit the coefficients directly to each core's missing truth response.
These oracle fits measure an optimistic representation bound. They use
evaluation truth intentionally and are NOT deployable predictions. An oracle
failure indicates insufficient basis flexibility at that core size; oracle
success does not establish that the coefficients can be inferred.

Separately fit standardized ridge regressions to training-phase coefficients,
with fixed penalty1 and an unpenalised intercept. Features are supported-core
means of the six observed tensor components, true observed density mean and
standard deviation, support fraction, mean boundary distance, and physical
core position. No phase identifier, evaluation coefficient, or evaluation
truth enters the conditional fit. Fit a Gaussian coefficient covariance from
leave-one-training-phase-out residual second moments, retaining residual bias
in that second moment. Use64draws with seed271 for each evaluated core.

This is a deliberately small conditional model of exterior tidal RESPONSE,
not a sampled full exterior density field. It conditions on summaries of
true observed density, not directly on galaxies. Independent core-local
draws do not constitute a coherent joint exterior realization across the
survey. Its performance cannot establish that every possible conditional
exterior model succeeds or fails.

The Gaussian spreads only these retained coefficients. It does not add the
spatial response left outside the harmonic basis, a density-inference error
field, or uncertainty in the fitted regression parameters. Its covariance is
shared across cores, rather than adapted to distance from the boundary.
Coverage therefore tests this specific restricted uncertainty model, not an
exact marginalisation over all possible exterior matter fields.

## Scores and numerical controls

Compute tensor Frobenius RMSE, ascending-eigenvalue RMSE, class confusion
and class changes at lambda0.2. Class0/1/2/3 means void/wall/filament/cluster,
defined by the number of eigenvalues exceeding the threshold. For conditional
draws, also compute class Brier scores and central interval coverage.
The64draw interval uses order-statistic indices3and60, so its exchangeable
reference coverage is57/65=87.6923percent, not exactly90percent.

Weight cores equally within each phase and the two evaluation phases equally.
Retain phase, cap, shell, support-stratum and boundary-distance results.
Do not attach voxel-binomial confidence intervals. Distance is measured to
non-full support, including actual holes; no tile boundaries are introduced.

Interpret distance-bin averages cautiously: equal-core weighting gives a core
with one contributing cell the same bin weight as a densely populated core.
In the farthest bin, ph017_NGC_s1_interior_00 side1 contributes one cell with
a zero-exterior class mismatch. This drives a misleading-looking upturn in
the descriptive curve, not evidence of increasing physical error far from
the boundary. Preserve all bins; use the full boundary/interior core strata
for the primary coverage visualization. The registered scores are unchanged.

Numerical checks include phase-access guards, padded extraction and exact
raw/fine core alignment, strict-periodic DC behavior, source linearity,
harmonic basis trace/rank/recovery, training-only fitting, synthetic score
controls, and the existing kernel quadrature/direct/FFT tests. Independently
recompute the saved tensor errors as full3x3matrix norms and compare original
condition masks. Check selected full-footprint FFT results by direct cuboid
summation at the same physical target points.

Repeat ph016 with3.383Mpc/h sources and average tensors back to the original
evaluation cells. This tests source discretization and mask-density averaging;
the reference smoothing and evaluation targets remain unchanged. The isolated
piecewise-constant source operator and full-box spectral reference are not
identical discretizations. Do not attribute every residual to exterior matter
unless this refinement is small relative to the measured error.

## Artifacts and reproduction

Scratch root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/boundary_mock_20261002_v1`.
Each phase writes source fields, scored-core inputs, source bindings and
completion receipts. Input science products remain read-only. Repository
implementations are `workflows/sbi/e2e_boundary_mock.py`,
`workflows/sbi/e2e_boundary_models.py`, and
`workflows/sbi/e2e_boundary_verify.py`.

Run on an approved compute node in cosmic_env, using fresh output directories:

```sh
python -m workflows.sbi.e2e_boundary_mock --phase ph007 --out <root>/ph007
python -m workflows.sbi.e2e_boundary_mock --phase ph008 --out <root>/ph008
python -m workflows.sbi.e2e_boundary_mock --phase ph016 --out <root>/ph016
python -m workflows.sbi.e2e_boundary_mock --phase ph017 --out <root>/ph017
python -m workflows.sbi.e2e_boundary_models --root <root> --out <root>/scores
python -m workflows.sbi.e2e_boundary_mock --phase ph016 --factor 1 --out <root>/ph016_refined_v2
python -m workflows.sbi.e2e_boundary_verify --root <root> --scores <root>/scores --refined <root>/ph016_refined_v2
python -m workflows.sbi.e2e_boundary_mock --phase ph007 --factor 1 --out <root>/ph007_refined_v2
python -m workflows.sbi.e2e_boundary_mock --phase ph008 --factor 1 --out <root>/ph008_refined_v2
python -m workflows.sbi.e2e_boundary_mock --phase ph017 --factor 1 --out <root>/ph017_refined_v2
python -m workflows.sbi.e2e_boundary_models --root <root> --out <root>/scores_refined --factor 1 --suffix _refined_v2
python -m workflows.sbi.e2e_boundary_verify --root <root> --scores <root>/scores_refined --refined <root>/ph016_refined_v2
```

No density-model retraining, production VAC changes or website replacement
is part of this bounded experiment.

Allocation59204439 was relinquished after55m53s, with the full scientific
pipeline completed0:0. Together with the first30m15s, actual allocated time
was86m08s. Delivery-only CLI/preview issues were repaired; final chart and
source receipts pass360/736px layout checks, legend-state interaction and
zero browser-error checks. Preview CDN dependencies were blocked, not fetched.
The repository graph refresh succeeded on the login host after the compute
node rejected its filesystem lock. No global lock behavior was changed.
