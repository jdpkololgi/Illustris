# E2E physical-error attribution — 2026-09-08

## Frozen diagnostic specification, before numerical execution

The user requested scientific error-budget tests before deciding whether to
regenerate separate E2E density/tensor targets. This authorizes the bounded CPU
audit, not new fits, alterations to D2/P12-A, or a science-training release.

Configuration: `configs/e2e_field_error_budget_v1.json`.
Implementation: `workflows/sbi/e2e_field_error_budget.py`.
Use all 96 previously response-selected training anchors in ph000/ph002/ph003;
do not read selection/confirmation or ph001/ph006 payloads. All three phases
have historical exposure. Parents/caps/voxels are correlated; report per-phase
and shell/support results descriptively, without voxel-IID significance tests.

Construct a new full-periodic-box spectral reference from the same hashed
2048-cubed TSC counts and exactly one Gaussian R7 smoothing. Reuse the native
count normalization arithmetic. This reference eliminates derivative truncation,
not particle subsampling, TSC assignment, finite simulation resolution or model
error. It is not an independent cosmological simulation.

Comparisons, frozen before reading their outcomes:

1. Native second-order versus spectral tensor at identical legacy lookup sites.
   Reconstruct native tensors again for numerical parity. Test fourth- and
   eighth-order centred first differences applied twice as alternative operators.
2. Spectral tensors sampled by legacy floor, nearest integer, trilinear and local
   cubic Lagrange rules. The installed Abacus TSC kernel deposits around integer
   lattice sites; floor is not generally nearest-site sampling. Preserve the
   existing observer-to-box mapping. Cubic versus linear is a convergence
   diagnostic, not proof of continuum interpolation accuracy.
3. Periodic parent solves from clean Gaussian density at 64/96/128 cells, compared
   at the same central 32-cubed region. Repeat floor versus linear sampling.
   Residuals combine exterior tides with observer-grid discretization; parent
   enlargement alone does not mathematically separate these effects.
4. Existing trace-target parent solves at 64/96 cells versus native tensor truth.
   At 96 cells, decompose the total tensor residual exactly into propagated
   scalar-filter error, parent/resampling error and spectral-minus-native tensor.
   Save their Gram matrix: these correlated errors must not be summed in
   quadrature. Include isotropic-DC removal and an oracle constant-tensor
   residual diagnostic; neither is a proposed deployable correction.

For the complete science region and its separately observed subset, measure
eigenvalue bias/RMS, eigengaps, tensor residuals, four-class and lambda2-threshold
disagreement. Science functionals use lambda2 > 0.2 regional filling fraction,
all supported nonperiodic pairs in 10--20/20--40/40--80 Mpc/h bins, fixed interior
terminal connectivity under six neighbours, and largest void-component fraction.
Connectivity terminals are fixed in the JSON; no favourable endpoint selection.
These are deterministic truth-functional changes, not posterior event calibration.

There is no E2E posterior uncertainty to use as a denominator yet. No arbitrary
scientific acceptance tolerance is introduced: the report quantifies error
sources and science sensitivity, while R0-PHYSICS release remains undecided.
Neither small relative RMS nor a stable average filling fraction alone licenses
connectivity claims. Report sparse support and phase ranges rather than pooling
them away. No production P12-A/D2 labels or frozen evaluations are redefined.

## Verification and execution

Five lightweight tests pass: spectral plane-wave sign/trace normalization,
native-FD parity and higher-order convergence, periodic/cubic sampling,
FFT pair counts versus brute force, and constant-tensor residual attribution.
Heavy numerical work runs in one authorized interactive CPU allocation with
foreground ownership and a durable log; no GPU or learned model is required.

All three phases completed. Allocation 58074689 on nid004205 and its numerical
step both have state COMPLETED, exit 0:0. Elapsed allocation time was 29m05s;
peak recorded RSS was 276,715,460 KiB (about 264 GiB). The allocation is released.
The NERSC allocation workflow kept the full-box work on a CPU node; D2 was not
modified, no GPU was used, and no model fits or full training-product regeneration
were started. All 34 focused E2E tests pass, including six new audit tests.

The three reference shards total 2,668,620,576 bytes under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/error_budget_20260908`.
Source/native/parent payload hashes were verified before each phase. Maximum
native tensor replay difference was 1.20e-7. Small reports, the frozen audit,
source hashes and scheduler receipt are archived in
`docs/evidence/e2e_field_v1/error_budget_20260908/`; start with `SUMMARY.json`.
The source AST/global knowledge graphs were refreshed.

## Completed findings

**Finite differences are small in ordinary RMS but not harmless for topology.
Floor sampling and finite-parent tides are larger problems in typical RMS.**
This is an error-attribution result, not a calibrated-posterior or R0-PHYSICS
release. No scientific acceptance tolerances were chosen retrospectively.

### Eigenvalue error hierarchy

The following are median per-anchor RMS divided by that anchor's reference
eigenvalue scatter, in percent. Each entry is lambda1 / lambda2 / lambda3.
The observed science mask is used; complete-core and all eight shell/support
strata are retained in the machine-readable report. These are not errors in
units of E2E posterior uncertainty and are not IID confidence intervals.

| Comparison | ph000 | ph002 | ph003 |
| --- | --- | --- | --- |
| Native FD2 versus spectral, same sites | 0.907 / 0.580 / 0.665 | 0.904 / 0.576 / 0.646 | 0.918 / 0.595 / 0.646 |
| FD4 versus spectral, same sites | 0.00820 / 0.00608 / 0.00585 | 0.00812 / 0.00609 / 0.00573 | 0.00821 / 0.00625 / 0.00571 |
| FD8 versus spectral, same sites | 4.06e-6 / 3.21e-6 / 2.80e-6 | 3.92e-6 / 3.28e-6 / 2.80e-6 | 3.99e-6 / 3.31e-6 / 2.81e-6 |
| Spectral floor versus linear sampling | 10.41 / 9.01 / 7.42 | 10.07 / 8.91 / 7.58 | 10.42 / 8.95 / 7.53 |
| Spectral nearest-round versus linear | 5.22 / 4.56 / 3.79 | 5.14 / 4.52 / 3.75 | 5.18 / 4.62 / 3.84 |
| Spectral linear versus cubic sampling | 0.391 / 0.239 / 0.279 | 0.384 / 0.240 / 0.274 | 0.386 / 0.245 / 0.266 |
| Clean linear 64-cell parent versus full box | 13.91 / 9.99 / 8.03 | 12.73 / 10.29 / 8.11 | 13.41 / 10.43 / 8.08 |
| Clean linear 96-cell parent versus full box | 5.73 / 4.37 / 3.35 | 6.44 / 4.93 / 3.82 | 6.22 / 4.78 / 3.72 |
| Clean linear 128-cell parent versus full box | 3.40 / 2.68 / 2.07 | 4.09 / 3.35 / 2.59 | 4.11 / 3.10 / 2.55 |

FD2 has absolute eigenvalue RMS of order 0.001 and changes the four-class label
at a median 0.21--0.23% of observed voxels per phase. Floor versus linear changes
about 3.46--3.52% of those labels. The FD4 improvement is roughly two orders of
magnitude; FD8 closely matches the spectral reference at this native resolution
and smoothing. These findings do not transfer automatically to coarser grids.

### Science-function changes

These are worst absolute differences across the 96 screened training anchors,
not population error estimates. Filling/void columns are **percentage points**;
pair differences are absolute values of C_m in the three registered bins.
Connectivity counts indicate anchors with at least one changed fixed-axis
terminal event, out of 96; those anchors are not independent universe draws.

| Comparison | Max filling change (pp) | Max pair change | Changed connection anchors | Max largest-void change (pp) |
| --- | ---: | ---: | ---: | ---: |
| FD2 versus spectral | 0.0872 | 0.000599 | 1 | 8.206 |
| FD4 versus spectral | 0.00906 | 0.0000427 | 0 | 0.00947 |
| FD8 versus spectral | 0 | 0 | 0 | 0 |
| Floor versus linear | 0.534 | 0.00448 | 7 | 17.878 |
| Linear versus cubic | 0.0434 | 0.000199 | 1 | 0.0815 |
| Clean linear 64-cell parent | 1.204 | 0.00581 | 5 | 17.519 |
| Clean linear 96-cell parent | 0.327 | 0.00199 | 2 | 17.444 |
| Clean linear 128-cell parent | 0.278 | 0.00122 | 1 | 17.626 |

The zero FD8 changes above apply to the observed subset. On the complete cores,
one unobserved voxel in ph000 changes web class, with a one-voxel largest-void
fraction change (1/32768), illustrating threshold sensitivity even at this
precision. Do not describe the operators as bitwise identical.

The decisive FD2 topology example is `ph003_SGC_s3_interior_00`: the largest
void occupies 23.28% of the observed science region with native FD2 versus 15.07%
with spectral, FD4 and FD8 tensors. The complete-core result is also different,
22.99% versus 14.88%, so this is not merely a survey-mask disconnection.

The largest floor-sampling example is `ph000_NGC_s2_boundary_00`: the observed
largest-void fraction is 17.49% under floor sampling versus 35.37% under linear.
The 128-cell clean linear parent still gives 17.75%, despite much improved RMS.
The analogous complete-core jump is also present. Small perturbations near a
connecting bottleneck join/split large components. Parent enlargement helps
average accuracy, but topology need not improve monotonically for each anchor.

### What dominates the remaining parent error

For clean linear parents, a constant tensor accounts for median residual-energy
fractions 63--72% at 96 cells and 79--83% at 128 cells across phases. In ph000,
the trace of this mean residual is below 2.1e-10. This is consistent with missing
long-wavelength exterior traceless shear, not failure of numerical trace closure.
It is an oracle diagnostic, not a licensed truth-assisted correction or a proof
that every residual is exterior shear.

The exact residual Gram matrix also shows strong cancellation between the
propagated FD trace filter and spectral-minus-native tensor terms. For example,
the ph000 mean squared contributions are 5.48e-6 and 5.75e-6, with cross term
-5.47e-6, versus 7.54e-4 for the parent/resampling term. Independent quadrature
addition would misattribute the inherited pipeline's error.

Removing the parent mean is not a fix: the clean 96-cell no-DC control has
4.57--7.95% median relative eigenvalue RMS, worst filling change 1.72 percentage
points and eight changed connection anchors. Preserve the fluctuating parent
mean under the declared completion; that still does not recover exterior shear.

### Recommendation and remaining boundary

Regenerate a separate E2E Gaussian-density/spectral-tensor product with a
consistent, convergence-tested interpolation rule. The user is willing to
regenerate data; the audit supports that choice over trying to repair an old
floor-sampled trace. Existing TSC counts can be reused, or redeposited under a
separately declared resolution/assignment contract. FD8 is a quantitatively
validated alternative for the current potential/stencil workflow. Neither
requires changing P12-A/D2's frozen native estimand or historical evaluations.

For subsequent training, qualify the science region and domain separately.
The 128-cell diagnostic (433.024 Mpc/h) reduces typical error but has not earned
a topology pass. Freeze acceptable absolute/uncertainty-scaled errors for the
chosen science functionals, then verify the matched target/sampling/domain
contract. E2E posterior uncertainties, calibration, diagnostic power and model
contracts are not supplied by this truth-only audit. `training_ready=false`
and `r0_physics_pass=false` remain correct; no full training dataset was replaced.

## Alternatives to assess against these measurements

The cleanest separate E2E contract is Gaussian-smoothed density and all six
spectral tensor components generated from the same full-box count spectrum,
with one declared observer-grid sampling rule. This can reuse the current TSC
counts; redepositing particles is needed only to change particle selection,
mass assignment or native resolution. Keep new files, hashes, normalization and
labels under an E2E-specific version rather than altering inherited products.
The FFT is intrinsically periodic, so full-box generation is the appropriate
place to make this target definition; applying it to a small crop is a different
boundary problem. See the [SciPy FFT reference](https://docs.scipy.org/doc/scipy/tutorial/fft.html).

Higher-order finite differences are a second option when retaining the existing
potential/stencil workflow is operationally useful. Regenerate tensor labels
and their trace consistently; merely changing a downstream operator cannot undo
the attenuation already embedded in an inherited trace. The audit tests orders
four and eight at exactly the same sites against the spectral reference.

A third option is to retain the native discrete estimand explicitly. On its
complete unsampled native lattice, its trace-to-tensor multiplier is

```text
T_ij(k) = s_i s_j / sum_a s_a^2 * delta_trace(k),  s_i = sin(k_i dx)/dx,
```

where the denominator is nonzero. This restores operator/target consistency,
not equivalence to a continuum spectral tensor. It cannot simply be applied to
the already resampled observer grid: sampling, aliases, boundary conditions and
the native derivative symbol must all match. Null modes remain unidentifiable
from the trace. Direct deconvolution by k^2/sum(s_a^2) is consequently not the
preferred repair when the original counts are available; it is singular at
discrete null modes and would require a justified band limit.

The legacy floor lookup and a nearest-site lookup should not be called the same
operation. The [Abacus TSC implementation](https://github.com/abacusorg/abacusutils/blob/main/abacusnbody/analysis/tsc.py)
uses rounded integer lattice coordinates for deposition. Local installed code
and both source density-build scripts were checked for the zero-offset rule.
Linear/cubic resampling should be considered alongside spectral regeneration,
not conflated with derivative order.

None of these changes recovers exterior harmonic tides from a finite density
crop. Larger generated domains, restricted science interiors, or a separately
authorized boundary-state model address that issue; extra observational context
alone is not missing matter in the generated tidal solve. The current audit
does not authorize a new learned external-tide model.
