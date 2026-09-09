# E2E v2 finite-domain gate — 2026-09-09

The user authorized spare interactive capacity for the spectral finite-domain
audit, a domain decision and downstream model/transform/diagnostic contracts
before a training canary. No learned fit, external-tide model, D2 modification,
selection/confirmation opening or new training geometry is included here.

## Frozen comparison before this run

Use all 96 response-selected training anchors in ph000/ph002/ph003. Read only
the six training shards in spectral_20260909, hash-check them and retain both
the complete and observed fixed 32-cell science masks. Compare the registered
64/96-cell Gaussian R7 parents with the independent full-box spectral tensors
at identical cubic-sampled sites. The existing periodic parent FFT includes the
parent mean as mean(delta)/3 on its diagonal and applies **no second smoothing**.
This is a declared finite-domain completion, not a survey-periodicity claim.

The configuration is `configs/e2e_field_domain_gate_v2.json`. Tolerances apply to
every anchor, eigenvalue or pair bin and both masks, not just pooled averages:

| Quantity | Maximum allowed discrepancy |
| --- | ---: |
| Eigenvalue RMS | 0.01 absolute **and** 5% of within-core truth scatter |
| Absolute eigenvalue bias | 0.005 |
| Four-class disagreement | 1% of supported voxels |
| Two-collapsed-region filling fraction | 0.5 percentage points |
| Environment pair probability, each 10–20/20–40/40–80 Mpc/h bin | 0.0025 absolute |
| Largest-void fraction | 1 percentage point |
| Fixed terminal connection events | No changed axes |

These conservative **internal design tolerances** are now frozen before the
v2 run, with the prior v1 audit already known. They are not blind preregistration,
posterior-calibration thresholds or an empirically established cosmological
error budget. The aim is to rule out substantial deterministic distortions
before spending on learned fits. One-percent volume errors and event reversals
are retained separately because small eigenvalue RMS can hide topology changes.
Report a predeclared 0.5/1/2/5-times tolerance sensitivity envelope without using
it to replace the primary rule; multiplying zero still permits no event flips.

If neither domain passes, select neither for tidal-science training. If one
passes, it is only an eligible screen candidate: R0 additionally requires a
posterior-scaled budget. The frozen future fraction is 0.1 of posterior standard
deviation for the matched eigenvalue and continuous science functionals, for
bias and RMS separately. That test cannot be marked passed without an independent
uncertainty reference and cannot be made true by widening a fitted posterior.
No such reference is available in this truth-only audit. This is a genuine
remaining gate, not a reason to claim the present run measured posterior error.

Two 96-cell attribution controls remove the parent mean or the oracle constant
traceless tensor residual. The latter is fitted separately on each evaluated
mask and uses truth: it is explicitly non-deployable. Neither control may qualify
a domain, change the stored targets or become an inference correction. A larger
128-cell domain had only a linear-sampled v1 audit; it is not silently substituted
for either registered v2 domain or declared adequate from that earlier result.

## Downstream contract and stop sequence

The common target remains the physical Gaussian R7 scalar field; the independent
spectral tensor is the tidal reference, never the tensor reconstructed from the
same bounded density crop. Preserve whole-phase roles, existing channels and
train-only affine normalization, the parent mean, shared parent/sample identity,
and independent confirmation access. Do not change P12-A/D2's estimand.

R0 is upstream of choosing a science-training model. No architecture, compute
budget or wavelet arm is declared ready if the density-to-tide domain fails.
The proposed matched CFM/DIFF comparison and conditional WCFM challenger remain
the existing plan; their final architecture/transform settings must be frozen
against an adequate target/domain, not a convenient failed geometry. Existing
Haar/Fourier/RNG fixtures are engineering evidence, not that final freeze.

For diagnostic power, freeze the following requirements now:

- Cosmological phase/source blocks are the independent units. The 96 anchors,
  their caps and all voxels are not 96 independent universes. Report per-phase
  and shell/support strata; a failed stratum cannot be rescued by pooling.
- Keep marginal eigenvalue, environment-pair, largest-void and connection checks
  distinct. For later posterior tests, evaluate the corresponding draw-wise
  quantities, not functionals of a posterior-mean field.
- A calibration/power study must include a calibrated dependent-field null and
  explicit prior-only, half-width, independent-sibling and low-mode/shear-bias
  alternatives, under the same response masks and sample count as evaluation.
- Use disjoint simulation seeds to calibrate and test critical values. Require
  the upper 95% binomial interval for familywise false positives <=0.10 and
  the lower 95% interval for power >=0.80 at predeclared practically relevant
  deviations. Use at least 1,000 independent synthetic panels for each evaluated
  null/alternative and keep this synthetic evidence separate from cosmological
  source-block uncertainty. These are future execution requirements, not
  measurements performed by the domain audit.
- MIRA stays supplementary until its realistic response-conditioned activation
  study passes. The existing 12-cell analytic fixture is insufficient to activate
  it; three training phases cannot estimate a high-dimensional independent-box
  covariance reliably. Do not invent a measured effective sample size.

A domain failure blocks the requested final science-training freeze/canary.
Continue only after an explicit change of scope to a density-only benchmark,
a separately specified larger-domain audit, or an exterior-tide-aware target/model.

## Execution

Four focused domain tests pass, including the mean/trace convention, separate
topology gate and non-deployable traceless control. The CPU audit needs no new
2048-cubed FFT or GPU. Use the standard one-node CPU interactive route with a
bounded 30-minute limit, as documented by
[NERSC](https://docs.nersc.gov/jobs/interactive/#perlmutter-cpu-nodes), and release
the allocation when its foreground step exits. Run and terminal evidence will
be recorded after execution.
