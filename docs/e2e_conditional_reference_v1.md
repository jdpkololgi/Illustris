# Bounded learned conditional-posterior reference, v1

Registered 2026-09-19, before fitting. This is the prerequisite to the prepared
coupled-field campaign, not approval to run that campaign. P12 is unchanged.
User approved one shared A100 for at most two hours, CPU work within the same
allocation, and at most 20 GiB new Scratch. No Abacus training or sealed data.

## Scientific question

We seek joint draws from p(delta | observed galaxies, survey response; simulator
assumptions), not only plausible marginal densities or a good posterior mean.
The present test separates numerical integration error, fixed-distribution
learning error, and the additional burden of amortising over observations.
Failure at finite budget is not a proof of representational impossibility.

## Registered Gaussian rung

An 8^3 periodic Gaussian field has known spectrum proportional to
[1+(|k|/1.5)^2]^-2, normalized to unit voxel variance, including stochastic DC.
Observe y=Mx+epsilon with two known binary masks and a known spatially varying
noise standard deviation (0.35 to 0.9). Missing y is zero-filled; mask and noise
are separate input channels. Covariance and conditional means are computed in
float64 from the full precision matrix, independently checked against the
Gaussian conditioning identity. This is a deliberately small spatial diagnostic,
not a demonstration of production-scale representation or DESI robustness.

Within each mask/noise template the true conditional covariance is independent
of observed y; only the conditional mean changes. Thus this first rung is easier
than the nonlinear cosmological problem: it tests learning two covariance
operators and observation-dependent means, not arbitrary y-dependent covariance.
The lognormal/Poisson rung would add that further difficulty. The small network
has 286,947 parameters; it is the existing backbone family, not the full-size
coupled production-shape configuration.

There are four fixed, independently generated evaluation observations, two per
mask. For each objective and seed: fit one amortised model on fresh joint (x,y)
draws with both masks; separately fit one fixed-observation model per first two
cases on fresh *exact posterior* draws. This is an easier distribution-fitting
control, not a claim those posterior training examples are available for DESI.
The paired comparisons use cases 0 and 1; cases 2 and 3 are amortised-only checks.

Twelve fits: 2 objectives x 2 seeds x (1 amortised + 2 fixed). All use the existing
ConditionalVDM 3D U-Net, base 8, two downsampling levels, three observation-only
channels, unchanged state-residual output. VDM uses existing continuous VLB and
fixed gamma [-13.3,13.3]; CFM uses the independent Gaussian-to-data linear bridge
and velocity MSE. Adam, lr 3e-4, batch 32, no clipping or auxiliary loss.
4096 updates each; checkpoints 1024/4096. No per-field normalization,
truth-as-condition, adaptive loss changes or data-dependent model selection.

## Numerical and metric controls first

Propagate exact Gaussian means and covariances through the current VDM
ancestral transitions and CFM Heun integrator at 32/64/128/256/512/1024 network
evaluations. This isolates finite endpoint, initialization and discretization
errors without Monte Carlo noise. VDM uses its existing sampler; CFM uses two
evaluations per Heun step. Compare oracle moment propagation against sampled
oracle paths in tests. A <=5% relative covariance error is the numerical gate;
neural conclusions at a failing NFE remain sampler-confounded.

At each checkpoint draw 512 samples per observation at NFE 128 and 256, with
identical seed addressing across models. Compare to exact conditional mean,
voxel variances, regional-mass distributions and the full covariance of a fixed
16-dimensional orthonormal spatial-probe basis (eight octants, six Fourier
sin/cos modes, two mixed modes). Report residual power by Fourier shell and
mean-field error. Probe coverage is the *exact posterior probability mass*
inside sample 5%-95% intervals, not repeated truth rank tests on four cases.

Exact-reference Monte Carlo ensembles (128 repeats at the same 512 draws)
calibrate mean/covariance estimator fluctuations. Operational per-case gates:
normalized mean RMS <= max(0.1, exact-reference 99th percentile); relative
projected covariance Frobenius error <= max(0.15, reference 99th percentile);
mean voxel-variance ratio within [0.9,1.1]; mean octant coverage in [0.85,0.95].
These are declared toy tolerances, not DESI science tolerances or simultaneous
99% tests. Covariance controls retain the correct marginals while deleting
off-diagonal covariance, and separately shrink posterior variance by 30%.
Metrics must distinguish these from the exact posterior before interpreting
learned-model results. Retain all case/seed results; no pooling seeds as
independent universes and no post-hoc tolerance changes.

## Decision and next rung

- Oracle fails: diagnose sampler before attributing error to learning.
- Fixed fits fail: architecture/objective/optimization remains a cause; cannot
  blame amortisation alone, even if amortised fits are worse.
- Fixed passes, amortised fails reproducibly: evidence for an additional
  conditional-learning bottleneck at this budget, not a theorem against it.
- Both pass across seeds/cases, metrics and numerical controls pass: qualify a
  Gaussian learned rung only. Next use lognormal matter with Poisson counts,
  masks and varying exposure, comparing to independent converged numerical
  posterior chains (split R-hat, bulk/tail ESS and step-size/resolution checks).
  That posterior is NOT analytic. It needs its own frozen reference/protocol;
  no automatic nonlinear or Abacus run is authorized by passing this toy.

The 10% improvement gate in the earlier coupled proposal must be re-derived
using conditional references and phase-aware power calculations before launch.
Sharing fields does not by itself prove a chi-square statistic has fewer than
seven degrees of freedom: its joint covariance/rank or simulated null must be
used. Do not replace one invalid significance conversion with guessed dof.

The fixed-observation versus amortised decomposition follows the concern raised
by Doeser & Jasche, arXiv:2606.10023; it is a diagnostic adaptation, not a
reproduction of their cosmological forward model. Exact-sampler and metric-null
controls incorporate the useful additions in Opus's follow-up response.

## Operational contract

Source/config hashes are saved at launch and checked on resume. Checkpoints
contain model, optimizer and RNG state. Atomic publication; finite runner with
110-minute internal deadline; no successor allocation or model changes. tmux
preserves the approved interactive shell, Slurm bounds compute. A completion
receipt means all registered fits/evaluations exist, not scientific gate success.

### Launch record

Ten focused unit tests passed, including independent sampled-oracle moment
checks and exact optimiser/RNG replay. Frozen source is under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/conditional_reference_20260919_v1/source`.
Allocation58594516, shared_interactive, one A100 on nid008337, two-hour limit;
tmux `conditional_reference_20260919` on login31. Both objective GPU smoke
tests passed. The shell exits after the finite step, releasing the allocation.
The `results/manifest.json` binds source/config hashes and the Slurm job.

The run has now completed within its budget. See the
[scientific conclusions](e2e_conditional_reference_conclusions_20260919.md) and
[complete evidence](evidence/e2e_conditional_reference_20260919/README.md).

Early exact controls identify a genuine numerical distinction: VDM covariance
error is about9.7% at128NFE and5.0% at256NFE for mask0, below5% at512NFE;
CFM Heun error is0.022% at128NFE. These are exact affine moment calculations,
not neural results. The registered neural NFE128/256 results must retain this
VDM qualification; no tolerance or training recipe was changed after seeing it.
