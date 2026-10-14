# Exact block-conditional sampler: prospective protocol, 2026-10-09

## Temporary runtime amendment (approved before execution)

Following two Scratch-license-blocked allocation requests and stalled SciPy
imports from Scratch, the user explicitly approves a temporary Scratch-independent
runtime and output location. Use the read-only NERSC `pytorch/2.8.0` module;
do not modify cosmic_env or any shared installation. Record Python and dependency
versions in every result receipt. Small generated JSON/logs go to
`/global/u2/d/dkololgi/diagnostics/exact_block_sampler_20261009_temporary_v1`.
The CPU allocation declares U2 rather than Scratch because this synthetic
diagnostic uses no Scratch inputs, runtime or outputs. This is a task-specific
exception to the normal cosmic_env convention, not a repository default change.
Numerical targets, seeds, gates, tests and the entire fixed panel below remain
unchanged. Source hashes bind this amendment and the runtime-selection wrapper.

User authorizes implementation of the exact block-conditional sampler, analytic
invariance test, and global/coarse-update extension, including required compute.
First stage: one CPU interactive allocation, at most one hour. No Abacus data,
training, GPU use, production edits, sealed-phase access or posterior promotion.
Existing numerical implementations and prior receipts remain untouched.

## Target and constructions

Target is an analytically known Gaussian posterior N(mu,Sigma). Reuse the existing
reference prior and masked Gaussian observation machinery. For precision J and
an orthonormal update basis U, update only the coordinates along U conditional on
their orthogonal complement:

    x' = x + U [(U^T J U)^-1 U^T J (mu-x) + eta]
    eta ~ N(0, (U^T J U)^-1).

Local U selects every voxel of a patch. Forward systematic sweeps cover every
coordinate. The extension appends a rank-at-most-seven conditional update of
the constant and sin/cos fundamental modes along each axis. The basis is fixed
by geometry, never posterior eigenvectors. It is a joint move of long modes
CONDITIONAL on the residual field, not an independent splice or rescaling.

Compare local-only and local-plus-global. Recover their affine transitions M,b,Q
by composition of the update equations, accumulating independent innovation
covariances explicitly; NEVER define Q from the desired stationary covariance.
Check M mu+b=mu and M Sigma M^T+Q=Sigma, and spectral radius(M)<1. Separately
compare actual sampled one-sweep projections against these analytic moments.

## Fixed panel before execution

- n=4 and n=8 periodic domains, preserving the repository's reference spectral
  convention (these sizes are not a physical resolution-convergence claim).
- Priors: independent; existing spectral covariance; long-range covariance
  0.2*C_spectral + 0.8*11^T. All have unit marginal prior variance.
- Two observation regimes: template0 unchanged; template1 thinned by
  (i+2*j+3*k)%8==0 with noise standard deviation multiplied by2.
- For each target, one deterministic seeded data realization; no realized-truth
  coverage inference across this small panel. Target covariance is analytic.
- Unconstrained n4 patch side2/stride1; n8 side4/stride2, non-wrapping placement.
  Both still condition on the FULL exterior, including periodic cross-edge
  dependence. n4 also tests reverse patch order.
- Additional n8 targets for all3priors/both observations in the actual factor4
  block-zero-mean latent subspace. Use a block-local orthonormal basis; patches
  are full aligned4^3blocks. Project global directions into the same subspace,
  dropping null directions. Invariance is checked in independent coordinates.
- Reproduce the existing actual-canvas all-observation marginal-oracle baseline
  at n4 with side3 and full-domain controls, NFE512, on these SAME targets.
- Include the same-local-marginals/different-long-range-covariance construction
  as a deterministic information-insufficiency check.

This is18targets and48sampler/order configurations. No configuration is selected
away after results. Long-range/sparse cases are stress tests, not estimates of
Abacus correlation structure. Exact J is globally available: this is not yet a
bounded-halo or matrix-free survey implementation.

## Numerical gates and convergence

One-sweep relative invariance residuals <=1e-8, stable spectral radius<1.
Propagate affine moments from x=0 at 1,2,4,...,512sweeps, not Monte Carlo.
Record first registered sweep checkpoint with all of:

- posterior-whitened worst-direction mean contraction <=0.01 for a unit
  whitened initial perturbation;
- worst-direction relative covariance error <=0.01;
- analytic posterior probability in each nominal90% probe interval within0.005
  of0.90. Probe set: regional means, low modes and neighbouring-region contrasts;
  null-variance directions in the constrained subspace are explicitly omitted.

Report finite-budget nonconvergence as such, not a wrong invariant distribution.
Compute stationary integrated autocorrelation times analytically for each probe
from M,Sigma; report effective samples per sweep as well as burn-in. A global
update need not improve every probe or every target. Report timings of kernel
construction and measured repeated actual sweeps separately from analytic QA;
small-domain timings do not establish DESI scaling.

Independent sampler check: on one n4 correlated target, use16384independent
initial exact-posterior draws, apply one actual sweep, and compare means,
variances and projected covariances against exact target using analytic Gaussian
Monte Carlo standard errors (maximum standardized discrepancy <7). This is an
implementation check, not the primary correctness proof. Focused tests additionally
check full-block draws, precision-vs-covariance conditional identities, constraint
preservation, sampler-vs-affine deterministic noise propagation, and deliberately
wrong independent marginal updates.

Outputs are exclusive-create and source/hash bound at
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/exact_block_sampler_20261009_v1`.
Engineering completion, Gaussian invariance, finite-budget convergence, and
cosmological qualification are separate fields. No Abacus sampler change follows
automatically from a Gaussian pass.
