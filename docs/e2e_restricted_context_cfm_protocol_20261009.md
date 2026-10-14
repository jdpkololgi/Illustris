# Restricted exterior context and learned Gaussian CFM pilot

## Negative-case handling correction

The first oracle run stopped when a restricted-context chain had spectral
radius>=1: the harness incorrectly treated that scientific negative result as
an engineering exception. The correction records instability and omits
nonexistent stationary metrics instead of terminating the panel. No target,
context, model or gate changes. Preserve the first partial output under
`oracle`; the complete rerun uses `oracle_negativecases_v2` with new source
bindings. Full-information positive controls still fail closed if unstable.

Prospective contract frozen before numerical execution on 2026-10-09. The user
approves the restricted-context Gaussian oracle comparison followed by one
exterior-conditioned CFM pilot, including required compute. This does not
authorize Abacus interventions, production changes or access to sealed phases.

## Scientific distinction

The target is a synthetic density-contrast field x with a known Gaussian prior,
observed through masked additive Gaussian noise. This is not a galaxy catalogue
likelihood. Analytic Bayes conditioning supplies p(x|y). An exterior-conditioned
update draws a block given the CURRENT sampled exterior and fixed observations;
the exterior is not assumed to equal the true field. Repeated updates integrate
over exterior uncertainty. Exact global updates redraw the constant and six
fundamental sine/cosine mode coordinates conditional on their orthogonal
complement and the observations. They are not independent coarse-field draws.

## Restricted-context oracle panel

All observation information remains available through the exact posterior.
Only the field-exterior information is restricted. Missing exterior coordinates
are marginalized analytically, not clamped to zero and not assumed independent.

Twenty targets: n=4,6,8 with independent, spectral and long-range priors under
standard and sparse/noisier observation regimes, plus two n6 nonperiodic targets.
Periodic spectral covariance uses the discrete Laplacian spectrum with fixed
correlation parameter 1.5 CELLS; long-range is 0.2*C+0.8*11-transpose. This
differs intentionally from the earlier fixed-domain-frequency prior to permit
domain-size comparisons at fixed cell spacing and correlation parameter.
The open-domain covariance is a squared-exponential kernel of length1.5cells
with a0.05diagonal term, normalized to unit variance. The observation templates
and sparse thinning follow the preceding panel. One seeded observation per
target; no claim about cosmological repeated-truth coverage.

Every local block is2^3 cells at every domain size. Six exterior contexts:
full exterior, none, one-cell collar, two-cell collar, one-cell collar plus
exterior mean, and one-cell collar plus seven geometric modes. Modes exclude
the updated block; far-exterior modes plus collar values carry the same
information as full-exterior modes plus collar. Redundant directions are
removed. Periodic distances are used only for periodic targets.

For each context compare local-only sweeps and local sweeps followed by an
EXACT global conditional move. This hybrid does not assume a global move repairs
wrong local conditionals. All targets use forward nonoverlapping aligned sweeps;
n8 long-range/sparse additionally uses reverse order and a one-cell periodic
tiling shift. This yields264 rows, retaining every result.

Report one-sweep target invariance, exact stationary Gaussian moments from the
discrete Lyapunov equation, worst posterior-whitened covariance error, probe
90% probability, and expected conditional KL caused by discarded exterior
information. Stationary autocorrelation is measured against the chain's ACTUAL
stationary distribution and must not be interpreted as accuracy when that
distribution is wrong. Positive full-exterior and independent controls require
invariance residual <=1e-8 and spectral radius<1. An approximate-context screen
requires covariance Frobenius error<=1%, worst whitened covariance error<=1%,
whitened mean error<=0.01 and probe90% probability within0.005 of0.90.
Approximate passes are not labelled exact invariance. Apply the existing
1,2,...,512 target-convergence checkpoints only when stationary accuracy passes.
Report setup and actual sweep costs separately. This remains a dense oracle,
not evidence of matrix-free survey scalability or finite-context sufficiency
outside the tested family. Fine-stage constraints remain a separate extension;
their exact full-information feasibility was established in the preceding test.

## One learned CFM pilot

The pilot uses full-exterior conditioning as a deliberately sufficient
small-domain reference; a failed restricted interface is not silently promoted
to training. It uses n4, the new long-range prior and the sparse observation
regime. Observations vary across training examples; mask, noise law and prior
are fixed. Generate fresh joint samples (x,y) from that synthetic prior and
likelihood, then use exterior x_E and y as context for each of eight disjoint
2^3 blocks. Held-out observation seeds are separate. Existing marginal-model
checkpoints do not have this interface and are not reused as conditional models.

Use ONE time-conditioned affine velocity network with eight block heads: a
two-hidden-layer128-wide SiLU network of time features outputs the affine
coefficients acting on noisy block state, exterior/observational context and
an intercept. Affinity in state/context is appropriate for this Gaussian pilot
and permits exact moment accounting even AFTER learning. It is NOT the current
3D convolutional CFM architecture and does not establish its learnability.
Context whitening uses the fixed synthetic context covariance as preprocessing;
the network never receives a per-observation posterior mean, covariance,
precision or analytic conditional target as an input.

Train the usual independent-endpoint linear CFM objective:
z_t=(1-t)epsilon+t*x_B with target velocity x_B-epsilon. Fresh synthetic fields,
observation noise and base noise supply stochastic training examples. One seed,
20,000 updates, batch2048 (32stratified times by64examples), Adam with initial
learning rate0.0003 and cosine decay to0.00003. No architecture sweep, loss-based
run extension, teacher-velocity training or checkpoint cherry-picking. Save and
report checkpoints0,5000,10000,20000; the final checkpoint controls conclusions.

First compare exact Gaussian conditional draws with exact conditional CFM
flows at NFE128/256/512 using Heun integration. Then evaluate the learned
conditional flows using the SAME integration and affine accounting. Exact
positive controls must agree to<=0.1% covariance error at the finest NFE before
learning conclusions are interpreted. Test actual model calls against extracted
affine transitions to catch interface/integrator mismatch.

Evaluate conditional mean/covariance errors and the assembled Gaussian chain
on eight held-out observation realizations. Report local learned sweeps alone
and learned local sweeps plus an EXACT global update; that hybrid is an oracle
diagnostic, not a fully learned method. Learned stationary accuracy uses the
same1%/0.005 probability screen as the context comparison, with explicit
NFE refinement and stability results. Numerical exact-invariance1e-8 is NOT
imposed as a neural approximation tolerance. A completed but failed pilot is
a valid result; no unregistered rescue training follows.

## Execution and provenance

Use bounded interactive development allocations under the two-allocation limit:
one CPU allocation for the oracle comparison and one single-GPU allocation for
the pilot when ready. Scratch remains scheduler-reserved; the previously
approved temporary read-only NERSC pytorch/2.8.0 runtime continues for these
synthetic tests. Do not modify cosmic_env or shared installations. Small logs,
JSON and model checkpoints go under
`/global/u2/d/dkololgi/diagnostics/restricted_context_cfm_20261009_v1`.
No large stored ensembles, external uploads or source commits are needed.
Freeze source hashes per stage, never overwrite receipts, and release each
allocation promptly. Scientific failure and engineering completion are separate.
