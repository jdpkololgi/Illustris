# Classical/hybrid decision experiment: start now after evaluation closeout

The user explicitly replaces perfection as a stopping criterion with downstream
T-Web relevance or a simpler equally useful calibrated alternative. Existing
plans listed classical reconstruction and a later converged lognormal/Poisson
reference, but no executable cost/accuracy comparison or hybrid was specified.
This document makes the immediate Gaussian comparison executable. No further
neural fitting or alpha tuning is authorized by this comparison.

## Immediate arms (development only)

- Selected stochastic alpha, EMA, final checkpoint, both seeds: existing draws.
- Classical Wiener constrained realizations: x*=x_prior + K(y-y_mock), with
  y_mock drawn using the SAME mask and heteroscedastic noise. Independent
  algebra verifies mean/covariance against precision-form Gaussian conditioning.
  This needs no network training and is a genuine sampler, not a point estimate.
- Conditional low-mode hybrid: retain sampled high modes, redraw the seven real
  Fourier low-mode coordinates (|k|<=1, including DC) from their Gaussian
  conditional GIVEN those high modes and observations. Schur complement keeps
  cross-mode dependence. This preserves the target when the high marginal is
  correct; it cannot fix an incorrect retained high marginal.
- Independent low-mode splice: negative control, draws low modes independently
  from their marginal; do not mistake correct low marginals for correct joint law.
- Oracle-through-hybrid control verifies the conditional replacement construction.

Four existing development observations,2048draws/arm. Evaluate original field
metrics, R={0,1,2}cell smoothed uncertainty, width2/4/8regional intervals and
threshold-zero toy tidal-class probabilities. Independent exact draws provide
the nonlinear Monte Carlo floor; linear coverage uses exact Gaussian integrals.
Record classical setup/draw and hybrid correction times. Hybrid retains neural
generation cost; no unfair CPU-vs-GPU speedup or scalable-Abacus claim.
Use existing allocations if idle; no cancellation of other work. Preserve the
two-allocation cap. Bound this diagnostic to one short compute step.

## Decision sequence

1. Finish the two missing alpha evaluations and aggregate all384ensembles.
2. Execute these classical/hybrid arms now; do not wait for mathematically perfect
   neural Gaussian performance. Keep original gates as diagnostic records, not
   a requirement to spend indefinitely on matching them.
3. Gaussian classical sampling is analytically exact, so its success establishes
   the baseline cost/accuracy, not new evidence that it works on real galaxies.
   If hybrid adds no practical benefit, do not train a more complicated hybrid
   merely to salvage the neural approach.
4. The next nonlinear decision experiment is lognormal matter/Poisson galaxies,
   with a converged multichain sampler as both reference and competitor, a
   cheap Laplace/log-density Gaussian approximation, and a neural/hybrid arm only
   when the reference has verified convergence. That requires a separate concrete
   generator/noise/selection and convergence protocol; it is not implemented or
   silently launched by this Gaussian test. No Abacus/DESI likelihood is implied.
5. Before scientific qualification, define voxel size, smoothing, tidal threshold
   and tolerable probability/coverage errors for the intended DESI use. The toy
   is dimensionless; its R=1 is not7Mpc/h. Report dependence errors after smoothing
   and cost, rather than choosing tolerances after seeing a favorable result.

The hybrid here is a privileged Gaussian diagnostic using known conditional
covariances, NOT a learned nonlinear residual model. A useful Gaussian result
must still earn its complexity on the nonlinear rung against classical sampling.
