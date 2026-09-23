# Gaussian target / optimization / resolution controls

The original target, resolution and adaptive affine panels are complete and
all42checkpoints pass the state/provenance audit (2026-09-23). Protocol:
[registered controls](e2e_reference_target_controls_20260922.md). Machine-readable
evidence: [summary](evidence/e2e_reference_controls_20260922/summary.json).

## Matched U-Net interventions

All six CFM parents were restored at65536updates and continued16384steps with
identical starting model, Adam state and random streams. Endpoint2048draws and
256NFE are matched. Mean error is normalized by posterior RMS uncertainty;
covariance error refers to16probes, not the full field covariance.

| Mode / intervention | Mean error | Probe covariance error | Highest-shell power ratio | All five gates |
|---|---:|---:|---:|---:|
| Amortised, unchanged |0.11943|0.08866|1.38537|0/8|
| Amortised, LR decay |0.09271|0.07567|1.37901|0/8|
| Amortised, exact targets |0.07643|0.07213|1.34292|0/8|
| Fixed, unchanged |0.05744|0.07751|1.34761|0/4|
| Fixed, LR decay |0.02315|0.06602|1.33643|0/4|
| Fixed, exact targets |0.02577|0.06681|1.27039|0/4|

The fifth gate requires EVERY shell ratio in[0.9,1.1]; averaging shells or seeds
does not make a passing model. Exact targets improve amortised mean error by
36.0% versus the matched unchanged branch; LR decay improves it22.4%. Frozen
fresh-example exact-target MSE falls from0.01132 to0.00931/0.00622. Thus target
noise and optimization materially contribute; raw stochastic/exact training
losses cannot be compared directly. All eight exact-target amortised mean errors
are below0.1, but none of the36U-Net seed/case/branch cells passes the full gate.
The fixed fits improve means too, without eliminating relative small-scale error.

This rules out the strong claim that the observed mean gap is demonstrably
irreducible amortisation error. It does not establish a unique cause of the
remaining spectrum error or a finite-capacity impossibility.

## Learned affine controls

The control knows the posterior eigenvectors and analytic Gaussian-bridge time
form, but learns variances and either a fixed mean or dense observation-to-mean
map. These are deliberately privileged controls, not ordinary oracle predictions
or deployable survey models. In particular they only handle the two prescribed
mask/noise templates; they do not demonstrate transfer to unseen operators.

At16384steps both fixed-control variants pass4/4cells, including all-shell power.
Highest-shell ratios average1.04731(stochastic targets) and1.00153(exact targets).
The fixed Gaussian posterior is therefore learnable with this representation,
and the evaluation/sampler does not impose the U-Net's30%power error.
Fresh amortised affine fits have not passed at that budget: exact targets give
mean0.04534 but highest-shell ratio1.16711, while stochastic targets give
mean0.19119 and ratio1.22192. Their separately archived continuation must be
read before attributing that miss to conditional-model representation.

That continuation is now verified at65536steps. Exact-target amortised affine
mean error improves to0.03536, but highest-shell ratio stays1.16949 (0/8full
passes). Stochastic targets give mean0.19307 and highest-shell1.11264 (0/8).
Longer training does not by itself settle the remaining mean/variance coupling.

## Physical resolution control

This is a NEW nested physical prior, not a continuation of the original prior.
Eight fresh exact-target U-Net fits share the same physical Fourier series and
same512possible observation locations/noise; the16^3grid gets no extra data.
Nested exact prior and posterior identities were verified numerically.

On the IDENTICAL8^3subsampled observables, amortised mean error improves from
0.11459 to0.07608. Probe covariance error changes0.08866 to0.09061, and mean
octant coverage0.90036 to0.91334. With all common-observable gates, counts change
0/8 to4/8: better, not reproducibly qualified. Fixed common-observable fits pass
2/2on each grid. Native16^3highest-shell power averages0.64862(amortised) and
0.66201(fixed): a substantial deficit, not a repaired native spectrum. All ten
native16^3cells fail the full gate. Native8^3fixed fits pass2/2; amortised0/8.

Native Fourier shells differ in alias content. This experiment does not locate
one artifact at a unique physical k versus k/Nyquist, nor identify receptive
field, sparsity or target dimension as its sole cause. Equal update counts are
not equal convergence or equal compute; seeds do not create identical training
realizations across grids. It does establish sensitivity to numerical
representation under fixed physical observation information.

## Numerical controls and limits

Paired128/256NFE maximum absolute shell-ratio differences are0.000690(target
panel) and0.001248(resolution panel), much smaller than the spectral failures.
Coverage here is exact Gaussian probability inside generated octant intervals;
it is NOT population TARP or DESI validation. Two seeds/four observations and two
observation operators support this controlled diagnostic, not universal claims.

## Decision

Keep nonlinear and Abacus promotion gated; P12 and prepared survey-like data are
unchanged. The next practical hypothesis is scale/time parameterization rather
than another undirected training extension. A candidate is an invertible Fourier
preconditioner estimated only from the training PRIOR, with correctly transformed
noise and inverse transformation for evaluation. It must not use an observation's
true posterior covariance. Compare against the same U-Net at matched budgets,
retain physical-coordinate power/coverage gates and both seeds. This is a
recommendation, not an additional launched experiment or an established cure.
