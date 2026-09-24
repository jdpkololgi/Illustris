# Bounded Abacus Wiener comparison: calibration remains the objective

## Edge-likelihood correction after failed launch

Job58828297 failed in51s during training-data loading, before fitting or draws;
three solver tests passed. The original check incorrectly equated zero tapered
exposure with necessarily zero raw counts. The producer explicitly retains raw
counts while multiplying expected counts by support and apodization.

Corrected contract: at native resolution let w=apodization where expected>0,
and w=0 elsewhere. Sum w*counts and sum expected over the SAME2^3cells.
Since expected=w*lambda, the weighted Poisson variance is sum(w*expected),
not sum(expected). Apply train-only normalization to both expectation and this
variance; add the fitted unresolved term as before. No arbitrary exposure floor
or zero-density constraint. Entirely unsupported cells have zero likelihood
response and retain conditional prior uncertainty. The threshold1e-4 matches
the producer; contradictory positive expected counts below it still fail closed.
Audit raw, weighted and excluded counts for every training/evaluation anchor.

This baseline consequently uses tapered supported counts; the frozen neural
channels retain raw counts, so observational information is not identical at
edges. Report exclusion fractions and boundary/interior strata; do not call
an edge-dominated difference a clean architecture comparison. The original
physical targets, coordinate frame and scoring operators remain unchanged.
Repair tests cover partial-block masks, fractional-weight shot variance,
untapered conservation and invalid exposure, plus the existing solver controls.
The failed artifacts remain unchanged. Any approved retry uses a new v2 root.

Repair verification: six tests pass in2.772s. A bounded read of the first training
anchor ph000_NGC_s0_boundary_00 succeeds:232352.257 raw deposited counts,
11428.132 excluded (4.918%),212540.202 weighted counts; all unsupported pooled
counts zero and all supported variances positive. This is one-patch verification,
not a full data audit. Counts use cloud-in-cell deposition, not integer independent
voxel counts. Thus the propagated shot term is explicitly a diagonal Poisson
WORKING approximation; CIC introduces additional fractional-weight variance and
cross-cell covariance not represented here. The fitted white unresolved term
does not prove those correlations are repaired. Do not claim exact shot noise
or a perfectly matched physical likelihood from this correction.

User authorizes Step 1 only, with one CPU batch node for <=2 h and <=10 GiB
new Scratch. No interactive allocation use, neural training, nonlinear sampler
scaling, confirmation-phase expansion or Production VAC changes. Tests and a
representative anchor run inside the batch job before full evaluation. Failure
stops the job; no automatic retry or parameter tuning.

## Estimand and scope

Use the existing frozen context experiment's 384 training cores in ph000/002/003
and 32 previously exposed evaluation anchors in ph004/005. The 21 newly prepared
phases are NOT opened. These two evaluation phases cannot establish robust
phase-level confidence intervals. Voxels, anchors and optimization seeds are not
independent simulation phases. The comparison is exploratory, not fresh confirmation.

Keep original 48^3 parents, central16^3 owned cores,6.766Mpc/h cells and already
R7-smoothed block-averaged matter targets; no second smoothing. Keep the legacy
observer coordinate mapping for both methods. Its later documented distance
error is a shared limitation, not silently repaired for one arm. The P3b random
response is not the complete audited DESI observing process.

## Frozen classical model, no held-out nuisance fitting

Fit stationary isotropic FFT covariance of physical delta from all384training
parents after subtracting ONE training-wide mean (never each patch's mean).
Preserve DC variance. Shell-average orthonormal Fourier power; floor eigenvalues
at1e-6of maximum shell power for invertibility, recording affected-mode fraction.
Read original counts and random expected counts and sum2^3native cells. Neural
channels pool logarithms and cannot be inverted to recover counts correctly.
Check original raw/VDS and prepared-product hashes before access.

Fit eight cap×redshift-shell expected-count normalizations from training counts
only. Fit one positive scalar galaxy response b by Poisson-weighted regression
of count residuals on training target fluctuation. Fit nonnegative unresolved
variance s² from excess squared residuals above shot noise, training only.
Freeze everything in FIT.json before evaluation inputs are read.

Approximate likelihood after the edge correction: y=n_weighted-mu=A(delta-m), A=b*mu,
N=normalized_weighted_shot+mu²*s², diagonal, with zero response outside exposure. This is a Gaussian
linear working likelihood, not a Poisson model or a claim that shot noise and
bias are truly local. The target is smoothed while counts are not; s² is a
deliberately simple approximation to unresolved/nonlinear galaxy fluctuations.
Scalar bias, stationary isotropic prior, white extra noise and periodic parent
boundaries are explicit misspecification risks. Negative-density draws are
reported, never clipped. No calibration rescaling on evaluation truth.

Sample64constrained realizations per anchor using Q=C^-1+AᵀN^-1A, FFT covariance
operations and preconditioned CG. Require relative residual<=2e-7 for every
mean and draw solve, <=2000iterations. Preserve log/error on failure. Validate
the solver against a dense Gaussian posterior, masked/prior limit and invalid
inputs before touching scientific data.

## Matched evaluation and decision

Reuse first64saved draws from every final20,480-update A/B/C/D seed, same anchors.
Score density fair CRPS, rank/finite-draw attainable90% coverage and width;
eight regional means and total core mean; spectra/cross-correlation; eigenvalue,
eigengap scores; tidal energy and threshold-zero class Brier score. Use the
same periodic-parent tidal operator for ALL arms, including D; this deliberately
does not evaluate D's additional coarse external-tide information. Compare
against original full-box truth and report true-parent closure error separately.
No new smoothing. Tidal summaries use fixed256spatial probes per core to bound
cost. Report by phase and observational stratum, not only a pooled winner.

The goal is a calibrated field posterior, NOT to establish classical superiority.
A useful result identifies whether an inexpensive working posterior already
captures missing regional uncertainty, or instead exposes likelihood/prior
misspecification. Calibration alone is insufficient (an uninformative prior can
pass some coverage tests); require proper scores, sharpness and joint quantities.
Truth-based repeated tests are falsifiable but finite summaries do not prove
the full conditional distribution correct. Nor are old neural scores a rigorous
lower bound on attainable neural performance; improved training may help but is
not guaranteed. Laplace's four-case failure does not imply every approximation
fails on Abacus. Scaling nonlinear inference includes scientific likelihood and
boundary choices, not just FFT engineering.

Stop after this comparison. Do not launch another classical or toy programme.
Return a focused recommendation for the shortest path toward calibrated fields,
with numerical, mock-conditional and real-survey validity kept separate.
