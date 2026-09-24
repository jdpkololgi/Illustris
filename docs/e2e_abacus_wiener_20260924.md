# Bounded Abacus Wiener comparison: calibration remains the objective

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

Approximate likelihood: y=n-mu=A(delta-m), A=b*mu,
N=mu+mu²*s², diagonal, with zero response outside exposure. This is a Gaussian
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
