# Spectral preconditioning: registered comparison

User requests implementation and committed scientific records,2026-09-23.
Scope: the original8^3masked Gaussian reference, not Abacus or P12. No nonlinear
or production promotion is implied. Preserve prior results without overwriting.

## Claims to test, not assume

Mean errors can favor inflated variance in the restricted affine Gaussian-bridge
family. This is not an intrinsic inconsistency of CFM: the exact conditional
velocity remains the population optimum, and the effect depends on model/mean
misspecification and optimization. Check Opus's quantitative prediction against
saved affine parameters using the population observation covariance, not just
four held-out observations. Draw-based U-Net absorption predictions are
diagnostic approximations, NOT a theorem for nonlinear velocity fields. Remove
finite-ensemble mean-estimation noise before attributing mean error to learning.
The earlier resolution panel does not uniquely distinguish variance range from
aliasing, physical receptive field or time/noise parameterization.

## Transform and leakage boundary

Estimate per-mode prior power from8192fresh TRAINING-prior realizations using
independent seed912731. No held-out observations, posterior covariance, truth
field or fitted checkpoint enters the transform. Use the real periodic FFT,
W(k)=1/sqrt(max(P_train(k),1e-6*max(P_train))). Keep the DC mode; no per-field
mean removal, normalization or clipping. Store weights, estimation count/seed,
hashes and inverse tests. Estimate once and freeze across seeds and branches.

## Four arms, with matched starting weights and fresh data streams

1. `physical`: original physical coordinates, unit-white physical base noise,
   ordinary velocity MSE.
2. `weighted`: SAME model input and bridge/base noise, loss ||W(vhat-v)||^2.
   This isolates spectral loss weighting.
3. `coordinates`: train u=W x, start with W epsilon (colored noise in u-space),
   target W v, ordinary u-space MSE. This preserves the physical bridge and
   weighted objective but changes the network's numerical coordinates.
4. `white_bridge`: train u=W x with unit-white u-space base noise. This is an
   EXPLICITLY DIFFERENT bridge, equivalent to colored physical base noise
   W^-1 epsilon. Recompute exact conditional-velocity targets for that bridge.

Conditions remain the same physical y/mask/noise channels in every arm. Sampling
uses the matching base distribution, vector field and inverse transform. Do not
mistake unchanged white noise after a coordinate change for the same bridge.

Factorial panel: four arms × stochastic/exact targets × fixed observation0/
amortised × seeds17/29 =32fresh U-Net fits. Same base8/levels2 model, Adam3e-4,
batch32,32768updates. Checkpoints8192/32768;512draw learning curves,2048final
draws at128/256NFE. Fixed fits evaluate case0; amortised fits all four cases.
Do not compare these fresh fits causally with older continued fits.32k is a
finite matched budget, not a convergence claim. No automatic extension/selection.

## Registered predictions and gates

- Fixed U-Net: weighted/whitened arms should reduce all-shell relative spectral
  error, especially the top shell, without degrading physical mean/coverage.
- Amortised: measure mean and variance separately; improved high-k means may
  reduce affine-style absorption. Do not require nonlinear U-Net errors to obey
  the scalar affine formula. A residual unexplained by that diagnostic remains
  a variance/velocity-model error candidate, not proof of one mechanism.
- Weighted vs coordinates tests representation at the SAME physical bridge;
  coordinates vs white_bridge tests noise/time geometry as a package.

All promotion metrics are in PHYSICAL coordinates: mean<=max(.1,matched-null
p99);16-probe covariance<=max(.15,matched-null p99);variance ratio[.9,1.1];
octant coverage[.85,.95];EVERY shell power ratio[.9,1.1]. Require all cases and
both seeds, not just their average. Include full per-shell results, NFE changes,
mean-error absorption estimates (raw and Monte-Carlo-corrected), frozen exact
velocity risk and all failures. Exact targets remain privileged diagnostic
supervision; practical success requires stochastic-target performance too.

## Execution and persistence

Close out previous42checkpoints first; run transform/teacher/sampler tests and
one-GPU throughput smoke. Freeze committed source/config into Scratch, then
use independent workers on four GPUs if measured runtime warrants. At most two
allocations; checkpoints and Slurm steps with durable tmux launch logs. Record
job IDs, status, hashes, completion manifest and wall time. Commit plan before
scientific fitting, then execution/results. Do not claim launch or success before
the corresponding receipt. Full E2E and nonlinear training remain paused.
