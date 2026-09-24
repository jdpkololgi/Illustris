# Paired development evaluation — no confirmation access

User requests evaluation of6656and13312updates, both seeds17/29. All models
are frozen EMA weights; no training or hyperparameter selection in this run.
ph012/013only,16prepared pairs each. Same observations, truths, operator and
draw identities across checkpoints. Same sampled coarse realization identity
feeds each joint fine draw. NO truth-derived coarse conditioning in inference.

## Bounded ledger and cost

Main:32pairs×2seeds×2checkpoints×32draws =4096joint+wide samples,128NFE per
factor (64Heun steps; two evaluations per step). Refinement: first geometry-ID
in each development phase,2seeds×2checkpoints×8paired draws at256NFE =64extra
samples. Compare refinement with the SAME first8main draws, not32versus8.
This tiny refinement panel can flag gross discretization error, not certify a
2ppcoverage change. Microbatch2is fixed across all runs and resumes.

Earlier qualified benchmarks: coarse4.9736s+fine6.5720s per sample at128NFE;
main estimate13.14GPUh, refinement0.41GPUh, plus load/physical scoring/I/O.
Requested one4GPU node/4h (16GPUh max), tmux and atomic2draw chunks, no automatic
renewal. Actual throughput may differ; stop before deadline with durable draws,
report incomplete ledger rather than reduce draws or claim completion. Source,
checkpoints, chart, chunks and truth references hashed. Scratch approximately
9GiB uncompressed density outputs; no particle copies or changes to preparation.
User approved this full development resource bound. First2draws in each worker
are regenerated for exact in-process addressed replay (8additional technical
draws total). This is not a fresh-process sampler replay certificate.

## Measurements

Use corrected common rectangular consistent wide+fine tensor operator and
independent full-box truth. No additional R7smoothing. Every draw must conserve
coarse block mass to2e-6relative, remain positive/finite, and satisfy physical
trace identity. Same physical layout and threshold-zero tidal classes as plan.
Density scores on all owned core voxels; eigen/gap/Brier on fixed256probes/core.
Regional masses: both16³owned cores; eight24³block-aligned probes;
eight12³offset fine-sensitive probes from registered locations. These probes
overlap and are not independent trials. Fair CRPS,50/68/90/95coverage/width,
rank histograms, error/spread and coherent per-region bias retained.
At32draws nominal90%attainable coverage is29/33=87.8788%, not90%.
Report both phases and seeds; no voxel-derived significance or universal pass.

Dependence: seven unit-L2 block-orthogonal DCT modes per core; scale from the
same fixed26training-only pairs used for profile fitting (no development scales).
Fair14vector energy and matched7cross-core variogram, no mandatory percentage
improvement gate. Common unwindowed rectangular pseudo-spectra: sample, mean,
unbiased residual, truth-minus-mean and cross-correlation. DC/regional quantities
reported separately. No claim that rectangular pseudo-power is a periodic-box
isotropic power estimate. Bands fixed0,.04,.08,.16,.32,1h/Mpc; no pass/fail based
on poorly resolved/high-kbands in this exploratory screen.

Class probabilities: Brier and per-class calibration-in-the-large, NOT error
against an unavailable exact conditional probability. Absorption is not exactly
identifiable here; residual/mean-error decomposition is diagnostic only.

Tests cover forbidden-phase access, addressed noise, coarse-annihilating probes,
known-Gaussian coverage versus severe underdispersion, and equal-marginal but
erased-dependence variogram control. These establish metric direction and gross
sensitivity, NOT a phase-cluster detection limit for Abacus. Actual32pair panel
remains too small to certify tight phase-conditional calibration. Confirmation
metric-power controls and maturity qualification remain separate future gates.

## Interpretation

First ask: finite coherent samples, checkpoint improvement, and whether sampler
refinement materially changes the answer. Then inspect regional calibration
and proper scores jointly; neither better plots nor broader intervals alone
qualifies a posterior. A coarse failure cannot be rescued by fine coupling.
OldD/Wiener inputs differ, so historical scores are not a matched causal baseline.
No confirmation, architecture sweep, classical expansion or automatic retraining.
