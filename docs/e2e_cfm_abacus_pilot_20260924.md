# Bounded coupled Abacus CFM pilot: protocol before training

## Scope and authorization

Objective: a useful, calibrated mock-conditional R7 field posterior, not prettier
draws or winning against a classical model. One coupled architecture, two seeds
(17,29), each with a wide coarse factor and joint64x48x48fine residual factor.
User approves first segment: one four-A100 node <=2h,8GPUh; no automatic resource
renewal. Benchmark and train all four factors independently. Preserve checkpoints
under tmux; no Production VAC modification. Confirmation remains sealed.

This supersedes the main plan's additional nonlinear-toy prerequisite and the
large four-arm launch proposal. It does NOT rewrite preparation authorities.
Use corrected cartesian_v2 products:13train phases/1664pairs, ph012/013development
(32pairs), ph014–019confirmation (96pairs). ph001/006 remain forbidden and legacy
004/005 excluded. Retain fixed cosmology/epoch/HOD and survey-selection limitations
from the preparation closeout. Pair coherence is not whole-survey coherence.

## Model, optimization and chronology

Reuse qualified CoupledBackbone, base24/three levels, full joint-local and spatial
wide observations in BOTH factors. Fine draws share one sampled wide coarse
field and a genuinely joint residual, not two independent parent draws.
Training uses true coarse conditioning; inference must use sampled coarse and
report the resulting conditional-distribution shift. Never evaluate oracle
coarse conditioning as the deployable posterior.

CFM linear independent-Gaussian bridge, projected block-zero-mean fine noise and
velocity; positive mass-conserving decoder unchanged. Batch2, AdamW1e-4 to1e-5
cosine decay over26624updates (32epochs), weight decay1e-5, clip0.5, EMA0.999.
Checkpoints832/1664/3328/6656/13312/26624; durable latest every256updates and at
segment deadline. Each epoch visits all1664pairs, deterministically reshuffled
per seed. Initial pilot uses zero context offset, explicitly no augmentation.
Do not equate update counts across toy and Abacus; report presentations/epochs.

Initial spectral loss preconditioner alpha=.30 is a provisional candidate, NOT
an Abacus-selected optimum. Estimate radial latent spectra from26training-only
pairs (two fixed geometry-selected pairs/phase); preserve DC, floor eigenvalues
at1e-3positive-median and normalize mean-square weights. Multiply Fourier velocity
error by P^-alpha. This is LOSS weighting, not a change of coordinates or noise
path; projected base noise remains unchanged. Positive fixed weights preserve
the population velocity-regression target, though capacity/optimization tradeoffs
change. Coarse and fine profiles are separate. Profile IDs and spectra are saved.
No target leakage from development or confirmation.

After first segment: report wall time, unique training exposure, loss/DC error,
memory and resumable steps. Do not call early training mature. Development-only
comparison of alpha in{.20,.30,.40} is a later bounded selection decision: choose
using regional-mass and marginal proper scores together, not best high-k power
alone; it is NOT bundled into this first resource approval. No silent switching
of alpha within a continuation. The current runner trains/checkpoints only;
posterior evaluation and alpha-selection orchestration remain follow-on work,
not falsely represented as already implemented by starting this job.

## Success criteria: what is justified and what is not

All scientific quantities use the registered common rectangular operator, owned
cores, fixed R7 target, and region definitions in the coupled resource proposal.
128draws give attainable central90% coverage117/129=90.6977% under the tested
order-statistic implementation; record this from code.
Do not count voxels, overlapping probes or seeds as independent universes.

| Quantity | Pre-recorded assessment | Evidence and qualification |
|---|---|---|
| Numerical correctness | finite draws; positive density; coarse-mass conservation; target-free inference; reproducible checkpoint/RNG | hard implementation requirements, not empirical calibration |
| Core/regional mass | aim within5percentage points of attainable coverage, phase90%CI compatible with reference; show per-seed and all50/68/90/95levels | practical target, NOT yet an empirically justified equivalence margin; distinguish bias from underdispersion |
| Density/eigenvalue/gap CRPS | lower is better; no claimed gain unless paired phase evidence supports it; >5%worsening flags regression | proper-score direction is justified;5%is a provisional practical margin, not measured downstream utility |
| Tidal classes | Brier score and calibration-in-the-large per class; fixed bins and phase-level uncertainty | one truth per observation does not give exact conditional probabilities; no invented probability-error oracle |
| Dependence | matched cross-core block-orthogonal probes; fair variogram and energy, with permutation controls at FIXED coarse field | energy improvement may be<1%even for an oracle; NO mandatory10%energy gain or automatic10%variogram gate |
| Spectral shape | sample/mean/residual decomposition in train-defined well-resolved bands;10%is a diagnostic target, not Nyquist perfection gate | assess downstream regional/tidal consequences; mask bands from train power/mode counts only |
| Stability | two consecutive saved checkpoints AND both seeds; sampler128/256NFE agreement relative to Monte Carlo uncertainty | unresolved drift is inconclusive; falling loss alone is not posterior convergence |

Coverage count precision must be shown before confirmation. Even under optimistic
independence,96Bernoulli trials near.907have SE~3percentage points and32trials
SE~5points. Six/two independent phases and overlapping regions can make this
worse. Therefore a requirement that the entire confidence interval fit within
±5points is NOT imposed; development cannot certify tight calibration. A wide
interval is inconclusive, not pass or failure. Report phase means, all six phase
values at confirmation, leave-one-phase-out sensitivity and phase-cluster intervals.
Do not shrink intervals by bootstrapping voxels. Two development phases guide
selection and catch gross errors; they cannot establish generalization precision.

Before selecting a scientific gate, the evaluation implementation must run its
finite-ensemble oracle and deliberately biased/underdispersed/decorrelated
controls with the ACTUAL panel/probes. Record null variability and smallest
detectable degradation. If a metric cannot resolve the desired effect, keep it
descriptive; do not launch more tuning to satisfy it. This control is required
before confirmation, not a reason to block exploratory training. No existing
empirical evidence certifies all proposed margins as science requirements.

Absorption/DC/per-shell reporting: on Abacus an exact posterior-mean error and
exact Gaussian absorption prediction are unavailable. Report truth-minus-ensemble
mean, squared-error/spread decomposition with finite-M correction, DC offset by
phase/selection, spectra and conditional ranks. Label any inferred absorption
mechanism a hypothesis, not the exact toy calculation. The first training log's
DC velocity error is NOT a posterior regional-mass calibration measurement.

## Decision and stop rules

1. First segment must pass technical checks and produce replayable checkpoints.
   Failure means fix implementation; no scientific conclusions from a crash.
2. Assess development draws after adequate exposure; if substantial improvement
   remains between checkpoints, report under-training and propose measured
   continuation, not architecture failure. First2h is not a maturity budget.
3. Freeze configuration, sampler, metrics and uncertainty/control receipts before
   confirmation prediction. A single final-checkpoint pass is insufficient.
4. If mature, numerically stable models still exclude true regional masses or
   show reproducible joint-score defects, do not promote. Diagnose the coarse
   posterior separately: fine coupling cannot repair block-aligned coarse masses.
   An analytic long-mode hybrid is a possible separately authorized fallback.
5. If scores/calibration are practically acceptable and stable but precision is
   inadequate, conclude encouraging/inconclusive rather than impossible or proven.
   Do not pursue mathematical perfection at irrelevant fine scales.

Old D and Wiener results are historical controls in a different coordinate/input
contract. Wiener was already fitted for galaxy bias, and its edge-supported
information and periodic closure differ. No claimed causal architecture win.
An information-matched baseline may be evaluated later at bounded cost, but is
not a prerequisite that delays this pilot or a new classical-method programme.
No four-node/day-scale claim, no710GPUh campaign, no production DESI release.
