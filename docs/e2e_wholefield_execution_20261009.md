# Whole-field campaign execution record

## Current outcome

Gate 2 rejects the registered sensitivity requirement; the campaign is stopped
before Gate 1, its cost benchmark or any training. See
[the verified result](e2e_wholefield_gate2_results_20261009.md).
Thirteen tests and 840 independent checks pass. The optimistic necessary-event
upper bounds at the 20% effect are 4.9–51.1%, all below the required 80% under the
declared synthetic laws. No claim of architecture failure or measured population
power follows. GPU usage is zero. CPU release requires separate user permission:
auto-review blocked exiting allocation 59602162; it remained RUNNING at 19:34 UTC,
with no scientific work active and scheduled expiry at 20:17:04 UTC.

## Authority and budget

Approved plan/scope/log commit: `6d3b993`. Sensitivity runs first. All gates must
be reported before advancement; an insufficient panel stops the campaign.
Execution begins with CPU allocation 59602162 at 2026-10-09 19:17:04 UTC.
The 14-day deadline is 2026-10-23 19:17:04 UTC. GPU ceiling is 32 allocated
hours, including at most 8 for Gate 1. GPU use so far: zero.

Allocation 59602162: interactive CPU, desi, nid200076, one hour maximum,
Scratch license, 64 requested logical CPUs (exclusive node allocation exposes
256). No other allocation was present at start. Gate 2 output root is
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/gate2`.
No new fields, training or sealed-phase access is needed for preparation.

## Gate 2 preparation

`configs/e2e_wholefield_gate2_v1.json` fixes the panel, physical coordinates,
32-draw budget, effect and power requirements before synthetic sensitivity runs.
The six-dimensional fine-sensitive pair vector consists, per core, of density
averaged in the fixed offset subcube [2:14]^3 and two orthonormal trace-free
contrasts of volume-mean ordered tidal eigenvalues. Identity scaling uses the
existing dimensionless physical units; no development-fitted whitening is used.
Full-core mass and all three regional eigenvalues remain separately recorded.
The offset is geometry-defined, not selected by residuals. Full-core masses
aligned to coarse blocks remain controls, not fine improvement targets.

Preparation verifies saved parent receipts and hashes on the compute node,
checks all 32 pair IDs across draw batches, and reconciles regional eigenvalues
and physical truth against the earlier Stage A2 summaries. Six focused tests
pass. The saved PREPARED.json and PANEL.npz bind exact inputs and coordinates.
Preparation took 3.9 seconds for the four caps plus summary/receipt writing.
This is not a sensitivity pass.

All-pair continuous ratios are 1.4687444108 (ph012) and 0.7665127685 (ph013).
Interior-only ratios are 1.4697373317 and 1.3046031679. Both strata are retained;
the primary all-pair panel is not replaced by the more favorable interior subset.
The class-fraction targets 2.5 to 1.5 are secondary and not these baselines.

Historical true-coarse saved products lack regional continuous tidal draws.
Do not relabel the local-oracle products as full-cap true-coarse evidence.
A necessary-condition screen on the available sampled-coarse panel may reject
the complete gate, but cannot pass it: full advancement implies the necessary
event. If a simultaneously valid upper confidence bound for that relaxed event's
detection probability is below 80%, additional stricter criteria cannot rescue
the complete rule under that synthetic law. Otherwise continue to the complete
rule and obtain any missing matched summaries within the approved budget.

## Necessary-event sensitivity screen frozen before simulation

`configs/e2e_wholefield_gate2_screen_v1.json` specifies a rejection-only screen.
For each phase, construct the whole-phase covariance of the six-coordinate
vectors from shared-address saved draws, with fixed 10% diagonal shrinkage.
This is a synthetic nuisance fixture, not a new trained normalizer, a replacement
physical truth or a model correction. Work in its orthogonal eigenbasis for
trace statistics; this retains the correlated quadratic-form distribution.

Generate 4,096 independent two-phase synthetic panels for each of four laws:
Gaussian scale error, bias plus scale error, a 25%-trace coherent common mode,
and elliptical Student-t noise with five degrees of freedom. Baseline expected
ratios equal the registered phase-specific continuous ratios. The 20% alternative
moves the population |R-1| 20% toward zero, not 20% of R itself. Additional
0/40/60/100% scenarios distinguish null behavior and stronger-effect sensitivity.
Pair posterior innovations at correlations 0, 0.9 and 1, with 0.8 correlation
between checkpoints of a configuration. Truths remain shared across every
candidate/checkpoint; phases are independent and spatial regions are not.

The optimistic necessary event allows either configuration at any checkpoint
to satisfy the ratio criterion in eligible phases. It drops the second seed,
true-coarse mode, neighboring-checkpoint stability, actual selector, proper
scores, coverage/marginal guardrails and MC significance. A phase whose fixed-
truth whole-draw MC interval includes R=1 is also dropped, because its prescribed
joint-dependence alternative can only make the full rule harder. If no ratio
phase is eligible this screen is inapplicable, not failed. Every full-rule pass
implies this optimistic event, so its probability is an upper bound on full-rule
detection probability under the stated law. An exact-moment version separately
shows the synthetic panel limit without finite posterior-draw noise.

Use simultaneous one-sided Hoeffding bounds, total MC error at most 1% across
all reported event probabilities. Reject Gate 2 if the optimistic upper bound
is below 80% at the registered effect in a declared law. The no-effect event
frequency is an upper bound, not a measurement of the complete rule's false-
advancement rate. This screen cannot pass Gate 2 or qualify any posterior;
absence of rejection requires completing the full rule and its <=5% null check.
Secondary class-fraction sensitivity cannot reverse a primary-gate rejection.
