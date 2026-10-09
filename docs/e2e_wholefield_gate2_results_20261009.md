# Whole-field Gate 2 sensitivity result

## Decision

Stop the approved whole-field campaign at Gate 2. The registered two-phase
panel and advancement rule do not achieve 80% detection probability for a 20%
reduction in |R-1| under the declared synthetic laws. Even an optimistic necessary
event has simultaneous upper bounds between 4.9% and 51.1%. Adding the omitted
requirements cannot increase the full rule's probability of success.

This is a failed sensitivity prerequisite, not an architecture result, an Abacus
population-power measurement or proof that whole-field generation cannot help.
No Gaussian campaign, Gate 1 cost pilot, Abacus fit or new field draw was run.
No GPU was allocated. The user's instruction to stop on insufficient sensitivity
takes precedence over the later campaign stages. Restart requires a new decision;
the registered gate was not relaxed after inspection.

## Panel and estimand

The source is the complete frozen parent development panel: ph012/ph013, 16
pairs per phase, two cores per pair and 32 shared-address draws. Each phase has
eight interior and eight boundary pairs. All are retained, with equal pair/core
weight and the corrected density-only tidal operator against physical full-box
R7 truth. No held-out phase payloads were read.

Each core contributes density in the fixed [2:14]^3 subcube and two orthonormal
trace-free contrasts of the volume-mean ordered tidal eigenvalues. Both cores
give six fine-sensitive coordinates per pair. Coordinates use dimensionless
physical units with identity scaling, not development-fitted whitening.
The 32 registered core starts were checked to be divisible by four on the fine
grid; the offset subcube therefore cuts coarse cells. Full-core density and all
three mean eigenvalues are retained separately; coarse-aligned mass is a control.

| Phase | Continuous R, all pairs | Fixed-truth 95% draw-MC range | Interior R, descriptive |
| --- | ---: | ---: | ---: |
| ph012 | 1.46874 | 1.32975–1.78530 | 1.46974 |
| ph013 | 0.76651 | 0.69090–0.98635 | 1.30460 |

R is the trace of squared mean error minus finite-draw mean noise, divided by
predictive covariance trace. It is not coverage. The ranges resample whole draw
addresses, preserving spatial dependence; they are not population confidence
intervals. The class-fraction ratios near 2.5 are a different estimand.

## Rejection-only test

The configuration and source were committed before simulation (`b06670e`).
There are 4,096 independent synthetic two-phase panels for each combination of
four error laws and three paired-innovation correlations. Whole-phase covariance
comes from the saved draws with fixed 10% diagonal shrinkage; this covariance is
only a nuisance fixture for the synthetic experiment. Truths are shared across
all candidate configurations/checkpoints within a trial. Regions, draws and
checkpoints are not treated as independent phase realizations.

The tested alternative moves the population |R-1| 20% toward zero. The optimistic
event lets either of two configurations at any of four checkpoints achieve the
observed 20% ratio improvement in both phases. It omits the second training seed,
true-coarse evaluation, adjacent-checkpoint stability, actual checkpoint selector,
proper-score improvement, marginal/coverage guardrails and MC significance.
Neither phase needed the baseline-near-one alternative in this run.

Every pass of the full registered rule implies this more permissive event.
Consequently, an upper bound below 80% for that event rejects the full rule's
80% requirement under the same law. This logic can reject but never pass Gate 2.

The table gives simultaneous 99% **upper** bounds on the optimistic event's
probability at the registered effect. Each column specifies correlation of paired
posterior innovations, not correlation of independent universes.

| Synthetic law | Correlation 0 | Correlation 0.9 | Correlation 1 |
| --- | ---: | ---: | ---: |
| Gaussian scale error | 44.72% | 28.14% | 10.44% |
| Bias plus scale error | 51.14% | 42.69% | 16.40% |
| Added coherent common mode | 41.84% | 26.97% | 8.83% |
| Elliptical Student-t, five degrees of freedom | 22.26% | 10.56% | 4.90% |

All 12 settings reject the sensitivity requirement. Hoeffding bounds use a union
bound over 120 reported upper probabilities, with total upper-bound MC error at
most 1%. Lower bounds in the receipt are descriptive, not part of this simultaneous
upper-bound claim. Independent exact-binomial bounds confirm every rejection.

The fixture covariance effective ranks are 13.70 and 14.19, or 8.30 and 8.89 with
the added common mode. These are covariance diagnostics, not counts of independent
cosmological phases. Additional 0/40/60/100% effect scenarios and exact-posterior-
moment versions are retained in the receipts. At the 20% effect, the exact-moment
event also falls far below 80%; finite posterior-draw noise is not the only issue.

## Interpretation and limits

The acceptance threshold and the alternative used to demand 80% power were both
set at 20%. That boundary creates a demanding operating point even before asking
for success in both phases, both seeds and both conditioning modes. More generally,
a realized two-phase error/variance ratio can move away from one even when the
underlying posterior improves. The fixed-panel ratio is not a noiseless measure
of population calibration.

The result is conditional on declared summary-level Gaussian/bias/common-mode/t
surrogates and covariance estimates from 32 saved draws. These are not fresh
Abacus simulations, a cosmology/HOD robustness study or a complete field-generative
model. The bias/scale scenario and common-mode choices are documented interventions,
not fitted explanations of the original failure.

The complete conjunction of scores and its <=5% false-advancement calibration
was not simulated: the necessary-condition bound already rejects it. Likewise,
true-coarse continuous tidal extraction, calibrated/decorrelated full-score
controls, secondary class-fraction power and Gate 1 benchmarking were not run
after this stop. Their absence cannot turn the failed necessary condition into
a pass. The loose screen's no-effect frequencies are not the full rule's false-
positive rate. No posterior, full gate, or architecture has been qualified.

A future proposal should distinguish the minimum accepted effect from the
larger effect at which high power is required, and revisit the panel-level
decision rule and independent validation support. Do not change those choices
silently or spend additional training compute under this stopped campaign.

## Verification and provenance

- Approved plan/scope/log commit: `6d3b993`.
- Tested panel extraction and fixed coordinates: `6cb540f`.
- Frozen power-screen source/configuration: `b06670e`.
- Thirteen focused tests pass. Independent verification performs 840 checks,
  re-extracts every continuous summary from the original saved cores, verifies
  provenance hashes, recounts every event, and checks the probability bounds.
- Synthetic screen: 11.10 seconds on CPU allocation 59602162; scheduler step .3
  completed 0:0 in 16 seconds including startup/tests. Verification step .4
  completed 0:0 in eight seconds.
- Total campaign GPU use: **0 allocated GPU-hours**. The single CPU allocation
  completed 0:0 at 19:47:18 UTC after 30m14s (0.5039 CPU-node-hours), including
  setup and idle time. Manual release was blocked by auto-review pending separate
  permission; the idle shell then logged out automatically and relinquished the
  allocation. No agent-issued exit/cancellation bypass occurred. All eight steps
  completed 0:0, and the final allocation inventory is empty.
- Required graph refresh was attempted in cosmic_env but failed at the filesystem
  lock with errno 524. No bypass was attempted; scientific verification passed.

Receipts under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/gate2`:

- `PREPARED.json`, `PANEL.npz`, `screen_step.log`;
- `necessary_screen_v1/BINDING.json`, `SIMULATIONS.npz`, `COMPLETE.json`,
  `VERIFIED.json`.

COMPLETE SHA256: `15ec566f7a5ee5323ec6bb979eb00fa2b8e263171bf330ba48469dfdc034e026`.
PREPARED SHA256: `48a352cbaa600f2753f30a0ce0cc5b434ddf599de9a718728922c9bfccfc6631`.
The saved products and frozen model artifacts are unchanged. The execution ledger
is [e2e_wholefield_execution_20261009.md](e2e_wholefield_execution_20261009.md).
