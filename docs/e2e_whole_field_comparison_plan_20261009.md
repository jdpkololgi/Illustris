# Coupled CFM whole field comparison plan

Execution outcome, 9 October: **stopped at Gate 2 for insufficient sensitivity**
under the registered synthetic tests. No Gate 1 campaign, cost pilot or training
was run; zero GPU-hours used. The contract below is retained, not relaxed.
See [the verified gate result](e2e_wholefield_gate2_results_20261009.md).

Status: execution approved on 9 October 2026. Commit this plan, the P12-A scope
note and science log before compute. Run Gate 2 first or in parallel with Gate 1;
stop if the panel cannot detect the registered effect. Measure Gate 1's full cost
before its campaign and pack several small fits per GPU. Report each gate before
proceeding. The approved ceiling is 14 days, 32 allocated GPU-hours total and
8 for Gate 1, interactive salloc/srun only, at most two allocations. Existing
frozen artifacts remain unchanged; sealed phases and follow-on coarse training
remain outside this execution authorization.

## Decision and scope

Use a gated sequence: real-panel sensitivity and Gaussian comparison, then
a fine-only Abacus pilot with at most two surviving configurations. Whole-field
coarse training and full-chain testing are a separate follow-on, not initial
fits. The candidate families are whole-domain U-Net and transformer generation,
each with direct velocity and clean-field prediction. The central question is whether
learning a joint velocity over the generated domain improves regional and
cross-region posterior uncertainty, rather than only cellwise predictions.
The Gaussian comparison is a mandatory screening gate, not Abacus qualification.
This is an exploratory comparison for useful improvements, not a promise or
requirement to deliver a production-qualified calibrated posterior. Failed or
inconclusive calibration results remain informative and must remain visible.

Initially preserve the coarse/fine factorization, R7 physical target, resolution,
observation selection, latent transforms, projected fine noise, positive
mass-conserving decoder and downstream density-to-tidal operator. Whole field
means one evolving state and one jointly evaluated velocity over each factor's
declared domain, not an average of independently learned patch velocities.
Transformer tokens are not independently sampled spatial patches.

The eventual target is the full padded cap domain. First make a matched finite
domain comparison spanning multiple original generation windows, with identical
observation information and identical central science regions across arms.
This is explicitly a whole-domain pilot, not whole-survey qualification. NGC
and SGC must not be called a joint posterior merely because each is generated
coherently. Shared source-volume identities and any cross-cap assumptions remain
part of the inference contract.

A single-stage density generator removing the coarse/fine split changes another
factor and is deferred. Likewise, higher spatial resolution, new cosmology/HOD
coverage and survey observation resampling are not bundled into this experiment.

The 8 October true-coarse result motivates prioritizing the fine stage: the
regional fraction defect persists despite true coarse conditioning. It does
not establish that the fine network alone causes the defect, or that coarse
inference is correct. Assembly, conditional learning, exterior information and
target/operator mismatch remain possible contributors.

## Candidate configurations

| Arm | Training and sampling | Purpose |
| --- | --- | --- |
| F | Existing frozen coupled CFM, existing canvas and numerical settings | Reused reference, no training |
| W-v | Whole-domain U-Net, direct velocity prediction | Gaussian candidate; Abacus fine fit only if selected |
| W-x | Whole-domain U-Net, clean-field prediction converted to velocity | Gaussian candidate; Abacus fine fit only if selected |
| T-v | Whole-domain transformer, direct velocity prediction | Gaussian candidate; Abacus fine fit only if selected |
| T-x | Whole-domain transformer, clean-field prediction converted to velocity | Gaussian candidate; Abacus fine fit only if selected |

F starts from the current coarse-13312/fine-26624 EMA baseline, with model,
normalizer, source and sampler hashes resolved in the preflight. Preserve seeds
17 and 29 where the corresponding references exist; disclose incomplete panels.
Do not silently substitute an augmentation continuation or historical operator.
Reuse existing reference results wherever galaxies/regions, targets, conditions,
operators and evaluation settings match. Additional frozen-model draws are only
needed for missing matched comparisons, not an automatic repeat of old work.

The full two-by-two backbone/parameterization comparison happens on the Gaussian
testbed. Abacus receives only one or two selected configurations, each with two
fine-model seeds; it is not necessarily a complete factorial comparison and
cannot then estimate every backbone/parameterization interaction. Those fits
use fresh initialization, the same training phases, geometry-selected domains,
target-voxel exposure and observation information. Match effective
batch and parameter scale where feasible; separately report wall time, FLOPs,
peak memory, target volume presentations and actual optimizer steps. No claim
that these budgets can all be equal simultaneously.
Comparison with F answers whether a tested configuration improves on the existing
reference. Without a newly trained Abacus patch control, it cannot uniquely attribute
improvement to removal of patch assembly rather than training or preparation
differences. Abacus patch retraining is excluded, not deferred as an automatic
requirement. A matched Gaussian patch-canvas control is included below. The
initial Abacus stage has two to four fine-model fits, not sixteen factor fits.

An optional cheap F whole-array replay can precede fitting if its interfaces
permit it. Larger-domain inference with patch-trained weights changes their
input distribution; failure cannot reject whole-field learning. A transformer
cannot inherit the frozen U-Net weights by a sampler switch.

## Gate 1 Gaussian comparison

Run the four candidates plus one matched direct-velocity U-Net patch-canvas arm
on the existing side-16 correlated 3D Gaussian problem. The patch arm must use
the actual overlap-averaged canvas, not the previously tested exterior-conditioned
block Markov chain. Retain the exact whole-domain and exact patch-oracle controls.
The latter isolate assembly error without neural approximation; the learned
comparison measures whole-domain versus patch training and assembly together.

Use three training seeds and checkpoints at 4,096, 16,384 and 65,536 updates,
with a 32,768 checkpoint for the adjacent-checkpoint stability check. Continue
each trajectory rather than fitting three independent budgets: five arms times
three seeds means fifteen fits. Side 8 is an implementation/parity check, not
another full training grid. Match prior/likelihood draws, all-observation
information, time law, parameter scale and supervised-voxel exposure. A patch
update must aggregate sufficient locations to match full-field supervision;
report its extra computation rather than repeating the earlier eightfold voxel
exposure mismatch. Freeze patch size, stride, overlap weights and boundaries.

Use the existing observations only for development. Register 16 fresh independent
observation/truth realizations, balanced across the two existing mask templates,
for the locked Gaussian gate. Select the checkpoint budget using development
data before opening that panel; retain the complete predeclared learning curves
and every seed, with no retraining or threshold adjustment after gate inspection.
Freeze the selector, probe family, multiplicity treatment and numerical margins
before fitting. Exact conditional means/covariances are evaluation-only, never
training teachers or conditioning inputs.

Evaluate normalized mean error, covariance and cross-region errors, interval
content, joint-summary ranks and proper scores against the exact conditional
law. Validate the finite-draw audit using exact independent posterior draws;
failure to reject is not a posterior certificate. Keep the corrected Gaussian-null
audit, not the earlier invalid four-standard-error screen. Refine NFE with paired
noise until discrepancies are stable relative to measured Monte Carlo precision.
Check the constrained Gaussian fine subspace and clean-endpoint conversion
numerically before any Abacus promotion; this does not add another training grid.

An arm advances only if, in all three seeds, it either satisfies the registered
absolute accuracy/equivalence criteria or meets a predeclared relative-improvement
gate: at least 20% reduction against the matched learned canvas in both worst-probe
mean error and probe-covariance discrepancy, resolved beyond Monte Carlo error,
without material interval-content or proper-score regression. Require stable
development performance at the selected and adjacent checkpoint and resolved
integration error. The 20% margin is a proposed engineering screen, not a changed
posterior-qualification tolerance. Report failures on individual observations;
do not hide them behind pooled scores. Rank eligible arms by the frozen joint
score and retain at most two; do not force one of each backbone through the gate.

A relative-only pass licenses a bounded exploratory Abacus test, not a claim of
Gaussian calibration. If none passes within the cap, stop this campaign. Gaussian
failure is a resource-prioritization decision, not proof of Abacus impossibility.
The enlarged panel, draws and refinement may dominate training cost: "minutes"
or "a few GPU-hours" is not a measured estimate for this complete design.

## Gate 2 Real-panel sensitivity before Abacus training

Run this gate first, or in parallel with Gate 1, using saved summaries and
receipts wherever sufficient. A failed sensitivity gate stops the campaign.
Freeze the actual comparison geometry: currently two phases with 16 region pairs
per phase, including only eight interior pairs per phase in the published interior
ratio. If the multiwindow pilot changes support, re-evaluate sensitivity on that
exact subset before fitting. Do not count all 32 pairs as interior or independent.

Validate the complete decision rule on calibrated, biased, underdispersed and
decorrelated synthetic ensembles at the intended draw count and panel geometry.
Preserve within-phase, cross-region and shared-draw dependence; shuffling regions
independently is a deliberate negative control, not a bootstrap null. Include
heavy-tail/sparse-support and common-mode perturbations, and MC uncertainty in
the sensitivity estimate. Predeclare at least 80% detection probability at at
most 5% false advancement under the tested synthetic nulls for the practical
primary effect; calibrate the complete rule, including checkpoint selection and
multiple candidates/metrics, not each score separately.
These are conditional design checks, not a power guarantee across cosmologies.

The illustrative ratio change 2.5 to 1.5 belongs to the historical class-fraction
statistic, not a measured continuous-summary baseline. Test that scenario as a
secondary sensitivity check, alongside phase-specific starts 2.6958 and 2.1147.
For the continuous primary vector defined below, freeze its own baseline and
effect size from existing development summaries before any new fit. Use a proposed
20% reduction in absolute deviation of its error/variance ratio from one, plus
improvement in the registered joint score, rather than imposing a fraction-derived
target. If that effect is not detectable, stop or obtain approval for a revised
panel; more posterior draws can reduce MC error but cannot create new phases.
Sensitivity established only for class fractions is insufficient to launch.

## Stage 3 Fine-only Abacus pilot

Train only the fine factors of the one or two Gaussian survivors, with seeds
17 and 29. Keep the frozen coarse model and original fine-training conditioning
law unchanged; do not bundle a conditioning-augmentation intervention. Evaluate
each fine checkpoint under both true coarse conditioning and a shared, hashed
set of sampled coarse endpoints from the frozen baseline, with the same coarse
draw identities across candidate and reference. These are two evaluation modes,
not twice as many fits. Match the frozen fine reference separately in each mode.

True coarse is an oracle conditional diagnostic; sampled coarse is the deployable
composition test. Improvement only under true coarse is not enough to advance.
Both modes must improve the fine-sensitive primary summaries in both development
phases and both seeds, with stable neighboring checkpoints and no material
marginal regression. Test frozen-coarse compatibility explicitly rather than
assuming a new fine generator tolerates its errors.

The positive mass-conserving decoder fixes total mass within each coarse cell.
Changing only fine generation cannot repair uncertainty of regions that are exact
unions of those cells. Retain those regional masses as conservation/coarse-stage
controls, not fine-stage improvement gates; under true coarse their uncertainty
may be identically zero. Use predeclared sub-coarse/non-aligned regions and tidal
or cross-region summaries with genuine fine sensitivity for the pilot's primary
comparison. Never divide by structurally zero variance.

Only reproducible success here justifies proposing whole-domain coarse training
and a full-chain experiment. That is a separately scoped and approved stage with
its own measured budget, not remaining work automatically owed by this pilot.

## Transformer and prediction parameterization

[Remy et al.](https://arxiv.org/html/2606.31988#S3.SS3) provide a useful design:
field and observation tokens interact through global attention, with time
conditioning. Their network predicts a clean endpoint and converts it to a
velocity for a conditional flow-matching loss and ODE sampler. This remains
CFM; it is not direct deterministic regression replacing posterior generation.

For a bridge z_t=(1-t)epsilon+t*x, direct velocity prediction fits x-epsilon.
Clean-endpoint prediction d_theta instead defines
v_theta=(d_theta-z_t)/(1-t), with an explicitly controlled endpoint denominator.
The induced objective is not ordinary unweighted clean-field MSE.

W-v/T-v retain the current direct velocity target. W-x/T-x test clean-field
prediction from the outset, with the same underlying flow-matching objective.
Adapt the paper's field/condition token idea to physical 3D positions and the
actual observation channels. Keep the transformer near the U-Net parameter
scale; do not copy a large model indiscriminately. No cosmology-output tokens
are learned from this fixed-cosmology dataset. All masked-domain target tokens
remain where the physical target requires unobserved matter; observation holes
are not zero-density constraints. No target information enters conditioning.

Freeze common time sampling, loss weighting and optimizer policy for the initial
two-by-two comparison; do not simultaneously import a new complete training
recipe. Specify and test the clean-prediction endpoint denominator/clipping and
its interaction with the constrained fine latent before fitting. This is not
ordinary unweighted clean-field MSE. Record bottleneck/patch information loss,
residual paths and numerical endpoint behavior as part of each architecture.

## Data scarcity and overfitting

The current scientific record identifies 21 prepared coupled phases: 13 training,
012/013 development, 014--017 exposed replication, and 018/019 fresh confirmation.
The October 2 production inventory is a historical planned task list, not proof
that later records' sealed phases are available for tuning. Audit actual exposure
before any future launch. Preserve 001 and existing legacy access restrictions;
004/005/006 are not fresh independent confirmation or automatic new training data.

Multiple crops, caps, periodic views, noise draws and training seeds are not
independent simulation phases. Full volumes contain useful local-mode diversity,
but more voxels do not provide more independent long-mode realizations. A smaller
number of simulations is a reason for stronger generalization controls, not a
reason to assume a transformer will resist memorization.

For the fine-only Abacus pilot:

1. Construct and freeze phase/source-box groups before crop selection. Prevent
   overlap and periodic replicas from crossing roles. Fit all normalizations and
   spectral profiles on training phases only.
2. Use common finite domains and phase-balanced training. Define one epoch as a
   pass over that common domain inventory, with identical voxel weighting across
   selected configurations. Save learning curves at 4, 8, 16 and 32 epochs,
   subject to an approved measured resource cap.
   No automatic continuation beyond the cap; compare exposure, not raw steps.
3. Select on 012/013 using a frozen proper-score criterion with calibration
   guardrails; report every checkpoint and both seeds. Require stability at
   adjacent checkpoints rather than picking an isolated favorable fluctuation.
   Successful pilot replication on 014--017 is a separately scoped follow-on,
   acknowledging prior exposure; it is not a rescue search after pilot failure.
4. Track training versus held-out loss, posterior mean error, spread and coverage
   separately. Improving training loss with worsening held-out scores or shrinking
   uncertainty is a stop signal for that continuation, not motivation to keep
   fitting. Compare two seed trajectories; neither is another universe.
5. Use modest capacity, weight decay and EMA with an equal, bounded tuning policy
   for both families. Accept only joint geometric transformations preserving
   observer coordinates, line of sight, masks and physical targets. Do not import
   image rotations or independently perturb galaxy counts without validating the
   observation law. Fresh bridge noise does not supply new cosmological structure.
6. If generalization remains ambiguous at the cap, park the campaign. A later
   training-phase learning curve or larger model requires an explicit reopening,
   not an automatic extension of this experiment.

Opening 018/019 is a final, separately approved comparison after model selection
and evaluation are frozen. Two fresh phases have limited statistical power;
voxels, regions and posterior draws cannot turn them into precise population
calibration evidence. Fixed cosmology, epoch and HOD remain scientific limits.

## Feasibility and numerical gates

Before Abacus fitting, audit actual cap cuboids, padding, data availability and the
common observation channels. Choose the pilot domain from training-only geometry
and measured forward/backward cost, not favorable scientific scores. It must
contain several generation windows; the current single joint rectangle already
has joint fine generation and alone cannot test removal of multiwindow assembly.

Measure the selected architectures' training and sampling at the real tensor
shapes, including conditioning and optimizer state. Attention token count includes
condition tokens; dense attention has quadratic pair count. Memory-efficient
attention can avoid storing that matrix without removing its quadratic compute.
Do not extrapolate the 16-cubed Gaussian timing to a survey cap. If cap-scale
execution needs approximations, freeze and label them as different architectures;
windowed attention or independent block sampling is not silently whole attention.

Test target-free inference, coordinate and mask parity, positivity, fine
projection, coarse-mass conservation, checkpoint/RNG replay and declared boundary
conditions. Exact kernel chunking is acceptable only with whole-evaluation parity;
it is distinct from assembling learned patch marginals. Do not assume a finite
CutSky cuboid is periodic. Keep the corrected density-only tidal analysis and its
physical-truth closure fixed across arms; retain original-operator sensitivity.

## Evaluation and decision

Register aligned draws, NFE refinement, region weights, primary metrics and
practical margins before fitting. Keep established absolute benchmark gates
visible; relative pilot advancement is a separate decision, not a relaxation of
posterior qualification. Gates 1 and 2 must be resolved before Abacus fitting.

Primary outcomes are continuous regional density/mass, regional tidal eigenvalues
and joint uncertainty of a fixed low-dimensional vector of regional summaries,
including cross-region contrasts. For the fine-only gate use the fine-sensitive
subset specified above. Fix region weights from geometry, not galaxy counts or
observed residuals. Define regional eigenvalues as volume means of the ordered
voxel eigenvalues; eigenvalues of a mean tensor are a distinct, separately labeled
statistic. Freeze vector coordinates, train-only scales/whitening, regularization
and a TARP-style random-reference rank/coverage procedure with validated finite-draw
and dependence-aware nulls. Two phases cannot establish population joint calibration;
these remain finite-panel diagnostics with explicitly limited power.

For the fixed scaled continuous vector s, report R = tr(Q)/tr(V), where
Q averages (posterior-mean minus truth) outer products minus V/M and V is the
sample predictive covariance at M draws. Use the same registered region weights
for both, retain bias and variance separately, and do not clip negative MC-adjusted
errors. Report distance from one, joint ranks and fair energy/variogram scores
together: covariance inflation alone must not earn advancement. At the selected
checkpoint require the predeclared 20% reduction in |R-1| and a resolved joint
score improvement in every phase/seed and both conditioning modes, with no material
coverage or marginal-score regression. Freeze score margins in Gate 2, before
training. Stability at a neighboring checkpoint is required, not a new chance to
select after failure. If baseline R is already indistinguishable from one, this
ratio is not an improvement target; Gate 2 must instead register a detectable
joint-dependence effect and its decision rule, or stop as uninformative.

Galaxy-weighted and volume-weighted class fractions are secondary. Retain the
entire fixed threshold sweep 0, 0.1, 0.15, 0.2, 0.25, 0.3 and 0.4, all original
regions, support counts and leverage diagnostics; no threshold or sparse-core
exclusion may be selected for better scores. New products must save enough
eigenvalue information to support that sweep. Historical true-coarse products
with only threshold-0.2 labels cannot supply an invented full sweep; label missing
baseline entries or budget additional frozen evaluation explicitly.

Also retain marginal density, ordered eigenvalue/gap CRPS, Brier, 50/68/90/95
percent rank curves and mean/sample/residual power. Score all spatial strata.
Cellwise R-squared or attractive images are insufficient. One realized truth per
observation is not an exact conditional posterior mean or probability oracle.

Report paired phase-level differences and every phase, with posterior Monte Carlo
uncertainty separately, without treating cores, seeds or draws as independent
phases. The pre-training sensitivity gate controls interpretation. Increase NFE
within the fixed cap until the tested observables are stable relative to Monte
Carlo precision; otherwise label solver
and learning error unresolved. State how many draws are needed from measured
precision, then freeze that count and its compute budget before scoring.

Advance from the pilot only on the registered regional/joint rule above; later
broader claims require exposed-phase replication and separately approved fresh
confirmation. A whole-domain improvement supports that domain and target only.
All-arm failure
does not prove architecture impossibility; useful errors may instead concern
optimization, data diversity, likelihood mismatch, resolution or exterior matter.

Per-galaxy P12-A comparison is outside this campaign; its estimand and evidence
contract are in [a separate note](e2e_cfm_p12a_comparison_scope_20261009.md).

## Sequence and resource approval

1. Commit the approved documents, verify Scratch access and frozen Abacus
   input/phase manifests, and run Gate 2 first or alongside Gate 1. Report its
   result; stop if the intended panel cannot detect the registered effect.
2. Freeze the Gaussian implementation, selector and gate; measure representative
   training AND sampling cost, including packing multiple small fits on each GPU.
   Run Gate 1 only within its approved subcap and report before moving on.
3. Measure survivor feasibility at actual fine-domain shapes, bind F, freeze
   domains, checkpoints, margins, draw counts, runtime/storage and source hashes.
   If the complete two-seed comparison cannot fit the remaining cap, stop before
   training rather than silently dropping a seed or primary diagnostic.
4. Only after both gate reports and passes, run two to four fine fits, evaluate both
   conditioning modes and apply the frozen stop/advance rule.
5. Only success warrants a new proposal for coarse training, full-chain testing
   and exposed-phase replication. Fresh 018/019 confirmation remains separately
   approved after selection is frozen. No VAC, DESI, website or default changes.

Approved hard envelope: 14 calendar days from the first authorized compute start,
32 aggregate allocated GPU-hours for Gates 1--2 and the fine pilot combined,
including feasibility, training, posterior draws, refinement, controls and failed
attempts. Gate 1 has an 8 GPU-hour subcap included in that total. These are spending
ceilings, not timing predictions. Execution within these bounds is authorized.
Measure first; if the complete design exceeds them, report it and request an
explicit scope/budget decision. Count GPU count times allocated elapsed time,
not only optimizer time. Record actual CPU hours and storage, and freeze their
allocation/storage limits in the execution request as well.

Stop at the first exhausted cap, failed prerequisite, numerical/physical invalidity
or missing reproducible primary improvement. If no survivor improves in both
phases and both seeds under both coarse modes, park this field-model campaign
and retain the result as negative or inconclusive according to its sensitivity.
Unresolved numerical error, insufficient panel sensitivity or incomplete work
is inconclusive, not evidence of architecture failure. There is no automatic
extension, rescue fit, new seed, added training phase or threshold search. A
Scratch outage does not silently reset the clock; any extension needs approval.

This is bounded research, so use explicitly approved interactive salloc/srun
only, with at most two pending/running allocations. Remove the former automatic
sbatch handoff: a later production submission would require a separate user
decision, and is not part of this plan. The user's execution approval authorizes
the bounded interactive allocations and steps, not unrelated scheduler changes.

Scratch reservation blocked work earlier on 9 October; that is historical
evidence, not a live availability check. Before any Scratch-backed stage verify
current scheduler/filesystem status, required Scratch license, input readability
and output capacity in the approved environment. Stop if unavailable; do not
omit the license, bypass the reservation or move Abacus payloads into Home.
The Gaussian stage can use the previously demonstrated small Scratch-independent
runtime/output design only with an explicitly approved bounded exception; it
does not establish Abacus feasibility. Preserve cosmic_env and shared software.

## Why the control was delayed

Whole-volume generation was discussed on 19 September in
`e2e_inference_strategy_decision_20260919.md`. The same-day SCIENCE_LOG correction
rejects its one-GPU survey-feasibility claim: it extrapolated patch measurements
without padded geometry, global attention and other required costs. Those earlier
figures must not be revived as measured feasibility.

The development path emphasized bounded coarse/fine rectangles, wider observation
context and later shared-state canvas construction. Those were reasonable
engineering choices, but numerical coherence is not posterior-law correctness.
The exact-canvas counterexample on 8 October makes that distinction concrete.
The missing test was an early, small, information-matched comparison of patch
assembly with a genuinely joint generator, not the mere idea of whole fields.
It should have been prioritized earlier; historical costs did not justify
omitting the small control.

Supporting local records: `e2e_cfm_abacus_pilot_20260924.md`,
`e2e_threshold_canvas_results_20261008.md`,
`e2e_matched_gaussian_3d_results_20261009.md`,
`e2e_gaussian_transformer_results_20261009.md`, and current `SCIENCE_LOG.md`.
