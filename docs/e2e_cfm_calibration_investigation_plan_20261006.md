# CFM calibration investigation plan

## Decision and authority

Continue towards an informative, approximately calibrated density-field posterior
for a specified DESI BGS selection. A restricted, mock-validated domain is an
acceptable first success. Finding genuinely modest property–environment
relationships is also success; stronger contrasts are not an acceptance criterion.

The user requested this working protocol on 6 October 2026. This is documentation
authority only: no new allocation, submission, cancellation, training,
confirmation access or production replacement. Resource envelopes below are
proposals for subsequent execution approval, not extensions of earlier permissions.

P12-A writing and submission by end-2026 remain independent of CFM success.
Japan travel is 21 October–12 November, with a talk on Monday 26 October and work
continuing during the trip. These dates set review points, not calibration deadlines.

Three questions guide the investigation:

1. Where is uncertainty wrong in the coarse model, conditional fine model and
   composed field posterior?
2. Does that failure affect galaxy class probabilities, regional class fractions
   and a defined property–environment estimator?
3. Does one defensible intervention improve uncertainty against a matched control
   without sacrificing useful information or physical coherence?

## Starting evidence

The controlling record is [SCIENCE_LOG.md](../SCIENCE_LOG.md). These are existing
development results, not findings of this new investigation.

| Evidence | Meaning for the next stage |
| --- | --- |
| Positive central-linear reconstruction reduced true-density operator width deficit from 3.871% to 1.366% across 64 cores; eight-core posterior replay improved from 5.885% to 3.463%. | Promising operator candidate, not an all-core posterior repair or production promotion. |
| Common-operator canvas coarse/fine shear RMS ratios were 0.9783/0.9271; oracle-coarse fine RMS was 0.9931. | Near-correct amplitudes do not establish calibration or identify the sole defective factor. |
| Nominal-90% four-gap joint coverage was 68.75% in each phase for canvas and both local variants; oracle conditioning gave 81.25%/75%. | Conditional and composed uncertainty remain unresolved. Dependent cases are not binomial trials. |
| Equal-size conditioning perturbations moved fine shear in opposite directions; within-pair coarse-distance/fine-shear correlations were +0.0636/+0.0107. | Scalar error amplification is unconfirmed; sensitivity alone does not justify augmentation. |

Sources: [density and sampler controls](e2e_density_sampler_diagnostics_20261005.md),
[conditioning interventions](e2e_conditioning_ablation_20261005.md),
[exterior campaign](e2e_exterior_campaign_20261002.md), and
[earlier calibration panel](e2e_global_calibration_20261001.md).

## Fixed contract

**Physical target.** Preserve R7-smoothed matter density, the potential-Hessian
tidal definition, the existing distance convention and 6.766 Mpc/h cap cells.
Do not smooth a second time. Ordered eigenvalues satisfy lambda1 <= lambda2 <=
lambda3. At primary threshold 0.2, count eigenvalues strictly above threshold:
0 void, 1 wall, 2 filament, 3 knot. Threshold 0 is a fixed sensitivity check,
not an alternative selected for better scores. These are tidal classes, not
thin filament spines or bound-cluster membership.

**Posterior object.** Each operational draw is a positive density field with its
declared coarse block masses. Tensors, eigenvalues and classes must follow from
that same draw. Independent probability corrections or regional covariance
inflation do not by themselves qualify a field posterior. Preserve the current
independent-cap limitation: this is not a demonstrated joint NGC–SGC posterior.

**Baseline.** Bind seed-17 coarse-13312/fine-26624 EMA weights, train-only
normalization, Heun NFE128, 32 draw addresses, shared-state sampler, learned
exterior law, padding and operator hashes to the existing freeze. Retain the
original operator beside any correction. Local sampler-factor comparisons use
their common local-plus-wide operator; their numbers cannot be combined with
full-cap free-space results as an additive error budget.

**Phase discipline.** Initially read ph012/ph013 only: all 32 registered pairs,
64 cores and original cap/shell/support geometry. There are two independent
phase realizations. Ph014–017 are exposed, not blind; legacy004–006 have separate
exposure/compatibility qualifications. Later use requires a replication amendment.
Ph001 remains excluded and ph018/ph019 sealed, including prediction/truth payloads.

**Domain.** Freeze shell edges and supported galaxy selection from bound input
manifests before analysis. The existing four-shell mock panel spans z=0.15–0.55,
not the full observational BGS range. Retain each shell, cap and existing
interior/boundary stratum. Additional one-dimensional count/support splits use
training-derived boundaries, not evaluation residuals. A narrower domain may be
selected in development but exclusions stay visible and are frozen before
confirmation. Wider high-z posteriors are not inherently miscalibrated.

## Stage A Locate the uncertainty failure

### A0 Verify inputs and measurements

Inventory saved regional draws, fields, tensors, truths and galaxy-coordinate
products through receipts before large-array reads. Record missing products;
reuse saved summaries where sufficient. Do not rebuild native matter FFTs or
resample full caps merely for convenience.

Regional means of voxel eigenvalues differ from eigenvalues of a mean tensor.
For oriented shear use S = T - trace(T) I/3, not eigengaps as if they were its
five components. Keep physical full-box truth and operator-matched truth separate.
The latter diagnoses error but never replaces physical truth for qualification.

Before interpretation, validate calibrated, biased, underdispersed and wrong-
dependence synthetic ensembles. Exchangeable oracle truth/draws must reproduce
the exact finite-M scoring null. Shuffling draw IDs across regions must preserve
marginals while degrading a dependence-sensitive score. Test discrete/tied class
fractions, deterministic conditioned quantities and empty support explicitly.

### A1 Score factors and composition

| Component | Measurements | Required control |
| --- | --- | --- |
| Coarse posterior | Regional mass/trace, coarse-only shear RMS, fixed traceless-tensor projections, paired-region vector ranks and proper scores | Identical operator on coarse draws and coarse truth; power is not a calibration substitute. |
| Fine posterior given true coarse field | Fine-residual shear, regional gaps, conditional proper scores and joint ranks | Reuse oracle draws; exclude structurally fixed mass/trace from spread-normalized tests. Not a deployable posterior. |
| Composed field posterior | Density/eigenvalue marginals, mass, six-eigenvalue and four-gap regional vectors, spatial sums/differences and cross-scale dependence | Full-cap draws, with original and operator-matched targets separately; distinguish mean improvements from uncertainty improvements. |

For scalars report bias, widths, ranks, central coverage at 50/68/90/95%, fair
CRPS, raw MSE and MSE minus posterior variance/M. Do not clip a negative corrected
estimate then take its square root. For vectors retain the established joint-rank
test, fair energy/variogram scores and mean/covariance diagnostics. Freeze
whitening, regularization and projections; validate them with oracle controls
rather than fitting a correction to evaluation residuals.

Score the posterior distribution of each regional anisotropy statistic against
its own truth, not only average amplitude. A coarse marginal pass does not
establish correct coarse/galaxy association or coarse–fine dependence. Produce
a failure map, not a binary assignment of blame.

### A2 Test the operator candidate on posterior draws

After a throughput pilot, replay the fixed positive central-linear reconstruction
on all registered development cores and saved draws. Each draw supplies its own
density correction. Retain the original distant tensor contribution and unchanged
physical truth; report reconstructed-truth diagnostics separately. Never apply
true-density corrections to generated draws.

Measure scalar/joint coverage and class/proper scores as well as mean shear.
All-core local correction is still not a converged full-cap implementation.
Production candidacy additionally needs representative full-cap, extent and
precision checks. Improved class fractions are a hypothesis, not a guaranteed
consequence of improved average shear.

## Stage B Test consequences for the science

### B1 Galaxy class probability reliability

For each selected mock galaxy g, p(g,k) is the fraction of field draws assigning
class k at its declared position. Pin IDs, selection, real-space/redshift-space
convention and tensor interpolation or cell lookup for BOTH predictions and
truth. Where interpolation is used, interpolate tensors before eigenvalues.
Do not extrapolate beyond verified truth support or substitute pooled cell labels.

Use fixed probability edges 0, 0.2, 0.4, 0.6, 0.8, 1, with explicit endpoint
handling. Plot truth frequency against mean predicted probability for all four
classes. Retain empty/sparse bins and publish galaxy, spatial-block and phase
counts. Report classwise/multiclass Brier, reliability residuals and hard-label
confusion/purity/completeness as separate measurements. Include train-only class
prevalence as a non-informative comparator under identical selection.

Keep M fixed in raw Brier comparisons and also report a validated finite-ensemble
correction, so Monte Carlo noise in estimated probabilities is not mistaken for
model degradation. For screening, a probability bin with fewer than four
contributing registered cores in either phase is explicitly sparse; exceeding
that minimum still does not establish independent support or adequate precision.

Prioritize walls/filaments without hiding other classes. Report phase, shell,
cap and support splits before pooling. The earlier voxel filament-fraction
shortfall is not a galaxy misclassification rate or a known property-contrast
bias. If positional truth cannot be aligned, label cell diagnostics as cell-level
and mark B1 incomplete.

### B2 Regional class fractions

For each coherent draw and registered region compute both volume-weighted cell
class fractions and selected-galaxy class fractions. Use identical fixed support
and denominators across draws and truth. Empty regions are unavailable, not zero;
selection-weighted variants need separately frozen weights and estimands.

Report fraction bias, widths, coverage/ranks and paired-region differences.
Four fractions sum to one: use three fixed coordinates for nonsingular joint
tests and retain all four marginals. Validate tie handling with discrete oracle
controls. Means of marginal class probabilities cannot replace the joint
distribution of regional fractions.

### B3 One end to end property estimator

Proposed first demonstrator: filament-minus-wall low-sSFR fraction contrast,
log10(sSFR/yr^-1) < -11, with common mass/redshift support and standardization.
This does not assert that suitable physical mock properties already exist.
Freeze strata, eligibility and weighting before development contrasts. A
density-controlled version is a separately defined question, not an interchangeable
estimate of the same effect.

Compare truth labels and the complete CFM inference procedure; add P12-A only
where IDs, selection, smoothing and physical targets match. New GraphWeb_DESI
work requires its own implementation scope. Independently drawn P12-A galaxy
posteriors cannot be described as coherent field uncertainties.

If physical mock properties are unavailable, use a declared synthetic-property
law with zero and nonzero class associations and shared spatial residuals,
fixed before evaluation. It tests estimator recovery, not the physical size of
an sSFR effect. Repeated property draws do not create new density phases.

Test the actual estimator and interval procedure. Neither soft weighting by
p(g,k) nor coherent label draws automatically yields unbiased class-conditional
property means; omitted property information can matter. Separate field,
finite-sample and measurement uncertainty without double counting. Failure may
require an estimator or generative-model change, not stronger class discrimination.
B3 does not block completion of A/B field diagnosis.

## Statistical decision rules

### Finite draws and dependence

Every metric uses its own reference. At M=32, sorted entries 1 and 30
(zero-based) define the current continuous central interval with attainable
coverage 29/33 = 87.8788%, not 90%. Record attainability at other levels too.
Randomized TARP targets its stated nominal level. Validate the existing joint-rank
null independently rather than borrowing either convention. Class-fraction ties
require discrete oracle controls. Additional draws improve Monte Carlo precision,
not the number of independently observed universes.

Report phasewise and equal-phase/equal-case core-panel summaries. Galaxy
reliability also reports its explicit galaxy-weighted estimand. No voxel-, pair-,
draw- or TARP-reference-iid confidence intervals. Spatial resampling within a
phase is a sensitivity check, not a replacement for independent phases. Two
phases cannot establish tight population equivalence or exact conditional calibration.

### Acceptance and escalation

| Gate | Rule | Interpretation |
| --- | --- | --- |
| G0 Integrity | Input/phase bindings and oracle controls pass; positive density, shared-state and mass contracts hold. | Otherwise fix measurement before interpretation. |
| G1 Scalar coverage | In each phase, absolute error <=0.05 from metric-specific attainable coverage at 50/68/90/95% for density, eigenvalues and regional scalars. No registered cap/shell/boundary split with >=4 cores has a 90% deficit >0.10. | Inherited descriptive screen, not independent-trial qualification. Smaller splits remain unresolved. |
| G2 Joint and class coverage | Proposed extension: <=0.05 phase-level coverage error for predeclared regional joint summaries and class fractions. Fixed-bin class reliability uses proposed absolute residual <=0.05. | Sparse bins, rare knots and dependent samples can prevent a decision even when point estimates pass. |
| G3 Information value | Nominated primary proper score improves in both phases; other nominated primary proper scores worsen by no more than 2% versus matched control. Retain widths, discrimination and the train-only non-informative baseline. | No repair claim from inflated intervals alone. Near-zero score denominators use absolute changes and need explicit review. |
| G4 Numerical stability | Preserve mass tolerance <2e-6 and inherited solver/overlap contracts. Representative precision/NFE/extent changes <=0.1 posterior SD RMS for nondegenerate targets. | Conditioned quantities use absolute identities. Failure blocks promotion even if coverage improves. |
| G5 Fresh confirmation | Freeze method, domain, metrics, tolerances, draw count and phase inventory before separately authorized confirmation; apply gates without retuning. | Only the declared mock/domain summaries can qualify; ph001 stays excluded. |

G2 and the 2% G3 margin are proposed research tolerances, not universal standards.
Accept or amend them before new scores, never to rescue a candidate afterward.
Do not weaken the earlier scalar/stability gates. Report numerical screen outcome
separately from evidential status: supported, failed or insufficient evidence.
An estimate inside tolerance is not an equivalence demonstration.

Before G5, assess whether available independent phases and region sampling can
resolve those tolerances. If not, deliver a limited replicated screen or seek
approval for a larger validation design. Two fresh phases do not guarantee
certification. Synthetic metric-control repetitions validate the scorer, not
the learned posterior's sampling distribution across universes.

B3 has an additional science-estimator gate before property-science claims.
Provisional targets: absolute bias <=0.25 of the across-realization estimator SD
and interval-coverage error <=0.05 from its validated reference, for both zero
and nonzero controls. Report precision and null false-positive behaviour. If
independent realizations cannot resolve these, the estimator remains unqualified
even when field screens pass. These are proposed tolerances requiring a freeze.

## Stage C Select one defensible intervention

Register one hypothesis, one candidate and one primary calibration discrepancy
from the A/B scorecard before fitting. The following routes guide investigation,
not exclusive attribution:

| Evidence | Next route |
| --- | --- |
| Operator closure or stability dominates | Complete deterministic operator validation with unchanged draws. |
| Coarse summaries fail against coarse truth | Investigate coarse conditioning/objective/dependence; fine-oracle checks remain separate. |
| Fine conditional summaries fail with true coarse input | Test one fine-stage intervention versus unchanged continuation. |
| Selected factor summaries pass but composition fails | Probe missing conditional, cross-scale or sampler/context dependence; neither factor is globally exonerated. |
| Field improves but property estimator fails | Investigate estimator, selection and property conditioning, not stronger physical trends. |
| No consistent testable mechanism | Stop expansion of the model search and consolidate the failure map. |

Preserve the physical target and positive exact-mass decoder. Its latent is
block-centred log density, not an additive density residual. Independent coarse
draws paired with unchanged fine targets remove genuine conditional dependence
and are not a valid default. Any augmentation declares its changed objective
and train-only perturbation construction.

Candidate/control start from identical checkpoint and optimizer states with
paired seeds, training stream, update count and evaluation draw addresses.
Match updates; report runtime/FLOPs if per-update cost differs. Pin checkpoint
selection and evaluation cadence before fitting. Start with seed17; replicate
with seed29 only after checking its own baseline compatibility.

Advance beyond the first paired test only if its nominated calibration discrepancy
falls by >=20% in each development phase, G3/G4 pass and no previously passing
G1/G2 primary screen becomes a failure. This is a proposed investment gate, not
qualification. A baseline already inside tolerance needs a different frozen
objective, not a tiny-denominator improvement. Further work after failure needs
a new hypothesis and explicit budget.

## Compute proposal

The first execution request should cover A0/A1/B1/B2 preparation and saved-draw
analysis only. A2, new draws, B3, training and confirmation are separate decisions.

| Package | Proposed maximum envelope | Boundary |
| --- | --- | --- |
| A0/A1/B1/B2 | One CPU node up to 2h; one shared GPU up to 2h only if needed; <=64 GiB new artifacts | Pilot one registered case per phase, measure I/O/memory/throughput, then stream saved draws. No new full-cap generation. |
| A2 operator replay | Additional shared GPU up to 2h; <=64 GiB | Pilot first; save tensor corrections/summaries, not duplicated full density ensembles. Full-cap implementation may need a separate budget. |
| Decision-limiting Monte Carlo precision check | Additional shared GPU up to 2h; <=32 GiB | Preselect four geometry cases and M=128 before generation. A local resample cannot certify full-cap Monte Carlo precision; sampler/context must match. |
| C matched training/evaluation | Four single-GPU 2h training segments maximum, 8 GPUh total; <=4 GPUh evaluation; <=128 GiB | Two arms x two seeds; second seed conditional on first paired gate. Pilot sets feasible equal update counts. No automatic continuation. |
| B3 property demonstrator | Budget after property/estimand audit | Do not assume physical properties or matched P12-A products exist. |
| Fresh-phase/full-cap confirmation | Not budgeted here | Full freeze, measured costs and separate phase-access/compute approval required. |

These are caps, not forecasts or monetary estimates. If the complete panel cannot
fit, stop with a partial receipt and request a measured revision; do not drop
expensive cases. Full-cap generation is substantially different from scoring
saved draws and may exceed these envelopes.

Follow the NERSC allocation workflow: compute nodes for large reads, FFTs and
scoring; at most two pending/running interactive allocations; reuse compatible
ones. Validate current accounts, QOS, memory and filesystem requirements at
launch. Frozen longer production work requires a passed interactive smoke and
explicit batch approval. No unrelated cancellation, autonomous resubmission or
multi-node expansion is implied. Use cosmic_env and the clean Python convention.

Record peak host/GPU memory, I/O, seconds per case/draw and projected total cost.
Existing read-only inputs are under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1`.
Proposed new outputs are under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/calibration_investigation_20261006_v1`.
The new root is not created by this planning task. Never overwrite earlier receipts.

## Deliverables and review points

Each implemented package delivers a frozen configuration, source/input hashes,
phase-exposure ledger, engineering controls, per-case metrics, phase/stratum
scorecard, compute receipt and short decision record. Include reliability,
coverage-level, width/proper-score and paired-change plots. Missing/partial cases
stay visible. Code/configuration and decisions belong in Git; arrays in Scratch.

Scorecard fields: question, physical target, method/operator, phase,
cap/shell/support, independent-phase count, region/galaxy counts, draw count,
metric/reference, value, tolerance, numerical screen, evidential status, caveat,
artifact path/hash and next action.

| Review point | Decision |
| --- | --- |
| By 20 October, subject to execution approval | Review A/B and decide whether one intervention is justified; no deadline-driven gate relaxation. |
| 21–26 October | Protect travel/talk preparation; show calibration limitations alongside visualisations. |
| 27 October–12 November | If justified and approved, complete one matched intervention while maintaining P12-A writing. |
| After 12 November | Continue, restrict or pause CFM based on evidence and writing capacity; protect P12-A year-end submission. |

## Working checklist

- [x] Record objective, restricted scope and independence from P12-A submission.
- [x] Specify questions, controls, decision rules and proposed compute envelopes.
- [x] Approve Stage A execution and freeze the initial metric manifest on 6 October.
- [x] Complete A0 controls and A1/A2 full-panel numerical integrity checks.
- [x] Complete A1 factor/composition scorecards; cached density widths beyond 90% remain unavailable.
- [x] Authorize and complete bounded A2 all-core operator replay; no precision expansion or promotion.
- [x] Authorize and complete B1/B2 science-facing scorecards; G2 still fails (see `e2e_stage_b_execution_20261006.md`).
- [x] Audit B3 properties in the verified observation/parent-linkage products; stellar mass and sSFR absent there.
- [ ] Obtain an appropriate B3 property source and freeze its estimator, or separately freeze a synthetic generating law.
- [x] Complete the authorized saved-draw bias/spread/dependence diagnosis; see `e2e_stage_c_mechanism_20261006.md`.
- [x] Explain why no training intervention is yet justified: global shifts, uniform widening and cross-region pairing removal do not provide a phase-stable repair; conditional within-region structure remains unresolved.
- [ ] Execute matched comparison only after objective/budget approval.
- [ ] Review information value, remaining failures and writing opportunity cost.
- [ ] Design and authorize fresh confirmation only after a complete freeze.

Documentation review on 6 October checked source claims, target distinctions,
finite-draw references, phase restrictions, local links and scope. No new numerical
experiment, posterior qualification or execution approval follows from this plan.

The subsequent user authorization and live execution record are in
[Stage A execution](e2e_stage_a_execution_20261006.md). That record supersedes
the documentation-only status above for Stage A, not for training, Stage B or
fresh confirmation.
