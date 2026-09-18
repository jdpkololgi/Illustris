# Controlled VDM field-posterior experiment: scientific closeout

Final scientific closeout, 2026-09-18: **the approved experiment is complete;
the field posterior is not production-ready**. All ten fits, thirty checkpoints,
688 case reports, 33,280 fine draws and 7,872 distinct shared coarse draws are
verified. The final report and independent closeout audits pass. No new fit or
validation-selected checkpoint was introduced after results were opened.

The registered diversity contrast passes; the additional-context and multiscale
package contrasts do not. Sampled coarse uncertainty improves regional mass
and adjacent-core scores, but does not establish calibrated joint fields.
The next justified step is one bounded, genuinely coupled-field VDM/CFM
protocol, contingent on paired-data feasibility and a new resource proposal,
not an automatic extension of the present training.

## Experiment and claim scope

This implements the approved [contract](e2e_vdm_context_diversity_v1.md) and
[audit clarifications](e2e_vdm_context_audit_proposal_20260917.md).
The desired object is a conditional density-field posterior given galaxy and
survey observations, not a deterministic density reconstruction. This remains
a fixed-cosmology/HOD/epoch, conditional-on-mock experiment: the P3b angular
response is not an audited DESI fibre/redshift-success response.

| Arm | Distinct training cores / source phases | Observation information | Generated matter |
| --- | --- | --- | --- |
| A | 32 / ph000, ph002 | Local spatial channels plus wide means | Fine parent |
| B | 384 / ph000, ph002, ph003 | Same as A | Fine parent |
| C | Same 384 / 3 | Adds spatial wide context | Fine parent |
| D | Same 384 / 3 | Spatial wide context; coarse factor sees wide channels only | Shared coarse plus conditional fine |

Two optimization seeds; 20,480 updates and 40,960 presentations per factor;
checkpoints 5,120/10,240/20,480. Ten factors total, with 8,371,907 parameters
per fine factor and an additional 8,188,395 per D coarse factor. Fixed full-VLB
VDM, gamma [-13.3,13.3], deterministic FP32, AdamW 1e-4, weight decay 1e-5,
gradient clip 0.5, batch 2. No post-result fitting or best-checkpoint selection.

All arms share the same 48-cubed, 324.768 Mpc/h density parent and owned
16-cubed, 108.256 Mpc/h central core, with 6.766 Mpc/h cells. The target is the
physical-density 2-cubed average of the cubic-sampled R7 field at z=0.2, not a
second R7 smoothing. Observer redshift 0.15--0.55 is a selection coordinate.
D additionally samples a 48-cubed coarse field over 1299.072 Mpc/h. The parent
and wide fundamental wavenumbers are 0.0193467 and 0.00483667 h/Mpc.

All shared normalization/scoring scales use A32 only. No patchwise target
mean/std fit or density-DC subtraction. Spectral windows remove their weighted
DC; regional masses are therefore reported separately. Train phases are
000/002/003, development 004, internal confirmation 005; no ph001/ph006 access.
The two evaluation phases have programme history, not globally blind status.
384 patches are not 384 independent universes; overlapping parent/context
volumes and seven offset augmentations do not increase the phase count.

D enforces positive density and exact block mass through its coarse/log-residual
chart. Adjacent owned cores reuse the same sampled coarse realization, with
distinct fine noise. Its restricted coarse conditioner and conditional fine-core
independence remain approximations. C:D also changes capacity, compute, chart
and physical tidal closure; it is a package contrast, not pure stochasticity.

The original pre-fit tidal-composition gate failed and is preserved. The
explicitly approved consistent-density v2 uses the tidal operator on the
replicated wide density, plus the fine-parent residual tide; it does not
commute replication and the tidal operator. Its unchanged A32 physical gate
passes with 84.17% / 84.96% / 84.88% median per-anchor eigenvalue-RMSE reductions.
This is a truth-representation check, not learned-posterior calibration.
Computational wide FFTs are 192-cubed; the generated coarse grid remains
48-cubed. Six tensor components are averaged before eigenvalue computation.

## Registered decisions and common-panel results

Final-checkpoint summaries give equal weight to anchors, seeds and evaluation
phases. CRPS is the fair finite-ensemble score standardized by A32 scales.
Coverage is over correlated core probes, not independent full-field SBC trials.

| Arm | Density CRPS | Density-mean RMSE | Density C90 | Physical tidal energy |
| --- | ---: | ---: | ---: | ---: |
| A | 0.35214 | 0.33604 | 79.09% | 0.76876 |
| B | 0.31384 | 0.31177 | 89.32% | 0.69840 |
| C | 0.32443 | 0.32122 | 87.35% | 0.73209 |
| D | 0.31580 | 0.30671 | 86.97% | 0.70620 |

The final attainable C90 target is 90.7513%, versus 87.8788% for the earlier
32-draw checkpoints. Compare gaps to these targets, not raw coverage alone.
Final main-panel summaries use sixteen anchors per phase, two phases and two
optimization seeds: 64 cases per arm, not 64 independent cosmological volumes.

| Contrast | Pooled primary-score gain | Registered decision | Reason |
| --- | ---: | --- | --- |
| H1, A to B: diversity | +10.88% density CRPS | PASS | All four seed/phase gains positive (10.35--11.24%); coverage, mean-error and power nonregression pass |
| H2, B to C: spatial observed context | -3.37% density CRPS | NOT ESTABLISHED | Three of four score directions worsen; power nonregression fails all four; one coverage gap exceeds five percentage points |
| H2, C to D: multiscale package | +3.54% physical tidal energy | NOT ESTABLISHED | Below the registered 10% primary gain; confirmation variograms worsen in both seeds; one tidal-component coverage gap exceeds five points |

All four C:D tidal-score directions improve, but by only 0.79--4.97%. Do not
convert that into a passing contrast by dropping its predeclared requirements.
B has the best pooled density CRPS and physical tidal energy; D has the lowest
density-mean RMSE and substantially better regional mass. No arm dominates
all scientific requirements. A failed contrast is inconclusive for the
physical hypothesis at this budget, not proof that context or VDM is useless.

Paired spatial bootstrap summaries use 1,024 resamples with fixed phases:
13 occupied 500 Mpc/h blocks per phase and six/seven 1,000 Mpc/h blocks.
Every A:B score-difference interval favors B. Some C:D intervals include zero.
These are descriptive within-phase uncertainties, not independent-universe
confidence intervals or a cosmological-population significance claim.

## Checkpoint progression and sampler convergence

Development ph004 density CRPS, at 5,120 / 10,240 / 20,480 updates:

| Arm / seed | 5,120 | 10,240 | 20,480 | Late improvement |
| --- | ---: | ---: | ---: | ---: |
| A / 0 | .38674 | .36278 | .36204 | +0.20% |
| A / 1 | .37877 | .34495 | .36007 | -4.38% |
| B / 0 | .41124 | .34224 | .32135 | +6.10% |
| B / 1 | .35879 | .35252 | .32084 | +8.98% |
| C / 0 | .38612 | .35632 | .33856 | +4.98% |
| C / 1 | .36359 | .34513 | .33051 | +4.24% |
| D / 0 | .34517 | .33077 | .32221 | +2.59% |
| D / 1 | .33359 | .32856 | .32145 | +2.16% |

D's late physical tidal-energy gains are 2.00%/2.18%. These losses/scores have
not all plateaued. Nevertheless B/C sample-power discrepancy worsens in both
seeds while CRPS improves; D's highest-band power falls from .793/.635 to
.610/.615. This is not demonstrated joint convergence or a reason to extend
unchanged training automatically. At two predeclared fitted anchors, A's CRPS
improves from .318/.315 to .216/.215 while transfer plateaus or regresses,
consistent with overfitting but not a causal proof from only two anchors.
D still undercovers fitted fields (final fitted C90 .828/.848, target .879),
so failure is not exclusively unseen-field generalization.

All sixteen coupled 250/500/1,000-step screens pass. Worst late density
power/width changes are 1.399%/.757%, tidal/eigengap width .745%, and D-wide
power/width .758%/.667%; worst late paired MC95 bound is 1.447%. Main draws
use the registered 250 steps. These small numerical changes make integration
error an unlikely complete explanation of the much larger statistical
deficits; they do not certify exact sampling or calibration. M=8 refinement
intervals have their own attainable coverage and are not the main C90 panel.

![Registered checkpoint progression](evidence/e2e_field_v2/vdm_context_20260918/checkpoint_progression.png)

The shaded coverage region spans finite-ensemble targets, not an acceptance
band for arbitrary conditional populations. Earlier/final main-panel C90
targets are .87879/.90751 respectively.

## Field statistics, power and spatial uncertainty

Marginal improvements do not establish spatial calibration. The eight
162.384 Mpc/h octants of each generated parent have regional-mass C90 of
56.64% / 46.88% / 31.64% / 74.80% for A/B/C/D, against 90.75% attainable coverage.
These regions include the auxiliary parent halo; they are not the smaller
owned science core. Regional CRPS is .03819 / .04368 / .05239 / .02046.
A/B/C RMS spread is .02997 / .02647 / .02177 versus mean-prediction RMSE
.06167 / .06720 / .07641. D makes substantial progress without reaching
calibration; B's good voxel intervals conceal much worse regional uncertainty.

Parent one-point mean density contrast is -.02114 / -.04310 / -.05670 / +.00518,
versus truth -0.00218; mean within-field standard deviation is
.43855 / .40163 / .37519 / .41338, versus truth .45237. B's core density bias improves
from A's -0.06518 to -0.03553, while its parent mass bias worsens. Distinguish
core reconstruction improvement from the wider generated-field distribution.
D's core bias is only +.00200, but within-core standard deviation .39070
remains below truth .43840 and its density q99 is 1.243 versus truth 1.503.
Repairing the mean is not repairing the whole distribution.

Parent bands have edges [0,.04,.08,.16,.32,infinity] h/Mpc, measured on the
324.768 Mpc/h parent, not the 108.256 Mpc/h core. Pooled sample/truth ratios:

| Arm | Band 0 | Band 1 | Band 2 | Band 3 | Band 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | .880 | .961 | .970 | .617 | .462 |
| B | .691 | .803 | .814 | .785 | .707 |
| C | .478 | .605 | .695 | .756 | .725 |
| D | .929 | .918 | .781 | .737 | .628 |

These are individual-sample powers averaged over the panel, not posterior-mean
powers. Per-cell B ratios span .562--.893 (11--44% suppression). B's
posterior-mean ratios are [.423,.465,.476,.368,.204], finite-M-corrected mean
ratios [.419,.460,.471,.362,.197], and unbiased posterior residual ratios
[.272,.343,.342,.423,.509]. The decomposition passes for every case. B's
mean-field cross-correlations [.701,.750,.739,.659,.475] improve over A
[.690,.735,.704,.576,.346], without establishing correct posterior fluctuations.
No requirement is imposed that each draw match its paired truth spectrum.

D's wider coarse bands have edges [0,.01,.02,.04,.08,infinity] h/Mpc and
pooled ratios [1.357,.949,.958,1.000,1.003]. Wide voxel C90 is 89.24%.
Thus blanket suppression at every scale is not the diagnosis: the lowest
wide band has excess power, while the fine highest band is about 37% low.
Nearly correct wide one/two-point statistics do not prove conditional
cross-location dependence. A--C have no generated wide matter counterpart.

![Sample and posterior-mean field statistics](evidence/e2e_field_v2/vdm_context_20260918/field_statistics.png)

## Tidal/eigengap and conditional calibration

Final physical C90, percent; ordered eigenvalues are lambda1 <= lambda2 <=
lambda3 and gaps are lambda2-lambda1, lambda3-lambda2. Target is 90.7513%:

| Arm | Density | lambda1 | lambda2 | lambda3 | gap12 | gap23 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 79.09 | 78.55 | 80.48 | 79.46 | 82.19 | 79.63 |
| B | 89.32 | 87.98 | 88.41 | 86.88 | 88.30 | 85.51 |
| C | 87.35 | 84.83 | 86.21 | 83.67 | 86.01 | 82.15 |
| D | 86.97 | 86.63 | 87.05 | 86.10 | 87.43 | 86.79 |

Density C50/C68/C95 for A/B/C/D is respectively
[.396,.490,.468,.461] / [.555,.669,.644,.638] / [.842,.930,.915,.913],
against attainable [.50674,.68784,.94222]. All widths, biases, CRPS and
rank histograms remain in the case records. For example B's density RMS
spread is .26920 versus mean RMSE .31177. Density first/last rank-bin masses
are A .084/.166, B .059/.083, C .067/.096, D .093/.074 (16-bin uniform mass
.0625); D also has lambda1 left-tail .120 and lambda3 right-tail .106.
Near-C90 aggregate agreement is not uniform rank or conditional calibration.

Physical and matched-boundary tidal scores are separately retained. A/B/C
periodic physical energies .76876/.69840/.73209 are close to their
matched-closure energies .76635/.69609/.72970. D's composite physical energy
.70620 is close to common-periodic physical .70734 and matched .70586;
reflect/zero alternatives do not remove the error. Thus closure alone is
not the main learned-posterior bottleneck here. The earlier truth-only 84%
representation-gate improvement is not an 84% gain in learned calibration.
Unrepresented matter beyond the 1.3 Gpc/h domain remains a limitation.

Train-only bins cover angular response, boundary distance, observed log-count,
log nbar, observer redshift and support fraction. Each occupied bin contains
only three to eight anchors per seed/phase. Density C90 ranges are:

| Observational stratum | A | B | C | D |
| --- | --- | --- | --- | --- |
| Angular response | .742--.835 | .859--.948 | .821--.924 | .836--.897 |
| Boundary distance | .664--.837 | .804--.930 | .786--.911 | .830--.894 |
| Mean log(1+count) | .690--.882 | .848--.931 | .838--.910 | .833--.897 |
| Mean log nbar | .687--.845 | .861--.942 | .827--.933 | .833--.894 |
| Observer redshift | .668--.896 | .839--.931 | .830--.913 | .829--.900 |
| Support fraction | .739--.835 | .844--.932 | .818--.908 | .845--.890 |

Observed/unobserved core probes also differ: B's four seed/phase observed
coverages span .876--.913 versus .758--.860 unobserved; D spans .859--.880
versus .758--.855. Unobserved bins use only anchors with such probes (12 or
13 per phase). These are descriptive associations, not proof that selection
shift, normalization, or redshift itself causes the error. Selection matters
for later deployment even if its present causal contribution is not isolated.

## Shared-parent attribution: useful but insufficient

The preselected adjacent-core panel contains four physical pairs (two caps
in two phases), both optimization seeds, shell 2 interior only. It is separate
from the main panel. Sampled and fixed-mean controls use 32 draws; the true-
coarse oracle uses 16. Oracle conditioning is not available at deployment.

| D control | Density CRPS | Physical tidal energy | Pair energy | Variogram |
| --- | ---: | ---: | ---: | ---: |
| Sampled shared coarse | .25562 | .58839 | .33415 | .012932 |
| Fixed predicted coarse mean | .26715 | .63285 | .58455 | .019102 |
| True-coarse oracle | .17377 | .41693 | .28066 | .003608 |

Sampled versus fixed-mean coarse improves both pair scores in all eight
seed/phase/cap cases. Coarse stochasticity is scientifically useful; replacing
it with its mean structurally removes regional-mass variance. Nevertheless,
D summary-difference C90 across density/eigenvalues/gaps averages only 33.33%,
versus 12.50% fixed-mean and 87.88% attainable for nondegenerate M=32
ensembles. Density differences cover 2/8 and gap23 differences 0/8 cases.
The fixed-mean density summary is itself deterministic, so its pooled coverage
also mixes a collapsed quantity with stochastic tidal/gap components; use the
proper-score comparison rather than treating that number as continuous SBC.
Those repeated-seed cases are not eight independent universes, nor are their
six summaries 48 independent calibration trials.

C:D pair energy improves in all four seed/phase aggregates, but ph005
variograms regress .01128 to .01273 and .00930 to .01491. Shared-parent
receipts prove shared draws, not correct cross-core covariance. D's sampled
density covariance varies in sign over this small panel; fixed-mean covariance
is essentially zero. Fine cores remain independent conditional on the parent.

**Deterministic oracle caveat:** block mass fixes oracle regional masses and
the density core-mean difference to truth (spread/bias of order 1e-15). Their
generic interval coverages are tie/roundoff-sensitive, not tests against the
continuous M=16 target .88235. The raw pooled oracle difference C90 of 68.75%
mixes this deterministic quantity with stochastic tidal/gap differences and
must not be called a clean calibration gain. Preserve raw outputs, interpret
nondegenerate components and proper scores separately. Improved oracle scores
localize missing coarse information; they are not an E2E result or a theorem
that every score must improve. Even oracle highest-band fine power is .634,
so perfect coarse information does not by itself fix the fine spectral deficit.

![Predeclared posterior field example](evidence/e2e_field_v2/vdm_context_20260918/posterior_fields.png)

This fixed example shows parent 48-cubed slices, not only the owned 16-cubed
science core. Density panels share truth's 1st--99th percentile color limits;
spread panels are individually scaled, with ranges in their titles. Visual
plausibility and individual spread colors are not calibration tests.

## Decision, CFM and literature sanity check

**No production release and no unchanged training extension. Yes to developing
one bounded, joint-field follow-up.** The positive lead is not merely better
marginals: stochastic parents improve joint scores, but the remaining
fine/coarse dependence and power are inadequate. P12's local per-galaxy
eigenvalue posterior remains a different, separately useful product; it
cannot stand in for a joint density-field posterior.

The next protocol should first verify additional independently phased, paired
matter/BGS complete/altmtl data, retaining sealed phases. Then compare a
retained D baseline with a common genuinely coupled spatial design under
matched VDM and conditional-flow-matching objectives, two seeds, identical
observational information and train-only invertible scale conditioning.
Jointly evolving adjacent regions must allow residual dependence after
conditioning on the shared parent. Benchmark and approve a finite budget
before launch; freeze phase roles, tolerances and joint-specific promotion
gates before predictive results. No per-realization power penalty in the first
comparison. If joint power, regional uncertainty, pair/tidal/gap scores and
conditional calibration fail again, stop this neural route and reassess the
forward/selection model or a bounded physical-likelihood reference.

The [literature and data assessment](e2e_vdm_context_joint_decision_literature_20260918.md)
reviews CAMELS, 3D diffusion, WSGM/GUD, Cosmo3DFlow, Cosmo-FOLD, BORG and
FMPE. Cosmo-FOLD's shared evolving spatial state is a relevant design precedent;
CFM provides a different transport objective and potentially cheaper draws,
not a guaranteed calibration repair. Earlier F3-L2/F3-L2c flow results and
the difference between an invertible wavelet model and a flat Haar CNN are
retained rather than treating CFM as either untested or already disproved.
Raw additional Abacus phase directories and BGS mock products exist, but their
phase/HOD/epoch/coordinate/truth pairing is not yet verified as a usable set.

The [Gemini primary-paper audit](e2e_gemini_literature_audit_20260918.md)
confirms the four references but rejects their overextended interpretation:
deterministic transport of random initial states is not MAP collapse; DPS,
an unconditional prior and fresh Langevin noise are not necessary for a
conditional posterior. Denoising MSE is not a Gaussian galaxy likelihood.
Poisson/NB observations and differentiable HOD likelihoods are modelling
choices requiring validation, not ready exact likelihoods. Conicus3D's
noise-dependent spectral covariance is a useful reference mechanism, not
permission to recolor samples or to double-count BGS observations as both
conditioning and an independent likelihood. Full DESI use still requires
audited RSD, HOD/velocity nuisance, epoch evolution and actual survey response.

## Completion, reproducibility and resource accounting

Run root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1`.
Frozen scientific source: `16398511d4552bd041e8d1e97cebd3e168088064`.

- All ten fits finished; all 30 checkpoint payload hashes verified in
  `analysis/CHECKPOINT_INTEGRITY.json`, including source and data-receipt bindings.
- All sixteen coupled 250/500/1000-step sampler screens pass. Worst late density
  power/width changes are 1.399%/0.757%; worst late paired MC95 bound is 1.447%.
  This is numerical stability within registered tolerances, not calibration.
- Exact draw ledger: 688 cases, 33,280 fine and 7,872 distinct coarse draws.
  `DRAW_LEDGER_INTEGRITY.json` verifies identities, bindings and paired RNG
  receipts. `DRAW_INTEGRITY.json` independently hashes all 5,144 payloads,
  34,634,901,688 bytes, with no mismatch.
- Actual SIGUSR1/fresh-process replay and full-size scalar/batch sampling tests
  pass. Sampling survived four planned pause/resume handoffs; nonzero code75
  is the registered clean pause, not an unexpected failure.
- Full frozen report CPU58524609 completed 0:0 in 2,896 seconds. Every case
  includes the registered field, physical/matched tidal, gap, conditional and
  spatial diagnostics. All three figures have been visually inspected.
- Optional case-audit step4 ended when that allocation naturally released;
  no partial audit was promoted. One 15-minute-max JSON-only recovery,
  CPU58526345, completed 0:0 in 12 seconds. CASE_REPORT_AUDIT and CLOSEOUT_AUDIT
  pass all 688 cases, reconcile all 384 preview cases, and independently
  recompute all 32 progression cells. No new fields, fits or draw generation.
- Final allocated totals including that recovery: **78.0075 GPUh and 2.536111
  CPU-nodeh**; Scratch **122,030,674,932 bytes (113.65 GiB)**. Resource closeout
  occurred 23.03 hours after the first allocation, within approved
  112 GPUh / 8 CPU-nodeh / 300 GiB / 48-hour limits. Eight of eight allowed
  GPU requests were used; at most two interactive allocations concurrently.
  Full-node GPU jobs used four workers. No new compute remains pending.
- Final resource authority is `analysis/closeout_resources/ACCOUNTING.json`:
  it includes all 18 jobs and the supplemental audit. The original immutable
  EXPERIMENT_COMPLETE receipt predates that 12-second recovery. Do not add
  shared report steps or the already-accounted preview twice.

RESULTS SHA256:
`2ca834400a766de7d6c357679f823fc9ee848f12b40ce7a562949a8186c3303f`.
MANIFEST SHA256:
`67e1f9d0edc73b58c761e35f1f2dc13d4fe102e27f260104ecdb87b725a9bf10`.
MODELS_FROZEN SHA256:
`04abc713e9a1baa87288640ac10672b318ee7fd218ebcb2229eb78323bd3ea57`.

Small final results, figures, audit and accounting receipts are archived with
their original hashes in
[the evidence directory](evidence/e2e_field_v2/vdm_context_20260918/README.md).
Bulk checkpoints, arrays and all case-level metrics remain under the Scratch
root (not backed up). The large REGISTERED_DIAGNOSTIC_SUMMARY.json summarizes
already registered case metrics; it is not an additional model-selection run.
The original failed physical-v1 gate and all expected pause/failure statuses
are retained. The science log and field-model plan link this final report;
earlier preliminary/running entries are chronological history, not current state.
