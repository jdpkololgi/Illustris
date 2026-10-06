# Coherent density fields with uncertain exterior matter

## Aim and current status

The user has authorized a comparison of exterior-density models aimed at a
publishable conditional field posterior, followed by all available
training-unseen simulation phases and a full-cap environmental website update.
Necessary compute is preapproved. The two-phase development comparison is
complete and its marginal coverage is encouraging, but every candidate fails
the joint regional tidal diagnostic. No posterior qualification is claimed.
The numerical checks and selection freeze are complete. The full-resolution
production pilot and remaining targetless input preparation have been submitted;
final full-cap scores and website publication are still pending. Fresh-phase
scores cannot change the selected model or numerical settings.

The previous compact exterior-response experiment did not qualify its model.
The present target is different: each sample must be a coherent density field,
including an explicit treatment of unobserved matter. Tidal tensors, ordered
eigenvalues and web classes must then be deterministic functions of that same
draw. Independent per-core tensor adjustments cannot stand in for this object.

## Data and phase discipline

### October 5: true-density operator closure

Authorized development diagnostic: use phases012/013 only and the immutable
production source `source_78156ac3417555ef269f`. The harness
`workflows/sbi/e2e_true_density_closure.py` rebuilds the native2048 R7-smoothed
contrast and uses the original cubic interpolation/two-raw-cell averaging to
fill each cap's entire source cuboid at6.766Mpc/h, including holes and32-cell
padding. No learned completion, footprint weighting, new smoothing, or model
fit enters this oracle. The exact frozen `CapTidal` maps it to tidal tensors.
The target is the regional average of ordered voxel eigenvalues, not the
eigenvalues of the regional average tensor.

Compare against the independently saved periodic full-box target in every
registered two-core case, retaining32unchanged posterior draws. Report
`operator_target - fullbox_target` divided by each posterior standard deviation,
the same gap in fixed20percent-shrinkage covariance units, and its alignment
with `posterior_mean - fullbox_target`. Recompute the original joint rank as a
baseline check and evaluate the unchanged ensemble against operator-matched
truth. This counterfactual test neither redefines the production estimand nor
qualifies a posterior. It isolates combined representation/domain error; it
does not distinguish discretisation from missing matter beyond the cuboid.

Acceptance checks: true source agrees with all saved core densities within
2e-6; true-source tensor trace within1e-8; exact recovery of original rank;
source/freeze hashes; direct-cuboid small GPU control; zero/fully-explained/
opposite-direction synthetic residual controls. Count phases, not cores, as
independent units. Artifacts: campaign `verification/true_density_closure_20261005_v1`.

Completed results (32draws,16pairs perphase):

| Diagnostic | ph012 | ph013 |
| --- | ---: | ---: |
| RMS operator offset / component posteriorSD | 0.5806 | 0.6032 |
| Largest absolute offset / component posteriorSD | 1.4876 | 1.7373 |
| Original joint90coverage | 37.5% | 50.0% |
| Operator-matched joint90coverage | 50.0% | 62.5% |
| Original regional marginal90coverage | 67.71% | 73.96% |
| Operator-matched regional marginal90coverage | 75.00% | 84.38% |
| Mean squared covariance-weighted residual, original | 16.899 | 17.316 |
| Mean squared covariance-weighted residual, matched | 10.681 | 9.482 |

Finite-draw90reference is29/33=87.879%. In each phase13of16pairs have smaller
covariance-weighted residual; two previously failing joint regions pass and
none reverse. Mean eigenvalue shifts are approximately(+.009,-.0003,-.009),
compressing the eigenvalue separation. This is suggestive of representation
effects but does NOT separate discretisation from finite-domain tides.
Exact agreement with all saved core density targets, oracle trace errors below
2.58e-14 and exact original-rank recovery pass. Five focused controls pass.
This identifies a material partial explanation, not a mostly-resolved failure
or a validated posterior. Keep both the original physical-target assessment and
the counterfactual matched-target assessment visible. The remaining gap requires
posterior bias/spread/covariance diagnosis. Do not inflate eigenvalue intervals
independently or change the physical truth to manufacture qualification.

### October 5: exact trace/gaps, grid/extent split and checkpoint test

Completed development-only follow-up on ph012/ph013. The 32 individual draws
already existed in full-cap HDF5; they have now been extracted into regional
NPZ records. Define trace as the sum of ascending eigenvalues (equal to density
contrast), gaps as lambda2-lambda1 and lambda3-lambda2, and width as
lambda3-lambda1. These are means of voxelwise eigenvalues: their gaps diagnose
anisotropy but do not retain the five oriented traceless-tensor components.
Uncertainties below use actual draw covariance, not independent eigenvalues.

| Pooled descriptive diagnostic, 64 correlated cores | Full-box truth | Operator-matched truth |
| --- | ---: | ---: |
| Mean true width | 0.469058 | 0.450900 |
| Mean posterior width | 0.433109 | 0.433109 |
| Posterior width deficit | 7.664% | 3.946% |
| Posterior mean width below truth | 54/64 | 44/64 |
| Mean exact width posterior SD | 0.027731 | 0.027731 |
| Mean standardized width bias | -1.544 | -0.821 |
| Width marginal nominal-90 coverage | 59.375% | 75.0% |
| Joint four-gap nominal-90 coverage | 46.875% | 68.75% |

Trace has pooled bias -0.002210, marginal coverage 85.9375%, and joint two-core
coverage 87.5% in each phase. It is less problematic than the gaps, but a small
average bias is not proof of conditional calibration. There are only two
independent phases, not 64 independent sign trials; even calibrated posterior
means need not fall on either side of truth with equal probability.

For the remaining matched-target error, subtracting each phase's global bias
does not remove direction-dependent excess residual scatter. After subtracting
estimated 32-draw mean noise, gap-scatter eigenvalues in mean-posterior-covariance
units are [0.491, 1.305, 1.987, 2.478] and [0.425, 1.011, 1.504, 2.300].
Trace values are [0.662, 1.626] and [0.359, 0.955]. This is a descriptive
bias/spread/covariance split, not a calibration fit: observation-dependent bias
and covariance error cannot be uniquely separated using these dependent cases.

The operator controls hold true density and geometry fixed:

- Extent: double the full true-source cuboid padding from 32 to 64 cells at
  6.766 Mpc/h, verifying exact overlap with the original source. Across all
  32 pairs, eigenvalue shifts are only 0.00531/0.00459 posterior-SD RMS and
  at most 0.02904/0.02036 SD for ph012/ph013. The width deficits barely change:
  3.8149% to 3.8136%, and 3.9271% to 3.9173%.
- Grid: on four geometry-selected pairs (eight cores), evaluate the same
  cuboid operator at 3.383 Mpc/h using true subcell density, average tensors
  onto the original cells BEFORE eigendecomposition, and add the local
  fine-minus-coarse tensor correction to the cap tensor. Recover 73.1–75.0%
  of the operator width deficit in every tested core; residual deficit is
  0.81–1.16%. Increasing the refinement collar from 16 to 32 coarse cells
  changes components by less than 0.000331 posterior SD. Trace corrections
  remain below 7.4e-15.

This implicates local grid representation as the dominant tested operator
effect. Extent doubling is not infinite-domain convergence, and local refinement
is not a whole-cap half-cell solve. Its use of true subcell density makes it an
oracle diagnostic, NOT a deployable correction: never transfer the truth-based
tensor increments to posterior draws. A repair must reconstruct tides using
only the inferred density (or an explicitly modelled subgrid posterior).

The early-stopping test replays both coarse checkpoints with the SAME fine
26,624 checkpoint, normalizer, NFE128 and 32 random addresses, for both seeds
and every pair. All saved input hashes pass and CPU/GPU tensor parity is
7.77e-16. There are 4,096 field replays. These use the historical local+wide
sampler/operator, not the current full-cap sampler.

| Seed / phase | Mean width, coarse 13,312 | Mean width, coarse 26,624 | Mean width SD, early to late |
| --- | ---: | ---: | ---: |
| 17 / 012 | 0.451718 | 0.451852 | 0.028335 to 0.026257 |
| 17 / 013 | 0.462445 | 0.465885 | 0.028146 to 0.026272 |
| 29 / 012 | 0.451245 | 0.446842 | 0.028182 to 0.025346 |
| 29 / 013 | 0.458651 | 0.462458 | 0.027733 to 0.025164 |

Longer training does not consistently recover width and reduces its spread by
6.7–10.1% in every comparison. This does not support a general claim that early
stopping sacrificed shear for calibration, nor uniquely establish a low-k cause.
An F3-L2-style intervention remains a hypothesis; its earlier repaired dependence
did not establish conditional coverage for this task.

The proposed inference that compressed eigenvalues necessarily underpredict
voids and knots is false. At fixed trace, compression can increase either extreme
class depending on trace and threshold (a synthetic counterexample passes).
Actual threshold-zero knot fractions OVERpredict truth by 1.093/1.114 percentage
points, while threshold-0.2 void fractions overpredict by 2.396/2.605 points.
Aggregate class fractions are not a substitute for probability calibration.

Next research gates: a density-only, grid-consistent operator reconstruction,
then a frozen scale-resolved/mode-ablation test of the remaining posterior
deficit before selecting a training or architectural change. Preserve the
original physical target; no independent eigenvalue rescaling or model promotion.
No confirmation phases were opened by these diagnostics, and no production
model, scorer or website changed.

The proposed density-only reconstruction and matched sampler/factor follow-up
is now complete; see [the dedicated October 5 results](e2e_density_sampler_diagnostics_20261005.md).
It supports a deterministic operator improvement, a partial sampler/context
effect and strong coarse-conditioning sensitivity, but no posterior promotion.

Reproduction: `e2e_trace_shear_diagnostic.py`, `e2e_operator_split.py`,
`e2e_checkpoint_shear.py`, `e2e_shear_followup_summary.py` and
`e2e_operator_split_summary.py` in `workflows/sbi`. Receipts are under campaign
`verification/trace_shear_20261005_v1`, `operator_split_20261005_v1`, and
`checkpoint_shear_20261005_v1`. Four focused follow-up tests pass; original
closure parity, trace, frozen-source and geometry checks remain enforced.
CPU 59381367 and GPU 59381720 completed successfully and were released with
explicit user approval. The data-validation workflow kept paired comparisons,
units of replication, oracle information and supported claim limits explicit.

The inherited CFM and normalizer use thirteen training phases. The prepared
training-unseen set is ph012 through ph019. Ph012/ph013 are development phases;
ph014 through ph017 have already been exposed by earlier evaluations. Only
ph018/ph019 remain fresh confirmation for this programme. Freeze the complete
candidate and evaluation contract before opening their prediction or truth
payloads. Repeated use of exposed phases must be disclosed, not relabelled blind.

The user explicitly chose to preserve the sealed ph001 programme reserve in
the campaign clarification. Do not prepare, predict or evaluate ph001.
The phrase all available unseen phases also requires a metadata audit of
ph004/ph005/ph006. Legacy native grids for ph004/ph005/ph006 exist, but
the coupled-model input products and exposure histories require qualification.
The historical selection recipe used some legacy phases; absence from CFM
training does not imply absence from all upstream fitting. Retain legacy access
guards and register separate compatible inputs if these phases are included.
List any unavailable or incompatible phase and its reason explicitly.

The user subsequently explicitly requested rebuilding ph004/ph005/ph006 as
additional non-blind replication. The final requested inventory is therefore
eleven phases: 004/005/006 and 012–019. Their separate rebuild authority is
`configs/e2e_exterior_legacy_inputs_20261002.json`. No old guard or registry is
relaxed, no legacy output is overwritten, and no newly rebuilt phase may be
used to choose or fit the exterior model. Numerical truth preparation is
distinct from predictive scoring; all final scores require the later freeze.

## Comparisons and gates

Compare zero exterior contrast and periodic controls against spatially
correlated unconditional exterior density, exterior density conditioned on
each sampled interior, and learned CFM exterior draws. These are candidate
families, not yet a frozen numerical specification. Fit new prior or likelihood
parameters on training phases only. Register the concrete model choices,
selection rule and numerical tolerances before their development scores.

First isolate the exterior problem using matched true-density controls where
useful; then evaluate the actual galaxy-conditioned sampler plus tidal operator.
Never report truth-conditioned performance as operational inference. Larger
connected domains and full-cap checks are required; two-window agreement alone
is not survey-wide joint calibration. Zero exterior contrast means a mean-density
background, not empty space.

Report finite-ensemble attainable coverage at several levels, ranks, widths,
proper scores, regional masses, joint summaries, spatial dependence and
environment probabilities. Retain phase, cap, redshift and boundary-distance
results. Check positivity, identical overlapping values, long modes, solver
precision, source resolution and exterior-domain extent. Treat correlated
cells within a phase as dependent and preserve the existing R7 estimand.

After development selection, freeze weights, sampler, exterior law, operators,
draw count, metrics and final phase inventory. Immediately execute the frozen
full evaluation. A candidate may be best among those tested without being
qualified: publication claims remain conditional on the observed diagnostics
and the mock cosmology, epoch, tracer and selection assumptions.

## Registered spatial-prior comparison (before new fits or scores)

The new prior is a stationary Gaussian field for log density, with covariance
equal to a nonnegative mixture of three discrete squared-inverse-Laplacian
spectra. Length parameters are fixed at 1, 4 and 16 fine cells (6.766, 27.064
and 108.256 Mpc/h). Mixture variances are fitted by nonnegative least squares
to training log-density covariance at lags 0, 1, 2, 4, 8, 12 and 16 cells.
Average within each training phase and then equally across all thirteen
phases. Do not subtract individual patch means: that would erase measured
long-mode variance. Use all registered training pairs, with no score-driven
pair selection. Set the log mean to minus half the total variance so the
prior's ensemble density mean is one; report its empirical lognormal mismatch.

Conditional simulation uses the Gaussian conditional formula and an explicitly
checked conjugate-gradient solve, not interpolation of a deterministic mean.
The sampler preserves every supplied in-footprint density value. Fractionally
occupied fine cells retain their inferred full cell-average density; these are
an effective-cell representation, not resolved subcell boundary predictions.
The isolated zero-contrast control retains the old fractional-occupancy source
convention and must be identified separately. The learned arm keeps all valid
CFM-generated cells, including those outside the footprint, and conditions the
same prior on those cells to extend beyond the CFM domain. The unconditional
arm replaces only the exterior with an independent spatial prior draw. The
conditional arm conditions the spatial prior only on in-footprint sampled cells.
All stochastic arms therefore return an entire positive density realization.

The initial connected panel uses the registered two-window shared-state CFM,
seed-17 coarse step 13312 and fine step 26624, Heun NFE 128, and 32 posterior
draws. Use every registered development pair in ph012/ph013. This panel is a
development comparison, not the full-cap joint qualification. The prior's
finite periodic domain is not the tidal boundary condition: apply the analytic
free-space cuboid operator to density contrast on the declared completed domain,
without another R7 smoothing. Primary padding is 16 fine cells on each side;
repeat representative cases with 32 and 64 cells to measure extent dependence.
The periodic tensor remains a separately labelled control. New density models
must not be compared with different tidal operators without identifying that
as a confounded control.

Numerical gates: Gaussian conditional reference tests must pass; exact interior
values, density positivity, finite tensors, relative CG residual below 2e-7,
shared-window density agreement below 1e-10 and mass consistency below 2e-6.
Repeat representative posterior draws at NFE 256. Report per-target extent and
solver changes relative to posterior standard deviations; an RMS change above
0.1 posterior standard deviations is a failed stability gate, not permission
to call the result converged. Fixed-cell R7 truth remains the primary estimand;
the prior cannot claim to resolve density below that representation.

Select among the three stochastic exterior arms by lowest equal-phase,
equal-cap mean fair CRPS of the three ordered eigenvalues on development
cores. This selects a best-tested approximation, not a qualified posterior.
Report paired changes by phase and cap; no voxel-iid significance claims.
Publishable qualification additionally requires finite-ensemble 50/68/90/95
percent coverage within 0.05 of attainable nominal on each fresh phase for
density, each eigenvalue, and regional means; no cap/boundary/redshift split
with at least four cores may have 90-percent coverage deficit above 0.10.
Report ranks, widths, fair energy and variogram scores, shuffled-dependence
controls, environment Brier scores and joint regional diagnostics. Two fresh
phases cannot establish population-level calibration even if these screening
gates pass. If every arm fails, still freeze and evaluate the best-tested arm
as explicitly unqualified; never retune on the fresh confirmation scores.

Full-cap generation and website publication require the selected law to be
applied to one connected cap density per draw. Independent local completions
cannot be stitched into a purported global posterior. The two physical cap
lattices are not commensurate; either demonstrate a common-world completion
or disclose and test the independent-cap approximation. This remains a
separate qualification condition, not something two-window tests establish.

## Development evidence and full cap approximation

All 32 connected development cases in phases 012 and 013 completed with
32 draws each. The equal-phase, equal-cap, three-eigenvalue fair CRPS and
joint six-regional-eigenvalue coverage are:

| Exterior treatment | Fair CRPS | Joint regional 90 percent coverage |
| --- | ---: | ---: |
| Zero contrast control | 0.062786 | 21.875 percent |
| Unconditional correlated density | 0.057819 | 46.875 percent |
| Conditional correlated density | 0.057508 | 50.000 percent |
| Learned CFM exterior plus conditional extension | 0.057463 | 46.875 percent |
| Periodic operator control | 0.056851 | 65.625 percent |

The attainable finite-draw reference is 87.879 percent. The learned arm
narrowly wins the registered comparison among spatial-density candidates,
but the difference from the conditional arm is small and only two independent
phases contribute. The periodic control scores better than all three; it is
not the same exterior-density model evaluated with the same operator. None
passes joint calibration. Spatial-arm density marginal coverage is about
87.7–87.9 percent, while eigenvalue marginal coverage spans 83.7–86.6 percent
in phase 012 and 85.9–86.8 percent in phase 013. Marginal success must not be
used to claim a calibrated field posterior.

The direct cuboid-sum check agrees with the CUDA operator within 2.23e-16.
Four saved-density cases replay exactly; doubling the sampler resolution to
NFE 256 changes density by at most 0.259 percent of posterior standard
deviation. Holding a larger-domain density realization fixed and cropping
only its gravitational sources, padding 16 cells reaches 10.07 percent of
posterior standard deviation and narrowly fails the registered 10 percent
threshold. Padding 32 cells stays below 4.57 percent in all four tested cases;
use 32 for the full-cap numerical implementation. These checks do not establish
convergence of the prior distribution with domain size or convergence to a
finer native-density representation.

Full-cap completion is explicitly capwise: each cap has a coherent density
draw and a spatially correlated exterior conditioned on that draw. The
opposite cap's observations do not condition this exterior, and multiplying
the two cap laws is an independence approximation. Do not add both completed
cap boxes as gravitational sources: their modelled exteriors overlap. Do not
claim a common-world or joint two-cap posterior. Final evaluation must test
the resulting capwise approximation and retain that limitation on the website.

Source results are under `scores/ph012`, `scores/ph013` and
`verification/numerical_v1` in the registered campaign artifact root. All six
targetless full-cap input caches for development phases and Loa completed.
The NGC throughput benchmark uses 1537 windows, four draws and NFE 2: 92.03
seconds sampling, 38.91 seconds loading and 26.73 GiB GPU memory. Scaling its
sampling time to NFE 128 predicts about 99 minutes per four-draw batch; this
is a planning extrapolation, not a completed posterior run. The full-cap
completion smoke test and independent saved-file verification now pass.
An initial float32 log/exp round-trip changed some known interior values by
rounding. The implementation was corrected to copy supplied values exactly;
the final smoke has zero difference in every covered cell. Saved eigenvalues
retain the trace identity within their expected float32 rounding. All 32
focused tests passed after that change. Both complete development phases were
then repeated at padding 32 without resampling their inferred interiors.

The refined fair CRPS is 0.05745935 learned, 0.05752063 conditional and
0.05791410 unconditional. The same spatial candidate wins, narrowly; no
significance or calibration claim follows. Learned eigenvalue 90-percent
coverage is 86.36/86.38/83.81 percent in phase 012 and
86.02/86.82/86.50 percent in phase 013. The periodic control retains its better
primary score. Keep it visible rather than presenting the selected spatial
candidate as the best performer across all controls.

For the final padding-32 development repeat, learned joint regional tidal
coverage is 37.5 percent in phase 012 and 62.5 percent in phase 013, or 50
percent across the 32 pairs. Joint regional mass coverage is 81.25 and 87.5
percent; individual regional-mass interval coverage is 81.25 and 90.625
percent. This is still a major joint tidal deficit against 87.879 percent.
It is not confined to the footprint edge: in phase 012, cells more than
27.064 Mpc/h inside support have eigenvalue coverage 85.05/85.68/82.93
percent. The corresponding phase-013 values are 86.78/87.62/86.77 percent.
Distance-bin figures are cell-count-weighted within each phase, with no
voxel-independent uncertainty estimate. They do not identify a unique cause
or license changing the frozen model after the final evaluation starts.

## Frozen production handoff

At 23:32 UTC on October 2, `SELECTION_FREEZE.json` pinned the selected learned
completion, 32 draws, NFE 128, padding 32, all eleven final mock phases, the
sealed phase 001 exclusion, input normalizer, checkpoints, training-only prior,
software versions and 856 runtime source files. Its SHA256 is
`6bd49651231448443fe171ad7699032fee7151dcf34a33caf7952484b0745232`.
The immutable source snapshot is `source_78156ac3417555ef269f` below the
campaign artifact root. Provenance explicitly records both Git HEAD and the
included uncommitted task files; this is not claimed to be a clean commit.

`configs/e2e_exterior_tasks_20261002.json` contains 192 unique four-draw units:
both caps of eleven mocks and Loa, each with 32 draws. Pilot job 59237546_1
runs phase 012 SGC draws 0–3 through density sampling, complete-density tidal
pushforward and independent saved-file verification. Job 59237550 prepares
the targetless phase 014–019 inputs. Main array 59238002 is queued with 191
units, excluding the pilot, and at most 16 concurrent GPUs. Its scheduler
dependencies require the successful full-NFE pilot and both qualified input
jobs. Earlier short-NFE timing remains an extrapolation until the pilot
completes. Legacy 004/005/006 rebuilding is complete and all 48 target products
were rehashed successfully. CPU allocation 59232476 completed and was released;
dependent job 59237994 prepares those phases' targetless full-cap caches.

Scoring job 59238010 depends on all production units. Website-preparation
job 59238614 depends on scoring and performs packaging, recoverable removal
of only superseded hash-verified generated cap chunks, and real-browser QA.
It does not publish. Its source and current Site state are separately pinned
by `WEBSITE_PREPARATION_FREEZE.json`, SHA256
`964f03c67caef8cb0095a22ee283b4c1a0afcf8925c191523db99b1e6eb09d1d`.
The runtime copy is `website_source_fa791612c371db5810bb`. Native private
publication remains a separate final action after successful preparation.
At this checkpoint the production jobs are queued without a scheduler start
estimate. No final full-cap coverage or new website publication is claimed.

Each production unit fails closed on incomplete outputs, changed source,
changed inputs or mismatched freeze. Completed outputs are reused only with
matching hashes and identities. Failed jobs are not automatically resubmitted.
The frozen final scorer tests the same registered cores within the entire
coherent cap draws, including joint 16-regional-mass and 48-regional-tidal
vectors per cap and explicitly approximate paired-index two-cap summaries.
The final summary must separate development reuse, exposed replication,
legacy non-blind replication and the two fresh confirmation phases.

## Native-resolution uncertainty products

### October 5 operational acceleration and interim evaluation

With explicit user approval, all 150 pending tasks (42–191) in array 59238002
were changed from four-hour limits to 150 minutes for NGC and 90 minutes for
SGC. Existing completed runtimes supported these limits with headroom. This
changes scheduling eligibility, not the calculations, source freeze, outputs,
dependencies or concurrency. The 80-GB constraint is still unchanged.

CPU allocation 59377709 completed successfully in 4m13s and was released.
The frozen final scorer was run separately on completed phases 012 and 013,
writing `interim_scores/ph012/SCORES.json` and `interim_scores/ph013/SCORES.json`.
Each includes all 16 registered two-core cases across both caps and all 32
draws, plus passing synthetic metric controls. Equal-sized core cases are
averaged within phase. These are reused development phases, not fresh tests.

| Nominal 90% diagnostic | Phase 012 | Phase 013 |
| --- | ---: | ---: |
| Density marginal | 87.244% | 88.608% |
| Eigenvalue 1 marginal | 86.208% | 85.803% |
| Eigenvalue 2 marginal | 86.297% | 86.861% |
| Eigenvalue 3 marginal | 84.120% | 86.638% |
| Regional-mass marginal | 78.125% | 93.750% |
| Joint two-core mass | 87.500% | 87.500% |
| Joint six regional eigenvalues | 37.500% | 50.000% |

The attainable reference is 87.879% with 32 draws. Marginal values describe
the registered cores within coherent full-cap draws, not a whole-cap voxel
census. The joint tidal deficit remains substantial. All four phase-cap
48-dimensional regional tidal vectors are outside their nominal 90% rank
region; all four 16-dimensional mass vectors are inside. These four dependent
cap outcomes are diagnostics, not a population coverage estimate. Final-phase
evaluation and existing calibration caveats are unchanged.

The separate 45-minute 40-GB GPU allocation is 59377708. Hardware-test source
`workflows/sbi/e2e_exterior_40gb_check.py` imports the immutable production
snapshot without modifying it. Four simultaneous full-size sampler draws use
NFE2 only as a memory/throughput test; these are not production samples.
The tidal test replays all four saved phase-014 density draws at frozen
padding32 and uses independent saved-file verification. Outputs remain under
`verification/hbm40_20261005_v1`, never the production directories.

The 40-GB test completed successfully in 16m16s; the GPU was released. Peak
allocated memory for NGC/SGC was 26.95/14.24 GiB for sampling and 35.25/19.92
GiB for tides. CPU comparison of every saved density and eigenvalue element
found zero difference from the 80-GB production arrays. The 24 targetless
input-cache headers show that tested NGC has the largest FFT geometry, but
Loa has more conditioning windows and a larger SGC canvas than tested. Only
phase 014 was executed on 40 GB; do not represent this as every phase tested.
The production 80-GB restriction has not been changed.

### Independent score audit and training-duration question

At the user's request, CPU 59378079 independently reconstructed both cores
from the registered layout and physical origins, assembled the true symmetric
tidal matrices and diagonalized them, and extracted every one of the 32 draws
for all 32 phase-012/013 cases. A separately implemented centered-covariance
and Cholesky calculation reproduced all stored joint ranks exactly. No core
offset, draw identity, ascending-eigenvalue or density/tidal trace discrepancy
was found. Known inferred interiors match exactly; the largest saved trace
residual is 4.77e-7. This is a targeted audit, not proof against every physical
approximation or implementation defect.

An independent 1,200-case six-dimensional Gaussian oracle gives 88.833%
nominal-90 joint coverage, compatible with its 87.879% finite-draw reference;
a 200-case underdispersed control gives 0.5%. The actual individual regional
mean-eigenvalue interval coverage is only 67.708% and 73.958%. Thus poor joint
coverage is accompanied by low marginal coverage of spatial averages: it is
not merely a product of six ordinary cellwise intervals. The joint statistic
is an exchangeable leave-one-out outlyingness rank, not an intersection of
six marginal 90% intervals. Audit source is `e2e_exterior_score_audit.py` and
the hashed receipt is `verification/score_audit_20261005_v1/AUDIT.json`.

| Same joint regional-tidal diagnostic | Phase 012 | Phase 013 |
| --- | ---: | ---: |
| Earlier local learned exterior, padding 32 | 37.5% (6/16) | 62.5% (10/16) |
| New coherent full-cap learned exterior | 37.5% (6/16) | 50.0% (8/16) |

Checkpoint hashes and NFE are identical between these two rows. Full-cap
integration, exterior context and random draws differ, so this is not a
paired estimate of a training effect. Earlier approximately 87–88% tidal
figures were cellwise marginal coverage, while earlier TARP curves tested
regional density vectors. Neither is the same diagnostic as the table.
There are only two independent development phases, not 32 independent skies.

The saved coarse regularization screen provides evidence against assuming
plain longer training will fix this: its baseline from 13,312 to 26,624
updates reduces development block spread from 0.031142 to 0.025632 while RMSE
barely changes (0.030244 to 0.029928). Block coverage drops from 87.5% to
79.30%, and core coverage from 87.5% to 73.44%, while training errors improve.
Source: `coarse_regularization_20260926_v1/SUMMARY.md` in the parent Scratch
directory; its matched checkpoint entries are retained in `COMPLETE.json`.
This supports a generalization/underdispersion concern, not a uniquely proved
cause of the new tidal failures. The selected coarse-13,312/fine-26,624 model
has not changed. A matched truth-density/operator-resolution closure test
and checkpoint-specific regional bias/spread comparison would distinguish
operator error from learned-posterior error more directly than more training
alone. No such new training or production retuning is authorized by this audit.

Postprocessing job 59241627 depends on the pilot and main production array.
It writes `loa_field_summary/{NGC,SGC}/DENSITY_TIDAL_SUMMARY.h5` under the
campaign root, with support and coverage masks, density and ascending tidal
eigenvalue means, sample standard deviations and 50/68/90/95-percent central
order-statistic intervals. These use the same finite-draw interval convention
as the frozen scorer, without Gaussian approximations or interpolated
quantiles. Native 6.766 Mpc/h class probabilities at thresholds zero and 0.2
and normalized class entropy are also retained. All 32 source draws must pass
identity, provenance and full-file hash checks before summaries are accepted.
Slabwise processing bounds memory; saved means also undergo a trace check.

These are pointwise model intervals, not simultaneous field bands. Real-DESI
coverage is unverified, and the two caps still use the explicitly independent
approximation. The postprocessor does not fit, retune or change the frozen
science pipeline. Its independent immutable source directory is
`field_summary_source_6fcbefe6429873b995c9`, with exporter SHA256
`bb26d28c6c1a3ead214f8eb91fa5dfeb07e539431abde3846979d8896e3626b6`.
Three tiny unit tests pass. A compute-node write/read smoke on 59241517
verified known 32-draw summaries for two synthetic caps, including all four
interval levels, classes, entropy and metadata; it read no observed fields
or mock truth. That allocation completed successfully and was released.
At registration the production exporter is queued; no actual Loa summary
or scientific success is inferred from its synthetic smoke.

## Delivery and provenance

Keep source and scientific decisions in this repository and generated products
under the registered Scratch root in
`configs/e2e_exterior_campaign_20261002.json`. Use measured interactive smoke
tests before appropriately sized, frozen batch jobs. Preserve unrelated work.

The final website update must use full-cap draws and their own derived tidal
and environment fields, not transplant earlier per-galaxy probabilities onto
new density samples. Preserve private access, original VAC products and real
mask holes. Display uncertainty and qualification limits, including any failed
or inconclusive calibration tests, rather than implying a publishable result
merely because a sampler produced a smooth field.

The planned display uses the density mean of all 32 draws, four browsable
individual draws, and environment fractions from native-cell eigenvalues in
all 32 draws. Classify first, then pool eight fine cells: the result is an
expected fine-cell volume fraction, not the class of a spatially pooled tidal
tensor. Compute entropy across draws separately at each fine cell and only
then spatially average it; entropy of the pooled class fractions would confuse
unresolved spatial mixture with posterior uncertainty. Tiny analytic controls
distinguish these cases exactly. These summaries are exploratory uncertainty
from the model, not empirically certified real-DESI class probabilities.
