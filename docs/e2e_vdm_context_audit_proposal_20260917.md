# VDM diversity/context experiment: implementation audit and approval proposal

Final status2026-09-18: the subsequently approved implementation, bounded runs
and full assessment are complete. See [final results](e2e_vdm_context_results_20260918.md)
and [completion evidence](evidence/e2e_field_v2/vdm_context_20260918/README.md).
The audit-time approval request and pre-fit observations below are historical;
no additional budget or automatic follow-up experiment is implied.

Date: 2026-09-17. Status: **planning deliverable; approval requested before further
implementation or new allocations under this request**. No training was launched
by this audit. Existing jobs were inspected, not submitted, cancelled or modified.

Subsequent status: the user-provided active-goal continuation authorizes completing
implementation and bounded CPU/GPU execution. This audit is now incorporated into
the active experiment contract; its audit-time statements above/below remain
historical snapshots. Implementation and gate results are recorded in SCIENCE_LOG.

This is a continuation/review of the existing
[experiment contract](e2e_vdm_context_diversity_v1.md), not a second additive
resource budget. The live science log already records an earlier authorized goal
and CPU preparation. This document resolves the remaining design ambiguities and
lists work still needed before that matrix is safe to run. Existing source,
receipts, checkpoints and running frozen builders remain unchanged.

## 1. What the completed experiment establishes

Verified artifact:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_assessment_20260917_v2/analysis/RESULTS.json`
with SHA256
`89cf73cab5878ed89708f9585bbc50efad6e4da201cc099af108465928932cde`.
It contains all 3,584 registered draws: fixed/learned schedules, two optimization
seeds, checkpoints 512/2048/5120, and paired sampler refinements.

- All 16 recorded refinement screens pass. At 500 -> 1000 steps the largest
  power change is 0.590327%, its largest paired MC95 endpoint is 0.604442%, and
  the largest reported central-interval width change is 0.528116%.
- From 2048 -> 5120, density CRPS worsens in 6/8 model-by-patch comparisons.
  Density C90 falls in all eight: from 69.3--95.3% to 44.9--71.5%; posterior RMS
  spread also falls in all eight. Final power remains biased; correlation alone
  is not a success criterion.
- Example, fixed schedule/seed0/NGC: CRPS 0.09434 -> 0.10261, C90 94.9% -> 67.2%,
  spread 0.2379 -> 0.1010, despite substantially better correlations in the
  measured bands. The SGC result is worse, not a replication of NGC calibration.
- These are two cutouts from **one development phase**, not eight independent
  universes or a blind posterior-calibration validation. Old empirical CRPS and
  interval conventions remain attached to those historical results; new fair
  scores must not be silently substituted in an old/new numerical comparison.

Conclusion: sampler discretization is small for these frozen checkpoints and
tested observables. It has not been excluded universally for future models or all
joint statistics. Reuse this assessment; do not repeat the old draws. Neither H1
nor H2 is already established, and the results do not support blind continuation
of the same small-data experiment.

## 2. Geometry, information and estimand audit

All lengths below are comoving Mpc/h; wavenumbers are h/Mpc, with k_f = 2*pi/L.

| Object | Grid | Cell spacing | Physical side | k_f |
| --- | --- | ---: | ---: | ---: |
| Original sampled R7 density parent | 96^3 | 3.383 | 324.768 | 0.0193467 |
| Direct VDM generated density parent | 48^3 | 6.766 | 324.768 | 0.0193467 |
| Common central science output | 16^3 | 6.766 | 108.256 | 0.0580401 |
| Existing wide observational grid | 96^3 | 13.532 | 1299.072 | 0.00483667 |
| Proposed spatial context / coarse matter | 48^3 | 27.064 | 1299.072 | 0.00483667 |

Target: the 2x2x2 physical-density average of cubic-sampled, Gaussian-R=7 density
at matter epoch z=0.2; then log(1+delta) and pooled training-only affine scaling.
The averaging window is part of the target; do not smooth by R7 a second time.
The central core has a 108.256 Mpc/h fine-parent halo on each face.
Observer redshift 0.15--0.55 changes selection/geometry, not the matter epoch.

`e2e_direct_experiment.observation()` supplies **24 channels**, not a purely local
catalogue: 12 spatial local channels plus 12 spatially broadcast wide means.
Local channels are counts, random support, angular response, apodized exposure,
expected counts, log count/expected ratio, support-boundary distance, n-tilde,
three line-of-sight components and observer redshift. Wide channels contain
counts, expected counts, support, angular response, exposure, n-tilde, geometric
validity, count ratio, line-of-sight components and observer radius.

The old model loses **wide spatial arrangement**, not all information about the
wide footprint. The new wide encoder pools transformed observation channels to
48^3 at the same physical extent. Retain the existing transforms; averaged
log-count channels are not literal raw counts, and label their diagnostics so.
Padding observational arrays outside the survey is nonperiodic and carries
support/validity information; it must not create observed galaxies by wrapping.

No per-patch target mean/std fitting or density-DC subtraction occurs in the
direct model. Pooled affine normalization is invertible. GroupNorm inside the
network is not evidence of data-level DC removal. Cropping limits available
modes, and averaging attenuates small scales. Spectral diagnostics subtract a
window-weighted mean, so regional mass/DC diagnostics are essential separately.

Keep three issues distinct:

1. Unobserved galaxies: a correct conditional density posterior can marginalize
   their uncertainty; missing context does not mathematically force undercoverage.
2. Exterior matter: density samples on the central parent do not specify the
   external contribution to its physical tidal field.
3. Tidal boundary closure: periodic/zero/reflect padding is an approximation,
   not recovered external matter. Even D omits matter beyond the wide domain.

Survey limitation: the frozen P3b response represents angular support/targetability;
it is not an audited DESI fibre-assignment/redshift-success response. This remains
a conditional-on-mock, fixed-cosmology/HOD/epoch experiment, not real-DESI release.

## 3. Resolved comparison and phase roles

| Arm | Distinct training cores / phases | Observation information | Generated matter |
| --- | --- | --- | --- |
| A: small current-context | 32 / ph000, ph002 | local fields + wide means | fine parent |
| B: diverse current-context | 384 / ph000, ph002, ph003 | same as A | fine parent |
| C: diverse spatial-context | same 384 | local fields + spatial wide context | fine parent |
| D: diverse multiscale | same 384 | same available observation products as C | shared coarse + conditional fine |

Fresh fits are required. A is a **balanced small-data control**, not an exact
rerun of the old ten-NGC-cutout training distribution. Both caps, all four
redshift shells and both support strata are represented in A and B; keeping the
old biased selection only in A would confound diversity with coverage of the
observation distribution. Historical checkpoints provide motivation, not a
matched A arm. All four new fine models use the same context-capable architecture
and initialization policy; A/B collapse spatial wide inputs to means internally.

A is nested in B. B has 3 phases x 2 caps x 4 shells x 2 support strata x 8 anchors.
The geometry screen has already found all 416 primary training/evaluation anchors
under the unchanged quotas. Core centres must be separated by at least 162.384
Mpc/h in the periodic source box, including NGC/SGC aliases; central cores do not
overlap. Parent/context overlap within a phase is recorded, not called independent
data. Seven context offsets (0 and +/-108.256 along each axis) are augmentation,
not seven additional cosmic fields. No retrospective quota/distance relaxation.

- Train: ph000/ph002/ph003. ph003 has been used for development previously; it
  is now explicitly training data and cannot be reported as held-out evidence.
- Development: ph004, 16 geometrically selected anchors.
- Internal confirmation: ph005, 16 anchors, predictive scores unopened until the
  full model/protocol freeze. These phases have programme history and are **not
  globally untouched blind tests**.
- Forbidden: ph001 and ph006. No new access to their data, labels or predictions.
- Fit all shared preprocessing and scoring scales on A32 only; D-only charts on
  those same training fields. No eval moments or truth-dependent anchor selection.

A:B tests increased training diversity jointly in patches and parent phases,
not a pure independent-universe count effect. It grows two phases to three, not
to hundreds. B:C isolates spatial arrangement at fixed data/architecture. C:D
tests a multiscale **package**, not an unconfounded capacity or stochasticity
effect. This minimal study cannot estimate every data-by-context interaction.

## 4. Model and shared-draw contract

Retain full VLB, fixed gamma in [-13.3,13.3], the residual 3D U-Net with bottleneck
attention, AdamW (lr=1e-4, weight decay=1e-5), clip=0.5, batch=2 and deterministic
FP32. Choose fixed scheduling once to avoid another schedule factorial; this is
a control choice, not a claim that learned schedules are inferior.

Two optimization seeds (0/1); 20,480 updates and 40,960 presentations **per factor**.
Checkpoints 5,120/10,240/20,480; durable checkpoints every 256 updates. No equal-epoch
comparison. Ten factors total: eight fine models and two D coarse models. Current
counts are 8,371,907 parameters per fine model and 8,188,395 extra per D coarse model.
D therefore gets extra parameters and compute, which must remain visible.
No new identity loss, EMA, AMP, transformer, CFM or optimizer sweep.

Physical split, with rho=1+delta:

    rho_c = blockmean_4(rho_f)
    u = log(rho_f) - blockmean_4(log(rho_f))
    rho_f = rho_c * exp(u) / blockmean_4(exp(u))

Use stable block softmax for decoding, positive physical density, and exact block
mass. Project D fine states/noise/predictions onto the 63-dimensional zero-mean
subspace of each 64-voxel block; use the corresponding prior/decoder/VLB degrees
of freedom. The old Fourier-lowpass product cannot simply be renamed rho_c.
Different chart likelihood values are not directly comparable across C and D;
compare their decoded physical predictive distributions.

An important audited approximation is retained explicitly in this minimal pilot:

    q_D(rho_c, u | X_local, X_wide, S)
      = q_c(rho_c | X_wide, S_wide)
        q_f(u | rho_c, X_local, X_wide, S).

The existing coarse trainer receives only the 12 wide channels. This is a valid
restricted conditional generative family, **not a general exact factorization**
of the posterior given every local observation. It assumes the coarse observation
summary is sufficient for coarse mass; that may fail. Record this restriction
and assess coarse mass/calibration. A negative D result cannot rule out coherent
multiscale posteriors generally. Do not silently describe this as a complete
implementation of p(rho_c | all fine observations). Expanding that conditioner is
a separately justified next experiment if this restriction becomes decisive.

Inference readers may open observations only. True coarse density is legitimate
supervised conditioning during fine training, but not an inference answer. The
oracle uses a separate labelled diagnostic path. A sampled parent is keyed by
model/seed/checkpoint/domain/draw/solver, cached and reused by both adjacent cores.
Fine random noise belongs to each owned core; sharing an identically indexed noise
array across shifted cores would create artificial fine dependence. Retain the
explicit approximation that fine cores are independent conditional on the parent;
shared coarse alone is not a full-survey joint-posterior guarantee.

For physical tidal diagnostics use
`U T_wide(delta_c) + T_parent(delta_f - U delta_c)` with the same declared block
replication U. Report this separately from matched-patch periodic/zero/reflect
closures and the full-box tensor reference. Average six tensor components before
diagonalizing, never average eigenvalues. C:D changes this physical closure too;
report both arms under the common patch closure as well as D's composite result.

## 5. Evaluation and decision rules fixed before new predictive outcomes

Use the registered 688-task ledger: **33,280 fine draws + 7,872 distinct coarse
draws**, including refinements and attribution controls. No extra fits.

- At 5,120 and 10,240: 32 draws on each of 16 ph004 anchors.
- At 20,480: 64 draws on all 32 ph004/ph005 anchors, increased to 128 on eight
  geometrically selected sentinels. Final checkpoint is primary, not the best
  checkpoint chosen on confirmation.
- Two fixed training anchors: 32 draws at all checkpoints.
- Four adjacent-core pairs (one per cap/evaluation phase): 32 joint draws.
  D additionally uses 32 fixed-mean-coarse and 16 true-coarse oracle draws per
  core, sharing the appropriate fine random numbers. The mean uses the registered
  sampled coarse ensemble, not a new fit or additional parent ensemble.
- Two development anchors per model: eight coupled draws at 250/500/1000 steps;
  reuse the 250-step draws. New-model refinement is necessary, not a repetition
  of the completed old-checkpoint test.

Measure sample power, power of the posterior mean, residual power, correlations,
posterior-mean RMSE, one-point distributions, regional masses, bias/spread,
C50/68/90/95/ranks and fair finite-ensemble CRPS. Add ordered eigenvalues/gaps,
standardized joint energy scores and cross-core summary energy/variogram scores.
Wide low-k power is assessed on D's 1299 Mpc/h domain; absent A--C quantities are
N/A, not inferred from a 108 Mpc/h crop. Never demand individual samples reproduce
their paired truth's spectrum or correlation one.

Use train-only observation-stratum edges: log-count proxy, support, angular
response, n-tilde proxy, observer redshift and boundary distance. Truth environment
bins are descriptive. Report observed/unobserved regions separately. Equal anchor
weights within the balanced panel; show all seed-by-phase cells. Paired 500 Mpc/h
source-block bootstrap with a 1000 Mpc/h sensitivity is descriptive fixed-phase
uncertainty, not a substitute for independent simulation phases or full-field SBC.

Resolve the previously ambiguous success threshold as follows:

1. Primary score reduction >=10% on the equally weighted pooled seed/phase
   comparison **and a positive reduction in each of the four seed-by-phase cells**.
   A:B and B:C use standardized density CRPS; C:D uses physical tidal joint energy.
   The present report code instead requires >=10% in every cell; amend it and test
   heterogeneous-but-consistent fixtures before any predictive score is opened.
2. In each seed/phase cell, reduce the relevant C90 coverage gap by >=5 percentage
   points, or finish within 5 points of its finite-ensemble target. For tides,
   compute absolute gaps for each eigenvalue before averaging, preventing under-
   and overcoverage of different eigenvalues from cancelling. Show every component.
3. Per-cell density-mean RMSE must not worsen by >5%; the aggregate sample-power
   discrepancy must not worsen by >0.05 absolute log units. Aggregate expected
   sample spectra across anchors, not a demand for every draw to match truth.
4. A positive multiscale-package result also requires better joint cross-core
   energy/variogram scores. Attribute a gain specifically to **stochastic coarse
   uncertainty** only with supporting D versus frozen-mean-coarse proper-score and
   calibration evidence. D's oracle is diagnostic and never a contender.

Report partial/mixed/inconclusive when these do not agree; do not promote a score
fluctuation hidden by averaging. Four pairs/two phases cannot establish population
joint calibration, even if the point-estimate screens pass. No requirement that
every metric improve monotonically at every checkpoint.

Sampler gates: 250->500 power/width change <5%; 500->1000 <2% with paired MC95 upper
bounds <5%. With eight draws the attainable central interval is 7/9, **not 90%**;
label it honestly. Extend the same already-generated refinement comparisons to
D wide power and the tidal/gap interval widths before main sampling; no additional
draw budget is needed. Retain sample-mean-power MC bias information (or report the
variance/M correction) when comparing 32-, 64- and 128-draw panels.

## 6. Implementation audit and mandatory preflight work

Observed working tree: commit `9683a21` plus uncommitted sampling, analysis, report,
control, CPU launcher and GPU smoke modules, and dataset/train edits. Do not
mistake uncommitted code for a frozen tested production snapshot.

The audit reran 41 focused/regression tests successfully (3.567 seconds reported
test time). They cover model/geometry algebra, phase guards, exact optimizer/RNG
resume, corruption rejection, shared-parent addressing, chunk resume, fair scores
and Gaussian/log-Gaussian finite-ensemble coverage. This is not a full-size GPU
test or a learned-posterior result.

| Milestone | Verified status / required action after approval |
| --- | --- |
| Geometry and role ledger | 416 primary anchors pass; preserve identities and quotas. |
| Physical input products | ph000/ph002/ph003 receipts present; mass/trace errors <=2.67e-15; evaluation products still building at audit. Hash-verify payloads on compute. |
| Representation and normalization | Not run. On A32 only: positivity, round-trip/mass/trace <=2e-6, exact declared plane/DC transfer, >=25% median per-anchor reduction in each normalized eigenvalue RMSE versus parent-only closure. Stop if it fails. Then freeze A32 charts. |
| Report contract | Implement the pooled threshold, componentwise coverage gaps, explicit coarse-conditioning restriction and same-draw tidal/coarse refinement checks above. Add tests before opening validation scores. |
| Confirmation access | Sampling calls the strong full-matrix freeze verifier; reporting currently relies on a weaker flag/geometry check. Require the strong verifier before report-time ph005 target access too; test tampering/missing branches. |
| Compute preflight | Commit/freeze source and config; full-size GPU training/sample parity, exact reload, observation-only inference, generated-coarse mass test and actual signal/pause/process-resume test. Small replay tests alone do not establish interruption safety. |
| Resource controller | Complete and test the finite GPU launcher: aggregate actual allocated GPU time including idle time, request count, retries, elapsed deadline and storage cap; no automatic scientific redesign. CPU launch helpers are not the complete GPU controller. |
| Report scalability | Benchmark representative CPU scoring before committing all draws: fair joint scores are pairwise in ensemble size and repeated across closures/masks. Cache/reuse calculations; parallelize bounded CPU work. Do not omit diagnostics to conceal overruns. |
| Final evidence | Implement figures and final expected-versus-actual receipt/draw-count audit. Preserve failed gates, runtimes, hashes, checkpoint progression, H1/H2 assessment and one next justified step in the science log/model plan. |

Existing preparation is real: geometry job 58469111 completed in 57s; ph000/ph002
builder 58469318 completed 0:0 in 50m21s. Remaining-phase builder 58470146 was still
running at audit, with ph003 complete. It is a frozen CPU data job, not neural
training. No new GPU training, sampler release or representation-gate pass exists.
This audit did not start, stop or alter those allocations.

## 7. Resource proposal and stop conditions

Recalculation using the exact current ledger and the **old-model**, not new-model,
throughput (0.18 s/update; 5.5 s/batched 250-step draw):

| Component | Baseline-equivalent GPU-hours |
| --- | ---: |
| Ten fits x 20,480 updates | 10.24 |
| All fine/coarse sampling, step-weighted | 63.85 |
| 15% scheduling/I/O/recovery allowance on both | 11.11 |
| Full-size smoke/controller allowance | 4.00 |
| Total extrapolation | **89.20** |

Use **87--99 GPU-hours as a provisional range, not a measured promise**. New
context attention, hierarchical conditioning, cache/serialization and GPU-memory
limits must be measured. Sampling dominates the budget. The measured five-model
smoke recomputes the full forecast before any expensive matrix fit; stop if it
cannot fit the ceiling. Benchmark CPU scoring as well.

Approval envelope, inclusive of existing work in this experiment root:

- **112 allocated GPU-hours hard cap**, including idle/recovery; no banking a
  full-node charge as one GPU's kernel time.
- **8 CPU-node hours total**, including builds, physical gate, analysis and any
  retry; do not reset the consumed ledger on continuation.
- **300 GiB additional Scratch** within the named experiment root. One float64
  scalar field for every registered fine/coarse draw is about 34 GiB before
  compression; checkpoints, inputs, caches and analysis are additional. This is
  not permission to duplicate native full-box data or alter CFS sources.
- **48 hours elapsed including queue waits** from the existing first request:
  2026-09-17 14:51:16 UTC -> **2026-09-19 14:51:16 UTC**. If approval comes too late,
  stop and request an explicit revised deadline; do not silently restart the clock.
- At most **8 GPU allocation requests**, smoke/tail/replacement included; at most
  **4 hours per allocation** and one full four-GPU node at a time. Use independent
  workers on all four GPUs; use shared allocations for a one-/two-GPU tail.
  At most two pending/running interactive allocations total, including CPU.
- One bounded **infrastructure replacement**, only with verified resumable state
  and remaining budgets. Clean planned wall-time continuation is not a failure
  retry. An unexpected numeric/hash/scientific-gate failure stops for review;
  no automatic objective, grid, step-count or draw-count changes.

Execution preference: fixed, reviewed `salloc`/`srun` commands supervised through
tmux, with atomic checkpoints and chunk receipts in Scratch. The user's explicit
interactive preference takes precedence over the skill's normal batch-handoff
recommendation. Only deterministic authorized work survives disconnect; no agent
is left making resource or scientific decisions. tmux does not extend Slurm time.
Use desi_g/desi accounts, explicit GPU requests on allocation and steps, Scratch
license, and --immediate=600. Choose near-four-hour windows for the long measured
sampling work; do not inherit the unrelated old 75-minute allocation split.

The [NERSC QOS table](https://docs.nersc.gov/jobs/policy/) checked for this audit
allows four-hour interactive jobs and two submitted/running jobs; the
[interactive guide](https://docs.nersc.gov/jobs/interactive/) requires explicit
GPU resources on srun. These maxima are not a guarantee of immediate availability.

## 8. Approval-ready goal and completion definition

Proposed approval text:

> Complete implementation and execution of the audited A/B/C/D VDM experiment,
> including the preflight corrections and gates in this proposal, two seeds,
> fixed exposure, the registered posterior-draw ledger and final H1/H2 assessment.
> Approve a total ceiling of 112 GPU-hours, 8 CPU-node hours and 300 GiB, with the
> existing elapsed deadline, at most eight GPU requests of up to four hours and
> one bounded infrastructure replacement. Use all four GPUs on full nodes and
> durable interactive execution. Stop at failed gates or resource limits; keep
> ph001/ph006 sealed. Update SCIENCE_LOG and the field-posterior plan with verified
> results and the single next justified step.

Matrix completion means saved, hash-verified artifacts and an honest
distributional assessment, not finding a preferred winner. If a predeclared gate
fails, preserve and report that valid negative result and identify the matrix as
incomplete; do not claim that the full experiment ran. No production claim, DESI
catalogue inference, further architecture search or unbounded extension is
included. A negative gate is not a licence to repair the experiment until it
passes.
