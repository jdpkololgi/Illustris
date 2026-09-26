# Minimal context correction for the P12-A VAC

Status: authorized production chain submitted, not a promoted replacement. Original P12-A remains immutable. Active parent plan:
`plan_desi_p12a_vac_20260924.md`.

## Evidence and intervention

The actual epoch20 production encoder with nominal24-voxel context fails
registered context/subdivision gates. Changing only context to48 passes24
ph002 dense/sparse/edge cores. Keep the Planck18 Mpc chart, 5-Mpc grid, cap
separation, counts/exposure construction, frozen per-fold selection/scalers,
weights, epoch choices and seven-feature posterior definition unchanged.
Use48 context voxels and alignment8 consistently. Pin cuDNN TF32 on/matmul
TF32 off to reproduce the original export's precision. Treat this as a new
input protocol, not a claim that the original mock calibration never existed.

## Execution sequence and stop rules

1. Add an explicit export context option with legacy default24; serialize context,
   alignment, precision policy and source hashes in each completion record. No
   overwrites or reuse of halo24 arrays under a halo48 marker. Stage the candidate
   below a new Scratch root, not inside canonical P12 directories.
2. Verify the observer-only field reconstruction and candidate export on bounded
   training cores. Reproduce saved predictions under legacy24 before evaluating
   candidate48. Require exact parent-set/identity consistency and finite ordered
   base predictions; preserve response variables independently of changed base
   predictions. Time32 representative cores/phase before estimating full-export
   job sizes. Check peak GPU memory; use the smallest suitable GPU allocation.
3. Run the five omitted-phase exports with their own frozen contracts and
   checkpoint weights; run the full-fit ph006 validation export separately.
   Do not mix full-fit training predictions into posterior training. Preserve
   exposure/selection roles and all previously accessed-phase disclosures.
4. Build a new seven-feature response dataset with the exact P3b supported-row
   policy and random-support boundary feature. Reuse valid response caches only
   after parent/manifest/hash bindings match. Refit the posterior using the
   registered P12-A training/selection procedure, not a new architecture search.
5. Compare original versus candidate on registered validation/calibration and
   information criteria, including caps, shells, sparse and boundary populations.
   Reject or investigate any regression; do not choose thresholds after scoring.
   A numerical convergence pass alone is insufficient to select the candidate.
6. Independent confirmation must follow the exposure ledger. Do not reopen
   ph001 for tuning, relabel an exposed phase as blind, or use E2E reserved
   confirmation phases014-019. Resolve eligible confirmation identity before
   reading its payload.
7. Freeze the selected candidate and Loa observed-success/response contract,
   finish the observation-adapter golden replay, then notify the user of
   readiness for bounded DESI inference. Public VAC release retains the parent
   plan's replication, domain-shift and truth-free closure requirements.

Start with frozen encoder weights. Weight retraining is conditional on the
larger-context candidate failing scientific validation. The user has already
approved VAC compute; routine resource requests do not require another approval.
Use benchmarked ordinary Slurm batch for full exports/fitting, preserve the
two-interactive-allocation limit, and return for scientific decisions or failed
gates that materially change this correction path.

## Authorized run, 2026-09-24

The user approved regeneration, posterior refitting and revalidation. Candidate
root: `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_halo48_candidate_20260924_v1`.
All 718 source/configuration files are snapshotted and SHA256-verified before
each stage. The manifest and submission receipt are mirrored under
`docs/evidence/p12/HALO48_{RUN_MANIFEST,SUBMISSION_RECEIPT}_20260924.json`.
Original P12-A outputs and encoder weights remain untouched.

Benchmarks passed in all six phases: legacy-context parity plus 32 candidate
cores per phase selected across cap/shell/population strata. Median per-core
runtime is 0.205-0.246 seconds; peak allocated GPU memory is below 1.57 GB.
GPU reconstruction of the posterior returned 32 finite three-coordinate draws.
Seven exporter tests and six base-response tests pass.

| Stage | Job | Bound and dependency |
|---|---|---|
| Five OOF exports plus ph006 full-fit validation export | 58827028, array 0-5 | Two concurrent, one GPU each, 2h30m/task |
| Parent/response/hash validation, dataset preparation, FMPE refit | 58827034 | After all exports succeed; one GPU, 4h |
| Calibration diagnostics and comparison bundle | 58827036 | After fit succeeds; one GPU, 2h |

Each job requests 32 CPUs and 48 GiB host memory with shared GPU QOS. Invalid
Slurm dependencies are cancelled automatically; failed or partial exports are
not silently resumed or overwritten. Inspect logs and preserve a failure receipt
before creating a corrected version. Benchmark allocation 58825781 was released.

The fit retains seed42, hidden128/layers5, batch4096, early stopping20 and maximum
300 epochs. Dataset sampling, fold roles, seven features and response policy
remain the registered procedure. The historical report includes temperature
experiments; its `calibration_pass` alone is **not** a promotion gate. Compare
untempered coverage, physical-coordinate diagnostics, conditional errors and
information scores against the registered original criteria before selection.
The comparison output deliberately keeps `ready_for_desi_canary=false`.

Next after the batch chain: review scientific comparison, resolve independent
confirmation eligibility without opening reserved payloads, freeze the Loa
selection/response adapter and complete its golden-mock replay. Notify the user
when those gates permit bounded DESI inference; successful fitting alone does
not authorize a readiness claim. No real DESI inference is part of these jobs.


Scheduler recovery, 2026-09-24: ph000 and ph002 exports completed successfully
(9,956,825 rows total). Remaining tasks had a stale-looking JobArrayTaskLimit
reason despite no running task; refreshing the unchanged two-task throttle
cleared it, with tasks2/3 subsequently waiting on Priority. Existing export,
refit and audit job IDs/dependencies remain in place. No repeat export or
interactive allocation needed. See `docs/evidence/p12/HALO48_SCHEDULER_RECOVERY_20260924.json`.


## Optimized resubmission (supersedes original pending export IDs)

At the user's request, pending array tasks2-5 were replaced by ordinary named
jobs: ph003 **58831285**, ph004 **58831286**, ph005 **58831287** (after ph003),
ph006 **58831289** (after ph004). Each requests one GPU,32CPUs,16GiB and90min;
two may run together. The completed ph000/ph002 products are retained.
Actual completed export times60-65min and MaxRSS<2.7GiB support the tighter
requests, with runtime and memory headroom. Scientific configuration and frozen
source hashes are unchanged. Refit **58827034** depends on all four new jobs;
audit **58827036** follows successful refit. Jobs are accepted/released but
queue priority/resource availability still determines start time. Receipt:
`docs/evidence/p12/HALO48_OPTIMIZED_RESUBMISSION_20260924.json`.


2026-09-25: conditional coverage and eight-core observer golden replay pass;
Loa full-data/random content hashes verified. Supersedes pending replay status.
Next is bounded Loa input construction/QA then diagnostic inference; no further
retraining indicated. Full release and independent confirmation remain open.
See Illustris `docs/p12a_golden_conditional_results_20260925.md` and the explicit
bounded-trial handoff manifest. Real Loa inputs/posteriors are not yet produced.
