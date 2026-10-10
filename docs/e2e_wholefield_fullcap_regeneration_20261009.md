# Full-cap matched baseline: execution authority and resource preflight

## Current parallel continuation: both interactive slots occupied

User requests the spare interactive capacity and asks not to routinely monitor
a normally progressing run. s010/59644879 remains running on nid200264 with one
GPU, ph016/NGC/seed17/d008; its log reaches coarse step 8. Second allocation
s011/59645231 starts on nid200424 with four independent single-GPU workers,
claiming d012, d016, d020 and d024 of the same phase/cap/seed. Startup receipts
and a one-time device check verify four ranks on four distinct physical GPUs.
The frozen sampler and exclusive task-claim logic are unchanged; eight focused
parallel/contract tests pass. No failed claim is retried or overwritten.

This is five concurrent tasks over exactly two allocations, not five completed
tasks. Each allocation has a three-hour limit; s011 reserves at most 12 GPUh.
Accounted use plus both live reservations is 44.418611 of 250 GPUh; actual use
must be reconciled after termination. No batch or third allocation is submitted.
No routine polling or automatic follow-on allocation is configured. This user
instruction supersedes the continuous-monitoring procedure below. Interactive
session teardown remains a disclosed failure risk, not repaired persistence.

## Current user direction: supervised interactive run

User rejects the overnight batch handoff and explicitly requests interactive
jobs. Commit 89938c1 adds a source-bound `supervised-interactive` mode, tested
alongside the batch/smoke paths (20 focused tests pass). No batch is submitted.
The numerical worker and failed claims remain unchanged. This changes the
supervision procedure, not the app's disconnect behavior: retain an active agent
turn and monitor through verified completion and allocation release. Do not end
the turn with a live unattended interactive job. Connection/session loss remains
a residual risk; no persistence claim or detached process is introduced.

s010 / 59644879 starts on nid200264 with one GPU and a three-hour maximum,
using shared_interactive and the tested shell/entrypoint. It claims the next
untouched task, ph016/NGC/seed17/d008. Actual use before launch is 29.418611h,
plus a 3h live reservation; no scale-up before verified completion. All gates,
the 250h ceiling and at-most-two-allocation limit remain fixed.

## Repaired launch ownership and verification, 10 October

The investigation found a shared launch-session failure: the runtime recorded
teardown of this thread's listener (no active turn), an explicit Shutdown
submission and session shutdown at **16:32:14 UTC**. Both Slurm steps died at
**16:32:16 UTC**. This strongly identifies app-session teardown as the common
trigger. The inspected records do not explain why the app requested shutdown.
There is no reported OOM/node-failure reason or Python exception in the eight
sampler logs. This is not evidence that the numerical sampler needs changing.

The foreground change was insufficient because the salloc/srun processes still
belonged to the app session. Repair 99b026b provides a scheduler-owned batch
shell that execs srun and retains durable Slurm stdout/stderr. Its entrypoint
binds/checks launcher source and reservation hashes, requires batch ownership for
production, propagates worker errors, and rejects an empty completed-task list.
There is no background login process, automatic retry, requeue, submission loop,
failed-claim overwrite or change to frozen numerical sources.

Verification:

- Eighteen focused tests and shell syntax checks pass.
- s008 / 59644691: real CUDA startup succeeds; allocation releases in 12s,
  COMPLETED 0:0, with BATCH_COMPLETE.json explicitly marked non-scientific.
- s009 / 59644697: injected error writes BATCH_FAILED.json, exits through srun
  with code 1 and releases in 21s, FAILED 1:0 as expected.
- Both use one GPU and make no generation claims. Total actual campaign use
  is now **29.418611 / 250 GPUh**. No allocations remain active.
- Slurm accepts the actual repaired script in a non-submitting dry run,
  estimating 09:14:32 UTC (10:14:32 BST) on 11 October; no guaranteed start.

The interactive smokes establish startup and error propagation; they do not
prove survival of app shutdown. Moving production ownership to Slurm removes
that dependency. No production batch has yet run. Explicit acceptance of the
long queue is requested before a single-GPU, three-hour validation; further
scale-up remains gated on verified completion and accounting. The bank is
unchanged at 57/192 tasks. The failed claims remain preserved.

Sources: local runtime lifecycle events and sacct records; launcher
`workflows/sbi/e2e_wholefield_fullcap_batch.{py,sh}`; receipts in
`fullcap_v1/allocations/s008/` and `s009/`. The
[OpenAI tool guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide)
describes exec sessions but does not promise persistence through session shutdown.
[NERSC interactive documentation](https://docs.nersc.gov/jobs/interactive/)
describes the salloc/srun ownership model. Neither public source establishes
the incident cause; that comes from the local timestamped evidence above.

## Current checkpoint: parallel interruption, 10 October evening

The two parallel segments did not complete. Jobs 59633119 and 59633169 have
allocation state TIMEOUT and DerivedExitCode 0:9 despite allocation ExitCode
0:0. Both compute steps were cancelled with signal 9 at 16:32:16 UTC. Each
of the eight logs ends at fine step 56; each attempted task contains only
START.json and density/BINDING.json, with no payload, restart checkpoint or
COMPLETE.json. No worker/rank completion or failure receipt exists. The shared
termination time is suggestive of an external event, not proof of a cause.
The foreground wrapper did not prevent stranded allocations on this path.

| Segment | Actual allocated GPU-hours | Outcome |
| --- | ---: | --- |
| s006 / 59633119 | 12.017778 | Compute killed; allocation timed out |
| s007 / 59633169 | 12.015556 | Compute killed; allocation timed out |
| Campaign cumulative | **29.409444 / 250** | 57 complete, 135 still missing |

Accounting includes the idle tails. Both ACCOUNTED.json receipts are saved.
No allocations remain active. Existing completed products are unchanged;
preserve all failed claims/logs without retry or overwrite. Gate 2 and training
cannot advance until the required bank is complete. No numerical gate result
is inferred from this infrastructure failure.

Stop further use of this interactive launch path. The latest non-submitting
three-hour, one-GPU shared-batch estimate around 20:09 UTC is 09:12:31 UTC
on 11 October (10:12:31 BST), about 13 hours away and not guaranteed.
Request acceptance of that wait for one bounded single-GPU batch validation;
the user's existing batch permission was conditional on a reasonable wait.
No new job or retry is submitted in this checkpoint. Wider batch continuation
requires verified completion and reconciled remaining-budget admission.
The earlier launch sections below are historical, not current success claims.

## Validation passed and parallel continuation, 10 October

The foreground validation job 59628287 completed successfully at 13:29:38 UTC
(14:29:38 BST), released normally, and used 1.761389 GPUh. It completed four
ph014/NGC seed-29 draws, d004--d007, with the frozen NFE128 sampler. Density
mass error is 4.87e-16. The saved tidal-field verifier passes 159,252,480 interior
voxels, with zero interior density difference and trace residual 4.77e-7.
The measured sampler peak is 26.95 GiB GPU memory.

Corrected-core extraction 59633091 also completed and released normally in
58 seconds (0.016111 GPUh): original-tide replay error 9.53e-8 and correction
trace residual 8.98e-15. There are now 57 completed/extracted task products,
including the 56 reused ones, and 135 generation tasks remain. This does not
establish the complete baseline, Gate 2 or posterior qualification.

Total actual GPU allocation use before parallel continuation is **5.376111h**.
A fresh shared-batch dry run at 14:54 UTC estimates 02:47:25 UTC on 11 October,
an approximately 11h53m wait. Continue with the user-approved interactive fallback.

The parallel launcher (5a55d97, atomic startup barrier correction 59ff989) retains
the unchanged frozen numerical worker, and binds its own source in each rank's
receipt. Four independent ranks use one GPU apiece and atomically claim distinct
tasks. One foreground srun owns the ranks; failure propagates to the allocation.
Seventeen focused tests pass. No automatic retries or allocation-submission loop.
The full node is justified by four ready independent tasks and the measured
single-worker memory footprint; this is not a distributed-model rewrite.

Segment s006, job 59633119 on nid200404, starts four workers with a three-hour
limit (12 GPUh reserved). Startup verifies four distinct task claims and four
different physical GPU UUIDs via a read-only device diagnostic. After that
check, s007 starts as job 59633169 on nid200324 after a short queue wait, using
the second permitted allocation for four further workers, also limited to three
hours. All four ranks have passing startup records. Combined reservations plus prior actual use are
29.376111 GPUh against the unchanged 250h ceiling. No third allocation, sealed
phase, training run or scientific advancement is authorized by this launch.

Evidence: `fullcap_v1/allocations/s004/ACCOUNTED.json`,
`fullcap_v1/allocations/s005/ACCOUNTED.json`, the d004 generation and extraction
receipts, and `fullcap_v1/allocations/s006/` and `s007/`. The failed d000 attempt
remains preserved and is not silently retried. Each segment must be reconciled
after termination before further resource decisions.

## Queue-based continuation, 10 October

User authorizes batch execution if queue waits are reasonable, otherwise
sequential interactive segments. At 11:42 UTC, `sbatch --test-only` for one GPU,
three hours, `shared`, `desi_g`, Scratch access and 80GB A100 memory estimated
a start at 23:51:40 UTC: about 12h09m later (00:51:40 BST on 11 October).
The same request without the 80GB constraint estimated 23:51:41 UTC. These
were non-submitting scheduler estimates, not guaranteed start times. The
initial host-local timestamp was rechecked explicitly in UTC before comparison.

Selected the authorized interactive fallback because this wait is long relative
to the bounded validation run. Segment s004, job 59628287, started promptly on
nid200256 with one 80GB GPU and a three-hour limit. Actual use before this segment
is 3.598611 GPUh, plus its 3 GPUh reservation against the unchanged 250h ceiling.
The new launcher, committed in 151ac06, runs coordinator and worker inside one
foreground compute step; no login-host background workers, automatic resubmission
or unattended interactive-allocation chain. Three launcher lifecycle tests and
12 existing contract tests pass. Frozen numerical generation sources and all
existing outputs are unchanged; the failed d000 claim remains preserved.

This is a started technical validation, not a successful completion. Further
generation scale-up requires verified task completion; Gate 2 and training
remain downstream. Source/status authority:
`workflows/sbi/e2e_wholefield_fullcap_foreground.py`,
`fullcap_v1/allocations/s004/`, and `fullcap_v1/logs/s004/`.
Relevant scheduler references: [NERSC scheduling](https://docs.nersc.gov/jobs/scheduling/)
and [QOS limits](https://docs.nersc.gov/jobs/policy/).

## 10 October execution checkpoint

Extraction finished, but generation did not. Independent CPU audit 59628225
verifies all 56 available corrected-core products: payload hashes and bindings,
physical identities, exact summary reconstruction and 32 exact corrected-core
replays against the earlier products. All 21 focused tests pass. This is an
engineering verification, not a partial-panel sensitivity result.

Generation job 59606367 has allocation state TIMEOUT. Its compute step was
cancelled with signal 9 after 48m47s, while the allocation remained reserved for
3h00m09s. The last sampling log is coarse step 64; no new density field or
completed task was saved. Available Slurm records and logs do not establish
the termination cause. The old launcher did not release its allocation after
this step termination. Preserve the failed claim and partial directory; no
automatic overwrite/retry or concurrent generation scale-up is authorized by
this technical outcome.

| Segment | Outcome | Actual allocated GPU-hours |
| --- | --- | ---: |
| s001 / 59606367 | Interrupted step; allocation timed out | 3.002500 |
| s002 / 59606430 | Initial extraction replay passed | 0.017778 |
| s003 / 59606480 | Remaining available extraction passed | 0.578333 |
| Total | 56 extracted products; 136 generation tasks outstanding | 3.598611 |

The 250-hour ceiling, original deadline and scientific gates remain unchanged.
No new GPU job has been submitted during this checkpoint. A switch from the
standing interactive-only restriction to bounded Slurm batch execution has been
requested explicitly: one single-GPU validation job first, then wider execution
only after verification. It is pending user approval, not an automatic handoff.
Without the missing fields the complete baseline and Gate 2 remain unavailable;
no training, sealed-phase access or posterior qualification follows.

Audit: `fullcap_v1/CHECKPOINT_AUDIT_20261010.json`; accounting:
`fullcap_v1/allocations/s001/ACCOUNTED.json`, `s002/ACCOUNTED.json`,
`s003/ACCOUNTED.json`, all under the generation root recorded below.

## Current decision

On 9 October the user chose to retain the exact registered full-cap corrected
operator and explicitly authorized field regeneration, GPU use and training as
needed to obtain results. This resolves the operator/data-scope choice. It does
not waive the sensitivity/Gaussian gates, sealed-phase restriction, or the
earlier explicit resource ceiling without a numerical amendment. The user has
now explicitly approved **250 total allocated GPU-hours**, with the deadline,
interactive-only execution, two-allocation limit and scientific gates unchanged.
This authorizes bounded execution; the preflight itself launched no compute.

The target is now a full-cap baseline over ph012--017, model seeds 17 and 29,
32 draws per cap, and the same 16 pairs per phase. This matches the intended
phase/seed/region panel but is not the original pair-local draw distribution.
Keep the physical estimator unchanged: learned exterior completion, padding 32,
free-space full-cap tides and the registered conservative-linear local density
reconstruction correction. Do not substitute periodic patch tides or an
augmentation continuation. Frozen EMA checkpoints remain coarse 13,312 and
fine 26,624. No new baseline training is required.

## Current inventory and net cost

The bounded preflight inspected JSON receipts and file metadata only, not field
or target payloads. Twelve targetless phase/cap input caches are present, with
plan hashes matching their prepared receipts. Reuse candidates have passing
four-draw receipts and existing payloads of the registered sizes. They still
need source/checkpoint and payload-hash verification on compute nodes before
reuse. Incomplete directories are not counted and will not be overwritten.

The two-seed/six-phase panel has 192 four-draw cap tasks. There are 56 completed
density/tide task pairs available: 51 seed-17 tasks in the original full-cap
campaign and five seed-29 SGC tasks in the stopped mixture campaign (three in
ph012, two in ph013). **136 density and 136 tidal tasks remain.**

| Phase | Seed-17 reuse / 16 | Seed-29 reuse / 16 | Net GPU-hours, median rates | Net GPU-hours, maximum observed rates |
| --- | ---: | ---: | ---: | ---: |
| ph012 | 16 | 3 | 18.14 | 20.57 |
| ph013 | 16 | 2 | 19.12 | 21.65 |
| ph014 | 14 | 0 | 22.76 | 25.68 |
| ph015 | 5 | 0 | 33.26 | 37.48 |
| ph016 | 0 | 0 | 42.09 | 47.57 |
| ph017 | 0 | 0 | 41.89 | 47.35 |
| Total | 51 | 5 | **177.26** | **200.31** |

These are planning estimates, not new measured runtimes, statistical intervals,
or guaranteed upper bounds. For each cap/product, use the median or maximum
observed seconds per window in completed receipts, multiplied by the prepared
window count for each missing task. Density timing includes sampling/loading;
tidal `seconds` already includes its setup, which is not added a second time.
The initial v1 metadata artifact double-counted that setup; v2 corrects this
and is the controlling estimate. Both artifacts are preserved.

The estimate excludes source/payload/parity verification, corrected-operator
extraction, Gate 2 sensitivity/calibration, allocation idle/tail time, failures,
Gaussian Gate 1 and all training. Thus the complete design cannot responsibly
be launched under the standing 32-hour forecast gate. Even the remaining
seed-29 work in ph012/013 alone is estimated at 37.26--42.22 GPU-hours.

Expected additional saved density/tide storage is approximately 344.2 GiB from
the same window-scaled metadata. This is not a disk-space reservation or a
complete storage bound: allow additional compact summaries, staging, logs and
failed outputs, and check Scratch capacity before launch. Original inputs are
never deleted or overwritten.

## Approved budget amendment

Approved ceiling: **250 total allocated GPU-hours**, including rather than
adding to the original 32, with the original 8-hour Gate 1 subcap and deadline
unchanged (23 October, 20:17 BST). At most two interactive allocations; no batch
submission. This is a spending ceiling, not a promise that every later gate and
training arm will fit. Measure and account at each stage; stop and report if a
complete required stage cannot fit the remaining budget.

The user's explicit numerical approval supersedes the earlier 32-hour total.
It does not alter the frozen model/operator, statistical gates, phase scope,
training-only normalizations or sealed confirmation restrictions.

## Execution sequence after the budget is resolved

1. Commit the authorization, immutable source/config binding, exact phase/seed/
   draw inventory, reusable receipt map and hard accounting/admission limits.
   Preserve ph018/019 and all unrelated outputs. No scientific score selection
   or new model fitting during baseline preparation.
2. Verify existing source/checkpoint/normalizer bindings and payload hashes.
   Extend the current seed-29 development-only sampler guard through a narrow,
   separately named six-phase comparison contract, not a global weakening of
   the old mixture or sealed-phase guards. Test negative permission cases.
3. Test the registered sampler on representative shapes and establish unchanged
   seed-17 replay where code interfaces change. Preserve failed/partial outputs;
   any regenerated task uses an exclusive new destination with explicit lineage.
4. Complete missing full-cap fields with the unchanged NFE128 sampler, then
   exterior completion and corrected core extraction. Reuse qualified products.
   Use independent single-GPU workers only at demonstrated memory/resource
   scale. Historical peaks are about 27 GiB for NGC, but host-memory and full
   extraction peaks must also fit the actual allocation. Use measured admission
   limits, allowing completion/verification before the allocation deadline.
   Foreground bounded workers should exit naturally when finished so allocations
   do not remain idle while waiting for manual release approval.
5. Verify the resulting 192-case continuous-summary bank and report every
   phase/seed R. Freeze the complete amended Gate 2 rule before simulations:
   pooled six-phase statistic, ph014--017 out-of-fit report, phase resampling,
   frozen seed noise floor, minimum 10% accepted gain, power at both 20% and 40%
   true effects, at least 80% power and at most 5% false advancement. Preserve
   b06670e's laws/correlations and valid rejection-only screen logic. A relaxed
   necessary-event screen cannot pass the full gate.
6. Report Gate 2 before advancing. Only if it passes may the budgeted Gaussian
   gate proceed; only its survivors may receive the gated fine-model training.
   Training permission is available but does not override failed prerequisites.

## Evidence

### Execution checkpoint

Authorization committed as 35beaf7; generation contract/runner as a61cdcf.
CPU 59606194 passed 12 tests, verified all 12 prepared caches and all 56 reusable
density/tide pairs, then released automatically. The immutable generation
manifest SHA256 is
`62dd809fc73bc056ab9ffd0168d8b71a8daefa307e9f25acb862341abfbc463f`.
The unchanged sampler inherits four verified full-NFE seed17 replay receipts.
Segment s001 (59606367, nid200277) starts one GPU for at most three hours;
its first task is ph014/NGC/seed29/d000. Completion remains pending.

Generation root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_v1`.
`MANIFEST.json` binds source/configuration/input identities;
`REUSE_VERIFIED.json` records the 56 qualified reuses;
`allocations/s001/` holds reservation/start/final-accounting records;
`logs/s001/` and `tasks/` retain task execution and completion evidence.

The independent extractor is `workflows/sbi/e2e_wholefield_fullcap_extract.py`.
It uses only the generation snapshot's numerical dependencies, binds its own
source, restricts target access to ph012--017 and checks original-tide replay,
correction trace and corrected ph012/013 seed17 parity. It saves compact cores
and the fixed six-coordinate summary, not an altered operator or gate. Five
extractor unit tests plus 12 contract tests pass.

The first GPU replay, 59606430, passed all four ph012/NGC seed-17 draws, with
exact corrected-core agreement, original-tide max difference 5.96e-8, and
correction trace residual 8.93e-15. The extraction itself took 39.99 seconds and
peaked at 35.47 GiB GPU memory. Actual allocation: 64 seconds, 0.01778 GPUh,
COMPLETED 0:0, released automatically. Segment s003 (59606480) now extracts
currently verified products within a one-hour/one-GPU reservation, retaining
the two-allocation limit. No Gate 2 decision, new fit or posterior claim follows
from this engineering replay.

Metadata preflight source:
`workflows/sbi/e2e_wholefield_fullcap_preflight.py`.
Controlling artifact:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_resource_preflight_v2.json`.
SHA256: `6840dcd9acdfd6151d4408e26ccd48c6aa9e29d0fdd3b8debb47dac91d2e22c9`.
The artifact retains per-task inventory, timing rates, prepared cap geometry and
all read JSON hashes. Independent `jq` totals reconcile 192 tasks, 56 reuse
candidates, 136 missing tasks and both timing sums. No new numerical payload
analysis, scientific result, model training or posterior qualification is claimed.
