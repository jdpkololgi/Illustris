# RUN_LOG.md — Operational & Slurm Receipt Ledger

## 2026-11-26 — Flexible 48-hour OOF continuations

All 20 first-stage tasks running. Replaced pending chains with launcher
f0a23c9. Stage 2 requests 48 hours with 24-hour backfill minimum. Stage 3
requests 24 hours and export six hours. Verified 20 aftercorr tasks in each
stage. Arrays (second / third / export), grouped by omitted phases:

- 015–017,020–024: 62047832 / 62047833 / 62047835
- 000: 62047859 / 62047863 / 62047864
- 001–003: 62047872 / 62047879 / 62047882
- 004,005,007,008: 62047884 / 62047885 / 62047886
- 009,010: 62047887 / 62047888 / 62047889
- 011,014: 62047891 / 62047892 / 62047893

## 2026-11-26 — Corrected pending OOF chains and development recovery

Replacement arrays 62037192 through 62037237 (nonconsecutive) use baf0ace.
Verified 20 pending tasks each for stages 2, 3 and export. Development
recovery job 62037278 (1590c6d, six hours, one A100) verifies original
ph006 arrays and then exports ph012/013. Pending at verification.

## 2026-11-25 — Per-fold immediate summary publication

Preserved all first stages. Replaced pending stages 2/3/export for the six
groups below in their original order with arrays 61986117/118/120,
61986121/122/123, 61986124/125/126, 61986127/128/129,
61986169/170/171 and 61986173/174/175 (suffixes abbreviated).
Verified 20 pending tasks per stage and no duplicate downstream tasks.
Launcher f6b1d47 publishes each verified shard under the campaign summaries
directory when six hours remain after training, or in the next allocation.

## 2026-11-25 — Full OOF training and summary export queue

Committed launcher 1e6e3e5. Each fold has up to three 24-hour shared A100
segments (32 CPUs, desi_g, Scratch) and a six-hour terminal export. Later
segments return immediately on completion. Slurm accepted 20 tasks per stage.
Array IDs in stage order (training 1/2/3, export):

- ph015–017/020–024: 61980028/029/030/031 (last three digits abbreviated).
- ph000: 61980054/055/056/057.
- ph001–003: 61980058/059/060/061.
- ph004–005/007–008: 61980062/063/064/078.
- ph009–010: 61980079/080/081/082.
- ph011/014: 61980083/084/085/086.

Fresh eight have no predecessor. Other groups wait for their prior allocations.
Per-task aftercorr dependencies within each new chain. Both interactive pairs
running at verification. No new first-wave start estimate from Slurm.

## 2026-11-25 — Additional interactive OOF phases for December 2 Loa target

Requested two free interactive slots from login18 in tmux. 61979976
(ph009/ph010) and 61979977 (ph011/ph014), source 175905f, each two A100s,
64 CPUs, desi_g/shared_interactive, Scratch, four-hour ceiling, ten-minute
allocation wait, no retries. Initial state pending. tmux sessions and Scratch
allocation logs use oof-ph009-ph010 and oof-ph011-ph014 respectively.

## 2026-11-24 — Four OOF checkpoint continuations queued

Interactive segments 61932924 and 61932921 reached their runtime budgets.
Submitted array 61952194 (tasks 4/5/7/8 for omitted ph004/005/007/008).
Epoch 1 resume cursors are 106797, 106797, 106796 and 106796 respectively.
Checkpoint hashes pinned in launcher a18d9bc. Each uses its original trainer
checkout 02bbd35, one A100, 32 CPUs, desi_g/shared, Scratch, six hours,
no requeue or follow-on. All four tasks verified pending after submission.

## 2026-11-24 — Interactive OOF progress and architecture queue hold

User explicitly requested two interactive allocations with two phases each,
overriding the usual batch preference for these bounded production segments.
61932924 runs ph004/ph005; 61932921 runs ph007/ph008. Both allocated on
nid200337, each two A100s, 64 CPUs, desi_g/shared_interactive, Scratch,
four hours maximum, no retries. tmux owners on login35:
`oof-pair-004-005`, `oof-pair-007-008`. Source 02bbd35, launcher
`workflows/sbi/lowz_oof_interactive_pair.sh`; each worker gets one GPU,
32 CPUs and 56 GB RAM, a fresh seed-42 fit, four-epoch schedule and
12600-second checkpointed runtime limit. All four contract hashes passed.
Logs: `$SCRATCH/abacus/p12-oof-phNNN-JOBID.log`, allocation logs
`$SCRATCH/abacus/oof-pair-004-005.log` and `oof-pair-007-008.log`.
Per user request, held still-pending sweep jobs 61685645 (G-PATCH),
61860649 (LitePT), 61872278 (window transformer); verified JobHeldUser.
Running CNN 61664814 and all OOF/development/CFM jobs were left unchanged.

## 2026-11-24 — Four OOF fits and development export queued in parallel

User approved four concurrent OOF fits plus development export. Existing
61929552 continues omitted-ph000. New array 61930823_1/2/3 starts fresh
omitted-ph001/ph002/ph003 fits respectively, at most three array tasks
concurrent. Separate 61930825 exports ph006/ph012/ph013 sequentially from
the full encoder's four-epoch terminal checkpoint. Each new task: one
40/80 GB A100, 32 CPUs, desi_g/shared (gpu_shared), six hours, Scratch,
no requeue, retries or follow-on jobs. Training seed 42, fixed four-epoch
schedule; segments checkpoint before budget exhaustion. New maximum
authorized allocation is 24 GPU-hours across four tasks.
Launcher `workflows/sbi/lowz_oof_parallel.slurm`, revision 6bcb666. Shell
syntax and four input-contract hashes/phase counts verified. All submitted
jobs initially pending, start times unknown. Logs use
`$SCRATCH/abacus/p12-parallel-%A_%a.log`; immutable source snapshots and
outputs carry job IDs. No duplicate ph000 submission. Batch jobs survive
disconnects without tmux.

## 2026-11-24 — First omitted-ph000 production continuation submitted

User approved the six-hour segment. Submitted batch 61929552 from revision
5cf63a21246b7ca6f6bd0b3e1406d78fc1c27870 using
`workflows/sbi/lowz_oof_ph000_segment.slurm`. Verified one 40/80 GB A100,
32 CPUs, desi_g/gpu_shared (submitted shared), six hours, Scratch, seed 42,
Requeue=0 and no dependency. Checkpoint and smoke-shard hashes matched
the preceding receipt before submission. Resumes from update 256 toward
the fixed four-epoch budget; this allocation need not finish all four epochs.
Initial state PENDING/Priority, StartTime unknown. Log:
`$SCRATCH/abacus/p12-oof-ph000-61929552.log`. Batch persists across disconnects.

## 2026-11-24 — OOF export retry passed; production continuation prepared

Approved export-only job 61929126 completed on nid200273 in 33 seconds,
exit 0. tmux `oof-export-retry` on login35; one A100, 32 CPUs,
desi_g/shared_interactive, Scratch, 30-minute ceiling, no retry; allocation
released. Source fce24c4, log `$SCRATCH/abacus/oof-export-retry.log`.
Checkpoint before/after SHA256:
`3cdadd1006c5a69835bedeee4cf2ba712b6a5ad13111b407041eb3329612f3a8`.
Smoke shard SHA256:
`7660daf251010fe0b8d87dd59dffd82be9f214e45017d6b39d05b3cd35cb6033`.
Prepared `workflows/sbi/lowz_oof_ph000_segment.slurm`: six-hour shared batch,
one 40/80 GB A100, 32 CPUs, desi_g, seed 42, no requeue/follow-on. Syntax
passes and all nine trainer source hashes match the saved checkpoint run.
Not submitted: specific production allocation approval remains outstanding.

## 2026-11-24 — matched point-panel restart

61911376 exited 1 before training: archived source lacked Git metadata.
Verified detached graph 449c60d and LitePT fc0b407 checkouts on Scratch
before requesting replacement interactive window 61916473. tmux
point-panel-restart on login35; $SCRATCH/abacus/point_panel_retry_hgykYw/launch.log.
One A100, 32 CPUs, desi_g/shared_interactive, 4h, 10-minute allocation wait,
no automatic retry. First observed pending for priority; not a result.

## 2026-11-24 — bounded corrected point-encoder diagnostic

Interactive 61911376, nid200261, tmux point-panel-1124 on login18. One A100,
32 CPUs, desi_g/shared_interactive, Scratch, four hours maximum; no retry.
Graph then LitePT, fresh seed 42, 2048 updates each, registered 24 groups
of ph006/012/013; diagnostic outputs only in $SCRATCH/abacus/point_panel_diag_61911376.
Launcher log: $SCRATCH/abacus/point-panel-1124.log. Production jobs untouched.

## 2026-11-23 — transformer epoch-3 continuation

Replaced held 61786757 after verifying epoch 3 cursor 0, step/scheduler
692728 and completed epoch-2 history. New 61872278: one A100 (40/80 GB),
32 CPUs, desi_g/shared, Scratch, 24h, no requeue. Pending/Priority, no ETA
at verification. Launcher: workflows/abacus_tweb/window_epoch3_resume.slurm.
LitePT 61860649 untouched.

## 2026-11-23 — LitePT first segment shortened

Cancelled 61671100 while pending (zero elapsed). Replacement 61860649:
desi_g/shared, one 40/80 GB A100, 32 CPUs, Scratch, six hours, no requeue
or dependency. Corrected low-z inputs, fresh seed 42, same pinned trainer
and readout. Internal runtime limit 19800 s for a checkpointed pause.
Pending for priority with no start estimate at verification.

## 2026-11-23 — window-attention epoch-2 interactive comparison

Allocation 61847641 on nid200257: one A100, 32 CPUs, desi_g/shared_interactive,
Scratch, 4h maximum; no retries. tmux window-e2-61786757 on login21.
Batch 61786757 confirmed JobHeldUser before allocation. Python step 61847641.0
started; original corrected run, epoch 2 cursor 300912. Launcher stops after
validation and saving the epoch-3 cursor-zero checkpoint. Keep the batch held
pending checkpoint review. Log: $SCRATCH/abacus/window-e2-61786757.log.

**Purpose:** Append-only operational ledger (newest first) for Slurm submissions, `PENDING`/running status checks, tmux sessions, wrapper syntax checks, and SHA-256 receipts across `Illustris` and `GraphWeb_DESI`.
**Boundary:** Put completed scientific results, negative results, bug discoveries, and methodological decisions in `SCIENCE_LOG.md`, and keep current live state in `CONTEXT.md` (≤ 8 KB overwritten).

---

## 2026-11-24 — Paper 3 blind scoring and real Loa observation preparation

Blind full-box CPU score 61911489 on nid200007 (desi/interactive, 64 logical
CPUs, Scratch, 4 h max, no retry); runner f6f5aef. Outputs under
`$SCRATCH/abacus/e2e_field_v3/lowz_cfm_blind_confirmation_v1/blind_score_61911489`.
Require SCORES_COMPLETE.json and matched phase/cap denominators.
Loa observation CPU preparation 61911547 on nid200068 (same resources,
2 h max, separate authorization, no retry), runner b56af45. Outputs:
`$SCRATCH/abacus/e2e_field_v3/lowz_loa_preparation_61911547`. Require
OBSERVATIONS_PREPARED.json. Logs: `$SCRATCH/abacus/lowz-score-61911489.{log,err}`
and `lowz-loa-61911547.{log,err}`. No real-data GPU inference in either job.

Both steps and allocations completed 0:0, no retry. Blind score 16:48
(step 16:41, peak RSS 329415608 KiB), receipt SHA-256
`1539818124eb0b7843bb6c7f89f01c940c136329f567e5cdd144165b927d6281`.
Loa inputs 17:49 (step 17:44, peak RSS 5668300 KiB), receipt SHA-256
`79355192e0f4fbe96ca4f1486d9623b3b321f127a0ddd2dc65f12fddbfef39bc`.
Following separate approval and receipt verification, response 61911856
launched on nid200068: desi/interactive, one CPU node, 64 logical CPUs,
Scratch, 2 h max, foreground release, no retry, no GPU. Runner a4d40c3;
logs `$SCRATCH/abacus/lowz-response-61911856.{log,err}`. Require
`e2e_field_v3/lowz_loa_response_61911856/RESPONSE_PREPARED.json` under
`$SCRATCH/abacus` before interpreting transfer or planning GPU work.
61911856 and its step completed 0:0 in 8:08 and 8:07; peak step RSS
19530164 KiB, allocation released. Verified both cap hashes in response
receipt SHA-256 `2b82368046841046d1dddfb7e0bcb74401f63ca50adcf21b7f6e48a2db0ba451`.
No retry or GPU allocation.

## 2026-11-24 — Real Loa draw zero submitted on both caps

NGC 61912248 (3h30) and SGC 61912247 (2h) verified PENDING/Priority.
One 80 GB A100, 32 logical CPUs, desi_g/shared, Scratch per job, no retry;
5.5 GPU-hours combined ceiling. Source revision 132676d, launcher
`workflows/sbi/lowz_loa_draw0.slurm`, runner SHA-256
`01a4ed3372075d7486586e275377e6b20587c6c1f2ee4a4489f150adc66d2844`.
Both use the verified 61911856 response digest above. Logs:
`$SCRATCH/abacus/lowz-loa-draw0-{61912248,61912247}.log`.
Require density and tides COMPLETE receipts under each cap's
`predictions/draw_000` and successful accounting before reporting outputs
or estimating ensemble cost. No ensemble submitted.

## 2026-11-23 — ph018 confirmation draw 0 submitted

Subsequent baseline freeze: CPU allocation 61911278 on nid200013, desi/interactive,
64 logical CPUs, one-hour ceiling, released after 42 seconds (step 35 seconds,
both 0:0). No retry. Runner 75853a9, logs `$SCRATCH/abacus/lowz-reference-61911278.{log,err}`.

Continuation: remaining blind draw arrays 61871857 (NGC, 3:10:00)
and 61871864 (SGC, 1:46:00), 63 tasks each verified pending at launch.
Task 1..31 = ph018 draw 1..31; task 32..63 = ph019 draw 0..31.
One 80 GB A100, 32 logical CPUs, desi_g/shared, Scratch, no requeue.
Runner 25de41a. Logs: `$SCRATCH/abacus/lowz-blind-ensemble-%A-%a.log`.
No baseline, truth access or scoring in these jobs.

Verified outcome: both jobs and both steps per job COMPLETED 0:0; elapsed
2:50:56 NGC and 1:27:16 SGC. Density and tides each emitted a passing
receipt. Remaining 63 draws per cap project to 271.1 A100-hours at these
rates, excluding baseline and scoring. No remaining draw submitted here.

NGC 61857305 (3:10), SGC 61857310 (1:46), each one 80 GB A100,
32 logical CPUs, desi_g/shared, Scratch, no requeue. Both initially pending
for Priority. Commit aad3932. Density then tides only; no truth or scores.
Logs: `$SCRATCH/abacus/lowz-blind-draw0-{61857305,61857310}.log`.
Verify both density and tides COMPLETE receipts before any ensemble decision.
No remaining draws submitted.

## 2026-11-23 — Blind geometry registration completed

CPU 61857133, nid200043, 64 logical CPUs, desi/interactive, Scratch, 1h limit.
Foreground allocation exited 0:0 after 1:15; step 0 exited 0:0 after 1:13.
No retry. Adapter 45bbae6. Geometry SHA-256
`eb20683dbbf052f19ec651b8711c403eab3f75a967d235faf12011f316b2e737`.
Logs: `$SCRATCH/abacus/lowz-geom-61857133.{log,err}`.

## 2026-11-23 — CFM blind input preparation: one approved recovery

Outcome: allocation COMPLETED 0:0 in 15:11; step 61855386.0 COMPLETED
0:0 in 15:07. Foreground execution relinquished the node; empty stderr.
All four complete markers report this job, the registered phase/cap,
exact angular replay and the bound release, selection and source hashes.
All eight output paths and sizes match their compute-node SHA-256 records.

Job 61855386 on nid200029, one CPU node, 64 logical CPUs, desi/interactive,
Scratch, two-hour limit. Foreground `salloc ... srun ...` runs the committed
confirmation adapter with 16 workers, ph018 then ph019, stopping on error.
The command's exit releases the allocation. No cleanup watcher, additional
retry, GPU, matter truth, or DESI inference. Same release 5028056.
Logs: `$SCRATCH/abacus/lowz-confirm-retry-61855386.{log,err}`.
Output: `$SCRATCH/abacus/e2e_field_v3/lowz_cfm_blind_confirmation_v1/`
under `{ph018,ph019}/{NGC,SGC}/RESPONSE_COMPLETE.json`.
The previous 711234803-byte file was preserved by renaming its directory
to `ph018/NGC.partial-61854512`; it is not a run input or qualified output.

## 2026-11-23 — CFM blind input preparation interrupted (no retry)

Allocation 61854512, nid200038, desi/interactive CPU, 64 logical CPUs,
Scratch, limit 2 h. Release 5028056. The allocation exited COMPLETED after
3:07; the actual srun step 61854512.0 was CANCELLED (0:15) after 2:41.
The agent's cleanup watcher failed to recognize the live step and exited
the allocation prematurely. No scheduler timeout, model error or scientific
failure was observed. The ph018 NGC native writer was reached, downstream
of development parity and angular replay checks. There is no completed
response receipt for any cap. ph019 did not start. No truth, GPU or retry.
Preserved incomplete (711234803 bytes; do not use):
`$SCRATCH/abacus/e2e_field_v3/lowz_cfm_blind_confirmation_v1/ph018/NGC/NATIVE.h5`.
Stdout/stderr: `$SCRATCH/abacus/lowz-confirm-input-61854512.{log,err}`.
Recovery requires new approval: quarantine the incomplete directory without
deleting it; let a single foreground srun command control the allocation
lifetime, never a queue-display watcher.

## 2026-11-23 — CFM finishing, CPU 61848915 and GPU 61849771

CPU desi/interactive nid200023: 64 logical CPUs requested, Scratch, 1 h
maximum, completed and released after 14:18. Steps 0–4: native oracle fields
on three development phases, fitted count benchmark + scoring, scorecard and
historical regional replay, paired density PDFs/local spectra, 5/95 replay.
No sealed data. All six input receipts matched the bound core truth with zero
max-absolute error on supported scoring cells.

GPU desi_g/shared_interactive nid200261, 1×80 GB A100, 32 logical CPUs,
Scratch, 1 h maximum. Step 0 produced a passing ph006/NGC tidal field; its
reader incorrectly requested padding from COMPLETE rather than BINDING.
Step 1 verifies the binding hash and reads the recorded padding, reuses that
field without regeneration, then runs the remaining phases/caps serially.
All six scores and combined readout verified. Development method binding
passed: `lowz_mock_dev_method_binding_v1/METHOD_FROZEN.json` SHA-256
80436676ce3cc07e4364a1ee62d524452a0635e2e08fb12975d5167b8a437be1;
no confirmation input access or authorization. GPU allocation released,
sacct COMPLETED 0:0 after 28:42. Ancillary Graphify update failed on
filesystem flock (errno 524); repository index not refreshed. Scientific
outputs and checks are independent of the index.

## 2026-11-22 — CFM development replay and input audit, CPU 61811977

One CPU node, desi/interactive, 64 logical CPUs requested, Scratch; released
after both steps finished. sacct allocation COMPLETED 0:0, elapsed 00:07:11;
step .0 replay 00:04:56, step .1 input check 00:01:29, both 0:0. First
salloc invocation rejected an unsupported option before making an allocation.
Scratch outputs under `abacus/e2e_field_v3/lowz_finish_dev_61811977` and
`lowz_loa_input_check_61811977`. No GPU, CFM sampling, sealed phase or DESI inference.

## 2026-11-22 10:05 PST — development calibration readout verified

NGC/SGC production arrays completed 93/92 tasks respectively. Including
the earlier draws, the report lists 192 distinct paths. Job 61647276 exited
0:0 in 00:05:04; CALIBRATION.json and all six phase/cap PNGs were present
in `$SCRATCH/abacus/e2e_field_v3/lowz_calibration_61647276/`.

## 2026-11-22 07:16 PST — final four CFM draws running

188/192 distinct development draws complete (NGC 92/96, SGC 96/96).
After the approved in-place throttle removal, NGC array `61646846` tasks
92–95 (ph013 draws 28–31) started at 06:14:39 PST; all four were RUNNING
at the check. No duplicate submissions or retries. Scorer `61647276`
was PENDING/Dependency; no calibration JSON yet. On completion, verify
`$SCRATCH/abacus/e2e_field_v3/lowz_calibration_61647276/CALIBRATION.json`
and six phase/cap panels before interpreting ranks, coverage or reliability.

## 2026-11-22 07:16 PST — encoder sweep checkpoint

Slurm accounting: corrected jobs 61786757 (window continuation), 61664814
(large-kernel), 61671100 (LitePT-S), and 61685645 (host-cone G-PATCH) were
pending with zero elapsed time. Historical 61662008 was confirmed cancelled
after 04:51:39; no replacement on the historical contract. No job changed
during this state save.

## 2026-11-22 — Ledger initialized; live queue snapshot (no jobs modified)

Operational ledger split out from `SCIENCE_LOG.md` (pre-2026-11 entries archived verbatim in `docs/log_archive/SCIENCE_LOG_2026_pre_nov.md`; Nov 2026 entries retained in `SCIENCE_LOG.md` so active Codex chats keep uninterrupted context).

Active Perlmutter queue at initialization (`squeue --me`, 2026-11-22 05:51 PST):
- `61786757` (`p12a-win`, PD/Priority): Shifted-window attention Ep2 continuation (resumes `61664810` from cursor `300,912 / 346,364`).
- `61664814` (`p12a-lar`, PD/Priority): Corrected low-z large-kernel CNN first 24h segment.
- `61671100` (`litept-l`, PD/Priority): Corrected low-z LitePT-S first 24h segment.
- `61685645` (`gpatch-l`, PD/Priority): Corrected low-z host-cone G-PATCH first 24h segment.
- `61646846_[92-95]` (`cfm-cap-`, PD): Final 4 tasks of 32-draw NGC CFM array (SGC array `61646847` complete).
- `61647276` (`cfm-cali`, PD/Dependency): Dependent 192-receipt CFM calibration scorer (`CALIBRATION.json` + 6 map panels).
