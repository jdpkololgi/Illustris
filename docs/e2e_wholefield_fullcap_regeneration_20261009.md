# Full-cap matched baseline: execution authority and resource preflight

## Current decision

On 9 October the user chose to retain the exact registered full-cap corrected
operator and explicitly authorized field regeneration, GPU use and training as
needed to obtain results. This resolves the operator/data-scope choice. It does
not waive the sensitivity/Gaussian gates, sealed-phase restriction, or the
explicit 32 allocated GPU-hour ceiling. A separate numerical budget amendment
has been requested because the saved full-length timings exceed that ceiling.
No new allocation, generation or training has been launched in this preflight.

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

## Requested budget amendment

Requested ceiling: **250 total allocated GPU-hours**, including rather than
adding to the original 32, with the original 8-hour Gate 1 subcap and deadline
unchanged (23 October, 20:17 BST). At most two interactive allocations; no batch
submission. This is a spending ceiling, not a promise that every later gate and
training arm will fit. Measure and account at each stage; stop and report if a
complete required stage cannot fit the remaining budget.

This amendment is pending explicit user response. General permission to use
GPUs is not recorded as approval to exceed the earlier numerical cap.

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

Metadata preflight source:
`workflows/sbi/e2e_wholefield_fullcap_preflight.py`.
Controlling artifact:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/fullcap_resource_preflight_v2.json`.
SHA256: `6840dcd9acdfd6151d4408e26ccd48c6aa9e29d0fdd3b8debb47dac91d2e22c9`.
The artifact retains per-task inventory, timing rates, prepared cap geometry and
all read JSON hashes. Independent `jq` totals reconcile 192 tasks, 56 reuse
candidates, 136 missing tasks and both timing sums. No new numerical payload
analysis, scientific result, model training or posterior qualification is claimed.
