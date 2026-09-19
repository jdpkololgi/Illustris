# Coupled-field preparation: session ownership reconciliation

The user directed the Mac-connected execution of `Conditional field models`
(task `01a06ce0-480c-71c1-9e40-a32a06b743a4`) to lead all subsequent work.
This is one preparation programme, not two independently approved experiments.

## Imported phone-side findings

The shared transcript contains completed turns at 18:32:54, 18:37:36 and
18:38:51 UTC. They reported concurrent edits, documented the distance-convention
error, and blocked further edits/launches on execution ownership. The phone's
[coordinate audit](e2e_coupled_distance_convention_audit_20260918.md) is retained
unchanged as primary evidence, not overwritten by the earlier approximate
frame check. Its restart-publication concern is also part of the repair.

Live inspection found two app-server processes opening the same task transcript:
login21 PID166579 and login30 PID1631531. The older process held a deleted task
writer lock while the newer held a replacement. Overlapping recorded turns
support duplicate execution, not merely a stale UI. Reconnection through a
different login node is a plausible trigger, not a proven explanation for the
lock replacement. This does not establish which physical device opened every
connection or constitute a repaired upstream Codex synchronization bug.

After the explicit ownership instruction, the old process was revalidated by
executable and exact task-transcript descriptor, then terminated with SIGTERM.
Its only other open task was its guardian. Subsequent inspection confirms that
PID166579 is gone. The Mac-connected login30 backend and unrelated tasks were
not terminated. No lock files, transcripts, or research artifacts were deleted.
The pairing tmux session survived; Slurm allocations were not cancelled.

The goal tool reported no current goal, so the same approved preparation
objective was re-established here, explicitly carrying forward prior spending.
It is active, not complete. The original resource registry and frozen v1 source
snapshots remain authoritative for already launched work.

## Scientific reconciliation

This is NOT the earlier constant h-unit adapter error (64 Mpc/h versus
94.5906000591 Mpc) documented in SCIENCE_LOG. The inherited calculation already
converted Planck18 Mpc distances to Mpc/h with 0.6766. Its remaining discrepancy
is a redshift-dependent cosmology distance-curve mismatch. Do not repair it by
inserting another h, changing H0 alone, or fitting phase-specific shifts.

New `cartesian_v2` products use a pinned DESI/Abacus distance table directly in
Mpc/h. Raw/fine/wide spacings remain 3.383/6.766/27.064 Mpc/h, preserving the
registered physical geometry. All positions and Fourier/tidal cell sizes use
that frame. Observed redshift remains RSD; host auditing alone uses Z_COSMO.
Inherited selection density is transformed by its radial volume Jacobian,
including the Mpc^-3 to (Mpc/h)^-3 conversion, with no selection refit.

The v1 DATA_AUTHORITY and configurations must not change under frozen jobs.
Catalogue RA/DEC/Z joins, angular maps, native TSC counts and verified B particles
are reusable. Legacy Planck18 Cartesian products remain diagnostic-only and
are excluded from new readers by coordinate hashes and explicit unit keys.
All later experimental arms must use the same corrected products.

## Execution safeguards

Only this Mac-led task may launch or edit the programme. Phone access should
control this same connected backend, not independently resume a stale server.
The deterministic continuation controller now holds a persistent exclusive
filesystem lock; never unlink it while a controller is running. This prevents
duplicate preparation controllers but is not a general fix for Codex clients.
No model training or full-particle restore is authorized. Keep the approved
4.5 TiB Scratch, 64 CPU-node-hour, 4 technical GPU-hour and two-allocation caps.

## Verified correction and resumed execution

Twenty focused tests pass, including rejection of extra/missing unit metadata,
radial measure conservation, explicit new-frame matter interpolation, and
injected crashes after NPZ/HDF5 publication but before completion receipts.
Immutable generation filenames make these interruptions recoverable without
accepting partial files or overwriting source data. Unreferenced generations
remain included in the storage budget; this is not automatic garbage collection.

On CPU58540511/nid004202 the pinned correction passes4,077 resolved centrals
across three training phases, using the same table/axis/offset without fitting:

| Phase | Centrals | Old median Mpc/h | Corrected median Mpc/h | Corrected p95 Mpc/h |
| --- | ---: | ---: | ---: | ---: |
| 007 | 1357 | 0.510020 | 0.00003617 | 0.00012373 |
| 008 | 1361 | 0.535280 | 0.00003926 | 0.00012161 |
| 020 | 1359 | 0.527499 | 0.00003793 | 0.00010126 |

Maximum corrected error across these samples is0.000316705Mpc/h. Diagnostic
extra/missing h controls instead have median errors224--339Mpc/h. Even fitting
the best constant scale leaves0.162--0.168Mpc/h median errors; this diagnostic
fit is not used in any coordinate transformation. The radial mismatch is not
a single constant unit multiplier. These checks concern resolved host centrals,
not all galaxy types or an all-phase independent host validation.

Receipts: `cartesian_v2/coordinate_audit/ph007,ph008,ph020/CORRECTED_COMPLETE.json`
under the registered preparation root. New coordinate config SHA256:
`355db36fbcfacf7e37fd300d351ab7aad905e46fbb4c4b9e251a2264fa7a9d49`.
The original base config hash remains
`af5917cfb89a7a0be00b4f91c2aa3a8f7d73b8de2fc339e53d58d48f185d2f2b`.

Continuation58540511 is frozen as `source_snapshots/cpu_prepare_cartesian_v2`,
launched in login30 tmux `coupled-prep-mac02`, with four-hour limit and exactly
one request. Its preflight coordinate gate passed before the pairing, native
matter and corrected-observation workers started. Prior CPU58538574 returned
the registered pause code75 after3391seconds (0.941944CPU-nodeh); it completed
catalogue joins007--011,020--023 and saved nativeph007 at48/136 input files.
Its legacyph007 Cartesian maps remain comparison-only. Transfer58539034 is
unchanged. Completion of this coordinate repair is not completion of the full
data-preparation goal or authorization for scientific model training.

The first full corrected observation build (ph007, both caps) also passes:
NGC observed/expected shell ratios1.013307/1.005377/0.995736/0.975588;
SGC1.027473/1.010771/0.998837/0.952732. Counts are conserved and all fields are
finite. The committed receipts explicitly label Mpc/h geometry and
(Mpc/h)^-3 selection density. Building all remaining phases and validating the
joint products/normalization/physical references is still in progress.

## Additional stale-runner discovery at 19:19--19:26 UTC

An automatic goal continuation also arrived on an older login34 backend,
PID1027746 (started September17). This was not a user-authorized change of
leader. The login30 leader continued independently, performing step58540511.4
and publishing the corrected-product smoke receipt and the results above.
The login34 continuation performed read-only reconciliation and did not launch
another allocation or modify scientific products. Its open tasks were this
same transcript and its guardian only. It relinquishes execution to login30;
the shared goal must remain active, not be cleared or marked blocked, because
the designated leader and deterministic preparation jobs are making progress.
This reinforces that host-level duplicate execution, not just transcript UI
refresh, needs care. Avoid independently resuming this task through other
login-node backends; keep the Mac-selected leader/controller as the owner.

The Mac owner's subsequent user-requested recheck confirmed that PID1027746
logged this task from19:18:57 through19:27:43UTC (last process log19:27:44).
Read-only process/descriptor checks onlogin20--34 andlogin40 found no remaining
competing backend for this task; login30 PID1631531 remains the owner. Its original
writer-lock descriptor points to a deleted inode and the pathname disappeared,
so this is not evidence that the application-level cross-device lock bug is fixed.
No scheduler cancellation or scientific-product deletion occurred in this check.
The two existing allocations and goal remain active under the Mac owner.

## Mac reconnection and continued preparation, 21:19 UTC

The current Mac-connected backend is now login04 PID213499 (Codex0.154.0),
opening this same task transcript and a non-deleted task-writer lock. Read-only
inspection confirms that the oldlogin30 PID1631531 is gone; the remaining Cursor
backend there does not have this task open. No server was killed in this check.
This host change does not change the user's Mac ownership decision.

The original deterministic allocation controllers/tmux sessions remain onlogin30
and their Slurm jobs58540511/58544227 remain RUNNING. New condition recovery and
physical-gate tmux sessions are onlogin04 and reuse those allocations. A login
node's tmux list is local to that node: not seeing the older sessions onlogin04
does not imply that their compute stopped or authorize duplicate allocations.

The goal API returned no active goal after the reconnect. Following the user's
explicit request to continue as a goal, the full preparation objective was
re-established here, retaining the SAME Scratch root, registry, original caps,
spent resources and outstanding requirements. No new scientific-training or
full-particle authority was inferred. This is not evidence that all cross-device
application behavior is repaired; it records the currently verified owner.
