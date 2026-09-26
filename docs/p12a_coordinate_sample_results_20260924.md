# P12-A sampled coordinate and native-label results

Job58823054 completed on nid004164 and was released after5m22s. The approved
30-minute CPU budget covered the first coverage failure, corrected ph002 replay,
and ph003-005 replication. Nine synthetic tests pass. All VAC compute requests
are now explicitly authorized by the user; scientific/phase gates remain.

| Phase | Observed rows | Native label rows | Cap/shell groups | Max point replay error (Mpc) | Max exact-distance error (Mpc) | Max native label difference | Partial checks |
|---|---:|---:|---:|---:|---:|---:|---|
| ph002 |16384|128|8/8|0|6.85254e-7|0|pass|
| ph003 |16384|128|8/8|0|6.85262e-7|0|pass|
| ph004 |16384|128|8/8|0|6.85257e-7|0|pass|
| ph005 |16384|128|8/8|0|6.85261e-7|0|pass|

Each phase also passed exact TARGETID/sky/host/label joins, Galactic cap identity
and native CWEB comparison. The observed-point test reproduces the stored
Planck18 interpolation; exact Astropy integration is an independent arithmetic
check of that interpolation. The native label comparison uses independently
loaded uncleaned halo x_com positions and original R7 T-Web cells, with exact
float32 label comparison. No coordinate mapping was fitted to labels.

Evidence: `docs/evidence/p12/P12A_COORDINATE_SAMPLE_PH00{2,3,4,5}_20260924_v2.json`.
Each report binds the final source hash, v3 input inventory, phase/row IDs,
sampled-data hashes, native source paths/metadata and every numerical check.
Large FITS/ASDF/NPZ contents were not fully rehashed; that distinction remains
explicit. These are deterministic samples, not population-level calibration.

The first ph002 report (`...PH002_20260924_v1.json`) is preserved. All its tested
numbers matched, but the three most populated slabs covered NGC only (62 native
rows). Its coverage check correctly failed. Before the next measurements, the
geometry-only sampling rule was amended to prioritize missing cap/shell groups
using at most six slabs. Tests cover cap imbalance and missing strata. Thresholds
were unchanged. Passing runs sampled16 native rows in each cap/shell group.

Successful per-phase runtimes were8.98,12.48,12.31,11.33s. `/usr/bin/time` reported
maximum process RSS of640960,648640,632032,634092KiB respectively. Slurm reported
7202868K for the three-phase extension step; use this larger accounting value
when sizing subsequent runs, not just Python process RSS. The allocation closed
COMPLETED0:0. Step.0 failed1:0 for missing coverage; step.1 shows FAILED2:0
because this audit deliberately returns2 when partial checks pass but the full
VAC gate remains unresolved. The extension driver checked reports/exit2 and
completed0:0. Scheduler success is not the scientific gate.

## Decision and next work

No sampled ph002-005 point/label error was found. These results do not trigger
retraining and do not qualify full P12-A deployment. Continue frozen-model
handoff while closing:

1. Imported ph000 lineage: its P1 marker only says legacy import. The manifest
   points to the older path1 fibre-assignment catalogue and pre-existing graph
   points, so the newer ph002-005 parent-row mapping cannot simply be assumed.
2. Full-fit/OOF encoder, normalization, selection and response-field lineage;
   actual annotation-run x_com provenance rather than only today's source default.
3. Central/satellite/unresolved-host coverage and pinned native-distance/volume
   comparison, with physical offsets distinguished from fiducial coordinates.
4. Loa source/refreeze and response consistency, then golden mock and bounded
   DESI trial. `coordinate_audit_status=unresolved` and
   `ready_for_desi_canary=false` remain correct for all reports.

No held-out/confirmation phase, real-DESI posterior inference, fitting or
retraining was performed by these runs.

## Subsequent handoff checks

The outstanding list above records the initial run state. Later same-day work
closed ph000 import/numerical checks and added encoder/annotation/Loa evidence,
plus ph002 central/satellite sample recovery. Current consolidated status:
`p12a_handoff_followup_results_20260924.md`. Full qualification remains open.
