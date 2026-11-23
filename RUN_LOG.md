# RUN_LOG.md — Operational & Slurm Receipt Ledger

**Purpose:** Append-only operational ledger (newest first) for Slurm submissions, `PENDING`/running status checks, tmux sessions, wrapper syntax checks, and SHA-256 receipts across `Illustris` and `GraphWeb_DESI`.
**Boundary:** Put completed scientific results, negative results, bug discoveries, and methodological decisions in `SCIENCE_LOG.md`, and keep current live state in `CONTEXT.md` (≤ 8 KB overwritten).

---

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
