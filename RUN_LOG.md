# RUN_LOG.md — Operational & Slurm Receipt Ledger

**Purpose:** Append-only operational ledger (newest first) for Slurm submissions, `PENDING`/running status checks, tmux sessions, wrapper syntax checks, and SHA-256 receipts across `Illustris` and `GraphWeb_DESI`.
**Boundary:** Put completed scientific results, negative results, bug discoveries, and methodological decisions in `SCIENCE_LOG.md`, and keep current live state in `CONTEXT.md` (≤ 8 KB overwritten).

---

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
