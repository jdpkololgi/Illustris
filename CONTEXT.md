# CONTEXT.md — Illustris Live State Dashboard

**Last Modified:** 2026-11-27 (Paper 3 blind field readout)
**Rule:** Overwrite in place on `save-state` (keep ≤ 8 KB / ~150 lines). Durable science goes in `SCIENCE_LOG.md`; Slurm/tmux/receipt telemetry goes in `RUN_LOG.md`. Pre-2026-11-22 context archived in `docs/log_archive/CONTEXT_2026_pre_nov22.md`.

---

## 1. Active Deliverables & Scope

- **Paper 2 (Primary — Per-Galaxy Probabilistic Cosmic Web VAC for DESI DR2 BGS_BRIGHT):**
  - **Core contribution:** Encoder-independent methodology yielding calibrated joint posteriors over the three ordered Gaussian-$R_7$ ($7\ h^{-1}\mathrm{Mpc}$) tidal-tensor eigenvalues ($\lambda_1 \le \lambda_2 \le \lambda_3$, softplus increment parameterisation) and derived T-Web classes ($\lambda_{\mathrm{th}}=0.2$) over **$0.01 \le z < 0.55$**.
  - **Current stage:** Whole-phase FMPE development readout complete. Primary seed 42 passes physical interval and TARP checks but fails the rank-shape test. Production freeze, blind opening and Loa application held.
- **Paper 3 (Separate / Experimental — Coherent 3D Field-Level Posterior via CFM):**
  - Two-stage coarse + fine 3D Conditional Flow Matching over full survey caps ($0.01 \le z < 0.55$). Not a dependency for Paper 2.
  - **Current stage:** Frozen model scored on both blind phases (128 draws); better single-cell density, eigenvalue and class forecasts than the matched count baseline on each phase. Low-z density intervals remain cautious. This is not a calibration test on DESI.

---

## 2. Claim & Provenance Table (Matched Low-z Corrected Contracts)

*All Paper 2 encoder rows below share identical contract hashes: 20 train / 3 dev (`ph006, 012, 013`, 21,876,905 authoritative rows) / 2 sealed (`ph018, 019`), $0.01 \le z < 0.55$, own-galaxy `RA/DEC/Z_COSMO` $R_7$ targets (`training_ready_sha256: 63b3c61c...`). Never mix with historical 5-phase or $z \ge 0.15$ runs.*

| metric | split_type | phase_set | z_range | artifact_id |
| --- | --- | --- | --- | --- |
| CFM fair CRPS improvement vs matched count baseline: 19.3/23.6/20.7/27.1% (density, ordered tides) | preregistered supported blind cells, both caps | ph018/019 | [0.01,0.55) | `docs/evidence/lowz_cfm_blind_score_table.json` |
| CFM nominal 81.82% interval: observed coverage 83.95% (density), 81.86% / 82.33% / 81.48% (ordered tides) | supported registered development cells, cell-weighted, both caps | ph006/012/013 | [0.01,0.55) | `$SCRATCH/abacus/e2e_field_v3/lowz_calibration_61647276/CALIBRATION.json` |
| P12-A epoch 1, macro R² λ1: 0.4351 | whole-phase dev, seed 42 | ph006/012/013 | [0.01,0.55) | `$SCRATCH/abacus/p12a_lowz_full_seed42_61548250/full/unet/seed_42/epoch_history.jsonl` (epoch 1) |
| Window attention epoch 1, macro R² λ1/λ2/λ3: 0.3814 / 0.4758 / 0.4570 | whole-phase dev, seed 42 | ph006/012/013 | [0.01,0.55) | `docs/evidence/p12/lowz_encoder_epoch1_window_attention.json` |
| P12-A epoch 2, macro R² λ1/λ2/λ3: 0.5092 / 0.5677 / 0.5779 | whole-phase dev, seed 42 | ph006/012/013 | [0.01,0.55) | `docs/evidence/p12/lowz_encoder_epoch2_p12a.json` |
| P12-A epoch 4, macro R² λ1/λ2/λ3: 0.5825 / 0.6747 / 0.7215 | whole-phase dev, seed 42 | ph006/012/013 | [0.01,0.55) | `docs/evidence/p12/lowz_encoder_epoch4_p12a.json` |
| CFM draw-zero T-Web agreement: 63.79% | supported registered NGC development cells, single draw | ph006 | [0.01,0.55) | `$SCRATCH/abacus/e2e_field_v3/lowz_cap_draw_61610899/NGC/SCORES.json` |
| CFM draw-zero T-Web agreement: 66.39% | supported registered SGC development cells, single draw | ph006 | [0.01,0.55) | `$SCRATCH/abacus/e2e_field_v3/lowz_cap_draw_61582948/SGC/SCORES.json` |

Only epoch 1 has scores from two encoders. No matched full-development R² yet for large-kernel CNN, LitePT-S or corrected host-cone G-PATCH. Exclude the historical G-PATCH extension: its input catalogue differs.


---

## 3. Frozen Contracts, Split Policy & Key Caveats

- **Shared phase policy (`configs/shared_phase_policy_20261027.json`):** Train `ph000–005, 007–011, 014–017, 020–024` (20); Dev/calibration `ph006, 012, 013` (3); **Final test `ph018, 019` remains sealed for Paper 2; separate Paper 3 frozen-CFM confirmation authorized.** Not for tuning.
- **Low-z bindings (`configs/lowz_retraining_binding_20261118.json`):** Supervision `0.01 ≤ z < 0.55`, positive-z context to `0.60`, matched-CIC selection/response, v2 own-galaxy `RA/DEC/Z_COSMO` labels (fixes legacy `HALO_INDEX` bug).
- **Material scientific caveats (carry into Paper 2 & 3):**
  1. All low-z CutSky labels currently derive from the **$z = 0.2$ Abacus snapshot** (not an evolving lightcone).
  2. Per-galaxy FMPE/NPE posteriors embed the Abacus training prior and are marginal, not spatially joint—do not stack them as population PDFs without hierarchical reweighting.
  3. T-Web is the potential Hessian ($R_7 = 7\ h^{-1}\mathrm{Mpc}$, $\lambda_{\mathrm{th}} = 0.2$), not V-Web velocity shear.

---

## 4. Next Steps & Active Queue (Two Live Codex Tracks)

1. **Paper 2 recommendation (not launched):** Pin epoch-4 encoder + seed-42 FMPE and its ph006 width correction. Verify corrected golden-mock inference parity, then run a bounded internal Loa pilot spanning caps/redshift, flagged provisional. This checks deployment, not calibration. Hold the release VAC and blind opening. Do not extend all 20 folds on the epoch-1/4 contrast alone. See `docs/evidence/p12/epoch_parity_20261128/`.
2. **Paper 3:** Both real caps have 32 draws; viewer live. Blind power, localization matrices and mock/Loa interpretation boundaries: docs/paper3_blind_field_handoff.md. Next: write up, retaining high-z transfer caveat.
3. **Observational handoff (`~/GraphWeb_DESI`):** Keep Loa VAC defaults frozen until the new low-z encoder + FMPE passes dev calibration, sealed `ph018/019` check, and golden-mock parity.

---

## Pending Ideas

- **Candidate Theory Paper (with Ofer & Krishna):** Compare nonlinear volume-weighted Abacus ordered tidal-eigenvalue 1-point PDFs and auto/cross-power spectra against Gaussian predictions (Doroshkevich 1970; Lee & Shandarin 1998; Desjacques & Smith 2008). Existing pilot: `workflows/tidal_statistics/` & `docs/abacus_tidal_statistics_20261006.md` (`c000/ph000` $z=0.2$, $512^3$ integrated abs diff `0.428/0.343/0.355` vs `~0.001` Gaussian control). Keep cleanly separate from Paper 2 per-galaxy VAC; observational measurement requires coherent CFM fields or hierarchical reweighting, never stacked FMPE draws.
- **Historical baselines & archival context:** See `docs/log_archive/CONTEXT_2026_pre_nov22.md` and `docs/log_archive/SCIENCE_LOG_2026_pre_nov.md`.
