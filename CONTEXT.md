# CONTEXT.md — Illustris Live State Dashboard

**Last Modified:** 2026-11-22
**Rule:** Overwrite in place on `save-state` (keep ≤ 8 KB / ~150 lines). Durable science goes in `SCIENCE_LOG.md`; Slurm/tmux/receipt telemetry goes in `RUN_LOG.md`. Pre-2026-11-22 context archived in `docs/log_archive/CONTEXT_2026_pre_nov22.md`.

---

## 1. Active Deliverables & Scope

- **Paper 2 (Primary — Per-Galaxy Probabilistic Cosmic Web VAC for DESI DR2 BGS_BRIGHT):**
  - **Core contribution:** Encoder-independent methodology yielding calibrated joint posteriors over the three ordered Gaussian-$R_7$ ($7\ h^{-1}\mathrm{Mpc}$) tidal-tensor eigenvalues ($\lambda_1 \le \lambda_2 \le \lambda_3$, softplus increment parameterisation) and derived T-Web classes ($\lambda_{\mathrm{th}}=0.2$) over **$0.01 \le z < 0.55$**.
  - **Current stage:** Matched multi-encoder architecture comparison on the corrected 20/3 low-z contracts (same 346,364 cores/epoch, same 21,876,905 dev rows, 4-epoch cosine schedule, seed 42). Freeze the winning encoder once matched evidence is in, then train and calibrate the corrected low-z FMPE posterior head and run DESI Loa inference via `~/GraphWeb_DESI`.
- **Paper 3 (Separate / Experimental — Coherent 3D Field-Level Posterior via CFM):**
  - Two-stage coarse + fine 3D Conditional Flow Matching over full survey caps ($0.01 \le z < 0.55$). Not a dependency for Paper 2.
  - **Current stage:** Seed-17 fine + coarse trained; 32-draw coherent development calibration ensemble finishing across all 6 dev caps (`ph006/012/013` × NGC/SGC).

---

## 2. Claim & Provenance Table (Matched Low-z Corrected Contracts)

*All Paper 2 encoder rows below share identical contract hashes: 20 train / 3 dev (`ph006, 012, 013`, 21,876,905 authoritative rows) / 2 sealed (`ph018, 019`), $0.01 \le z < 0.55$, own-galaxy `RA/DEC/Z_COSMO` $R_7$ targets (`training_ready_sha256: 63b3c61c...`). Never mix with historical 5-phase or $z \ge 0.15$ runs.*

| Model / Track | Metric (`macro R² λ1/λ2/λ3` or Field) | Split / Eval Set | Status / Cost | Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **P12-A U-PATCH (Ref)** | Ep1 λ1 `0.4351`; Ep2 `0.5092 / 0.5677 / 0.5779`; Ep3 λ1 `0.5666`; **Ep4 `0.5825 / 0.6747 / 0.7215`** | 20/3 dev (`ph006/012/013`), $0.01 \le z < 0.55$ | **4/4 epochs DONE** (~13.2 A100-h/ep) | `docs/evidence/p12/lowz_encoder_epoch4_p12a.json` |
| **Window-Attention Grid** | Ep1 `0.3814 / 0.4758 / 0.4570` | Matched 20/3 dev, $0.01 \le z < 0.55$ | Ep1 done; Ep2 paused at 300,912/346,364; cont. `61786757` PD | `docs/evidence/p12/lowz_encoder_epoch1_window_attention.json` |
| **Large-Kernel CNN** | Pending | Matched 20/3 dev, $0.01 \le z < 0.55$ | First 24h segment `61664814` PD | `RUN_LOG.md` |
| **LitePT-S (Point)** | Pending (preflight passed; stochastic restart accepted) | Matched 20/3 dev, $0.01 \le z < 0.55$ | First 24h segment `61671100` PD (~54.7 A100-h/ep proj.) | `abacus/lowz_litept_fork_61668057/` |
| **G-PATCH (Host-Cone)** | Pending (receiver-streamed + 1.40× exact host cone, max FP32 diff `2.98e-7`) | Matched 20/3 dev, $0.01 \le z < 0.55$ | First 24h segment `61685645` PD (~170.2 A100-h/ep proj.) | `docs/evidence/p12/lowz_gpatch_host_cone_61683377.json` |
| **Historical G-PATCH** | **EXCLUDED** (`61662008` cancelled; old 5-phase $z \ge 0.15$ contract) | Do not enter into low-z table | Cancelled | `SCIENCE_LOG.md` (2026-11-22) |
| **CFM Low-z Seed-17** | `ph006` draw-zero T-Web agree: 63.8% NGC / 66.4% SGC (75.7% / 73.2% at $z<0.15$); RMSE `0.3819 / 0.3652` | 20/3 dev (`ph006/012/013`), 6 caps, $0.01 \le z < 0.55$ | SGC 32-draw array DONE; NGC 4 tasks (`92–95`) left; scorer `61647276` PD | `docs/figures/lowz_ph006_draw_zero/`, `lowz_calibration_61647276` |

---

## 3. Frozen Contracts, Split Policy & Key Caveats

- **Shared phase policy (`configs/shared_phase_policy_20261027.json`):** Train `ph000–005, 007–011, 014–017, 020–024` (20); Dev/calibration `ph006, 012, 013` (3); **Final test `ph018, 019` (2 — SEALED, do not open)**.
- **Low-z bindings (`configs/lowz_retraining_binding_20261118.json`):** Supervision `0.01 ≤ z < 0.55`, positive-z context to `0.60`, matched-CIC selection/response, v2 own-galaxy `RA/DEC/Z_COSMO` labels (fixes legacy `HALO_INDEX` bug).
- **Material scientific caveats (carry into Paper 2 & 3):**
  1. All low-z CutSky labels currently derive from the **$z = 0.2$ Abacus snapshot** (not an evolving lightcone).
  2. Per-galaxy FMPE/NPE posteriors embed the Abacus training prior and are marginal, not spatially joint—do not stack them as population PDFs without hierarchical reweighting.
  3. T-Web is the potential Hessian ($R_7 = 7\ h^{-1}\mathrm{Mpc}$, $\lambda_{\mathrm{th}} = 0.2$), not V-Web velocity shear.

---

## 4. Next Steps & Active Queue (Two Live Codex Tracks)

1. **Encoder comparison track:** Let queued jobs (`61786757` window-attn Ep2, `61664814` large-kernel, `61671100` LitePT-S, `61685645` host-cone G-PATCH) run and update the matched Claim Table at equal epochs & GPU-hours; freeze the winning Paper 2 encoder and proceed to low-z FMPE calibration.
2. **CFM calibration track:** Wait for the final 4 NGC draws (`61646846_[92-95]`) and dependent calibration scorer (`61647276`) to produce `CALIBRATION.json` + 6 cap map panels, then audit marginal/joint calibration (fair CRPS, TARP, energy score, class reliability).
3. **Observational handoff (`~/GraphWeb_DESI`):** Keep Loa VAC defaults frozen until the new low-z encoder + FMPE passes dev calibration, sealed `ph018/019` check, and golden-mock parity.

---

## Pending Ideas (Park Here — Do Not Hijack Active Deliverables)

- **Candidate Theory Paper (with Ofer & Krishna):** Compare nonlinear volume-weighted Abacus ordered tidal-eigenvalue 1-point PDFs and auto/cross-power spectra against Gaussian predictions (Doroshkevich 1970; Lee & Shandarin 1998; Desjacques & Smith 2008). Existing pilot: `workflows/tidal_statistics/` & `docs/abacus_tidal_statistics_20261006.md` (`c000/ph000` $z=0.2$, $512^3$ integrated abs diff `0.428/0.343/0.355` vs `~0.001` Gaussian control). Keep cleanly separate from Paper 2 per-galaxy VAC; observational measurement requires coherent CFM fields or hierarchical reweighting, never stacked FMPE draws.
- **Historical baselines & archival context:** See `docs/log_archive/CONTEXT_2026_pre_nov22.md` and `docs/log_archive/SCIENCE_LOG_2026_pre_nov.md`.
