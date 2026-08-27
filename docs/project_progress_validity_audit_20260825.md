# GraphWeb project progress and validity audit

**Audit date:** 2026-08-25 (UTC snapshot ≈ `2026-08-25T13:41:46Z`)  
**Authority:** `SCIENCE_LOG.md` supersedes `docs/plan_generalisable_graphweb_vac.md` on conflicts.  
**Companion canvas:** [graphweb-validity-audit](/global/homes/d/dkololgi/.cursor/projects/global-homes-d-dkololgi-TNG-Illustris/canvases/graphweb-validity-audit.canvas.tsx)

## 1. Executive verdict and scope

### Verdict (one paragraph)

GraphWeb is **code-ready and scientifically progressing**, but **not release-ready**. The P10 deterministic U-PATCH Bright-only reference on sealed-blind-protocol ph006 selection is an **established** independent-phase result (`macro R2(λ₁)=0.57310`). Live response-ladder and strict-control runs are **preliminary**. P12 posterior calibration is **incomplete** (partial OOF summaries only). P13 DESI canary / production VAC columns remain **gated**. Current NPE/flow samples, when calibrated, may represent **approximate per-galaxy amortized posteriors over ordered eigenvalue triplets under the mock training prior** — they are **not** jointly coherent, eigenvector-bearing, Poisson-consistent compatible tidal fields.

### Scope and confidence limits

| Layer | Status | Confidence |
| --- | --- | --- |
| Code readiness (P0–P8 adapters, P10 contracts, U-PATCH train/eval, P12 exporters/tests) | Strong | High — tracked tests + frozen contracts |
| Runtime completion (P10 Arm-A epoch 20; live R2/R3-RF/strict) | Mixed | High for Arm-A; interim for live jobs |
| Scientific validity of deterministic U-PATCH ph006 selection | Established under protocol | High for Bright-only Arm-A; **not** for FAINT causal claims |
| Posterior / VAC release readiness | Blocked | High that release is premature |

**Artifacts searched:** `SCIENCE_LOG.md`; `docs/plan_generalisable_graphweb_vac.md`; `docs/evidence/p10/*`; P12 OOF roots under `/pscratch/.../p12_oof_summaries/`; live response/strict training trees; GraphWeb_DESI `CONTEXT.md` / `ACTIVE_WORKFLOWS.md`; graphify global graph (`~/.graphify/global-graph.json`); source `shared/eigenvalue_transformations.py`, `workflows/sbi/jraph_sbi_flowjax.py`, `workflows/sbi/p12_export_unet_summaries.py`.

**Live Slurm snapshot (read-only):**

| JobID | Name | State | Node | Role |
| --- | --- | --- | --- | --- |
| `57582458` | `p10resp` | RUNNING | `nid008641` | R2 + R3-RF response ladder |
| `57583442` | `p10strict` | RUNNING | `nid008600` | R3-RF-DM + cross-phase Null |

---

## 2. Progress and experiment-validity ledger

Claim labels: **established** | **preliminary** | **open** | **stale** | **contradicted**.

### 2.1 Protocol gates P0–P13

| Gate | Plan status (plan file) | Audit status | Evidence |
| --- | --- | --- | --- |
| P0 / P0S evidence freeze | COMPLETE / near-complete | **established** | `docs/evidence/p0/`; plan §P0 |
| P1–P4 catalogue/graph/fields/manifest | COMPLETE (P3b-R active response gate) | **established** | plan §P1–P4; P10 multiphase P1–P4 closeout |
| P5–P7 adapters | COMPLETE | **established** | plan §P5–P7 |
| P8 matched deterministic (ph000) | COMPLETE / frozen development | **established** (same-phase only) | `docs/evidence/p8/`; SCIENCE_LOG P8 closeouts |
| P9 residual hybrid | Deferred | **open** / deferred | plan §P9 |
| P10 multi-phase + blind | Arm-A complete; selection on ph006; ph001 sealed | **established** for Bright U-PATCH Arm-A; **open** for FAINT/response causality and ph001 open | `docs/evidence/p10/arm_a_unet_training_complete_20260818.json`; `blind_evaluation_frozen_20260814.json` |
| P11 representation pretraining | Bounded, non-blocking | **open** / optional | plan §P11; SCIENCE_LOG |
| P12 posterior calibration | Start after ph006 selection | **incomplete / preliminary** | OOF only for `ph000`, `ph002`, `ph006`; SCIENCE_LOG 2026-08-20–22 |
| P13 DESI canary / VAC | Gated on P10 winner (+ P12 for posterior columns) | **gated** | plan §P13; SCIENCE_LOG |

### 2.2 Deterministic U-PATCH (P10 Arm-A) — established

- **Claim:** Four-shell macro `R2(λ₁)=0.573104` on ph006 at best epoch 20 (Bright-only Arm-A seed 42).  
- **Status:** **established** as the frozen independent-phase deterministic reference under the P10 protocol.  
- **Citation:** `docs/evidence/p10/arm_a_unet_training_complete_20260818.json` (`best_primary_macro_r2_lambda1`: `0.5731041810246436`); SCIENCE_LOG 2026-08-18 / 2026-08-20.  
- **Caveats:** ph006 is the **selection** phase — do **not** treat ph006 selection metrics as blind evidence. ph001 remains sealed (`blind_evaluation_frozen_20260814.json`). Old random-split TARP/SBC is **stale** for P10-protocol calibration (SCIENCE_LOG 2026-08-14 pathway assessment).

### 2.3 Response ladder (R2 / R3-RF) — preliminary

Matched **epoch-10** two-seed means (verified from `epoch_history.jsonl`):

| Arm | Seed 42 | Seed 43 | Two-seed mean |
| --- | ---: | ---: | ---: |
| R2 assignment | 0.542117 | 0.547435 | **0.54478** |
| R3-RF | 0.521878 | 0.547605 | **0.53474** |
| Real FAINT Proxy (SCIENCE_LOG) | — | — | **0.66394** (epoch 10) |
| Existing FAINT Null (SCIENCE_LOG) | — | — | **0.64185** (epoch 10) |

**Live maturity (2026-08-25):** validated histories complete through **epoch 11**; loss traces show **mid epoch 12** training (`cursor` ≈ 39k–83k of 84,446). Worker dirs also carry `ALLOCATION_PAUSED.json` while job `57582458` is RUNNING — treat epoch ≥11 metrics as **preliminary** until a clean epoch-complete validation row is frozen without pause ambiguity.

**Status:** **preliminary**. Response arms sit below real FAINT and old Null at matched epoch 10; SCIENCE_LOG (2026-08-25) correctly weakens “BRIGHT+FAINT wins via survey-response modelling” and notes R3-RF is a dense random-response field, not a sparse tracer match.

### 2.4 Strict controls (R3-RF-DM + cross-phase) — preliminary / open

- Products frozen for visible phases; canaries passed (SCIENCE_LOG 2026-08-25).  
- Job `57583442`: all four scientific workers mid **epoch 1** (~49k–50k / 84,446 updates); **`epoch_history.jsonl` empty** → **no validation yet**.  
- **Status:** infrastructure **established**; scientific contrast **open**.

### 2.5 FAINT / multitracer causal claims — preliminary / open

- Proxy vs Bright gains and Proxy−Null contrasts are logged but **not** causal until R3-RF-DM and cross-phase Null finish (SCIENCE_LOG 2026-08-25 decision).  
- Old FAINT Null retains projected clustering within Δz strata → **not** structure-free.  
- **Status:** **preliminary / open**. BGS_FAINT is **not** yet a production dependency.

### 2.6 P12 — incomplete

- Leave-one-phase-out encoders and OOF summary export exist for **ph000, ph002, ph006** only (`/pscratch/.../p12_oof_summaries/`).  
- Missing training-phase OOFs for **ph003–ph005**; flow calibration (SBC/TARP/conditional coverage) on the P10 protocol is **not** closed.  
- **Status:** **incomplete**. Old Abacus wedge TARP/SBC remains **stale** as production calibration.

### 2.7 P13 / GraphWeb_DESI — gated / demonstrator

| Path | Status | Notes |
| --- | --- | --- |
| Legacy GAT classifier VAC path | Active code; not the P10/P12 posterior product | `workflows/graph_inference/graph_catalog.py` |
| G3 / Jraph wedge inference | **Demonstrator**, not production VAC | `CONTEXT.md`: G3 NPE as current interface; wedge + Abacus-domain TARP/SBC; DESI closure separate |
| P13 canary / posterior VAC columns | **Gated** | Requires frozen P10 winner; posterior columns require P12 |

### 2.8 Release readiness summary

| Question | Answer |
| --- | --- |
| Can we ship a production DESI environmental VAC with calibrated eigenvalue posteriors? | **No** |
| Is the simulation truth-chain and U-PATCH protocol strong enough to continue? | **Yes** |
| Binding constraint | **P12 calibration + finish strict controls + freeze ph001 predictions before one-open** |

---

## 3. Simulation truth-chain audit

### 3.1 Canonical P10 chain (established)

```
AbacusSummit particles (c000, z=0.2, 2000 Mpc/h)
  → 10% A+B subsample
  → TSC density on ngrid=2048
  → Gaussian smooth Rsmooth=7 Mpc/h (≈10.4 comoving Mpc)
  → CACTUS/FFT T-Web Hessian → ordered eigenvalues (λ₁≤λ₂≤λ₃), λ_th=0.2 classes
  → Host-halo linkage → mock galaxy annotation (P1)
  → Cap graphs / metrics (P2) + CIC/response fields (P3 / P3b-R)
  → Fixed-comoving patch manifest (P4)
  → U-PATCH: 3-D U-Net on canonical patches → trilinear sample at BRIGHT coords
    → point head: scaled linear ordered increments → eval invert to (λ₁,λ₂,λ₃)
```

**Citations:** SCIENCE_LOG 2026-08-11 / 2026-08-13 (uniform 10% A+B TSC contract; ph006 T-Web); `docs/evidence/p10/multiphase_p1_p4_complete_with_ph006_20260813.json`; plan P10 particle/response contracts.

### 3.2 Legacy-versus-P10 parameter drift

| Item | Canonical P10 | Legacy risk |
| --- | --- | --- |
| Particle definition | Uniform **10% A+B** TSC | Older builds / reduced-grid canaries / phase-scattered roots |
| Grid / smooth | **ngrid=2048**, **R=7 Mpc/h** | Alternate grids or pre-P10 scratch layouts |
| Phase roles | train `ph000,ph002–ph005`; select `ph006`; seal `ph001` | Random-split or same-phase-only claims |
| Response | P3b-R / Loa DR2 randoms (all-18 frozen) | Mixing mutable DESI LSS versions |

**Status:** P10 multiphase products closed through ph006 with sealed-blind separation — **established** for the training contract. Any analysis using pre-P10 density/T-Web paths is **stale** for production claims.

### 3.3 Units, frames, and label/feature mismatch

- **Smoothing vs U-PATCH lattice:** SCIENCE_LOG 2026-08-25 — production U-PATCH is a **per-galaxy estimator**; the 5 cMpc input lattice need **not** equal R=7 Mpc/h. Required parity is coordinate frame, grid origin/cell, fractional voxel index, and target row.  
- **RSD features vs real-space labels:** mocks carry RSD-aware observed positions/features against real-space T-Web labels — known domain mismatch; ablation remains **open**.  
- **Float threshold / class columns:** P10 gates count native float32 threshold behaviour (SCIENCE_LOG 2026-08-13).  
- **Host linkage:** P1 annotation via host-halo; catalogue–field closure tests cover host consistency (`p3a_catalogue_field_closure.py` / tests).  
- **Phase isolation / leakage:** transforms fit on training phases only; OOF P12 exporters reject in-sample checkpoints (`validate_oof_checkpoint`); ph001 truth absent until predictions frozen.

### 3.4 U-PATCH product semantics

U-PATCH does **not** decode a voxelwise three-eigenvalue tidal field. It maps patches → latent voxels → point-sampled increments. A future voxelwise tensor decoder needs a separate truth-grid contract (SCIENCE_LOG 2026-08-25).

---

## 4. Posterior interpretation verdict

### 4.1 What is implemented

1. **Deterministic U-PATCH head:** scaled **linear** ordered increments → physical eigenvalues only at evaluation (`p8_deterministic_common` / Arm-A path).  
2. **Graph NPE / FlowJAX path:** encoder + normalizing flow over **ordered softplus increments** (`shared/eigenvalue_transformations.py`; `workflows/sbi/jraph_sbi_flowjax.py`). Samples invert via `increments_to_eigenvalues` / `samples_to_raw_eigenvalues`.  
3. **P12 path:** OOF U-Net latents + base eigenvalue predictions + response covariates as conditioning for calibrated flows (`p12_export_unet_summaries.py`; partial OOF on disk).  
4. **GraphWeb_DESI consumers:** G3 wedge Jraph inference applies Abacus-trained weights; GAT path is a separate 4-class classifier.

### 4.2 Explicit verdict

**When (and only when) calibration gates pass under the P10 protocol,** drawn samples may be interpreted as:

> Approximate **per-galaxy amortized posteriors** \(p(\lambda_1,\lambda_2,\lambda_3 \mid x_i, H_{\mathrm{fid}})\) over **local ordered eigenvalue triplets**, under the **mock training prior / observation model**, for galaxy \(i\) with features/context \(x_i\).

They are **not**:

- samples of a **jointly coherent** tidal field \(\mathbf{T}(\mathbf{x})\) across the survey;  
- **eigenvector-bearing** (directions of the tidal tensor are not in the head);  
- **Poisson-consistent / compatible** density→Hessian reconstructions (no enforced Fourier/Poisson consistency between sampled eigenvalues at neighbouring galaxies);  
- population-level PDFs obtained by stacking DESI galaxy posteriors without reweighting / hierarchical SBI (`GraphWeb_DESI/CONTEXT.md`).

### 4.3 Stronger “compatible tidal field” posterior (required before that language)

A defensible field-level posterior would need at least:

1. A generative model that maps a latent density (or potential) field → fixed FFT tidal operator → eigensystem (plan field-level path / SCIENCE_LOG direction).  
2. Joint structure across galaxies (spatial GP, field latent, or autoregressive field) — not independent amortized draws.  
3. **Posterior-predictive tests:** recover smoothed δ and T-Web maps; Poisson/FFT residual checks; eigenvector alignment metrics; cross-galaxy covariance coverage — not only per-λ marginal SBC/TARP.  
4. Separate BRIGHT/FAINT response contracts and phase-held-out calibration.

Until then, VAC language must stay at **per-galaxy eigenvalue posteriors (calibrated)** or **deterministic eigenvalue estimates**, not “compatible tidal fields.”

---

## 5. Methodology improvements and minimum experiment set

### 5.1 Release blockers (do these before any production VAC claim)

1. **Finish strict controls** to epoch-comparable validation (R3-RF-DM ≥2 catalogue seeds; forward cross-phase; promote reverse/2718 only if primary comparison warrants).  
2. **Complete P12 OOF** for all training phases; fit/calibrate flow; pass SBC/TARP **and** conditional coverage on the P10 protocol (retire legacy random-split TARP as current evidence).  
3. **Freeze ph006-selected predictions on ph001 inputs** before any truth open (`P10_BLIND_PREDICTIONS_FROZEN` procedure).  
4. **DESI response/schema canaries** + GraphWeb_DESI transfer tests under frozen Loa family (P13), after deterministic winner freeze.  
5. **Keep ordering parameterization consistent** in docs and products: softplus (NPE) vs linear increments (U-PATCH) must be explicit in VAC schema.

### 5.2 High-value, non-blocking research

- Multi-seed / multi-phase sensitivity beyond the two-seed response means.  
- RSD vs real-space label ablation.  
- Analytic / reference T-Web validation on controlled densities.  
- Sparse-shell / boundary stratification improvements.  
- Field-level encoder challengers (P11 / F-tier) without delaying P12.

### 5.3 Ranked methodology risks

| Rank | Risk | Severity |
| --- | ---: | --- |
| 1 | Claiming calibrated VAC posteriors before P12 closes | Release-blocking |
| 2 | Interpreting FAINT Proxy−Null as causal tracer information | Science-blocking for FAINT dependency |
| 3 | Treating ph006 selection or old TARP as blind/calibration evidence | Validity |
| 4 | “Compatible tidal field” language for independent amortized draws | Interpretation |
| 5 | Legacy density/T-Web parameter drift in analyses | Silent bias |
| 6 | Stacking DESI posteriors into population PDFs | Cosmology misuse |
| 7 | Calling GraphWeb_DESI G3 wedge a production VAC | Product misuse |

---

## 6. Quality checks (this audit)

| Check | Result |
| --- | --- |
| Epoch-10 means R2=0.54478, R3-RF=0.53474 | **Verified** from live `epoch_history.jsonl` |
| U-PATCH ph006 macro R2(λ₁)=0.57310 | **Verified** vs tracked JSON |
| Interim epochs labelled preliminary | **Yes** (response epoch 11–12; strict epoch 1) |
| ph006 not treated as blind | **Yes** |
| Old TARP/SBC not treated as current calibration | **Yes** |
| G3 wedge not treated as production VAC | **Yes** |
| Posterior verdict explicit | **Yes** (§4.2) |

### Citation gaps / follow-ups

1. Full five-phase P12 OOF completion markers for `ph003–ph005` (absent on disk at audit time).  
2. Frozen epoch-15 / terminal response-ladder validation JSON (not yet; histories stop at completed epoch 11).  
3. Strict-control first validation reports (none yet).  
4. Written VAC schema distinguishing U-PATCH linear increments vs NPE softplus increments (policy clear in code/docs; product schema still pending P12/P13).  
5. Allocation-pause vs RUNNING reconciliation for `57582458` worker markers (operational; does not change epoch-10 verification).

---

## Refs (primary)

- `SCIENCE_LOG.md` (2026-08-11 … 2026-08-25)  
- `docs/plan_generalisable_graphweb_vac.md` (§P0–P13)  
- `docs/evidence/p10/arm_a_unet_training_complete_20260818.json`  
- `docs/evidence/p10/blind_evaluation_frozen_20260814.json`  
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/response_training/p10_{r2_assignment,r3_rf}_v1/`  
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/strict_control_training/`  
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12_oof_summaries/`  
- `shared/eigenvalue_transformations.py`  
- `workflows/sbi/jraph_sbi_flowjax.py`  
- `workflows/sbi/p12_export_unet_summaries.py`  
- `GraphWeb_DESI/CONTEXT.md`  
