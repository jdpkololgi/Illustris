# Plan — Field-level & multimodal program: "stop throwing the field away"

Durable plan written 2026-07-07 (Claude Code + JDPK brainstorm). Companion to
`plan_g4_proper_equivariant_tensor.md` (input-representation axis) and
`roadmap_environmental_vac.md` (production gates). This document owns the
**output/target-representation and multimodal axis**. Running narrative:
`SCIENCE_LOG.md`.

## 0. The observation that started this

Simulations hand us the complete density / tidal / eigenvalue **fields** — full
3-D grids — and the current pipeline reduces all of that to 3 numbers per galaxy
before the model ever sees it. We might be throwing too much information away.
Three distinct proposals fell out of the discussion, and they are NOT the same
idea wearing different clothes:

1. **Privileged-information fusion** — a 3-D CNN sees the sim fields at train
   time, latent-regularizes the GraphNet, and is absent at inference.
2. **Semi-/self-supervised sim+real training** — unlabeled DESI and labeled
   mocks share an encoder / latent space.
3. **Field as OUTPUT** — the graph model *decodes* a density field, and the
   tidal tensor is obtained by an exact physics layer, not learned.

## 1. Information accounting (the discipline that ranks the ideas)

At DESI inference time the input is the sparse, biased, redshift-space galaxy
catalog. Full stop. **No training-time-only modality can raise the mutual
information between input and target** — it can only reshape the hypothesis
space (regularization, sample efficiency, calibration). Consequences:

- Idea 1 is a **regularizer**, not an information source. It has a name in the
  literature: *learning using privileged information* (Vapnik) / *generalized
  distillation* (Lopez-Paz et al. 2016). Gains concentrate in low-data regimes;
  our sims are not data-poor ⇒ expected gain modest. Cheap, so worth one
  disciplined ablation (T3), not a program.
- Extra subtlety specific to us: the density field *deterministically implies*
  the tidal target (Poisson), so a field-side teacher is trivially near-perfect
  — the distillation signal is "the answer, encoded". Any benefit is purely in
  how it shapes the graph latents on sparsely-sampled structure.
- Idea 3 dissolves the "image branch is dead at inference" problem by making
  the field a **product** instead of an input. This is where the science value
  is (§3).
- Idea 2 is the riskiest: latent alignment can *hide* sim→real shift instead of
  fixing it, silently biasing NPE posteriors. It already has a home as the
  GateM→JEPA escalation in the roadmap (Track 2); this plan only adds the
  mock-on-mock control it must pass first (T5).

**The floor every learned model must be measured against** is the classical
ceiling: textbook density reconstruction + exact tidal solve (T1, below). If a
Wiener-filtered linear reconstruction recovers most of the GraphNet's R², the
"headroom for clever architectures" narrative changes completely.

## 2. T1 — Classical-reconstruction baseline (RUN 2026-07-07)

`workflows/abacus_tweb/classical_tidal_baseline.py`. Same estimand as the
GraphNet: observer-frame redshift-space positions in, real-space eigenvalues at
galaxies out, scored with sklearn R² on the *identical* test split (masks from
the `path1 ... lineareig_si` cache; GraphNet reference 0.775/0.811/0.891).

Pipeline: galaxies → density estimate on a padded Cartesian grid (analytic
wedge mask, radial n̄(r) selection) → T̂_ij(k) = (k_i k_j/k²) W₇(k) δ̂(k)
(exactly the cactus convention; kernel exp(−½(kR)²), R = 7 Mpc/h = 10.39 Mpc)
→ trilinear sample at galaxies → `eigvalsh` (ascending = LAMBDA1/2/3) →
per-eigenvalue linear calibration fit on the train split → test-split scores.

Estimators: raw CIC counts; CIC + Gaussian pre-smoothing (3, 5 Mpc/h); DTFE
(per-point ρ = 4/V_star from the Delaunay star volume, linearly interpolated);
data-driven Wiener filter (noise plateau from the high-k P(k) tail).

Solver validation (`--mode validate-solver`): recompute eigenvalues on a 512³
subbox of the 10% particle grid and compare voxelwise against the stored cactus
`eig_vals` (rs7 fullgrid_v3 slabs) — this doubles as the §3 physics-layer
correctness anchor AND the Tier-B `tidal_tensor_fullgrid` validation the G4
plan calls for.

**RESULTS: see `RESULTS 2026-07-07` block at the end of this file.**

Decision rules (frozen before the run):
- best classical λ1 R²_cal ≥ ~0.70 → the GraphNet's edge over linear theory is
  thin; deprioritize architecture hunts, prioritize calibration + field-level
  *products* (§3) and sim-to-real robustness.
- ≤ ~0.55 → large nonlinear headroom exists; T2/T4 tell us whether graphs or
  fields capture it better.
- in between → the gap IS the budget for T2–T4; report it per eigenvalue and
  per environment.

## 3. The centerpiece: graph → field → Poisson (F-tier)

### 3.1 Why

- The tidal tensor is a **nonlocal linear functional of the density field**
  (1/k² inverse-Poisson kernel — same observation that motivated the union
  graph in G4-PROPER). Predicting δ̂ and applying the operator exactly is
  better-posed than predicting T (or its eigenvalues) pointwise:
  symmetry, trace = δ̂, and rotational consistency are guaranteed by
  construction instead of being learned or regularized in.
- Eigen**vectors** come for free → filament orientations, spin/shape
  alignments, intrinsic-alignment (IA) priors — the Tier-B science — WITHOUT
  e3nn irreps, without tensor targets, and without the box→observer tensor
  rotation crux (§4 of the G4 plan).
- The predicted δ̂ field is itself a release-able product (wedge density maps
  with uncertainty), and the natural bridge to field-level inference.

### 3.2 "Will the model know how to transform density to tidal field?" (JDPK)

**It never has to learn it — we hard-code the mathematics.** The physics layer
is a fixed, parameter-free, exactly differentiable function:

```
delta_hat (grid)  --rFFT-->  delta_k
T_ij(k) = (k_i k_j / k^2) * W_R(k) * delta_k        # Poisson + Hessian, exact
T_ij(x) --irFFT--> 6 component grids
T_ij at galaxy positions  (differentiable trilinear sample)
lambda_1<=2<=3 = eigvalsh(T)                        # differentiable
```

Every step is linear algebra + FFTs: autodiff flows gradients from an
eigenvalue loss back through `eigvalsh`, the interpolation, and the FFTs into
δ̂ — the network's ONLY learnable job is producing a good density field from
the galaxy graph. The classical baseline (T1) uses this exact layer with a
non-learned δ̂, which is why it is simultaneously (a) the floor, (b) the unit
test for the layer, and (c) proof the plumbing exists. JAX gotchas, both
already known: use a float64 or jittered `eigvalsh` for near-degenerate
eigenvalues (G4 plan §7), and beware cuSOLVER eigvalsh blowups on GPU
(memory: gpu/jax gotchas) — eigenvalue-space losses can also be computed from
invariants (tr T, tr T², det T) if the eigensolver misbehaves in training.

### 3.3 Architecture sketch

- **Encoder (reuse, frozen or finetuned):** the production attentional
  GraphNet on the G3 union graph — node latents h_i. No new input machinery;
  this axis is orthogonal to P1a/P1b input questions.
- **Graph→grid bridge:** differentiable scatter of node latents onto a coarse
  wedge grid (CIC weights at galaxy positions — the transpose of the sampling
  operator in the physics layer), channels = latent dims + a counts channel.
- **Field decoder:** small 3-D U-Net (or FNO — natural pairing with the
  spectral physics layer) refining scattered latents → δ̂ on the T1 grid
  (~3 Mpc cells, padded, apodized wedge mask as an input channel).
- **Physics layer:** §3.2, with W_R at R = 7 Mpc/h fixed to match targets.
- **Heads/losses:**
  - L_eig: existing eigenvalue loss at galaxies (softplus-increment
    parameterisation unchanged) — **zero new training data needed**.
  - L_field (optional, F2): MSE/log-cosh on δ̂ vs the true 10% particle δ
    sampled at the observer-frame grid points. KEY SIMPLIFICATION vs Tier B:
    δ is a SCALAR — supervising it needs only the box↔observer affine map
    (already fit to RMS 0.07 Mpc/h in the spotlight-figure work), NOT the
    tensor rotation R·T·Rᵀ that makes Tier B risky.
  - NPE: unchanged — invariant global/node latents condition FlowJAX exactly
    as today; or, later, generative δ̂ (F3) for field-space uncertainty.

### 3.4 Relation to the other plans (how it all fits)

| Existing item | Relation |
|---|---|
| G4-PROPER P1a/P1b (input axis) | Orthogonal. F-tier reuses whatever encoder wins wave 2. A P1b equivariance failure does NOT block the F-tier: δ is scalar, no irreps needed. |
| G4-PROPER Tier B (tensor targets + frame rotation) | Largely SUPERSEDED as the route to eigenvectors/IA: F-tier gets them from physics with no tensor targets and no rotation crux. Tier B survives only as a comparison (direct tensor supervision vs physics-layer) if F1 fires. |
| Roadmap G5 (sparse U-Net field-level baseline, LOW priority) | T2 concretizes G5's spirit cheaply (CNN-on-counts control); F-tier is the full version. If F1 fires, G5 as originally written is absorbed. |
| Roadmap GateM → JEPA escalation | Idea 2 lives THERE, not here. This plan adds only T5's mock-on-mock control as a prerequisite before any latent alignment touches DESI. |
| Classical baseline (T1) | New permanent floor for every model row in every table, and the physics-layer unit test. |
| P2 workshop paper | T1+T2 rows strengthen the representation-ablation story (classical / tabular / graph / field cells of the zoo). |

## 4. Test list derived from the 2026-07-07 discussion

Ordered; each gated; costs assume Perlmutter interactive.

| # | Test | What it isolates | Cost | Gate / decision |
|---|---|---|---|---|
| **T1** | Classical ceiling (§2) — CIC/Gauss/DTFE/Wiener + exact tidal solve | information already in linear reconstruction | ½ d CPU (DONE) | per §2 decision rules; floor for all tables |
| **T2** | CNN-on-counts control: 3-D U-Net on voxelized wedge counts (+mask channel) → eigenvalues at galaxies, same splits | does GRAPH structure add anything over a dumb voxel view? (G5 concretized) | 2–3 d, 1 GPU | CNN ≈ GraphNet ⇒ graph story needs rewriting; CNN ≪ ⇒ sparse-tracer geometry matters |
| **T3** | LUPI distillation: 3-D CNN teacher on sim δ/T patches (box-frame, sampled at halo x_com per memory gotcha), auxiliary cosine/L2 latent loss on GraphNet node embeddings, α-weighted; teacher absent at eval | privileged fields as regularizer | 2–3 d, 1 GPU (bolt-on to jraph stack) | ΔR²(λ1) > seed noise (≥3 seeds) ⇒ keep as cheap add-on; else close idea 1 |
| **T4 = F1** | Graph→field→Poisson, eigenvalue-supervised only (§3.3 minus L_field) | output representation axis | ~1 wk, 1 GPU | λ1 R² ≥ G3 (0.804) AND calibration ≥ current flow ⇒ GO F2; else field-as-output shelved with negative result |
| **F2** | + L_field on true δ via affine map (scalar, no tensor rotation) | does field supervision beat eigenvalue-only? | +2–3 d | field loss must improve λ R² or eigenvector stability, else drop |
| **F3** | Generative δ̂ (flow/diffusion decoder head) for field-space uncertainty | posterior fields | ~2 wk, gated on F2 | TARP-style coverage in eigenvalue space ≥ NPE baseline |
| **F4** | Eigenvector science eval: predicted vs truth tidal eigenvectors (needs truth tensor grid from `tidal_tensor_fullgrid.py` — the T1 validate-solver code is 80% of it) | IA-readiness | few d CPU + eval | median misalignment angle small enough for IA priors (quantify vs random) |
| **T5** | Sim↔sim domain-alignment control: TNG↔Abacus (or path1↔staged-mock) as a truth-known domain pair; measure whether DANN/contrastive alignment helps or silently distorts, using the existing closure-test machinery as validator | de-risks idea 2 before GateM/JEPA ever touches DESI | ~1 wk | alignment must not degrade truth-known posteriors; else JEPA branch inherits a documented hazard |

Sequencing: T1 (done) → T2 ∥ T3 (independent, cheap) → T4/F1 (the decision
point) → F2–F4 only on GO. T5 is independent and only urgent if GateM fails
after Phase B.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Wedge boundary artifacts in FFT layer (non-periodic survey) | padding + apodized mask (T1 machinery); interior-vs-all diagnostic already in T1 scoring |
| eigvalsh gradient instability at degenerate λ | invariant-space loss fallback (tr T, tr T², det T); jitter; float64 head |
| Graph→grid scatter too lossy at 3 Mpc cells | counts channel + multi-resolution scatter; cell-size ablation |
| F-tier looks good on eigenvalues but δ̂ is unphysical off-galaxy | F2 field loss; visual + P(k) checks of δ̂ (we have the truth field) |
| Scope creep (this plan re-absorbing G4/JEPA) | §3.4 table is the contract: input axis stays in G4-PROPER, sim-to-real stays in Track 2 |

## 6. RESULTS 2026-07-07 — T1 classical baseline

**Solver validation (physics layer):** 512³ subbox of the 10% particle grid at
(512,512,512), interior trim 64 cells, vs stored cactus rs7 `eig_vals`:
λ1/λ2/λ3 voxelwise R² = **0.9918 / 0.9946 / 0.9967** (rms 0.012 vs truth std
0.13–0.22). Residual = subbox-missing long-wavelength tides, not convention
error. **The §3.2 physics layer is validated end-to-end** — this same code is
the F-tier layer and the Tier-B `tidal_tensor_fullgrid` anchor.

**Classical ceiling (path1 wedge, test split, train-calibrated R²):**

| Estimator | λ1 R² (cal) | λ2 R² (cal) | λ3 R² (cal) | λ1 raw | λ1 ρ_s | λ1 R² interior |
|---|---|---|---|---|---|---|
| CIC | 0.546 | 0.630 | 0.654 | 0.251 | 0.774 | 0.596 |
| CIC + Gauss 3 Mpc/h | 0.527 | 0.610 | 0.636 | 0.280 | 0.768 | 0.579 |
| CIC + Gauss 5 Mpc/h | 0.489 | 0.577 | 0.608 | 0.312 | 0.740 | 0.538 |
| Wiener (data-driven) | 0.546 | 0.630 | 0.654 | 0.256 | 0.774 | 0.596 |
| **DTFE** | **0.552** | **0.641** | **0.663** | 0.349 | 0.766 | 0.599 |
| GraphNet+NPE (Delaunay, curated) | **0.775** | **0.811** | **0.891** | — | — | — |

Artifacts: `/pscratch/sd/d/dkololgi/abacus/classical_baseline/` (scores JSON,
per-estimator predicted eigenvalues, solver validation JSON).

**Reading (per the frozen §2 decision rules):**
- Best classical λ1 ≈ 0.55 ⇒ **large nonlinear/learned headroom is REAL**:
  the GraphNet's +0.22/+0.17/+0.23 margin is genuine learning (RSD
  de-distortion + bias correction + nonlinear structure), not something linear
  reconstruction already gives away. Architecture work (G4-PROPER, F-tier) is
  chasing real signal.
- The margin is LARGEST for λ3 (+0.23) — collapse-axis information is where
  learning pays most; consistent with the cluster-recovery focus.
- Edge effects cost the classical method ~0.05 (interior 0.60 vs 0.55) — even
  edge-free, classical stays far below the GraphNet.
- Extra smoothing monotonically hurts (target already at 7 Mpc/h); Wiener ≈
  CIC (shot-noise filtering is not the binding constraint — sparsity+bias is);
  DTFE best by a hair. Raw-R² ≪ calibrated-R² confirms galaxy bias must be
  calibrated out even classically (single linear map suffices for most of it).
- T2 (CNN-on-counts) inherits these numbers as its floor; T4/F1's gate stays
  "beat G3 = 0.804", now known to sit ~0.25 above the classical ceiling.
