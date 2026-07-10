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
| **T2** *(DONE §7)* | CNN-on-counts control: 3-D U-Net on voxelized wedge counts (+mask channel) → eigenvalues at galaxies, same splits | does GRAPH structure add anything over a dumb voxel view? (G5 concretized) | 2–3 d, 1 GPU | **RESULT: λ1 0.876±.004 > GraphNet 0.775; 4-class acc 0.882. Reads as "fixed-scale > Delaunay", not "CNN > graphs" — controls in §8** |
| **T3** *(RE-RUNNING §7)* | LUPI distillation: 3-D CNN teacher on sim δ/T patches (box-frame, sampled at halo x_com per memory gotcha), auxiliary cosine/L2 latent loss on GraphNet node embeddings, α-weighted; teacher absent at eval | privileged fields as regularizer | 2–3 d, 1 GPU (bolt-on to jraph stack) | ΔR²(λ1) > seed noise (≥3 seeds) ⇒ keep as cheap add-on; else close idea 1 |
| **T4 = F1** *(DONE-accuracy §7)* | Graph→field→Poisson, eigenvalue-supervised only (§3.3 minus L_field) | output representation axis | ~1 wk, 1 GPU | **RESULT: λ1 0.841 ≥ G3 0.804 → GATE PASSED on accuracy (graph encoder, no attention); calibration half pending the flow head** |
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

## 7. RESULTS 2026-07-08 — T2 (CNN-on-counts) + T4/F1 (graph→field→Poisson)

Same wedge, same test split, same MSE point-estimate estimand as the G4 gates
(so directly comparable to G4 runs, but +~few pts vs the NPE posterior-mean
baseline — the known estimand caveat). Scripts: `workflows/sbi/gate_t2_cnn_counts.py`,
`workflows/sbi/gate_t4_graph_field_poisson.py`. Artifacts: `/pscratch/sd/d/dkololgi/abacus/field_level_tests/`.

| Model | λ1 R² | λ2 R² | λ3 R² | notes |
|---|---|---|---|---|
| classical DTFE (floor) | 0.552 | 0.641 | 0.663 | T1 |
| GraphNet+NPE Delaunay (baseline) | 0.775 | 0.811 | 0.891 | NPE posterior mean |
| G3 GraphNet+NPE union | 0.804 | 0.846 | 0.895 | production anchor |
| **T4/F1 graph→field→Poisson** | **0.841** | **0.897** | **0.931** | MEAN agg, no attention; clears the F1 gate |
| **T2 CNN-on-counts (3 seeds)** | **0.876 ± 0.004** | **0.905** | **0.933** | seeds 0.876/0.871/0.880 |

**T2 — CNN-on-counts.** A plain 3-D U-Net on the voxelized galaxy COUNT field
(5 Mpc cells, 1.4M params) beats the GraphNet on all three eigenvalues, tight
across seeds. Spatially validated (viz agent): T-web **4-class accuracy 0.882**
(void/wall/filament/cluster F1 = 0.906/0.878/0.872/0.845), predicted class
fractions within ~0.3% of truth, confusion only between adjacent classes,
Spearman(pred λ1, truth λ1) = 0.965 — the win is real spatial structure, not a
metric artifact. **Framing (JDPK):** a CNN IS a GNN on a lattice graph, so this
is not "CNN beats graphs" but "fixed-scale regular sampling + density-valued
nodes beats the Delaunay receptive field" — the same lesson as G3
(union > Delaunay), taken further. **Caveats gating interpretation:** MSE head
(vs NPE), and the RAW wedge is ~2.4× DESI density (easiest regime for a grid).
→ controls in §8.

**T4/F1 — graph→field→Poisson.** GraphNet(EGNNlite, mean-agg) → differentiable
CIC scatter → 3-D U-Net → δ̂ → fixed FFT physics layer → analytic 3×3
eigensolver → eigenvalues. λ1 R² **0.841 clears the F1 gate** (≥ G3 0.804),
with a graph encoder and NO attention. This is the key result for the
"graphs vs CNN" worry: the graph encoder feeding a field/physics decoder
nearly matches the pure CNN (0.841 vs 0.876) and beats every prior graph
baseline — graphs are vindicated as the encoder; the win is the field-shaped,
fixed-scale OUTPUT representation. **Engineering note:** `torch.linalg.eigvalsh`
on CUDA tried to allocate 51 GiB for (N,3,3) matrices (the documented cuSOLVER
blowup) — replaced with an analytic Cardano 3×3 eigensolver (matches numpy to
7e-15, finite grads). The F1 gate's calibration half (≥ current flow) is NOT
yet tested — needs the invariant-latent→FlowJAX head (that's F1-calibration /
P2 in §6-plan), so F1 is "GO on accuracy, calibration pending".

**T3 — LUPI distillation. SHELVED 2026-07-08 (no valid result).** Four launch
attempts; the script's execution path never actually uses the GPU — even with
`device: cuda` confirmed and a hard `assert torch.cuda.is_available()` passing,
the process sits at 0 MiB GPU / ~85% CPU (the `torch.tensor(delta, device="cuda")`
at line 494 should show 207 MB on the card and does not), so it runs at CPU speed
and never finishes an allocation. A genuine, non-obvious bug in the agent-written
LUPI script, unresolvable by inspection. LUPI was the lowest-value test by design
(modest expected gain — "keep as cheap add-on OR close idea 1"), so shelved rather
than chase a rewrite reactively. Box-frame teacher mapping WAS validated (25×
random-median density, 0 bad links), so the idea is not disproven — just untested.
Revisit only as a focused device-path rewrite if LUPI becomes important.

## 8. Controls before "CNN beats graph" / "attention non-essential" are recorded

Motivated by the T2/T4 results (tracked; not yet run):
1. **Matched-estimand graph control** — union/radius GNN with an MSE point head
   (not NPE) vs the CNN's 0.876. Isolates estimand from representation. (T4/F1's
   0.841 with a graph encoder already suggests the gap is small.) **PARTIAL
   2026-07-08:** union+curated+MSE with the *lightweight EGNN-lite* smoke model,
   MEAN agg = λ1 **0.704**; the attention run was time-limit-killed. NOTE this is
   the weak smoke net, NOT the production GraphNet — an MSE-head *production*
   GraphNet is the clean comparison and is still TODO. So this control is not yet
   decisive; do not read 0.704 as "the graph's matched-estimand number".
2. **DESI-density re-run** — T2 + T4 on the n(z)-harmonized `nzharm` cache
   (`.../sbi_caches/path1_flowjax_3d_lineareig_si_nzharm/`), since the raw wedge
   is ~2.4× DESI density = easiest regime for a grid; the CNN edge should shrink.
3. **Cell-size sweep** — T2 at 3/4/5/6 Mpc scored on the cluster slice (λ1>0.2),
   on dense AND nzharm — finer resolves peaks but adds shot noise (JDPK: 5 Mpc
   may wash out cluster cores; target is 7 Mpc/h≈10.4 Mpc so grid ≤ target).
4. **Clean attention test** — attention-on vs -off within the SAME T4 F-tier
   encoder at matched estimand. G4 smoke had attention +0.05 over mean, but the
   representation lever is bigger; hypothesis: attention was partly patching the
   Delaunay scale-mismatch, so its value drops on a regular grid/field. Decides
   "attention is second-order for this operator" as a defensible claim.
5. **JEPA de-risking (broader):** T2's grid/CNN result makes a grid-native
   I-JEPA (mask blocks, predict latents; degrade-sampling-at-fixed-field views
   trivial on a grid) a low-risk instantiation of the GateM→JEPA branch and the
   substrate for the P5 foundation-encoder direction. Gated on GateM.

## 9. WHERE THIS LEAVES THE THREE PLANS (2026-07-08 synthesis)

Written to address the honest worry — "has a dumb CNN shown the graph PhD work
useless?" **No, and the results say why.** Read this section as the current
cross-plan state.

**The one-line result:** the field-level program is *validated* and the graph is
*vindicated as an encoder*; what changed is the **framing of where the
architectural lever sits** — it is representation *scale* + a physics-grounded
*output*, not attention and not equivariance. Nothing here touches the SBI /
calibration / VAC core or the published classification lineage.

**Why "CNN beats graph" is NOT the finding (be precise):**
- **T4/F1 is the direct rebuttal.** The *graph* encoder feeding a field+physics
  decoder scores **0.841** — above every graph baseline (0.775 Delaunay, 0.804
  G3) and near the pure CNN (0.876), with mean aggregation and NO attention. The
  winning architecture *is* a graph net; the graph was never the problem.
- **A CNN IS a GNN on a lattice.** So T2 = "fixed-scale regular sampling beats the
  Delaunay receptive field" — a statement about graph *construction* (your G4
  thesis: the Delaunay scale-mismatch), not about abandoning graphs. You had
  already half-found this: G3 union (0.804) > Delaunay (0.775).
- **The comparison is confounded and unfinished.** CNN/T4 use MSE point heads;
  the 0.775 baseline is an NPE posterior mean (MSE optimises R² directly). The
  matched *production*-GraphNet MSE head is still TODO (the 0.704 EGNN-lite mean
  is a weak smoke net). And it is all on the RAW wedge (~2.4× DESI density), the
  easiest regime for a grid. "CNN > graph" is not established at matched
  estimand / model / density.
- **The metric is not the thesis.** The VAC deliverable is *calibrated per-galaxy
  posteriors* (SBI/NPE, TARGETID VAC, closure, DESIVAST) plus the published
  RASTI classification. A point-estimate R² on one dense wedge does not touch any
  of that.
- **The regime that matters is untested.** Everything here is one dense wedge. At
  survey scale (voids, huge n(z) range) a fixed grid becomes mostly-empty and
  memory-prohibitive — exactly where sparse graphs/point-clouds are natural. The
  grid's edge here may not survive the regime the VAC actually runs in.

### 9.1 Multimodal / field-level plan (THIS doc): central hypothesis CONFIRMED
T4/F1 passed its accuracy gate — "field as OUTPUT" is the winning direction, and
it is built on the graph encoder. This plan is in the strongest position of the
three. Next: **F1-calibration** (invariant-latent → FlowJAX; the gate's second
half), then **F2** (field supervision) and **F4** (eigenvectors → IA). T2 (0.876)
is the field-representation reference. §8 controls fix the interpretation. T3
shelved; T5 unchanged (gated on GateM).

### 9.2 G4-PROPER plan: equivariance now DEPRIORITISED (its own gate fired)
G4-PROPER's decision rule was explicit: *P1a ≥ 0.80 ⇒ deprioritise equivariance,
invest in graph construction + attention*. That threshold is now cleared three
times — G3 0.804, **T4 0.841 (mean agg, no attention, no equivariance)**, CNN
0.876 — so the ≈+0.09 headroom over baseline is captured by
representation/output, NOT by steerable equivariance. Consequences: **shelve the
heavy SEGNN/Equiformer P1b line and Tier B**; the F-tier supersedes Tier B as the
eigenvector/IA route (physics gives the tensor, no irreps, no frame rotation).
Attention is demoted to *second-order* (pending the clean §8.4 on/off test). The
wave-1 point-cloud/graph-construction findings (D, E, G3) stand and feed the
"correct discrete support for a nonlocal operator" paper story — which the
CNN/T4 results strengthen, not refute.

### 9.3 Roadmap → VAC: production encoder reopened (favourably); SBI core intact
The Phase-B production question "what encoder?" is reopened in a good way:
**evaluate the F-tier (G7: graph encoder → field → physics) as the production
architecture**, since it beats G3 on accuracy; keep the flow head for calibration
(F1-calibration decides). The CNN result argues for a field/grid-aware
representation in production. **Unchanged and unthreatened:** the calibrated-NPE
machinery, TARP/SBC, the TARGETID-keyed VAC, closure tests, DESIVAST cross-match,
and the published classification paper. The sparse survey-scale regime remains
the frontier where a graph/point-cloud advantage most plausibly reasserts — an
argument FOR the graph encoder at production scale, not against it.

**Bottom line for the thesis:** 2.5 years produced the labelled T-web dataset, the
graph pipeline, the calibrated-NPE apparatus, the DESI application, AND the
understanding that representation scale is the lever — which is precisely what led
to the physics-grounded F-tier that now beats every prior model. A simpler
baseline you can *explain* (CNN = fixed-scale lattice GNN) and *out-design* (with
your own graph→field→physics net) is a stronger thesis chapter than "graphs win by
default". The framing matures from "graphs are the architecture" to "the graph is
an excellent encoder for a physics-grounded field inference, and here is why" —
that is a contribution, not a refutation.

## 10. Experiment plan — F-tier vs G3 GraphNet vs 3-D U-Net (2026-07-08)

Removes the two confounds in the earlier runs (MSE-vs-NPE estimand; raw
~2.4×-DESI-density wedge that favours grids) and adds calibration. Formalises §8.

**Posterior estimator decision:** the tests were MSE point-estimates, so they do
not directly rank estimators — but the strong latent (F-tier 0.841) + the hard
λ1-tail/cluster regime point to **running G6 now: MAF-NPE vs FMPE on the frozen
best-encoder latent, judged by SBC/TARP + cluster-tail coverage** (not NLL). Prior
= **FMPE**, but it is an empirical gate. Forward: the F-tier makes a **generative
δ̂ field posterior (F3, flow-matching/diffusion → physics)** the natural estimator
(BORG-adjacent, spatially coherent) — gated behind F1-calibration + F2.

| Phase | Experiment | Models | Head | Data | Metrics | Decides |
|---|---|---|---|---|---|---|
| **P1** | matched-estimand accuracy | G3-GraphNet · F-tier · CNN | all **MSE**, ≥3 seeds | raw wedge | λ1/2/3 R², λ1>0.2 slice, 4-class | is the CNN 0.876 edge real at matched estimand? |
| **P2** | DESI-density re-run | G3 · F-tier · CNN | MSE | **nzharm** | as P1, Δ vs P1 | does the grid edge shrink at survey sparsity? |
| **P3** | calibrated head + G6 | G3 · F-tier (CNN opt) | **MAF vs FMPE** on frozen latent | raw+nzharm | **SBC/TARP**, tail coverage, NLL, width | posterior estimator + best-calibrated model = **production choice** (= F-tier F1-calibration gate) |
| **P4** | CNN cell-size sweep | CNN | MSE | raw+nzharm | λ1>0.2 slice vs cell 3/4/5/6 Mpc | finer cores vs shot noise |
| **P5** | F-tier field-level | F-tier | **generative δ̂ (F3)** + eigvec (F4) | raw | field TARP, eigvec misalignment | field posteriors + IA — gated on P3 GO |

**Implementation:** G3-GraphNet MSE control = `jraph_pipeline.py --prediction_mode
regression` on the UNION cache (the *production* GraphNet, not the EGNN-lite smoke);
F-tier = `gate_t4_graph_field_poisson.py`; CNN = `gate_t2_cnn_counts.py` (already 3
seeds: 0.876/0.905/0.933); estimator bake-off = `gate_g6_fmpe_frozen_head.py`.
Caches: raw `..._sbi_cache_3d_lineareig_si.pkl`; nzharm `path1_flowjax_3d_lineareig_si_nzharm/`.
Order: **P1 first** (cheap, resolves the graph-vs-CNN worry) → P2 → P3 (production
decision) → P4 → P5 (on P3 GO). All GPU via tmux + salloc interactive + CUDA assert.

**P3 RESULT (2026-07-10) — FMPE > MAF on accuracy; calibration comparison INCOMPLETE.**
Frozen union GraphNet encoder, identical splits, MAF vs FMPE head (CPU-only). Posterior-
mean R² (n_eval=1500): MAF 0.819/0.881/0.916 vs **FMPE 0.850/0.896/0.928** (+0.031 λ1,
wins all; cluster λ1 Spearman +0.68→+0.70). **Accuracy gate MET.** FMPE calibration
(this run only): SBC KS-uniform p 0.000/0.003/0.001, λ1 coverage 59.4%@68% / 82.9%@90% →
under-covers. **CAVEAT: MAF SBC/coverage NOT computed here** — so FMPE is imperfectly
calibrated in absolute terms but we can't yet say worse-than-MAF; plus FMPE was quick-
trained (144 epochs, sbi defaults, 1 seed) on the raw over-dense wedge. **RESOLVED (symmetric rerun):** MAF calibration SBC KS-uniform p 0.009/0.006/0.017,
λ1 coverage 0.610@68% / 0.837@90% — vs FMPE 0.000/0.003/0.001, 0.594/0.829. **BOTH
under-cover near-identically**; the deficit is the frozen-encoder / over-dense wedge /
default flow training, NOT the estimator. ⇒ FMPE wins accuracy with calibration COMPARABLE
to MAF → **G6 gate GO: adopt FMPE** as the production posterior head. Calibration is a
SEPARATE fix (SBC-aware training / tempering + the P2 nzharm re-run). Scripts:
`gate_g6_fmpe_frozen_head.py`, `generate_maf_selfeval.py`; result
`field_level_tests/P3/g6_result.txt`.

**P1 STATUS (launched 2026-07-08, tmux `p1work`):** CNN done (0.876±.004). Running:
F-tier seeds 43/44 (+ the seed-42 0.841 already in hand) and the same-framework
graph control (EGNN-lite on union+curated, MSE, mean & attention, seeds 42–44) to
complete the matched-estimand torch-framework table. The *production*-GraphNet MSE
number (jraph regression on the union cache) is P1b — set up separately as the
strongest graph datapoint.
