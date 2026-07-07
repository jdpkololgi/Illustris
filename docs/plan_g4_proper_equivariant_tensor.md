# Plan — G4-PROPER: equivariant point-cloud model, tensor-valued

Durable plan for the real G4 test (roadmap v2 symmetry axis). Written 2026-07-03
(Claude Code + JDPK) after the G4-SMOKE naming correction. Running narrative:
`SCIENCE_LOG.md`.

## WAVE-1 RESULTS (2026-07-04) — read this first

| Run | Model | Graph | Features | λ1 | λ2 | λ3 | clu ρ |
|---|---|---|---|---|---|---|---|
| baseline | GraphNet+NPE | Delaunay | curated | 0.775 | 0.811 | 0.891 | — |
| G3 | GraphNet+NPE | union | curated | **0.804** | 0.846 | 0.895 | — |
| A | GraphNet+NPE | radius | curated | 0.752 | 0.799 | 0.876 | — |
| **D** | point-attention MPNN | radius | **positions** | **0.726** | 0.807 | 0.838 | 0.54 |
| E | attentional DGCNN | **dynamic** | positions | 0.507 | 0.662 | 0.681 | 0.36 |
| B | SEGNN steerable+attn | union | positions | 0.536 | 0.610 | 0.653 | 0.35 |
| C | SEGNN steerable+attn | radius | positions | 0.423 | 0.411 | 0.513 | 0.37 |

**Verdicts (preliminary — single seed; steerable capacity-confounded):**
1. **Point-cloud works:** D (positions-only) ≈ A (curated) → raw geometry recovers ~96%
   of the curated-feature signal (matched estimand). Point-cloud route validated.
2. **Dynamic graph HURTS: E−D = −0.22 λ1** — the §1(d) subsumption hypothesis confirmed.
   Useful long edges are geometry-anchored (Delaunay bridges), not feature-anchored.
3. **Steerable SEGNN worst (0.42–0.54)** but capacity-confounded (fast 179k config) → NOT
   a symmetry verdict yet; RPP-relaxed / matched-param needed (plan branch below).
**Production stays G3.** Nonlocality upgrade (GPT-5.5 review): T̂ij(k)=(ki kj/k²)W_R(k)δ(k)
— the 1/k² makes the tidal tensor a NONLOCAL inverse-Poisson operator, so the union graph
is a discrete quadrature of it (radius = fixed aperture, Delaunay = adaptive void bridges,
attention arbitrates). Paper framing: "correct discrete support for a nonlocal cosmological
operator", not a failed architecture search.

**WAVE 2 (in progress):** D seeds 43/44 (seed variance) + run F (attentional DGCNN + curated
features, §5A). Deferred diagnostics: environment-sliced eval, connectivity-residuals, union
edge-type attention attribution. NOT queued: heavy Equiformer/MACE, matched-capacity SEGNN,
Tier B — all gated on the wave-2 + diagnostic readout.

**CLASSICAL FLOOR (2026-07-07, T1 of `plan_field_level_multimodal.md`):** textbook
density reconstruction (best: DTFE) + the exact FFT tidal solve, same estimand/test
split, train-calibrated: λ1/λ2/λ3 R² = **0.552 / 0.641 / 0.663**. Every row in the
table above should be read against this floor, not just against 0.775: the baseline
GraphNet's margin over LINEAR reconstruction is +0.22/+0.17/+0.23 — the learned
headroom is real, and even the "failed" SEGNN runs (0.42–0.54) sit at/below what a
non-learned method achieves. Add the classical row to all future result tables.

## 0. Scope discipline (what G4-SMOKE did NOT test)

G4-SMOKE held the **prebuilt Delaunay graph** and the **curated 7 node features** fixed
and varied only invariant edge-message aggregation (λ1 R²: mean 0.603, attention 0.654
vs baseline 0.774). G4-PROPER must remove BOTH leaks:

- **input = positions + LOS only** (no curated Degree/Density/I_eig features anywhere);
- **model builds its own neighbourhoods** (radius graph ~10 Mpc/h, not the offline
  Delaunay);
- **architecture is equivariant** (steerable messages under SO(3)-about-observer);
- **output is the tidal TENSOR** (type-2), from which eigenvalues fall out ordering-free.

The question it answers: *is an equivariant point-cloud model competitive with the
curated-feature GraphNet, and does the tensor formulation help?*

## 1. The key de-risking insight — TWO TIERS

Learning the tensor does NOT automatically require T-web module changes. Split it:

### Tier A — architecture test, eigenvalue-supervised (NO tweb changes, do FIRST)
A steerable net outputs a symmetric rank-2 tensor per node; we differentiably
diagonalise it and **supervise on the existing eigenvalues (LAMBDA1/2/3)** — which are
rotation-INVARIANT, so no tensor targets and NO box↔observer frame rotation are needed.
The tensor is an architectural device that gives ordering-free eigenvalues by
construction; eigenvectors emerge unsupervised (consistent because the net is
equivariant). This isolates "is the equivariant/point-cloud architecture competitive?"
with ZERO new data. **This is the quick-test JDPK wants, done correctly.**

### Tier B — orientation science, tensor/eigenvector-supervised (NEEDS tweb work)
Only if Tier A validates the architecture AND we want real-space tidal ORIENTATIONS
(intrinsic alignments). Requires tensor targets in the observer frame → the tweb module
additions in §3, plus the frame rotation in §4.

Decision rule: run Tier A → if it clears the bar, then (and only then) scope Tier B.

**2026-07-07 amendment — Tier B is no longer the only route to eigenvectors.** The
graph→field→Poisson F-tier (`plan_field_level_multimodal.md` §3) obtains the full
tensor + eigenvectors from a fixed differentiable FFT physics layer applied to a
*predicted scalar density field* — no tensor targets, no e3nn irreps, and no §4
box→observer tensor rotation (a scalar field needs only the already-fit affine map).
Tier B survives as the *comparison* arm (direct tensor supervision vs physics-layer
derivation) if F1 fires; do not start §3–4 tweb work for its own sake before the F1
readout. Bonus already banked: the F-tier/T1 solver validation (voxelwise R² ≥ 0.992
vs stored cactus `eig_vals` on a 512³ subbox) IS the §3 "validation anchor" — the
standalone FFT tensor code exists in `workflows/abacus_tweb/classical_tidal_baseline.py`.

## 2. Baseline data reality (verified this session)

- T-web producer `abacus_cactus_tweb_fullgrid_mpi.py` calls external `cactus`
  (`cactus.src.tweb.mpi_run_tweb`) on the density field
  `AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_..._density_field.npy`; **saves only
  `cweb` + `eig_vals`** (eigenvectors/tensor discarded).
- Per-galaxy labels assigned by `annotate_cutsky_with_tweb_eigs.py`:
  (FILE_NUM, HALO_INDEX) → host-halo box-frame x_com → grid voxel → LAMBDA1/2/3. Targets
  are REAL-space (box frame); inputs are redshift-space (observer frame). Eigenvalues are
  frame-invariant, which is why the current pipeline needs no frame handling.
- Env: **e3nn NOT installed** in cosmic_env/rapids-gnn (torch_geometric 2.7.0 is). Needs
  an isolated env.

## 3. Tier-B tweb module changes (additive; do NOT touch cactus)

Build the tidal tensor ourselves via FFT (self-contained, and doubles as a cactus
cross-check) rather than modifying the external package:

- **NEW `tidal_tensor_fullgrid.py`** — from the same density field δ: φ_k = -δ_k/k²;
  T_ij(k) = (k_i k_j / k²) δ_k; Gaussian smooth at Rsmooth=7 Mpc/h; save the 6 unique
  real-space components on the grid (float32). **Validation anchor:** diagonalise → the
  eigenvalues MUST equal the stored cactus `eig_vals` (frame-invariant); trace MUST equal
  the smoothed δ. Free, decisive correctness test.
- **NEW `annotate_tensor_at_galaxies.py`** — mirror `annotate_cutsky_with_tweb_eigs`:
  sample the 6 tensor components at the same (FILE_NUM, HALO_INDEX) voxels → per-galaxy
  T_box columns. Reuses that script's indexing wholesale.
- **NO edits** to cactus, the eigenvalue producer, or the existing cache builder.

## 4. Tier-B frame rotation (the crux subtlety)

Target tensor is box-frame; equivariant net operates in the observer frame of its inputs
⇒ rotate per galaxy: T_obs = R·T_box·Rᵀ, R = box→observer basis rotation from the cutsky
construction (rigid remap ⇒ piecewise-constant per tile). **Phase-0 task: pin down R**
(from the cutsky/remap code or by fitting box↔sky coordinate pairs). Validate: eigenvalues
of T_obs still equal LAMBDA (rotation-invariant) — guards the rotation. This step is the
main Tier-B risk and the reason Tier A goes first.

## 5. Architecture (both tiers share the encoder)

- **Env:** e3nn 0.6.0 + opt_einsum_fx 0.1.4 installed into **cosmic_env** (2026-07-03,
  `--no-deps`, zero churn — torch 2.9.1 / numpy 2.3.5 unchanged; PyG 2.7.0 already
  present). Equivariance P0 gate PASSED early: `o3.Linear(4x0e+2x1o → 1x0e+1x2e)`
  round-trip max|Δ|=0.0, and 1x0e+1x2e = the 6 symmetric-tensor components. Isolated
  `equiv_env` remains the fallback only if a future version conflict appears.
- **Input:** node = optional scalar (luminosity only, or none); geometry = relative
  vectors rᵢⱼ (type-1) on a **radius graph (~10 Mpc/h)** built at load time; per-node LOS
  r̂ᵢ (type-1) supplied to break isotropy → exact SO(3)-about-observer equivariance.
- **Encoder:** steerable message passing (e3nn tensor-product convolutions), 4–6 layers,
  invariant + ℓ=1,2 features. Battaglia GN block with equivariance-constrained φ.
- **Attention is REQUIRED, not optional** (amended 2026-07-03): invariant (ℓ=0)
  query/key logits + steerable values preserve equivariance exactly (SE(3)-Transformer
  construction) and act as a learnable adaptive smoothing kernel matched to the fixed
  7 Mpc/h target (widen in voids, narrow in clusters). Mean/sum aggregation is a fixed
  low-pass kernel; the smoke already showed mean forfeits ≈+0.05 R². Mandatory
  regularisation parity so the comparison is not confounded: attention dropout (not just
  feature dropout), weight-decay/dropout parity with baseline (0.2 / 0.08), early stopping
  on **val NLL**, matched parameter count. The smoke's attention variant overfit hard
  (train 0.073 vs val 0.253) because it was *undisciplined*, not because attention is wrong.
  Full candidate taxonomy and the fixed execution order are in §5A (they drive the
  P1a/P1b bake-off).
- **Heads:**
  - Tier A: a single ℓ=2 (+ℓ=0 trace) output → symmetric 3×3 tensor → torch symeig →
    sorted eigenvalues; loss = eigenvalue MSE (matched to LAMBDA), + small trace-vs-δ
    regulariser.
  - **NPE integration:** to keep calibrated posteriors, use the encoder's INVARIANT
    latent to condition the existing FlowJAX flow (drop-in for the GraphNet embedding).
    Point-tensor head is the smoke; the flow head is the production path.

## 5A. Candidate architectures & execution order (per the equivariant-GNN review)

The gamut of equivariant/geometric GNNs organises by **which irreps ℓ the latent features
carry** (the GWL expressivity axis, Joshi et al. 2023). Only the steerable/tensor family
carries ℓ≥2 and can natively emit the 1x0e+1x2e tidal-tensor head, so the bake-off draws
its candidates from there; the other families are either skipped or used as a control.

**Family taxonomy (organising the gamut):**

| Family | ℓ carried | Members | Role here |
|---|---|---|---|
| (a) Scalarised / invariant-message | ℓ≤1 | EGNN, PaiNN, GVP-GNN | **SKIP** — cannot natively emit the ℓ=2 head; shadowed by the 0.603 smoke |
| (b) Steerable / tensor-field | ℓ≥2 | TFN, **SEGNN**, **SE(3)-Transformer**, **Equiformer**, MACE | **the candidate pool** (see order below) |
| (c) Frame-based / canonicalisation | n/a | Frame Averaging, local frames, LEFTNet | FA **SKIP** (PCA-frame degeneracy); LOS local-frame kept only as a cheap fallback if tensor products prove too costly |
| (d) Point-cloud, self-built graph | n/a | DGCNN, **Point Transformer**, PointNet++ | one model as the **P1a graph-construction control**, not an equivariant competitor |
| (e) Long-range / global attention | varies | GATr, Erwin, full-attention transformers | **SKIP** — long-range capacity wasted on a compact-support 7 Mpc/h target |

**Fixed execution order (matched-compute, matched-param; run in this sequence, each gated):**

| # | Model | Family | Gate | Rationale |
|---|---|---|---|---|
| 1a (control i) | **Existing attentional GraphNet + curated features**, ONLY the edge set swapped to radius (in-stack) | — | **P1a-i** — run FIRST | The report's Rec. #1: cheapest single-variable ablation of the Delaunay scale mismatch. Same model/features/seed/splits as baseline. |
| 1b (control ii) | **Positions-only attention control** (run D — attention MPNN on a load-time radius graph, POSITIONS+LOS ONLY). *Relabelled 2026-07-04: previously mis-badged "Point-Transformer-class" — a point-cloud network IS an attention GNN on a coordinate-derived graph, so D is architecturally the same object as run A; its scientific content is the FEATURE axis (D−A), not a new model family.* | (d) | **P1a-ii** | Fills the (positions-only, non-equivariant, fixed-graph) cell. For a fixed radius rule, "load-time" and "prebuilt" edges are IDENTICAL (verified: same 1,816,273 pairs). |
| 1c (dynamic graph) | **Attentional DGCNN** (run E): dynamic kNN recomputed PER LAYER in learned feature space (layer 0 = coordinate kNN), EdgeConv max-pool REPLACED by GAT/GAPNet-style multi-head attention, positions+LOS only. Rotation-invariant feature-space kNN (verified 1.4e-16). | (d) | after P1a-ii | The ONE candidate whose graph is genuinely not fixed. Promoted from SKIP (JDPK). **RESULT: E 0.507 < D 0.726** — dynamic graph hurt by 0.22; subsumption confirmed. |
| 1d (dynamic + curated) | **Run F**: attentional DGCNN (dynamic feature-space kNN) but with **curated Delaunay features** as inputs instead of positions-only. `gate_g4_p1e_dgcnn_attn.py --curated-features`. | (d) | wave 2 | F vs E = feature axis for the dynamic graph; F vs A = dynamic vs radius at matched curated features. Tests whether dynamic selection helps once the model already has the field estimators. |

**Dynamic-graph construction — control knobs (JDPK "how much control?", 2026-07-04):** we have
essentially full control over the dynamic local graph, exposed in `gate_g4_p1e_dgcnn_attn.py`:
(i) **k** (neighbour count — fixed count ⇒ density-adaptive physical scale); (ii) **which space
per layer** (layer 0 coordinate, layers>0 feature — configurable to all-coordinate = static, or
all-feature = pure DGCNN); (iii) **candidate-pool cap** `--knn-radius-cap` — restrict feature-kNN
to nodes within a physical radius = "learned selection WITHIN a physical envelope", the direct fix
for why E lost (non-local roaming); void nodes fall back to physical-kNN; (iv) distance metric;
(v) what defines the kNN space (raw inputs seed it — positions vs curated — or a dedicated learned
coordinate head); (vi) recompute frequency. The capped variant is the ready follow-up if F or E's
void slice shows a dynamic-selection signal worth keeping under a locality constraint.
| 2 (first equivariant) | **SEGNN-style steerable MPNN** + invariant-logit attention + 1x0e+1x2e tensor head | (b) | **P1b** | Minimal model that carries ℓ=2 natively, emits the tensor head, has a mature e3nn recipe, fewest confounds. PyTorch e3nn sidecar. Must beat 0.774 **and** the P1a controls. |
| 3 (second equivariant) | **SE(3)-Transformer / Equiformer-class** attentional steerable | (b) | after P1b GO | Non-linear equivariant attention + higher capacity to chase the 0.86 real-space ceiling. Only if #2 clears P1b (or within seed noise). SE(3)-Transformer is SEGNN's attentional cousin — fold its attention into the SEGNN build rather than running it separately. |

**Attribution algebra (why all controls are needed):** with G3 = GraphNet×union,
A = GraphNet×radius (P1a-i), D = positions-only attention×radius (P1a-ii),
E = attentional DGCNN dynamic graph (P1a-iii), C = SEGNN×radius, B = SEGNN×union:
**D−A** = raw geometry vs curated features at matched graph+attention; **C−D** =
equivariance alone at matched inputs+graph; **A−G3** = radius vs union at matched
everything; **B−C** = union vs radius within the equivariant family; **E−D** =
learned dynamic candidate selection vs fixed physical candidates at matched
inputs+aggregation. Without D, a P1b win could not be attributed to equivariance
rather than "any attention net on raw geometry"; without E, the §1(d) subsumption
claim would remain an untested assumption.

**Attention vs dynamic graphs — weighting ≠ candidate selection (JDPK discussion,
2026-07-04):** the over-smoothing analogy is half right. Mean/max aggregation is a
fixed low-pass kernel — that is the AGGREGATION axis, and attention fixes it
(hence attention required everywhere). The dynamic-graph concern is a different
axis: **attention reweights the candidates an edge set offers but cannot attend to
an absent edge.** With fixed k in feature space, feature-similar distant nodes can
evict physical neighbours from the candidate list entirely — a failure attention
cannot repair. Run E mitigates this with layer-0 coordinate kNN (physical
locality guaranteed once) and per-edge geometry scalars at every layer (the net
can learn geometrically coherent feature neighbourhoods if that is optimal).
Honest prior: for a compact-support (7 Mpc/h) target we expect D ≥ E overall —
but voids are where E could win (physical neighbours are few/far there, and
feature-similar void galaxies elsewhere can share signal: a form of environmental
parameter sharing). An E win concentrated in voids would be genuinely informative
and would revive the dynamic/multi-scale graph line.

**Supervision policy (wave 1 / Tier A):** ALL models train on the 3 sorted
eigenvalues. The steerable candidates predict the tensor INTERNALLY
(1x0e+1x2e → differentiable diagonalisation) — an architectural device, not a
target. Non-equivariant models supervise eigenvalues directly and must NOT emit a
fixed-frame 3×3 (frame-dependent for a non-equivariant net — logged rule). Tensor
TARGETS (+ eigenvectors) are Tier B, gated behind P1b.

**G3-driven amendments (2026-07-04, JDPK):** G3 (Delaunay∪radius in-stack) passed
its gate early (λ1 R² 0.8041@epoch-3749 vs 0.7750 Delaunay-full) → (i) G3 is
REUSED as the control×union cell; (ii) union-graph variants added to P1b
alongside radius-only (2×2 factorial); (iii) P1a-i doubles as the
radius-vs-union attribution ablation.

Note the two orderings compose: the **P1a control (#1) runs before the equivariant #2/#3**,
because the equivariant machinery must clear the *higher* bar of beating the radius-graph
control, not just the Delaunay baseline. Within the equivariant pool the order is
**SEGNN-with-attention first, Equiformer-class second, MACE skipped** for the smoke phase.

**Explicit SKIP list (with one-line justification):**
- **Plain EGNN / PaiNN / GVP** — ℓ≤1; cannot natively emit the ℓ=2 tensor head; already
  shadowed by the G4-SMOKE 0.603.
- **GATr / full-attention geometric transformers** — long-range capacity wasted on a
  compact-support (7 Mpc/h-smoothed) target that is a local functional of the field.
- **MACE** — many-body inductive bias calibrated to fixed-coordination atomic systems,
  mismatched to scale-free galaxy clustering; revisit only if body-order turns out to matter.
- **Frame Averaging (PCA frames)** — degeneracy/discontinuity under repeated eigenvalues;
  the per-galaxy LOS r̂ᵢ breaks symmetry more cleanly.
- **PointNet(++)** — PointNet has no neighbourhood structure (global pooling only, cannot
  see local density); PointNet++ groups by fixed radius, redundant with runs A/D.
- ~~**DGCNN**~~ — **PROMOTED to run E** (2026-07-04) as the *attentional* variant: the
  subsumption argument deserved a test, not an assumption, and EdgeConv max-pool is
  replaced by attention so E isolates candidate selection with aggregation matched.

## 6. Phased plan with validation gates

- **P0 (½ d):** stand up `equiv_env`; radius-graph builder; on the EXISTING wedge, sanity
  a tiny steerable net forward/backward; unit-test equivariance (rotate inputs+LOS →
  outputs rotate; tensor → R T Rᵀ) to machine precision. GATE: equivariance holds.
  **Escape hatch (amended):** if P1b fails *strict* equivariance, re-run once with a
  **Residual Pathway Prior** (RPP) relaxed pathway before concluding equivariance is
  unhelpful — distinguishes "symmetry is wrong here" (survey wedge breaks SO(3)) from
  "strict constraint too rigid."
- **P1a — TWO controls (RUN FIRST; cheapest; NO equivariance, NO tweb change,
  amended; see §5A attribution algebra): P1a-i** = non-equivariant ~10 Mpc/h
  **radius-only** attentional GraphNet in the existing JAX/jraph stack (same curated
  features / target / seed / splits as baseline; radius-only, not Delaunay∪radius, to
  isolate the construction axis). **P1a-ii** = positions-only attention control
  (`gate_g4_egnn_smoke.py --positions-only --build-radius-mpc`), filling the
  (positions-only, non-equivariant, fixed-graph) cell. **P1a-iii** = attentional
  DGCNN dynamic-graph run (`gate_g4_p1e_dgcnn_attn.py`), the unfixed-edges test
  (§5A row 1c). Tests the strongest first-principles
  hypothesis — that the Delaunay receptive field (fixed count, density-varying scale) is
  mismatched to the fixed 7 Mpc/h target. **GATE: within seed noise of 0.774 → graph
  construction matters; ≥ 0.80 → construction is the lever, DEPRIORITISE equivariance and
  invest in graph + attention.** (Relationship to G3: G3 tests Delaunay∪radius; P1a is the
  ablation attributing any gain to the radius edges specifically — sequence it after the
  G3 readout.)
- **P1b — Tier A equivariance smoke (1–2 d GPU):** eigenvalue-supervised steerable net,
  positions+LOS only, radius graph, SEGNN-with-attention. Pre-register discipline: ≥ 3–5
  seeds, matched compute/params, regularisation parity, frozen thresholds. Compare λ1 R² +
  cluster metrics on the SHARED test split. **GATE: λ1 R² ≥ ~0.75 AND beats the P1a control
  beyond seed noise → architecture competitive; proceed to the SE(3)-Transformer/Equiformer
  second test. < 0.70 or fails to beat P1a → run ONE RPP-relaxed variant (P0 escape hatch),
  then stop/log and keep GraphNet.** Because equivariant machinery must clear a *higher* bar
  than the Delaunay baseline (it must also beat the cheaper radius-graph control), a strict
  win is what justifies its cost.
- **P2 — Tier A + flow (few d):** swap point head for invariant-latent→FlowJAX; SBC/TARP;
  compare to the production NPE. **GATE (amended): calibration/coverage ≥ the current
  eigenvalue-regression flow.** Tensor supervision is better-posed, so calibration should
  improve; if it regresses, that is evidence the diagonalisation injects noise → revert to
  eigenvalue regression and keep the tensor head for IA (eigenvector) science only.
- **P3 — Tier B (only if orientation science wanted, ~1–2 wk):** build §3 tensor grid +
  §4 frame rotation (each with its eigenvalue-match validation); supervise full tensor;
  evaluate eigenvector accuracy vs box-frame truth. GATE: eigenvector error small enough
  for IA use.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Frame rotation R wrong (Tier B) | eigenvalue-invariance check catches it; Tier A avoids R entirely |
| Differentiable 3×3 symeig instability (degenerate λ) | analytic symmetric eigensolver / eigval-only loss / small jitter |
| e3nn data-hunger on ~58k train nodes | start small (ℓ≤2, few layers); the radius graph + weight sharing keep params low |
| Env drift breaking cosmic_env | fully isolated equiv_env; no shared installs |
| Scope creep back into G4-SMOKE territory | enforce §0: NO curated features, NO offline graph |

## 8. Effort, sequencing, decision

- **Framing (amended):** the strongest first-principles lever here is **graph
  construction** (radius graph fixing the Delaunay scale mismatch), which is *separable
  from* equivariance. Equivariance proper is only an **approximate** win — the survey
  wedge breaks SO(3) (footprint, fibre-assignment anisotropy, radial selection), so it
  is a useful regulariser, not a guaranteed gain; with ~58k train nodes and a 3-parameter
  group the predicted gain is modest, consistent with the empirically-bounded ≈ +0.09 R²
  headroom (G1.5 ladder: 0.774 → 0.86 ceiling). This is why P1a runs first and why P1b
  must beat the P1a control, not just the Delaunay baseline.
- Tier A (P0–P2): ~1 week, gated at P1a then P1b. Sequence BEHIND the G3 readout (GPU
  contention).
- Tier B (P3): ~1–2 weeks, only if Tier A passes AND IA orientations are a goal.
- **Implementation staging:** run the P1b smoke in a PyTorch e3nn sidecar (fastest path to
  the physics answer; e3nn 0.6.0 already in cosmic_env); port to e3nn-jax only after it
  passes, for the P2/production flow-head integration. A negative smoke in PyTorch saves
  the entire JAX porting cost.
- Production adoption only if Tier A + flow ≥ baseline on λ1 R², cluster recovery, AND
  calibration. Otherwise the GraphNet stays and this becomes a P2-paper negative/idea
  result ("equivariant point-cloud vs curated-feature graph, at matched effort").
- **Permanent-shelf condition:** if (i) P1a captures the headroom, (ii) strict *and*
  RPP-relaxed steerable models fail to beat it beyond seed noise under matched compute,
  and (iii) the tensor head does not improve calibration → equivariance is confirmed a
  modest regulariser dominated by graph construction; shelve with a documented
  negative-result ablation. A neutral result is publishable, so EV is positive as long
  as the gates hold.
- **Thresholds that change the plan:** P1a ≥ 0.80 → deprioritise equivariance, invest in
  graph construction + attention. Strict steerable underperforms but RPP-relaxed reaches
  parity → "symmetry is approximate," adopt the relaxed variant. Tensor-head calibration
  regresses → revert to eigenvalue regression, tensor head becomes IA-only.

## 9. What this does NOT change

Phase B production bundle proceeds independently on the GraphNet + nzharm cache + union
graph (G3) + FMPE (G6) + luminosity (G2). G4-PROPER is an exploratory branch that can
only ADD to Phase B if it clears P1/P2 — it never blocks it.

## 10. Companion plan (output axis / multimodal)

G4-PROPER owns the **input-representation** axis (graph construction, features,
equivariance). The **output/target-representation and multimodal** axis — classical
floor (T1), CNN-on-counts control (T2), privileged-information distillation (T3),
graph→field→Poisson (T4/F-tier), sim↔sim alignment control (T5) — lives in
`plan_field_level_multimodal.md` (2026-07-07). The two compose: the F-tier reuses
whatever encoder wins the G4 bake-off, and a P1b equivariance failure does not block
the F-tier (scalar field target, no irreps).
