# Plan — G4-PROPER: equivariant point-cloud model, tensor-valued

Durable plan for the real G4 test (roadmap v2 symmetry axis). NOT executed — this is
the design to review before any build. Written 2026-07-03 (Claude Code + JDPK), after
the G4-SMOKE naming correction. Running narrative: `SCIENCE_LOG.md`.

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
  invariant + ℓ=1,2 features; attention optional (invariant logits) — motivated given the
  smoke result. Battaglia GN block with equivariance-constrained φ.
- **Heads:**
  - Tier A: a single ℓ=2 (+ℓ=0 trace) output → symmetric 3×3 tensor → torch symeig →
    sorted eigenvalues; loss = eigenvalue MSE (matched to LAMBDA), + small trace-vs-δ
    regulariser.
  - **NPE integration:** to keep calibrated posteriors, use the encoder's INVARIANT
    latent to condition the existing FlowJAX flow (drop-in for the GraphNet embedding).
    Point-tensor head is the smoke; the flow head is the production path.

## 6. Phased plan with validation gates

- **P0 (½ d):** stand up `equiv_env`; radius-graph builder; on the EXISTING wedge, sanity
  a tiny steerable net forward/backward; unit-test equivariance (rotate inputs+LOS →
  outputs rotate; tensor → R T Rᵀ) to machine precision. GATE: equivariance holds.
- **P1 — Tier A smoke (1–2 d GPU):** eigenvalue-supervised steerable net, positions+LOS
  only, radius graph. Compare λ1 R² + cluster metrics vs baseline 0.774 on the SHARED
  test split. **GATE: ≥ ~0.75 → architecture competitive; proceed. < 0.70 → stop, log,
  keep GraphNet.**
- **P2 — Tier A + flow (few d):** swap point head for invariant-latent→FlowJAX; SBC/TARP;
  compare to the production NPE. GATE: calibration ≥ baseline.
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

- Tier A (P0–P2): ~1 week, gated at P1. Sequence BEHIND the G3 readout (GPU contention).
- Tier B (P3): ~1–2 weeks, only if Tier A passes AND IA orientations are a goal.
- Production adoption only if Tier A + flow ≥ baseline on λ1 R², cluster recovery, AND
  calibration. Otherwise the GraphNet stays and this becomes a P2-paper negative/idea
  result ("equivariant point-cloud vs curated-feature graph, at matched effort").

## 9. What this does NOT change

Phase B production bundle proceeds independently on the GraphNet + nzharm cache + union
graph (G3) + FMPE (G6) + luminosity (G2). G4-PROPER is an exploratory branch that can
only ADD to Phase B if it clears P1/P2 — it never blocks it.
