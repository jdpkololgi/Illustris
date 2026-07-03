# Roadmap v2 — DESI BGS cosmic-web environment VAC + papers

Canonical decision-gated plan. Written 2026-07-03 (Claude Code brainstorm with JDPK),
superseding v1 (2026-07-02). Running narrative: `SCIENCE_LOG.md`. Every avenue below
is either **closed** (decision recorded), **gated** (explicit GO/NO-GO experiment), or
**scheduled**. Nothing expensive runs before its gate.

## 0. The problem, stated exactly

Learn the mapping: **sparse, biased, redshift-space galaxy configuration → real-space
matter tidal field**, with calibrated posteriors.

- **Targets:** Hessian eigenvalues (λ1,λ2,λ3) of the field from the FULL particle
  distribution, real-space comoving Cartesian, Gaussian-smoothed at 7 Mpc/h
  (retained — mass-anchored result: clusters need fine smoothing). Classes = λ_th
  thresholds; **headline product is the soft posterior P(λ1>λ_th)**, not argmax.
- **Inputs:** galaxy positions from observed z (RA,Dec,Z → Planck18 ≡ Abacus c000
  comoving distance) ⇒ redshift space. The regression fuses field interpolation,
  bias correction, and statistical RSD de-distortion.
- **Exact symmetry:** SO(3) about the observer (translations broken by radial RSD).
  Realized exactly by an E(3)-equivariant net fed per-node LOS r̂ᵢ as a type-1 input.
  Frame consistency holds (single Cartesian basis) ⇒ tensor/eigenvector prediction is
  well-posed; eigenvectors would be real-space tidal orientations (IA-ready).

## 1. Decisions already closed (do not reopen without new evidence)

| Topic | Decision | Evidence |
|---|---|---|
| Velocity-dispersion features | REJECTED | AUC≈0.5 on truth at Delaunay scale; +0.018 λ1 R² aperture-matched; doesn't fix threshold zone |
| Target smoothing scale | KEEP 7 Mpc/h | mass-anchored recovery monotonic toward fine; global-R² optimum is bulk artifact |
| Hard-class deficit | It's λ1-tail shrinkage | report P(λ1>λ_th); Spearman .56 global but 97% tail miss at λ_th=0.2 |
| Baryonic/HOD input features | REJECTED (claim purity) | property ceiling: geometry ⊥ properties; painted-mock systematics not worth it. Sanctioned middle path = luminosity weighting (G2) |
| Graph-partitioned SBI | LEGACY | wedge subvolumes are the path |
| Graph transformers (global attention) | MAPPED, NOT PLANNED | most data-hungry cell of the zoo; ~100k labeled nodes can't feed it |
| Nature/Nat Astro | Not a target | only if a headline science result emerges from the VAC |
| Encoder identity | Attentional GraphNetwork (Battaglia+2018), NOT GAT/GATv2 | GAT = separate PyTorch classification lineage (RASTI paper, JDPK author) |

## 2. Track 1 — Phase 0 gates (representation & information; CHEAP FIRST, PARALLEL)

Two-axis model-zoo framing: connectivity (Delaunay / radius / dynamic / global) ×
symmetry (hand-crafted invariants / raw-geometry / equivariant). Current cell =
(fixed Delaunay, hand-crafted invariants). Each gate moves one axis.

| Gate | Experiment | Cost | GO criterion | Decides |
|---|---|---|---|---|
| **G1** | GNN vs GBM, identical features/splits | ½ d CPU+1 GPU eval | GNN λ1 R² − GBM > ~0.03 | capacity vs information; rungs a–b of ablation |
| **G1.5** | **RSD-penalty decomposition**: rebuild features from Z_COSMO (real-space) vs Z, compare λ1 R² (also mass-anchored slice) | ½ d CPU | penalty ΔR² > ~0.05 | upper bound on ALL LOS-aware/equivariant machinery; splits ceiling into RSD vs sparsity+bias |
| **G2** | Luminosity-weighted density features, truth-gated (cutsky R_MAG_ABS) | ½ d CPU | ΔR²(λ1) > ~0.03 on truth | flux weighting in/out (claim-preserving) |
| **G3 (b′)** | Connectivity axis: same GraphNet on **Delaunay ∪ radius(≈10 Mpc/h)** graph | days, 1 GPU | cluster completeness or λ1 R² up materially | was the GRAPH the restriction? (fixed-physical-scale receptive field vs density-dependent Delaunay) |
| **G4 (c)** | Symmetry axis: equivariant encoder (EGNN/SEGNN, e3nn), rᵢⱼ + r̂_LOS in; tensor out or invariant-latent→flow | 1–2 wk GPU | ≫ b′ (ΔR² > ~0.05 or cluster metrics) | adopt equivariance; unlocks eigenvector/IA product. **Gated on G1 AND G1.5 both firing** |
| **G5 (d)** | Sparse U-Net field-level baseline | ~1 wk | informational | bounds graph-representation loss; LOW priority, only if time before P2 |
| **G6** | FMPE head swap (flow-matching vs MAF; same conditioning) | few days GPU | SBC/TARP + NLL improve | posterior-head modernization; independent of encoder gates |

Deferred (not gated yet): multi-scale *target* heads (fine-scale cluster head) — only
revisit after Phase B if cluster recovery still lags; dynamic/learned graphs — only if
G3+G4 both disappoint.

## 3. Track 2 — Phase A sim-to-real (PARALLEL with Track 1; architecture-independent)

1. **A1** Fix sentinel-z injection bug (2.07M phantom z≈0.59 rows in mock parent;
   wedge currently clean; blocks z-expansion). Do immediately.
2. **A2** Measure clean n(z), mock vs DESI, per z-shell (wedge count offset ~12% known).
3. **A3** Harmonize: density-match training mocks to DESI n(z) + inject DESI-like
   z-errors. (= the "degraded mock" idea in minimal, targeted form = supervised
   domain randomization / nuisance-marginalized SBI.)
4. **GateM** (after Phase B retrain): summary-space MMD within ~2σ of split-half floor
   → done. **If it fails → JEPA escalation**: full nuisance randomization + JEPA-style
   self-supervised pretraining (predict clean-view embedding from degraded view; dual
   encoder + EMA; pretrain on unlabeled DESI + 63.9M cutsky; fine-tune flow head).
   View design rule: degrade the SAMPLING at fixed field, never the field. This branch
   doubles as the only ICML/NeurIPS main-track candidate (P5).

## 4. Track 3 — Production (sequential; starts when gates close)

- **B** ONE bundled retrain: all GO winners (G2/G3/G4/G6 as applicable) + multi-scale
  aperture-density features (proven +0.08 λ1 R²) + softplus increments (best-calibrated)
  + A3-harmonized mocks. No incremental retrains.
- **C** Scale-out: tile BGS bright into overlapping wedges; graph→features→NPE per
  wedge via sbatch; de-dup overlaps by posterior averaging. **Start DESI collaboration
  publication process here.**
- **D** Validation: SBC/TARP; mass-anchored massive-halo recovery; n(z)-perturbation
  stability; full-footprint property closure; DESIVAST void cross-match.
- **E** Release: **GraphWeb-BGS** — per-galaxy TARGETID-keyed calibrated posteriors:
  P(λ1>λ_th) + 4 class probs (headline), λ means/stds, posterior width, OOD/MMD flag,
  model+trainset version. (If G4 fired: tensor/eigenvector columns → IA use case.)

## 5. Track 4 — Papers (interleaved; timeliness targets)

| Paper | Venue | Content | Trigger / deadline |
|---|---|---|---|
| **P1 Letter** | MNRAS Letters / RASTI | smoothing vs cluster recovery (mass-anchored anti-correlation), aperture≈1.4×scale matching, AUC-vs-completeness trap | results DONE — draft now, submit ~3 wk |
| **P2 Workshop** | NeurIPS ML4PS (deadline ~late Aug) | representation ablation (a,b,b′,c[,d]) + RSD-penalty decomposition + misspecification-gated transfer | needs G1–G4 by mid-Aug |
| **P3 Methods** | MNRAS / RASTI | calibrated NPE for web environments; SO(3)-about-observer symmetry analysis; production architecture | after Phase B |
| **P4 VAC** | MNRAS / ApJS + DESI review | GraphWeb-BGS + closure + DESIVAST | after C/D |
| **P5 ML main-track** | ICML/NeurIPS (conditional) | nuisance-invariant JEPA pretraining for amortized SBI, benchmarked beyond cosmology | ONLY if GateM fires |

## 6. Timeline (aggressive but honest)

- **Jul wk 1–2:** G1 + G1.5 + G2 (parallel CPU) · A1 + A2 (parallel) · P1 drafting.
- **Jul wk 3–4:** G3 (b′) · G6 · A3 cache rebuild · P1 submitted.
- **Aug:** G4 (if gated GO) · assemble P2 → ML4PS (~Aug 29) · Phase B bundled retrain.
- **Sep–Oct:** GateM check · Phase C scale-out · P3 drafting · DESI process starts.
- **Nov–Dec:** Phase D validation · VAC assembly · P4 drafting (DESI review into 2027).
- JEPA/P5 branch: only on GateM failure; 3–6 mo, runs alongside C/D.

## 7. Standing methodology rules (learned this cycle, non-negotiable)

Gate every retrain with a cheap mock-truth pre-check · evaluate in TARGET (eigenvalue)
space, never label space alone · anchor rare-class claims to physical labels (mass),
never AUC alone · match feature scale to target smoothing (~1.4×) · one bundled
retrain, never incremental · independence-from-geometry ≠ usefulness — always test
incremental value on truth.
