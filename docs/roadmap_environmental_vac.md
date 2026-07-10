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
| Classical-reconstruction floor | MEASURED 2026-07-07 (T1) | best classical (DTFE + exact FFT tidal solve, same split, calibrated) λ1/λ2/λ3 R² = 0.552/0.641/0.663 vs GraphNet 0.775/0.811/0.891 ⇒ learned headroom is REAL (+0.22/+0.17/+0.23); classical row mandatory in all future tables. `plan_field_level_multimodal.md` §6 |

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
| **G5 (d)** | Field-level baselines, SUPERSEDED/CONCRETIZED 2026-07-07 by `plan_field_level_multimodal.md`: **T1** classical ceiling (DONE — see §1 decisions) + **T2** CNN-on-counts control (3-D U-Net on voxelized wedge counts, same splits) | T2: 2–3 d GPU | T2 ≈ GraphNet ⇒ graph story needs rewriting; T2 ≪ GraphNet ⇒ sparse-tracer geometry matters | bounds graph-representation loss with a concrete floor already in hand |
| **G6** | FMPE head swap (flow-matching vs MAF; same conditioning) | few days GPU | SBC/TARP + NLL improve | posterior-head modernization; independent of encoder gates |
| **G7 (NEW)** | **Graph→field→Poisson (F-tier)**: GraphNet encoder → scatter → 3-D U-Net → δ̂ grid → fixed differentiable FFT tidal layer → eigenvalues (T4/F1 of the field plan). Eigenvalues fall out of physics; eigenvectors/IA product for free; classical-baseline solver already validated (voxel R² ≥ 0.992 vs cactus) | ~1 wk GPU | λ1 R² ≥ G3 (0.804) AND calibration ≥ current flow | output-representation axis; replaces G4 Tier B as the default eigenvector route |
| **G8 (NEW, cheap)** | **LUPI distillation** (T3): 3-D CNN teacher on sim δ/T patches latent-regularizes the GraphNet; teacher absent at inference | 2–3 d GPU bolt-on | ΔR²(λ1) > seed noise (≥3 seeds) | privileged sim fields as regularizer — the honest version of "multimodal training, unimodal inference" |

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
5. **T5 prerequisite (added 2026-07-07):** before ANY sim+real latent alignment
   (JEPA/DANN/contrastive) touches DESI, run the truth-known sim↔sim control
   (TNG↔Abacus or path1↔staged-mock) — latent alignment can HIDE domain shift and
   silently bias NPE posteriors; the closure-test machinery is the validator. See
   `plan_field_level_multimodal.md` T5.

## 3b. Track 2b — S-TRACK: full-z-range selection function (ADDED 2026-07-09)

**The problem (previously implicit, now explicit):** the VAC must cover BGS z≈0.05–0.6,
where the magnitude limit drives a **~165× density falloff** (measured, A2 full-range
JSON: n̄ at z 0.15–0.25 vs 0.45–0.55). Training so far = ONE shell (z 0.2–0.3). SI
per-graph medians absorb a *uniform* scale (validated at 0.73×), NOT two decades — and
even perfectly rescaled features cannot fix *calibration*: an amortized NPE trained at
one density is overconfident at sparser densities. Posterior width MUST grow with z.
NEW measurement: mock/DESI ratio degrades with z (0.91 → 0.54 → **0.28** at z 0.45–0.55)
⇒ full-range shape-match-by-dilution would cost 72% of training data — dilution alone
CANNOT extend A3 to the full range.

**Design decision (default): ONE amortized model, conditioned on the selection —
"amortize over the sampling intensity."** The maglim+fiberassign mock already forward-
models the selection; we train across the full range and expose the sampling intensity
to the model as **ñ(zᵢ)** = smooth fit to each dataset's OWN n(z) (DR2 spline at
inference; mock spline in training):
- node feature, **by-name EXCLUDED from SI normalization** (like Clustering — it is the
  covariate; the median would erase it), and
- appended to the **FMPE conditioning vector** (heteroscedastic amortization — the
  mechanism by which posteriors widen at high z; FMPE = confirmed production head).
Conditioning on **ñ, not z**: density regimes overlap mock↔DESI even where z-profiles
diverge (the 0.28 problem dissolves — mock high-z shells teach the density regime DESI
occupies slightly deeper). Precedent: radial selection functions are explicit inputs in
all field-level reconstruction (Wiener/BORG/ELUCID); conditioning amortized NPE on a
nuisance is standard SBI practice. Per-shell models remain the PRE-REGISTERED FALLBACK
(cons: K× training, seam discontinuities at shell boundaries in a public VAC, no
statistical sharing exactly where data is scarcest — high z).

| Gate | Experiment | Cost | GO criterion / decides |
|---|---|---|---|
| **S0** | Full-range selection atlas: DR2-vs-sentinelfix n(z) 0.05–0.6 full footprint; smooth ñ(z) splines (both datasets); per-shell graph stats (Delaunay edge length, degree@10 Mpc/h, union edge counts, **voxel occupancy** for the U-Net/F-tier) | ½ d CPU | quantifies per-shell OOD for EACH winner architecture; produces the ñ(z) conditioning functions |
| **S1** | Cutsky-truth **shell-transfer matrix** (GBM/aperture harness): train-shell-i → test-shell-j R²(λ1) grid + pooled+ñ-conditioned row + per-shell diagonal | 1 d CPU | pooled-conditioned ≥ per-shell − 0.02 on every shell ⇒ single-model default confirmed cheaply BEFORE any retrain |
| **S2** | Training-data extension: multi-shell buffered wedges from the **sentinelfix parent** (A1 unblocked z-expansion) with LIGHT per-shell gradient trimming only (no global dilution); optional dilution-ladder augmentation (±20% around ñ) for between-shell robustness | ~1 d | full-range caches for Phase B |
| **S3** | Winner-specific conditioning: GraphNet → ñ node feature (SI-excluded) + FMPE conditioning; **F-tier/U-Net → ñ(z)·V_voxel expected-counts input channel** (net can form Poisson contrast; prevents grid OOD); union graph unchanged (fixed comoving radius correct; Delaunay guarantees connectivity at any density) | in Phase B build | implementation spec |
| **S4** | Production showdown at matched compute: pooled-conditioned vs 2 shell-models (low/high z) | part of Phase B eval | conditioned within ΔR² 0.02 AND per-shell-calibrated ⇒ ONE model ships; else shells + overlap-blend seam protocol (posterior mixture over Δz≈0.02) |
| **S5** | Per-shell validation battery folded into Phase D: SBC/TARP **per shell**, GateM MMD per shell, closure per shell, and the **width-monotonicity sanity check** (posterior width must grow toward high z — the physically required signature that conditioning worked) | Phase D | conditional calibration, not just marginal |

**S-track is ALSO the encoder arbiter:** T2's U-Net 0.876 and F-tier's 0.841 were
measured on a DENSE wedge (~2.4× DESI); grids degrade with sparsity while
Delaunay∪radius adapts — so the G3-vs-F-tier-vs-U-Net production decision MUST be made
on S2's full-range data (S0's voxel-occupancy stats will preview it). Do not crown an
encoder on the dense wedge.

**Known systematic (documented, not blocking):** single-snapshot (z=0.2) cutsky labels
carry no growth evolution across 0.05–0.6; the VAC inherits a "z=0.2-epoch tidal field"
convention, with sim-to-real mismatch growing with |z−0.2|. Mitigation: document; optional
post-hoc D(z)/D(0.2) rescaling of λ; long-term = multi-snapshot lightcone labels. S5's
per-shell GateM partially detects it.

## 4. Track 3 — Production (sequential; starts when gates close)

**FIELD-LEVEL UPDATE (2026-07-08) — production encoder question reopened, favourably.**
Per `plan_field_level_multimodal.md` §7–9: the **F-tier (G7) graph→field→Poisson beat G3**
on accuracy (λ1 0.841 vs 0.804; T2 CNN-on-counts 0.876), so Phase-B **B** should evaluate
the F-tier as the production encoder/output (graph encoder → decoded δ̂ → fixed FFT physics
→ eigenvalues), with the flow head retained for calibration (the F1-calibration gate
decides). Equivariance (old G4) is deprioritised — do NOT block B on it. **UNCHANGED and
NOT threatened by the T2 CNN result:** the calibrated-NPE core, TARP/SBC, the TARGETID VAC,
closure tests, DESIVAST, and the published RASTI classification — a point-estimate R² on one
dense (~2.4× DESI) wedge does not touch any of these. The sparse survey-scale regime (voids,
n(z) range) is where a grid degrades and the graph/point-cloud encoder most plausibly wins —
an argument FOR the graph encoder at production scale. See field-level §9.3.

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
  model+trainset version. (If G7 fired: tensor/eigenvector columns → IA use case, plus
  optional wedge δ̂ density maps as a field-level product; G4 Tier B is the fallback
  route to the same columns.)

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
