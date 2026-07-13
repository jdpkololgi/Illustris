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
| **G7 config FROZEN (2026-07-13)** | F-tier point-estimate config = **v2 variant A** (`gate_ftier_v2.py` scatter=tsc, decoder=unet), consuming the **shared `path1_wedge_union_r10hmpc_gnn_arrays.npz` union graph = the G3 production connectivity** (v1 rolled its own radius graph — dropped). Numbers: λ1 **0.841** (v1 0.840), λ2 0.900, λ3 0.932, clu-Sp +0.57 — tiny gain, but one preprocessing lineage shared with the calibrated λ1 product. v2_B (fno+survey-mask, 0.839) → the survey-mask is a **Phase-C full-footprint deployment toggle** (real boundaries), NOT part of the frozen accuracy config. F-tier remains the **point-estimate/eigenvector (IA) product, badged** — NOT the v1 calibrated headline (that stays G3+FMPE-λ1). | — | — | firms the encoder-consistent field branch for the Phase-B parallel retrain |
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
| **S1** | Cutsky-truth **shell-transfer matrix**, TWO TIERS (amended 2026-07-12): **(a)** GBM/aperture harness — train-shell-i → test-shell-j R²(λ1) grid + pooled±ñ rows + per-shell diagonal, cutsky downsampled per shell to the DESI ñ(z) spline (DESI-realistic), north wedge box only; **(b)** WINNER zero-shot — existing trained G3-GraphNet and CNN/F-tier evaluated per-shell on S2 caches, two SIMULTANEOUS tmux GPU sessions (hbm80g) = the S2.5 encoder-at-sparsity readout at production fidelity | (a) ½–1 d CPU; (b) ~½ d GPU after S2 | pooled+ñ ≥ per-shell − 0.02 on every shell ⇒ single-model default confirmed BEFORE the retrain; (b) decides the production encoder under real sparsity |
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
- **E** Release: **GraphWeb-BGS** — per-galaxy TARGETID-keyed posteriors. **v1 CONTRACT
  (amended 2026-07-13, Codex review §1):** the CALIBRATED science columns are λ1 only —
  posterior mean/std/quantiles for λ1 and **P(λ1>λ_th) = the three-axis-collapse (knot)
  probability** (ordering λ1≤λ2≤λ3 makes these identical) — plus width/information
  diagnostics, selection/boundary/OOD flags, and full provenance. λ2/λ3 and 4-class
  columns may be written for continuity but are badged **EXPERIMENTAL/UNVALIDATED**
  until the SBC-aware v1.1 work lands. Do NOT present 4-class as calibrated science.
  (If G7 fired: tensor/eigenvector columns → IA use case, point-estimate-badged.)

## 4b. VAC v1 hardening (Codex review adoptions, 2026-07-13)

1. **Spatial holdout (MANDATORY, affects the Phase-B split design):** hold out a
   contiguous RA block (e.g. RA 150–160) across ALL shells from training entirely;
   random transductive splits are optimistic for new sky. **Tempering fitted on a val
   region, assessed on a DISJOINT test region — never tuned on the final test shell.**
2. **Symmetric scope guard:** S1(b) proved the DENSE low-z end fails like the sparse
   end (GraphNet zero-shot −1.09; CNN best-case 0.002 at z0.05–0.15). If S4/S5 cannot
   validate z<0.15 or z>0.45, those rows ship OOD-FLAGGED or outside the validated
   range — symmetric, pre-registered.
3. **S5 battery additions:** reliability diagram + Brier score for P(λ1>0.2), global
   AND per shell AND mass-anchored slice (it IS the product); width-vs-realized-|error|
   + conditional coverage vs ñ, degree, boundary distance; **prior-dominated /
   low-information flag** (posterior var ÷ unconditional var) — calibrated-but-
   uninformative rows must say so.
4. **ñ spline discipline:** freeze the bandwidth/knot prescription BEFORE inference;
   run a two-bandwidth (smoother/rougher) sensitivity check; the claim is "conditioning
   on a smooth expected sampling intensity", NOT on the measured galaxy density (real
   radial modes must not be absorbed). Randoms-grounded selection = v1.1.
5. **Tile aggregation (scale-out):** overlapping-tile predictions are CORRELATED views
   — aggregate as a **centrality-weighted posterior mixture** (μ=Σwμ_t;
   Var=Σw(σ_t²+μ_t²)−μ²; P=ΣwP_t), or select the most-interior tile. NEVER multiply
   posteriors; never average variances. Buffered tiles, trim after inference; flag
   Delaunay edges bridging mask holes + extreme edge lengths; atomic idempotent
   per-tile outputs + completion manifests; resumable chains.
6. **Schema:** multi-bit OOD reasons (z-range, ñ-support, degree/edge-length, boundary,
   completeness, MMD, prior-dominated, tile-inconsistency) + provenance (checkpoint
   hash, repo SHAs, cache+spline hashes, TARGET_EPOCH=0.2, smoothing, λ_th, ordering).
7. **Ops:** golden-wedge canary end-to-end BEFORE scale-out; incremental CFS backups
   DAILY from now (not just Jul 21); checksum manifests for FITS + checkpoints.

## 5. Track 4 — Papers (interleaved; timeliness targets)

| Paper | Venue | Content | Trigger / deadline |
|---|---|---|---|
| **P1 Letter** | MNRAS Letters / RASTI | smoothing vs cluster recovery (mass-anchored anti-correlation), aperture≈1.4×scale matching, AUC-vs-completeness trap | results DONE — draft now, submit ~3 wk |
| **P2 Workshop** | NeurIPS ML4PS (deadline ~late Aug) | representation ablation (a,b,b′,c[,d]) + RSD-penalty decomposition + misspecification-gated transfer | needs G1–G4 by mid-Aug |
| **P3 Methods** | MNRAS / RASTI | calibrated NPE for web environments; SO(3)-about-observer symmetry analysis; production architecture | after Phase B |
| **P4 VAC** | MNRAS / ApJS + DESI review | GraphWeb-BGS + closure + DESIVAST | after C/D |
| **P5 ML main-track** | ICML/NeurIPS (conditional) | nuisance-invariant JEPA pretraining for amortized SBI, benchmarked beyond cosmology | ONLY if GateM fires |

## 6. Timeline — DEADLINE-COMPRESSED (rewritten 2026-07-12)

**HARD CONSTRAINT: NERSC shutdown Jul 22 – Aug 3.** The production VAC must be BUILT,
INTERNALLY VALIDATED, FROZEN and BACKED UP (scratch→CFS) by **Jul 21**. Operational
constraints in force: **sbatch unusable → everything via salloc+tmux chains**; use
**hbm80g** GPU nodes when memory-bound (40-vs-80GB roulette); test the two winners in
SIMULTANEOUS allocations wherever possible.

- **Jul 12–13:** S1(a) proxy matrix (CPU) · S2 shell-cache chain (5 shells, buffered
  builds from sentinelfix parent, union edges, NO dilution — ñ-conditioning replaces it)
  in tmux · S1(b) winner zero-shot on the shell caches (2 parallel GPU tmux) = S2.5.
- **Jul 14–15:** S3 conditioning build (ñ node feature SI-excluded + FMPE vector;
  ñ·V_voxel expected-counts channel for the field branch) · **spatial holdout baked into
  the pooled cache**: three RA-disjoint regions applied identically across all 5 shells —
  **train RA<145 · val/tempering 145–150 · test RA≥150** (test never trained; τ fit on val,
  assessed on disjoint test) · **Phase B**: full-range ñ-conditioned G3+FMPE retrain (hbm80g)
  with **F-tier v2_A** (tsc+unet, shared union-graph arrays = G3) point-estimate retrain in
  parallel · S4 readout + tempering re-fit on the val slab.
- **Jul 16–17:** **Phase C**: DR2 full-range inference — tile the BGS footprint via
  salloc/tmux chains (fallback scope if churn is slow: north Galactic cap first, rest
  post-shutdown) · GateM + S5 per-shell battery (SBC/TARP, MMD, width-vs-z).
- **Jul 18–20:** **Phase D**: mass-anchored, full-range closure, DESIVAST cross-match ·
  VAC v1 assembly (TARGETID-keyed FITS + docs + versioning).
- **Jul 21:** FREEZE + full backup to CFS (scratch is purge-prone and the shutdown is an
  outage) + push all repos. Buffer day for slippage.
- **Post-shutdown (Aug 3+):** DESI collaboration review process (human-paced; explicitly
  NOT achievable by Jul 22), P2 ML4PS (~Aug 29) from banked results, P3/P4 drafting,
  λ2/λ3 SBC-aware training for the 4-class columns (v1.1), JEPA/P5 only if GateM fails.

**VAC v1 scope guard (pre-registered):** if S1/S5 show the z≳0.45 regime (median union
degree ~3, deg0 10.5%, mock deg0 42%) cannot be validated in time, v1 ships z 0.05–0.45
with the 0.45–0.6 rows included but OOD-FLAGGED (posterior width + flag columns), and
v1.1 extends after the shutdown. Better a clean validated range than a rushed full one.
Dropped for v1 under the deadline: G3-to-7000 paper number (sbatch hostage; non-blocking),
G4/G8 exploratory branches, P1 letter drafting (post-shutdown).

## 7. Standing methodology rules (learned this cycle, non-negotiable)

Gate every retrain with a cheap mock-truth pre-check · evaluate in TARGET (eigenvalue)
space, never label space alone · anchor rare-class claims to physical labels (mass),
never AUC alone · match feature scale to target smoothing (~1.4×) · one bundled
retrain, never incremental · independence-from-geometry ≠ usefulness — always test
incremental value on truth.
