# SCIENCE_LOG.md — shared brain: Claude Desktop (science) ⇄ Claude Code (NERSC)

### 2026-07-08 — [science] Where the field-level results leave the three plans (and: the graph PhD work is NOT shown useless)
- **The honest worry** (JDPK): "has a dumb CNN beating my GraphNet shown 2.5 yr of graph
  work useless?" **No — and precisely why is worth stating.** Full argument in
  `plan_field_level_multimodal.md` §9; summary here.
- **Result in one line:** the field-level program is VALIDATED and the graph is VINDICATED
  as an encoder; what changed is the FRAMING of the architectural lever — it's representation
  *scale* + a physics-grounded *output*, not attention and not equivariance. The
  SBI/calibration/VAC core and the published RASTI classification are untouched.
- **Why "CNN beats graph" is not the finding:** (1) **T4/F1 is the rebuttal** — a GRAPH
  encoder + field/physics decoder = 0.841, above every graph baseline (0.775/0.804), near
  the CNN (0.876), with mean-agg and NO attention. The winning net IS a graph net. (2) A
  **CNN is a GNN on a lattice**, so T2 = "fixed-scale sampling > Delaunay receptive field" —
  a graph-CONSTRUCTION result (your G4 thesis; you'd already seen G3 union 0.804 > Delaunay
  0.775). (3) The comparison is **confounded/unfinished**: MSE head vs NPE posterior-mean;
  the matched *production*-GraphNet-MSE control is still TODO (the 0.704 is a weak EGNN-lite
  smoke); raw wedge is ~2.4× DESI density (easiest for a grid). (4) The **metric isn't the
  thesis** — the VAC deliverable is calibrated posteriors (SBI/NPE, TARGETID VAC, closure,
  DESIVAST), which a point-estimate R² doesn't touch. (5) All on **one dense wedge**; the
  sparse survey-scale regime (voids, n(z)) is where grids degrade and graphs win — untested.
- **Per-plan status:** MULTIMODAL — central hypothesis CONFIRMED (F-tier passed accuracy
  gate); next = F1-calibration (flow head) → F2 → F4. G4-PROPER — equivariance DEPRIORITISED
  (its own P1a≥0.80 gate fired 3×); shelve SEGNN/Equiformer P1b + Tier B; F-tier supersedes
  Tier B for eigenvectors; attention demoted to second-order. ROADMAP→VAC — evaluate the
  F-tier (G7) as the production encoder in Phase-B (it beat G3); SBI core unchanged.
- **Thesis framing matures** from "graphs are the architecture" to "the graph is an
  excellent encoder for a physics-grounded field inference, and here is why" — a stronger,
  more defensible contribution (you built the dataset, the calibrated-NPE apparatus, the DESI
  application, AND the understanding that led to the F-tier that beats everything).

### 2026-07-08 — [code] Field-level tests run: T2 CNN-on-counts + T4/F1 graph→field→Poisson both beat every graph baseline; T3 shelved
- **T4/F1 (the centerpiece) PASSES its accuracy gate:** graph encoder (EGNNlite,
  mean-agg, NO attention) → differentiable CIC scatter → 3-D U-Net → δ̂ → fixed
  FFT physics layer → analytic 3×3 eigensolver → eigenvalues. **λ1/λ2/λ3 =
  0.841/0.897/0.931**, ≥ G3 (0.804). The graph encoder + field/physics decoder
  nearly matches the pure CNN and beats every prior graph baseline ⇒ the "CNN is
  killing my graph work" worry is answered: graphs are vindicated as the encoder;
  the win is the fixed-scale, field-shaped OUTPUT. Calibration half of the gate
  (≥ current flow) still pending the invariant-latent→FlowJAX head.
- **T2 CNN-on-counts is real and robust:** 3-D U-Net on voxelized galaxy COUNTS
  (5 Mpc, 1.4M params) → **λ1 0.876 ± 0.004** (3 seeds), beats GraphNet 0.775.
  Spatially validated: T-web **4-class accuracy 0.882** (void/wall/filament/cluster
  F1 0.906/0.878/0.872/0.845), class fractions within ~0.3% of truth, confusion
  only between adjacent classes, Spearman(pred,truth λ1)=0.965. Interactive 3-D
  viz built (artifact + PNGs) — predicted cosmic web tracks truth cleanly.
- **Framing (JDPK):** a CNN IS a GNN on a lattice, so T2 = "fixed-scale regular
  sampling + density-valued nodes > Delaunay receptive field" (same lesson as G3),
  NOT "CNN > graphs". Attention looks second-order, not non-essential (needs the
  clean within-F-tier on/off test). T2 also de-risks a grid-native JEPA (P5).
- **Caveats gating interpretation** (NOT yet resolved → §8 controls in the plan):
  MSE point head vs NPE posterior mean; RAW wedge is ~2.4× DESI density (easiest
  for a grid). Controls queued: matched-estimand graph MSE head, nzharm-density
  re-run, cell-size sweep on the cluster slice, clean attention on/off.
- **Ops lessons (bit us):** (1) `srun --overlap` can silently fail to bind the GPU
  for a *later* step in an allocation → T3 CPU-trained for 1.5 h invisibly; fix =
  a fail-fast `torch.cuda.is_available()` guard on every step. (2) `torch.linalg.
  eigvalsh` on CUDA tried to allocate 51 GiB for (N,3,3) (cuSOLVER blowup) →
  replaced with analytic Cardano 3×3 (matches numpy 7e-15). (3) interactive-only
  runs per JDPK — reuse idle allocations via `srun --jobid`; no sbatch.
- **T3 (LUPI)** re-running in a fresh 4 h GPU alloc (capped steps, guarded);
  box-frame teacher validated (25× random density). Verdict pending.
- Refs: plan `docs/plan_field_level_multimodal.md` §7–8; scripts
  `workflows/sbi/gate_t2_cnn_counts.py`, `gate_t4_graph_field_poisson.py`,
  `viz_t2_wedge_clusters.py`; `/pscratch/.../abacus/field_level_tests/`.

### 2026-07-07 — [science] Multimodal/field-level direction: "we might be throwing too much information away" + the classical floor that motivates it
- **The idea (JDPK):** simulations give us complete 3-D density/tidal/eigenvalue FIELDS,
  and the pipeline reduces them to 3 numbers per galaxy before any model sees them.
  Proposals discussed: (1) train-time 3-D CNN on the sim fields latent-regularising the
  GraphNet (absent at inference); (2) semi-/self-supervised sim+real joint training;
  (3) tensor-valued outputs via a more physical model.
- **Information accounting (the discipline):** at DESI inference the input is the sparse
  redshift-space catalog, full stop — a train-only modality cannot raise the mutual
  information, only regularise (idea 1 = Vapnik LUPI / generalized distillation; cheap
  ablation, modest expected gain). Idea 2 is risky: latent alignment can HIDE sim→real
  shift and silently bias NPE posteriors → truth-known sim↔sim control required first.
- **The reframe that stuck: flip the field from INPUT to OUTPUT.** Graph → decoded δ̂
  grid → **fixed differentiable FFT physics layer** T̂ij(k)=(ki kj/k²)W₇(k)δ̂(k) →
  eigenvalues at galaxies. Nothing about the density→tidal map is learned — it is
  hard-coded mathematics; gradients flow through eigvalsh+FFTs into δ̂. Eigenvalues
  "fall out from the physics"; symmetry/trace/rotational consistency guaranteed;
  eigenVECTORS (→ IA science) come free WITHOUT e3nn irreps or the Tier-B tensor
  rotation crux (scalar field needs only the affine map). Composes with, does not
  compete against, the G4 nonlocal-operator story (same 1/k² kernel).
- **T1 classical-reconstruction baseline RUN (the "first move"):** textbook density
  estimation + the exact tidal solve, same estimand + test split as the GraphNet.
  Best classical (DTFE): **λ1/λ2/λ3 R²(cal) = 0.552 / 0.641 / 0.663** vs GraphNet+NPE
  0.775/0.811/0.891 (CIC 0.546, Wiener≈CIC, extra smoothing hurts; interior-only λ1
  0.60). **⇒ The GraphNet's +0.22/+0.17/+0.23 margin over linear reconstruction is
  genuine learning** (RSD + bias + nonlinearity) — architecture work is chasing real
  signal, and the biggest margin is on λ3 (collapse axis / clusters).
- **Physics-layer validation banked:** the same FFT solve reproduces the stored cactus
  eigenvalues voxelwise on a 512³ subbox of the 10% particle grid at
  **R² = 0.992/0.995/0.997** — the F-tier physics layer AND the G4 Tier-B validation
  anchor are both de-risked in one shot.
- **Plans updated:** new canonical doc `docs/plan_field_level_multimodal.md` (test list
  T1–T5 + F-tier phases); `plan_g4_proper_equivariant_tensor.md` (classical floor row,
  Tier-B amendment, §10 companion pointer); `roadmap_environmental_vac.md` (G5
  concretized→T2, NEW gates G7 field-decoder + G8 LUPI distillation, T5 prerequisite on
  the JEPA branch, G7 columns in the release spec).
- Refs: `workflows/abacus_tweb/classical_tidal_baseline.py`;
  `/pscratch/sd/d/dkololgi/abacus/classical_baseline/` (scores + per-estimator
  predictions + solver validation JSONs).

### 2026-07-04 — [code] G4-PROPER WAVE 1 COMPLETE (A–E): point-attention wins, dynamic graph & steerable lose; wave 2 (D seeds + F) launched
- **Full wave-1 board (all λ1/λ2/λ3, positions-only unless noted, point-estimate MSE except G3/A/baseline = NPE posterior mean):**
  - baseline GraphNet+NPE, Delaunay, curated: **0.775** / 0.811 / 0.891
  - G3 GraphNet+NPE, union, curated: **0.804** / 0.846 / 0.895  (production anchor)
  - A GraphNet+NPE, radius, curated: **0.752** / 0.799 / 0.876
  - **D point-attention MPNN, radius, POSITIONS-ONLY: 0.726 / 0.807 / 0.838, clu ρ 0.54**
  - E attentional DGCNN, DYNAMIC feature-space kNN, positions: 0.507 / 0.662 / 0.681, clu 0.36
  - B SEGNN steerable+attn, union, positions: 0.536 / 0.610 / 0.653, clu 0.35
  - C SEGNN steerable+attn, radius, positions: 0.423 / 0.411 / 0.513, clu 0.37
  - (all C/D/E converged: C ran full budget val-plateaued, D early-stopped @4000, E early-stopped @1925 — not truncated.)
- **THREE headline reads:**
  1. **Point-cloud from raw geometry works & ≈ curated features.** D (positions+LOS+radius
     graph) λ1 0.726 vs A (same graph, curated features) 0.752 — matched estimand
     (MSE point est ↔ NPE posterior mean both target E[λ|x]) → positions recover ~96% of
     the curated-feature signal; the cuGraph features add ~+0.03. JDPK's point-cloud
     instinct validated.
  2. **Dynamic feature-space graph HURTS: E−D = −0.22 λ1** (matched inputs+attention). The
     §1(d) subsumption hypothesis is CONFIRMED, not assumed: for a compact-support target,
     letting edges roam in feature space imports non-local neighbours and costs ~0.22.
     Attention did NOT rescue it (it reweights candidates, cannot attend to evicted physical
     neighbours). Clean publishable negative for the dynamic-graph line.
  3. **Steerable SEGNN underperformed everything** (B 0.536, C 0.423) — but CAPACITY-
     CONFOUNDED (fast 179k config vs D's larger net; e3nn-TP-throughput-limited). B(union)
     0.536 > C(radius) 0.423 → union>radius holds inside the steerable family too.
- **GPT-5.5 deep-review memo — synthesis (JDPK relayed):** (a) upgrades the graph story via
  the inverse-Laplacian: T̂ij(k)=(ki kj/k²)W_R(k)δ(k); the 1/k² makes the smoothed tidal
  tensor genuinely NONLOCAL (real-space kernel ~anisotropic power-law, not exp-suppressed),
  so the union graph is a **discrete quadrature of a nonlocal operator** — Delaunay bridges =
  adaptive void connectivity, radius edges = fixed aperture, attention arbitrates. This also
  explains E: useful long edges are GEOMETRY-anchored (Delaunay bridges help), not FEATURE-
  anchored (DGCNN dynamic edges hurt), and softens the original review's "compact support ⇒
  skip long-range". (b) Its pre-written decision tree lands on branch 2 (D high + B low ⇒
  steerable IMPLEMENTATION may hurt ⇒ RPP-relaxed / hybrid, NOT immediate heavy Equiformer).
  Convergent with our plan's RPP escape hatch. (c) New paper narrative endorsed: "discovering
  the correct discrete support for a nonlocal cosmological operator", not a failed arch search.
  (d) Production stays G3 (unchanged). (e) Tier-B physics constraints logged for later (trace–
  Poisson TrT_R=δ_R; Hessian integrability ∂k Tij=∂j Tik — a free symmetric tensor field is
  NOT automatically a Hessian field). Full memo lives with JDPK.
- **Redo policy: NO redos** (runs internally valid). Needed instead = diagnostic layer.
  **Wave 2 LAUNCHED now** (tmux `g4_wave2` on login node, SSH-independent):
  **D seeds 43 & 44** (seed variance for the headline result; single-seed so far) + **run F**
  = attentional DGCNN + CURATED Delaunay features (F vs E = feature axis for dynamic graph;
  F vs A = dynamic vs radius at matched curated features). D-seed43 already allocated
  (55499510).
- **DGCNN control knobs added** (answering JDPK's "how much control over the dynamic graph"):
  `gate_g4_p1e_dgcnn_attn.py` now takes `--curated-features` and `--knn-radius-cap` (restrict
  feature-space kNN to a physical envelope — 'learned selection WITHIN locality', the direct
  fix for why E lost; void nodes fall back to physical-kNN). Verified: capped kNN keeps edges
  local, rotation-invariance holds with curated+capped (1.2e-16). F runs UNCAPPED (= E arch)
  to isolate the feature axis; capped variant is the ready follow-up if F or E's void slice
  motivates it.
- **Deferred diagnostics (not yet built, next after wave 2):** environment-sliced eval
  (V/W/F/C) across G3/A–F; connectivity-residual diagnostics; union edge-type attention-mass
  attribution. Explicitly NOT queued: matched-capacity heavy SEGNN, Equiformer/MACE, Tier B.
- **Standing:** G3 resume (55441429) still PENDING on plain gpu&a100 — JDPK to protect with
  `scontrol update JobId=55441429 Features="gpu&hbm80g"` (Claude permission-blocked from the
  queued job).

### 2026-07-04 — [code] RUN B RESULT (P1b SEGNN×union, positions-only): λ1 0.536 — MISS on gate, but capacity-confounded; chain hardened
- **B (SEGNN steerable+attention, union graph, positions+LOS only, 179k params, 1348
  steps, point-estimate MSE):** test-set λ1 R² **0.536**, λ2 0.610, λ3 0.653;
  cluster-slice λ1 Spearman **+0.35** (baseline 0.54). Clean training: train 0.234 ≈
  val 0.218, NO overfitting (the DropEdge + attention-dropout discipline held).
- **Verdict: clear MISS** on the P1b gate (needs λ1 ≥ 0.75 AND beats the P1a controls
  radius 0.752 / union 0.804). **BUT heavily confounded — do NOT read as "equivariance
  fails":** (1) this is the FAST config (179k params vs baseline ~992k; hidden halved,
  3 layers) forced by e3nn TP throughput (~9.5 s/step even so); ~5× under-capacity.
  (2) positions-only, so the 0.536→0.774 gap largely re-expresses the value of the
  curated features (G1: features carry most of the recoverable signal). (3) point-
  estimate MSE head is R²-FAVOURED vs the baseline's posterior mean, so 0.536 is if
  anything optimistic. Net: positions-only steerable+attention at reduced capacity does
  not reconstruct what curated cuGraph features encode — expected direction, but the
  capacity confound must be resolved (matched-param SEGNN) before the gate is honestly
  called or an RPP-relaxed variant is run.
- **Clean comparisons still pending:** B−C (union vs radius WITHIN steerable, matched
  everything else) needs C; the equivariance contrast needs the positions-only non-
  equivariant runs D (radius) / E (dynamic). Only then does the factorial read.
- **tmux chain hardened + restarted (login30):** the liveness check now distinguishes a
  live run from a failed-salloc/crashed attempt (tail terminal-marker grep, not mere
  file freshness) — was waiting 25 min on stale failed-attempt logs; now retries
  promptly when a slot frees. Verified: on restart it correctly skipped B (results) and
  C (live), and launched D (job 55484009). B's slot freed → D pending a node.
- Interim conversion note (for the record): B's pooled val MSE 0.218 in globally-
  standardised space → pooled R²(Y) 0.78 is INFLATED by between-component ordering; the
  honest per-component decomposition predicted mean ~0.55, matching the measured
  0.536/0.610/0.653. Global-standardised val loss is NOT comparable to the FlowJAX runs'
  val NLL — different metric entirely.

### 2026-07-04 — [code] RUN A RESULT (P1a-i): radius-only 0.752 < union 0.804 at matched budget — COMPLEMENTARITY is the lever; unattended tmux chain armed
- **Radius-only control @3749 epochs (posterior-mean, 128 samples, shared test split):**
  λ₁ R² **0.7519**, λ₂ 0.7989, λ₃ 0.8757, mean 0.8088; best val NLL 1.4248.
  Anchors: union@3749 **0.8041**/0.8461/0.8955 (val NLL 0.856); Delaunay-full@7000
  0.7750/0.8105/0.8912 (val NLL 1.107).
- **Readings:** (1) **JDPK's prediction CONFIRMED: union > radius-only** (+0.052 λ₁ at
  matched budget) — Delaunay edges carry information the radius ball misses (registered
  mechanism: 172k Delaunay pairs longer than 14.78 Mpc = void bridges; radius-only has
  110 isolated nodes / 178 components). (2) Radius-only ≈ Delaunay-full (0.752@3749 vs
  0.775@7000, budget-mismatched in Delaunay's favour) — the pure constructions are
  roughly comparable; **neither alone is the lever: the UNION (complementary edge
  populations + attention arbitration) is.** (3) P1a threshold: radius-only does NOT
  hit ≥0.80 alone ⇒ "construction is the lever" holds in the ADD-radius form, not the
  replace form. P1b's bar remains the union control (0.804@3749; higher at G3-7000).
- Follow-up registered: environment-sliced eval (V/W/F/C) on A vs G3 checkpoints to
  test the mechanism (radius deficit in voids; radius gain over Delaunay in clusters).
- **SSH-independence (JDPK disconnecting):** all remaining wave-1 automation moved to a
  tmux orchestrator on **login30** (`tmux attach -t g4_chain`;
  `workflows/sbi/run_g4_chain.sh`): idempotent — skips runs whose results exist, leaves
  live runs alone (log-freshness), (re)launches anything missing incl. B/C if the
  Claude-owned sallocs die at disconnect. D's smoke script now writes a results file
  (`--out-file`) for completion detection. Claude-bound D/E watchers stopped.

### 2026-07-04 — [code] Run E added (attentional DGCNN, unfixed graph); D relabelled; B throughput fixed; wave now A/B/C/D/E
- **D relabelled (JDPK caught it):** "Point-Transformer-class" was an over-badge — a
  point-cloud network IS an attention GNN on a coordinate-derived graph, so D is
  architecturally the same object as run A. D's honest label: **positions-only attention
  control**; its scientific content is the FEATURE axis (D−A), not a new model family.
- **Run E built (P1a-iii, JDPK design):** attentional DGCNN — the ONE wave-1 model whose
  graph is genuinely NOT fixed: kNN recomputed PER LAYER in learned feature space
  (layer 0 = coordinate kNN, k=20), EdgeConv max-pool REPLACED by GAT/GAPNet-style
  4-head attention, positions+LOS only, eigenvalue supervision.
  `gate_g4_p1e_dgcnn_attn.py` + runner; selftest: forward/backward finite AND full-model
  **rotation invariance 1.4e-16** (all learned features derive from invariant scalars →
  feature-space kNN itself rotation-invariant, dodging DGCNN's usual symmetry breakage).
  **E−D = learned dynamic candidates vs fixed physical candidates** at matched
  inputs+aggregation — the §1(d) subsumption claim gets TESTED, not assumed.
- **Conceptual point logged (plan §5A):** attention fixes the WEIGHTING axis (the
  over-smoothing-like fixed-kernel problem, JDPK's analogy — half right), but NOT the
  CANDIDATE-SELECTION axis: attention cannot attend to an absent edge, and fixed-k
  feature-space kNN can evict physical neighbours entirely. E mitigates via layer-0
  coordinate kNN + per-layer geometry scalars. Prior: D ≥ E overall, but **voids are
  where E could win** (environmental parameter sharing between feature-similar,
  spatially distant void galaxies) — an E win in voids would revive the multi-scale
  graph line.
- **B throughput crisis fixed:** e3nn TP einsum kernels (not matmul — TF32 did nothing)
  gave 30–49 s/step ⇒ ~244 steps/budget = void. Config: hidden halved
  (16x0e+8x1o+4x2e → 179k params), 3 layers, DropEdge 0.5 (full edges for val/eval),
  val every 50 → **9.55 s/step, T_max=1256 steps**. Matched-param discipline vs the
  992k GraphNet is broken for the smoke (noted in results); returns in multi-seed phase.
- **Queue chain (QOS 2-slot):** A (RUNNING, epoch ~1400/3750, on pace) + B (RUNNING,
  fast config) → C (SEGNN radius) → D (positions-only radius) → E (DGCNN) via watchers.
- SKIP-list updates: DGCNN → promoted to run E; PointNet(++) stays out (no/fixed-radius
  neighbourhoods — redundant with A/D).

### 2026-07-04 — [code] Run D added (P1a-ii point-cloud control); P1a split into two controls; plan §5A/§6 reconciled
- JDPK caught a real ambiguity: the plan's §5A table named Point Transformer/DGCNN as
  "the" P1a control while §6 (and the source report's Rec. #1) specified the in-stack
  GraphNet radius control — these are TWO different controls answering different
  questions, and the launched run A implements only the latter. Fixed by splitting:
  **P1a-i** (run A, in-stack GraphNet, single-variable edge-set ablation) and **P1a-ii**
  (**run D, NEW**): Point-Transformer-class attention MPNN, POSITIONS+LOS ONLY, node
  features [1, |pos|/median], neighbourhoods built at LOAD TIME from the point
  distribution (`gate_g4_egnn_smoke.py --positions-only --build-radius-mpc 14.78`).
  Verified: load-time radius build reproduces the prebuilt npz EXACTLY (1,816,273
  pairs) — for a fixed rule, "no preconstructed graph" and "prebuilt" are the same
  edges; the genuinely dynamic family (DGCNN feature-space kNN) stays SKIPPED per the
  §1(d) subsumption argument (compact-support target).
- **Attribution algebra now complete** (added to plan §5A): D−A = raw geometry vs
  curated features; C−D = equivariance alone (matched inputs+graph); A−G3 = radius vs
  union; B−C = union vs radius within equivariant. Without D, a P1b result could not
  be attributed to equivariance vs "any attention net on raw geometry".
- **Supervision policy recorded in plan:** ALL wave-1 models train on eigenvalues
  (Tier A); steerable nets predict the tensor internally only; non-equivariant nets
  must not emit a fixed-frame tensor. Tensor TARGETS = Tier B, gated.
- Run D queued behind the QOS 2-slot cap via watcher (fires after C is training and a
  slot frees). Sequencing: A ends → C; B ends → D.

### 2026-07-04 — [code] G4-PROPER wave 1 LAUNCHED: 2×2 factorial {control, SEGNN} × {union, radius}; P0 at machine precision
- **Design (JDPK-approved):** complete the factorial with 3 runs, reusing G3 as the
  4th cell: control×union = G3 (0.8041@3749 ✓), **A** control×radius (existing GraphNet+
  curated features+FlowJAX, ONLY edges swapped; `--epochs 3750` = exact matched budget),
  **B** SEGNN×union, **C** SEGNN×radius (positions+LOS only, §0 purity). Interactive QOS
  caps 2 concurrent jobs → C auto-launches via watcher when A or B frees a slot.
- **Radius-only graph built** (`build_union_graph_arrays.py --radius-only`, +connectivity
  diagnostics): 1,816,273 pairs @14.78 Mpc. **Void-fragmentation prediction confirmed
  structurally:** 178 components, 110 isolated nodes, largest component 99.64% (union =
  single component via Delaunay's triangulation guarantee); 172,459 Delaunay pairs are
  LONGER than 14.78 Mpc (the void bridges radius-only loses). Cache
  `path1_flowjax_3d_lineareig_si_radiusgraph` built by an edge-transplant procedure
  **gold-validated by exactly reproducing the production union cache on all 9 fields**;
  nodes/masks/targets byte-identical to baseline.
- **P1b script** `workflows/sbi/gate_g4_p1b_segnn.py`: e3nn steerable MPNN (hidden
  32x0e+16x1o+8x2e, 4 layers, 4 heads, 921k params), invariant-logit segment-softmax
  attention, 1x0e+1x2e head → sym 3×3 → ANALYTIC eigvals → MSE on existing LAMBDA
  (Tier A). Single GLOBAL affine target scaling (per-component would be
  tensor-inconsistent). Regularisation parity: wd 0.08, attention dropout 0.1,
  early-stop on val.
- **Debugging harvest (all committed, each a real finding):** (1) 40-vs-80GB A100
  roulette — FlowJAX needs ~43GiB/device on union/radius; P1a pinned `hbm80g`; the
  PENDING G3 resume sbatch 55441429 still has plain gpu&a100 = OOM RISK if it lands on
  40GB (JDPK to decide: `scontrol update JobId=55441429 Features="gpu&hbm80g"`).
  (2) **JAX preallocates 75% of GPU when unpickling the cache's jnp arrays** under a
  PyTorch job — the "non-PyTorch 30GB" both B OOMs; fixed `JAX_PLATFORMS=cpu`.
  (3) cuSOLVER batched eigvalsh wants ~25GiB workspace at N~1e5×3×3 → closed-form
  trigonometric sym-3×3 eigensolver (plan §7 mitigation; matches LAPACK to 1e-14).
  (4) e3nn per-edge TP weights blow memory at 4M edges → edge-chunked checkpointed
  values (chunk-equivalence 4.4e-16); nested layer+chunk checkpoints break e3nn
  TorchScript on recompute → flat inner-only checkpointing + narrow shared logits MLP.
  (5) **P0 precision forensics:** GPU index_add atomics (~1e-7 float64 noise) → CPU
  test; CartesianTensor float32 change-of-basis → float64 buffer; **e3nn bakes CG
  constants in default dtype at module creation** (.double() doesn't recast) → build
  test model under float64 default. Result: **equivariance 5.0e-16 = machine
  precision** (the earlier 6.6e-9 was baked-float32 noise, not architecture).
- **Status at log time:** A RUNNING (~1.6 s/epoch, will hit 3750 + auto-eval well within
  4h), B RUNNING (step 0: train 1.64/val 1.42), C armed. Prediction registered: union ≥
  radius-only with the gap concentrated in voids (connectivity), radius ≥ Delaunay in
  clusters (aperture). Anchors: Delaunay-full 0.7750, union@3749 0.8041.

### 2026-07-03 — [code] G3 readout ASSESSED: connectivity axis wins; Phase B needs union×nzharm merge; G4 bar quantified
- Verified on disk (`..._uniongraph_EVAL3749/flowjax_sbi_results_*.txt`): λ₁ 0.8041 / λ₂
  0.8461 / λ₃ 0.8955 (mean 0.8486) vs baseline 0.7750/0.8105/0.8912 (0.8256); test NLL
  0.8964 vs 1.1392 (~0.24 nats/galaxy sharper posteriors) — at 56% of training budget.
- **Ladder update:** union claims +0.029 of the ~+0.085 gap between the Delaunay GraphNet
  and the G1.5 real-space bound (0.86–0.88). Remaining headroom for EVERYTHING else
  (equivariance, FMPE, features) ≈ 0.05–0.08. The Delaunay receptive-field scale-mismatch
  hypothesis is now validated at production level — graph construction was the
  first-order lever, exactly as the amended G4-PROPER reframe (§8) argues.
- **Integration gap found:** the two data-side wins live in SEPARATE caches (nzharm =
  Delaunay-on-harmonized; union = Delaunay∪radius on UN-harmonized). Phase B requires a
  **union×nzharm merged cache** (union edges on harmonized points; ~20 min) — tasked.
- G4-PROPER implications: bar formally raised to the FINAL union number; P1a
  (radius-only) control gains importance (distinguishes radius-sufficient vs
  union-necessary); the surviving unique payoff of Tier B is EIGENVECTORS/orientations
  (IA science) — a capability no connectivity fix can provide, which is the honest
  physics case for keeping the symmetry branch alive within its ~0.05 accuracy budget.
- Held sbatch 55441429 NOT released (honoring JDPK's earlier correction) — release is
  his call. Status figure: `docs/roadmap_status_20260703.png` (+ canonical figures dir).

### 2026-07-03 — [code] G3 interactive done@3749 (NOT trained-out); existing G4-PROPER plan AMENDED (attention required, P1a/P1b split)
- **G3 status + PRELIMINARY EVAL (GO signal):** the interactive run (job 55442933) ended —
  4 h `salloc` wall revoked it at 13:42 local. It banked to **epoch 3749/7000** (last
  checkpoint on disk), val NLL still descending, no plateau. Ran an eval-only read of the
  3749 checkpoint (interactive salloc job 55454981; `--resume_from` + `--epochs 1` → empty
  train loop → pipeline's own 128-sample posterior-mean block; outputs in
  `..._uniongraph_EVAL3749`, kept run dir pristine). **Result (raw eigenvalues, posterior
  mean, vs SI Delaunay baseline):** λ₁ **0.8041** vs 0.7750 (**+0.029**), λ₂ 0.8461 vs 0.8105
  (+0.036), λ₃ 0.8955 vs 0.8912 (+0.004), mean 0.8486 vs 0.8256; best val NLL **0.8563** vs
  1.1065, test NLL 0.8964 vs 1.1392. **The union graph BEATS the fully-trained Delaunay
  baseline on all three λ at only 56% of training** — a floor, not the final number. **G3 is
  a GO.** (No cluster-slice Spearman in the pipeline results txt — a separate diagnostic if
  wanted.) NB: there is **no eval-from-checkpoint for the FlowJAX/NPE stack** — only
  `jraph_{regression,classification}_eval_from_checkpoint.py` exist (regression/clf models);
  the NPE eval is inline in `main()`, hence the resume+epochs=1 trick.
- Held resume sbatch **55441429** (`--resume`, 4×A100, 12 h) remains **JobHeldUser**. Recommend
  RELEASE to finish to 7000 for the final (≥0.804) gate number — held pending JDPK's call
  (he corrected an earlier premature release, so not auto-releasing).
- **Existing G4-PROPER plan AMENDED IN PLACE** (`docs/plan_g4_proper_equivariant_tensor.md`,
  per the sharpened equivariant-GNN review) — NOT a new doc. The doc already had the Tier A
  (eigenvalue-supervised, no tweb) / Tier B (tensor+eigenvector, needs tweb) split and the
  self-contained FFT tensor-build (§3: `tidal_tensor_fullgrid.py` building
  T_ij(k)=(k_ik_j/k²)δ_k, validated against cactus `eig_vals`; NO cactus edits) + frame
  rotation (§4). Amendments layered on:
  - **Attention promoted optional→REQUIRED** (§5) in every equivariant candidate (invariant
    logits preserve equivariance exactly; = adaptive smoothing kernel for the fixed 7 Mpc/h
    target) with pre-registered regularisation parity (attn dropout, wd/dropout 0.2/0.08,
    early-stop on val NLL, matched params) — the smoke overfit because undisciplined.
  - **Fixed bake-off order + SKIP list** (§5): SEGNN-with-attention first, SE(3)-Transformer/
    Equiformer-class second, one point-cloud model as the P1a control. SKIP EGNN/PaiNN/GVP
    (ℓ≤1), GATr (long-range wasted on compact target), MACE (many-body mismatch), PCA
    frame-averaging (degeneracy).
  - **P1 split into P1a/P1b** (§6): P1a = non-equivariant ~10 Mpc/h *radius-only* attentional
    GraphNet in existing JAX/jraph (no tweb, no equivariance) — isolates the Delaunay
    scale-mismatch lever; GO within seed noise of 0.774, ≥0.80 ⇒ deprioritise equivariance.
    P1b (Tier A steerable) must beat BOTH 0.774 AND the P1a control beyond seed noise (≥3–5
    seeds, matched compute/params). RPP-relaxed escape hatch (P0) if strict fails. P2 gate
    amended to calibration/coverage ≥ eigenvalue-regression flow.
  - **Reframe (§8):** the strongest first-principles lever is graph construction, not strict
    SO(3) — the wedge breaks the symmetry, so equivariance is a regulariser (modest gain at
    ~58k nodes / 3-param group, consistent with the +0.09 G1.5-ladder bound). Added
    permanent-shelf condition + plan-changing thresholds + PyTorch-e3nn-sidecar-first staging.
  - Correction: my first pass wrongly created a duplicate `workflows/sbi/G4_PROPER_PLAN.md`
    with a "save-the-Hessian-in-run_tweb_memory_optimized" rebuild — DELETED; the existing
    plan's standalone-FFT approach is the canonical one.

### 2026-06-16 — [code] graphify knowledge graph setup complete (Mac + NERSC)
- What: Installed graphify across Mac (`~/Developer/Illustris`, `~/Developer/GraphWeb_DESI`) and NERSC (`~/TNG/Illustris`, `~/GraphWeb_DESI`). Per-repo graphs built on both machines. Global cross-repo graph built at `~/.graphify/global-graph.json` (tags: `Illustris`, `GraphWeb_DESI`) on both machines. Claude Code PreToolUse hooks (grep/Read/Glob interception) and Cursor `.cursor/rules/graphify.mdc` updated in both repos to reference global graph for cross-repo queries. `graphify-out/` gitignored; `CLAUDE.md` + `.claude/settings.json` + `.cursor/rules/graphify.mdc` travel via git.
- Why: LLMs (Claude Code + Cursor) now consult a knowledge graph before grep/file reads, reducing token cost and surfacing cross-file dependency edges. Global graph captures the Illustris→GraphWeb_DESI model/module dependency that per-repo graphs can't see.
- Next: after any significant code change, run `graphify update .` in the repo then `graphify global add graphify-out/graph.json --as <tag>` to keep global current. Mac global uses `~/Developer/` paths; NERSC uses `~/TNG/Illustris/` paths.

The boundary object that keeps **science discussions** (Claude Desktop, local Mac)
and **agent work** (Claude Code, NERSC) aware of each other. It rides git, so it
reaches every machine.

- NERSC path: `~/TNG/Illustris/SCIENCE_LOG.md`
- Mac path:   `~/Developer/Illustris/SCIENCE_LOG.md`
- Syncs through GitHub (`jdpkololgi/Illustris`).

## How to use this file (both assistants)

1. **Read it at the start of a session** to load current direction before doing
   anything substantive.
2. **Append a short entry** (newest at the top of the Log) when a decision is
   made, a result lands, or direction changes. Don't rewrite history; add.
3. **It's only current after `git pull`.** Pull before reading/editing; commit +
   push after writing. (Stage just this file: `git add SCIENCE_LOG.md`.)
4. Keep entries terse and skimmable. Prune "Open threads" as items close.

Tags: `[science]` = decisions/hypotheses/conclusions from Desktop discussions.
`[code]` = what was run/changed/found, blockers, next actions from NERSC.

Entry shape:

```
### YYYY-MM-DD — [science|code] short title
- What: ...
- Why / decision: ...
- Next: ...
- Refs: files, commits, run dirs
```

## Open threads / current focus

- **NPE on Abacus wedge subvolumes** is the path forward for SBI. Graph-partitioned
  FlowJAX is legacy ("a nightmare") and to be retired when convenient.
- **GraphWeb production VAC** needs a real `sbatch` submit script — the DESI
  graph→features→inference chain currently only runs interactively.
- Jraph regression on wedge/cube caches is active and fine.

## Log (newest first)

### 2026-07-07 — [code] Mass-colour (M*-g/r) by-environment contours + wedge-count clarification
- What: `workflows/sbi_inference/plot_mstar_color_environment.py` (GraphWeb_DESI) makes
  `mstar_gr_by_environment` (2x2 nested KDE contours per inferred class, reproduces the
  reference fig) and `mstar_gr_overlay` (4 classes overlaid, class-coloured 50/90%).
- Result: mass-colour shows the environmental trend clearly where SFR-M* hexbins did NOT
  (env at fixed mass is 2nd-order → hexbin panels differ mainly in count). Void is bimodal
  (blue lobe + red lobe); blue lobe shrinks void→wall→filament→cluster to a single
  red-sequence peak. Median (g-r) 0.782/0.827/0.870/0.908; median logM* 10.60→10.71.
- COUNT CLARIFICATION (user asked): full DESI wedge = 111,503 galaxies (110,251 unique
  after 1,252 hemisphere dups; 111,171 with FastSpecFit photometry/sSFR). The "~22,000"
  is the thin 3° Dec slice (Dec 21-24° = 22,486) used by the FAN/skewer visualisations,
  NOT the whole wedge. z-shell is 0.20-0.30. Talk slide-5 updated to prefer the
  mass-colour contour fig + this count note.

### 2026-07-07 — [code] DR3-KP talk: SFR-M* (SFMS) figures by inferred environment + eigenvalue-continuous version
- What: New `workflows/sbi_inference/plot_sfms_environment.py` (GraphWeb_DESI) makes
  three SFR-vs-M* diagrams from the SI closure-join parquet (desi_wedge_env_props,
  N=77,459 with valid FastSpecFit sSFR): `sfms_all` (whole wedge, hexbin counts),
  `sfms_by_environment` (2x2 hexbin by inferred hard_class), `sfms_eigen_continuous`
  (1x3 hexbin coloured by mean posterior λ1/λ2/λ3 per cell). Blue-cloud/green-valley/
  red-sequence via MS fit to SF pop (logSFR=0.83 logM*-8.67) + offsets (MS-0.6/-1.2).
- Result: clean environmental-quenching signal — SF blob dominates Void, red sequence
  dominates Cluster (Wall/Filament intermediate). Eigenvalue version: red-sequence
  corner sits at higher mean λ (more collapsed tidal env), signal strengthening
  λ1→λ2→λ3. Figures in `figures/desi_wedge_flowjax_linear_si/`.
- Talk use: user DROPPED the mass-control slide; `sfms_by_environment` becomes slide 5
  hero (script §Slide 5 updated), eigen-continuous as its backup/alt. Also confirmed
  closure-slide text is correct (Loa=DR2; quenched=log sSFR<-11; g-r independent check).

### 2026-07-07 — [code] DESI DR3 KP (Cosmic Web & Galaxy Environment) 4-min talk: script v2 written, figures all existing
- What: Slide-by-slide script for the 4+2 min DR3 KP parallel-session talk written to
  `GraphWeb_DESI/docs/desi_dr3_cwge_4min_talk_script.md` (5 slides: title/map →
  why-posteriors → mock-trained+calibrated (class-fractions hero) →
  spotlight_cartography_closure composite hero → mass-control + take-home/VAC;
  + 8-item Q&A prep incl. fibre incompleteness, RSD, classical DTFE floor
  0.55/0.64/0.66 vs GNN 0.78/0.81/0.89, closure-circularity).
- Decision: NO new figures needed — the 2026-07-06 `desi2026_spotlight` set
  (esp. `spotlight_cartography_closure.png`) + SI-run + closure figures cover all
  slides; environment-fluent audience ⇒ closure/mass-control promoted to payload,
  SBI machinery compressed to one slide, calibration figures to backup.
- Refs: figures under `/pscratch/.../graphweb_desi/figures/{desi2026_spotlight,
  desi_wedge_flowjax_linear_si,desi_wedge_flowjax_linear_si_closure}/`; numbers
  reconciled to 2026-06-21/22 closure + calibration entries.
- v2 (same day): session schedule received — talk 5 of 7, fixed title "Learning
  the cosmic web: dynamical environment value added catalogues for DESI BGS",
  after ASTRA VAC / DisPerSE / MTV-reconstruction / void-science talks, NO
  per-talk Q&A (10-min consolidated + 40-min discussion). Script updated: zero
  web setup, positioning-vs-other-VACs framing, bridge lines to Li (MTV) and
  Rincon (voids), discussion seeds (common validation wedge; closure test as
  shared truth-free VAC metric; uncertainty propagation as KP standard).
- v3 addendum: deck review vs ASTRA slides (one idea/visual per slide; drop both
  videos) appended to the script; NEW fan ("pie-slice") figures built in the
  cmlamman polar style — `figures/desi2026_spotlight/fan/` on scratch:
  `fan_{desi_inferred,abacus_truth}_dec{2,3,4}` + recommended `_dec3_crop`
  variants (theta=RA 120–160°, r=z 0.20–0.30, Dec 21–24° slice, rorigin 0.12,
  class-styled points, script `make_fan_figure.py` + README). These replace the
  wedge video and the slide-30 sky map as the deck's hero pair.

### 2026-07-03 — [code] NAMING CORRECTION: the two "G4" runs are G4-SMOKE, not G4 (JDPK)
- The 2026-07-03 runs titled "G4 smoke" and "G4 part 2 (attention)" are hereby renamed
  **G4-SMOKE (graph+features held FIXED)**. They held the prebuilt Delaunay edge set AND
  the curated 7 node features fixed, and varied only invariant edge-message aggregation
  (mean 0.603 → attention 0.654 λ1 R² vs baseline 0.774). They test message/aggregation
  design on the EXISTING graph+features — NOT equivariance-as-architecture and NOT
  point-cloud input. The gate G4 as originally defined (symmetry axis) is STILL OPEN.
- **G4-PROPER** = the real test: positions + LOS only, model builds its own ~10 Mpc/h
  neighbourhoods, NO curated features, steerable messages → the tidal TENSOR (eigenvalues
  + eigenvectors fall out). Requires T-web module changes to emit tensor/eigenvector
  targets. Planned separately: `docs/plan_g4_proper_equivariant_tensor.md` (not executed).

### 2026-07-03 — [code] A3 COMPLETE: n(z)-harmonized wedge rebuilt end-to-end; nzharm SI cache ready
- What: full GPU/CPU rebuild of the harmonized training wedge via a BUFFERED box
  (RA 118–162, Dec 12.5–32.6, z 0.185–0.315) so boundary nodes keep real neighbours —
  mirroring subset-from-full semantics: (1) buffered shape-match dilution (C=0.733 core;
  148,222 of 176,349 kept; σ_v=35 km/s z-errors; sentinels excluded) → (2) gudhi
  Delaunay (CPU-guarded script — resource guard correctly refused a GPU alloc; pipeline
  split CPU→GPU→CPU) → (3) cuGraph 7 features (rapids) → (4) trim to exact wedge:
  **82,650 nodes, 605,126 undirected pairs (avg degree 14.6 ✓)**, alignment=identity,
  single hemisphere → (5) SI cache (splits 57,854/17,357/7,439) at
  `sbi_caches/path1_flowjax_3d_lineareig_si_nzharm/`.
- Parity verified: edge storage convention matches baseline (undirected pairs; cache
  doubles internally: 605,126→1,210,252 like 740,623→1,481,246); no self-loops/dupes.
- G4 record CORRECTED (JDPK probe): both smokes used the PREBUILT Delaunay edge_index +
  curated node h0 — raw geometry entered only as edge scalars. They test message/
  aggregation design on the same restricted graph, NOT "point cloud vs graph." The true
  point-cloud experiment (positions+LOS only, self-built ~10 Mpc/h neighbourhoods,
  attention EGNN, no curated features) is well-posed, ~½ day, NOT yet run — optional,
  sequenced behind the G3 readout. Tensor-target rule reaffirmed: eigenvalues for
  invariant nets (fixed-frame tensor INVALID there); tensor only with steerable
  (or LOS-frame spin-2-careful) heads.
- Phase B inputs now ready: nzharm cache (this) + union graph (G3 training, ~epoch 900+,
  sbatch auto-release armed) + FMPE head (G6 GO, calibration tune pending) + luminosity
  features (G2).
- New scripts: `build_harmonized_buffered_catalog.py`, `trim_wedge_from_buffered.py`,
  orchestrator `logs/run_a3_rebuild.sh` (CPU→GPU→CPU, tmux a3_rebuild).

### 2026-07-03 — [code] G4 part 2 (attention): helps (+0.05 λ1) but NO-GO stands; gap decomposed
- What: JDPK hunch tested — same EGNN-lite smoke with the SINGLE change mean→4-head
  attention (invariant logits from [h_src,h_dst,egeo]; segment softmax; symmetry story
  intact). λ1 R² 0.603→**0.654** (λ3 .791→.825; cluster Spearman .44) vs baseline
  0.774 — still below the pre-registered 0.75 bar.
- Attribution now clean: of the 0.17 part-1 gap, **aggregation ≈ +0.05**; remaining
  ~0.12 points at (a) regularization/training parity (attention overfit HARDER:
  train .073 vs val .253; baseline uses dropout .2 + wd .08 over 7000 epochs) and
  (b) curated EDGE features (baseline edge_attr includes density_contrast; pure-geometry
  egeo omits it) and/or the NLL/flow head's representation learning.
- Decision: **NO-GO confirmed for the full steerable build** — closing the rest of the
  gap = rebuilding the baseline's tuning, i.e. the very effort the gate protects.
  Standing lesson: any future equivariant build must be ATTENTIONAL
  (SE(3)-Transformer-class, not plain EGNN). Optional part 3 (dropout/wd parity) noted
  but not expected to flip the decision.
- G3: epoch 840, val NLL 3.47→2.50, healthy; auto-release watcher armed for the held
  sbatch handoff at window end.
- Refs: `gate_g4_egnn_smoke.py --aggregation attention`; log `logs/g4b_attn_*.log`.

### 2026-07-03 — [code] G6 GO (FMPE beats MAF), G4 smoke NO-GO (EGNN-lite < GraphNet), G3 training in tmux
- Parallel interactive wave (tmux sessions g3_union / g6_fmpe / g4_egnn; sbatch 55441429
  HELD as resumable fallback so no double-writer on the shared checkpoint dir).
- **G6 (FMPE vs MAF, frozen 80-d GNN embeddings, sbi 0.26.1, CPU): GO on accuracy.**
  Same conditioning, same train pool: λ1 R² 0.807 vs 0.772 (+0.035), λ2 .842/.829,
  λ3 .916/.898; cluster-slice λ1 Spearman +0.77 vs +0.61 (n=71). Converged in 144
  epochs on CPU — cheap swap. **Calibration caveat:** λ1 slightly overconfident
  (68% central coverage 0.63; 90% → 0.84; KS p .004; other dims fine) — needs a tune
  + a same-slice MAF rank comparison before production adoption in Phase B.
  (`gate_g6_fmpe_frozen_head.py`; embeddings cached at
  `sbi_runs/.../frozen_embeddings_all_nodes.npz`.)
- **G4 smoke (EGNN-lite, scalarized observer-rotation invariants of raw geometry +
  curated features, same Delaunay edges/splits, MSE head): NO-GO at the
  pre-registered bar** (λ1 R² 0.603 vs baseline 0.774; needed ≥0.75; λ2 .756/.810,
  λ3 .791/.891; cluster Spearman .45/.54) — despite the point-estimate head being
  R²-favoured. Raw geometry does NOT trivially beat curated features + attention;
  overfitting gap (train .15 vs val .29) says budget/regularization, not capacity.
  **Full steerable e3nn build DEPRIORITIZED** — revisit only if G3 or P2 motivates.
  (`gate_g4_egnn_smoke.py`; OOM on full-graph backward fixed via per-layer gradient
  checkpointing.)
- Conceptual note (logged for the papers): equivariance and attention are ORTHOGONAL
  axes of the Battaglia+2018 GN framework — equivariance constrains what φ_e/φ_v may
  compute; attention is a choice of aggregation weighting. EGNN = GN block with
  invariant-restricted φ_e, no attention; SE(3)-Transformer/Equiformer = steerable φ +
  invariant-logit attention. Attention is most motivated on the UNION graph (two edge
  populations to arbitrate).
- **G3:** training live in tmux (job 55442933, 4×A100). Union graph is ×2.7 edges →
  ~5.2 s/epoch → ~10 h total: this 4 h window banks ~2700 epochs to checkpoint; then
  RELEASE the held sbatch (`scontrol release 55441429`) to resume overnight.
- Phase B bundle shape so far: n(z)-harmonized mocks (A3) + luminosity features (G2)
  + FMPE head (G6, after calibration tune) + union graph pending G3 readout;
  equivariant encoder out (G4 smoke).
- Refs: `workflows/sbi/gate_g6_fmpe_frozen_head.py`, `gate_g4_egnn_smoke.py`,
  `run_flowjax_union_interactive.sh`; logs `/pscratch/.../logs/g{4,6}_*.log`.

### 2026-07-03 — [code] A3 designed+built, G3 training SUBMITTED (job 55441429), G6 deferred (no FMPE deps)
- **A3 harmonization (design + artifacts):** shape-match the mock wedge n(z) to DESI by
  per-shell dilution — keep fractions f_i = C·(DESI_i/mock_i), C = min ratio = 0.733;
  the residual amplitude offset becomes UNIFORM (scale-only), exactly what SI features
  absorb. Kept 82,693/100,935 (81.9%). z-errors injected (σ_v=35 km/s, median |dz|≈1e-4,
  original Z kept as Z_ORIG). Artifacts: `abacus/harmonized_20260703/path1_wedge_nzharm_
  {keepmask.npy, targets.fits, selection.json}` (`harmonize_wedge_nz.py`). Remaining A3
  step: rebuild Delaunay graph + cuGraph features on the kept points (GPU, rapids-gnn),
  then SI cache — commands recorded in selection.json.
- **G3 (union graph) fully launched:** built Delaunay ∪ radius(10 Mpc/h=14.78 Mpc)
  edges — 740,623 → 1,988,732 undirected pairs (×2.69), median edge 9.3→11.3 Mpc; new
  edge_attr in the original convention (`build_union_graph_arrays.py`; new npz+metadata,
  originals untouched). SI cache built with IDENTICAL splits to baseline (seed 42;
  70,654/21,196/9,085) at `sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph/`.
  Training sbatch SUBMITTED: **job 55441429** (4×A100, 12h, regular), output
  `sbi_runs/path1_wedge_flowjax_3d_linear_si_uniongraph`; compare vs the SI baseline on
  the shared test split (gate: λ1 R² / cluster metrics vs 0.774).
- **G6 (FMPE) deferred honestly:** cosmic_env has NO sbi/diffrax; flowjax 17.2.1 is
  discrete-flows-only. Plan: next session, new ISOLATED conda env with `sbi` (keeps
  cosmic_env pristine), train FMPE/NPSE heads on FROZEN GNN embeddings (one forward
  pass extracts them) vs the MAF baseline — head-only comparison, ~a day.
- Refs: scripts above; A2 n(z) JSON feeds harmonize_wedge_nz directly.

### 2026-07-03 — [code] Phase 0 gates EXECUTED: G1 GO (GNN≫GBM!), RSD penalty huge but GNN closes it, n(z) gradient, sentinel-z FIXED
- What: ran the roadmap-v2 first wave in parallel (G1, G1.5, G2, A1, A2). All new
  standalone scripts (`gate_g1_gnn_vs_gbm.py`, `gate_g15_g2_rsd_luminosity.py`,
  `measure_nz_mock_vs_desi.py` in GraphWeb_DESI/workflows/sbi_inference/); only edit =
  sentinel-z patch (backed up, new output filenames; originals untouched).
- **Physics verification first (JDPK request):** eigenvalue targets confirmed sampled at
  the REAL-SPACE host-halo x_com via (FILE_NUM,HALO_INDEX) — annotate_cutsky code
  explicitly avoids sky-coordinate inversion. So targets carry no RSD sampling offset;
  satellites inherit their HOST's λ (note for papers).
- **G1 (GNN vs GBM, identical features/splits): GO — and it FALSIFIES the 06-25/06-26
  "information-limited" claim.** GNN posterior-mean λ1 R²=0.774 vs GBM 0.272 (λ2 .810
  vs .334; λ3 .891 vs .326); cluster-slice Spearman 0.54 vs 0.04. Message passing does
  massive implicit field interpolation/de-distortion. (Transductive eval — matches
  deployment.)
- **G1.5 (RSD penalty, Z vs Z_COSMO on cutsky truth): +0.60** — real-space aperture
  features reach λ1 R²=0.86 vs 0.26 z-space (median RSD displacement 3.3 Mpc);
  mass-completeness 0.36→0.56. **The ladder: z-space GBM 0.26 → z-space GNN 0.77 →
  real-space bound 0.86.** The GNN already recovers most of the RSD-destroyed info;
  bounded headroom for G4 equivariant/LOS work ≈ +0.09 (motivated, tempered).
- **G2 (luminosity weighting): marginal GO (+0.036** z-space; +0.02 on real-space) —
  include flux-weighted features in Phase B, low cost.
- **A2 (n(z)): mock/DESI = 0.895 in the wedge and Z-DEPENDENT** — ratio slides
  1.02 (z≈0.20) → 0.73 (z≈0.275) → 0.89 (z≈0.30). A radial gradient a single per-graph
  SI median cannot absorb ⇒ per-shell harmonization (A3) confirmed as the right fix.
  Phantoms in box = 19.1%, excluded. Outputs: `abacus/nz_comparison_20260703/`.
- **A1 (sentinel-z): FIXED + REGENERATED + VALIDATED.** `_inject_chunk` now preserves
  ZWARN=999999 (never injects a pass onto fibre-unobserved rows); backup
  `inject_loa_spec_from_zall.py.bak.20260703`. New catalogs (originals untouched):
  `BGS_BRIGHT_full_noveto_loa_spec_sentinelfix.fits` and
  `/pscratch/.../abacus/mock_bgs_maglim_sentinelfix.fits` — 7,472,725 rows (was 9.54M),
  ZWARN==0 = 100%, sentinel-window frac 2e-5 (was 0.217), z-median 0.1966 (= audit's
  predicted clean value). z-expansion unblocked; A3 has its clean parent.
- Next: A3 per-shell harmonization (now fully unblocked); G3 union-graph and G6 FMPE
  (next sessions); G4 motivated with ~+0.09 tempered expectation; P1 letter drafting.
- Refs: gate scripts above; logs `/pscratch/.../logs/a1_sentinelfix_*.log`.

### 2026-07-03 — [code] Roadmap v2: decision-gated, parallelized; RSD-penalty gate added; problem stated exactly
- What: consolidated the whole brainstorm into **`docs/roadmap_environmental_vac.md` v2**
  (canonical; supersedes v1 and the 07-02 entry's sketch). Everything is now closed /
  gated / scheduled — no expensive step runs before its GO/NO-GO gate.
- Key sharpenings since v1:
  1. **Problem stated exactly**: z-space biased sparse tracers → REAL-space matter tidal
     field (targets from full particle field, comoving Cartesian, rs7; inputs from
     observed Z). The regression fuses interpolation + bias + statistical RSD
     de-distortion. Exact symmetry = **SO(3) about the observer**, realized by an
     E(3)-equivariant net fed per-node LOS r̂ᵢ (translations correctly broken). Frame
     consistency holds ⇒ tensor/eigenvector prediction well-posed (IA-ready).
  2. **New gate G1.5 — RSD-penalty decomposition** (mock has both Z and Z_COSMO):
     features from real-space vs z-space positions; the R² gap upper-bounds what ANY
     LOS-aware/equivariant machinery can recover, and splits the "information ceiling"
     into RSD vs sparsity+bias parts. Gates the equivariant rung together with G1.
  3. **Graph-construction critique formalized** (gate G3/b′): Delaunay's receptive
     field is fixed in neighbour count but variable in physical scale — narrowest in
     clusters — while the target lives at fixed 7 Mpc/h. Cheap fix under test:
     Delaunay ∪ radius(~10 Mpc/h) union graph, same GraphNet.
  4. Model zoo reduced to 2 axes (connectivity × symmetry); point-cloud models are
     GNNs that build their own graph (input = positions ± luminosity, nothing else).
  5. Timeline: P1 letter drafts now; G1/G1.5/G2 parallel CPU gates in wk 1–2 of July;
     ML4PS (P2) ~Aug 29 from the ablation; ONE bundled retrain end of Aug; JEPA/P5
     only if the post-retrain MMD gate fails.
- Refs: `docs/roadmap_environmental_vac.md` (v2, canonical — read this first on the
  Desktop side); task list mirrors the July gates.

### 2026-07-02 — [code] Method-frontier brainstorm + roadmap: triage plan, equivariant GNNs, papers
- What: strategy session (no production runs). Reviewed the whole diagnostic arc and
  set a forward plan. **Durable plan doc: `docs/roadmap_environmental_vac.md`** — point
  the Desktop side there; this entry is the summary.
- Architecture correction (propagate this — `~/.claude/CLAUDE.md` is NOT git-tracked so
  its fix won't sync): the NPE encoder is a jraph **Attentional GraphNetwork
  (Battaglia+2018)**, NOT GAT/GATv2. The GAT is the separate PyTorch classification
  lineage (the RASTI paper — JDPK is author; stop citing it as a third-party sibling).
- Method-frontier assessment (candidate upgrades, all gated): equivariant GNN encoder
  (EGNN/SEGNN fed rᵢⱼ + LOS r̂ᵢ, output the tidal *tensor* → eigenvalue ordering free,
  RSD anisotropy first-class); graph transformer (long-range = multi-scale); FMPE /
  flow-matching or Simformer posterior head (replace discrete MAF); luminosity weighting
  (claim-preserving, truth-testable on cutsky R_MAG_ABS); field-level reconstruction as
  the classical rival we complement, not beat.
- Key reframe (corrects the 06-26 "information-limited" claim): the measured ceiling is
  a property of the HAND-CRAFTED features, not proven for the raw point cloud. The
  representation ablation (below) is what settles "are we too constrained by
  distribution-only, hand-engineered features."
- Degraded-mock idea = supervised **domain randomization / nuisance-marginalized SBI**;
  related to but not the same as **JEPA** (self-supervised, dual-encoder, predicts
  masked embeddings). JEPA is the escalation if the misspecification gate fires.
- Decision structure = **Phase 0 Triage** (cheap GO/NO-GO, parallel), gating one bundled
  production retrain: (1) GNN-vs-GBM capacity check ½d; (2) luminosity-weighting gate
  ½d truth-tested; (3) representation ladder rungs c/d = equivariant GNN + sparse U-Net
  field-level, GPU, GO only if ≫ current GraphNet; (4) FMPE head swap. Phase A n(z)
  harmonization runs in parallel (fix sentinel-z bug first). Full detail + Phase B–D +
  publication ladder (Letter → Methods → VAC → ML4PS → conditional ICML/NeurIPS JEPA
  paper) in the roadmap doc.
- Next: start Phase 0 steps 1 & 2 (cheap CPU); step 1 doubles as rungs a–b of the ML4PS
  paper. Gate step 3 (GPU equivariant run) on step 1.
- Refs: `docs/roadmap_environmental_vac.md`; two roadmap flowcharts + a feature-type
  schematic produced in-session (not committed — regenerable).

### 2026-06-26 — [code] DEFINITIVE (mass-anchored): clusters favour FINE smoothing; current 7 Mpc/h good
- What: the "ideal test" — anchor 'cluster' to a SCALE-INDEPENDENT physical label
  (halo mass) instead of the smoothing-dependent λ1>λ_th. The master cutsky
  (`mocks_with_eigs_23032026/cutsky_..._with_tweb_eigs.fits`) carries **HALO_MASS**
  directly (units 1e10 Msun/h; also R_MAG_ABS, G_R_REST, CEN) and is row-aligned with
  the rs6–24 λ catalogs → no fragile CompaSO index join needed.
  (`mass_anchored_cluster_test.py`; the earlier CompaSO-join attempt was abandoned —
  path1 wedge_targets BOX_INDEX didn't reproduce host_halos x_com, and the master
  catalog has mass anyway.)
- Sanity passed: Spearman(logM, λ1@rs7)=+0.24; features(aperture density)→P(logM>13)
  AUC=0.736 (density genuinely predicts mass).
- Findings (identical across logM>12.5/13/13.5):
  1. **Massive-halo recovery decreases MONOTONICALLY with smoothing — finest is best, no
     interior optimum.** AUC(true λ1 → M>1e13): rs6 0.789 → rs7 0.770 → rs10 0.721 →
     rs20 0.637. Model recovery (aperture density → predicted λ1 → massive) same trend
     (rs6 0.722 → rs20 0.665).
  2. **Anti-correlation confirmed with a physical anchor:** mass-cluster recovery falls
     with smoothing while global R²(λ1) PEAKS at rs10. Optimising global accuracy
     actively harms real cluster recovery — the global "optimum" is a bulk artifact.
  3. **T-web cluster label is a real but imperfect mass finder:** purity (λ1>0.2 ∩
     M>1e13)/(λ1>0.2) ≈ 0.54 at rs7 (~0.61 at M>3e12) — about half of T-web clusters
     are genuine >1e13 halos; the rest are collapsed-environment outskirts. Massive halos
     ARE recoverable from geometry (AUC ~0.72), contra the earlier units-bugged run.
- Decision: **7 Mpc/h is a good cluster choice** (rs6 marginally better but noisier
  globally); do NOT raise the target smoothing for accuracy — it trades away clusters.
  Levers unchanged: soft posterior P(λ1>λ_th) + scale-matched ~10 Mpc/h *feature*
  aperture. Mass anchor removes the circularity in the smoothing-dependent cluster def.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/mass_anchored_cluster_test.py`,
  `plot_mass_anchored_recovery.py`; figure `tng_illustris/figures/smoothing_scale_study/
  mass_anchored_cluster_recovery.png`. Master cutsky has HALO_MASS/R_MAG_ABS/G_R_REST/CEN
  (useful for any future property-augmentation study, now testable on truth).

### 2026-06-26 — [code] CORRECTION: global-R² smoothing optimum ≠ cluster optimum; clusters favour ≤7 Mpc/h
- What: JDPK critique of the previous entry — clusters are compact, so larger smoothing
  washes them out; the global R²(λ1) peak at ~10 Mpc/h could be raising bulk accuracy
  while *erasing* clusters. Tested with CLUSTER-CONDITIONED metrics vs smoothing
  (`cluster_recovery_vs_smoothing.py`).
- Findings — critique CONFIRMED:
  1. **Global R²(λ1) and cluster completeness are anti-correlated.** Global R² peaks at
     rs10 (0.592); rank-based cluster completeness@true-rate peaks at the FINEST scale
     (rs6–7 ≈ 0.55) and falls through rs10 (0.518) → collapses by rs16 (0.17). rs7→rs10:
     global +0.045 but completeness −0.031.
  2. **AUC(cluster) is a trap for rare classes** — it *rises* with smoothing (0.90→0.98)
     because survivors become a trivially-separable extreme tail; completeness exposes
     the real (opposite) trend. Use completeness/fate, not AUC, for rare clusters.
  3. **Cross-scale fate:** of clusters defined at rs6, only 75% survive to rs7, **33% to
     rs10**, ~0% by rs16 — compact clusters dissolve fast under smoothing.
- Decision / correction: **RETRACT the "~10 Mpc/h optimum" as a target choice** — that is
  the bulk's optimum, not the clusters'. The current **7 Mpc/h is near-optimal for
  clusters** (was well-chosen). No single smoothing serves both bulk and clusters. The
  cluster levers remain: soft posterior P(λ1>λ_th) (shrinkage) + scale-matched ~10 Mpc/h
  *feature* aperture for the 7 Mpc/h *target* (feature scale ≠ target scale).
- Caveats: "cluster" defined via λ1(s)>λ_th (smoothing-dependent); a scale-independent
  anchor (halo mass via HALO_INDEX→CompaSO) would be the ideal further test.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/cluster_recovery_vs_smoothing.py`,
  `plot_smoothing_scale_study.py`; figure `tng_illustris/figures/smoothing_scale_study/
  smoothing_scale_learnability.png`.

### 2026-06-26 — [code] Target smoothing scale vs λ learnability: optimum ~10 Mpc/h; aperture≈1.4×scale
- What: JDPK follow-up — we have cutsky BGS eigenvalues at many smoothing scales (rs
  6–24 Mpc/h, same 63.9M galaxies, only target smoothing varies). Held galaxy features
  fixed (aperture density at 3–28 Mpc/h on a wedge footprint downsampled to BGS-like
  ~12.6 Mpc spacing) and measured how λ1/λ2 distribution + learnability depend on the
  target smoothing. (`smoothing_scale_investigation.py`, fitsio column reads.)
- Findings:
  1. **λ1 learnability is non-monotonic, peaks at ~10–11 Mpc/h** (geom R²(λ1): rs6 0.51,
     rs7 0.55, rs9 0.59, **rs10 0.592**, rs11 0.59, rs12 0.58, rs16 0.53, rs20 0.47).
     The current **7 Mpc/h is slightly below the learnability optimum**.
  2. **Scale-matching quantified:** λ1 at smoothing s is best predicted by density at
     aperture ≈ **1.3–1.5×s** (rs6→7, rs7→10, rs9–12→14, rs16–20→20). Design rule: for a
     target smoothed at s, build density features at ~1.4×s. Explains why few-Mpc Delaunay
     features under-serve the 7 Mpc/h target.
  3. **λ2 is the well-behaved eigenvalue** (R²≈0.64, peaks ~7 Mpc/h); web-classification
     difficulty is almost entirely in λ1.
  4. **Resolution↔learnability tradeoff is steep for clusters:** cluster frac (λ1>0.2)
     6.6%→3.2%→~0 over rs7→rs10→rs16. Can't smooth away the cluster problem without
     erasing clusters — smoothing scale and λ_th must be tuned jointly.
- Decision / implication: highest-value claim-preserving change = use **scale-matched
  ~10 Mpc/h aperture-density node features for the 7 Mpc/h target** (bigger lever than
  velocity dispersion, per the 06-26 entry). Smoothing scale + λ_th are a joint choice;
  ~10 Mpc/h is the λ1 sweet spot if fewer clusters are acceptable. No model change yet
  (investigation only); pre-veldisp default preserved.
- Caveats: aperture-density proxy (not full GNN; trends robust, absolute R² conservative);
  single footprint/realization; large-s decline partly aperture-capped at 28 Mpc/h.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/smoothing_scale_investigation.py`;
  cutsky catalogs `mocks_with_eigs_*_rsmooth_*` (rs6–24, _15d set).

### 2026-06-26 — [code] Cluster fix is FEATURE SCALE-MATCHING (+ soft posterior), NOT velocity dispersion
- What: followed up the 06-25 diagnosis by gating a proposed kinematic feature (local
  line-of-sight velocity dispersion / Fingers-of-God anisotropy) with cheap mock-truth
  pre-checks BEFORE any retrain. Arc:
  1. **Delaunay-scale veldisp — null.** σ∥/σ⊥ over Delaunay neighbours (~few Mpc) does
     NOT separate true cluster vs filament (AUC 0.507). Also showed the first signal test
     was circular (used the model's own geometry-derived `hard_class` as target).
  2. **Aperture-matched veldisp — modest.** JDPK insight: T-web targets are the tidal
     field Gaussian-smoothed at **7 Mpc/h**, so the kinematic scale must match. Sweeping
     fixed apertures, FoGAniso signal peaks at ~7–10 Mpc/h (cluster-vs-filament AUC
     0.51→0.61; on a balanced HistGBM proxy, cluster recall +0.02–0.04, cluster→filament
     −0.01–0.03). Real but small.
  3. **Eigenvalue-space reframe (JDPK) — the decisive view.** Classes are just λ_th=0.2
     thresholds on the regressed eigenvalues, so test λ1 directly. geom→λ1 R²=0.286;
     FoGAniso adds only +0.018; it does NOT explain geometry's λ1 residual (r≈0.03) and
     does NOT resolve λ1 in the threshold zone.
  4. **Why-the-limit investigation — the payoff.** (a) FEATURE-SCALE MISMATCH dominates:
     adding **aperture density at 7 & 10 Mpc/h** lifts λ1 R² 0.286→0.366 (**+0.080, 4×
     the kinematic gain**) — purely spatial, claim-preserving. (b) The "cluster deficit"
     is largely **regression shrinkage**: globally λ1 ranks fine (Spearman 0.56) but a
     point estimate predicts cluster frac 0.004 vs true 0.059 at λ_th=0.2 (**97% tail
     miss**) → the calibrated **posterior P(λ1>λ_th)** (NPE `p_exceed`) is the honest
     product. (c) Boundary is hard but not irreducible: feature-space λ1 noise floor
     0.74→0.66 and zone Spearman 0.13→0.17 with better-scaled features (earlier "R²<0"
     was a narrow-variance metric artifact).
- Why / decision: **did NOT add velocity dispersion** — gated off; the cheap pre-checks
  saved a ~9h retrain that would have bought ~+0.018 λ1 R² without fixing the boundary.
  Code reverted to pre-veldisp default (no production-pipeline file touched; the precheck
  scripts are standalone/opt-in only).
- Methodology lessons: (i) independence-from-geometry ≠ usefulness — test new features in
  the TARGET (eigenvalue) space, not on discrete labels; (ii) match FEATURE scale to the
  target smoothing scale (7 Mpc/h); (iii) hard-thresholding a shrunk point estimate
  manufactures the cluster deficit — report the posterior.
- Next (future, when requested): (a) scale-matched spatial features — aperture density at
  ~7–10 Mpc/h appended to the node set (biggest, claim-preserving λ1 lever); (b) report
  & calibrate soft P(λ1>λ_th) rather than hard argmax class fractions.
- Refs (standalone, opt-in): `GraphWeb_DESI/workflows/sbi_inference/`
  `velocity_dispersion_precheck.py`, `velocity_dispersion_aperture_precheck.py`,
  `velocity_dispersion_eigenvalue_precheck.py`, `threshold_limit_investigation.py`;
  plan shelved at `~/.claude/plans/keen-twirling-prism.md`.

### 2026-06-25 — [code] Cluster-deficit diagnosis rewritten: not coverage/FoG, it's spatial-only feature limit
- What: ran a diagnostic battery on the SI run (`path1_wedge_flowjax_3d_Bcorrected_linear_si`
  / `desi_wedge_flowjax_linear_si`) to localise the known cluster under-recovery
  (clusters→filaments) and the small DESI misspecification:
  1. **Posterior degeneracy** — none. DESI matches the in-distribution Abacus self-eval on
     all checks: collapse-to-marginal ratio 2.4–3.8 (≫1), no width/spike collapse, moderate
     inter-λ corr (genuine 3D posterior), embedding eff. rank ~5–6/80 (same both sides).
  2. **Summary-space MMD misspecification** (BayesFlow/Schmitt-style, GNN embedding = summary
     net) — DESI vs Abacus MMD²=+0.0043 vs in-dist floor ≈0 (split-half), p<0.01, 51σ, but
     MMD distance only 0.066 on standardised 80-d → DETECTABLE but small; downstream λ
     distributions still overlay. Don't chase it.
  3. **Training-support coverage in SI space** — CLEAN. frac DESI > Abacus train *max* ≈0 every
     feature; cluster-cands only ~0.7% past p99.9 in density family. **SI normalisation closed
     the extrapolation gap** the old non-SI check flagged → coverage is NOT the cluster cause.
  4. **Property-information ceiling** (geometry→FastSpecFit property, cross-val, closure
     parquet) — geometry encodes ~nothing about galaxy properties: R² LOGMSTAR 0.06 / g−r 0.03 /
     log_sSFR 0.003 / DN4000 ~0; quenched AUC 0.56. In model-cluster galaxies it's at chance
     (AUC 0.53). Properties are nearly orthogonal to the spatial features.
- Why / decision: combined with the earlier mild FoG/LOS result, this **rules out extrapolation
  and FoG** and **supports the intrinsic-information-limit hypothesis**: the closure test showed
  properties correlate with environment (survive mass control); (4) now shows that signal is
  independent of geometry. So positions-only features are demonstrably leaving environment-
  relevant information (galaxy properties) on the table, most so in clusters. Answers "are we
  over-limiting with spatial-only features?" → yes, now shown not asserted.
- Caveats: redundancy measured via the eigenvalue-optimised embedding (may understate what raw
  geometry could predict — raw-7-feature check pending); low R² includes property scatter, only
  the env-correlated part is usable headroom (closure sizes it); exploiting it needs properties
  painted onto Abacus mocks (HOD/SHAM → modelling systematics) and changes the claim to
  "positions + properties". MMD per-repeat "ratio" log line is a cosmetic /0 artifact.
- Next: decide on a deliberate property-augmentation study — paint a minimal property
  (luminosity / SHAM M*) onto the path1 mock, retrain, measure cluster-recall gain. Real GPU
  retrain, not a quick check. Optional: raw-7-feature redundancy robustness check.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{property_ceiling_ablation,desi_abacus_coverage_report,mmd_misspecification_check}.py`,
  `TNG/Illustris/workflows/sbi/degeneracy_check_abacus.py`; FoG: `plot_fog_los_alignment.py`.

### 2026-06-22 — [science] Correction: softplus, not linear, is best-calibrated; talk drops the comparison entirely
- What: Re-examined `Plots/three_way_tarp.png` directly (was working from a stale table note).
  Max|ECP−α|: softplus 0.01, linear 0.03, raw 0.08. Softplus is the best-calibrated
  parameterisation, not linear — corrects the v2.1 script's "linear increments …
  best-calibrated" line, which had the calibration ranking backwards (likely conflating it
  with the ordering-violation-rate column, where linear/raw do better than softplus). This
  matches the pre-existing SCIENCE_LOG/long-term framing ("softplus gives better TARP
  coverage than linear despite similar NLL") — it was only the script that drifted from it.
- Decision: production parameterisation is UNCHANGED (linear increments — chosen on
  NLL/R²/ordering-violation grounds, and it's what every downstream DESI result in the talk
  already uses). What changes is the TALK: the three-way parameterisation comparison is cut
  from the main flow entirely — no slide framed as a comparison between target increments,
  not even a one-line mention. Slide 8 ("Training results"/calibration headline) states only
  that the production model's posteriors are calibrated on the simulator (TARP+SBC), with no
  reference to alternative parameterisations. The 3-way comparison figure moves to backup/Q&A
  (a reviewer in this room may ask "why this parameterisation" — worth having on hand) but
  carries no main-deck claim now that the deck doesn't reference it.
- Also scoped down: the "diagnosing & closing the gap" material (graph-scale/edge-length
  diagnosis, per-graph normalisation fix, cluster-recovery sequence) is cut from the talk too
  — no time in a 12-min slot once background/motivation was expanded to 3 slides. Talk now
  goes straight from zero-shot transfer ("it works, aggregate near-truth") to the DESI closure
  test (validation = first science) to take-home. The gap-diagnosis work itself is unaffected
  (still real, still in `SCIENCE_LOG`/code), it's just not presented Thursday.
- Next: slides 8 and 14 rewritten as standalone bullet content (Dakshesh is speaking without a
  script, so slide text needs to carry the point on its own). Deck still has cleanup pending:
  slide 15 is an unrelated leftover from the original 33-slide purge (delete); slides 9 and 16
  are duplicate drafts of "Zero-shot to DESI" (merge); slide 14 (Summary) needs to move to the
  final position.
- Refs: `Plots/three_way_tarp.png`; `SBI-Galev-2026.key` slides 8, 14, 17;
  `cambridge_sbi_galev_2026_script.md` (needs v3 pass to match — not yet done).

### 2026-06-22 — [code] Closure/embedding figure fixes: UMAP class-colour bug + mass-control bins redrawn
- UMAP embedding plot (`GraphWeb_DESI/.../plot_desi_wedge_flowjax.py`): the DESI panel
  coloured classes from a SEPARATE `rng.choice` draw than the embedding subsample → labels
  misaligned with points → inferred classes looked scattered at random (while the PCA panel,
  which shares one index, was fine). Fixed to one shared index. Regenerated UMAP now shows the
  expected ordered cluster→filament→wall→void gradient — and the right mental model: the T-web
  classes are thresholds (λ_th=0.2) on a *continuous* tidal-field manifold, so a smooth
  gradient (not separated blobs) is the correct picture.
- Mass-control closure plot (`plot_property_environment_closure.py`): replaced quantile
  tertiles (boundary at 10.49 — sat ON the M*≈10.5 quenching transition, and the low bin
  spanned 8.1–10.49 → poor mass control) with FIXED bins [9.8,10.4,10.6,11.0,11.6]
  (`--mass-edges`). Edges bracket the transition (10.4/10.6, not 10.5); it's isolated in the
  narrow [10.4,10.6) bin; and the **below-transition bin [9.8,10.4) still rises** void→cluster
  (f_q 0.62→0.68; ≥850 clusters/bin) ⇒ intrinsic mass quenching alone cannot explain the trend
  ⇒ environmental quenching. Ordered plasma colours for the 4 mass lines; suptitle states it.
- Uncommitted (await go-ahead): both edited scripts (GraphWeb_DESI).
- Refs: SI run dir `embedding_umap.png`, `closure/closure_mass_control.png`; `FIGURE_GUIDE.md` updated.

### 2026-06-22 — [science] Talk script finalized v2.1: all figures locked, deck-build is the only remaining task
- What: Recovered the interrupted talk-prep thread and confirmed against `Plots/` that all
  6 previously-"STILL NEEDED" figures (slide 3 idealised skewer, slide 6 three-way TARP,
  slide 7 Abacus TARP+SBC, slide 9 edge_scale + cluster recovery bars) now exist, matching
  the NERSC 06-22 "6/6 complete" entries below. Updated
  `~/Developer/Working Files/SBI-Galev-2026/cambridge_sbi_galev_2026_script.md` to v2.1:
  every slide's Visual line now points at a FINAL figure filename instead of a placeholder;
  numbers reconciled to the actual figures (slide 6 max|ECP−α| softplus/linear/raw =
  0.012/0.029/0.082; slide 9 recovery sequence 0.027→0.034→0.040→0.046 vs truth 0.058,
  ~60% of gap closed — corrected from the earlier "56%/two-point" figure). All 4 OPEN
  DECISIONS and the STILL-NEEDED list marked resolved/closed in the script.
- Why / decision: with content locked, the only remaining work before Thursday is
  mechanical — building the actual Keynote deck (`SBI-Galev-2026.key`, currently still a
  33-slide copy of the classification-era FLATS talk) from the v2.1 script. Confirmed via
  keynote-mcp the file is open at 33 slides; identified reuse candidates by content audit:
  slide 1 (title layout), slide 3 "The Cosmic Web" (T-web explainer layout), slide 8
  "Combining graphs and machine learning" (pipeline-stage layout) — consistent with the
  06-15/06-21 plan to inherit theme + 3 layouts and rebuild everything else.
- Next: confirm rebuild plan with Dakshesh (strip/repurpose old slides vs keep-and-renumber),
  then build the 11 main + ~8 backup slides via keynote-mcp using the v2.1 script + `Plots/`
  images; convert the two HTML animations (`skewer_idealised.html`,
  `skewer_posterior_animation_real.html`) to mp4/gif for embedding (no live HTML in-talk);
  timer dry-run once built (OPEN DECISIONS #4, still pending).
- Refs: `cambridge_sbi_galev_2026_script.md` v2.1; `SBI-Galev-2026.key`; `Plots/` (all);
  see 06-21/06-22 [code] entries below for figure provenance.

### 2026-06-22 — [code] Dropped three-way TARP; SBC + TARP (with error band) for production linear only
- Decision (user): stop framing results as a parameterisation comparison. Removed
  `three_way_tarp.png` and the band-less copied `abacus_tarp_coverage.png` /
  `abacus_sbc_calibration.png` from the SI folder + canonical mirror.
- NEW `TNG/Illustris/workflows/sbi/plot_calibration_linear.py` (GPU; reuses
  plot_flowjax_posteriors' loaders): SBC + TARP for the **production linear-increment SI
  model**. Outputs `tarp_linear.png` — TARP with a **bootstrap 1σ error band** (N=3000 test
  points, max |ECP−α| = 0.015, i.e. at the MC noise floor √(0.25/N)≈0.009 → consistent with
  perfect calibration) — and `sbc_linear.png` — rank histograms for λ₁/λ₂/λ₃ with the
  expected ±2σ band (flat within band; mild λ₁-upper / λ₂-lower edge structure).
- WHY drop the comparison: the three-way TARP's softplus(0.012) < linear(0.029) ordering sat
  inside the 500-point MC noise floor (per-bin SE≈0.022), so it was NOT a real calibration
  ranking — and claiming "linear best-calibrated" would be a hostage to fortune. Linear's
  case rests on Pareto-dominance (NLL +1.13 vs +2.25, R² 0.82 vs 0.74, ordering-viol 0.8%),
  not "tightest TARP". (Supersedes/corrects the 2026-06-19 "linear tightest TARP" note.)
- `FIGURE_GUIDE.md` updated to the new pair; `plot_tarp_threeway.py` now unused.
- Uncommitted (await go-ahead): + `plot_calibration_linear.py` (Illustris).
- Refs: SI run dir `tarp_linear.png`, `sbc_linear.png`.

### 2026-06-22 — [code] Skewer HTML → GIF for Keynote (NERSC ffmpeg has no h264)
- `render_skewer_video.py` NEW (`GraphWeb_DESI/workflows/sbi_inference/`): extracts the
  embedded JSON payload from a skewer HTML (`var D={…}`) and re-renders the same 3 panels
  (galaxy strip / three λ posteriors + λ_th / class-prob bar) natively with matplotlib +
  shared.plot_style → mp4 or gif. No browser/headless-Chromium needed.
- WHY gif not mp4: the NERSC system ffmpeg (`/usr/bin/ffmpeg`) is built WITHOUT libx264
  (encoders present: only gif, libvpx-vp9; h264/mpeg4 decoders disabled), and
  imageio-ffmpeg isn't installed — so no H.264. Keynote plays + loops GIFs natively, so
  produced GIFs: `skewer_idealised.gif` (3.0 MB), `skewer_real.gif` (0.97 MB) in the SI
  folder + canonical. The renderer auto-falls-back mp4→gif on encode failure.
- For a true H.264 mp4: convert the gif on the Mac (its ffmpeg has libx264):
  `ffmpeg -i x.gif -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" x.mp4`.
- Also fixed `sync_figures_to_canonical.sh` to include *.gif/*.mp4 (were excluded → the
  gifs weren't reaching the canonical dir).
- Uncommitted (await go-ahead): + `render_skewer_video.py` + sync-script edit (GraphWeb).
- Refs: SI run dir `skewer_{idealised,real}.gif`.

### 2026-06-22 — [code] Idealised skewer (slide 3) + per-figure interpretation guide; STILL NEEDED 6/6
- Slide 3 — `skewer_idealised.html` NEW (`GraphWeb_DESI/workflows/sbi_inference/build_skewer_idealised.py`):
  the discrete→continuous bookend. IMPORTS the real skewer's HTML template so layout + the
  standardised style (true-black / IBM Plex Sans / COSMIC_WEB_COLORS / λ_th=0.2) are identical
  — fixes the gap that the Desktop mockup had — but is driven by CLEAN synthetic structure:
  one density peak the sightline crosses, λ₃/λ₂/λ₁ cross λ_th at d≈0.33/0.60/0.86 so the class
  bar morphs void→wall→filament→cluster→…→void (peak P(cluster)=0.78). Slide 10 = the same
  layout on real posteriors. **STILL-NEEDED list now complete (6/6).**
- `FIGURE_GUIDE.md` written into the SI run dir: per-figure interpretation + error-bar
  provenance + variable definitions, esp. closure — inferred env = hard_class (argmax of the
  λ>λ_th class probs) / trace_lambda (E[Σλ] ∝ density); properties = LOGMSTAR, SFR,
  log sSFR=log10(SFR)−LOGMSTAR, quenched≡log sSFR<−11, g−r=ABSMAG01_SDSS_G−R from loa
  FastSpecFit; CIs = Wilson (fractions) / bootstrap 16-84 (medians) / TARP bootstrap bands /
  SBC rank uniformity.
- Uncommitted (await go-ahead): + `build_skewer_idealised.py` (GraphWeb).
- Refs: SI run dir `skewer_idealised.html`, `FIGURE_GUIDE.md` (+ canonical mirror).

### 2026-06-22 — [code] Slide 6 three-way TARP done; label fixes; "STILL NEEDED" now 5/6
- Slide 6 — `three_way_tarp.png` NEW (`TNG/Illustris/workflows/sbi/plot_tarp_threeway.py`,
  GPU): recomputes TARP coverage for the 3 parameterisation models on their own test
  splits and overlays them (bootstrap 1σ bands), reusing plot_flowjax_posteriors' eval
  machinery. Result = the slide-6 message exactly: max|ECP−α| **softplus 0.012, linear
  0.029, raw 0.082** — raw eigenvalues win NLL but are clearly OVER-confident (curve bows
  above the diagonal); softplus/linear hug it. (One JAX gotcha: flowjax needs new-style
  typed keys — `jax.random.key`, not `PRNGKey`.) Written to the SI folder + synced.
- Fixes from user review of yesterday's batch: (a) class-fractions legend now consistent
  "[domain] NPE" → `Abacus NPE` / `DESI NPE` (was "NPE DESI"); (b) recovery-bars 4th stage
  relabelled `Per-graph norm. (production)` (dropped "SI" — the fix is per-graph
  domain-relative normalisation, per the talk-v2 correction, not "scale-invariant").
- STILL NEEDED now just (d) slide-3 idealised fake-data skewer (generative; matches the
  real-skewer panel layout or reuse the 06-18 Desktop mockup) — user's call.
- Uncommitted (await go-ahead): `plot_tarp_threeway.py` (Illustris);
  `plot_desi_wedge_flowjax.py` + `plot_cluster_recovery_bars.py` (GraphWeb_DESI).
- Refs: SI run dir figures + canonical `figures/desi_wedge_flowjax_linear_si/`.

### 2026-06-21 — [code] Slide-figure batch: 4 of 6 "STILL NEEDED" plots done into the SI folder
- What: Actioned the STILL-NEEDED list from the plot→slide entry below. All new/updated
  figures are themed (`shared.plot_style`) and live in the SI run dir
  `/pscratch/.../flowjax_inference_outputs/desi_wedge_flowjax_linear_si/` (also synced to
  canonical `figures/desi_wedge_flowjax_linear_si/`). DONE:
  * Slide 8 — `class_fractions_comparison.png` REGENERATED without the "Regression DESI"
    bars (added `--no-regression` + `--only-class-fractions` to `plot_desi_wedge_flowjax.py`;
    generic bar spacing). Now 3 shades: Abacus truth / Abacus NPE / NPE DESI. Cluster group
    reads 0.06 / 0.05 / 0.05 → on-message "transfer lands near truth incl. rare clusters".
  * Slide 9 — `cluster_recovery_bars.png` NEW (`plot_cluster_recovery_bars.py`): reads the
    four DESI variant summaries live → cluster fraction 0.027→0.034→0.040→0.046 vs Abacus
    truth 0.058, annotated "60% of the gap closed". Gold bars, dashed truth line.
  * Slide 9 — `edge_scale.png` copied in from the baseline linear run (graph-level, identical
    across runs; already themed).
  * Slide 7 — Abacus calibration copied in as `abacus_tarp_coverage.png` +
    `abacus_sbc_calibration.png` (the themed 06-21 SI-run plots; provenance-renamed).
- NOT done (need heavier work, flagged for user decision):
  * Slide 6 three-way TARP (softplus/linear/raw overlay): the per-run npz saves only class
    probs (no posterior samples / no true θ), and `plot_tarp_coverage` computes ecp/alpha
    inline without saving them → a single overlay needs a GPU re-eval of all 3 models.
    User pre-authorised the table fallback; GPU overlay on request.
  * Slide 3 idealised fake-data skewer: generative; `build_skewer_animation.py` is the
    real-data version (panel layout = galaxy strip + width ribbon + class strip + 3 λ
    posterior KDEs + class-prob bar). A synthetic twin can be built to match, or reuse the
    06-18 Desktop mockup — user's call.
- Code: `plot_desi_wedge_flowjax.py` (2 new flags) + new `plot_cluster_recovery_bars.py`
  are UNCOMMITTED in GraphWeb_DESI pending user go-ahead.
- Refs: SI run dir figures; `GraphWeb_DESI/workflows/sbi_inference/{plot_desi_wedge_flowjax,
  plot_cluster_recovery_bars}.py`; this batch maps to the plot→slide entry directly below.

### 2026-06-21 — [science] Plot→slide mapping decided; graph-construction question closed
- What: Audited the conference folder `~/Developer/Working Files/SBI-Galev-2026/`
  (deck .key + 17 figures/animations) and assigned heroes per slide; full mapping
  appended to the script (§PLOT → SLIDE MAPPING v2.1). Curated, not exhaustive
  (we deliberately do not use every plot).
- Decisions:
  * Slide 8 hero = `class_fractions_comparison.png`. KEY REALISATION: this is the
    SI-corrected production run — NPE-DESI cluster fraction is already 0.05 (truth
    0.06), all four classes land near truth. So the transfer result and the cluster
    recovery are the SAME figure; slide 8 = "transfer works incl. rare clusters,"
    and slide 9 becomes the under-the-hood "this didn't come for free" methods
    slide. Reinforces the demotion of cluster-fraction from headline to support.
    ACTION: regenerate this plot WITHOUT the "Regression DESI" bars (off-message
    for an SBI room; clutters the legend) — keep 3 shades (Abacus truth/Abacus
    NPE/NPE DESI).
  * Slide 10 = `closure/closure_categorical.png` (hero: f_quench/colour/sSFR rise
    void→cluster) + `closure_mass_control.png` (survives at fixed M*) + the real
    skewer animation. Strongest part of the deck: validation = first science.
  * Slide 10 animation: `skewer_posterior_animation_real.html` → record to mp4/gif
    and embed in Keynote (no live HTML in a 12-min talk). Bookend to slide 3.
  * Backup/Q&A only: eigenvalue_corner, closure_continuous_{trace,p_cluster},
    posterior_width_sky_map, class_sky_map, width_vs_boundary (counterintuitive:
    width tracks tail extremity not class ambiguity — needs 30s, keep off main deck),
    embeddings (umap/pca/3d), lambda_th_sweep, posterior_width_3d.
- STILL NEEDED (not in folder — sync from NERSC canonical or generate): (a) **Slide 7
  TARP coverage plot + SBC rank histogram (Abacus)** — CRITICAL, the calibration
  headline currently has no figure; sync from the SI run via
  `scripts/sync_figures_to_canonical.sh`. (b) Slide 6 three-way TARP curves
  (softplus/linear/raw) to make "raw wins NLL, fails calibration" visual (table
  alone is acceptable fallback). (c) Slide 9 `edge_scale.png` + cluster-fraction
  recovery-sequence bars (else slide 9 is text-only). (d) Slide 3 idealised
  fake-data skewer — generate or reuse the 06-18 Desktop mockup; must match the
  real-skewer panel layout for the bookend.
- CLOSED: graph construction is **Delaunay on BOTH** the Abacus training wedge and
  the DESI inference wedge (user-confirmed; alpha-complex was tried earlier and
  worked worse, abandoned). => NO train/inference graph mismatch; removed that as a
  slide-9 open question. The 2026-06-21 v2 entry's open-item (1) is hereby resolved.
  Pipeline slide states "Delaunay" cleanly; slide-9 gap is purely the n(z)/edge-scale
  story.
- Closure test: confirmed done and strong per the 2026-06-21 [code] entry (99.7%
  TARGETID→FastSpecFit join; every trend monotonic void→cluster; survives mass
  control). Skewer animations exist; user may iterate.
- Deck file: still a copy of the old FLATS talk inside the SBI-Galev-2026 folder;
  it's the working base for building the new slides.
- Next: sync the four missing-figure sets from NERSC; generate the idealised skewer;
  regen class-fractions without the regression bars; then build slides on the FLATS
  copy. Timer dry-run after.
- Refs: `~/Developer/Working Files/SBI-Galev-2026/` (deck + figures);
  `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (§PLOT→SLIDE v2.1);
  2026-06-21 [code] closure-test entry; 2026-06-19/06-20 [code] diagnostic/SI runs.

### 2026-06-21 — [code] Property–environment closure test runs on DESI LOA wedge (slide 10)
- What: Built the TARGETID→FastSpecFit join + closure-test plots for slide 10's
  truth-free validation (validation item (4) from 2026-06-18). Two new scripts in
  `GraphWeb_DESI/workflows/sbi_inference/`:
  `build_desi_wedge_property_join.py` (recovers TARGETID from preds
  `global_node_id` → source BGS catalog row index; de-dups the 1,252
  hemisphere-split duplicate targets by averaging their posteriors; joins loa
  FastSpecFit SPECPHOT by TARGETID → LOGMSTAR/SFR/Dn4000/ABSMAG g,r; writes
  `desi_wedge_env_props.parquet` + `join_report.json`) and
  `plot_property_environment_closure.py` (themed categorical + continuous +
  mass-control figures).
- Join quality: loa-vs-loa (same release — no Y1/Y3 mismatch), FastSpecFit dir
  `/global/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs/` (already the
  `config_paths.DESI_FASTSPEC_CATALOGS_DIR` default). **99.7% coverage**
  (111,171/111,503 unique targets); RA/Dec cross-check 0.027″ → join verified.
  Used the SI production run preds (`desi_wedge_flowjax_linear_si`).
- RESULT — every known trend recovered, monotonic void→wall→filament→cluster:
  f_quench 0.743→0.777→0.815→0.835; median (g−r)₀.₁ 0.782→0.827→0.870→0.908;
  median log sSFR −11.51→−11.63→−11.77→−11.96. Continuous E[Σλ] (tidal-tensor
  trace ∝ density): Spearman ρ = +0.076 (quench), +0.138 (g−r), −0.090 (sSFR),
  all correct sign at N=111k. **Mass control: trend SURVIVES at fixed M*** —
  within every logM* tertile f_quench rises void→cluster (high-mass tertile
  0.855→0.882→0.906→0.924), so it is not merely the mass–environment relation
  (median logM* itself rises only mildly, 10.60→10.71). Overall f_quench 0.78
  (massive BGS-bright sample; the TREND is the result).
- Caveats (carry to slide): inferred environment derives from galaxy positions
  so a density→quenching trend is partly expected — the point is the inferred
  T-web environment carries real physical signal AND survives mass control; loa
  fibre incompleteness is mildly env-dependent (biases absolute fractions, not
  the trend); quenched ≡ log sSFR < −11 (the colour panel is the independent
  colour-cut view).
- Figures (deck theme) synced to canonical
  `/pscratch/.../graphweb_desi/figures/desi_wedge_flowjax_linear_si_closure/`:
  `closure_categorical.png`, `closure_continuous_{trace,p_cluster}.png`,
  `closure_mass_control.png`. Slide 10 now has a real figure, not text-only.
- Next: drop the categorical (headline) + mass_control (defensible) figures into
  slide 10; optionally rerun on the baseline `desi_wedge_flowjax_linear` preds to
  confirm the trend is model-robust. The build script is `--run`/`--preds`-flagged
  so any variant rejoins in ~1 min.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{build_desi_wedge_property_join,
  plot_property_environment_closure}.py`; run dir
  `flowjax_inference_outputs/desi_wedge_flowjax_linear_si/`. See
  [[project-path1-wedge-npe-run]], 2026-06-18 validation-plan entry.

### 2026-06-21 — [science] Cambridge talk v2: narrative spine + factual corrections
- What: Reworked the talk around a single throughline question ("what cosmic-web
  environment is each galaxy in DESI, and how sure are we?") asked slide 1,
  answered slide 11, with a SKEWER BOOKEND (idealised fake-data skewer early as
  the discrete→continuous intuition pump; real DESI posterior skewer animation
  late as the payoff). Continuous-over-discrete justified with references: λ_th
  is arbitrary (Hahn 2007 used 0, Forero-Romero 2009 used 0.44, chosen for visual
  agreement); magnitude is the physics (tidal-torque theory couples spin to the
  tidal TENSOR — Codis 2015, Peebles 1969/Doroshkevich 1970); hard labels carry
  no uncertainty; and a continuous posterior SUBSUMES the discrete picture
  (threshold posterior samples → calibrated class probs at any λ_th with error
  bars). Back half reframed: the sim→real calibration-survival story is the
  differentiator for this SBI room, and the cluster-deficit / falsification work
  is demoted from headline to ONE supporting-evidence slide ("transfer is
  trustworthy and diagnosable"). Parameterisation slide reframed from a 3-way
  horse race to the methods-culture point "lower NLL ≠ a better posterior — why
  we calibration-test" (raw eigenvalues win NLL but fail TARP). Script v2 written
  to `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (supersedes v1).
- Corrections logged (these override earlier notes/memory):
  1. CORAL is NOT implemented (confirmed against 06-19/06-20 [code] entries — the
     adopted fix is per-graph feature normalisation, not CORAL). CORAL removed
     from the pipeline slide; appears only as a one-line future-work item.
  2. The Abacus wedge graph is built with DELAUNAY triangulation, NOT alpha
     complex. (Alpha complex was the recommendation for REAL survey data with
     gaps; the training wedge uses Delaunay.) Pipeline slide + diagram corrected.
  3. The slide-9 fix is "per-graph (domain-relative) feature normalisation"
     (normalise scale-carrying features relative to each graph's own mean so the
     GNN sees RELATIVE features and the mock-vs-DESI offset cancels) — NOT
     "scale-invariant features." Reworded everywhere to avoid confusing a physics
     audience and to describe what was actually done. (NB the 2026-06-20 [code]
     entry's `--scale-invariant-features` flag implements exactly this per-graph
     mean-relative normalisation; the talk wording is corrected, the code name
     is unchanged.)
- Emphasis added: n(z) differs mock-vs-DESI AND varies ACROSS the wedge, so the
  graph-scale shift is spatially varying — which is the physical reason a single
  global rescale fails and a per-graph relative normalisation is the right tool.
- Resolves the merge-conflict flagged earlier today: falsification walkthrough
  goes to BACKUP (aligns with the NERSC-side 12+3 entry's "appendix-only"
  conclusion), but for a story reason (supporting evidence, not a result), not
  merely time.
- References verified by web search (drop-in list in the script's §REFERENCES):
  Dressler 1980, Peng 2010, Kraljić 2018, Codis 2015, Peebles 1969/Doroshkevich
  1970; Hahn 2007, Forero-Romero 2009; Papamakarios & Murray 2016, Greenberg 2019,
  Cranmer/Brehmer/Louppe 2020, Papamakarios 2021; Talts 2018 (SBC), Lemos 2023 (TARP).
- Open items for next pass: (1) CONFIRM DESI-side graph construction — if it's
  alpha-complex while training is Delaunay, that train/inference graph mismatch is
  a second on-story contributor to the slide-9 domain gap and should be surfaced;
  (2) skewer animation + closure-test plot readiness by Thu drive slide 10's final
  form; (3) timer dry-run — slides 3/5/8/9 are the ~80–90s long ones, slide 9 the
  likely cut.
- Refs: `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (v2);
  2026-06-19 (DESI transfer + cluster-deficit diagnosis), 2026-06-20 (per-graph
  normalisation fix) [code] entries; conference site sbi-galev.github.io/2026.

### 2026-06-21 — [science] Cambridge talk: narrative finalized, format confirmed, deck strategy decided
- What: Format confirmed as **12-min talk + 3-min Q&A** (resolves the
  12+3 vs 15+5 open question from 2026-06-15). Finalized an 11-slide
  narrative built around the post-2026-06-15 results: brief T-web/cosmic-web
  motivation (1 slide) → classifier→posterior pivot with RASTI 2025 as
  single backstory slide → pipeline diagram (alpha-complex graph + GNN
  encoder + flow) → target-parameterisation result (linear increments
  Pareto-dominate softplus/raw — headline design choice) → TARP calibration
  on Abacus (credibility slide) → zero-shot DESI transfer result (class
  fractions land near truth, ordering-violation rate 0.8%→4.7% as the
  domain-shift fingerprint) → falsification walkthrough (ruled out
  under-density/FoG/extrapolation/shape, landed on graph-level edge-scale
  shift) → scale-invariant-feature fix (56% of cluster gap closed, zero
  in-domain cost, honest redistribution caveat) → truth-free validation +
  outlook (closure tests, skewer animation if ready, CORAL as next lever)
  → take-home. Full script with per-slide timing and visual notes written
  to `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md`.
- Why / decision: audience (SBI-GalEv 2026, Cambridge) is SBI-fluent but not
  T-web/DESI-fluent, so intro stays to ~3 sentences on cosmic-web physics
  and zero time explaining NPE/flows. Decided to sell the back half (slides
  7–9) as the differentiator for this specific room — a fully worked,
  falsifiable sim-to-real domain-shift diagnosis on an amortised NPE — since
  the conference's own theory stream lists domain adaptation, transfer
  learning, and calibration/convergence testing verbatim. Classification
  accuracy demoted to backstory only, consistent with the 2026-06-15
  framing decision but now made concrete in slide-by-slide form.
- Decision on deck file: do NOT edit `FLATs.key` in place. It's the
  classification-era (RASTI 2025) poster talk — visual theme (true-black,
  IBM Plex Sans, COSMIC_WEB_COLORS, FLATS accent palette) and several
  layout templates (title slide, T-web explainer, pipeline-stage slide) are
  worth inheriting, but essentially all result slides are from the retired
  4-class/partitioned-FlowJAX path and need rebuilding, not reuse. Plan:
  duplicate `FLATs.key` to a new file, strip outdated result slides, rebuild
  from the script above. Noted blocker: `FLATs.key` is 199MB and timed out
  via keynote-mcp AppleScript (`open_presentation`) — likely has heavy
  embedded assets from earlier iterations; worth auditing/trimming embeds
  before the new file becomes the working copy, to keep keynote-mcp
  reliable for the rest of the week.
- Next: (1) confirm whether the skewer animation (open thread since
  2026-06-18) lands before Thursday — it's the strongest single visual if
  ready, otherwise slide 10 stays closure-test text only; (2) duplicate
  FLATs.key → new working file, audit embedded asset size; (3) build slides
  from the script, regenerating figures already produced by the NERSC
  pipeline (TARP, class-fraction bars, edge_scale.png, lambda_th_sweep.png,
  fog_los_alignment.png) into the finalized PLOT_STYLE_GUIDE theme; (4)
  dry-run with a timer — slide 8 (falsification walkthrough) is the densest
  for its 90s budget and is the most likely cut candidate.
- **CONFLICTS WITH NEXT ENTRY:** this plan puts the falsification
  walkthrough on main-deck slide 8; the next entry (independent NERSC-side
  session) concludes it should be appendix-only. Unresolved — settle before
  building that slide.
- Refs: `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md`,
  `FLATs.key`, conference site (sbi-galev.github.io/2026), 2026-06-15 talk
  framing entry, 2026-06-18 validation plan entry, 2026-06-19/20 wedge-NPE
  + cluster-suppression + scale-invariant-fix entries.

### 2026-06-21 — [science] Cambridge SBI-GalEv talk slot confirmed: 12 min + 3 Q&A
- What: Organisers confirmed the talk slot is 12 minutes + 3 minutes for questions
  (resolves the open "12+3 vs 15+5" question from the 2026-06-15 framing entry).
- Why / decision: Tighter than the 15+5 option assumed in earlier figure planning —
  trim to the minimum figure set that carries the SBI-first narrative (calibration
  headline, DESI transfer result, SI fix) rather than including secondary
  diagnostics (FoG/shape/training-coverage falsification plots are appendix-only,
  not main-deck).
- Refs: FLATS deck; see 2026-06-15 "Cambridge SBI-GalEv talk" entry.

### 2026-06-21 — [code] Plot visual identity + centralised figure output fixed
- What: `workflows/sbi/plot_flowjax_posteriors.py` (Abacus-side TARP/calibration/
  posterior/class-fraction plots) had zero theming — plain matplotlib defaults,
  inconsistent with the GraphWeb_DESI true-black/IBM Plex Sans theme used on all
  DESI-side figures. Wired in `shared.plot_style.apply_style()` +
  `ACCENT_COLORS`/`COSMIC_WEB_COLORS` (the module was already mirrored into
  `TNG/Illustris/shared/`, just never imported by this script). Also added
  `scripts/sync_figures_to_canonical.sh` in both repos: rsyncs a run's finished
  figures into the existing-but-underused `CANONICAL_FIGURE_ROOT` /
  `GRAPHWEB_CANONICAL_FIGURE_DIR` (`/pscratch/.../{tng_illustris,graphweb_desi}/figures/`)
  under a run-named subfolder, so figures are centralised for picking per-conference
  without disturbing per-run experiment directories.
- Why / decision: per-run output dirs are right for active experimentation (compare
  testA/testB/Bcorrected/SI side by side) but meant the canonical figure roots,
  despite already being the documented default in `plot_flowjax_posteriors.py`,
  were being overridden by every launcher and sat empty/stale in practice.
- Next: regenerate `path1_wedge_flowjax_3d_Bcorrected_linear_si/plots/` with the
  fixed theme and sync both Abacus- and DESI-side SI figures into the canonical
  dirs (in progress).
- Refs: `shared/plot_style.py`, `scripts/sync_figures_to_canonical.sh` (both repos),
  `workflows/sbi/plot_flowjax_posteriors.py`.

### 2026-06-20 — [code] Route A: scale-invariant features fix — best cluster recovery (56%), no in-domain cost
- What: Implemented `--scale-invariant-features` in `build_abacus_sbi_cache.py` +
  `GraphWeb_DESI` inference (`abacus_gnn_parity.py`, `infer_desi_wedge_flowjax.py`):
  per-graph-median normalise the scale-carrying node (Degree/Density/NeighDensity/
  I_eig) and edge (edge_length) features into dimensionless **contrasts** before the
  box-cox/log, leaving density_contrast/directions/Clustering untouched. Rebuilt the
  linear cache (SI), retrained the linear NPE (4×A100), re-ran DESI inference with
  matched SI normalisation.
- Result (DESI cluster fraction, λ_th=0.2; truth 0.058):
  baseline 0.027 → edge-domain-adapt 0.034 (22%) → full node+edge adapt 0.040 (40%)
  → **SI retrain 0.046 (56% of the gap closed — best)**. Abacus-side UNCHANGED
  (Test NLL 1.14, posterior-mean R² 0.826, ordering-viol ~0) → the scale-invariance
  has ~zero in-domain cost, confirming the earlier prediction. Abacus TARP is the
  TIGHTEST of all runs (hugs the diagonal, dev <0.02) and SBC clean — scale-invariance
  preserved (slightly improved) calibration. So SI is the unambiguous production model:
  best Abacus calibration + best DESI transfer, no downside.
- Caveat (honest): not a clean across-the-board win — SI recovers clusters by
  REDISTRIBUTING: void 0.254→0.226 (truth 0.27, worse) and filament 0.273→0.295
  (truth 0.26, worse); wall 0.447→0.433 (toward 0.41, better). Total deviation from
  truth ~flat; it trades a little void/filament accuracy for much better cluster+wall.
  Favourable for a study where clusters are the rare science-critical class.
- Takeaway: scale-invariant features are the recommended DURABLE fix for the
  graph-scale/N(z) transfer shift (learned, not a post-hoc hack). Closes ~56% of the
  cluster deficit; the remaining ~44% needs Route B (mock densification at high z —
  the mock under-produces high-z galaxies; `--equal_data_dens` downsamples the wrong
  way). For Cambridge: present SI as the principled fix with the void/filament caveat.
- Refs: cache `path1_flowjax_3d_lineareig_si`; run `path1_wedge_flowjax_3d_Bcorrected_linear_si`;
  DESI `desi_wedge_flowjax_linear_si`; commits Illustris build-cache + GraphWeb parity/infer.

### 2026-06-19 — [code] DESI cluster-suppression diagnosis: not under-density, not FoG; a training-coverage shift
- What: Characterised why the wedge NPE (and the Jraph regression) under-predict
  **clusters** on real DESI (cluster fraction ~0.027 vs Abacus truth ~0.058 at
  λ_th=0.2). Built a diagnostic suite under
  `/pscratch/.../flowjax_inference_outputs/desi_wedge_flowjax_linear/`:
  eigenvalue corner (clean distributions) + environment corner; λ_th sweep
  (static `lambda_th_sweep.png` + animation `lambda_th_sweep_animation.html`);
  FoG magnitude test (`fog_anisotropy.png`) and the proper LOS-direction FoG test
  (`fog_los_alignment.png`). New scripts in `GraphWeb_DESI/workflows/sbi_inference/`.
- Findings (what the deficit is NOT):
  1. **Not under-density.** Real DESI cluster cores are *denser* than the mock
     (Density p99.9 0.18 vs 0.10; ~2× more galaxies above the Abacus 99th pct).
     Rules out fibre-incompleteness erasing cluster density.
  2. **Not Fingers-of-God.** Recomputed local inertia *eigenvectors*: DESI major
     axes are NOT more line-of-sight aligned than Abacus (median |cosθ| 0.421 vs
     0.456; both slightly transverse from wedge geometry). The earlier anisotropy
     *magnitude* excess (esp. void/wall-inferred) is transverse/generic, not RSD
     elongation. So FoG is ruled out as the differential cause.
  3. **Not training-coverage / extrapolation.** Only ~5% of DESI dense
     cluster-candidates (Density feature) exceed the Abacus train p99.9, ~0.01% of
     all DESI galaxies exceed the train max — the bulk is in-distribution
     (`training_coverage.png`). Too small to explain a halved cluster fraction.
- Findings (what it IS): a genuine **high-λ tail suppression** — persists at every
  threshold and *worsens* with λ_th (DESI/Abacus cluster ratio 0.65→0.45→0.30 at
  λ_th 0.1/0.2/0.3). The richest single cluster IS recovered (skewer P(cluster)=0.86
  in a thin 0.6° beam), so the model isn't blind to clusters — the bulk of moderate
  clusters degrade.
  4. **Not dense-structure shape either.** Oaxaca decomposition of the dense
     inferred-cluster-rate gap (Abacus 0.156 vs DESI 0.082, `shape_misclassification.py`):
     only **8%** is explained by DESI being more elongated; **92% is residual** —
     at MATCHED density AND matched anisotropy, DESI dense galaxies are classified
     as cluster ~half as often as Abacus ones. So the earlier shape hypothesis is
     refuted too.
- Bottom line: **every marginal node-feature explanation is falsified** (density,
  FoG, extrapolation, shape). The deficit is a *fixed-density-fixed-shape residual*
  → a joint / graph-level domain shift in the GNN encoder's feature→λ map (shared by
  regression+NPE, so it lives in the encoder, not the flow). Honest open question,
  not solved. Most concrete remaining lever to test: the **N(z) / number-density
  mismatch** (DESI ~12% more galaxies here, different N(z)) changes the graph EDGE
  SCALE (typical edge lengths) — which would shift GNN output even at fixed node
  features and is consistent with a graph-level, not node-level, residual.
- Edge-scale test (`edge_scale.png`) CONFIRMS a graph-level shift: DESI graph is 12%
  denser, edges 7% shorter, scaled log(edge_length) sits −0.11σ off the Abacus N(0,1)
  — the GNN sees a different graph scale even at fixed node features. Modest but the
  only lever that survives falsification.
- **Phase-0 cheap fix (inference-time domain correction, no retrain):** re-standardising
  DESI scaled EDGE features to training N(0,1) (`--edge-domain-adapt`) recovers ~22% of
  the cluster gap (0.027→0.034) and moves ALL classes toward truth — clean. Adding
  NODE-feature re-standardisation (`--node-domain-adapt`) reaches ~40% (cluster→0.040)
  but over-corrects (void 0.21<0.27, filament 0.30>0.26). So ~40% of the deficit is
  correctable input-domain shift; ~60% is a deeper feature→λ / connectivity residual.
  For the talk: use the clean edge-adapt; durable fix = scale-invariant edge features +
  retrain (Route A), graceful vs the blunt post-hoc rescale.
- Implication: expanding RA/Dec won't fix it. Levers, in order of evidence: (a) match
  the mock **N(z)/number density** to real DESI so the graph scale matches — the
  mock HAS the knobs (`upstream_mkCat_SecondGen_amtl.py --equal_data_dens y`,
  `upstream_prepare_mocks_Y3_bright.py --downsampling y`), both DEFAULT OFF in path1,
  which is why N(z) currently mismatches; (b) full **domain adaptation** (align joint
  feature/graph distribution, not marginals); (c) revisit the **HOD**
  (cluster richness/morphology); (d) pragmatic λ_th=0.1 (0.065 vs truth 0.10). Frame
  for Cambridge as a robust, RSD-NOT cluster systematic *localized* (not
  under-density/FoG/shape) to a graph-scale transfer effect — honest open problem.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{plot_eigenvalue_corner,plot_lambda_th_sweep,plot_fog_anisotropy*,plot_fog_los_alignment}.py`;
  diagnostic figures in the linear DESI run dir. See [[project-path1-sentinel-z-bug]].

### 2026-06-19 — [code] DESI transfer: linear NPE runs on real DESI LOA wedge (conference key result)
- What: Ran the linear-increment FlowJAX NPE on the real DESI LOA wedge (112,755
  bright BGS galaxies, same RA120–160/Dec14.5–30.6/z0.2–0.3 footprint). New
  scripts in `GraphWeb_DESI/workflows/sbi_inference/`:
  `infer_desi_wedge_flowjax.py` (GPU forward) + `plot_desi_wedge_flowjax.py`
  (themed figures). Mirrors the regression inference's preprocessing (node box-cox
  from cache `node_feature_scaler`; edge bidirectional+log+StandardScaler via
  `shared/abacus_gnn_parity.py`) but swaps the model for the GNN encoder + flow
  (128 posterior samples/galaxy → λ posteriors → class probs). Reuses
  `load_flowjax_model`/`create_gnn_and_flow`/`batched_sample_posterior`/
  `samples_to_raw_eigenvalues`/`posterior_to_classprobs`.
- Result: **transfer works, no collapse.** DESI NPE class fractions
  {void .254, wall .447, fil .273, clu .027} sit right between Abacus truth
  {.27/.41/.26/.06} and regression DESI {.246/.462/.266/.026} — wall-dominated,
  cluster-rare. Cluster .027 matches regression (.026), both below Abacus truth
  (.06) = the documented DESI cluster-tail transfer gap.
- NPE-specific findings the regression can't give: (a) per-galaxy posterior WIDTH
  sky map shows coherent structure tracking the cosmic web + survey-footprint
  gaps; (b) width rises near the survey edge (graph-truncation) AND rises with
  eigenvalue extremity (tail distance), NOT with class-boundary ambiguity;
  (c) ordering-violation rate is **4.7% on DESI vs 0.8% on Abacus** — a clean
  quantification of the domain shift (flow less certain on out-of-distribution
  real data; count-based class fracs stay robust). Fixable later with a post-hoc
  sort of posterior samples.
- CRITICAL parity guard: the edge scaler MUST be fit on the **path1 fiberassign**
  wedge npz (the model's training graph), NOT the regression wedge the old script
  uses — both have ~100k nodes so only a constants-assert (mean≈[2.15,0], scale≈
  [0.70,1.77]) catches a mix-up. Hard-asserted in the script.
- Outputs: `/pscratch/.../graphweb_desi/flowjax_inference_outputs/desi_wedge_flowjax_linear/`
  (preds npz + summary.json + 5 themed figures). TARP/SBC stay Abacus-side (no DESI
  truth). Stretch (post-conf): property–environment closure (needs TARGETID→
  fastspecfit join), skewer animation. See [[project-path1-wedge-npe-run]].
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{infer,plot}_desi_wedge_flowjax.py`,
  reused DESI wedge `desi_wedge_expanded_..._from_fullgraph/`.

### 2026-06-19 — [code] Wedge NPE parameterisation study: linear increments win; mock ruled out
- What: Ran the 3-d FlowJAX NPE on the path1 fiberassign wedge (z 0.2–0.3,
  100,935 nodes) under all three eigenvalue parameterisations, 4×A100, 7000
  epochs, seed 42, identical arch/reg/split. Result (Test NLL / posterior-mean
  R² / ordering-violation rate / TARP):
  | param | NLL | R² | viol. | TARP |
  |---|---|---|---|---|
  | softplus increments | +2.25 | 0.74 | 0% | tight |
  | **linear increments** | **+1.13** | **0.82** | **0.8%** | **tightest** |
  | raw eigenvalues | −0.07 | 0.85 | 2.5% | over-confident |
  Softplus's +2 NLL floor was the inverse-softplus heavy tail (init NLL 215 vs
  ~8–10 for linear/raw). Raw gets the lowest NLL but is *over-sharpened*
  (over-confident TARP, 2.5% out-of-order samples). **Linear increments
  Pareto-dominate softplus** (better NLL, R², calibration; negligible 0.8%
  violations) and are the best-calibrated overall — and match the 15-d wedge
  regression's own parameterisation. **Headline choice = linear** for the
  Cambridge talk (calibration-first venue). Hyperparameter tuning (dropout 0→0,
  wd 0.08→0.01) made no difference — not the lever.
- Mock audit cross-ref (separate session "Path1 mock generation audit"): found
  the spectro-injection resurrects fibre-unobserved targets as sentinel z≈0.59
  (2.08M rows, 21.8% of mock_bgs_maglim). **Our wedge is 100% clean** (Z∈[0.200,
  0.300], 0.00% at 0.59 — verified). So mock generation is **ruled out** as the
  NLL-floor cause; the floor is the wedge's genuine information content (sparse
  fiber-incomplete DESI-like selection). The sentinel flaw is a **z-expansion
  blocker**: expand in RA/Dec (or fix the injection first), never in z, until
  the injection is corrected.
- Code (uncommitted→committed this entry): added `--linear-increments` to
  `build_abacus_sbi_cache.py`; `eigenvalues_to_linear_increments` /
  `linear_increments_to_eigenvalues` / `resolve_increment_mode` and a 3-mode
  `samples_to_raw_eigenvalues` in `shared/eigenvalue_transformations.py`;
  `--increment_mode {softplus,linear,raw}` threaded through
  `jraph_sbi_flowjax.py` + `resolve_sbi_paths` (suffix `_linear_eig`) + the saved
  model + `plot_flowjax_posteriors.py`; vectorised the eval (chunked vmap, TARP +
  SBC + class-prob); checkpoint/resume.
- Next: (1) regenerate TARP + class-fraction figures from the linear run in the
  plot-style-guide theme for the deck; (2) LOCK the wedge — do not expand before
  the talk (multi-day rebuild, not needed; calibration is the headline);
  (3) post-conf: fix sentinel injection, then RA/Dec wedge expansion for tighter
  constraints. See [[project-path1-wedge-npe-run]], [[project-path1-sentinel-z-bug]].
- Refs: runs `path1_wedge_flowjax_3d_{testA_reg,testB_raweig,Bcorrected_linear}`
  under `/pscratch/.../abacus/sbi_runs/`; caches `path1_flowjax_3d{,_raweig,_lineareig}`.

### 2026-06-18 — [science] Posterior validation plan beyond TARP: class probabilities, skewer check, boundary degradation, closure tests
- What: TARP only certifies statistical self-consistency, not physical
  correctness on DESI where no per-galaxy truth exists. Defined four concrete,
  implementable validation steps for wedge-NPE q(λ|X) output:
  (1) **Class probabilities from samples.** Per galaxy, draw S posterior samples
  in increment space, invert to physical (λ₁,λ₂,λ₃) (ordering guaranteed by
  construction), threshold at λ_th, count crossings n=Σ1[λ_k>λ_th] (n∈{0,1,2,3}
  by ordering): P(void)=mean(n=0), P(wall)=mean(n=1), P(filament)=mean(n=2),
  P(cluster)=mean(n=3). Cross-check analytically via marginal CDFs: P(λ_k>λ_th)
  from each marginal. **NB ordering is ASCENDING λ₁≤λ₂≤λ₃ (CACTUS-native: eig1
  smallest, eig3 largest — verified end-to-end, reproduces CWEB exactly at
  λ_th=0.2), so the decomposition is P(void)=1-P(λ₃>λ_th),
  P(wall)=P(λ₃>λ_th)-P(λ₂>λ_th), P(filament)=P(λ₂>λ_th)-P(λ₁>λ_th),
  P(cluster)=P(λ₁>λ_th)** — the *smallest* eigenvalue λ₁ exceeding λ_th ⇒ cluster
  (all exceed). (An earlier draft of this entry flipped λ₁↔λ₃, matching a
  descending convention the codebase does not use.) Sample-based and analytic
  versions must agree to MC error — mismatch indicates an inversion or sampling
  bug. **λ_th must equal the CWEB-labeling threshold (Abacus = 0.2, the CACTUS
  default), not 0.0.** Implemented in
  `shared/eigenvalue_transformations.py::posterior_to_classprobs`.
  (2) **Reliability diagram for class probabilities** (discrete-output
  complement to TARP): on Abacus (CACTUS ground truth available), bin galaxies
  by predicted P(filament) (and other classes), plot empirical true-class
  fraction per bin vs predicted probability. Diagonal = calibrated; deviation
  flags over/under-confidence, expected to concentrate at wall/filament
  boundary per RASTI confusion matrix.
  (3) **Posterior-width vs distance-to-boundary.** Compute per-galaxy posterior
  width (e.g. mean marginal σ across λ₁,λ₂,λ₃, or posterior entropy) vs distance
  to wedge/survey edge (RA/Dec/z footprint boundary, alpha-complex hemisphere
  split) and vs distance to T-web class boundary (λ near λ_th). Expect
  monotonic widening near both. On Abacus (truth known) also plot width vs
  true prediction error to confirm widening tracks real degradation, not just
  graph truncation artifacts; confirm DESI shows the same *shape* of
  width-vs-edge-distance curve as Abacus.
  (4) **Property–environment closure test (DESI-only, no truth needed).** Bin
  real DESI BGS galaxies by inferred environment (E[trace(λ)] or dominant class
  probability) and check recovery of known relations: quenched fraction /
  colour / sSFR rising toward filament→cluster, morphology–density trend.
  Recovering established astrophysical trends from inferred-only environment is
  evidence the posteriors carry real physical information.
  Also scoped (5) cross-check against independent structure finder (DisPerSE
  filaments, or BORG T-web per existing on-the-horizon validation target):
  expect bulk agreement, divergence concentrated at class boundaries as the
  reassuring signature, not a failure mode.
  (6) **Animated line-of-sight skewer plot.** Pick a 1D skewer through a wedge
  (Abacus or DESI Loa) crossing void→wall→filament→cluster. At each position
  along the skewer: render the three eigenvalue marginal posterior densities
  (KDE over per-position-bin galaxy samples, not Gaussian — Gaussian only used
  for the Desktop mockup) with λ_th marked, a posterior-width ribbon, and the
  class-probability bar from (1)/(2)'s class-prob math. Animate by sweeping
  position; render as a saved video/gif, not just an interactive widget, so it
  drops into the talk deck. On Abacus, overlay the true CACTUS (λ₁,λ₂,λ₃) as
  moving markers on the posterior panel — credible bands containing truth
  through the transition, widening exactly where truth nears λ_th, is the
  core "physically sensible" evidence this whole validation plan is for.
- Why / decision: visualizing 3 correlated posteriors per galaxy across an
  entire wedge does not scale; need scalar/derived functionals overlaid on
  geometry instead of raw posterior dumps, and need validation that is
  meaningful on DESI specifically (no per-galaxy truth there), not just on
  Abacus. (1)+(2) give a calibration check at the class level; (3) gives a
  boundary-physics sense-check transferable Abacus→DESI; (4) gives a truth-free
  closure test directly on real BGS data. (6) is the figure that actually
  conveys physicality to a human audience — the others are diagnostic plots,
  this is the one that shows the posteriors tracking real structure.
- Next: implement (1) as a `posterior_to_classprobs(samples, lambda_th)`
  utility in `shared/eigenvalue_transformations.py` (sample-count path +
  analytic-CDF path, with an assert/warning if they diverge beyond MC
  tolerance); implement (2) reliability diagram + (3) width-vs-boundary-distance
  + (6) skewer animation in `workflows/visualization/` using
  `shared/plot_style.py` (COSMIC_WEB_COLORS for class bins, matplotlib
  FuncAnimation or equivalent for (6)); run (3) and (6) on one Abacus wedge
  first (truth overlay available), then the matching DESI Loa wedge, compare;
  (4) needs BGS property table joined to wedge-NPE output by target ID — check
  join keys exist in current DESI Loa wedge cache before scripting. (5)
  BORG/DisPerSE cross-check stays "on the horizon" pending BORG catalogue
  access — not in this sprint.
- Refs: `shared/eigenvalue_transformations.py` (ordered softplus increment
  policy — class-prob math must operate in this space then invert), CACTUS
  labels (Abacus truth), DESI BGS LSS catalogues at
  `/global/cfs/cdirs/desi` (property–environment closure), RASTI confusion
  matrix (wall/filament boundary expectation), `PLOT_STYLE_GUIDE.md`. Desktop
  mockup of (6) (Gaussian-approximation, interactive, not the real KDE version)
  built this session for layout reference.

### 2026-06-17 — [science] Plot style guide finalized: true-black theme, cosmic-web class colors, IBM Plex Sans
- What: Defined canonical matplotlib style for all GraphWeb_DESI plots: true-black
  (#000000) background, off-white (#F2F2F2) text/ticks; confirmed FLATS-deck
  cosmic-web class colors (Void #A1FCDD, Wall #4E84F7, Filament #EB336F, Cluster
  #F5C144) as a categorical-only palette, kept separate from the FLATS general
  accent colors (magenta #FF006E, blue #3A86FF, red #D62828); replaced Eurostile
  with IBM Plex Sans as the single font everywhere (Eurostile has no usable Greek
  glyphs, which matters for λ-notation); mathtext.fontset='dejavusans' handles
  λ-subscripts in labels regardless of body font. Wrote PLOT_STYLE_GUIDE.md +
  shared/plot_style.py (apply_style(), finalize_axes(), COSMIC_WEB_COLORS,
  CLASS_ORDER, ACCENT_COLORS) in GraphWeb_DESI.
- Why / decision: Every plot in the pipeline (regression diagnostics, NPE
  posteriors, TARP coverage, DESI BGS results) needs one consistent visual
  identity before Cambridge SBI-GalEv talk figures are regenerated from
  wedge-NPE output; relying on Eurostile directly would have broken λ-notation
  rendering.
- Next: Download IBM Plex Sans .ttf files into GraphWeb_DESI/assets/fonts/
  (command in PLOT_STYLE_GUIDE.md §2.3) on both Mac and NERSC; migrate existing
  plotting scripts/notebooks (workflows/visualization/*) to
  `from shared.plot_style import apply_style, finalize_axes`; consider mirroring
  plot_style.py into Illustris if TNG-side plots need the same theme.
- Refs: GraphWeb_DESI/PLOT_STYLE_GUIDE.md, GraphWeb_DESI/shared/plot_style.py,
  FLATS deck slide 9 (legend swatches).

### 2026-06-16 — [code] graphify global graph + cross-repo Cursor rules
- What: Merged `Illustris` and `GraphWeb_DESI` into `~/.graphify/global-graph.json`;
  updated `.cursor/rules/graphify.mdc` in both repos to name the sibling repo and
  require `--graph ~/.graphify/global-graph.json` for cross-repo queries.
- Why / decision: Per-repo `graphify query` and grep do not surface Illustris↔DESI
  dependencies; global graph does (e.g. `graph_net_models` → Jraph wedge inference).
- Next: After substantive code changes, `graphify update .` then
  `graphify global add graphify-out/graph.json --as <tag>`.
- Refs: `Illustris/.cursor/rules/graphify.mdc`, `GraphWeb_DESI/.cursor/rules/graphify.mdc`

### 2026-06-15 — [code] Environment policy clarified for agents
- What: Updated agent/runbook guidance: activate `cosmic_env` for normal work and
  `rapids-gnn` whenever calculating graph metrics/features.
- Why / decision: Phase 4 validation on an unprepared Cloud image failed on
  missing scientific dependencies; graph metrics also require the RAPIDS/cuGraph
  stack rather than the default environment.
- Next: Keep workflow launchers aligned with this policy as new graph-metric or
  wedge-NPE scripts are added.
- Refs: `CLAUDE.md`, `RUNBOOK.md`.

### 2026-06-15 — [code] Docs aligned to wedge-NPE Abacus path
- What: Updated top-level and workflow docs so Abacus-scale SBI points to
  RA/Dec/z wedge subvolume caches with `jraph_sbi_flowjax.py`, while partitioned
  FlowJAX is marked legacy/reference.
- Why / decision: Several public docs still described partitioned Abacus FlowJAX
  as active, contradicting the current wedge-subvolume SBI direction.
- Next: Add a production `sbatch` launcher for wedge NPE when the interactive
  recipe stabilizes.
- Refs: `README.md`, `ACTIVE_WORKFLOWS.md`, `RUNBOOK.md`,
  `workflows/abacus_tweb/README.md`, `workflows/sbi/README.md`.

### 2026-06-15 — [science] Target representation: ordered increments are canonical, not "raw eigenvalues"
- What: Cleared a doc ambiguity that conflated two independent axes. (1) Target
  *quantity*: eigenvalues beat shape-param (I₁,e,p) and invariant (I₁,I₂,I₃)
  reps, which are pathological for ML. (2) Target *parameterisation*: eigenvalues
  are trained as ordered softplus increments (λ₁ anchor + two non-negative
  increments), enforcing λ₁≤λ₂≤λ₃ by construction. The net trains/emits in
  increment space; inversion to (λ₁,λ₂,λ₃) is presentation/eval-only.
- Why / decision: "raw eigenvalues preferred" (true only on axis 1) could lead an
  agent to collapse the softplus head to a direct 3-output regressor and silently
  reintroduce ordering violations. Canonical policy now documented in code + docs.
- Next: retire the shape-param path properly — remove `--use_shape_params` and the
  dead `compute_shape_param_statistics` (builds `stats`, returns `None`) from
  `jraph_pipeline.py` + `eigenvalue_transformations.py` once confirmed unused.
- Refs: `shared/eigenvalue_transformations.py` (new module docstring + dep
  banners), `CLAUDE.md` (Physics Targets, module table, regression cmd).

### 2026-06-15 — [science] Cambridge SBI-GalEv talk: framing + reusable assets
- What: Talk narrative = open on environment→galaxy-evolution motivation, then
  frame the whole pipeline as amortised per-galaxy posterior inference of T-Web
  eigenvalues with TARP calibration as the headline. Regression = validated
  encoder/forward-model backbone; NPE + flows = frontier result. Reviewed prior
  poster (classification-era) and FLATS deck: deck already has FlowJAX
  posterior-comparison + TARP-coverage figures (from the retired partitioned
  path) — reuse as visual template, regenerate from wedge-NPE.
- Why / decision: venue is SBI-first, so calibration/posterior must lead, not
  cartography; self-diagnosed weak spot is astro motivation, so first ~3 min
  carry the science case. Slot Thu 25 Jun ~11:30, 15-min block (confirm 12+3 vs
  15+5 with organisers).
- Next: (1) NPE on one Abacus wedge + TARP plot = critical path; (2)
  R²-vs-smoothing-scale plot; (3) polish 3-way λ hists + class-fraction bar + 3D
  wedge render in deck palette (Eurostile, #FF006E/#3A86FF). Stretch: ξ(r_p,π)
  forward-model check, 2nd wedge for cosmic-variance scatter. Methods-slide
  architecture diagram drafted (draw.io).
- Refs: FLATS deck; `workflows/jraph/`, wedge caches; visualization notebook.

### 2026-06-14 — [code] Cross-tool infra + this log
- What: Set up `~/.claude/CLAUDE.md` as cross-repo canon (3-location map, conda
  envs, proven Perlmutter `salloc`/`srun` recipes; production = `sbatch`). Fixed
  the `nersc` skill's `qos.md` (no `gpu_`-prefixed QOS). Added two Claude Code
  skills: `desi-bgs-graph` and `jraph-eval`. Created this SCIENCE_LOG.
- Next: stand up the Desktop⇄NERSC bridge (filesystem MCP + git) so science
  decisions land here automatically.
- Refs: `~/.claude/CLAUDE.md`, `~/.claude/skills/{desi-bgs-graph,jraph-eval}`.

### 2026-06-14 — [science] SBI direction: wedges, not partitions
- What: Abacus-scale inference works via **wedge subvolumes** (graph per RA/Dec/z
  wedge); the graph-partitioning / partitioned-FlowJAX path is abandoned.
- Why / decision: graph partitioning was a nightmare; wedge training has worked
  over the last ~2 months. Future = NPE on the wedge subvolumes.
- Next: when convenient, retire `workflows/sbi/*partitioned*` + partition builders.
- Refs: `workflows/sbi/`, see also Claude Code memory `project-abacus-wedge-npe-direction`.
