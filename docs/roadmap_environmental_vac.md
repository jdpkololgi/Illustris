# Roadmap — DESI BGS cosmic-web environment VAC

Durable plan for turning the wedge-NPE method into a released DESI data product.
Living doc; the authoritative running narrative is `SCIENCE_LOG.md`. Written
2026-07-02 (Claude Code brainstorm with JDPK).

## Where we are

Amortized NPE: a jraph **Attentional GraphNetwork** (Battaglia+2018 — *not* GAT/GATv2;
the GAT is the separate PyTorch classification lineage / RASTI paper) encodes a
Delaunay graph of galaxy positions (7 geometric node features) into 80-d embeddings
that condition a **FlowJAX** flow over the tidal eigenvalues (λ1,λ2,λ3; 7 Mpc/h
smoothing; T-web classes = λ_th=0.2 threshold). Trained on Abacus path1 fiberassign
mocks with scale-invariant features; applied to a DESI BGS wedge. Calibrated
(SBC+TARP); property→environment closure on DESI validated truth-free.

Diagnosed this session: posteriors non-degenerate + transfer-stable; misspecification
small but real (summary-space MMD); coverage clean; **cluster deficit is largely
regression shrinkage of the λ1 tail** (report the soft posterior P(λ1>λ_th));
**clusters favour fine ≤7 Mpc/h smoothing** (mass-anchored, definitive — global-R²
optimum is a bulk artifact). Key caveat: the measured "information ceiling" is a
property of the *hand-crafted features*, not proven for the raw point cloud — that gap
is the main open question.

## Architecture / method menu (candidate upgrades)

- **Encoder:** equivariant GNN (EGNN/SEGNN/TFN) fed relative vectors rᵢⱼ + LOS r̂ᵢ,
  ideally outputting the tidal *tensor* (type-2) → eigenvalue ordering falls out for
  free, and RSD anisotropy becomes a first-class input. Alt: graph transformer
  (global attention → long-range/multi-scale in one hop).
- **Posterior head:** FMPE / flow-matching (continuous vector field) or Simformer,
  replacing the discrete MAF. Contained, low-risk modernization.
- **Inputs:** luminosity weighting (BGS flux, claim-preserving, truth-testable on
  cutsky R_MAG_ABS) is the cheapest relaxation. Multi-scale aperture density (3/7/10/14
  Mpc/h) is the proven +0.08 R²(λ1) lever. Explicitly NOT baryonic/HOD inputs.
- **Formulation:** field-level reconstruction (BORG/ELUCID) is the classical rival —
  frame our amortized per-galaxy NPE as its cheap complement, not competitor.

## Phase 0 — Triage (cheap GO/NO-GO; run in parallel; gate the expensive work)

1. **GNN vs GBM** (½ day, CPU): current GraphNet λ1 R² vs HistGBM, identical features.
   GO/NO-GO calibrates whether capacity or information is binding. (= rungs a–b below.)
2. **Luminosity weighting** (½ day, CPU, truth-gated on cutsky R_MAG_ABS): GO if λ1 R²
   lifts on truth.
3. **Representation ladder rungs c–d** (1–2 wk, GPU; gated by #1): c = equivariant GNN
   (rᵢⱼ+LOS→tensor), d = sparse U-Net field-level baseline. GO to production only if
   c or d ≫ b (current GraphNet). This definitively answers "are we too constrained by
   hand-crafted, distribution-only features." Either outcome is the ML4PS paper.
4. **FMPE head swap** (few days, GPU): GO if tails / SBC-TARP improve.

## Phase A — n(z) harmonization (parallel, architecture-independent)

Fix the sentinel-z injection bug first (2.07M phantom z≈0.59 galaxies in the mock
parent — current wedge clean, landmine for z-expansion). Measure clean n(z) mock vs
DESI; density-match training mocks to DESI n(z) + inject DESI-like z-errors (minimal
domain randomization = the "degraded mock" idea). Acceptance: summary-space MMD → the
split-half floor. Gate: if residual MMD stays high → escalate to full nuisance
randomization + **JEPA-style pretraining** (clean↔degraded views, unlabeled DESI +
cutsky; fine-tune flow head). JEPA = self-supervised; the degraded-mock idea as stated
is supervised domain randomization / nuisance-marginalized SBI — related, not identical.

## Phase B–D — production

- **B:** ONE bundled retrain = validated Phase-0 winners + n(z)-harmonized mocks +
  softplus-increment targets (best-calibrated). No incremental retrains.
- **C:** tile BGS bright into wedges; graph→features→NPE per wedge via sbatch; de-dup
  overlaps by posterior averaging.
- **D:** SBC/TARP, mass-anchored massive-halo recovery, n(z)-perturbation stability;
  full-footprint closure test; DESIVAST void cross-match.

## Product & uses

**GraphWeb-BGS** — per-galaxy calibrated environment posteriors keyed by TARGETID
(P(λ1>λ_th) + 4 class probs as headline columns; λ posteriors; OOD/MMD flag). Uses:
environmental quenching at fixed M*; posterior-weighted environment statistics;
intrinsic alignments (tidal field is the IA source); assembly-bias/HOD splits;
cross-survey/void-catalog comparison. Goes through DESI collaboration review — start
that clock early.

## Publication ladder

- Letter (now): smoothing vs cluster recovery, scale-matching, AUC-vs-completeness trap.
- Methods (after Phase B): calibrated NPE for web environment.
- VAC (after Phase C/D): GraphWeb-BGS — MNRAS/ApJS + DESI process.
- ML4PS @ NeurIPS 2026 (~late Aug deadline): the representation-ablation result.
- ICML/NeurIPS main track: only the JEPA branch, developed as a general method
  (sbibm + graphs + one more domain). Nature/Nat Astro: only a headline science
  result out of the catalog, not the methods.

## Immediate next steps

Start Phase 0 steps 1 & 2 (cheap CPU gates; step 1 is also rungs a–b of the workshop
paper). Fix sentinel-z bug + measure n(z) in parallel. Gate step 3 (GPU equivariant
run) on step 1.
