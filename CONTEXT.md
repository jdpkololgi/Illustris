# Research context — Illustris

## Role

Illustris is the simulation and methods hub for GraphWeb: cosmic-web inference
from galaxy catalogues using AbacusSummit mocks, IllustrisTNG development
workflows, graph neural networks, and simulation-based inference.

It is the source repository for shared models, target definitions, training
artifacts, and the shared `SCIENCE_LOG.md`.

## Scientific goal

Infer calibrated continuous posteriors over the ordered tidal-tensor eigenvalues

`lambda_1 <= lambda_2 <= lambda_3`

for DESI BGS galaxies. This supersedes a purely discrete T-web classification
into void, wall, filament, and cluster.

The published foundation is Kololgi et al. 2025 (RASTI,
doi:10.1093/rasti/rzag025), a graph-attention classifier trained on
IllustrisTNG-300.

## Current programme state — reconciled 2026-09-09

- **P12-A:** the untempered, uncorrected coordinate-aligned FMPE is the first
  per-galaxy posterior product. Its one-open ph001 evaluation reports green and
  all ten registered release gates pass on 4,897,905 galaxies. This is a
  registered conditional-on-mock/HOD claim, not exact conditional calibration,
  HOD marginalization, sim-to-real validation, or a coherent field posterior.
- **P12-B:** the matched point/frozen-latent/joint pilot and five-arm continuation
  are complete. Features are connected and used, but neither the pilot nor the
  continuation supports a production representation change. No further variant
  is automatically licensed.
- **D2:** seed42 passed its registered native-support-amended science and sampler
  ladder. The second training seed and combined-decision chain are submitted;
  no two-seed promotion is recorded. All galaxies were retained, and the
  registered conditional tolerance is 0.10 versus 0.05 for global/joint checks.
- **E2E-FIELD:** 160 response-screened anchors (320 nested 64/96-cell views)
  have source-verified arrays, whole-phase internal roles, train-only scaling
  and independent native tensor/eigenvalue references. Full-size engineering
  checks pass, but `training_ready=false`. The completed three-training-phase
  error audit finds FD2 sub-percent in typical eigenvalue RMS but topology-
  sensitive; floor sampling and finite-parent tides are larger typical errors.
  FD8 nearly matches the spectral reference. Separate spectral v2 products are
  verified complete (job 58115135, 44m05s). The matched cubic-target domain audit
  also completed (58120066): neither 64 nor 96 cells qualifies under the frozen
  internal tidal/topology screen. The 96-cell parent passes filling/pair limits
  but retains eigenvalue/class and topology failures. See
  `docs/e2e_field_domain_gate_20260909.md`. Final model/transform release and
  diagnostic-power gates remain blocked; no learned training
  has begun. See `docs/e2e_field_data_build_20260908.md` and
  `docs/e2e_field_error_budget_20260908.md`.
  A separate user-authorized wider-coarse/local-fine build and truth-only test
  launched as interactive CPU job 58131992 (code 5051f69): two larger extents
  with a fixed-extent resolution control, training phases only. Historical
  all-anchor criteria are diagnostics for this extension, not an automatic
  research-pilot veto. No model training or release is implied. See
  `docs/e2e_field_wide_coarse_20260909.md`. All 96 anchors / three variants and
  nine payloads (5.687 GB) are verified built; the truth-only test has started
  and its first anchor reports are saved. The full test is still running.
- **P13/Loa:** held pending separate authorization. P12-A's single blind opening
  does not authorize ph001 reuse by D2, P12-B or E2E.

See `docs/plan_generalisable_graphweb_vac.md`, the P12-B plans and
`docs/plan_end_to_end_conditional_field_posterior.md`. These status statements
come from the recorded terminal evidence, not continuous scheduler monitoring.

## Historical graph/NPE baseline

- Training data: AbacusSummit HOD cutsky mocks with CACTUS T-web labels.
- T-web convention: potential Hessian, `lambda_th = 0.2`, smoothing scale
  7 Mpc/h. Do not conflate this with V-web velocity shear.
- Baseline encoder: attentional Jraph Battaglia-style GraphNetwork.
- Baseline graph: union of Delaunay and radius-10 Mpc/h edges.
- Posterior model: FlowJAX normalising flow trained as NPE.
- Canonical target representation: ordered eigenvalue increments. Predict
  `lambda_1` and positive successive increments; invert only for evaluation
  and plots.
- G3 union-graph NPE performance: R2 = 0.804 / 0.846 / 0.895 for
  lambda_1 / lambda_2 / lambda_3.

## Field-level physics-grounded inference: motivation and historical evidence

The immediate research priority is field-level inference rather than further
equivariant-architecture exploration:

`graph encoder -> density grid -> fixed FFT tidal operator -> eigensolver -> eigenvalues`

The fixed layer uses

`T_ij(k) = (k_i k_j / k^2) W_7(k) delta(k)`

so the density-to-tidal mapping is physical and differentiable rather than
learned. This supports tensor consistency and supplies eigenvectors for future
intrinsic-alignment science.

Earlier deterministic evidence (not posterior-selection scores):

- T2: a 3-D U-Net on 5 Mpc voxelised galaxy counts reaches lambda_1
  R2 = 0.876 +/- 0.004 across three seeds.
- T4/F1: graph encoder -> CIC scatter -> 3-D U-Net -> fixed physics layer
  reaches R2 = 0.841 / 0.897 / 0.931.
- These deterministic scores do not establish posterior calibration. The
  current posterior status is the P12-A/P12-B/D2 summary above; full-cap field
  coherence remains a separate E2E research question.

The graph work remains central: the results indicate that representation scale
and a physics-grounded output are the main levers. A CNN is a GNN on a regular
lattice; graph encoders remain highly effective, especially in the
graph-to-field architecture.

## G4 evidence and architectural lessons

G4-PROPER is now supporting evidence rather than the active near-term gate.

- Union graphs beat either Delaunay-only or radius-only graphs.
- Delaunay void bridges and fixed-radius edges are complementary,
  geometry-anchored support for the non-local tidal operator.
- Position-only fixed-local attention recovers most curated-feature signal.
- Dynamic feature-space kNN / DGCNN harms performance: long connections should
  remain geometry-anchored, not feature-anchored.
- Tested SEGNN variants underperformed but were capacity-confounded; steerable
  models are shelved, not disproved.
- Attention is a second-order question to test cleanly inside the F-tier.

## Interpretation and safeguards

- The graph is a discrete quadrature of the non-local tidal operator
  (`1/k^2` inverse Laplacian), not merely a generic ML representation.
- NPE incorporates the Abacus training prior. Do not stack posteriors to infer
  population eigenvalue distributions without importance reweighting or
  hierarchical SBI.
- Feature-space domain shift and implicit-prior mismatch are distinct.
- TARP/SBC establish in-domain calibration, not sim-to-real validity.
- DESI closure tests remain necessary.
- Current field-level comparisons include unresolved controls: matched-estimand
  graph point estimates, DESI-like number-density re-runs, cell-size sensitivity,
  and attention on/off within the F-tier.

## Shared workflow

`SCIENCE_LOG.md` is the live source of truth. Read it before substantive work.
It wins if it conflicts with this file.

Use `[science]` for research decisions and `[code]` for implementation or run
results. Add only genuine decisions, hypotheses, results, and direction changes;
keep newest entries first.

Local desktop work is primarily science, planning, and interpretation. NERSC
work includes implementation, data processing, interactive development and
user-authorized frozen production jobs.
Synchronise through git and the science log:

1. Pull with `git pull --no-rebase` before pushing.
2. Preserve all log entries when resolving conflicts.
3. Commit substantive science-log updates; push when requested or authorized.
4. Use the NERSC allocation workflow: interactive allocations for development,
   explicitly authorized Slurm batch for frozen production. A login-node tmux
   supervisor is not a substitute for scheduler persistence. Preserve all
   scheduler approval rules and frozen source worktrees.
