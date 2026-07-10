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

## Current production anchor

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

## Active direction: field-level physics-grounded inference

The immediate research priority is field-level inference rather than further
equivariant-architecture exploration:

`graph encoder -> density grid -> fixed FFT tidal operator -> eigensolver -> eigenvalues`

The fixed layer uses

`T_ij(k) = (k_i k_j / k^2) W_7(k) delta(k)`

so the density-to-tidal mapping is physical and differentiable rather than
learned. This supports tensor consistency and supplies eigenvectors for future
intrinsic-alignment science.

Current evidence:

- T2: a 3-D U-Net on 5 Mpc voxelised galaxy counts reaches lambda_1
  R2 = 0.876 +/- 0.004 across three seeds.
- T4/F1: graph encoder -> CIC scatter -> 3-D U-Net -> fixed physics layer
  reaches R2 = 0.841 / 0.897 / 0.931.
- The field encoder is not yet production: it must meet the NPE calibration
  gate, beginning with the F1 FlowJAX posterior head.

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
work is implementation, data processing, and interactive GPU experiments.
Synchronise through git and the science log:

1. Pull with `git pull --no-rebase` before pushing.
2. Preserve all log entries when resolving conflicts.
3. Commit and push substantive science-log updates promptly.
4. Use tmux for NERSC work that must survive SSH/VPN interruptions.