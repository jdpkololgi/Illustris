# Actual legacy GAT: density × smoothing experiment

This completed experiment uses `workflows/gcn_paper/gnn_models.py::SimpleGAT(10,4,
num_heads=4)`, not the local-feature regressors used in the earlier
`tng_density_control_20261015` experiment. Results and final verification
are recorded below. The user's explicit request authorizes
this historical four-class classifier experiment; it does not replace the
programme's canonical ordered-eigenvalue target policy.

## Model and data contract

- Four GATv2 layers. The first three have four heads, width 20 per head,
  LayerNorm and dropout 0.1; layers 2/3 have the original residual paths.
  Edge length is the one-dimensional edge input. The final layer emits four
  class logits, with no invented eigenvalue-regression head.
- Ten historical node inputs: weighted degree, mean/min/max incident edge
  length, weighted clustering, tetrahedral density, mean neighbor tetrahedral
  density, and three neighborhood covariance eigenvalues. No stellar mass,
  SFR, colour, observed outcome or matter truth is a model input.
- Rebuild Gudhi alpha complexes and all features for normal and sparse
  populations. Normal: 283,969 tracers. Sparse: the existing seed-20261015
  random subset of 37,545 tracers. Alpha=1.5 n^(-1/3), in physical comoving
  Mpc: approximately 6.9063 and 13.5566. As in the historical GAT, the graph
  is nonperiodic, while matter truth is periodic. This differs from the
  earlier periodic-neighbor MLP experiment.
- Same TNG300-1 snapshot99 matter field and verified R2/R7 truth as the
  preceding experiment. Classes count eigenvalues >=0.2. Sparse training
  targets are retained graph nodes; removed tracers cannot survive in the
  features or topology. Dense versus sparse therefore changes both contexts
  and available training targets; it is not a fixed-query causal experiment.
- Spatial train/validation/test label splits from the preceding experiment.
  Validation and test galaxies are restricted to the sparse retained IDs in
  both arms, allowing a same-galaxy held-out comparison. Unlabelled graph
  context can cross the label-split boundaries, as in a transductive GNN.

## Optimization and verification contract

One paired initialization seed, 1,200 Adam updates at initial lr=0.003,
16 randomly selected training targets per update, inverse-training-frequency
class weights, float32, norm-1 gradient clipping. ReduceLROnPlateau applies
to weighted validation loss; choose the lowest validation-loss checkpoint.
Fit Box–Cox (+1e-6) only on training rows. This is a new bounded optimization
experiment, not bitwise replay of historical full-batch training. Do not
claim convergence, multi-seed stability, calibrated probabilities or an
architecture winner from these fits.

Exact incoming dependency blocks preserve **all** neighbors through four
message-passing layers. There is no random neighbor truncation. This reduces
memory by evaluating only nodes needed for each destination layer. It is
checked against the unchanged canonical full forward, initially and again
at the selected checkpoints. Loa inference evaluates each complete layer
in destination chunks and is separately checked against four-hop blocks.

Graph metrics use the required `rapids-gnn` environment with CPU NumPy/SciPy;
topology, PyTorch, and validation use `cosmic_env`. Weighted clustering and
covariance eigenvalues have independent NetworkX/direct-centered references.
Dense topology and rescaled features are compared with the archived training
cache; input truth/catalogue/feature hashes bind to the preceding manifest.

## Same-Loa transfer

Build one graph from the 587,896 padded Loa tracers used by the preceding
comparison. Keep all 103,788 query IDs and their exact catalogue binding.
All four fits use the same observed graph and galaxies. Historical DESI
alpha=6.906260417222935 Mpc; Planck18 comoving coordinates, redshift space.
The observed tracers are never thinned and outcome mass cuts do not rebuild
the graph. Boundary sensitivity is separate from the common parent sample.

Two explicit transfer conventions:

1. Historical default operation: fit a Box–Cox transform to observed graph
   features and recenter. This uses this padded graph, not the old full-sky
   low-redshift catalogue; it is operation parity, not identical moments.
2. Frozen training transform, with no observed refit or recentering.

The source's optional `first_moment_matching` branch instead transforms with
the saved scaler then recenters; neither convention here should be silently
renamed as that branch. Abacus uses saved P12-A probabilities. GAT softmax
values are class scores, not a posterior calibrated like a qualified SBI
product; both soft and hard descriptive estimators are retained.

Seven CG15 properties are overlaid before/after the recovered 90% limiting-
mass cut in z=0.20–0.30 and its lower/upper halves. No inference is extrapolated
to the historical z<=0.06 sample. A stronger association is not an accuracy
test or a reason to promote a model. See GraphWeb_DESI's companion report and
`workflows/presentation/loa_actual_gat_overlay_20261015.py`.

## Execution

Slurm inspection failed both inside/outside the sandbox. The user then
explicitly instructed: “Just use the login node.” Stages execute sequentially,
one CPU thread each, without scheduler submissions. Graphs/checkpoints are
compact Home artifacts because Scratch is read-only in this session.
The code, decisions and provenance remain in the owning repositories.

Entrypoints: `workflows/gcn_paper/legacy_gat_density_20261015.py` and
`workflows/gcn_paper/run_legacy_gat_density_20261015.py`.
Artifacts: `~/fmsu_tokyo_20261010/legacy_gat_density_20261015/`.

## Completed results

All four fits and eight Loa inference passes completed. On the identical
2,048 held-out TNG galaxies, balanced accuracy is R2 normal/thinned
71.03%/61.25%, R7 normal/thinned 67.56%/71.36%. Selected updates are
1000/1100/900/900, respectively. Do not interpret the R7 ordering as a
confirmed advantage of thinning; optimization/seed uncertainty remains.

Dense graph replay has zero directed-edge differences from the archived
3,890,098-edge cache. Position max difference 1.71e-13 Mpc; transformed
feature max difference 5.66e-7. All four selected models replay on identical
held-out IDs and truth. Layerwise Loa inference agrees with exact four-hop
evaluation to <=3.82e-6 in logits. `VALIDATION.json` and `MANIFEST.json`
retain the individual measurements and hashes.

The actual-GAT same-Loa result does not restore a large advantage over
Abacus. At a common parent/cut mass-redshift mixture and historical observed
normalization, probability-weighted cluster-minus-void quenching contrasts
[pp] before/after the 90% completeness cut are Abacus 9.460/9.647,
R2 normal 8.732/8.590, R2 thin 5.893/5.894, R7 normal 6.031/6.015,
R7 thin 7.595/7.644. Lower/upper redshift halves and a conservative interior
margin retain the qualitative conclusion. These are descriptive contrasts,
not matched truth-accuracy scores.

The frozen transform produces severe class-occupancy failures: normal R2 has
no predicted clusters; normal R7 is 95.45% void and reverses the raw quenching
trend. Sparse R2's 24.24-pp raw hard contrast coexists with only 0.046% voids
and 65.04% clusters. No shared controlled four-class support exists across
all five models under the frozen transform. Empty/unsupported results remain
visible rather than being used to select a preferred model.

Detailed observational report: GraphWeb_DESI
`docs/loa_actual_gat_overlay_20261015.md`. A separate 690-check MLP selection
experiment also finds little effect of the full-range completeness cut at
fixed reference mixture. Neither analysis repeats the old z<=0.06 selection
or establishes convergence, calibrated transfer or an architecture winner.
