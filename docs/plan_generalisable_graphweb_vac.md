# Generalisable GraphWeb — protocol-first training, blind validation, and DESI VAC

**Status:** ACTIVE WORKING PLAN
**Created:** 2026-07-16
**Programme:** Learning the Cosmic Web / GraphWeb-BGS
**Immediate objective:** replace interpolation-driven model selection with a common
spatially inductive training and validation protocol for graph, grid, field-physics,
and classical estimators.

## Authority and maintenance

Use the following authority order:

1. The newest empirical results in **SCIENCE_LOG.md**.
2. This implementation and validation plan.
3. **docs/roadmap_environmental_vac.md** for wider programme history.

This is a working document. Update the evidence ledger and decision register when a
result lands; do not silently erase failed hypotheses.

- **CLOSED:** evidence answers the question; do not reopen without new evidence.
- **ACTIVE:** currently being implemented or evaluated.
- **GATED:** run only after its named prerequisites pass.
- **DEFERRED:** scientifically interesting but outside the present production path.
- **BLOCKED:** awaiting a named data or infrastructure dependency.

Every experiment must state its hypothesis, controlled variable, inputs, spatial
split, metrics, adoption gate, artifact paths, hashes, and decision.

---

## 1. Correct reading of the evidence

### 1.1 Transfer results

| Model | Home/interpolation λ1 R² | Disjoint-wedge λ1 R² | Transfer reading |
|---|---:|---:|---|
| GraphNet, 8 passes | 0.804 | 0.421 | Strong interpolation; poor transfer |
| GraphNet, 2 passes | 0.628 | 0.369 | Reducing receptive field does not fix transfer |
| 3-D U-Net | 0.871 | 0.353 | High random-split accuracy does not imply transfer |
| DTFE + fixed tidal solve | — | 0.534 | More transferable in this dense wedge |

For the full-range spatial holdout:

| λ1 R² | Pooled | Macro | 0.15–0.25 | 0.25–0.35 | 0.35–0.45 | 0.45–0.55 |
|---|---:|---:|---:|---:|---:|---:|
| GraphNet R0 | 0.514 | 0.453 | 0.56 | 0.49 | 0.42 | 0.34 |
| 3-D U-Net | 0.549 | 0.461 | 0.570 | 0.545 | 0.502 | 0.226 |
| DTFE | 0.553 | 0.342 | 0.567 | 0.605 | 0.418 | -0.223 |
| CIC | 0.539 | 0.155 | 0.563 | 0.613 | 0.443 | -0.998 |

F-tier's dense-wedge result, R²(λ1) approximately 0.84, remains scientifically
interesting but was measured under the same spatially non-independent regime. Its
inductive transfer is not yet known.

### 1.2 Mandatory interpretation

- GraphNet does **not** presently beat DTFE where DTFE has adequate tracer support.
- The GraphNet macro advantage is entirely caused by DTFE/CIC collapsing in the
  sparsest shell. That shell must not manufacture a learned-model "win".
- The learned result in that shell is itself weak: GraphNet reaches only about 0.34.
- The U-Net is not generally inferior to GraphNet. On the full-range holdout it has
  higher pooled accuracy, wins three shells, and loses more strongly at high redshift.
- The different GraphNet and U-Net error profiles may contain useful complementarity.
- The two-pass test closes the simple "reduce GraphNet depth and transfer is solved"
  hypothesis.
- The disconnected-wedge U-Net test closes the simple "replace the GNN with a CNN and
  transfer is solved" hypothesis.
- F-tier retains unique scientific value through density, tensor, and eigenvector
  outputs, but must be tested under the same inductive protocol.
- Random-node validation is no longer an admissible production model-selection metric.
- A healthy random-split train/validation curve is not evidence of deployment
  generalisation.

> The current models were trained in ways that rewarded interpolation within one
> sampled cosmic field. The immediate task is to construct a training protocol that
> forces any encoder—graph, grid, or graph-to-field—to learn a transferable mapping.

### 1.3 What patches can and cannot do

Patches provide controlled optimization units, core/context separation, spatially
blocked validation, and regularisation against treating one wedge as one example.
They do **not** create independent universes or new long-wavelength modes. Independent
Abacus phases remain necessary for the strongest generalisation claim.

---

## 2. Scientific and product contract

The target remains the posterior over ordered real-space tidal eigenvalues,
conditioned on the observed galaxy catalogue and survey response.

- Ordered eigenvalues: λ1 <= λ2 <= λ3.
- Gaussian smoothing: R = 7 Mpc/h.
- Current target epoch: z = 0.2.
- Provisional input range: 0.15 < z < 0.55.
- T-web threshold: lambda_th = 0.2.

The immediate VAC science product is calibrated λ1 summaries and
P(λ1 > 0.2), with uncertainty, information, boundary, selection, and OOD flags.
Jointly calibrated λ2/λ3, four-class probabilities, tensor orientations, and adaptive
smoothing remain gated.

The winning production system may be GraphNet, a 3-D U-Net, F-tier, a
classical-plus-learned residual, a graph–field hybrid, or a validated ensemble. No
encoder family is the winner in advance. The training and validation protocol is the
primary hypothesis.

---

## 3. End-to-end dependency graph

The work packages are not a flat model queue.

~~~text
P0 evidence freeze + asset inventory
  |
  +--> P1 canonical catalogues and target alignment
          |
          +--> P2 canonical full-volume graph + global graph metrics
          |
          +--> P3 canonical count/response fields
          |
          +--> P4 shared fixed-comoving spatial manifest
                    |
                    +--> P5 GraphNet patch adapter + parity
                    +--> P6 U-Net patch adapter + parity
                    +--> P7 F-tier adapter + graph/field/FFT convergence
                              |
                              +--> P8 protocol-first deterministic showdown
                                        |
                                        +--> P9 complementarity/residual hybrids
                                        +--> P10 multi-phase training + blind test
                                                  |
                                                  +--> P11 JEPA gate, if justified
                                                  +--> P12 posterior calibration
                                                            |
                                                            +--> P13 DESI canary and VAC
~~~

P2, P3, and the first part of P4 can run in parallel after catalogue alignment. P5,
P6, and P7 can run in parallel after the shared manifest exists. Scientific training
must not start until the corresponding parity or convergence gate passes.

---

## 4. Preliminary work before model training

### P0 — Freeze evidence and inventory assets

**Status:** ACTIVE
**Duration:** 0.5–1 day CPU
**Blocks:** all later scientific comparisons

Tasks:

1. Recompute R0/A1 GraphNet validation and test metrics from frozen predictions.
2. Recompute the full-range U-Net on the identical scored rows.
3. Run DTFE/CIC on validation and test, with raw and source-calibrated predictions.
4. Add four-class metrics and P(λ1 > 0.2) Brier/reliability metrics.
5. Produce spatial-block uncertainties.
6. Inventory, for every phase/catalogue:
   - galaxy catalogue and target paths;
   - coordinate and unit conventions;
   - target epoch and smoothing metadata;
   - graph artifacts already available;
   - density grids/particles required for new T-web labels;
   - HOD and observer identifiers;
   - valid BOX_INDEX, halo, and TARGETID mappings.
7. Freeze a machine-readable evidence JSON and asset inventory.

**Gate:** all methods must align to the same target rows and target convention. Any
remaining target or row-alignment disagreement blocks graph and field generation.

### P1 — Canonical catalogue and target alignment

**Status:** GATED ON P0
**Duration:** 0.5–1 day CPU/high-memory
**Output:** one immutable raw catalogue per phase/HOD/observer

Store:

- catalogue, phase, HOD, and observer identifiers;
- global node ID and TARGETID;
- observer-frame Cartesian position for indexing;
- RA, Dec, redshift, and reporting shell;
- target eigenvalues and validity mask;
- halo/group identifiers;
- selection and completeness metadata;
- survey/mock boundary metadata;
- source-file hashes and target convention.

No train-fitted normalization belongs in this raw catalogue.

Required checks:

- unique IDs;
- finite valid targets;
- no BOX_INDEX < 0 in active rows;
- exact catalogue/target alignment;
- consistent units and cosmology;
- no train/validation/test filtering at this stage.

### P2 — Canonical full-volume graph and global graph metrics

**Status:** GATED ON P1
**Duration:** 1–3 days for the first catalogue; later catalogues pipeline in parallel
**Resources:** CPU/high-memory graph construction, then rapids-gnn GPU features

For every catalogue, construct the graph and graph metrics over the largest complete
contiguous input volume available **before** defining training patches. Do not build
separate redshift-shell graphs and concatenate them. Do not rebuild graphs inside
patches.

Graph build sequence:

1. Select the maximal parent catalogue providing context around the scored volume.
2. Construct the Delaunay graph.
3. Construct the existing fixed-physical-radius graph.
4. Form the Delaunay–radius union with edge-type provenance.
5. Do not connect across angular mask holes without an explicit mask-aware rule.
6. Save immutable senders, receivers, coordinates, edge types, and hashes.
7. Compute globally:
   - Degree;
   - Clustering;
   - tetrahedral Density;
   - Neighbour Density;
   - inertia eigenvalues;
   - edge length and direction;
   - edge density contrast.
8. Record the dependency support of every node and edge feature.
9. Create survey-boundary, graph-boundary, extreme-edge, and support flags.

"Global" describes the representation, not a mandatory single in-memory algorithm.
Distributed or overlapping graph construction is acceptable only after a canary shows
that interior topology and metrics match a monolithic calculation.

Graph gates:

- node count matches P1;
- senders/receivers are in bounds;
- no unexplained isolated populations;
- no redshift-shell seams;
- feature order and units are explicit;
- repeated builds are deterministic;
- global feature values survive serialization;
- graph and feature hashes are written.

Per-catalogue SLURM chain:

~~~text
catalogue_ready
  -> delaunay_job
  -> radius_union_job
  -> rapids_feature_job
  -> graph_validation_job
  -> GRAPH_COMPLETE manifest
~~~

Use successful SLURM dependencies, not a shared loose marker. Only the validation job
may write GRAPH_COMPLETE.

### P3 — Canonical full-volume count and response fields

**Status:** GATED ON P1; runs in parallel with P2
**Duration:** 0.5–1.5 days for the first configuration
**Resources:** CPU preprocessing; HBM80 GPU for model execution

Construct global voxel products once per catalogue, then extract U-Net/F-tier patch
views. Initial channels are:

- galaxy counts;
- expected counts from smooth selection and exposure;
- stabilized count contrast;
- luminosity-weighted counts where mock/DESI parity is established;
- mask/exposure;
- smooth ntilde(z);
- LOS unit-vector channels required by the established U-Net configuration.

Do not independently standardize or redefine fields inside patches. Fit learned channel
transforms on training cores after P4 and freeze them.

Field gates:

- counts conserve catalogue totals;
- axis convention and interpolation are verified;
- expected counts use no target information;
- zero exposure is distinguished from a physical void;
- channel metadata and hashes are stored;
- U-Net and F-tier use the same base fields where applicable.

### P4 — Shared fixed-comoving spatial manifest

**Status:** GATED ON P1; finalized after P2/P3 support atlases
**Duration:** 0.5–1 day CPU

Use observer-frame Cartesian positions to define fixed-comoving cores. Positions are
indexing metadata, not new GraphNet node features.

Evaluate:

~~~text
L_core = 32, 64, 96 Mpc/h
~~~

Default to 64 Mpc/h if at least 95% of context patches fit below 70% of an 80-GB GPU
and high-redshift occupied cores contain enough supervised galaxies. Use 32 if memory
is limiting. Use 96 only if high-redshift occupancy is limiting and memory is safe.
Select one global size, not one per redshift shell.

Super-block and fold construction:

1. Group cores into blocks with side approximately 4 * L_core.
2. Create five blocked folds.
3. Per rotation use three folds for training, one for validation, one for development
   testing.
4. Ensure all folds cover the usable redshift range.
5. Match validation and development-test volume and distance from training support.
6. Mark nodes whose complete representation support crosses split or survey boundaries.

RA 200–240 is development evidence, not a final blind test.

The manifest stores:

- patch and catalogue ID;
- fold and split ownership;
- core bounds and centroid;
- core galaxy and voxel indices;
- counts by reporting shell;
- distance to survey and split boundaries;
- graph, convolutional, and FFT support flags;
- catalogue, graph, field, and schema hashes.

Every model scores the same authoritative core galaxies. Context is
architecture-specific; target and core ownership are not.

Every eligible galaxy belongs to exactly one core in a fixed manifest and may appear
as context in many patches. Patch boundaries do not remove galaxies.

---

## 5. Architecture adapters and parity gates

### P5 — GraphNet patch adapter

**Status:** GATED ON P2/P4
**Duration:** 1–2 implementation days plus parity runs

Patches are views of the canonical graph:

- core nodes contribute to loss and metrics;
- context nodes participate in message passing only;
- outside nodes are absent;
- global node/edge features and connectivity are copied unchanged.

For K message passes include every node capable of influencing a core node through K
computational steps. Use reverse dependency traversal for directed edges.

Do not add raw RA, Dec, absolute Cartesian, or patch-relative coordinates as GraphNet
node features during the protocol gate. Keep established graph features so the first
experiment changes protocol rather than feature semantics.

No hard max_nodes or max_edges truncation is allowed. Subdivide oversized cores and
retain exact context. Use size buckets and explicit padding masks for XLA stability.

Graph parity requires:

- identical canonical features;
- identical relevant global edges;
- complete computational context;
- embeddings and predictions matching floating-point batching tolerance;
- no trend with core-boundary distance;
- invariance to core subdivision and patch order.

The existing local-subgraph-pipeline is scaffolding only. Its single-centre target,
hard caps, traversal-order truncation, small validation batch, and integrated FlowJAX
training are not production-safe.

### P6 — U-Net patch adapter

**Status:** GATED ON P3/P4
**Duration:** 1–2 implementation days plus parity runs

Extract field patches from canonical voxel products. Each has:

- an output core containing authoritative galaxies;
- a field-context halo covering the convolutional receptive field;
- frozen global channel definitions;
- training-core-fitted normalization applied elsewhere unchanged.

Sample predictions at the same core galaxies used by graph and classical models.

U-Net parity requires:

- identical global channels;
- stable core predictions as context grows;
- no independent patch normalization;
- no boundary-distance trend after the retained trim;
- identical galaxy-to-grid interpolation.

### P7 — F-tier graph/field/FFT adapter

**Status:** GATED ON P2/P3/P4
**Duration:** 2–3 implementation/convergence days

F-tier uses:

~~~text
canonical graph view
  -> graph encoder
  -> scatter to canonical field frame
  -> field decoder
  -> predicted density
  -> fixed FFT tidal operator
  -> tensor and eigenvalues
~~~

It inherits GraphNet graph parity and U-Net field requirements. The tidal operator is
nonlocal, so K-hop graph context alone is insufficient.

Run convergence over graph context, field-tile size, FFT padding/apodization, overlap,
central trim, and distance from tile/survey boundaries. Require stable density, tensor
components, eigenvalues, and eigenvectors. Record trace consistency and
eigengap-dependent orientation reliability.

---

## 6. Protocol-first deterministic model showdown

### P8 — Matched spatially inductive training

**Status:** GATED ON P5/P6/P7 PARITY
**Duration:** 2–4 GPU days for one-seed screening; three-seed finalists later

| ID | Candidate | Representation | Output |
|---|---|---|---|
| G-PATCH | GraphNet | canonical graph patches | ordered eigenvalues |
| U-PATCH | 3-D U-Net | canonical field patches | ordered eigenvalues |
| F-PATCH | F-tier | graph-to-field-to-fixed physics | density, tensor, eigenvalues |
| CLASSICAL | DTFE/CIC | global reconstruction | density, tensor, eigenvalues |

Common controls:

- deterministic ordered-increment heads first;
- no FMPE during representation selection;
- same spatial folds and core galaxies;
- transformations fitted on training cores;
- same target convention;
- same shell-weighted objective;
- no broad hyperparameter search;
- blocked validation for early stopping;
- complete validation evaluation;
- atomic best-checkpoint persistence;
- sealed test and blind phases.

For shell s, use w_i = N_s^(-1/2). For patch p, sample proportional to
W_p = sum_core w_i and optimize the weighted core mean. The expected objective is then
independent of patch subdivision.

Architecture controls:

- Start GraphNet with the R0/A1 eight-feature schema, not the failed aperture bundle.
- Start U-Net with the established selection-aware configuration.
- Start F-tier with the established v2_A concept under the new patch/FFT protocol.
- Do not rerun two-pass GraphNet.
- If exact eight-pass graph context is infeasible, compare matched four-pass full-graph
  and patch training to isolate protocol.

The scientific question is:

> Which representation generalises best when all serious candidates are trained and
> validated as spatial patches rather than random nodes within one field?

Screening order:

1. One seed on two folds for plumbing and obvious failures.
2. One seed across five folds for candidates that pass.
3. Three seeds for models within 0.03 macro R²(λ1) of the leader or offering unique
   physical outputs.
4. Freeze finalists before independent-phase truth is opened.

Select by fresh-region/fresh-graph generalisation, fold stability, knot probability,
macro and worst-shell accuracy, physical output value, deployment feasibility, then
pooled R².

---

## 7. Complementarity and hybrid models

### P9 — Residual correlation audit

**Status:** GATED ON P8
**Duration:** 0.5 day analysis before hybrid training

Measure out-of-fold residual correlations between GraphNet, U-Net, F-tier, DTFE, and
CIC versus shell, web class, tracer density, halo mass, and boundary distance.

Build a hybrid only if out-of-fold errors are genuinely complementary.

Candidates:

1. graph–field fusion using out-of-fold graph and field embeddings;
2. a zero-initialized ordered-eigenvalue residual around DTFE or F-tier;
3. a density residual around classical/F-tier density followed by the fixed tidal
   operator.

Never train a second stage on in-sample base predictions. Keep physical tensor outputs
and residual-corrected catalogue eigenvalues separately named where necessary.

---

## 8. Independent phases and blind evaluation

### P10 — Multi-phase target generation and training

**Status:** ACTIVE IN PARALLEL WITH P5–P9
**Duration:** scope after one-phase benchmark; likely days to weeks

Reserve:

- ph000: development and blocked protocol work;
- ph002–ph005: additional training phases;
- ph006: phase-level validation and calibration;
- ph001: sealed blind phase.

HOD seeds from one phase are population/observation variations, not independent cosmic
structures.

Preliminary phase work:

1. Audit particle/density-field availability.
2. Benchmark one 2048^3 target-generation run.
3. Validate its convention against ph000.
4. Record wall time, node-hours, storage, and failures.
5. Launch phase chains only after the benchmark passes.

Per-phase chain:

~~~text
density/particles
  -> CACTUS/T-web grid
  -> halo/galaxy target annotation
  -> P1 canonical catalogue
  -> P2 graph and metrics
  -> P3 canonical fields
  -> P4 manifest adapter
  -> PHASE_COMPLETE manifest
~~~

Blind protocol:

1. Freeze representations, weights, transforms, calibration strategy, classical
   calibration, and acceptance criteria.
2. Save a signed manifest.
3. Build fresh ph001 graph and field products without reading target metrics.
4. Run frozen finalists and classical baselines.
5. Save predictions and hashes.
6. Open the evaluator once.
7. Do not tune on ph001.

---

## 9. JEPA gate

### P11 — Representation pretraining

**Status:** GATED ON P8 AND MULTI-CATALOGUE DATA
**Duration:** 2–5 GPU days for bounded controls

JEPA is not GraphNet-only. Apply it to whichever graph, grid, or F-tier encoders remain
competitive.

Use matched controls:

1. random initialization;
2. masked reconstruction/denoising;
3. JEPA latent prediction.

For Graph-JEPA, prevent globally computed graph metrics leaking hidden targets: remove
target nodes/edges and exclude a feature-support guard. Target location may enter the
JEPA predictor but not production GraphNet node features.

For grid JEPA, mask 3-D blocks built from counts, expected counts, mask/exposure, and
luminosity channels. Reuse the U-Net encoder before considering a new transformer.

The preferred variant uses paired observed views of the same latent field with varied
HOD, velocity bias, magnitude selection, fibre assignment, completeness, and redshift
errors.

Do not pretrain on DESI until truth-known sim-to-sim controls pass. DESI pretraining is
transductive domain adaptation, not zero-shot generalisation.

Adopt JEPA only for consistent fresh-graph or blind-phase improvement, targeting at
least +0.03 R²(λ1) or comparably clear class-probability gain.

---

## 10. Posterior inference and VAC production

### P12 — Posterior calibration

**Status:** GATED ON A FROZEN DETERMINISTIC WINNER/HYBRID

1. Generate spatially out-of-fold embeddings or base predictions.
2. Fit FMPE/NPE on training phases.
3. Include ntilde(z) and response covariates directly in posterior conditioning.
4. Tune on ph006.
5. Evaluate once on ph001.

Require SBC, TARP, coverage, conditional coverage, knot-probability reliability, Brier
skill, width-versus-error, posterior contraction, and prior-dominated flags. Scalar
tempering that repairs average coverage while leaving shape failure is insufficient.

### P13 — DESI canary and scale-out

**Status:** GATED ON P10/P12

Reproduce the winning representation exactly:

- GraphNet: canonical DESI graph/global metrics, then graph patch views.
- U-Net: canonical count/response fields, then field patches.
- F-tier: canonical graph/fields plus converged FFT overlap and trim.
- Hybrid: all components with out-of-fold-compatible fusion semantics.

Run a golden mock and one DESI canary before full scale-out. Every eligible galaxy is
authoritative core exactly once. Overlapping contexts are not independent evidence;
never multiply overlapping posteriors.

Required flags include redshift support, graph/field support, boundary, mask hole,
extreme edge, completeness, prior domination, OOD, and overlap disagreement.

---

## 11. Evaluation and release gates

Every finalist reports pooled, macro, worst-shell, and per-shell eigenvalue metrics;
ordered-increment skill; Spearman; MAE; bias; slope; variance; class confusion;
balanced accuracy; macro-F1; void/knot recall; knot Brier skill; and results versus
halo mass, sampling density, degree, completeness, and boundary.

Use spatial-block or phase-level uncertainty. Do not bootstrap galaxies independently.

Report classical comparisons as pooled, per shell, macro over tracer-supported shells,
and sparse-shell failure separately. A learned model does not beat classical merely
because classical becomes undefined in the sparsest shell.

A learned/hybrid primary VAC estimator must improve source-calibrated DTFE on a blind
fresh phase with paired spatial uncertainty, avoid degrading tracer-supported shells,
provide real sparse-shell skill, and pass calibration/information gates.

If no learned model robustly improves DTFE, use classical reconstruction with calibrated
uncertainty as the defensible primary option and badge learned outputs experimental.

---

## 12. Near-term schedule: 2026-07-16 to 2026-07-21

This is an accelerated infrastructure/evidence milestone, not permission to skip blind
validation.

### July 16

- Complete P0 evidence and asset inventory.
- Freeze cache/manifest schemas.
- Launch P1 catalogue alignment.
- Register jobs and artifact roots.

### July 17

- Launch P2 global Delaunay/radius/union construction.
- Launch P3 canonical fields in parallel.
- Build the preliminary P4 manifest.
- Start the independent-phase target-generation cost benchmark.

### July 18

- Complete graph metrics and graph validation.
- Finalize the support atlas and core size.
- Implement P5/P6 adapters.
- Begin GraphNet/U-Net parity.

### July 19

- Close GraphNet/U-Net parity.
- Implement F-tier adapter and FFT convergence.
- Run one-patch and one-fold smoke tests.
- Do not train scientifically if parity fails.

### July 20

- Launch one-seed G-PATCH and U-PATCH blocked screens.
- Launch F-PATCH only if convergence passes.
- Complete the phase-generation cost decision.
- Produce the first protocol comparison without opening sealed data.

### July 21

Freeze a **protocol-ready VAC development bundle** containing:

- canonical catalogue/graph/field manifests;
- shared folds;
- passing patch adapters;
- parity/convergence reports;
- preliminary blocked-fold results;
- independent-phase generation schedule;
- provenance and CFS backup.

Do not call the catalogue science validated unless P10 and P12 actually pass. The honest
July 21 success condition is a reproducible generalisation-first pipeline, not a rushed
full-footprint VAC.

---

## 13. Longer schedule by dependency

| Stage | Earliest start | Typical duration | Exit artifact |
|---|---|---:|---|
| P0 evidence/inventory | immediate | 0.5–1 d | evidence + asset JSON |
| P1 catalogue alignment | P0 | 0.5–1 d/catalogue | immutable raw catalogue |
| P2 graph/metrics | P1 | 1–3 d first catalogue | GRAPH_COMPLETE |
| P3 canonical fields | P1 | 0.5–1.5 d | FIELD_COMPLETE |
| P4 spatial manifest | P1, finalize after P2/P3 | 0.5–1 d | shared folds/cores |
| P5/P6 adapters | P2/P3/P4 | 1–2 d each, parallel | parity reports |
| P7 F-tier adapter | P2/P3/P4 | 2–3 d | FFT convergence report |
| P8 showdown | adapter gates | 2–4 GPU d + seeds | blocked-fold ranking |
| P9 hybrids | P8 | 1–3 GPU d if justified | complementarity decision |
| P10 phases | benchmark | days–weeks | blind phase report |
| P11 JEPA | P8 + phases/views | 2–5 GPU d | JEPA decision |
| P12 posterior | frozen winner | 2–5 GPU d | calibration report |
| P13 DESI | P10/P12 | deployment dependent | golden canary + VAC shards |

---

## 14. Artifact and run discipline

Keep code, configuration, schemas, and decisions in Git. Store large assets under a
versioned scratch root resolved through shared/config_paths.py, with subdirectories for
catalogues, graphs, features, fields, manifests, patches, runs, and evaluations.

Every stage writes:

- configuration JSON;
- input hashes;
- Git SHA;
- environment;
- SLURM job ID/resources;
- row/node/edge counts;
- validation report;
- completion manifest written only by the authoritative process.

Jobs must be idempotent and resumable. Evaluation/finalization must not write a
trainer's completion marker. Copy irreplaceable manifests, scalers, checkpoints, and
results to CFS incrementally.

---

## 15. Explicit defaults and closed branches

- Primary hypothesis: protocol and independent training diversity.
- Active model families: GraphNet, 3-D U-Net, F-tier.
- Mandatory baseline: DTFE/CIC.
- Possible endpoint: physics-residual or graph–field hybrid.
- Core geometry: fixed comoving.
- Graph metrics: computed globally once per catalogue.
- GraphNet coordinate features: none added during protocol testing.
- U-Net fields: globally constructed before patch extraction.
- F-tier: graph and FFT context must converge.
- Initial range: 0.15 < z < 0.55.
- Smoothing: 7 Mpc/h.
- Threshold: lambda_th = 0.2.
- Target epoch: z = 0.2.
- Deterministic selection before posterior fitting.
- sqrt(N) shell objective.
- ph001 sealed blind.
- RA 200–240 is development evidence, not blind.
- Two-pass GraphNet transfer fix: **CLOSED—FAILED**.
- Broad random-split architecture search: **CLOSED**.
- GBM feature gate: **CLOSED**.
- Fifteen-feature aperture model: **CLOSED—FAILED**.
- JEPA: **GATED**.
- FMPE: **GATED** on deterministic generalisation.
