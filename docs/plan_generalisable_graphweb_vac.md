# Generalisable GraphWeb — protocol-first training, blind validation, and DESI VAC

**Status:** ACTIVE WORKING PLAN
**Created:** 2026-07-16
**Programme:** Learning the Cosmic Web / GraphWeb-BGS
**Controlling pre-shutdown objective:** **Demonstrate transferable deterministic
inference under the new patch protocol.**

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

The physical target is the ordered real-space tidal-eigenvalue triplet conditioned
on the observed galaxy catalogue and survey response.

- Ordered eigenvalues: λ1 <= λ2 <= λ3.
- Gaussian smoothing: R = 7 Mpc/h.
- Current target epoch: z = 0.2.
- Provisional input range: 0.15 < z < 0.55.
- T-web threshold: lambda_th = 0.2.

The **pre-shutdown deliverable is deterministic**, spatially transferable inference of
the three eigenvalues and derived threshold classes. It is acceptable—and preferable
under the time constraint—to select the training protocol with point-estimate R² and
class metrics before adding a posterior head. A seed ensemble is a robustness measure,
not a Bayesian posterior.

The later VAC may add calibrated λ1 summaries and P(λ1 > 0.2), with uncertainty,
information, boundary, selection, and OOD flags. Those posterior columns require P12;
they are not prerequisites for deciding whether patch training works. Jointly
calibrated λ2/λ3, four-class probabilities, tensor orientations, adaptive smoothing,
and nuisance-marginalised HOD uncertainty remain further gated.

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
  +--> P0S preservation manifest and environment specifications [parallel, no moves yet]
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
                                        +--> P9 complementarity/residual hybrids [if justified]
                                        +--> P10 multi-phase training + blind deterministic test
                                                  |
                                                  +--> P13 deterministic DESI canary
                                                  +--> P11 JEPA gate [optional]
                                                  +--> P12 posterior calibration [optional]
                                                            |
                                                            +--> P13 posterior VAC columns
~~~

P2, P3, and the first part of P4 can run in parallel after catalogue alignment. P5,
P6, and P7 can run in parallel after the shared manifest exists. Scientific training
must not start until the corresponding parity or convergence gate passes.

---

## 4. Preliminary work before model training

### P0 — Freeze evidence and inventory assets

**Status:** COMPLETE (2026-07-17)
**Runtime:** one 80 GB A100 interactive allocation plus CPU evidence evaluation
**Blocks:** cleared for P1–P4; independent-phase work remains gated separately in P10

Tasks:

- [x] Recompute R0/A1 GraphNet validation and test metrics from frozen predictions.
  Artifacts: `/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/{R0,A1_sqrt}_canonical_predictions.npz`.
- [x] Recompute the full-range U-Net on the identical scored rows.
  Artifact: `/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/evidence_point_dryrun.json`.
- [x] Run DTFE/CIC on validation and test, with raw and source-calibrated predictions.
  The evidence artifact records raw and frozen training-affine results separately;
  target-domain oracle calibration is excluded from deployment comparisons.
- [x] Add four-class metrics and P(λ1 > 0.2) Brier/reliability metrics.
  Posterior probabilities and deterministic point decisions are explicitly distinguished.
- [x] Produce 100 Mpc comoving spatial-block bootstrap uncertainties for every method.
- [x] Inventory, for every phase/catalogue:
   - galaxy catalogue and target paths;
   - coordinate and unit conventions;
   - target epoch and smoothing metadata;
   - graph artifacts already available;
   - density grids/particles required for new T-web labels;
   - HOD and observer identifiers;
   - valid BOX_INDEX, halo, and TARGETID mappings.
  Artifact: `docs/evidence/p0/asset_inventory.json`.
- [x] Freeze a machine-readable evidence JSON and asset inventory.
  Artifacts: `docs/evidence/p0/evidence_freeze.json` and
  `docs/evidence/p0/asset_inventory.json`; checksummed runtime copies are under
  `/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/`.

P0 implementation and runtime artifacts:

- evaluator: `workflows/abacus_tweb/p0_evidence.py`;
- frozen GraphNet exporter: `workflows/abacus_tweb/p0_export_graphnet_predictions.py`;
- inventory builder: `workflows/abacus_tweb/p0_inventory_assets.py`;
- interactive entry point: `workflows/abacus_tweb/run_p0_evidence.sh`;
- reusable allocation skill: `~/.codex/skills/nersc-interactive-allocation/`;
- tests: `tests/phase4/test_p0_evidence.py`;
- runtime directory: `/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/`.

**Gate: PASS.** All seven methods align to the same 219,929 canonical rows and target
convention; calibration is fitted on training rows only; test is evaluation-only;
posterior and deterministic probabilities are distinguished; uncertainty resamples
spatial blocks rather than individual galaxies.

### P0S — Preservation manifest before scratch migration

**Status:** ACTIVE IN PARALLEL; documentation only until the user approves copies
**Duration:** 0.5 day CPU/login-node metadata work
**Blocks:** does not block P1–P8, but must complete before the shutdown freeze

Perlmutter scratch is active workspace, not the source of truth. NERSC documents that
scratch is not backed up and that files not accessed for eight weeks are eligible for
purge. Do not evade purge policy by artificial access-time updates.

Use the following storage contract:

| Tier | Store here |
|---|---|
| Git/home | source code, workflow scripts, schemas, configs, manifests, hashes, environment specs, decisions, compact evidence |
| CFS | irreplaceable reusable catalogues, scalers, selected checkpoints/predictions, golden canaries, release bundles |
| HPSS | large expensive-to-rebuild density, T-Web, graph, staged-mock, and archival bundles |
| pscratch | active or reproducible intermediates, temporary caches, current training outputs |

Required preservation work, without moving data yet:

- [ ] Inventory every script and configuration that exists only under
  `/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/`, including the upstream
  preparation, fibre-assignment, mkCat, stage-2/stage-3, annotation, and cache runners.
- [ ] Seed that inventory with the known source candidates already present:
  - root documentation/runners: `README.md`, `STAGE3_DESI_ALIGNMENT.md`,
    `run_stage2_example.sh`, and `run_stage3_example.sh`;
  - `scripts/`: `build_mock_bgs_maglim_catalog.py`,
    `inject_loa_spec_from_zall.py`, `stage1_cutsky_subset.py`,
    `upstream_getpotaDA2_mock.py`, `upstream_prepare_mocks_Y3_bright.py`,
    `upstream_mkCat_SecondGen_amtl.py`, `audit_path1.py`,
    `audit_verify_sentinel.py`, `check_ph000_env.sh`, `run_loa_BCDE.sh`,
    `run_path1_prepare.sh`, `run_path1_fiberassign.sh`,
    `run_path1_mkcat.sh`, `run_path1_mkcat_fulld_only.sh`,
    `run_path1_mkcat_mkclusdat_only.sh`, `run_path1_maglim_from_fulld.sh`,
    `run_stage3_desi_aligned_mkcat.sh`, and `install_fba_to_univ000.sh`;
  - `wedge/`: `run_stage3_annotate.sh`, `run_stage3_annotate.slurm`,
    `run_stage3_sbi_cache_15d.sh`, `CACHE_TRAINING_NEW_WEDGES.md`, and
    `STAGED_MOCK_WEDGE_SBI_README.md`.
  Generated FITS/NPZ/catalogue trees, logs, `__pycache__`, fibre-assignment outputs,
  and backup files are data/evidence/archive candidates—not source-code candidates.
- [ ] Classify each item as source code, configuration, small evidence, reusable data,
  rebuildable cache, selected checkpoint, or archive.
- [ ] Record size, checksum, producing command, dependencies, downstream consumers,
  recommended destination, and copy priority in one migration manifest.
- [ ] Identify which staged-mock scripts are already represented by tracked code under
  `workflows/abacus_tweb/`; do not duplicate them blindly.
- [ ] Define a versioned repository destination for the genuinely scratch-only source
  before copying it. No migration occurs in this planning step.
- [ ] Export reproducibility records for both environments:
  - `/pscratch/sd/d/dkololgi/conda/envs/cosmic_env`;
  - `/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn`.
- [ ] For each environment save a from-history YAML, explicit package specification,
  conda package inventory, pip inventory, Python version, module/CUDA metadata,
  relevant environment variables, and a minimal smoke-test command.
- [ ] Include RAPIDS/cuGraph versions and a graph-feature smoke test for `rapids-gnn`;
  it is the required environment for large global node/edge metric construction.
- [ ] Produce a dry-run move table with total bytes for Git/home, CFS, and HPSS and
  identify anything that cannot be regenerated from tracked code plus retained inputs.
- [ ] Perform copies only after review; verify destination checksums before considering
  any scratch source disposable.

Official policy references:

- `https://docs.nersc.gov/filesystems/perlmutter-scratch/`;
- `https://docs.nersc.gov/policies/data-policy/policy/`;
- `https://docs.nersc.gov/filesystems/community/`;
- `https://docs.nersc.gov/filesystems/quotas/`.

### P1 — Canonical catalogue and target alignment

**Status:** COMPLETE — P1a wedge canary plus authoritative P1b full NGC+SGC index
**Duration:** 0.5–1 day CPU/high-memory
**Output:** one immutable raw catalogue per required phase/observer; HOD variants optional later

Scope is explicit:

- **P1a canary:** `ph000_path1_wedge_v1`, 374,537 rows, is retained for loader,
  parity, and resource smoke tests.
- **P1b authoritative development catalogue:** the full usable ph000 BGS footprint
  over both NGC and SGC, with the production redshift core and context buffers.
- NGC and SGC are stored in one row-indexed catalogue with an explicit component/cap
  identifier. They are not cropped to one rectangular RA/Dec fan.
- Preserve the stable parent row index so existing global graph and feature products
  can be fancy-indexed without changing identity or ordering.

P1b artifacts:

- `/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz`;
- `/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/manifest.json`;
- `/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/CATALOGUE_COMPLETE`.

Progress checklist:

- [x] Freeze the catalogue ID, phase, observer, target convention, redshift core,
  and context range in `manifest.json`.
- [x] Retain the 9,538,254-row parent FITS ordering as the canonical global node ID.
- [x] Build `canonical_index.npz` with parent ID, TARGETID, cap, shell,
  active/context masks, and target-validity mask.
- [x] Verify exact parent-FITS, XYZ, graph-row, and target alignment.
- [x] Verify unique TARGETIDs, finite active targets, and zero active
  `BOX_INDEX < 0` rows.
- [x] Record NGC/SGC, per-shell, halo-grouping, and source-hash provenance.
- [x] Write `CATALOGUE_COMPLETE` and track the compact P1b manifest under
  `docs/evidence/p1b_p2b/`.

Store:

- catalogue, phase, and observer identifiers; record HOD identifiers when present, but do
  not generate new HOD variants for the baseline protocol;
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
- exact alignment between canonical rows and the existing full-footprint graph parent;
- explicit NGC/SGC counts and component labels;
- no duplicated underlying halo/TARGETID may cross supervised folds later in P4.

### P2 — Canonical full-volume graph and global graph metrics

**Status:** COMPLETE — P2a wedge canary plus authoritative P2b full NGC+SGC union representation
**Duration:** 1–3 days for the first catalogue; later catalogues pipeline in parallel
**Resources:** CPU/high-memory graph construction, then rapids-gnn GPU features

For every catalogue, construct the graph and graph metrics over the largest complete
contiguous input volume available **before** defining training patches. Do not build
separate redshift-shell graphs and concatenate them. Do not rebuild graphs inside
patches.

For the current ph000 path1 catalogue, "largest complete contiguous input volume"
means the full usable **NGC plus SGC survey footprint**, not the RA 118–162 canary.
Because the footprint has two disconnected sky caps, "global" means one canonical
row-indexed graph object containing two components:

1. construct or retain NGC topology within NGC;
2. construct or retain SGC topology within SGC;
3. map both through stable parent/global indices;
4. concatenate them without any NGC–SGC edge;
5. compute or copy graph metrics from these full-cap components before patching.

The existing path1 full-footprint Delaunay and cuGraph products should be promoted
after exact provenance and row-alignment checks rather than recomputed gratuitously.
The completed 374,537-node union graph is **P2a**, a canary for the construction and
validation chain; it does not satisfy authoritative P2b by itself.

P2b reuses the exact 9,538,254-node parent Delaunay graph and global cuGraph
metrics. Within the 6,397,925-node context support it adds 141,819,389
non-Delaunay radius pairs, stored by cap. The resulting union has 190,563,017
undirected context pairs and zero cross-cap edges.

Artifacts: `/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/`
(`p2b_union_manifest.json`, `ngc_radius_only_*.npy`,
`sgc_radius_only_*.npy`, and `UNION_COMPLETE`).

Compact audit and provenance copies are tracked under
`docs/evidence/p1b_p2b/`.

Progress checklist:

- [x] Audit the existing 9,538,254-node full-footprint Delaunay graph against P1b.
- [x] Verify exact global node-feature and edge-index alignment with parent rows.
- [x] Verify the two-cap construction and zero NGC-SGC edges.
- [x] Retain global node metrics and Delaunay edge attributes without patch-level
  recomputation.
- [x] Define the authoritative 6,397,925-node context support from P1b.
- [x] Build fixed-radius pairs independently within NGC and SGC context.
- [x] Remove Delaunay overlaps and retain 141,819,389 radius-only additions.
- [x] Verify the 190,563,017-pair context-union arithmetic and edge provenance.
- [x] Store parent Delaunay plus per-cap radius-only arrays under stable global IDs.
- [x] Pass bounds, finite-feature, count, determinism, and cross-cap gates.
- [x] Write `UNION_COMPLETE` and track compact P2b evidence under
  `docs/evidence/p1b_p2b/`.

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
- exactly two expected cap components before any legitimate within-cap fragmentation,
  with zero cross-cap edges;
- NGC and SGC node/edge counts are recorded separately;
- any fancy-index projection preserves parent node features and edge attributes exactly;
- P2a wedge/full-parent parity is retained as a regression test, not used as the
  scientific training volume.

Per-catalogue development chain:

~~~text
catalogue_ready
  -> Delaunay build in a reused interactive allocation
  -> radius-union build in the same allocation where feasible
  -> rapids-gnn feature calculation in a reused GPU allocation
  -> graph validation
  -> GRAPH_COMPLETE manifest
~~~

For development, reuse `salloc` allocations through
`~/.codex/skills/nersc-interactive-allocation/`; never exceed the user's two-allocation
limit. Reserve `sbatch` chains for production-scale repeated phase processing after the
development commands are validated. Use stage-specific success manifests, not a shared
loose marker. Only graph validation may write `GRAPH_COMPLETE`.

### P3 — Canonical full-volume count and response fields

**Status:** P3a COMPLETE — canonical NGC+SGC fields and `FIELD_COMPLETE` frozen; P3b response upgrades deferred
**Duration:** 0.5–1.5 days for the first configuration
**Resources:** CPU preprocessing; HBM80 GPU for model execution

**Completion:** P3a passed all in-build and independent readback gates. P4 patch
manifest construction and P6 U-Net/F-tier adapters are unblocked. P3b remains a
post-protocol observation-model upgrade because the staged parent has no explicit
random-catalogue exposure, per-object completeness, or luminosity fields.
Use one fixed observer-frame Cartesian lattice definition per cap. NGC and SGC must
never be enclosed in one enormous dense cuboid or joined through empty sky. A
chunked/sparse-on-disk representation is acceptable, but every chunk must be indexed
in the immutable cap lattice so overlapping patch reads return identical voxels.

Construct global voxel products once per catalogue, then extract U-Net/F-tier patch
views. Initial channels are:

- galaxy counts;
- expected counts from smooth selection and exposure;
- stabilized count contrast;
- luminosity-weighted counts where mock/DESI parity is established;
- mask/exposure;
- smooth ntilde(z);
- LOS unit-vector channels required by the established U-Net configuration.

The shutdown-critical **P3a protocol baseline** is deliberately narrower:

- CIC galaxy counts from every P1b context galaxy, deposited once in its cap lattice;
- binary and apodized footprint/exposure support;
- smooth radial expected counts from the frozen selection prescription times exposure;
- stabilized count contrast;
- smooth ntilde(z);
- LOS unit-vector channels.

The authoritative parent FITS has no explicit random-catalogue exposure,
per-object completeness, or luminosity columns. Luminosity-weighted counts and
higher-fidelity random/completeness fields are therefore **P3b upgrades**, not
requirements for the deterministic patch-protocol gate. P3a must record the exact
exposure approximation and must not claim a production-complete observation model.

Progress checklist:

- [x] Confirm P1b catalogue/index/XYZ alignment, context mask, cap counts, and hashes.
- [x] Confirm wedge U-Net helpers are references for deposition, LOS, interpolation,
  and channel semantics—not a full-cap builder.
- [x] Audit the parent FITS schema and record the absent response/luminosity channels.
- [x] Resolve the unit convention independently: P1b graph points and historical
  U-Net points are observer-frame comoving Mpc, while the T-Web target smoothing is
  7 Mpc/h. The historical 5 Mpc cell is 3.383 Mpc/h for Planck18; literal 5 Mpc/h
  would be a distinct 7.390 Mpc cell. Unit evidence:
  `/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/unit_audit.json`.
- [x] Freeze `p3_field_schema_v1.json`: cell size, units, axis order, per-cap
  origins, context padding, chunks, deposition kernel, dtype, channels, and hashes.
- [x] Make the passing `unit_audit.json` a hard `FIELD_COMPLETE` dependency.
- [x] Estimate per-cap storage at 5 and 6 comoving Mpc (3.383 and 4.060
  Mpc/h under Planck18); select 5 Mpc using historical parity and the 12.06-GB
  raw-channel estimate across both caps.
- [x] Freeze P3a exposure support and radial selection sources; prove they use no
  targets or supervised split ownership.
- [x] Implement an idempotent, cap-separated full-field builder over P1b global IDs.
- [x] Run a P1a canary against the established U-Net field implementation.
- [x] Build NGC and SGC fields over the complete P1b context.
- [x] Verify CIC conservation globally, by cap/shell, and across chunk boundaries.
- [x] Verify finite channels, axis/interpolation parity, no cross-cap mixing, and
  stable overlapping reads.
- [x] Produce the exposure/occupancy/expected-count/support atlas by cap and shell.
- [x] Write `field_manifest.json`, validation report, channel hashes, and
  `FIELD_COMPLETE`; track compact evidence under `docs/evidence/p3/`.
- [x] Close the catalogue/field/target chain with independent CIC redeposition,
  repeated-host target equality, CWEB/eigenvalue consistency, and cap/shell
  count-contrast versus T-Web-trace correlations against shuffled controls.
  Runtime evidence: `/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/catalogue_field_closure.json`;
  tracked evidence: `docs/evidence/p3/catalogue_field_closure.json`.
- [x] Inventory candidate random/completeness products for P3b without blocking P3a.

P3a artifact root: `/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/`.
Authoritative products: `ngc_fields.h5`, `sgc_fields.h5`, `field_manifest.json`,
`validation_report.json`, `postbuild_validation.json`, `support_atlas.json`,
`unit_audit.json`, and `FIELD_COMPLETE`. Compact copies live in `docs/evidence/p3/`.

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

**Status:** COMPLETE — immutable 64-Mpc/h five-fold manifest plus P2/P3 support atlas
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

Progress checklist:

- [x] Freeze the candidate core-size resource probe over both NGC and SGC.
  The selected scientific core is exactly 64 Mpc/h = 94.5906 Mpc at Planck18
  `h=0.6766`; it is not rounded to P3 voxel edges. Runtime evidence:
  `/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/core_size_probe.json`;
  tracked evidence: `docs/evidence/p4/core_size_probe.json`.
- [x] Build fixed-comoving cap core cells independently of split ownership.
- [x] Group cores into spatial super-blocks before assigning folds.
- [x] Group repeated `(FILE_NUM, BOX_INDEX, HALO_INDEX)` halos and TARGETIDs so
  no underlying object crosses supervised folds. The full-sky parent contains 59,238
  periodically repeated images separated by 2.89–4.19 Gpc; retain one deterministic
  occurrence for supervision and keep the others context-only.
- [x] Assign five blocked folds with both caps and all reporting shells represented.
- [x] Match validation/development-test volume and distance from training support.
- [x] Assign every eligible active galaxy to exactly one authoritative core.
- [x] Attach P2b graph-support distances and exact 2/4-step union-graph dependency flags.
- [x] Attach P3 convolutional/exposure support after `FIELD_COMPLETE`.
- [x] Reserve FFT support fields without blocking GraphNet/U-Net.
- [x] Write immutable manifest, hashes, support atlas, and completion marker.
- [x] Test unique core ownership, fold isolation, shell/cap coverage, and deterministic
  rebuilds.

P4 artifact root: `/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/`.
Authoritative products: `spatial_manifest.json`, `cores.npz`, `super_blocks.npz`,
`context_assignment.npz`, `active_assignment.npz`, `rotations.json`,
`graph_support_{context,active}.npz`, `graph_support_manifest.json`,
`field_support.npz`, `core_field_support.npz`, `field_support_manifest.json`,
`p4_validation.json`, and `PATCH_MANIFEST_COMPLETE`. Compact evidence is tracked
under `docs/evidence/p4/`.

The 5,026,863 authoritative rows are balanced to 1.010 max/min active count across
folds and to 2.42% maximum relative deviation in any cap/shell cell. Exact union-graph
split safety is architecture-dependent: 62.08% are safe within two literal graph hops,
but only 29.30% within four. P5 established that the current attention
`GraphNetwork` has two graph-hop dependencies per nominal model pass, so its
two-pass strict subset is the four-hop mask (29.30%), not the two-hop mask. At
z=0.45–0.55 that four-hop fraction is effectively zero. A receiver-only two-pass
variant with one-hop dependence per pass may use the 62.08% two-hop mask; deeper or
different candidates require their computational dependencies to be derived, audited,
and re-gated rather than inferred from the layer count. P3 exposure supports 99.95%
of authoritative rows; 91.14% and 77.04% have at least 20 and 40 Mpc of
convolutional support.

---

## 5. Architecture adapters and parity gates

### P5 — GraphNet patch adapter

**Status:** COMPLETE (2026-07-19)
**Duration:** completed in one reusable interactive CPU allocation

Patches are views of the canonical graph:

- core nodes contribute to loss and metrics;
- context nodes participate in message passing only;
- outside nodes are absent;
- global node/edge features and connectivity are copied unchanged.

Model passes and graph dependency hops are separate quantities. For every candidate,
derive its computational dependency before assigning a P4 strict-support mask. The
current receiver-normalized attention `GraphNetwork` aggregates both sent and
received edges, so one nominal pass reaches two graph hops and two passes require
four-hop context. A receiver-only model would instead use one hop per pass. P5 uses
exact incident-edge traversal over the canonical union graph and stores both
`model_passes` and `dependency_hops`; it never assumes they are equal.

The primary loss and metric mask is frozen as every P4 authoritative core galaxy.
The architecture-specific hop-isolated mask is stored separately as
`strict_support_mask` and is a diagnostic, not the primary gate. Four-hop
strictness retains only 106 of 62,243 galaxies at z=0.45-0.55 and would make the
mandatory macro-shell metric undefined. Report robustness on three nested samples:

1. every authoritative core galaxy (primary);
2. architecture-independent physical margins beyond 10.4 and 20.8 Mpc from a fold
   boundary;
3. architecture-specific hop-isolated subsets, always with their retained fractions.

This same-catalogue P8 experiment is **spatial target generalization with a globally
observed representation**. No validation/test label enters training, but label-free
galaxies from the complete catalogue may enter global graph construction and model
context, as they will for DESI. Reserve **fresh-graph inductive generalization** for
P10, where each independent phase receives its own separately constructed graph and
the frozen training transformations. Do not use the word `inductive` without naming
which of these two settings is meant.

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

Completed evidence:

- Canonical adapter index:
  `/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter/`.
- Construction manifest: `adapter_manifest.json`; 9,538,254 parent nodes,
  190,563,017 undirected union pairs, 381,126,034 directed messages, 18,765 P4
  cores, and all construction gates pass.
- Parity report: `parity_report.json`; the actual shared two-pass attention
  architecture on the 100,935-node P1a graph matches full-graph embeddings and a
  fixed decoder exactly for four-hop patches (maximum embedding and prediction
  difference 0; subdivision difference below 2.4e-7).
- The full-footprint smoke suite samples a strict four-hop core from every NGC/SGC
  x fold stratum, verifies canonical-feature identity, exact masks, non-truncating
  subdivisions, and unique core ownership.
- Runtime readiness markers: `GRAPH_PATCH_READY` and
  `GRAPH_PATCH_PROD_GPU_READY`.
- Versioned schema and compact runtime evidence:
  `docs/evidence/p5/p5_graph_patch_schema_v1.json`,
  `docs/evidence/p5/p5_graph_patch_schema_v2.json`,
  `docs/evidence/p5/adapter_manifest.json`,
  `docs/evidence/p5/parity_report.json`,
  `docs/evidence/p5/parity_report_prod_gpu.json`,
  `docs/evidence/p5/GRAPH_PATCH_READY`, and
  `docs/evidence/p5/GRAPH_PATCH_PROD_GPU_READY`.
- Production-shape A100 parity used latent size 80, eight attention heads, 100,935
  full-graph nodes, 1,988,732 canonical undirected pairs, and a 5,737-node /
  194,584-directed-edge patch. Patch order is exactly invariant; maximum embedding
  and prediction differences are 1.54e-3 and 7.38e-4. These are 2.1e-4 and 5.7e-4
  relative to the corresponding output scales and pass the pre-registered 2e-3
  GPU tolerance.

The first G-PATCH protocol control is pre-registered as the existing two-pass
receiver-normalized attention GraphNetwork with the R0/A1 eight-feature schema and
four exact dependency hops. This isolates the patch-training protocol from an
architecture change. A receiver-only two-pass model is a separately named optional
challenger, never a post-hoc substitute. The historical two-layer disconnected-wedge
transfer failure motivates this retraining but does not answer the new protocol test.
Eight-pass attention is out of scope under the present P4 support because it requires
16 exact dependency hops.

Feature transformations are never fitted per patch. For each development rotation,
fit node SI medians per cap and Box-Cox parameters from authoritative training-fold
nodes; fit edge transforms from training-fold internal edges. Freeze and apply those
objects unchanged to validation/test patches. Blind phases and DESI must use frozen
training-ensemble transformations, not target-catalogue refits.

Progress checklist:

- [x] Implement P2b parent-Delaunay plus radius-only patch assembly by global ID.
- [x] Implement exact incident-edge dependency traversal for the required graph hops.
- [x] Separate authoritative core loss nodes from context-only nodes.
- [x] Preserve canonical node/edge features without patch recomputation.
- [x] Implement size buckets and padding masks with no node/edge truncation.
- [x] Pass full-graph versus patch embedding/prediction parity on P1a.
- [x] Pass subdivision, patch-order, and core-boundary parity tests.
- [x] Write adapter schema, parity report, tests, and `GRAPH_PATCH_READY`.
- [x] Separate the authoritative primary loss mask from strict-hop diagnostics.
- [x] Pre-register physical-margin and hop-isolated robustness subsets.
- [x] Freeze the two-pass attention G-PATCH protocol control before P8.
- [x] Pass a production-shape GPU parity point and write
  `GRAPH_PATCH_PROD_GPU_READY`.

### P6 — U-Net patch adapter

**Status:** COMPLETE (2026-07-19) — adapter-ready; final U-PATCH checkpoint must repeat the release gate
**Duration:** complete

Extract field patches from canonical voxel products. Each has:

- a P4 physical core that owns authoritative evaluation galaxies;
- a nominal voxel core for dense output and normalization bookkeeping;
- a field-context halo covering the convolutional receptive field;
- frozen global channel definitions;
- training-core-fitted normalization applied elsewhere unchanged.

P4 physical galaxy ownership is authoritative. Because rounded P3 lattice bounds do
not coincide exactly with physical P4 core faces, a small fraction of authoritative
galaxies lies just outside the nominal voxel-core slice. They remain legitimate core
targets if their interpolation stencil is inside the extracted context. Never drop
them merely to force voxel-core equality.

The structural adapter is implemented at
`/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/`. Its
`adapter_manifest.json` indexes 5,026,863 authoritative galaxies across 18,765 cores.
`structural_parity_report.json` passes exact canonical-channel identity, identical
galaxy order and coordinates, and interpolation parity across 4- and 8-voxel context
views. The structural readiness markers are `FIELD_PATCH_INDEX_READY` and
`FIELD_PATCH_PARITY_READY`. Compact evidence is tracked under `docs/evidence/p6/`.

P3 remains immutable and retains its wedge-derived `ntilde` channels for provenance.
P6 now supplies a versioned full-cap selection overlay instead of rewriting those
arrays. For each of the five P4 rotations it fits separate NGC and SGC curves using
only label-free observed galaxy redshifts and apodized effective exposure volume in
that rotation's three training folds. Validation and development folds never determine
the curve. The corresponding channel normalizer is fitted once per rotation on
supported training-core voxels, pooled over both caps, and is then frozen.

All selection gates pass. The maximum training-shell expected/observed error over
five rotations, two caps, and four reporting shells is 0.57%. Validation/development
ratios range from approximately 0.87 to 1.12 and are retained as spatial-variation
diagnostics rather than fitted away. Fixed knot-spacing sensitivity reaches 8.85%
and is recorded as a selection-model systematic. Patch overlays are exactly invariant
between 4- and 8-voxel views; expected-count and contrast identities agree to
6e-8 and 4.8e-7. Runtime products are under
`/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/fullcap_selection_v1/`;
the operative markers are `SELECTION_REFIT_COMPLETE` and
`SELECTION_CHANNELS_READY`, with tracked validation evidence in
`docs/evidence/p6/`.

The trained structural convergence suite then exposed a previously hidden patch
dependency in the historical T2 U-Net: PyTorch `GroupNorm` computes statistics over
all spatial positions in a patch, so otherwise identical core predictions changed
with patch size even when convolutional context was ample. The failing result is a
real architectural warning, not a reason to relax the gates. U-PATCH must use
patch-safe normalization: the adopted canary uses per-voxel channel LayerNorm with
the historical affine weights retained. Spatial `GroupNorm`, spatial `InstanceNorm`,
and patch-local input normalization are forbidden for production patch training.

With patch-safe normalization and an 8-voxel global-lattice phase lock for strided
pooling, the smallest stable context-growth tail begins at 24 voxels (120 Mpc) against
an 80-voxel reference. "Stable tail" means that the selected halo and every larger
non-reference halo pass; an isolated passing point is not convergence. At 24 voxels,
pooled galaxy-prediction NRMSE is 0.00155, latent-core NRMSE is 0.00387, and the
worst-core prediction NRMSE is 0.00301. Parent-versus-two-child subdivision gives
prediction NRMSE 0.00166 and p95 absolute error / reference standard deviation
0.00355. The retained boundary correlation is formally 0.203, but the pre-registered
trivial-effect branch passes because prediction NRMSE is below 0.002.

The suite is label-free: it never loads tidal targets or uses R-squared to select
patch geometry. It spans both caps and all four redshift shells. The historical
trained checkpoint is only a structural canary for learned filters and activations;
the final U-PATCH model must be trained from scratch with patch-safe normalization
and must repeat this suite before release.

Runtime evidence is under
`/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/trained_convergence_v1/`.
Tracked evidence is `docs/evidence/p6/p6_field_patch_schema_v2.json`,
`trained_convergence_report.json`, and `UNET_PATCH_READY`.

U-Net parity requires:

- identical global channels;
- stable core predictions as context grows;
- no independent patch normalization;
- no boundary-distance trend after the retained trim;
- identical galaxy-to-grid interpolation.

Progress checklist:

- [x] Implement cap-lattice field patch reads from the immutable P3 schema.
- [x] Separate output core, convolutional context, and unsupported survey boundary.
- [x] Fit channel normalization on supported training-core voxels only, pool both
  caps within each rotation, and freeze it for validation/development application.
- [x] Sample predictions at the shared P4 authoritative core galaxies.
- [x] Pass global-field versus patch-view channel and interpolation parity.
- [x] Refit separate full-cap NGC/SGC expected-count channels from each rotation's
  training folds only; pass shell closure and patch-overlay parity.
- [x] Pass trained-model context-growth, subdivision, and boundary-distance convergence.
- [x] Freeze the full schema and write `UNET_PATCH_READY`.

### P7 — F-tier graph/field/FFT adapter

**Status:** COMPLETE (2026-07-19) — adapter-ready; final F-PATCH checkpoint must repeat the release gate
**Duration:** complete

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

The initial composition gate is implemented in
`workflows/abacus_tweb/p7_ftier_patch_utils.py` and
`workflows/abacus_tweb/p7_validate_initial_composition.py`. On real NGC and SGC
cores it verifies exact authoritative-ID composition, conservative TSC scatter,
small/large-halo overlap parity, finite ordered eigensystems, and fixed-operator
trace consistency at approximately 3e-15. Runtime evidence is under
`/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter/`, with compact copies under
`docs/evidence/p7/` and the marker `FTIER_COMPOSITION_READY`.

The P6 selection blocker is resolved: the real NGC/SGC composition suite uses the
passed full-cap overlay and binds its manifest hashes. P5 separately establishes
exact graph-context parity for arbitrary encoder weights. Because no historical
full-range F-tier checkpoint exists, the P7 convergence suite uses the patch-safe
trained T2 latent field as a learned-spectrum structural canary, then applies the
fixed nonlocal tidal operator. It loads no tidal targets and spans NGC/SGC plus the
lowest and highest redshift shells.

The first coupled 64-voxel-halo configuration failed despite converged learned density:
tensor NRMSE was 0.0553 and eigenvalue NRMSE was 0.0502. The limiting operation was
therefore the nonlocal FFT, not the local decoder. Frozen one-factor controls confirm
that both padding and apodization are causal: at fixed 80-voxel context, reducing
padding from 24 to 16 voxels gives tensor/eigenvalue NRMSE 0.0982/0.1078, while
reducing apodization from 20 to 16 voxels gives 0.0339/0.0370.

The smallest passing eligible configuration is frozen as:

- learned-field context halo: 72 voxels = 360 Mpc;
- FFT zero padding: 20 voxels = 100 Mpc;
- cosine apodization: 20 voxels = 100 Mpc;
- authoritative output: the shared P4 physical core;
- reference: 80-voxel halo, 24-voxel padding, 20-voxel apodization.

Relative to that reference, tensor NRMSE is 0.02272, eigenvalue NRMSE is 0.01770,
and eigenvalue p95 absolute error / reference standard deviation is 0.04097. Trace
consistency is 1.9e-15. Orientation changes show the expected eigengap dependence;
for the large-gap half of each axis, the worst median and p95 changes are 0.744 and
1.817 degrees. Near-degenerate axes remain intrinsically unreliable and must be
reported with an eigengap quality measure.

The physical survey-support audit uses a retained trim of 2 smoothing lengths =
20.69 Mpc. It retains 8,869 galaxies and identifies 463 nearer-boundary galaxies.
A residual rank trend remains beyond the trim (Spearman = -0.342), but the retained
mean eigenvalue change is only 0.0158 of the reference standard deviation and passes
the pre-registered 0.02 trivial-effect branch; the near-boundary value rises to
0.0253. This is not described as zero boundary dependence. Near-support rows require
an explicit boundary-quality flag, and the final checkpoint must reproduce the audit.

Runtime evidence is under
`/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter/trained_convergence_v1/`.
Tracked evidence is `docs/evidence/p7/p7_ftier_patch_schema_v2.json`,
`trained_convergence_report.json`, and `FTIER_PATCH_READY`. This marker certifies the
adapter geometry and numerical operator. It does not certify F-tier inference
accuracy: the final F-PATCH checkpoint must repeat the complete decoder/FFT suite
before any tensor or eigenvector product is released.

Progress checklist:

- [x] Compose the passing P5 graph view and P6 canonical field frame.
- [x] Verify graph-to-field scatter conservation and overlap parity.
- [x] Consume the passed P6 full-cap selection overlay and frozen per-rotation
  normalization contract without modifying P3.
- [x] Freeze FFT padding, apodization, overlap, and central-trim candidates.
- [x] Run learned-field convergence for density, tensor components, and eigenvalues.
- [x] Verify eigengap-conditioned orientation stability under tile convergence.
- [x] Quantify residual error versus physical survey-support distance after a
  2-smoothing-length trim and freeze the near-boundary quality-flag requirement.
- [x] Write the full adapter schema, convergence report, and `FTIER_PATCH_READY`.
- [ ] Repeat the complete convergence suite with the final trained F-PATCH checkpoint
  before releasing tensor/eigenvector columns (release gate, not adapter blocker).

---

## 6. Protocol-first deterministic model showdown

### P8 — Matched spatial target-generalisation training

**Status:** READY FOR ONE-SEED SCREENING — P5, P6, and P7 adapter gates pass
**Duration:** 2–4 GPU days for one-seed screening; three-seed finalists later

G-PATCH, U-PATCH, and F-PATCH may now enter the matched deterministic protocol.
Readiness is adapter-specific, not an accuracy claim. Each final trained checkpoint
must repeat its own parity/convergence gate before it becomes release-eligible.

U-PATCH and the field decoder inside F-PATCH must use patch-safe normalization.
Per-voxel channel normalization or an equally local/frozen alternative is allowed;
normalization whose statistics include the patch spatial dimensions is prohibited.
This choice is frozen before training and may not be changed after validation results.

| ID | Candidate | Representation | Output |
|---|---|---|---|
| G-PATCH | GraphNet | canonical graph patches | deterministic eigenvalues |
| U-PATCH | 3-D U-Net | canonical field patches | deterministic eigenvalues |
| F-PATCH | F-tier | graph-to-field-to-fixed physics | density, tensor, eigenvalues |
| CLASSICAL | DTFE/CIC | global reconstruction | density, tensor, eigenvalues |

The P8 scientific question is:

> Can a model trained on many canonical core/context patches transfer labels to
> unseen spatial structures in the same globally observed catalogue, and which
> representation does so most reliably?

P10 asks the stricter fresh-graph question on independent simulation phases.

#### P8.1 Target contract

Use the current practical baseline:

```text
v1 = lambda1
v2 = lambda2 - lambda1
v3 = lambda3 - lambda2
```

This is the existing **linear-increment** convention. Fit its scaler on training-core
targets only. Convert to eigenvalues by cumulative sum for evaluation; do not silently
sort predictions. Record the ordering-violation rate.

The following target evidence is already closed:

- shape parameters `(I1,e,p)` and invariants `(I1,I2,I3)`: pathological, deprecated;
- raw, softplus, and linear increments: already compared under the wedge NPE protocol;
- 15-component derivative targets: already implemented with
  `(lambda1, Delta12, Delta23, R grad lambda1..3, R^2 laplacian lambda1..3)` and
  block weights `1.0/0.1/0.03`; informative historical work, but not demonstrated to
  improve spatial transfer;
- the failed July `R1 15-d` experiment was 15 **input features**, not this derivative
  target.

A literal log-gap target is one bounded ablation, not a new workstream:

```text
v1 = lambda1
v2 = log(max(lambda2 - lambda1, epsilon))
v3 = log(max(lambda3 - lambda2, epsilon))
```

Run it only after the linear-increment patch baseline passes parity and trains
successfully. Use the leading encoder, identical folds, seed, update budget, loss, and
training-core scaler policy. Its inverse uses positive exponentiated gaps and therefore
guarantees ordering. Adopt only if it improves the primary metric reproducibly without
materially degrading the worst shell, class metrics, or numerical stability. Do not
run raw/softplus/15-d/shape/invariant sweeps before the deterministic protocol gate.

#### P8.2 Common training controls

- deterministic point heads only; no FMPE/NPE during protocol selection;
- same spatial manifest, folds, authoritative core galaxies, target convention, and
  train-fitted transformations;
- context nodes/voxels never contribute to the loss;
- every eligible training galaxy is authoritative core exactly once per manifest
  rotation; patch overlap changes context, not scientific weight;
- no broad hyperparameter search;
- blocked validation for early stopping;
- evaluate the complete validation fold at every checkpoint decision;
- persist atomic best checkpoints and the predictions used to choose them;
- sealed development-test and blind phases.

For shell s, use `w_i = N_s^(-1/2)`. For patch p, sample proportional to
`W_p = sum_core w_i` and optimize the weighted core mean. The expected objective is
independent of patch subdivision. Log actual optimization exposure by shell and patch.

Architecture controls:

- The required G-PATCH protocol control is the existing two-pass
  receiver-normalized attention GraphNetwork, R0/A1 eight-feature schema, and four
  exact dependency hops. Retrain it under P4/P5; do not treat the historical
  disconnected-wedge transfer run as a substitute.
- A two-pass receiver-only GraphNet is an optional, separately named challenger after
  the control. Choosing it after observing control results is forbidden.
- Do not run the eight-pass attention model under the present manifest: it needs 16
  exact dependency hops and has inadequate valid support.
- U-PATCH and F-PATCH are first-class candidates, not GraphNet auxiliaries. Start
  U-Net with the established selection-aware configuration after the P6 selection
  refit; start F-tier with v2_A only after its nonlocal convergence gate.
- Keep architecture-specific safe fractions visible alongside accuracy; never change
  P4 fold geometry to rescue a preferred encoder.

#### P8.3 Controlling metric and checkpoint rule

For fold f and each of the four reporting shells s, compute `R²_f,s(lambda1)` over
**all authoritative validation-core galaxies in that fold and shell**. Never average
per-patch R² values.

```text
macro_R2_f = mean_s R2_f,s(lambda1)
primary_score = mean_f macro_R2_f
```

All four current shells are mandatory. If a shell lacks enough nodes, spatial blocks,
or target variance for a stable R², pre-register that failure and report it; do not
silently remove the shell after seeing model results.

Checkpoint selection uses `macro_R2_f` on the complete validation fold. Final model
selection uses `primary_score` across blocked folds, accompanied by fold scatter and
a spatial-block uncertainty interval. Pooled galaxy-level R² is tertiary.

Mandatory safeguards:

- worst-shell and every per-shell R² for all three eigenvalues;
- Spearman correlation, MAE, bias, slope, and predicted/true variance;
- balanced four-class accuracy, macro-F1, confusion, void recall, and knot recall;
- ordering-violation rate for unconstrained increments;
- source-to-transfer gap and residual spatial correlation;
- performance versus tracer density, graph degree, completeness, and boundary distance;
- runtime, memory, and patch-failure rate;
- primary metrics on every authoritative core galaxy;
- the same metrics beyond physical fold-boundary margins of 10.4 and 20.8 Mpc;
- architecture-specific hop-isolated diagnostics with retained fractions, never used
  to replace or veto the primary metric.

Brier skill and probability reliability are P12 posterior metrics. A deterministic
threshold decision must not be relabelled as a calibrated probability.

#### P8.4 Screening and success levels

1. Run one seed on two blocked folds for plumbing and obvious failure.
2. Run one seed across all five folds for candidates that pass.
3. Repeat candidates within 0.03 primary-score units of the leader—or candidates with
   unique physical outputs—across three seeds.
4. Freeze the target transform, architecture, preprocessing, and acceptance criteria
   before any independent-phase truth is opened.

A **P8 protocol pass** requires finite complete-fold results, stable patch execution,
no material boundary trend, improvement over the matched frozen/current learned
baseline on the spatial-fold primary score, and no shell whose degradation erases that
gain. A mean gain of at least 0.03 is the promotion target; smaller gains remain
provisional unless uncertainty clearly supports them.

The matched classical row is a hard scientific adoption baseline, not merely a
diagnostic floor. Define learned-model failure unambiguously as the best matched,
train-calibrated classical estimator exceeding every learned candidate on the primary
spatial-fold score, especially if the excess is reproduced across folds and is not
confined to one pathological shell. Conversely, classical methods failing to exceed a
learned candidate is necessary but not sufficient for production promotion: P10 blind
fresh-phase transfer must still pass. If no learned candidate at least ties the best
classical row within the spatial-block uncertainty, record a P8 learned-model NO-GO,
promote the classical reconstruction to the production reference, and do not rescue a
learned branch using pooled R-squared or the collapse of one classical high-redshift
shell.

A **production-transfer pass** additionally requires P10: a frozen fresh-graph test on
an independent phase. P8 same-phase blocked folds alone establish promising spatial
transfer, not universal simulation-to-DESI validity.

Rank candidates by primary score, fold/seed stability, worst-shell behaviour,
source-to-transfer gap, physical output value, and deployment feasibility. Use pooled
R² only after those criteria.

Progress checklist:

- [ ] Freeze the linear-increment target/scaler and complete-fold macro-R2 evaluator.
- [ ] Freeze shared P4 folds and classical comparison rows before training.
- [ ] Run one-seed/two-fold plumbing screens for every parity-passing encoder.
- [ ] Reject incomplete core coverage, boundary trends, or subdivision-dependent
  scientific weighting.
- [ ] Promote passing candidates to one seed across all five folds.
- [ ] Repeat near-leaders or uniquely physical candidates across three seeds.
- [ ] Run log-gap only on the stable leader if the baseline passes and time remains.
- [ ] Freeze out-of-fold predictions, mandatory metrics, checkpoints, and configs.
- [ ] Record GO/NO-GO per encoder without calling same-phase folds a production pass.

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

Progress checklist:

- [ ] Align spatially out-of-fold residuals for P8 finalists and classical methods.
- [ ] Measure complementarity by fold, shell, class, density, mass, and boundary.
- [ ] Pre-register one hybrid only if errors are demonstrably complementary.
- [ ] Train with out-of-fold base predictions and preserve output provenance.
- [ ] Retain only for blind/fresh-region improvement.

---

## 8. Independent phases and blind evaluation

### P10 — Multi-phase target generation and training

**Status:** ACTIVE IN PARALLEL WITH P5–P9 FOR PHASE/COST BENCHMARKING
**Duration:** scope after one-phase benchmark; likely days to weeks

Reserve:

- ph000: development and blocked protocol work;
- ph002–ph005: additional training phases;
- ph006: phase-level validation and calibration;
- ph001: sealed blind phase.

HOD seeds from one phase are population/observation variations, not independent cosmic
structures. The existing staged LSS catalogues are sufficient for the deterministic
protocol gate. Do not generate new HOD samples before shutdown. Add HOD-family
variation only after spatial and independent-phase transfer work, when testing
population robustness or nuisance marginalisation.

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

1. Freeze representations, weights, transforms, deterministic target/metric contract,
   classical calibration, and acceptance criteria. Posterior calibration is not part
   of this blind deterministic gate.
2. Save a signed manifest.
3. Build fresh ph001 graph and field products without reading target metrics.
4. Run frozen finalists and classical baselines.
5. Save predictions and hashes.
6. Open the evaluator once.
7. Do not tune on ph001.

Progress checklist:

- [ ] Audit density/particle and staged-LSS availability for ph001-ph006.
- [ ] Benchmark one 2048-cubed target-generation run and record cost/storage.
- [ ] Validate target convention and annotation parity against ph000.
- [ ] Freeze phase roles and a signed blind-evaluation manifest.
- [ ] Build fresh P1-P4 products for training/validation phases.
- [ ] Train and freeze finalists without reading ph001 truth metrics.
- [ ] Build ph001 graph/field products and save predictions before opening truth.
- [ ] Evaluate ph001 once and record the production-transfer decision.

---

## 9. JEPA gate

### P11 — Representation pretraining

**Status:** DEFERRED UNTIL P8 DETERMINISTIC GATE; OPTIONAL THEREAFTER
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

The preferred first variant uses paired observed views of the same latent field with
varied magnitude selection, fibre assignment, completeness, and redshift errors.
HOD/velocity-bias variation is a later nuisance-robustness extension, not a prerequisite
for the bounded JEPA test.

Do not pretrain on DESI until truth-known sim-to-sim controls pass. DESI pretraining is
transductive domain adaptation, not zero-shot generalisation.

Adopt JEPA only for consistent fresh-graph or blind-phase deterministic improvement,
targeting at least +0.03 spatial-fold macro R²(λ1) or a comparably clear balanced-class
gain.

Progress checklist:

- [ ] Reopen only if P8/P10 establish a specific representation-data bottleneck.
- [ ] Freeze random-init, masked-reconstruction, and JEPA matched controls.
- [ ] Implement leakage-safe spatial masks and feature-support guards.
- [ ] Compare on identical folds, compute, seeds, and independent phase tests.
- [ ] Adopt only for reproducible deterministic transfer gain.

---

## 10. Posterior inference and VAC production

### P12 — Posterior calibration

**Status:** DEFERRED; GATED ON A FROZEN DETERMINISTIC WINNER/HYBRID

P12 is not on the pre-shutdown critical path. Run it only if the intended VAC claims
posterior uncertainty or class probabilities. Deterministic protocol selection and a
deterministic canary do not require FMPE/NPE.

1. Generate spatially out-of-fold embeddings or base predictions.
2. Fit FMPE/NPE on training phases.
3. Include ntilde(z) and response covariates directly in posterior conditioning.
4. Tune on ph006.
5. Evaluate once on ph001.

Require SBC, TARP, coverage, conditional coverage, knot-probability reliability, Brier
skill, width-versus-error, posterior contraction, and prior-dominated flags. Scalar
tempering that repairs average coverage while leaving shape failure is insufficient.

Progress checklist:

- [ ] Reopen only after a deterministic representation passes P10.
- [ ] Generate leakage-safe out-of-fold conditioning summaries.
- [ ] Fit on training phases and tune on ph006 only.
- [ ] Pass marginal, multivariate, conditional, tail, and information gates.
- [ ] Evaluate once on ph001 and freeze calibrated posterior artifacts.

### P13 — DESI canary and scale-out

**Status:** GATED ON P10 AND A FROZEN DETERMINISTIC WINNER; P12 ONLY FOR POSTERIOR COLUMNS

Reproduce the winning representation exactly:

- GraphNet: canonical DESI graph/global metrics, then graph patch views.
- U-Net: canonical count/response fields, then field patches.
- F-tier: canonical graph/fields plus converged FFT overlap and trim.
- Hybrid: all components with out-of-fold-compatible fusion semantics.

Run a golden mock and one DESI canary before full scale-out. Every eligible galaxy is
authoritative core exactly once. Overlapping contexts are not independent evidence.
Use the pre-registered deterministic de-duplication rule; if P12 is added, never
multiply overlapping posteriors.

Required flags include redshift support, graph/field support, boundary, mask hole,
extreme edge, completeness, OOD, and overlap disagreement. Add prior-domination and
posterior-information flags only when P12 posterior columns exist.

Progress checklist:

- [ ] Reproduce frozen representation and schemas in GraphWeb_DESI.
- [ ] Pass a truth-known golden mock canary end to end.
- [ ] Run one DESI canary and audit support, systematics, boundaries, and throughput.
- [ ] Freeze deterministic de-duplication and quality-bit semantics.
- [ ] Scale idempotent shards only after canary and P10 gates pass.
- [ ] Assemble, checksum, document, and collaboration-review the deterministic VAC.
- [ ] Add posterior columns only if P12 separately passes.

---

## 11. Evaluation and release gates

The controlling deterministic score is the mean across blocked folds of the equal-shell
macro R²(lambda1), computed from all authoritative validation-core galaxies. The
worst-shell result is a mandatory constraint; pooled R² is tertiary.

Every finalist also reports per-shell metrics for all eigenvalues; Spearman; MAE; bias;
slope; predicted/true variance; ordering violations; class confusion; balanced
accuracy; macro-F1; void/knot recall; source-to-transfer gap; residual spatial
correlation; and results versus halo mass, sampling density, degree, completeness, and
boundary.

Use spatial-block or phase-level uncertainty. Do not bootstrap galaxies independently.
Knot Brier skill, reliability, coverage, and information diagnostics apply only after
P12 produces a posterior; deterministic threshold decisions are reported as decisions.

Report classical comparisons as pooled, per shell, macro over tracer-supported shells,
and sparse-shell failure separately. A learned model does not beat classical merely
because classical becomes undefined in the sparsest shell.

A learned/hybrid primary deterministic estimator must improve source-calibrated DTFE
on a blind fresh phase with paired spatial uncertainty, avoid degrading
tracer-supported shells, and provide real sparse-shell skill. Calibration/information
gates are additional requirements only for posterior columns.

If no learned model robustly improves DTFE, use classical reconstruction as the
defensible deterministic primary option and badge learned outputs experimental.
A posterior uncertainty layer may be added later only if P12 validates it.

---

## 12. Shutdown-critical schedule: 2026-07-18 to 2026-07-21

P0 is complete. The remaining critical path is:

```text
P1b canonical full NGC+SGC catalogue [COMPLETE]
  + P2b graph/metrics [COMPLETE]
  -> P3a canonical fields + P4 geometry/fold draft [NOW]
  -> P4 final shared patch/support manifest
  -> P5/P6 parity [P7 joins when ready]
  -> P8 deterministic blocked-fold transfer
  -> frozen protocol bundle
```

The pre-shutdown success criterion is evidence that at least one serious encoder can
train and transfer under the patch protocol, or a clear, reproducible finding that none
yet does. It is not an FMPE posterior, HOD-marginalised distribution, JEPA model, or
full DESI VAC.

### Resource scheduling

- Use at most two interactive allocations simultaneously.
- Reuse allocation A for CPU/high-memory catalogue, graph, manifest, and evaluation
  work.
- Reuse allocation B for `rapids-gnn` graph metrics, GPU parity, and deterministic
  training.
- Release an allocation when it has no queued critical-path work.
- Use `cosmic_env` for normal preprocessing/training/evaluation and `rapids-gnn`
  for large graph-metric computation.
- Use the reusable `nersc-interactive-allocation` skill before every `salloc/srun`.
- Do not use `sbatch` for development; reserve it for validated production repetition.
- Do not let an optional F-tier, JEPA, posterior, or HOD job occupy a slot needed by
  P1–P8.

### Wave 0 — July 18: freeze contracts and inputs

1. Freeze the deterministic target baseline: linear increments.
2. Freeze the primary metric and checkpoint rule from P8.3.
3. Complete P1 for the current staged LSS catalogue:
   - row/target/ID audit;
   - target convention and units;
   - catalogue/observer/phase identifiers;
   - immutable manifest and hashes.
4. Start P0S without moving files:
   - scratch-only staged-script inventory;
   - storage classification;
   - `cosmic_env` and `rapids-gnn` export specification;
   - dry-run migration manifest.
5. Select the phase required for the target-generation benchmark. Do not create HOD
   variants.

**Exit:** `CATALOGUE_COMPLETE`, frozen target/metric JSON, P0S draft manifest.

### Wave 1 — July 18–19: canonical representations and patches

P1b and P2b are complete. Run P3a and the geometry-only part of P4 in parallel;
finalize P4 support fields after P3 validation:

- P2: **complete** for the authoritative full NGC+SGC catalogue;
- P3: canonical count/response fields in `cosmic_env`;
- P4 draft: fixed-comoving cores and five super-block folds from catalogue geometry;
- P4 final: attach graph, convolutional, and FFT support after P2/P3 validation.

Do not recompute graph metrics inside patches. Patches are views of one canonical
per-catalogue graph/field representation.

**Exit:** `GRAPH_COMPLETE`, `FIELD_COMPLETE`, immutable P4 manifest, support atlas,
core-size decision, hashes.

### Wave 2 — July 19: adapter parity

1. Implement P5 GraphNet core/context extraction with exact K-step dependency support.
2. Implement P6 U-Net core/context field extraction.
3. Run full-representation versus patch-view parity on identical authoritative cores.
4. Begin P7 only if resources remain; F-tier must pass graph, field, and FFT convergence.
5. Record memory/size buckets and failure cases. Never truncate nodes or edges silently.

No scientific training begins for an encoder until its own parity gate passes.

**Exit:** P5 and/or P6 parity report; optional P7 convergence report; executable patch
loader smoke tests.

### Wave 3 — July 20: deterministic protocol screen

For every parity-passing encoder:

1. Train linear-increment deterministic heads with one seed on two blocked folds.
2. Checkpoint on complete-fold macro R²(lambda1), never pooled or per-patch R².
3. Report every mandatory P8.3 safeguard.
4. Compare against the frozen current learned baseline and source-calibrated DTFE/CIC
   on the same authoritative cores.
5. Promote only candidates with stable execution and a credible fresh-region gain.

If the linear baseline passes and time remains, run the literal log-gap ablation on the
single leading encoder with identical folds/seed/update budget. Do not run the
15-component derivative target, raw/softplus sweep, posterior head, or JEPA here.

**Exit:** two-fold deterministic comparison and explicit GO/NO-GO per encoder.

### Wave 4 — July 20–21: strengthen and freeze

1. Extend promoted candidates to all five blocked folds.
2. Use three seeds only for candidates within 0.03 of the leader or with unique
   physical value; do not delay a complete one-seed fold result for premature seeds.
3. Freeze the deterministic winner/set, transforms, target contract, preprocessing,
   metric implementation, and acceptance criteria.
4. Complete the one-phase P10 target-generation cost benchmark and write the
   independent-phase schedule. Do not open blind metrics.
5. Complete P0S manifests and environment specifications; review destinations before
   any later copy.
6. Reproduce one canary patch from the frozen bundle.

### July 21 freeze bundle

Freeze a **deterministic protocol-ready development bundle** containing:

- canonical catalogue, graph, field, and patch manifests;
- shared folds and authoritative core IDs;
- passing adapters and parity/convergence reports;
- linear-increment target/scaler contract and any approved log-gap result;
- complete blocked-fold deterministic predictions and metric tables;
- frozen checkpoints and configs;
- independent-phase generation cost/schedule;
- P0S migration manifest plus `cosmic_env`/`rapids-gnn` specifications;
- provenance, checksums, and reviewed CFS/HPSS destination plan.

A same-phase P8 pass is promising transfer evidence, not final DESI validation.
A production generalisation claim requires P10 blind independent-phase success.
P12 is required only if the released product claims posterior uncertainty or
probabilities.

---

## 13. Longer schedule by dependency

| Stage | Earliest start | Typical duration | Exit artifact |
|---|---|---:|---|
| P0 evidence/inventory | immediate | 0.5–1 d | evidence + asset JSON |
| P0S preservation manifest | P0, parallel | 0.5 d metadata | migration + environment specs |
| P1 catalogue alignment | P0 | 0.5–1 d/catalogue | immutable raw catalogue |
| P2 graph/metrics | P1 | 1–3 d first catalogue | GRAPH_COMPLETE |
| P3 canonical fields | P1 | 0.5–1.5 d | FIELD_COMPLETE |
| P4 spatial manifest | P1, finalize after P2/P3 | 0.5–1 d | shared folds/cores |
| P5/P6 adapters | P2/P3/P4 | 1–2 d each, parallel | parity reports |
| P7 F-tier adapter | P2/P3/P4 | 2–3 d | FFT convergence report |
| P8 showdown | adapter gates | 2–4 GPU d + seeds | blocked-fold ranking |
| P9 hybrids | P8 | 1–3 GPU d if justified | complementarity decision |
| P10 phases | benchmark | days–weeks | blind phase report |
| P11 JEPA | after P8/P10 if justified | 2–5 GPU d | optional JEPA decision |
| P12 posterior | after frozen deterministic winner | 2–5 GPU d | optional calibration report |
| P13 DESI deterministic | P10 + frozen winner | deployment dependent | golden canary + point VAC shards |
| P13 posterior columns | P10/P12 | deployment dependent | calibrated posterior VAC shards |

---

## 14. Artifact and run discipline

Keep source code, configuration, schemas, decisions, compact evidence, and
reproducibility specifications in Git/home. Use a versioned scratch root resolved
through `shared/config_paths.py` for active catalogues, graphs, features, fields,
manifests, patches, runs, and evaluations—but never treat scratch as durable source.

After P0S review:

- copy irreplaceable reusable catalogues, scalers, selected checkpoints/predictions,
  canaries, and release bundles to CFS;
- archive large expensive-to-rebuild density, T-Web, graph, and staged-mock bundles in
  HPSS;
- retain checksums and producing manifests in Git;
- do not delete or overwrite scratch sources until destination checksums pass.

Every stage writes:

- configuration JSON;
- input hashes;
- Git SHA;
- environment name and environment-specification hash;
- interactive allocation or production job ID/resources;
- row/node/edge counts;
- validation report;
- completion manifest written only by the authoritative process.

The environment contract explicitly includes both `cosmic_env` and `rapids-gnn`;
the latter is required for large graph-metric construction. Jobs must be idempotent and
resumable. Evaluation/finalization must not write a trainer's completion marker. No
storage migration is authorized merely by this plan; P0S first produces a reviewed
dry-run manifest.

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
- Controlling objective: transferable deterministic inference under the patch protocol.
- Primary metric: mean spatial-fold equal-shell macro R²(lambda1); pooled R² is tertiary.
- Target baseline: linear eigenvalue increments.
- Literal log gaps: one bounded post-baseline ablation.
- Fifteen-component gradient/Laplacian target: **DEFERRED** off the shutdown critical path.
- New HOD variants and HOD marginalisation: **DEFERRED** until spatial/phase transfer passes.
- Required environments: `cosmic_env` and `rapids-gnn`.
- Deterministic selection before posterior fitting.
- sqrt(N) shell objective.
- ph001 sealed blind.
- RA 200–240 is development evidence, not blind.
- Two-pass GraphNet transfer fix: **CLOSED—FAILED**.
- Broad random-split architecture search: **CLOSED**.
- GBM feature gate: **CLOSED**.
- Fifteen-feature aperture model: **CLOSED—FAILED**.
- JEPA: **DEFERRED/GATED** on deterministic generalisation and multi-catalogue data.
- FMPE/NPE: **DEFERRED/GATED** on a frozen deterministic winner; required only for
  posterior VAC claims.
