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

The fixed 7-Mpc/h target is frozen. The existing smoothing study closes replacing it
with a coarser target merely because bulk lambda1 R² peaks near 10 Mpc/h: cluster
completeness and mass-anchored massive-halo recovery both worsen with additional
smoothing, while 7 Mpc/h preserves compact collapsed structure. This closes
**target-scale retuning**, not the use of multiscale input features or auxiliary
supervision.

NEXUS+ is not another value of the T-web smoothing parameter. It combines
scale-normalized density-Hessian morphology signatures across a bank of log-Gaussian
smoothing scales and can return web significance and a locally dominant scale. Those
quantities are not the fixed-scale gravitational-potential Hessian eigenvalues or
tensor required by the VAC. Therefore:

- do not replace the 7-Mpc/h primary target with NEXUS+;
- use true-matter NEXUS+ first as an evaluation-only residual stratifier by morphology,
  signature strength, and dominant scale;
- open one auxiliary-head comparison only if that diagnostic exposes a reproducible
  morphology- or scale-dependent failure after P10;
- never use a true-matter NEXUS+ field as a production input; an observed-galaxy
  NEXUS+ feature would require its own mask, RSD, sparsity, and random-response closure.

The bounded auxiliary comparison, if opened, is current U-PATCH versus multiscale
T-web heads at 6/7/10 Mpc/h versus NEXUS+ signature/scale heads, all retaining only
the 7-Mpc/h eigenvalue or tensor output for production. Promotion requires fresh-phase
gain with no rare-knot, supported-shell, or boundary regression. The primary NEXUS
literature establishes multiscale simulation-field segmentation; it does not establish
per-galaxy recovery of this project's target under DESI observation operators.

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

**Status:** NEAR-COMPLETE (2026-07-21) — env specs exported, scratch-only `ph000` source
preserved to git (0 checksum mismatches), HPSS bulk archive verified (99 archives, 0 fail);
only reviewed CFS copies remain. Authoritative record: `docs/evidence/p0s/MIGRATION_MANIFEST.md`.
Original planning checklist retained below for reference.
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

#### P3b — Random-reference and deployable-response upgrade

**Status:** DEFERRED; does not modify the frozen P3a/P8 controls

A random catalogue is a high-density, unclustered Monte Carlo sampling of the survey
selection measure. It is not a set of negative galaxies, an estimate of the missing
matter, or an independent universe.

For each observation view `s`, deposit data and matched randoms on the same immutable
cap lattice with the same coordinate convention and kernel. The preferred project
construction is

~~~text
G_s(v)       = weighted observed galaxy count in voxel v
p_s(v)       = R_s(v) / R_base(v), after a manifest-frozen support regularisation
mu_s(v)      = ntilde_s(z_v) * V_voxel * p_s(v)
contrast_s(v)= log((G_s(v) + epsilon) / (mu_s(v) + epsilon))
~~~

Here `R_base` represents geometrically available support and `R_s` represents support
after the view's footprint, targeting, fibre, and redshift-success response. Reusing
the same base-random IDs across views is preferred because paired differences then do
not contain avoidable random-catalogue Monte Carlo noise. If an audited 3-D random
catalogue already samples the complete angular and radial selection, the equivalent
form is `mu_s(v) = alpha_s * R_s(v)`, where `alpha_s` is fitted only over a
manifest-frozen catalogue/tracer/cap normalisation domain—never independently inside
each patch.

Do not blindly use a clustering-random redshift histogram as `ntilde(z)`: random
redshifts may have been sampled from or inherited from the data and can absorb radial
large-scale structure. Retain the separately frozen smooth `ntilde(z)` unless the
random-redshift provenance and intended measure are audited. Ordinary footprint
randoms also do not by themselves encode density-dependent fibre collisions or
redshift failures; use matched assignment probabilities, PIP/completeness products,
or quality maps for those effects.

Minimum architecture contract:

- U-PATCH receives `G_s`, `mu_s` or its logarithm, stabilized contrast,
  random-derived exposure/support, completeness, boundary distance, and LOS as voxel
  channels. The first drop-in P3b arm should change only the P3a
  occupancy-derived exposure/reference field while preserving the frozen three-channel
  backbone; additional channels are a separately named challenger.
- G-PATCH keeps galaxies as message-passing nodes and interpolates the response fields
  at each galaxy. The first arm appends expected intensity, completeness/support, and
  boundary distance; it may add `d_ij / ell_s`, with
  `ell_s proportional to mu_s^(-1/3)`, alongside—not instead of—the physical edge
  length. Millions of random nodes are a later ASTRA-style ablation.
- F-PATCH may use response channels while reconstructing density, but only the
  reconstructed physical density contrast enters the fixed FFT tidal operator.

P3b gates:

- every stage/view has a matching response product or a documented factorization from
  common base randoms and stage-specific probabilities;
- random-only stabilized contrast is consistent with zero after the frozen
  normalization;
- results are stable across random seeds/densities, with random Monte Carlo noise
  subdominant to galaxy sampling noise;
- response fields use no targets, phase/split ownership, true matter, or local
  patch-wise renormalization;
- data, random, response, and topology hashes are view-specific;
- mock and DESI response columns have one deployable schema and units;
- raw counts, expected counts, and information support remain separate—randoms correct
  the expected selection baseline but do not recreate lost galaxies or remove shot
  noise.

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

**Status:** TWO-ROTATION RECOVERY/EXTENSIONS COMPLETE; U-CIC CLOSED; MT4 COMPLETE
WITH INFORMATION PASS / ENCODER NO-GO; P8.9 NEXT (2026-08-09)
**Duration:** recovery estimate follows the exposure audit; three-seed finalists later

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

#### P8 short-screen result and adequacy correction

The pre-registered rotations 0 and 2 are complete for the runnable learned models.
All values below are provisional means across two short validation screens for lambda1.

| Candidate | Four-shell macro R2 | First-three-shell diagnostic R2 | Worst/final-shell R2 | Decision |
|---|---:|---:|---:|---|
| CLASSICAL-CIC | 0.185 | **0.520** | -0.822 | reference; mechanically collapses in shell 4 |
| G-PATCH | **0.396** | 0.448 | **0.240** | provisional learned leader |
| U-PATCH | 0.363 | 0.423 | 0.183 | provisional short-screen result |
| F-PATCH v2_A | -- | -- | -- | resource NO-GO before training |

The four-shell macro advantage of G-PATCH and U-PATCH is not a win over classical
reconstruction: both learned models trail CIC in every one of the first three shells,
and their positive macro delta is created by CIC failing in the sparsest shell. These
numbers therefore emphasize the need for a generalisable encoder; they do not establish
that either learned model beats classical reconstruction.

The post-run optimization audit invalidates a scientific stop decision from these jobs.
Each job ran 2,000 randomly sampled patch steps but had 10,262--10,351 eligible training
cores. Exact replay of the frozen sampler finds only 1,554--1,560 unique cores were seen
(15.07--15.14%), representing 40.08--40.86% of the weighted sampling mass. The sparse
high-redshift shell saw only 103--121 unique cores out of 3,845--3,848 eligible cores.
Each run performed one complete-fold validation, at its terminal step; no learning curve,
plateau, or early-stopping decision exists. The jobs lasted approximately 6.6--13.3
minutes each, far below the registered 2--4 GPU-day screen envelope.

Consequently, the existing outputs are frozen as plumbing and optimization smoke evidence.
They justify no automatic five-fold promotion, but they cannot support a scientific
learned-model NO-GO. P8 remains open until the exposure-aware recovery below is complete.
F-PATCH v2_A remains a resource NO-GO for that exact configuration only.

The exact full-cap piecewise-linear DTFE row remains useful for characterising the
production reference and is required before the adoption gate closes. The independently
frozen P0 exact-DTFE evidence remains a contextual approximately 0.55 pooled deployment
bar, not a matched P8 result.

There is also a support mismatch that must be isolated. The learned estimators receive
finite patch context, whereas CLASSICAL-CIC reconstructs the full-cap field and applies a
global FFT tidal solve. Because the traceless tidal shear is nonlocal, an apparent
learned deficit may combine encoder error with missing long-wavelength information. This
does not weaken the classical adoption bar; it motivates a matched classical-anchored
residual test and an explicit global-versus-local tidal-support diagnostic.

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

For shell s, use `w_i = N_s^(-1/2)`. For patch p, define
`W_p = sum_core w_i`. In the exposure-aware recovery, construct a seeded
probability-proportional-to-`W_p`, without-replacement order and visit every eligible
core exactly once. Optimize `sum_core(w_i loss_i) / mean_p(W_p)` at each patch step,
so the arithmetic mean over one complete epoch is exactly the global row-weighted
objective. The objective is therefore independent of patch subdivision. Log actual
optimization exposure by shell and patch.

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

The two registered development rotations are fold-role rotations, not coordinate or
data-augmentation rotations:

- rotation 0: train folds `{2,3,4}`, validate fold `1`, keep fold `0` sealed as the
  development-test fold;
- rotation 2: train folds `{0,1,4}`, validate fold `3`, keep fold `2` sealed as the
  development-test fold.

Fold 4 is common training geography; the other roles move to test transfer across a
different part of the canonical full-cap catalogue. Neither rotation is an
independent-phase test; that remains P10.

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

#### P8.5 Exposure-aware recovery and support matching

Complete this recovery before interpreting G-PATCH or U-PATCH scientifically:

1. Replace step-based replacement sampling with complete exposure-aware patch epochs.
   Every eligible training core must be visited once per epoch in a seeded weighted
   permutation. Preserve the frozen square-root shell objective through explicit core
   loss weights; patch subdivision must not change the scientific objective.
2. Train for a minimum of five complete patch epochs and a maximum of twenty. Evaluate
   the complete blocked validation fold after every epoch. Early stopping is permitted
   only after epoch 5, with patience 3 and a registered minimum improvement of 0.005 in
   complete-fold macro-R². If the curve is still improving at epoch 20, record
   `NOT_CONVERGED` rather than selecting the terminal checkpoint.
3. Persist, by epoch and shell: eligible cores, unique cores seen, weighted exposure,
   repeats, loss numerator/denominator, validation metrics, wall time, and peak memory.
   A scientific run must have 100% eligible-core exposure in each completed epoch.
4. Rerun rotations 0 and 2 for the existing G-PATCH and U-PATCH controls before changing
   architecture, target parameterization, or loss. The short-screen checkpoints remain
   immutable and separately named.
5. Run a truth-field support diagnostic independent of ML. Starting from the true density
   field, compare central-core tidal tensors/eigenvalues obtained from the full field with
   solves using physical context radii of 60, 120, 180, 240, and 360 Mpc/h. Report trace
   and traceless-shear convergence separately by shell and boundary distance. This measures
   how much of the target is unrecoverable from the context offered to each patch model.
6. Build a support-matched physics-residual control. Supply every learned residual branch
   with the frozen, globally reconstructed train-calibrated CIC/DTFE tensor or eigenvalues;
   zero-initialize the residual so checkpoint zero reproduces the classical baseline
   exactly. The learned branch then has to add transferable information rather than relearn
   the nonlocal gravitational operator from finite context.
7. Complete the exact full-cap piecewise-linear DTFE row using the identical catalogue,
   split, target, calibration policy, and authoritative galaxies.
8. Close P8 only after converged G/U controls, the support diagnostic, the classical-
   residual control, and matched CIC/DTFE rows are available. Promote to five folds only
   if the converged result clears the registered gain/uncertainty rule. Record a learned
   NO-GO only if all converged learned and residual candidates fail the matched classical
   gate across folds rather than because of one shell's classical collapse.
9. Keep log-gap, JEPA, FMPE/NPE, HOD marginalisation, and broad architecture searches gated
   until this deterministic recovery answers whether optimization exposure and global
   physical support explain the short-screen deficit.

#### P8.6 Registered long-horizon convergence extension

The original recovery rule remains the immutable primary experiment. In rotation 0,
G-PATCH stopped at epoch 15 because the early-stopping comparator requires a gain of
at least 0.005 over the last qualifying score: epoch 12 scored 0.4663, while epochs
13--15 scored 0.4617, 0.4668, and 0.4682. Epoch 15 was the absolute best checkpoint,
but its gain of 0.0019 over epoch 12 did not reset patience; three consecutive stale
epochs therefore triggered the registered stop. This is correct execution of the
frozen rule, not evidence that the GNN had reached an optimization plateau. U-PATCH
reached the epoch-20 cap with its best score at epoch 20 and is also not demonstrably
converged.

To test the specific long-convergence hypothesis without rewriting the original result:

1. Complete the unchanged `recovery_v1` rotation-2 runs before using any longer-budget
   result for model selection. Never delete, resume with altered arguments, or relabel
   the original rotation-0/2 artifacts.
2. Name the continuation `convergence_extension_v1`. For each eligible parent run,
   initialize from its immutable `best_checkpoint.pt`; record its path, SHA-256,
   parent epoch, parent score, Git revision, and complete argument contract.
3. Reset the optimizer and scheduler explicitly. Use AdamW with learning rate `2e-4`,
   weight decay `1e-4`, gradient clipping at 5, and a cosine schedule over exactly 20
   additional complete-exposure epochs. This is a declared low-rate fine-tuning phase,
   not an in-place resume of the original optimizer trajectory.
4. Disable early stopping inside this 20-epoch extension. Validate the full fold after
   every epoch and retain every loss/validation curve. A noisy three-epoch window must
   not terminate the test that was created to measure slow convergence.
5. Preserve every other contract: model architecture, seed, folds, authoritative cores,
   global graph/field products, patch adapters, target/scaler, row-weighted objective,
   100% core exposure, and complete-fold primary metric. Continue deterministic epoch
   numbering from the parent epoch so the weighted patch permutations do not replay the
   parent's first epochs.
6. Run the rotation-0 G-PATCH and U-PATCH extensions first as development experiments.
   Apply the identical frozen extension to rotation 2 if either architecture improves
   its parent score by at least 0.005, or remains still improving at the extension cap.
   The architecture comparison uses the better of the immutable parent and extension
   checkpoints; fine-tuning is never allowed to erase a stronger parent result.
7. If the best extension score improves by less than 0.005 and the final five epochs
   contain no new best checkpoint, close the slow-convergence hypothesis for that
   architecture. If the best lies in the final three epochs, record
   `NOT_CONVERGED_EXTENSION_CAP`; do not call the terminal value converged.
8. Treat this as optimization diagnosis within `ph000`. It cannot replace rotation-2
   replication, the matched exact DTFE/global-residual controls, or P10 fresh-phase
   validation.

#### P8.7 Registered classical-anchored corrective model

The earlier P9 constant linear blends are complementarity diagnostics. They are **not**
trained corrective models and do not close P8.5 item 6. Register one primary corrective
model before opening any GraphNet/F-tier/hybrid sweep:

```text
U-CIC-RESID-v1:
  global full-cap counts -> CIC contrast -> fixed FFT tidal solve
                         -> train-fold affine eigenvalues
  local P3 fields       -> frozen-contract U-PATCH backbone -> bounded correction
  output                = corrected ordered eigenvalues
```

1. Re-run the frozen full-cap CIC workflow for rotations 0 and 2 and retain a
   `parent_node_id`-keyed classical prediction for every registered training and
   validation authoritative row. The affine response is fitted on training folds only;
   no test-fold truth enters the anchor.
2. Initialize the local field backbone from the corresponding converged
   `convergence_extension_v1` U-PATCH best checkpoint. Replace its point head with a new
   zero-initialized residual head and record the parent checkpoint path and SHA-256.
3. At checkpoint zero, require numerical reproduction of the CIC eigenvalues on a real
   patch. Abort if the maximum absolute difference exceeds `2e-6` or if any eigengap is
   non-positive.
4. Parameterize the correction as an additive bounded lambda1 shift and multiplicative
   positive eigengap shifts:

   ```text
   lambda1_hat = lambda1_CIC + sigma_train(lambda1) tanh(r1)
   gap12_hat   = gap12_CIC exp(1.5 tanh(r2))
   gap23_hat   = gap23_CIC exp(1.5 tanh(r3))
   ```

   This preserves ordering and prevents the residual from silently replacing the global
   baseline with an unconstrained second estimator. A `1e-6` gap floor is permitted only
   if the manifest reports how many anchor rows it affects.
5. Preserve the existing 64 Mpc/h core, 120 Mpc/h U-PATCH halo, selection-aware P3
   channels, P4 rotations, authoritative rows, target scaler, square-root shell weights,
   complete-exposure epoch sampler, complete-fold validation, and atomic resume contract.
6. Run seed 42 on rotations 0 and 2 for 20 complete-exposure epochs at learning rate
   `2e-4`, with a cosine schedule, no within-run early stopping, and full validation after
   every epoch. If the best checkpoint lies in the final three epochs, register a later
   extension rather than silently extending this screen.
7. Compare exactly matched rows against both components: train-affine CIC and standalone
   U-PATCH. Report the four-shell primary score, first-three-shell diagnostic, each shell,
   pooled metrics, class metrics, boundary dependence, and spatial-block intervals.
8. Adopt the corrective model only if it improves the supported-shell mean over U-PATCH
   without degrading the sparse shell by more than 0.01. The ordinary P8 `+0.03`
   promotion target remains decisive; smaller gains are provisional and require spatial
   uncertainty plus P10 replication.
9. Do not launch G-CIC, F-CIC, learned tensor residuals, or density-residual branches
   unless U-CIC-RESID-v1 demonstrates a reproducible gain or its residual diagnostics
   identify a specific representation failure worth testing.

The exact DTFE build may proceed in parallel. Its accelerated implementation must use
voxel-centric containing-tetrahedron point location, exact barycentric interpolation,
resumable cap/slab outputs, explicit unresolved-voxel coverage, and a synthetic parity
test against SciPy Delaunay. It must not fall back to a vertex splat while retaining the
DTFE label.

##### P8.7a Residual-bound feasibility correction

The original `U-CIC-RESID-v1` hard `+/-1 sigma_train(lambda1)` correction is a
registered feasibility NO-GO, not a completed accuracy screen. A machine-readable audit
at
`/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/classical/residual_bound_feasibility.json`
shows that the absolute train-fold CIC residual has a 99th percentile of `2.602` and
`2.553 sigma` in rotations 0 and 2. More decisively, even an oracle validation correction
subject to the one-sigma bound can reach only `R2_lambda1 = 0.281/0.196` in the sparse
shell, below the standalone U-PATCH scores `0.345/0.321` and therefore below the existing
no-degradation gate. No amount of optimization can make v1 eligible for adoption.

Freeze the three completed v1 epochs as `STOP_FEASIBILITY_NO_GO`; do not resume or
reinterpret them. Register `U-CIC-RESID-v2` before opening any further scores:

1. Preserve every v1 input, fold, seed, backbone checkpoint, optimizer, schedule,
   objective, logging, checkpoint, and evaluation contract.
2. Change only `lambda1_max_sigma: 1 -> 3`, using the ceiling of the largest
   rotation-level train-fold 99th percentile. This choice uses no validation score.
3. Retain zero-initialized exact CIC parity and the `exp(1.5 tanh(r))` positive eigengap
   corrections.
4. Use the separately named artifact root `u_cic_resid_v2`; never resume v1 weights or
   optimizer state.
5. Run the original 20 complete-exposure epochs with no within-run early stopping and
   reapply the unchanged supported-shell, sparse-shell, `+0.03`, uncertainty, and P10
   gates. The wider bound makes the registered question feasible; it does not weaken
   the adoption rule.

##### P8.7b U-CIC-RESID-v2 final closeout

**Status:** COMPLETE — `NO_GO_SPARSE_SHELL_REGRESSION` (2026-08-06)

Both registered 20-epoch screens completed. The result is not a marginal promotion:

| Rotation | Model | macro R2(lambda1) | first-three-shell macro | sparse-shell R2 |
|---|---|---:|---:|---:|
| 0 | U-PATCH-BRIGHT | 0.5070 | 0.5609 | 0.3453 |
| 0 | U-CIC-RESID-v2 | 0.5220 | 0.5933 | 0.3082 |
| 2 | U-PATCH-BRIGHT | 0.5197 | 0.5860 | 0.3210 |
| 2 | U-CIC-RESID-v2 | 0.5208 | 0.6033 | 0.2734 |

The corrective model gains only `+0.0150/+0.0011` in the registered macro score,
below the `+0.03` promotion target. More importantly, it loses
`-0.0371/-0.0476` in the sparse shell, far beyond the permitted `-0.01`.
Supported-shell gains cannot compensate for worsening the shell that motivated the
correction. Freeze all U-CIC variants and do not open G-CIC/F-CIC branches.

The machine-readable decision is
`docs/evidence/p8/ucic_v2_closeout.json`. Preserve standalone U-PATCH as
`U-PATCH-BRIGHT_REFERENCE`: the current leading learned Bright-only candidate under
the two-rotation P8 screen, not yet a production-approved VAC model. Five-fold/seed
replication and P10 fresh-phase transfer remain mandatory before production promotion.

#### P8.8 BGS_FAINT multitracer information gate

**Status:** MT1--MT3 COMPLETE; MT4 PROXY ROTATIONS 0/2 AND NULL ROTATION 0 COMPLETE;
NULL ROTATION 2 RUNNING; MT5 TECHNICALLY READY BUT FULL TRAINING GATED; NO MODEL PROMOTED

Implementation checklist (updated 2026-08-08):

- [x] Freeze the causal contract: BGS_BRIGHT alone owns supervision, evaluation, and
  catalogue output; BGS_FAINT is context-only in the first information screen.
- [x] Implement response-explicit Oracle and Proxy catalogue builders, with immutable
  truth/sky columns joined from unique `inputs/targ.fits` rows rather than repeated
  alternate-tile spectroscopic rows.
- [x] Stamp and audit the Oracle and Proxy catalogue products under
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/catalogues/`.
- [x] Implement tracer-separated Faint CIC/response overlays and independent Faint
  full-cap radial selection/normalization fits; preserve frozen Bright P3/P6 products.
- [x] Stamp the field and selection manifests under
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/fields/` and `selection/`.
  The exact incomplete-stencil exclusions and lost-weight audit are stored in
  `fields/grid_support_audit.json`; the voxel branch excludes only 4 NGC and 21 SGC
  Oracle edge rows (and the corresponding Proxy rows), while the graph branch retains
  them.
- [x] Implement the response-aware global Bright+Faint graph, P5-compatible adapter,
  and ten-feature GraphNet contract with globally computed graph metrics.
- [x] Stamp the PHOTSYS-marginal Proxy graph, cuGraph metrics, adapter, and feature
  transform under
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/graph/bf_proxy_response_v1_photsys_marginal/`.
  The disconnected NGC+SGC product contains `12,771,280` nodes, `92,674,381`
  Delaunay edges, `79,902,716` tetrahedra, and `299,551,997` union pairs. The
  production-shape adapter is exact on CPU to `4.77e-7`; GPU prediction/subdivision
  differences of `5.66e-4/1.13e-3` are retained as an amber round-off diagnostic,
  not relabelled as exact parity.
- [x] Implement the six-channel U-PATCH and ten-node-feature G-PATCH canary entrypoints,
  both with Bright-only target/loss ownership.
- [x] Complete the Oracle U-PATCH canary under
  `models/u_patch/bf_oracle_assigned_v1/rotation_0/seed_42/canary_steps100/`.
- [x] Complete the PHOTSYS-marginal Proxy U-PATCH canary under
  `models/u_patch/bf_proxy_response_v1/rotation_0/seed_42/canary_photsys_marginal_steps100/`;
  it covers all 999,683 Bright validation targets and is a technical diagnostic only.
- [x] Complete the PHOTSYS-marginal Proxy G-PATCH canary under
  `models/g_patch/bf_proxy_response_v1/rotation_0/seed_42/canary_photsys_marginal_steps100/`;
  its macro `R2_lambda1=0.27045` is a loader/optimization diagnostic, not a promotion result.
- [x] Complete MT2 information diagnostics and MT3 matched classical controls. The
  tracked summary is `docs/evidence/p8/multitracer_mt3_summary.json`; the information
  audit is `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/diagnostics/bf_proxy_response_v1/information_audit.json`.

- [x] Implement the neural Faint-position null at commit `6b80744`. The loader changes
  only Faint counts/derived density while retaining real Proxy exposure, selection,
  normalization and every Bright input/target. The full-cap smoke test proves exact
  identity of the retained channels and is executable as
  `workflows/abacus_tweb/p8_smoke_multitracer_null.py`.
- [x] Complete Proxy rotation 2 and Null rotation 0 from immutable commit `6b80744`.
- [x] Complete the Null rotation-2 causal replication and freeze the MT4 decision in
  `docs/evidence/p8/multitracer_mt4_decision.json` and runtime marker
  `models/recovery/mt4_closeout/MT4_UPATCH_MULTITRACER_DECISION`.

Completed preflight interpretation (2026-08-07):

- MT2 raises high-redshift occupied-voxel fractions from `1.18% -> 4.87%` in NGC
  and `1.08% -> 4.33%` in SGC and reduces mean separation from `42.30 -> 25.56`
  and `43.45 -> 26.47 Mpc`; this proves more measurements, not better inference.
- Combined Bright+Faint CIC reaches macro `R2_lambda1=0.4591/0.4470` on rotations
  0/2, but the angularly scrambled-Faint null already reaches `0.4449/0.4325`.
  The genuine-position increment is only `+0.0143/+0.0145`.
- Density-matched mixed controls average only `0.1024/0.1082`; per tracer, the
  Bright population remains more informative. Any neural Proxy gain must therefore
  survive a matched `U-BF-NULL-v1` run before being attributed to extra structure.
- A representative one-core U-PATCH overfit reduces the loss-window mean from
  `0.37727` to `1.0148e-4`. The interrupted-and-resumed epoch canary visits all
  `10,351/10,351` eligible cores exactly once, with zero repeats, all shells, and
  complete validation; its tracked summary is
  `docs/evidence/p8/multitracer_mt4_canary_rot0_summary.json`.
- The first MT3 evaluation correctly withheld completion after tiny affine-induced
  eigenvalue crossings. Commit `4556317` restores order after train-fitted affine
  calibration; all eight gates then pass without using validation truth.
- `U-BF-PROXY-v1` rotation 0 completes at epoch 20 with macro
  `R2_lambda1=0.60737` and shell scores `0.70792/0.65667/0.58036/0.48452`.
  This passes the registered replication trigger but remains causally uninterpretable
  until the matched neural Faint-position null completes.
- `U-BF-PROXY-v1` rotation 2 completes at epoch 20 with macro
  `R2_lambda1=0.53941` and shell scores `0.61297/0.59320/0.52131/0.43016`.
- Against converged Bright U-PATCH, Proxy gains `+0.10036/+0.01967` macro and
  `+0.13918/+0.10913` in the sparse shell on rotations 0/2. The two-fold mean gain
  is `+0.06002`, but rotation 2 misses the `+0.03` macro gate and degrades shell 2
  by `-0.01522`, beyond the allowed `-0.01`.
- `U-BF-NULL-v1` rotation 0 completes at epoch 20 with macro
  `R2_lambda1=0.47533` and shells `0.55735/0.52522/0.44922/0.36952`.
  Proxy minus Null is `+0.13204` macro and is positive in every shell
  (`+0.15057/+0.13145/+0.13114/+0.11500`). The real Faint positions therefore
  carry recoverable tidal information for this encoder on rotation 0; the large Proxy
  result is not a generic extra-occupancy or radial-selection shortcut.
- `U-BF-NULL-v1` rotation 2 reaches macro `R2_lambda1=0.51874`, only `-0.00100`
  from Bright-only. Proxy minus Null remains positive in every shell
  (`+0.00968/+0.01440/+0.01953/+0.03908`) and is `+0.02067` macro.
- Across the two rotations, Proxy minus Null averages `+0.07636` macro and
  `+0.08012/+0.07292/+0.07533/+0.07704` by shell. The sign therefore replicates,
  establishing additional Faint spatial information within `ph000`, while the effect
  size remains strongly geography dependent.
- Proxy fold spread is `0.06795`. Rotation 2 still misses the macro gate and degrades
  a supported shell by more than `0.01`, so current-encoder adoption is a NO-GO.
- Proxy rotation 2 and Null rotation 0 both finish at the epoch cap with learning rate
  zero. The mechanical `NOT_CONVERGED_MAX_EPOCHS` markers do not authorize extension:
  the registered rule explicitly requires non-zero remaining learning rate.

The frozen scientific result is a same-phase, two-rotation multitracer information PASS,
not a production encoder promotion. No fresh-phase or DESI generalisation claim follows.
The MT4 information trigger technically opens field/physics challengers, but P10.0 keeps
new `ph000` architecture sweeps closed: P8.9 `U-DENSITY-PHYS-v1` is the next registered
model experiment.

This evidence opens Proxy rotation 2 and the matched Null rotation 0. It does not open
full G-PATCH training, model adoption, or a claim that Faint traces additional cosmic
structure.

The active entrypoints are `workflows/abacus_tweb/p8_build_multitracer_catalogues.py`,
`p8_build_multitracer_fields.py`, `p8_refit_multitracer_selection.py`,
`p8_build_multitracer_graph_adapter.py`, `p8_prepare_multitracer_graph_features.py`,
`p8_build_multitracer_control_fields.py`, `p8_evaluate_multitracer_controls.py`,
`p8_train_multitracer_unet_patch.py`, `p8_train_multitracer_graph_patch.py`,
`p8_train_patch_recovery.py`, `p8_smoke_multitracer_null.py`, and
`run_p8_multitracer_recovery_supervisor.sh`.

The current plateau and the sparse-shell behaviour motivate a data-information test
before another encoder sweep. This branch asks whether an additional observed tracer
improves inference of the same Bright-galaxy targets. It does not change the target,
P4 folds, metric, smoothing scale, or authoritative evaluation rows.

##### P8.8.1 F0 feasibility audit — completed evidence

The executable audit is
`workflows/abacus_tweb/p8_multitracer_feasibility.py`; seven unit tests are in
`tests/phase4/test_p8_multitracer_feasibility.py`. Runtime evidence is
`/pscratch/sd/d/dkololgi/abacus/p8_multitracer_feasibility_v1/feasibility_audit.json`
with marker `F0_FEASIBILITY_COMPLETE`; the tracked digest and summary are in
`docs/evidence/p8/multitracer_f0_summary.json`.

The current staged construction is:

```text
SecondGen CutSky BGS
  -> upstream_prepare_mocks_Y3_bright.py
       Bright: R_MAG_APP < 19.5
       Faint proxy: 19.5 <= R_MAG_APP <= 20.175
       retain 69.5% of Faint with np.random.uniform
       promote 20% of retained Faint to high priority
  -> forFA0.fits (Bright + Faint proxy)
  -> multipass fibre assignment (both target types present)
  -> run_path1_mkcat.sh --tracer BGS_BRIGHT
       BGS_BRIGHT full LSS catalogue only
  -> Bright-only LOA spectroscopic injection
  -> build_mock_bgs_maglim_catalog.py default Bright-bit cut
  -> 9,538,254-row GraphWeb Bright parent
```

The exact audited unique-target counts are:

| Product | Bright unique | Faint unique | Interpretation |
|---|---:|---:|---|
| `forFA0.fits` | 10,547,983 | 7,559,142 | pre-FA target population |
| assigned-only TARGETID crossmatch | 8,314,754 | 3,601,878 | matched unique targets; 3,654,393 assigned IDs require separate provenance resolution |
| spectroscopic join | 9,920,755 | 7,110,292 | repeated-row product de-duplicated by TARGETID |
| final GraphWeb catalogue | 9,538,254 | 0 | Bright-only cut confirmed |

The upstream Faint population contains `553,830` unique targets in
`0.45 <= z < 0.55`; the spectroscopic-join product contains `150,919` unique Faint
IDs in that shell. This is enough to make an information-content experiment
worthwhile. It is not evidence that those rows already form a production-valid Faint
catalogue.

Three limitations are blocking:

1. The CutSky input exposes `R_MAG_APP` and `G_R` colours but not the `r_fiber`, `z`,
   and `W1` photometry used by the final DESI BGS_FAINT fibre-magnitude/colour
   selection. The current 0.695 random retention is a density proxy, not the final
   selection.
2. The current random Faint retention has no explicit RNG seed. The on-disk `forFA0`
   target IDs are now a fixed realization, but rerunning the source script is not
   bitwise reproducible. Never silently redraw this population.
3. `inject_loa_spec_from_zall.py` calibrates marginal spectroscopic success using
   `BRIGHT_BITS` only. The current Faint rows therefore lack a separately validated
   redshift-success response.

The official target-selection contract is more specific than the mock proxy:
BGS_FAINT uses `19.5 < r < 20.175` plus an `r_fiber` and
`(z-W1)-1.2(g-r)+1.2` selection to retain high redshift efficiency. Twenty per cent
of Faint targets receive high fibre priority; the rest receive lower priority. This
is why simply removing `--bright-only` from the final exporter is not acceptable.

##### P8.8.2 Frozen scientific question and labels

The first multitracer experiment is context-only:

```text
input context = observed BGS_BRIGHT + observed BGS_FAINT-proxy
supervised/evaluated rows = the unchanged 9,538,254 Bright parent
target = unchanged ordered R=7 Mpc/h tidal eigenvalues
metric/folds = unchanged P4/P8 contract
```

Faint galaxies must not contribute labels in the first screen. This cleanly separates
additional tracer information from additional supervision. A later Faint-supervised
experiment is allowed only after context-only performance and Faint truth linkage pass.

##### P8.8.3 MT1 — construct response-explicit multitracer catalogues

Build three separately named products; never let an oracle product become a production
input by path aliasing.

1. `BF_ORACLE_ASSIGNED_v1` — information upper bound.
   - Start from the immutable `forFA0` TARGETID/BGS_TARGET realization.
   - Join the existing multipass fibre-assignment records by TARGETID.
   - Retain one row per assigned target with simulated RSD redshift and truth linkage.
   - Do not impose redshift failure.
   - Mark every row and manifest `ORACLE_REDSHIFT`; this product may answer whether
     Faint tracers contain useful information, but can never support a DESI claim.
2. `BF_PROXY_RESPONSE_v1` — development-realistic proxy.
   - Preserve the same fixed Faint target IDs and actual multipass assignment outcome.
   - Fit separate Bright and Faint spectroscopic-success models from DESI zcatalog
     target bits. At minimum retain tracer type, apparent magnitude, cap, and available
     exposure/completeness covariates; do not recycle the Bright marginal draw for Faint.
   - If the mock lacks a response covariate used by DESI, marginalize or bin over that
     covariate and record the loss of fidelity. Never synthesize it from the tidal truth.
   - Use a frozen RNG seed and save the selected TARGETID vector plus checksum.
3. `BF_FINAL_SELECTION_v1` — production-realism target, deferred if necessary.
   - Requires a richer photometric mock containing the final BGS_FAINT selection
     observables, or a validated emulation/reweighting against DESI target catalogues.
   - Reproduce final target bits, priority split, fibre assignment, redshift success,
     angular mask, and separate Bright/Faint expected-count fields.
   - P10/P13 production claims must use this level or explicitly scope the released VAC
     to the proxy observation model.

Every MT1 product must contain:

- unique `TARGETID` and `TRACER_TYPE={BRIGHT,FAINT}`;
- `RA`, `DEC`, RSD `Z`, `Z_COSMO`, and magnitude/response covariates actually used;
- `FILE_NUM`, `BOX_INDEX`, `HALO_INDEX` with valid-link flags;
- target-bit, assignment, redshift-success, cap, shell, and completeness fields;
- original stage/path, row index, selection version, RNG seed, code SHA, and hashes;
- a de-duplication record.

De-duplication is deterministic: group by TARGETID; prefer an assigned successful
observation, then the highest available redshift-quality statistic, then the smallest
`TILELOCID`, then the earliest source row. Abort if repeated rows disagree in immutable
truth linkage or sky position beyond numerical tolerance. Resolve the 3,654,393
assigned unique IDs not found in `forFA0` before stamping `MT1_COMPLETE`; they may be
alternate-MTL bookkeeping rather than new physical targets and must not be silently
treated as galaxies.

MT1 gates:

- exact TARGETID uniqueness;
- zero Bright/Faint bit ambiguity unless documented by the mask definition;
- shell/cap counts and assignment fractions by tracer;
- redshift-success rates by tracer, `R_MAG_APP`, shell, and cap;
- 100% coordinate/unit parity with the Bright parent for shared TARGETIDs;
- valid truth-link fractions and catalogue-to-density correlation checks by tracer;
- no Faint row in the frozen Bright authoritative target set;
- explicit `ORACLE`, `PROXY`, or `FINAL_SELECTION` scope in the manifest.

##### P8.8.4 MT2 — preflight information and canonical field products

Before training, quantify whether the extra tracer changes the sparse observation
problem:

- number density, mean separation, and occupied-voxel fraction per shell and cap;
- counts-in-cells and Bright-Faint cross-correlation;
- fraction of previously empty 5 Mpc/h voxels filled by Faint context;
- distance from every Bright target to the nearest Bright and nearest Faint tracer;
- expected shot-noise reduction relative to Bright-only;
- mask/boundary coverage and overlap with existing P4 authoritative cores.

Build P3-compatible fields on the unchanged grid and patch geometry. Keep tracer
channels separate:

```text
N_B, mu_B, stabilized_delta_B,
N_F, mu_F, stabilized_delta_F,
support mask, boundary distance, LOS channels
```

Do not use `N_B + N_F` as the only count channel: Bright and Faint have different bias,
selection, and response. Fit `mu_B` and `mu_F` separately using the frozen full-cap
selection procedure; do not fit per patch. Store immutable observed fields and derive
contrasts on demand. Re-run P6 patch/context/subdivision parity for the enlarged input.

##### P8.8.5 MT3 — matched classical information controls

Run classical controls before neural training:

1. Bright-only CIC/TSC using the frozen P8 implementation.
2. Bright+Faint combined-count CIC/TSC.
3. Bias-aware two-tracer linear field estimate with tracer responses fitted on training
   folds only.
4. Density-matched thinning: randomly thin Bright+Faint to the Bright-only number
   density, preserving shell/cap selection, with three frozen seeds.
5. Faint-position null: retain Faint shell/cap counts but randomize positions within
   allowed support, destroying cosmic-web information while preserving selection.

These controls distinguish:

- more measurements of the same field;
- a useful second tracer population;
- a generic number-density gain;
- an apparent gain caused only by changing the radial selection function.

Classical affine or bias-response parameters must be fitted on training folds and
frozen before validation. Evaluate the identical Bright authoritative rows.

##### P8.8.6 MT4 — primary U-PATCH context-only screen

U-PATCH is the first neural gate because it already leads the Bright-only P8 screen
and accepts separate count/response channels without rebuilding a 17M-node graph.

Run the following matched models:

| ID | Input | Purpose |
|---|---|---|
| `U-B-v1` | frozen Bright channels | exact control rerun under the new data loader |
| `U-BF-ORACLE-v1` | Bright + oracle-assigned Faint channels | upper bound on information gain |
| `U-BF-PROXY-v1` | Bright + response-matched Faint channels | actionable development candidate |
| `U-BF-THIN-v1` | density-matched thinned multitracer channels | density-versus-population control |
| `U-BF-NULL-v1` | randomized Faint positions | selection/shortcut null |

Use the existing U-PATCH depth, grid, 64 Mpc/h core, 120 Mpc/h halo, target scaler,
loss, epoch sampler, rotations 0/2, seed 42, 20-epoch cosine schedule, resume contract,
and complete-fold logging. Change only the registered input channels. Train the primary
comparison from scratch. A zero-initialized extra-channel warm start may be run as an
optimization diagnostic but cannot replace the from-scratch comparison.

Open MT4 in stages:

- [x] One-core overfit for `U-BF-PROXY-v1`; the representative-core optimization
  gate passes by more than 99.97% loss-window reduction.
- [x] One complete rotation-0 epoch with exact exposure/resume/parity checks; all
  `10,351` cores are visited exactly once and complete-fold validation passes.
- [x] Run rotation 0 for 20 epochs from immutable commit `4556317`. Best epoch is
  the final epoch, with macro `R2_lambda1=0.60737` and shell scores
  `0.70792/0.65667/0.58036/0.48452`; outputs are under
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_proxy_v1/unet_multitracer/rotation_0/seed_42/`.
- [x] Rotation 0 passes the replication trigger. Rotation 2 was launched from frozen
  commit `6b80744` in tmux `p8_mt4_proxy_rot2`, allocation `56495027`; outputs are
  under
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_proxy_v1/unet_multitracer/rotation_2/seed_42/`.
- [x] Complete Proxy rotation 2. It reaches macro `R2_lambda1=0.53941` and sparse
  shell `0.43016`, but fails the per-rotation macro and supported-shell adoption gates.
- [x] The Proxy screen passes the trigger for the matched `U-BF-NULL-v1` neural
  control. Null rotation 0 was launched from the same frozen commit in tmux
  `p8_mt4_null_rot0`, allocation `56495031`; outputs are under
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_faint_null_v1/unet_multitracer/rotation_0/seed_42/`.
- [x] Complete Null rotation 0. Proxy exceeds it by `+0.13204` macro and in every
  shell, demonstrating that the real Faint spatial field—not the matched sampling
  shortcut—drives the rotation-0 neural gain.
- [x] The causal contrast is material enough to trigger geographic replication.
  Null rotation 2 was launched in tmux `p8_mt4_null_rot2`, allocation `56514867`,
  from immutable commit `6b80744`; outputs are under
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_faint_null_v1/unet_multitracer/rotation_2/seed_42/`.
- [x] Complete Null rotation 2 and freeze the two-fold matched-control contrast. It
  reaches macro `0.51874`; Proxy exceeds it by `+0.02067` macro and in every shell.
- [x] Apply the extension rule. All Proxy and Null schedules end at learning rate zero,
  so no extension is authorized despite best epochs at the cap.

MT4 closeout: `PASS_SAME_PHASE_TWO_ROTATIONS` for additional Faint spatial information;
`NO_GO_FOLD_INSTABILITY_AND_ROTATION_2_SAFEGUARDS` for the current Proxy U-PATCH encoder.
The full machine-readable record is
`/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_closeout/decision.json`
(SHA256 `4e840cde75606eefa0e5819e91fbb6c97f187c12a712e7fd24c6744cce88b0c3`).

Adoption requires, on both rotations:

- primary macro `R2_lambda1` improvement of at least `+0.03` over `U-B-v1`;
- sparse-shell improvement of at least `+0.03`;
- no individual first-three-shell degradation worse than `0.01`;
- Faint-position null consistent with the Bright-only control;
- retained improvement over the matched two-tracer classical estimator or a clearly
  stated conclusion that the gain is informational rather than ML-specific;
- no new boundary/context trend and complete P8 safeguards.

##### P8.8.7 MT5 — conditional G-PATCH union-graph screen

The original compute-saving rule was to open MT5 only after MT4 or a classical control
showed a meaningful sparse-shell gain. By explicit user instruction, the expensive
Proxy graph/metric construction and one short G-PATCH canary may now proceed in parallel
with the U-PATCH preflight so that both leading representations are technically ready.
This is an infrastructure and diagnostic screen only: it does not promote G-PATCH,
does not authorize a full G-PATCH training sweep, and does not weaken the requirement
that MT2/MT3 establish whether any gain is informational and whether learning adds value
beyond matched classical reconstruction.

If opened:

1. Construct one canonical global graph per catalogue over Bright+Faint nodes, with NGC
   and SGC as disconnected components.
2. Compute graph metrics globally in `rapids-gnn`; never recompute metrics per patch.
3. Add tracer identity, tracer-specific expected number density/completeness, and
   available magnitude/response covariates as node features.
4. Reuse P4 physical cores and fold roles. Bright nodes are authoritative supervised
   targets; Faint nodes are context-only.
5. Extract patch views of the canonical union graph with complete dependency-hop
   context and repeat P5 parity/support audits.
6. Compare `G-B`, union-metric-only, and full union-message-passing variants to isolate
   whether the gain comes from handcrafted geometry or learned Faint messages.

The same MT4 adoption gates apply. A union graph that helps only same-patch metrics or
only the macro through the sparse shell is not promoted.

##### P8.8.8 MT6/MT7 — optional F-tier and production validation

F-tier/U-Physics is not the first multitracer experiment. Reopen it only if MT2/MT4
demonstrate additional field information and the resource estimate fits the patch
contract. Its first test should use the separate Bright/Faint fields and fixed tidal
operator, not the previously infeasible v2_A graph scatter.

Any winning multitracer estimator must then pass:

- all five P4 folds and three seeds;
- P10 frozen independent-phase transfer with the same selection/response recipe;
- response perturbations and Bright/Faint ablations;
- DESI support checks for target density, assignment, redshift success, counts-in-cells,
  graph/field statistics, and mask dependence;
- the ordinary quality/OOD, tiling, provenance, and golden-wedge gates.

Posterior estimation, HOD marginalization, JEPA, and Faint supervision remain gated
until deterministic context-only transfer passes.

##### P8.8.9 Decision tree and stop rules

```text
F0 audit says no upstream Faint
  -> stop; acquire/regenerate richer mocks

Oracle Faint adds <0.02 sparse-shell R2
  -> stop multitracer ML; BGS_FAINT is not the missing information at this scale

Oracle helps, response-matched proxy does not
  -> observation-response modelling is the blocker; do not tune the encoder

Proxy helps U-PATCH and classical equally
  -> scientific result: tracer density was limiting; adopt the simpler best estimator

Proxy helps U-PATCH beyond matched classical controls
  -> open G-PATCH union graph and optional U-Physics/F-tier challengers

Same-phase gain fails P10
  -> no production promotion; expand simulation/selection diversity
```

No model is “rescued” by pooled R2, a target-wedge calibration, or a change to P4
geography. `U-PATCH-BRIGHT_REFERENCE` remains frozen throughout so every gain has a
stable baseline.

##### P8.8.10 Work-package schedule

| Package | Dependency | Compute | Estimated effort | Completion marker |
|---|---|---|---|---|
| MT0/F0 audit | existing stages | 1 CPU interactive | complete | `F0_FEASIBILITY_COMPLETE` |
| MT1 oracle/proxy catalogues | F0 | 1 CPU interactive, resumable | 0.5–1 day | `MT1_MULTITRACER_CATALOGUE_COMPLETE` |
| MT2 field/information preflight | MT1 | 1 CPU interactive | 0.5 day | `MT2_MULTITRACER_FIELD_COMPLETE` plus information audit |
| MT3 classical controls | MT2 | 1 CPU interactive | 0.5 day | `MT3_MULTITRACER_CLASSICAL_COMPLETE` |
| MT4 U-PATCH screens | MT2/MT3 | up to 2 interactive A100s, exact resume | 1–3 days | `MT4_UPATCH_MULTITRACER_DECISION` |
| MT5 union graph/G-PATCH | MT1 for authorized precompute; MT2/MT3/MT4 gain for full training | CPU + `rapids-gnn`, then A100 | 1–3 days | `MT5_GPATCH_MULTITRACER_DECISION` |
| MT6 F-tier/U-Physics | MT4 gain and resource pass | A100 | optional | `MT6_FIELD_PHYSICS_DECISION` |
| MT7 seeds/P10 | winning deterministic model | CPU/GPU | mandatory for production | `MULTITRACER_TRANSFER_COMPLETE` |

Keep at most two interactive allocations active; reserve batch submission for true
production. Each runtime package must be checkpoint-resumable and write its manifest,
logs, metric history, marker, and checksum before the next package opens.

The recovery is deliberately architecture-neutral. GraphNet, U-Net, a simplified
F-tier/U-Physics model, or a classical-residual hybrid may win; the scientific product is
the transferable estimator and validated protocol, not loyalty to a model family.

#### P8.9 — Bounded density-first baseline and learned long-mode closure

**Status:** D0 TARGET + TRACE + TENSOR CLOSURE PASSED; TRAINING CONTRACT NEXT
(2026-08-09)

The current evidence does not yet contain a matched DarkAI-style learned-density
baseline under the P4/P8 protocol. Keep the distinctions explicit:

- T1/CIC/DTFE reconstruct density classically and then apply the fixed tidal solve;
- historical T4/F1 produced an internal learned density field but trained it through
  the downstream eigenvalue loss;
- the planned F2 density-supervision experiment was never completed;
- P6 established numerical stability of an existing U-Net at a 120-Mpc field halo,
  not an accuracy gain from a larger effective receptive field;
- P7 showed that a learned field followed by a finite FFT solve required a 360-Mpc
  halo for numerical convergence, but did not train a P8 F-PATCH accuracy model;
- the completed true-field 60/120/180/240/360-Mpc diagnostic measures missing tidal
  support with perfect density, not what a learned reconstruction recovers.

Register one primary baseline, `U-DENSITY-PHYS-v1`, before any new graph-field,
density-residual, cell-size, or loss sweep:

```text
frozen Bright P3/P6 count, contrast, exposure, mask, and LOS fields
  -> patch-trained, patch-safe U-Net
  -> predicted R=7 Mpc/h smoothed matter contrast on core voxels
  -> deterministic overlap stitching on the complete NGC/SGC cap lattices
  -> one global fixed FFT tidal solve per cap
  -> tensor/eigenvalues sampled at the frozen Bright authoritative galaxies
```

The target field is

```text
delta_R7(k) = W_7(k) delta_m(k)
T_ij(k)     = (k_i k_j / k^2) delta_R7(k), with T_ij(k=0) = 0.
```

Do not apply `W_7` a second time in the tidal layer. Generate `delta_R7` at the frozen
target epoch on the canonical observer-frame cap lattices, verify its trace/eigenvalue
closure against the existing CACTUS labels, and store its construction manifest and
hashes. The matter field is privileged supervision only and can never enter a DESI
model input.

The observer-frame coordinate preflight is complete. On 16,000 authoritative galaxies,
host-halo `x_com` reproduces the frozen slab labels exactly, while the unique passing
simple sky mapping is `Z_COSMO` with observer origin
`(-1000,-1000,-1000) Mpc/h` (minimum eigenvalue `R2=0.987379`). This mapping is frozen
for target generation. The corresponding observed/RSD-`Z` row is materially worse,
especially for lambda1 (`R2=0.764040`). Consequently, density-first evaluation must
keep two non-interchangeable rows:

1. **real-space oracle sampling:** sample the reconstructed tensor at simulation
   `Z_COSMO` positions; this diagnoses the field/physics estimator but is not deployable;
2. **VAC-deployable sampling:** sample at observed `Z`, the coordinate available for
   DESI; this is the per-galaxy production row and includes the RSD/localization penalty.

Never use `Z_COSMO`, host positions, velocities, or halo linkage as model inputs. Never
quote the oracle row as DESI performance. If the real-space field passes but the
observed-`Z` row does not, the outcome is an RSD/localization blocker for a per-galaxy
physical-field VAC, not permission to hide the discrepancy with validation-fitted
rescaling.

Freeze the D0 training and inference contract:

1. Use Bright-only inputs first so the output-representation question is matched to
   `U-PATCH-BRIGHT_REFERENCE`. A later multitracer density row requires an MT4 gain and
   is separately named; it is not part of D0.
2. Reuse the P4 rotations, authoritative rows, 64-Mpc cores, P6 train-only response
   transforms, complete-exposure sampler, seed 42, patch-safe normalization, atomic
   resume, and complete-fold evaluation. Context voxels never own density loss.
   For voxelwise loss and stitching, ownership is defined by the P3 cell centre in
   exactly one half-open P4 lattice core. The older intersecting voxel ranges remain
   valid for context extraction but are not density-loss owners. Retain every nominal
   P4 core and add inference-only owner cores wherever supported voxels have no
   galaxy-occupied P4 core. Such rows use fold sentinel 255 and can never contribute
   loss, labels, evaluation, or model selection; they exist only to write the complete
   supported field after the model is frozen. This does not modify any P4 fold or
   authoritative galaxy. Unsupported rectangular volume is explicitly windowed for the
   final solve; supported voxels may never be zero-filled or replaced by a classical
   estimate.
3. Train the declared baseline with one training-fold-standardized voxelwise MSE on
   `delta_R7`. It has no eigenvalue/tensor target or direct point head. This deliberately
   measures the conventional density objective rather than silently turning D0 into
   another direct-eigenvalue model.
4. Infer contextual patches and stitch their uniquely owned density cores. Require
   order- and subdivision-invariant stitching, and measure disagreement under shifted
   or enlarged-context views before applying the P7-compatible global
   padding/apodization and fixed tidal solve.
5. Report both the raw physical eigenvalues and any train-fold-only affine diagnostic as
   separate rows. Never use validation labels to rescale the reconstructed field or
   silently substitute the affine row for the raw physical product.

Density MSE is an optimization loss, not the adoption metric. Report:

- field cross-correlation `r(k)`, transfer function, power ratio, mean/variance, and
  one-point PDF/tail recovery overall and by shell/cap/support distance;
- trace closure, tensor/eigenvalue error, ordering, and eigengap-conditioned orientation;
- the complete P8 per-galaxy macro, per-shell, pooled, class, knot/void, boundary, and
  spatial-block metrics on the identical authoritative rows;
- high-density and rare-knot failures explicitly, even if bulk field MSE is strong.

Use the stitched predicted field to run the missing learned long-mode diagnostic without
retraining: compare the global predicted-field tidal solve with matched
60/120/180/240/360-Mpc context solves, reporting trace and traceless shear separately
and comparing the curve with the completed true-field floor. Merely enlarging an input
halo beyond a fixed convolutional receptive field is not a context experiment and does
not satisfy this gate.

Run rotation 0 once. Continue to rotation 2 only if D0 is within `0.03` macro
`R2_lambda1` of the matched U-PATCH reference, improves a registered shell without a
supported-shell degradation worse than `0.01`, or yields a pre-declared tensor/
eigenvector benefit sufficient to justify retaining a secondary field product. If D0
has credible field `r(k)` but specifically fails downstream shear, tails, or eigenvalue
metrics, register at most one `U-DENSITY-PHYS-AUX-v1` run with the same architecture and
one fixed downstream tensor/eigenvalue auxiliary loss. Otherwise close density-objective
misalignment without a loss sweep. Do not open G-density, F-density, density-residual,
or generative-field branches from a failed D0.

Required artifacts are a target-field manifest, D0 config, stitched-field parity report,
field/downstream metric bundle, learned-context report, exact predictions, checksums, and
`DENSITY_FIRST_BASELINE_DECISION`. This bounded baseline may inform model choice but does
not delay P10 phase/cost benchmarking or replace independent-phase validation.

P8.9 granular execution checklist:

- [x] Implement and unit-test the target-coordinate preflight; verify immutable
  TARGETID joins, host-`x_com` label closure, coordinate units, observer origin, and
  observed-versus-cosmological redshift variants.
- [x] Pass the 16,000-row balanced runtime gate and freeze
  `Z_COSMO + (-1000,-1000,-1000) Mpc/h` for privileged target construction. Evidence:
  `/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/preflight/coordinate_alignment.json`
  and `docs/evidence/p8/density_target_alignment.json`.
- [x] Build cap-aligned `delta_R7` targets from the frozen T-web slabs using the exact
  P3 grid origin, cell size, shape, units, chunking, and observer mapping; do not smooth
  a second time. Evidence: runtime `p8_density_phys_v1/targets/target_manifest.json`
  and tracked `docs/evidence/p8/density_target_manifest.json`.
  The corrected mapping is `(observer_xyz_Mpc * h - 1000 Mpc/h) modulo 2000 Mpc/h`;
  the earlier mixed-unit artifact is retained only as superseded audit history.
- [x] Freeze the density-loss support as exact cell-centre ownership by a nominal P4
  core intersected with the P6 supported-voxel contract; quantify cap/shell coverage
  and prove that unsupported rectangular bounding-box volume never contributes loss.
  Context/intersection ranges are not loss ownership masks. The corrected radial
  support uses Planck18 comoving distances in the native P3 Mpc frame and contains
  115,089,625 voxels (78,099,094 NGC; 36,990,531 SGC).
- [x] Audit whether the P4 core tiling covers every supported output voxel needed for a
  complete-cap FFT. Register deterministic handling of any uncovered supported voxel;
  zero fill or silent classical substitution is forbidden. Exact nominal ownership
  leaves 5,660,978 supported voxels uncovered; 3,145 inference-only owner cores
  (1,788 NGC; 1,357 SGC) close coverage to exactly 100% without changing folds,
  supervised rows, or metrics. Evidence: runtime
  `p8_density_phys_v1/field_output_tiling/field_output_tiling_manifest.json` and tracked
  `docs/evidence/p8/field_output_tiling_manifest.json`.
- [x] Validate scalar trace closure at the same 16,000 authoritative galaxies.
  Host-`x_com`, supported `Z_COSMO`, and supported observed-`Z` rows give
  `R2=1.000000/0.991444/0.870115`, respectively. The oracle and deployable rows remain
  separate. Evidence: runtime `p8_density_phys_v1/target_closure/trace_closure.json`
  and tracked `docs/evidence/p8/density_target_trace_closure.json`.
- [x] Validate the separate one-global-FFT-per-cap tensor/eigenvalue closure before
  training. Report full-rectangle and survey-windowed solves, trace versus traceless
  shear, oracle `Z_COSMO` versus observed-`Z` sampling, and missing-external-tide
  sensitivity. Do not double-smooth the already `R=7 Mpc/h` target.
  The pre-runtime frozen rows are: rectangle plus 24-voxel padding; rectangle plus the
  P7 20-voxel cosine box taper and 24-voxel padding; hard science support; and science
  support times P3 apodized exposure times a 100-Mpc radial cosine taper. The latter
  two also use 24-voxel padding. Required gates on the balanced 16,000-row sample are:
  - tensor-trace versus transformed-input RMSE `<= 2e-4` for every cap, coordinate,
    and window row;
  - rectangle-raw `Z_COSMO` macro-shell `R2_lambda1 >= 0.90`;
  - apodized-science-window `Z_COSMO` macro-shell `R2_lambda1 >= 0.50` and worst-shell
    `R2_lambda1 >= 0.0`.
  These are target/operator and physical-floor gates, not model scores. `Z_COSMO` is
  privileged and may never be quoted as deployable DESI accuracy. Observed-`Z`, cap,
  shell, traceless-shear, box-taper sensitivity, and eigengap-conditioned window
  orientation rows are mandatory diagnostics. A failed gate blocks D0 until the
  field-output/context contract is revised; do not retune thresholds after inspection.
  Runtime PASS: trace-identity RMSE `2.997e-7`; full-rectangle `Z_COSMO` macro-shell
  `R2_lambda1=0.98648`; hard-support `0.97230`; apodized-window `0.92325` with worst
  shell `0.83536`. The apodized observed-`Z` macro is `0.69971`, with all shells
  `0.63450--0.75368`. Full rectangle versus P7 box taper is numerically immaterial,
  while the conservative radial taper owns most of the window loss. Hard-support
  traceless-shear R2 is `0.96692/0.93599/0.96378`; window-orientation disagreement
  decreases with privileged eigengap as expected. Evidence: runtime
  `p8_density_phys_v1/tensor_closure/{tensor_closure.json,sampled_tensors.npz}` and
  tracked `docs/evidence/p8/density_tensor_closure.json`. This authorizes the bounded
  D0 training contract; it is not a learned-model result.
- [ ] Freeze train-fold-only target scaling, D0 architecture/config, optimizer budget,
  logging, resume, and one-core overfit/canary tests; verify that no privileged target
  or `Z_COSMO` channel enters the model inputs.
  Pre-runtime contract: Bright U-PATCH's three-channel observation-only mapping;
  base-24 `UNet3D`; one scalar output; 24-voxel/120-Mpc context; 8-voxel alignment;
  seed 42. One training unit is a positive `(exact nominal owner core, shell)` pair;
  every supported owned training voxel appears exactly once per complete epoch and
  inference-only owners have no loss. Standardize `delta_R7` once on rotation-0
  training-fold voxels and use voxelwise MSE with `N_shell^-0.5` weights. Freeze 20
  complete AdamW epochs (`lr=0.002`, weight decay `1e-4`, clip 5) with a cosine schedule
  ending at zero and no early stopping. Save atomically every 250 updates and log every
  25. Select the field checkpoint by complete-validation macro-shell `R2(delta_R7)`;
  downstream eigenvalue adoption remains separate. Required preflight order is target/
  unit manifest, one-core overfit, exact resume, then one complete-epoch exposure
  canary. No direct eigenvalue/tensor target or loss is permitted in D0.
  Freeze the overfit probe at 200 AdamW updates with `lr=0.002` on the largest
  supported rotation-0 training unit; its final five-record mean standardized density
  MSE must be `<=0.25` times the mean over steps 1--10. The overfit checkpoint is never
  a scientific warm start.
  - [x] Build/hash the exact-owner unit manifest and rotation-0 train-only target
    scaler. There are 13,664/4,602/4,485 train/validation/development units; the scaler
    uses 65,869,177 training voxels with `mean=0.00383909`, `std=0.46607224`.
    Evidence: runtime `p8_density_phys_v1/training_contract/rotation_0/` and tracked
    `docs/evidence/p8/density_d0_{config,target_scaler}.json`.
  - [x] Pass the one-core overfit gate. Standardized density MSE contracts from
    `1.181697` to `0.027542` by the registered window means, ratio `0.023307 <= 0.25`,
    on 6,859 exact-owner voxels. Evidence: runtime
    `p8_density_phys_v1/overfit_probe/rotation_0/seed_42/overfit_report.json` and tracked
    `docs/evidence/p8/density_d0_overfit_report.json`.
  - [ ] Pass an interrupted/resumed trajectory test with identical next-step model,
    optimizer, scheduler, epoch order/cursor, loss trace, and RNG state.
  - [ ] Pass one complete-epoch canary with 100% unique training-unit exposure, zero
    repeats, all four shell voxel counts exact, persistent loss logging, and complete
    validation-unit field metrics.
- [ ] Train rotation 0 with complete-exposure density epochs and persistent loss/field
  validation curves; no direct eigenvalue/tensor loss is allowed in D0.
- [ ] Stitch overlapping density cores in an order- and subdivision-invariant manner;
  report overlap disagreement and supported-volume coverage before the one global FFT
  solve per cap.
- [ ] Evaluate field spectra/PDF/tails, trace/tensor/eigenvalue/orientation metrics,
  learned long-mode convergence, and the full P8 per-galaxy suite for both oracle and
  deployable sampling rows.
- [ ] Apply the registered rotation-2 continuation rule and write
  `DENSITY_FIRST_BASELINE_DECISION`; open at most the single pre-registered D1 auxiliary
  if its exact trigger is met.

Progress checklist:

- [x] Freeze the linear-increment target/scaler and complete-fold macro-R2 evaluator.
- [x] Freeze rotations 0/2, the P4 fold roles, authoritative cores, and train-only
  transformations before opening validation scores.
- [x] Run the matched full-cap CIC plus fixed FFT tidal reconstruction with affine
  calibration fitted on the registered training folds only.
- [x] Run G-PATCH seed 42 on rotations 0 and 2 with the exact four-dependency-hop
  P5 adapter and score every authoritative validation core.
- [x] Run U-PATCH seed 42 on rotations 0 and 2 with patch-safe per-voxel channel
  normalization and score every authoritative validation core.
- [x] Resolve F-PATCH v2_A before training as `NO_GO_FROZEN_V2_A_RESOURCE_INFEASIBLE`;
  the representative low-z view requires at least 91.6 GiB before autograd, decoder,
  FFT fields, or five-hop graph context.
- [x] Audit short-screen core coverage, fold-boundary dependence, graph degree/density,
  four-hop isolation as a diagnostic, residual spatial correlation, runtime, and memory.
- [x] Apply the no-macro-only-win interpretation: G-PATCH and U-PATCH trail CIC in all three
  tracer-supported shells, so neither clears the classical adoption gate.
- [x] Freeze the 2,000-step outputs as short-screen evidence; do not promote them to five
  folds or describe them as converged science runs.
- [x] Reproduce the historical samplers and record that only 15.07--15.14% of eligible
  training cores and one validation checkpoint were used.
- [x] Implement complete exposure-aware patch epochs, globally row-weighted patch
  objectives, per-epoch full-fold validation, atomic mid-epoch checkpoints, exact
  dropout-RNG resume, checkpoint-reconciled windowed loss traces, and registered early stopping in
  `workflows/abacus_tweb/{p8_epoch_training.py,p8_train_patch_recovery.py}`.
- [x] Unit-test complete/no-repeat epoch exposure, weighted subdivision invariance,
  partial-epoch accumulation/resume validation, and the epoch-5/patience-3/min-delta
  rule in `tests/phase4/test_p8_epoch_training.py`.
- [x] Run the registered U-PATCH one-core overfit diagnostic with
  `workflows/abacus_tweb/p8_probe_unet_overfit.py` and determine whether its output
  range expands under deliberate memorisation. Core 15211 (204 shell-3 galaxies)
  reaches scaled MSE 0.00173, lambda1 R2 0.9991, and recovers the full truth range;
  artifacts are under `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/probes/`.
- [ ] Run one complete rotation-0 canary epoch for G-PATCH and U-PATCH; verify 100%
  unique core coverage, all-shell weighted loss accounting, full-fold validation,
  allocation-interruption resume, and persistent loss curves.
- [x] Complete the registered rotation-0 `recovery_v1` runs: G-PATCH stopped by the
  frozen patience/min-delta rule at epoch 15 (best macro R2 0.4682), while U-PATCH hit
  the 20-epoch cap still improving (best macro R2 0.4943).
- [x] Complete the unchanged rotation-2 `recovery_v1` runs for G-PATCH and U-PATCH:
  G-PATCH macro R2 0.4708 at epoch 16; U-PATCH 0.5128 at epoch 20.
- [x] Implement and unit-test the guarded warm-start/fresh-optimizer extension contract
  in `workflows/abacus_tweb/{p8_train_patch_recovery.py,p8_epoch_training.py}` and
  `tests/phase4/test_p8_epoch_training.py`; the CLI enforces the registered run name,
  20-epoch budget, `2e-4` learning rate, disabled early stopping, matching
  model/rotation/seed, parent-best epoch offset, and complete checkpoint provenance.
- [x] Complete the pre-registered `convergence_extension_v1` long-horizon test
  without modifying or overwriting the primary recovery artifacts. Rotation 0
  improves G-PATCH `0.4682 -> 0.4910` and U-PATCH `0.4943 -> 0.5070`.
  Rotation 2 improves G-PATCH `0.4708 -> 0.4825` and U-PATCH
  `0.5128 -> 0.5197`. Both 20-epoch extensions exhausted their cosine
  schedules. Artifacts are under
  `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/convergence_extension_v1/`
  using the immutable `c92356a` training revision and exact-resume checkpoints.
- [x] Run the true-field context-growth diagnostic, separating trace from
  traceless shear. The implementation is
  `workflows/abacus_tweb/p8_true_field_context.py`; runtime evidence is
  under `/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/true_field_context_v1/`
  with marker `TRUE_FIELD_CONTEXT_COMPLETE`. The matched 1024-grid experiment
  uses true-density radii 60/120/180/240/360 Mpc/h. Overall eigenvalue RMSE
  falls from 9.27% to 0.88% of the full-periodic reference scatter between
  60 and 360 Mpc/h.
- [ ] Complete matched full-cap exact DTFE and global classical-plus-local-residual
  controls.
  - [x] Implement and unit-test the accelerated exact-DTFE locator. The production
    path now builds a vertex-to-incident-tetrahedron CSR once, queries grid voxels
    progressively through K=1/8/32/128 nearby vertex stars, and performs exact
    barycentric containment/interpolation. The synthetic test explicitly includes
    cases where the nearest site is not a vertex of the containing tetrahedron.
  - [x] Run the resumable exact NGC+SGC DTFE field build without approximate
    substitution. Exact Delaunay neighbour walking classifies every K-locator miss with
    zero singular/max-step failures; finite apodized-support coverage is `97.071%` NGC
    and `94.874%` SGC, with every remaining voxel proven outside the catalogue convex
    hull. The registered 99% finite-coverage gate therefore fails as a survey-geometry
    result rather than a numerical-locator failure.
  - [x] Evaluate exact DTFE on rotations 0 and 2 under the frozen training-only affine
    and authoritative-row contract. Macro `R2_lambda1 = -0.150/-0.185`; the first-three
    shell diagnostics are `0.125/0.104`. This exact full-cap DTFE is a negative matched
    control under the mask/selection geometry and does not replace CIC as the supported-
    shell classical anchor.
  - [x] Implement and unit-test `U-CIC-RESID-v1` with a keyed active-fold CIC anchor,
    frozen U-PATCH backbone provenance, zero-residual parity gate, bounded lambda1
    correction, and positive multiplicative eigengap corrections in
    `workflows/abacus_tweb/p8_train_unet_cic_residual.py` and
    `p8_train_patch_recovery.py`.
  - [x] Regenerate the rotation-0/2 keyed CIC anchors with the frozen classical code.
  - [x] Audit and freeze `U-CIC-RESID-v1` as `STOP_FEASIBILITY_NO_GO`: its one-sigma
    lambda1 correction cannot satisfy the sparse-shell adoption gate even with an oracle
    correction. Three completed epochs are retained as execution evidence, not accuracy
    evidence.
  - [x] Run the separately registered rotation-0/2 `U-CIC-RESID-v2` screens with the
    train-only-selected three-sigma bound and compare exactly matched rows to CIC and
    standalone U-PATCH. Freeze the branch as `NO_GO_SPARSE_SHELL_REGRESSION`.
  Evidence remains under
  `/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/classical/dtfe_fullcap_v1/`
  and `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/u_cic_resid_v1/`.
- [ ] Reapply the classical adoption and five-fold promotion gates to converged results.
- [ ] Spend three seeds only on candidates that pass the recovered two-rotation gate.
- [x] Freeze standalone U-PATCH as `U-PATCH-BRIGHT_REFERENCE`, the current two-rotation
  learned candidate; do not call it production-approved before five-fold/seed and P10.
- [x] Complete the BGS_FAINT F0 feasibility audit and register the conditional
  response-complete catalogue build in P8.8.
- [x] Keep log-gap, FMPE/NPE, JEPA, HOD, and broad architecture branches gated while the
  deterministic recovery remains open.
- [x] Freeze validation predictions, reports, diagnostics, configs, and the machine-readable
  screen decision under `docs/evidence/p8/` and the runtime root.
- [x] Record G-PATCH and U-PATCH as `INCONCLUSIVE_OPTIMIZATION_AUDIT_REQUIRED`, F-PATCH
  v2_A as resource NO-GO, and CIC/DTFE-style reconstruction as the reference direction.
- [ ] Close P8 only after the recovered scientific gate is actually complete.

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

#### P10.0 — `ph000` development boundary

It is reasonable to defer the expensive multi-phase training campaign until the
remaining representation and observation questions have been reduced to one or two
finalists. It is not acceptable to optimize `ph000` without a terminal boundary and
then describe the result as more generalisable. Repeated decisions against the same
blocked folds create adaptive benchmark overfitting even when the supervised rows are
spatially disjoint.

The remaining authorized `ph000` model-development decisions are therefore bounded to:

1. finish the active MT4 Proxy/null decision and any already-triggered registered
   rotation;
2. run P8.9 `U-DENSITY-PHYS-v1` and only its explicitly gated D1 auxiliary;
3. implement and canary the P10.1 final/degraded-view loaders and response channels;
4. use `ph000` only for technical or optimization screens of Arms A--C, never as their
   production-transfer decision;
5. open one paired-view P11 experiment only after the independent-phase Arms A--C
   diagnose a representation bottleneck.

No additional `ph000` architecture, loss, feature, context, cell-size, residual, or
posterior sweep opens from a null result. When the authorized decisions close, freeze
the surviving encoder(s), D0 field contract if retained, degradation recipes, response
schema, target/metric code, transformations, compute budgets, and acceptance rules in a
signed manifest. Later model selection occurs on ph006; ph001 remains a one-open blind
test and may not be used to tune this list.

P10 infrastructure is not deferred. Audit phase assets and benchmark one complete
target-generation chain in parallel with the remaining `ph000` work. This separates the
scientific choice to delay expensive multi-phase training from an avoidable late data,
runtime, or storage blocker.

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

#### P10.1 — Controlled observation-operator training

**Status:** SCIENTIFIC TRAINING GATED ON P8 CLOSURE; LINEAGE AUDIT AND TECHNICAL
CANARIES AUTHORIZED

The catalogue identifiers required to pair and group examples are not automatically
valid conditioning variables. Use this role contract:

| Variable class | Examples | Permitted use |
| --- | --- | --- |
| Grouping/provenance only | phase, observer ID, HOD ID/seed, stage ID, degradation seed, core ID, TARGETID, halo linkage, source hashes | Pairing, outer splits, hierarchical sampling, leakage audit, and stratified metrics; never production inputs |
| Deployable physical/response | continuous redshift or `log ntilde(z)`, LOS, random-derived expected intensity, exposure, completeness, fibre-assignment probability, redshift quality/error, mask distance, effective support | Concatenate as node/voxel channels; broadcast legitimate patch summaries or concatenate them before the output head |
| Unknown simulation nuisances | HOD and velocity-bias parameters, unobserved failure-recipe parameters | Vary during training and later marginalize; do not reveal their true values to the estimator |
| Privileged truth | `Z_COSMO`, true peculiar velocity, halo mass/linkage, matter density, T-web/NEXUS+ truth | Target construction, train-only auxiliary labels, and evaluation where declared; never a DESI production input |

Observer ID must not substitute for geometry: supply the actual LOS and response fields.
Stage ID must not substitute for the physical degradation: supply the quantities that
will exist for DESI. Prefer continuous redshift/selection variables over a shell
one-hot. Phase, observer, HOD, and stage remain outside the model call:

~~~text
sample.meta = phase, observer, HOD, stage, seeds, latent core, provenance
sample.x    = graph or field rebuilt for this observation view
sample.cond = deployable local response channels
sample.y    = shared ordered-eigenvalue increments or tensor target
prediction  = model(sample.x, sample.cond)
~~~

Begin conditional modelling by direct concatenation. Fit every response transform on
training phases/cores only and freeze it. Introduce FiLM or a separate conditioning
network only if a registered diagnostic shows that direct conditioning is ignored. In
P12 the same contract becomes `q(theta | encoder_output, response)`.

Sample in the order `phase -> latent spatial core -> observation view`. Keep every
stage, observer, HOD, and degradation realization of a latent core in the same outer
phase/spatial split, give the latent core one scientific weight, and rebuild topology,
topology-derived features, fields, and random-response products independently for every
view. Because distinct observers can revisit the same periodic-box structure, the first
P10 test should either use one observer or prove cross-observer grouping by base-box
identity.

Freeze a response-explicit degradation ladder before model training. It must include a
high-fidelity dense observed-galaxy view, the final Path1-like deployment view, and
intermediate views that vary magnitude selection, redshift-dependent tracer density,
angular completeness/masks, fibre assignment, redshift success/error, and tracer
composition where the current mock supports them. Uniform random thinning is a named
information control, not a substitute for the survey observation operator. Every view
must save its selected IDs, random seed, response fields, topology/field manifests, and
source hashes. Hold out at least one degradation recipe, not only a seed, from training.

The dense view is privileged training context, not a DESI input. Before it can become a
teacher in P11, its supervised model must transfer to unseen spatial blocks and ph006;
otherwise latent alignment would distill a phase- or geography-specific representation.
Multiple degradation views do not create new cosmic structures and cannot replace the
phase split.

Only three observation-training arms are mandatory:

| Arm | Training contract | Identifies |
| --- | --- | --- |
| A | Multi-phase, final-Path1-view supervised baseline | Transfer with deployment-like inputs |
| B | A phase-balanced mixture: final view plus one balanced auxiliary degradation view per latent core | Benefit of observation diversity |
| C | B plus continuous, DESI-available response conditioning | Benefit of modelling known survey response |

Select on the final production-like ph006 view, while reporting every stage, worst
stage/effect, and at least one held-out degradation recipe. A curriculum is only an
order-controlled replay ablation using the identical Arm-C examples and optimizer
updates. After any dense-to-sparse warm-up, retain balanced replay of dense,
intermediate, and final views; a one-way curriculum that forgets earlier views is not an
admissible comparison. Paired consistency is optional. Cross-stage JEPA remains P11 and
opens only if Arms A–C identify a representation bottleneck.

Minimal decision order:

1. finish P8 unchanged;
2. audit/re-export a current-Path1-compatible degradation ladder and matched randoms;
3. run the P3b drop-in response-field checks;
4. run P10 Arms A–C across independent phases;
5. test curriculum or paired consistency only for a remaining observation-transfer
   failure;
6. open P11 JEPA only for a diagnosed representation bottleneck;
7. open the bounded NEXUS+ auxiliary branch only for a diagnosed multiscale-morphology
   residual;
8. fit P12 uncertainty after the deterministic representation and response schema are
   frozen.

---

## 9. JEPA gate

### P11 — Representation pretraining

**Status:** DEFERRED; OPTIONAL ONLY IF P10 ARMS A--C DIAGNOSE A REPRESENTATION BOTTLENECK
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

#### P11.1 — Paired dense/degraded teacher--student gate

**Status:** OPTIONAL; GATED ON P10 ARMS A--C AND A TRANSFERRING DENSE TEACHER

The historical T3 LUPI attempt is not evidence against this branch: it used a
true-density CNN teacher, never completed a valid GPU run, and was shelved without a
scientific result. The first P11 test instead uses the same observed-galaxy modality at
different known response levels so it isolates observation robustness rather than
distilling a teacher that directly sees the answer field.

Open `PAIRED-DEGRADE-JEPA-v1` only if:

1. the dense-view teacher transfers under the frozen outer spatial/phase split and has
   clear headroom over the final-view student;
2. P10 Arm C still shows a reproducible final-view or held-out-recipe deficit consistent
   with a representation bottleneck rather than missing information alone; and
3. all views of a latent core have valid pairing/provenance and remain in one outer
   split with one total scientific weight.

Use the leading frozen encoder family; U-PATCH is the default while it remains the
deterministic leader. The teacher is frozen or EMA-updated and receives the dense view.
The deployable student receives the degraded view plus only DESI-available response
channels. The teacher, degradation label, phase, observer, HOD, and true failure-recipe
parameters are absent at inference.

Use the bounded objective

```text
L = L_target(student prediction, shared target)
    + alpha L_align(P(z_student), stopgrad(z_teacher))
    + beta L_spread(z_student)
```

where `P` predicts only teacher components recoverable from the degraded observation.
Align corresponding local/core and multiscale latents, with masks for unsupported
locations; do not force exact equality of one pooled embedding. Exact alignment is
physically wrong when sparsity has removed information and can create hallucination,
excessive shrinkage, or overconfident downstream posteriors. `L_spread` must prevent
collapse, and alpha/beta plus the aligned layers are frozen before validation scores.

Compare at identical outer splits, seeds, examples, optimizer updates, and compute:

1. Arm C supervised response-conditioned initialization with no alignment;
2. masked reconstruction/denoising on the same views;
3. paired latent prediction with the objective above;
4. an order-only dense-to-sparse curriculum using the identical view multiset and
   balanced replay.

Evaluate the student alone on the production-like ph006 view, every registered severity,
the held-out degradation recipe, and later ph001. Report primary/worst-shell metrics,
stage-wise transfer, response dependence, embedding collapse/spread, and deterministic
calibration diagnostics. A gain on `ph000`, the dense teacher, or an intermediate stage
does not promote the branch. Adoption requires the existing fresh-phase `+0.03` target
or comparably clear balanced-class gain, no supported-shell degradation worse than
`0.01`, and no deterioration on the held-out recipe. Posterior uncertainty remains P12;
alignment must never be presented as calibrated uncertainty.

Do not pretrain on DESI until truth-known sim-to-sim controls pass. DESI pretraining is
transductive domain adaptation, not zero-shot generalisation.

Adopt JEPA only for consistent fresh-graph or blind-phase deterministic improvement,
targeting at least +0.03 spatial-fold macro R²(λ1) or a comparably clear balanced-class
gain.

Progress checklist:

- [ ] Reopen only if P8/P10 establish a specific representation-data bottleneck.
- [ ] Freeze random-init, masked-reconstruction, and JEPA matched controls.
- [ ] Implement leakage-safe spatial masks and feature-support guards.
- [ ] Validate the dense teacher on unseen spatial blocks and ph006 before distillation.
- [ ] Keep all paired views in one outer split with one latent-core scientific weight.
- [ ] Freeze the degradation ladder, held-out recipe, alignment layers, and loss weights.
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
- NEXUS+: evaluation/auxiliary branch only; it does not reopen the primary target.
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
