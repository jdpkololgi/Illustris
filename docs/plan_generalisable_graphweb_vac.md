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
                                                  +--> P11 JEPA gate [optional, parallel]
                                                  +--> P12 posterior calibration [production VAC gate]
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

**Status:** P3a COMPLETE — canonical NGC+SGC fields and `FIELD_COMPLETE` frozen;
P3b source audit required before the P10 training interface is frozen and P3b exports
required before Arm C
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

**Status:** P3B-R IS THE ACTIVE PRODUCTION-RESPONSE GATE; SOURCE CONTRACT FROZEN;
RANDOM-RESPONSE EXPORTS AND R1 TRAINING PENDING; P3A IS THE OCCUPANCY-DERIVED R0
BASELINE, NOT THE FINAL OBSERVATION OPERATOR

A random catalogue is a high-density, unclustered Monte Carlo sampling of the survey
selection measure. It is not a set of negative galaxies, an estimate of the missing
matter, or an independent universe.

The response contract separates geometrical support from conditional completeness:

~~~text
M(v)   in {0,1}  = footprint/imaging-veto support, including holes
C_s(v) in [0,1]  = probability of surviving assignment and redshift-success stage s
p_s(v)           = M(v) * C_s(v), with separately registered factors where available
~~~

This distinction is mandatory at footprint edges and holes. A low observed count in a
supported voxel can be physical; a voxel with `M=0` contains no survey information.
For G-PATCH, remove any graph edge whose sampled physical segment crosses `M=0`; do not
allow a Delaunay or radius edge to bridge a masked hole or cap boundary. Boundary
distance is measured to the nearest `M=0` support cell and is both a candidate feature
and a required stratification variable.

**Source-availability audit (2026-08-13).** The upstream DA2 SecondGen BGS LSS tree
already contains 18 candidate final-view files
`BGS_BRIGHT_{0..17}_full_HPmapcut.ran.fits` for every `ph000`--`ph006`, under
`altmtl{phase}/kibo-v1/mock{phase}/LSScats`. Matching DESI-data names also exist under
the DA2 `kibo-v1` and `loa-v1` LSS releases. The inspected full randoms carry angular,
tile, hardware, imaging-veto and `FRAC_TLOBS_TILES` columns but no redshift; the
clustering randoms add `Z`, weights and `TARGETID_DATA`. Therefore the project should
**register and audit these products, not generate new final-view randoms by default**,
and should not use the clustering-random `Z` distribution until its data-linkage and
radial-selection provenance are accepted. This layout agrees with the
[DESI mock LSS data model](https://desidatamodel.readthedocs.io/en/latest/DESI_ROOT/survey/catalogs/RELEASE/mocks/AbacusSummit/OBSCON/VERSION/altmtlX/mockX/LSScats/index.html),
which defines 18 random realizations and distinguishes full from clustering products.
The P10 deployment source is frozen to the superseding Loa DR2 family, exactly
`DA2/loa-v1/LSScats/v2.1/PIP`; the official SecondGen mock products remain
Kibo-derived. This cross-family contract is explicit and must not be described as a
matched release. P13 may tighten the exact version only within the Loa family, never
silently fall back to Kibo or infer a release from a filename.

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

The existing final-view randoms need not imply a separately materialized catalogue for
every intermediate view. Prefer one audited base angular random set plus a
manifest-frozen factorization of `M`, fibre-assignment probability and redshift-success
probability. Materialize a stage-specific random only when the upstream product has a
meaningful persistent random identity and stage response. This follows DESI's use of a
matched random catalogue as an unclustered sampling of observability
([Ross et al.](https://arxiv.org/abs/2405.16593)); alternate-MTL/PIP products are the
appropriate precedent for density-dependent fibre assignment
([Lasker et al.](https://arxiv.org/abs/2404.03006)). ASTRA's merged galaxy+random
Delaunay construction is a useful alternative representation, especially for
underdense regions, but it is not the default encoder input
([Forero-Romero et al.](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf032/8221862)).

Source and implementation checklist:

- [x] Extend the tracked P10 phase registry to record all 18 full and clustering random
  paths, sizes, schemas and content hashes for ph000--ph006.
- [x] Freeze the candidate DESI-data LSS release/version and record the homologous full
  random, clustering random, completeness/PIP and quality products; do not silently mix
  `kibo-v1`, `loa-v1` or future public DR products.
- [x] Prove which identifiers, if any, persist across base and stage-specific randoms;
  otherwise build paired response fields from one base catalogue rather than claiming
  point-level pairing.
- [x] Audit the provenance and intended measure of `Z`, `TARGETID_DATA`, `WEIGHT*`,
  `FRAC_TLOBS_TILES`, `COMP_TILE`, `FRACZ_TILELOCID`, `GOODHARDLOC`, `NTILE`, `MASKBITS`
  and imaging-map cuts before assigning each to `M`, `C_s`, `ntilde_s` or diagnostics.
- [x] Write the atomic
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P10_RESPONSE_SOURCES_READY.json`
  binding the accepted mock and DESI sources, schema crosswalk and the rule that no
  target or local patch statistic enters the response.
  The passing audit enumerates and content-hashes all 18 full and 18 clustering
  randoms for every mock phase plus the exact Loa deployment products, freezes the
  semantic crosswalk, and explicitly declines unsupported point pairing. Tracked
  registry and evidence: `configs/p10_response_sources_v1.json`,
  `docs/evidence/p10/response_sources_20260814.json`, and
  `docs/evidence/p10/response_sources_ready_marker_20260814.json`. This is a source
  gate only: `response_fields_complete=false`, so Arms B/C remain gated.
- [ ] Build a small random-only and boundary-crossing canary before full P3b export.

##### P3b-R — Random-derived response products

**Priority:** immediate, before any new FAINT or JEPA run. Randoms are the canonical
unclustered response reference for BRIGHT-only production inference. BGS_FAINT remains
an optional tracer-information experiment rather than a response proxy or production
dependency.

Phase ownership and source contract:

- [x] Build training products for ph000/ph002/ph003/ph004/ph005 and a validation
  product for ph006. Do not open ph001 until the deterministic response, P12 and
  evaluation decisions are frozen. Products pass under each phase's
  `p3b_random_response_v1/`; tracked manifests and QA are in
  `docs/evidence/p3br/`. ph001 was not opened.
- [x] Use only the registered
  `BGS_BRIGHT_{0..17}_full_HPmapcut.ran.fits` angular randoms. Preserve the explicit
  Kibo-derived mock versus Loa deployment provenance; do not describe it as pointwise
  matched.
- [x] Keep the frozen BRIGHT `ntilde(z)` contract. Do not infer radial selection from
  clustering-random `Z` or `TARGETID_DATA`.

Random-density convergence:

- [x] On ph000 and ph006 compare random IDs `{0}`, `{0,1,2,3}`, and `{0..17}`.
- [x] Adopt fixed IDs `0..3` for every phase only when 4-versus-18 has support Jaccard
  at least `0.999`, median absolute fractional response difference at most `0.01`,
  99th-percentile difference at most `0.05`, and cap/shell expected-count differences
  at most `0.01`. Otherwise use all 18 everywhere. The ph000/ph006 comparison fails
  the pixel-amplitude gates (`median=0.027752`, `p99=0.119179`) and therefore freezes
  IDs `0..17` for every phase.
- [x] Record selected IDs, source hashes and the convergence decision atomically at
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P3BR_RANDOM_DENSITY_DECISION.json`
  (SHA-256 `2f0204b3cdc64d9f23408672cdc34da20033844e8bb747b7c6f12f40575bc6e7`).

Immutable output contract:

~~~text
/pscratch/sd/d/dkololgi/abacus/p10_multiphase/phXXX/
    p3b_random_response_v1/
        NGC/response_overlay.h5
        SGC/response_overlay.h5
        manifest.json
        qa.json
~~~

- [x] Match P3a exactly: HEALPix `nside=256` RING, **5 comoving Mpc** cells
  (`3.383 Mpc/h` at the registered Planck18 `h=0.6766`), cap origins, shapes and
  chunks. Store overlays rather than duplicate counts or LOS fields. Do not silently
  reinterpret the established P3a lattice as a literal `5 Mpc/h` grid.
- [x] Export `support_random`, `angular_response`, `exposure_apodized_random`,
  `expected_counts_random`, `log_count_ratio_random`,
  `distance_to_support_boundary` and audit-only raw random counts.
- [x] Define `angular_response` with mean one in each registered cap/PHOTSYS domain;
  define expected counts as
  `ntilde_BRIGHT(z) * V_voxel * angular_response * exposure_apodized_random`.
- [x] Freeze grid, channel units, normalization, selected random IDs, source hashes,
  `ntilde` hash, code commit and `sealed_phase_opened=false` in every manifest.

Validation and promotion gates:

- [x] Require exact P3a grid/parent parity, finite arrays, non-negative expected
  counts, no support outside the random mask, retained internal holes and identical
  ph000--ph006 schemas. All twelve cap products pass; binary support is sampled
  directly from the random map and never smoothed or filled.
- [x] Require a deterministic Poisson random-only canary to have mean standardized
  **count residual** `(G-mu)/sqrt(mu)` consistent with zero, and verify cap/shell
  expected-count closure under the frozen ensemble tolerance. Record, but do not gate
  on, the mean log-count ratio: its expectation is negative at low `mu` by Jensen's
  inequality and is not mathematically expected to be exactly zero. The twelve
  standardized means span `[-9.30e-4,+1.05e-3]`; observed/expected shell totals span
  `[0.9494,1.0526]`.
- [x] Build a co-chunked three-channel adapter
  `[BRIGHT counts, random exposure, random log-count-ratio]`; do not change network
  width, target rows, architecture, loss, patch geometry or optimizer-update budget.
  Frozen loader:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract_r1_random/TRAINING_LOADER_READY.json`.
- [x] Run a 1,000-patch throughput canary before full training. Use four-GPU DDP only
  after a measured speedup of at least `2.5x`; otherwise keep independent one-GPU
  scientific tasks. The one-A100 canary passed at `9.174 patches/s` (`1,000` updates
  in `109.005 s`) with report
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/response_training/p3br_r1_canary_1000_v1/unet/seed_42/THROUGHPUT_CANARY_REPORT.json`
  (SHA-256 `2ccaf66107d71cc1cff4ab44e45516345dda59b62b8c416a54999abe512a949e`).
  No DDP path is promoted; the four GPUs remain assigned to independent scientific
  tasks.
- [ ] Compare R1 with frozen R0 plus matched CIC/DTFE on ph006. Report pooled,
  four-shell macro, first-three macro, every shell, slopes, variance ratios, Spearman,
  response quantiles and boundary distance.
- [ ] Promote immediately for macro gain at least `+0.03` with no supported-shell loss
  worse than `0.01`; for gain `+0.01--+0.03`, require a second positive seed and mean
  gain at least `+0.02`; below `+0.01`, retain random response for posterior/deployment
  safety without claiming deterministic accuracy gain.

Implementation status (the scientific product gates above remain authoritative):

- [x] Implement and unit-test the streamed HEALPix builder, 1/4/18 convergence
  decision, cap-disconnected HDF5 overlays and exact-resume CPU orchestrator:
  `workflows/abacus_tweb/p3br_build_random_response.py`,
  `workflows/abacus_tweb/run_p3br_pipeline.py`, and
  `workflows/sbi/run_p3br_cpu_interactive.sh`.
- [x] Implement the capacity-matched stored-channel R1 adapter, frozen-normalization
  preparer, unchanged-width trainer wrapper and exact 1,000-update throughput gate:
  `workflows/abacus_tweb/p3br_prepare_r1_contract.py`,
  `workflows/abacus_tweb/p3br_training_contract.py`,
  `workflows/abacus_tweb/p10_train_random_response.py`, and
  `workflows/abacus_tweb/p3br_run_r1_throughput_canary.py`.
- [x] Implement independent one-GPU scheduling rather than unmeasured DDP, matched
  random-response CIC/DTFE scheduling, exact ph006 R0/R1/classical scoring, response
  quantiles, random-boundary bins and the registered promotion rule:
  `workflows/sbi/run_p3br_r1_p12_4gpu_interactive.sh`,
  `workflows/sbi/run_p3br_classical_4gpu_interactive.sh`, and
  `workflows/abacus_tweb/p3br_evaluate_r1.py`. While the legacy FAINT/P12 allocation
  remains active, `workflows/sbi/run_p3br_r1_sidecar_existing_gpu.sh` reuses only its
  idle fourth GPU. `workflows/sbi/run_p3br_transition_after_legacy.sh` then performs
  the non-overlapping handoff to the integrated supervisor automatically.
- [x] Implement tracked compact-evidence export with runtime-to-repository hash
  verification at `workflows/abacus_tweb/p3br_export_evidence.py`. Runtime products
  are not marked complete until the visible-phase manifests and QA below pass. The
  visible-phase products now pass; tracked evidence is in `docs/evidence/p3br/` and
  the runtime completion marker is
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P3BR_PIPELINE_COMPLETE.json`.

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
- random-only standardized count residual is consistent with zero; the known
  low-count Jensen bias of the log-count ratio is recorded rather than mislabeled as
  response error;
- results are stable across random seeds/densities, with random Monte Carlo noise
  subdominant to galaxy sampling noise;
- response fields use no targets, phase/split ownership, true matter, or local
  patch-wise renormalization;
- data, random, response, and topology hashes are view-specific;
- mock and DESI response columns have one deployable schema and units;
- graph components remain cap-disconnected and no retained edge crosses an `M=0` hole;
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

**Status:** COMPLETE — `ph000` DETERMINISTIC DEVELOPMENT FROZEN; U-PATCH-BRIGHT
REFERENCE AND CIC HANDED TO P10; D0 RETAINED AS ROTATION-0 SECONDARY FIELD/TENSOR
EVIDENCE ONLY (2026-08-10)
**Duration:** complete; independent-phase replication belongs to P10

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

**Status:** COMPLETE — SAME-PHASE INFORMATION PASS; CURRENT PROXY ENCODER NO-GO;
MT5 TECHNICALLY READY BUT FULL TRAINING CLOSED ON `ph000`; NO MODEL PROMOTED

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

**Status:** COMPLETE — ROTATION-0 D0 AND DARKAI-LIKE RESCORE COMPLETE; NO PRIMARY
POINT PROMOTION; ROTATION-2 AND D1 CLOSED WITHOUT RUN UNDER THE TERMINAL `ph000`
FREEZE (2026-08-10)

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
- [x] Freeze train-fold-only target scaling, D0 architecture/config, optimizer budget,
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
  - [x] Pass an interrupted/resumed trajectory test with identical next-step model,
    optimizer, scheduler, epoch order/cursor, loss trace, and RNG state. The continuous
    and update-2-resumed four-step trajectories agree exactly: model, optimizer, and
    loss-trace maximum absolute differences are `0.0`; scheduler, order/cursor, CPU RNG,
    and CUDA RNG are identical. Runtime evidence:
    `p8_density_phys_v1/resume_parity/rotation_0/seed_42/`; tracked report:
    `docs/evidence/p8/density_d0_resume_parity.json`.
  - [x] Pass one complete-epoch canary with 100% unique training-unit exposure, zero
    repeats, all four shell voxel counts exact, persistent loss logging, and complete
    validation-unit field metrics. The canary sees 13,664/13,664 units and all 65.87M
    owned voxels once, validates all 4,602 units, and reaches macro-shell
    `R2(delta_R7)=0.63158` after one epoch (`0.72340/0.71081/0.68496/0.40716` by shell;
    overall `0.58375`). This is a pipeline/learnability gate, not a converged or
    eigenvalue result. Runtime evidence:
    `p8_density_phys_v1/d0_runs/rotation_0/seed_42/canary_v1/`; tracked reports:
    `docs/evidence/p8/density_d0_canary_{summary,manifest}.json`.
- [x] Train rotation 0 with complete-exposure density epochs and persistent loss/field
  validation curves; no direct eigenvalue/tensor loss is allowed in D0. All 20 frozen
  epochs completed and epoch 16 was selected at complete-validation macro-shell
  `R2(delta_R7)=0.69678`. Runtime:
  `p8_density_phys_v1/d0_runs/rotation_0/seed_42/scientific_v1/`.
- [x] Stitch overlapping density cores in an order- and subdivision-invariant manner;
  report overlap disagreement and supported-volume coverage before the one global FFT
  solve per cap. All 21,910 cores give 100% NGC/SGC supported coverage. Expanded-
  context and subdivision NRMSE are `3.44e-4` and `1.96e-7`; both pass the frozen P6
  tolerances. Runtime:
  `p8_density_phys_v1/d0_stitched/rotation_0/seed_42/`.
- [x] Evaluate field spectra/PDF/tails, trace/tensor/eigenvalue/orientation metrics,
  learned long-mode convergence, and the full P8 per-galaxy suite for both oracle and
  deployable sampling rows. The stitched field has overall/macro-shell
  `R2(delta_R7)=0.62005/0.67563`; deployable raw lambda1 macro is `0.47234`, versus
  `0.50700` for matched U-PATCH. Tensor-component R2 spans `0.82490--0.86567`, and raw
  web-class behavior improves despite the lower point R2. The learned-context curve
  converges by approximately 240--360 Mpc but leaves an eigenvalue error plateau near
  11.4% of reference scatter. Runtime:
  `p8_density_phys_v1/d0_evaluation/rotation_0/seed_42/` and
  `p8_density_phys_v1/d0_learned_context/rotation_0/seed_42/`.
- [x] Apply the registered rotation-2 continuation rule and write
  `DENSITY_FIRST_BASELINE_DECISION`; open at most the single pre-registered D1 auxiliary
  if its exact trigger is met. D0 fails both the within-0.03 point gate (`-0.03466`) and
  supported-shell gate, so it is not promoted as the primary point estimator. It
  continues to rotation 2 only as a secondary field/tensor candidate because it passes
  the pre-declared tensor/eigenvector-benefit route and improves raw balanced accuracy,
  void recall, and knot recall. The D1 trigger is met: register exactly one fixed
  auxiliary before training; no loss sweep is authorized. Runtime decision:
  `p8_density_phys_v1/d0_decision/rotation_0/seed_42/`; tracked evidence:
  `docs/evidence/p8/density_first_rotation0_closeout.json`.
- [x] Rescore the frozen D0 field without retraining on the DarkAI-like diagnostic
  subset: NGC science support, cell-centre `0.15<z<0.4`, equal weight per 5-Mpc grid
  cell, separate selected-volume mean subtraction, no random/window deconvolution,
  and sign-threshold grid classes. The selected 33,192,463 cells give mode-weighted
  `P_cross/P_true=0.85516` and `r(k)=0.93689` over
  `0.02<=k<0.08 h/Mpc`, compared with `0.76037/0.88319` over
  `0.08<=k<0.20` and `0.46789/0.64592` over `0.20<=k<0.40`.
  Sign-threshold void/sheet/filament/knot recall is
  `0.68605/0.76551/0.76407/0.61941` (balanced accuracy `0.70876`). The subset and
  volume weighting recover some of the apparent external-method gap, but leave clear
  amplitude suppression and do not reverse the primary D0 point-estimator NO-GO.
  Exact spectra and confusion matrices are in
  `docs/evidence/p8/density_d0_darkai_like_rescore.json` and runtime
  `p8_density_phys_v1/d0_darkai_like_rescore/rotation_0/seed_42/`.
- [x] Close the previously opened D0 rotation-2 secondary replication as
  `NOT_RUN_SUPERSEDED_BY_PH000_FREEZE`. This is not a failed replication: the new
  diagnostic established that D0 remains scientifically interesting but does not earn
  further same-phase optimization before the independent-phase gate.
- [x] Close D1 as `NOT_RUN_SUPERSEDED_BY_PH000_FREEZE`. Do not tune a downstream
  auxiliary loss on repeatedly inspected `ph000` folds after the density objective
  failed primary adoption. A density/tensor auxiliary may be reconsidered only from a
  fresh-phase residual diagnosis, with a newly frozen contract.

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
- [x] Close the historical one-epoch canary line as superseded by the completed
  exposure-aware recovery and long-horizon runs: complete/no-repeat epoch exposure,
  full-fold validation, atomic resume, loss accounting, and persistent curves were
  exercised in the shipping trainers before the two-rotation science decisions.
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
- [x] Complete matched full-cap exact DTFE and global classical-plus-local-residual
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
- [x] Reapply the classical adoption and promotion gates. U-PATCH-BRIGHT is the frozen
  learned handoff; G-PATCH is the non-promoted runner-up; U-CIC is a sparse-shell
  NO-GO; D0 is not a primary point estimator; F-PATCH v2_A is a resource NO-GO.
- [x] Close five-fold/three-seed expansion as
  `NOT_RUN_DEFERRED_TO_P10_INDEPENDENT_PHASES`. Spend the replication budget on
  independent cosmic phases rather than adaptively optimizing the already inspected
  `ph000` folds.
- [x] Freeze standalone U-PATCH as `U-PATCH-BRIGHT_REFERENCE`, the current two-rotation
  learned candidate; do not call it production-approved before five-fold/seed and P10.
- [x] Complete the BGS_FAINT F0 feasibility audit and register the conditional
  response-complete catalogue build in P8.8.
- [x] Keep log-gap, FMPE/NPE, JEPA, HOD, and broad architecture branches gated while the
  deterministic recovery remains open.
- [x] Freeze validation predictions, reports, diagnostics, configs, and the machine-readable
  screen decision under `docs/evidence/p8/` and the runtime root.
- [x] Supersede the short-screen `INCONCLUSIVE_OPTIMIZATION_AUDIT_REQUIRED` state with
  the completed recovery/extension decisions; retain F-PATCH v2_A as resource NO-GO
  and CIC as the classical handoff.
- [x] Close P8 with runtime marker
  `/pscratch/sd/d/dkololgi/abacus/p8_closeout_v1/P8_COMPLETE` and tracked decision
  `docs/evidence/p8/p8_final_decision.json`. This freezes `ph000` development; it does
  not authorize a production VAC. P10 fresh-phase transfer remains blocking.

---

## 7. Complementarity and hybrid models

### P9 — Residual correlation audit

**Status:** DEFERRED TO P10 OUT-OF-PHASE RESIDUALS; NO NEW `ph000` HYBRID AUTHORIZED
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

**Status:** U-PATCH ARM-A COMPLETE AND FROZEN AT EPOCH 20; THE CORRECTED G-PATCH
SCHEDULE CONTROL AND MATCHED CIC/DTFE ROWS ARE COMPLETE; U-PATCH REMAINS THE
PH006-SELECTED DETERMINISTIC LEADER; PH001 REMAINS SEALED; MULTITRACER, OUT-OF-FOLD
P12, DENSE-TEACHER/VIEW-LADDER, AND ARMS B/C GATES ARE ACTIVE
**Duration:** scope after one-phase benchmark; likely days to weeks

Reserve:

- ph000, ph002–ph005: five phase-balanced training phases;
- ph006: phase-level validation and calibration;
- ph001: sealed blind phase.

#### P10.0 — `ph000` development boundary

It is reasonable to defer the expensive multi-phase training campaign until the
remaining representation and observation questions have been reduced to one or two
finalists. It is not acceptable to optimize `ph000` without a terminal boundary and
then describe the result as more generalisable. Repeated decisions against the same
blocked folds create adaptive benchmark overfitting even when the supervised rows are
spatially disjoint.

P8 closed `ph000` as an independent development/evaluation benchmark on 2026-08-10.
D0 rotation 2 and D1 were explicitly stopped without execution after the DarkAI-like
diagnostic; they are not failed experiments. This does **not** make `ph000` invalid as
training data. The final multi-phase models are initialized from scratch, and training
data are allowed to have influenced the historical architecture choice. The independent
generalisation evidence comes from ph006 model selection and the one-open ph001 test,
not from any subsequent score on ph000. The remaining authorized `ph000` activity is:

1. implement and canary the P10.1 final/degraded-view loaders and response channels;
2. include `ph000` with equal phase-level sampling weight in Arms A--C, while retaining
   its development-contaminated provenance and never treating its training-set score as
   production-transfer evidence;
3. open one paired-view P11 experiment only after the view ladder is frozen and the
   P12 summary-information headroom sub-gate passes; retain dense-teacher transfer as
   advisory evidence and run P11 as a non-blocking parallel challenger rather than a
   prerequisite for P12.

No additional **ph000-only** architecture, loss, feature, context, cell-size, residual,
or posterior sweep opens from a null result, and no old ph000 checkpoint initializes the
production run. When the authorized decisions close, freeze the surviving encoder(s),
D0 field contract if retained, degradation recipes, response schema, target/metric code,
transformations, compute budgets, and acceptance rules in a signed manifest. Model
selection occurs on ph006; ph001 remains a one-open blind test and may not be used to
tune this list.

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

1. Freeze a versioned phase registry for ph001--ph006, with phase roles, source
   paths, observation products, output roots, and target conventions.
2. Use one scientifically uniform particle sample in every phase:
   `field_rv_A + field_rv_B` and `halo_rv_A + halo_rv_B`, i.e. the documented
   3% A subsample plus 7% B subsample for 10% total. A phase may not silently fall
   back to A-only or a full particle snapshot.
3. Treat storage location as an operational difference, not a scientific one:
   A is online on CFS for every phase; B is restored from the per-phase HPSS halo
   tar archives for ph001--ph005; ph006 B is already online on CFS. The existing
   restored ph000 A+B products remain the convention reference.
4. Verify every B source before target generation: exactly 34 field ASDF slabs,
   exactly 34 halo ASDF slabs, both checksum manifests, expected archive/member
   sizes, POSIX checksums after restore, ASDF readability, simulation phase, and
   redshift 0.2 metadata.
5. Restore and process one phase at a time. Before restore, prove scratch free
   space exceeds the registered archive payload plus working headroom. Write an
   atomic `B_STAGE_COMPLETE` marker only after verification. Never make cleanup
   implicit; remove a verified staged B payload only after its compact truth
   products, manifests, and checksums exist and after explicit review.
6. Benchmark one complete 2048-cubed target-generation run on ph002. Record restore,
   density, T-web, annotation, wall time, node-hours, peak scratch, output size, and
   failures. Validate its convention against ph000 before launching the remaining
   phase chains.

The authoritative target contract is:

| Quantity | Frozen value |
| --- | --- |
| simulation/cosmology | AbacusSummit base c000 |
| target epoch | z=0.2 |
| particle sampling | `field_rv_A+B` and `halo_rv_A+B`, 10% total |
| box/grid | 2000 Mpc/h, 2048 cubed |
| assignment | TSC |
| tidal smoothing | Gaussian R=7 Mpc/h |
| eigenvalue order | lambda1 <= lambda2 <= lambda3 |
| web threshold | 0.2 |

The registry and preflight artifacts are part of the scientific contract. Phase and
storage-source identifiers are provenance only and may never enter model features.

Per-phase chain:

~~~text
phase registry + source preflight
  -> verified A + staged/verified B
  -> 10% density field
  -> CACTUS/T-web grid
  -> compact halo/galaxy truth annotation
  -> P1 canonical catalogue
  -> representation products required by each finalist
       -> P2 graph/metrics for G-PATCH
       -> P3 fields for U-PATCH/D0
  -> shared P4 patch/evaluation manifest
  -> PHASE_COMPLETE manifest
~~~

P2 and P3 are parallel representation branches, not universal serial prerequisites.
Build both for the current U-PATCH/G-PATCH comparison; a future finalist may consume
only one. All model families must nevertheless share the same phase, spatial examples,
authoritative target rows, folds, and evaluation contract.

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

- [x] Verify that the 7% B particle products exist for every required phase and
  register a uniform 10% A+B target contract. ph001--ph005 B is in HPSS; ph006 B is
  online on CFS.
- [x] Freeze and validate the versioned ph001--ph006 phase registry:
  `configs/p10_phase_registry_v1.json`.
- [x] Implement and test exact HPSS listing/payload preflight, POSIX checksum
  verification, ASDF phase/redshift checks, atomic completion markers, idempotent
  reuse, scratch-headroom enforcement, and the no-implicit-cleanup guard:
  `workflows/abacus_tweb/p10_{phase_assets,stage_particle_b}.py`.
- [x] Export a tracked particle, CutSky, forFA, potential-assignment,
  fibre-assignment, and LSS source-readiness inventory for ph001--ph006:
  `docs/evidence/p10/asset_inventory.json`. All six source gates pass.
- [x] Record the exact ph002 restore preflight (34 field-B slabs, 34 halo-B slabs,
  two checksum manifests, 168,059,673,012-byte payload):
  `docs/evidence/p10/ph002_b_restore_preflight.json`.
- [x] Complete the ph002 restore and post-restore verification. All 34 field-B
  and 34 halo-B ASDF slabs are readable, both checksum manifests pass, and the
  atomic marker is
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/particle_b/AbacusSummit_base_c000_ph002/B_STAGE_COMPLETE.json`.
  The machine-readable result is
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002_b_stage_result.json`; the
  restore log is
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/logs/ph002_b_stage.log`.
- [x] Implement the phase-explicit streaming A+B TSC density builder and verify its
  input contract against online ph006:
  `workflows/abacus_tweb/p10_build_density_field.py` and
  `docs/evidence/p10/ph006_density_input_preflight.json`.
- [x] Run the ph002 reduced-grid/four-slab technical A+B TSC canary and require
  particle-count conservation before the production-resolution benchmark. The
  authoritative 1024-cubed canary deposited 962,897,612 particles from one slab
  in each of field-A, halo-A, field-B, and halo-B with relative count error
  `3.6559e-7`, 101.45 seconds build time, and 20.49 GB peak RSS. Artifacts:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/targets/density/canary/AbacusSummit_base_c000_ph002_z0.200_ngrid1024_ab10_tsc_counts_1perdir.{npy,manifest.json}`;
  runtime logs:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/logs/ph002_density_canary1024_56669668.{out,err}`.
- [x] Build the complete ph002 2048-cubed, 136-slab 10% A+B TSC density field.
  It contains `33,022,530,364` particles, has relative count error
  `9.4491e-8`, required 2833.80 seconds, and is stored at
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/targets/density/AbacusSummit_base_c000_ph002_z0.200_ngrid2048_ab10_tsc_counts.{npy,manifest.json}`.
- [x] Implement and run the phase-generic T-web MPI stage. An eight-rank launch
  failed cleanly before output because its largest complex transpose message was
  exactly 2 GiB; `p10_run_tweb.py` now preflights that MPI limit. The registered
  16-rank/four-node layout completed in 7m39s and wrote 16 contiguous rank products
  totaling 111,669,184,864 bytes. Atomic gate:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json`.
- [x] Reconstruct the compact ph002 BRIGHT parent catalogue with halo linkage by
  exact matching to the frozen public forFA BRIGHT rows. All `10,619,510` rows
  pass TARGETID and RA/DEC/true-z/RSD-z/magnitude parity with zero maximum
  discrepancy. Artifacts:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/bright_parent/ph002_bgs_bright_parent_linkage.fits{,.complete.json}`.
- [x] Complete and validate compact ph002 T-web annotation on the BRIGHT parent.
  All `10,619,510` rows map, none are skipped, and every row has finite ordered
  eigenvalues and threshold-consistent CWEB class. Runtime was 5.81 minutes.
  Product and audit:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/annotated_parent/ph002_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits` and
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/logs/ph002_annotated_parent_audit_56762868.out`.
- [x] Join the annotated parent truth to the frozen **full** final LSS observation
  view by TARGETID and prove row, redshift, linkage, and label completeness
  contracts. A pre-P1 lineage audit caught that `BGS_BRIGHT-02` is a restricted
  tracer product, not the mock-2 spelling: its 1,427,814-row joined output is
  deprecated and forbidden as a training input. The correct unsuffixed
  `BGS_BRIGHT_full_HPmapcut.dat.fits` source contains `9,214,624` rows,
  `7,370,124` successful redshifts, and `4,988,277` successful rows in
  `0.15<=z<0.55`. The corrected 7,370,124-row join has unique TARGETIDs,
  complete labels and exact RA/DEC linkage; it completed in 51.25 seconds.
  Artifacts:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/observed/ph002_bgs_bright_full_observed_with_tweb.fits{,.complete.json}`.
- [x] Build and validate ph002 P1 canonical indexing. The phase-generic builder
  freezes the successful-redshift FITS rows, identity node IDs, Galactic-cap labels,
  shell/active/context masks, and Planck18 Cartesian points in Mpc:
  `workflows/abacus_tweb/p10_build_phase_index.py`. The result has `7,370,124`
  parent rows, `6,256,475` context rows, `4,988,277` active targets and shell counts
  `2,714,747 / 1,649,261 / 553,052 / 71,217`. The `2,379,758` rows with invalid
  BOX_INDEX remain context-only. Artifacts:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p1_canonical/{canonical_index.npz,points.npy,CATALOGUE_COMPLETE.json}`.
- [x] Build and validate the fresh ph002 P2 graph/metric branch.
  - [x] Run the first disconnected-cap Delaunay construction from the exact P1 points.
    Allocation `56763306` processed more than 148 million NGC and 61 million SGC
    simplices but ended without an atomic marker or saved output. The output directory
    is empty; treat this only as `INCOMPLETE_NO_ATOMIC_ARTIFACT`, not a graph product.
    Log: `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/logs/ph002_p2_graph_56763306.out`.
  - [x] Add cap-level checkpoint/resume and atomic validation to the phase-generic P2
    builder so a completed cap survives interruption of the second cap.
    Implemented in `workflows/abacus_tweb/p10_build_phase_graph.py`; NGC/SGC outputs
    retain immutable P1 parent-row IDs and are hash-gated before merge.
  - [x] Rerun/finish NGC+SGC graph construction, validate disconnected-cap identity and
    exact P1 row mapping, and write the atomic graph marker.
  - [x] Compute and validate the frozen global node/edge metrics in `rapids-gnn`, then
    write the P2 metric marker.
- [x] Finish sequential B restores for ph003, ph004, and ph005. The persistent
  supervisor completed cleanly at 2026-08-12T22:05:00Z. All three phases have verified
  34+34 B slabs, both checksum manifests, readable phase/redshift headers, and atomic
  `B_STAGE_COMPLETE.json` markers. Verified payloads are 168,068,368,057 /
  168,066,274,916 / 168,047,687,099 bytes. ph006 B remains verified online on CFS.
- [x] Benchmark one 2048-cubed target-generation run and record cost/storage.
  The ph002 phase directory occupies about 142 GiB and the staged B particles
  another 157 GiB. Density, T-web and annotation required 47.23, 7.65 and
  5.81 minutes respectively; B restoration required 2.39 hours. The failed
  eight-rank T-web attempt is retained as MPI-layout evidence and wrote no product.
- [x] Validate target convention and annotation parity against ph000. Both phases
  record c000, z=0.2, 2000 Mpc/h, 2048-cubed, R=7 Mpc/h and threshold 0.2;
  finite/order/class checks pass on all ph002 rows and a 100,000-row ph000 sample.
  Tracked evidence: `docs/evidence/p10/ph002_convention_parity.json`.
- [x] Normalize the frozen ph000 development/reference products under the same P10
  phase-tree interface. These products are now eligible as the fifth training phase for
  models initialized from scratch; the historical `reference` provenance and marker
  name remain unchanged. The passing
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/REFERENCE_PHASE_COMPLETE.json`
  binds 193 copied files totaling `162,057,995,430` bytes, preserves historical
  manifests byte-exactly, and exposes normalized target, catalogue, P1, P2, P3, P4
  and schema paths. The historical graph is already cap-disconnected but predates
  cap-checkpoint markers; record it as `legacy_global_graph`, not as missing or as
  newly cap-built. Tracked evidence:
  `docs/evidence/p10/ph000_reference_import_20260813.json`.
- [x] Complete the ph006 validation-phase truth chain.
  - [x] Build and validate the full 2048-cubed 10% A+B density field. It processes
    `33,022,530,364` particles with relative count error `9.4672e-8` in 2696.58
    seconds; the wrapped process exits zero after 45m05s with 54,538,056 KiB peak RSS.
    Runtime log:
    `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/logs/ph006_density_56762868.out`.
  - [x] Run and validate the ph006 T-web solve. Sixteen contiguous rank files cover
    x=[0,2048) and total `111,669,184,864` bytes. Atomic marker:
    `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json`.
  - [x] Build/validate parent linkage, annotation and the full observed-truth join.
    The observed product has `7,330,186` rows, complete labels, exact positional
    linkage, one explicitly recorded float32-threshold ambiguity and zero CWEB
    disagreements away from the threshold.
  - [x] Build/validate ph006 P1 canonical indexing: `6,248,640` context rows,
    `4,968,208` authoritative active targets and all four nonempty redshift shells.
  - [x] Build/validate ph006 P2 graph/metrics and P3 fields as parallel branches.
    P2 has `55,873,345` Delaunay edges, zero cross-cap edges and `187,094,689`
    context-union pairs; P3 passes all frozen nine-channel field gates.
  - [x] Build/validate the shared ph006 P4 validation manifest and
    `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/PHASE_COMPLETE.json`.
    Geometry, graph support, field support, deterministic rebuild and the stronger
    catalogue--field--target closure all pass.
- [x] Audit live per-phase readiness after the ph002/ph006 launches. Exact evidence:
  `docs/evidence/p10/multiphase_readiness_20260813.json`. No Slurm job was active at
  audit time; no phase currently carries `PHASE_COMPLETE`.
- [x] Implement the resumable phase-stage orchestration and cross-phase validation
  contract. `run_p10_phase_stage.sh` skips passing atomic artifacts and exposes P1,
  cap-graph, P2-post and P3/P4 work units; `p10_materialize_phase_schemas.py` permits
  only `catalogue_id` to differ from the tracked ph000 P3/P4 schemas; and
  `p10_validate_phase_products.py` separates exact physics gates from cosmic-variance
  diagnostics.
- [x] Implement and complete sealed ph001 P1--P4 input construction without target
  access. The
  blind observed FITS dtype has no CWEB/eigenvalue columns; density/T-web products are
  forbidden; all representation and P4 gates pass and the terminal marker is
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/BLIND_INPUT_COMPLETE.json`.
  Truth unsealing and
  scored evaluation remain downstream of frozen predictions.
- [x] Complete the automated ph001--ph005 campaign.
  - [x] Complete concurrent ph003--ph005 production density builds in interactive
    allocation `56857414`; every phase has a conserved 2048-cubed density manifest.
  - [x] Complete ph003--ph005 T-web/observed/P1 chains.
  - [x] Complete two cap checkpoints, merged graph, RAPIDS metrics and radius union for
    every ph001--ph005 catalogue.
  - [x] Complete P3 fields and shared P4 geometry/support for every ph001--ph005
    catalogue.
  - [x] Write passing `PHASE_COMPLETE.json` for ph002--ph005 and
    `BLIND_INPUT_COMPLETE.json` for ph001.
  - [x] Run the cross-phase physics/representation audit, including the stronger
    catalogue--field--target closure for ph002--ph005. Evidence:
    `docs/evidence/p10/multiphase_p1_p4_completion_20260813.json`; live product matrix:
    `docs/evidence/p10/multiphase_live_status.json`.
- [x] Complete ph002 model representations and shared examples.
  - [x] Complete target truth, full observed truth and P1 canonical indexing.
  - [x] Complete P2 graph and global graph metrics for G-PATCH.
  - [x] Complete P3 observational fields for U-PATCH/D0.
  - [x] Complete the shared P4 patch/evaluation manifest and `PHASE_COMPLETE`.
- [x] Complete ph003 training-phase products.
  - [x] Stage and verify 10% A+B particle inputs.
  - [x] Build density, T-web, parent linkage/annotation and full observed truth.
  - [x] Build P1 canonical indexing.
  - [x] Build P2 graph/metrics and P3 fields as parallel representation branches.
  - [x] Build the shared P4 patch/evaluation manifest and `PHASE_COMPLETE`.
- [x] Complete ph004 training-phase products.
  - [x] Stage and verify 10% A+B particle inputs.
  - [x] Build density, T-web, parent linkage/annotation and full observed truth.
  - [x] Build P1 canonical indexing.
  - [x] Build P2 graph/metrics and P3 fields as parallel representation branches.
  - [x] Build the shared P4 patch/evaluation manifest and `PHASE_COMPLETE`.
- [x] Complete ph005 training-phase products.
  - [x] Stage and verify 10% A+B particle inputs.
  - [x] Build density, T-web, parent linkage/annotation and full observed truth.
  - [x] Build P1 canonical indexing.
  - [x] Build P2 graph/metrics and P3 fields as parallel representation branches.
  - [x] Build the shared P4 patch/evaluation manifest and `PHASE_COMPLETE`.
- [x] Satisfy the deterministic training-readiness gate.
  - [x] U-PATCH training pool: P1+P3+P4 complete for ph000 and ph002--ph005.
  - [x] U-PATCH validation: build matching ph006 P1+P3+P4 products.
  - [x] G-PATCH training pool: P1+P2 graph/metrics+P4 complete for ph000 and
    ph002--ph005.
  - [x] G-PATCH validation: build matching ph006 P1+P2 graph/metrics+P4 products.
  - [x] Implement a phase-balanced training sampler over ph000 and ph002--ph005. Phase may
    determine grouping and provenance but must never enter the model features; no
    single phase or dense shell may dominate an epoch accidentally.
  - [x] Fit graph-feature, field-channel and target transforms on the five-phase
    ph000+ph002--ph005 training mixture only; serialize them once and apply them
    unchanged to ph006.
  - [x] Prove row identity, target convention, patch/core ownership, complete-epoch
    coverage, resume parity, weighted-loss accounting and deterministic validation
    for both U-PATCH and G-PATCH across every contributing phase.
  - [x] Write the passing atomic loader marker only after those canaries:
    `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/TRAINING_LOADER_READY.json`.
  - [x] Complete the cheap response-source/schema audit in P3b and write
    `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P10_RESPONSE_SOURCES_READY.json`
    before any new multi-phase GPU training. This freezes the future Arm-C interface
    and avoids discovering after Arm A that the mock and DESI response fields are not
    homologous; it does not require full P3b field export before Arm A.
  - [x] Freeze adapter inventory and shared transforms. The field, graph, selection,
    and target transforms are fit only on ph000+ph002--ph005 and applied unchanged to
    ph006 and sealed ph001. Evidence: `docs/evidence/p10/adapter_inventory_20260814.json`
    and `docs/evidence/p10/transforms_frozen_20260814.json`.
  - [x] Pass the registered real extraction/epoch canary: 84,446 training cores exactly
    once per epoch, 16,796 deterministic ph006 validation cores, balanced phase prefix,
    resume parity, exact weighted-loss reconstruction, row/target alignment, and
    deterministic graph/field extraction for every visible phase. Evidence:
    `docs/evidence/p10/training_loader_ready_20260814.json`.
- [x] Run the terminal P1--P4 cross-phase physics/representation audit through
  ph006. All exact gates pass; evidence:
  `docs/evidence/p10/multiphase_p1_p4_complete_with_ph006_20260813.json`. The
  machine-readable live matrix is `docs/evidence/p10/multiphase_live_status.json`.
- [x] Freeze phase roles and a signed blind-evaluation manifest. The one-open ph001
  contract is `configs/p10_blind_evaluation_v1.json`; tracked evidence and marker copies
  are `docs/evidence/p10/blind_evaluation_frozen_20260814.json` and
  `docs/evidence/p10/blind_evaluation_ready_marker_20260814.json`.
- [ ] Train and freeze finalists without reading ph001 truth metrics.
  - [x] Complete the freshly initialized five-phase U-PATCH Arm-A trajectory. All
    20 registered epochs and 1,688,920 optimizer updates completed; epoch 20 is the
    frozen ph006 selection checkpoint with four-shell macro `R2(lambda1)=0.57310`,
    first-three-shell macro `0.63464`, per-shell
    `0.69190/0.64159/0.57042/0.38850`, and cosine learning rate exactly zero.
    Epoch 18--20 improved only `0.00149`, below the registered `0.002` material-gain
    scale. Do not append epochs to the exhausted schedule; a new seed or low-LR
    experiment must be separately named and is not on the critical path.
    A prediction-conditioned calibration audit also passes: on ph006 the pooled
    `E[truth|prediction]` slopes are `0.9995/0.9942/1.0013` for lambda1--lambda3
    and normalized weighted mean absolute calibration errors are
    `0.0066/0.0075/0.0104`. Thus the sub-unity `prediction|truth` slopes diagnose
    conditional-mean shrinkage rather than an affine point-calibration defect. Evidence:
    `docs/evidence/p10/prediction_conditioned_calibration.json`.
  - [x] Resolve the G-PATCH optimization control before comparing representations.
    The original run is frozen after epoch 6 as diagnostic evidence: its LR remained
    `1.59e-3` after 506,676 updates because the epoch-scaled schedule was stretched by
    the five-phase epoch, validation oscillated, and prediction variance collapsed.
    Paired LR `2e-4` and `5e-4` fresh canaries completed three exact epochs with an
    explicit 400,000-update cosine horizon, pre-clip gradient telemetry and unchanged
    phase/patch objective. The `2e-4` canary is better: macro `R2(lambda1)=0.46986`,
    first-three-shell macro `0.51260`, and per-shell
    `0.53897/0.51258/0.48625/0.34164`, versus `0.45504` for `5e-4`. This repairs the
    optimization-invalid collapse but does not catch U-PATCH (`0.57310`); freeze G as a
    non-selected control rather than extending an architecture branch.
  - [x] Complete matched ph006 CIC and exact-DTFE rows with affine calibration fitted
    only on ph000+ph002--ph005 under the frozen phase/shell weights. CIC consumes the
    immutable P3 counts/expected-counts. DTFE must build a separate exact
    piecewise-linear density raster for every visible phase, convert it with that
    phase's expected-count response, and then apply the same fixed R=7 Mpc/h tensor
    solve. The expensive DTFE branch is resumable and may finish after CIC, but the
    older single-phase/P8 DTFE number is not a substitute for this matched row. The
    train-affine CIC row has macro `0.31139`, first-three-shell macro `0.51877`, and
    shells `0.51582/0.54325/0.49725/-0.31074`. Exact DTFE has macro `0.01607`,
    first-three `0.05337`, and shells `0.02077/0.05823/0.08111/-0.09583`. Under this
    fully matched independent-phase contract U-PATCH therefore beats both classical
    estimators overall and in every one of the first three supported shells; this is
    not a macro-only win caused by classical sparse-shell collapse. Evidence:
    `docs/evidence/p10/{cic,dtfe}_ph006_complete_20260820.json`.
  - [ ] Run the paired multi-phase BGS_FAINT information test only after its source
    audit and phase-matched view builder pass. Keep supervision and released targets on
    BGS_BRIGHT. Compare BRIGHT-only against (a) real BRIGHT+FAINT context and (b) a
    cap- and redshift-stratified angular-scramble FAINT null with identical tracer
    counts/response. Apply the same ph000+ph002--ph005 training and ph006 selection
    roles. A Proxy-minus-Null gain, not Proxy-minus-Bright alone, identifies additional
    spatial information; the Proxy remains non-production until the final Loa Faint
    selection/photometry contract exists. The six-visible-phase official-source audit
    is launched after CIC finalization; source feasibility is not itself a completed
    Proxy/Null view or a training result.
    - [x] Resolve every official assigned-FAINT source through the frozen phase
      registry rather than filesystem discovery. In particular, ph000 explicitly
      records the historical `fba0_bkp` path; this is provenance, not a silent
      fallback.
    - [x] Audit ph000/ph002--ph006 without opening ph001. Every phase contains
      `7.15--7.21` million unique assigned FAINT targets, every assigned FAINT
      TARGETID matches phase-correct forFA truth, and every matched `RSDZ` is finite
      and positive. Evidence:
      `docs/evidence/p10/multitracer_source_audit_20260820.json`.
    - [x] Implement one phase-generic builder that preserves BGS_BRIGHT supervision,
      writes separate FAINT count/expected-count/contrast channels on the immutable
      P3 grids, and creates the angular-scramble Null within cap and narrow-redshift
      strata while holding the radial distribution and tracer count fixed.
    - [x] Implement the matched six-channel U-PATCH adapter and trainer. The Proxy and
      Null must share architecture, initialization seed, phase weights, optimizer,
      patch manifest, Bright targets and validation contract.
    - [x] Freeze the six-phase source-audit and view-ready markers, including hashes,
      assignment/truth join counts, per-cap/per-shell counts and Null invariants.
      The context-limited assigned FAINT rows are `6,804,198--6,850,218` per phase;
      each Proxy and Null conserves its deposited tracer counts, the Null preserves
      every radius and permutes directions within cap and `Delta-z=0.01`, and the
      shared FAINT selection/normalization is fitted on the five training phases only.
      Evidence: `docs/evidence/p10/p10_multitracer_views_ready_20260820.json`.
    - [ ] Freeze Proxy and Null after complete epoch 15 on ph000+ph002--ph005, select
      on ph006,
      and report Proxy-minus-Null with spatial-block uncertainty. Do not promote from
      Proxy-minus-Bright alone. The four-GPU interactive chain launched as job
      `57292623`; at the 2026-08-21 audit it was healthy on attempt 7/24, job
      `57357340`. Both matched seed-42 runs passed distinct two-update technical
      canaries, completed five full epochs, and were active in epoch 6. Interim
      validation rows are explicitly non-decisional; paired evaluation waits for both
      20-epoch schedules and frozen best checkpoints.
      - [x] Reach 11 complete epochs for both matched models and verify complete
        84,446-core coverage, finite losses and checkpoint resume. On 2026-08-22 both
        were active in epoch 12 on job `57404937`.
      - [x] Record the interim causal contrast without promoting it. Proxy has best
        macro `0.66394` at epoch 10 and Null has best macro `0.65497` at epoch 11.
        At matched epoch 10 Proxy-minus-Null is `+0.02208` macro, but the overall
        contrast reverses at epoch 11. Proxy retains a fourth-shell advantage of
        `+0.06555/+0.05188` at epochs 10/11. This motivates completion and paired
        spatial uncertainty; it is not yet a production multitracer decision.
      - [ ] Complete both through epoch 15, freeze each best checkpoint and
        compute paired spatial-block confidence intervals for Proxy-minus-Null overall,
        in the first three shells, and in every individual shell.
      - [x] Reframe the pair as a bounded information-content diagnostic. At matched
        epoch 13 Proxy-minus-Null is `+0.02463` macro, `+0.01150` first-three, and
        `+0.00050/+0.00986/+0.02415/+0.06400` by shell. This is preliminary spatial
        information evidence, not a clean response estimate; no further FAINT run is
        authorized before P3b-R/R1 and baseline P12.
- [x] Build the truth-free ph001 graph/field products under the sealed blind-input
  contract.
- [ ] Save ph001 predictions before opening truth.
- [ ] Evaluate ph001 once and record the production-transfer decision.

#### P10.1 — Controlled observation-operator training

**Status:** U-PATCH ARM-A SELECTED ON PH006; G-PATCH OPTIMIZATION CONTROL AND MATCHED
CLASSICAL ROWS COMPLETE; ARMS B/C AWAIT THE FROZEN VIEW LADDER AND P3B RESPONSE-FIELD
EXPORTS; MULTITRACER AND P12 PREPARATION ARE ACTIVE

The catalogue identifiers required to pair and group examples are not automatically
valid conditioning variables. Use this role contract:

| Variable class | Examples | Permitted use |
| --- | --- | --- |
| Grouping/provenance only | phase, observer ID, HOD ID/seed, stage ID, degradation seed, core ID, TARGETID, halo linkage, source hashes | Pairing, outer splits, hierarchical sampling, leakage audit, and stratified metrics; never production inputs |
| Deployable physical/response | continuous redshift or `log ntilde(z)`, LOS, random-derived expected intensity, exposure, completeness, fibre-assignment probability, redshift quality/error, mask distance, effective support | Concatenate as node/voxel channels; broadcast legitimate patch summaries or concatenate them before the output head |
| Baseline-fixed simulation nuisance | fiducial BGS galaxy--halo prescription `H_fid` and its stochastic seed | The first released posterior is explicitly conditional on `H_fid`; HOD ID/seed remain provenance, never production inputs |
| Optional unknown simulation nuisances | alternative HOD/velocity-bias and unobserved failure-recipe parameters | Open only as later robustness interventions; vary and marginalize only if a registered stress test shows material sensitivity |
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
stage, observer, and degradation realization of a latent core in the same outer
phase/spatial split, give the latent core one scientific weight, and rebuild topology,
topology-derived features, fields, and random-response products independently for every
view. Because distinct observers can revisit the same periodic-box structure, the first
P10 test should either use one observer or prove cross-observer grouping by base-box
identity.

The baseline mock ensemble uses its existing BGS galaxy--halo prescription. The target
of the first SBI model is therefore
`p(lambda | x_s, response_s, H_fid)`, not a posterior marginalized over the unknown
DESI HOD. This is a valid conditional estimand and avoids putting new HOD lightcones,
fibre assignment and LSS processing on the critical path. DESI has studied 11 BGS HOD
models (`BGS_0` best fit plus ten posterior samples) controlled by 17 luminosity-dependent
meta-parameters, but only 11 `z=0.2` cubic mocks from one AbacusSummit realization were
available in that analysis; they are not drop-in replacements for the present
ph000--ph006 phase-indexed cut-sky products
([Findlay et al.](https://arxiv.org/html/2411.12023v4)). Locate them
only for the post-P12 held-out-HOD stress test defined below.

Freeze a response-explicit **forward-observation ladder** before Arms B/C. An observation
view is a different survey operator applied to the same latent galaxy population, not a
new universe. The minimum useful ladder is deliberately small:

| View | Catalogue state | Purpose and response requirement |
| --- | --- | --- |
| `V_dense` | Targetable BGS selection inside the accepted imaging footprint, before assignment/redshift losses | Privileged information ceiling; same `M`, explicit target-selection and `ntilde(z)` contract |
| `V_assign` | `V_dense` after the audited fibre-assignment/collision operator | Isolates assignment loss; matched `C_fibre`/PIP or alternate-MTL response |
| `V_final` | Final Path1-like successful-redshift and HP-map-cut selection | Deployment view; `M`, `C_fibre`, `C_z`, quality/error and final source hashes |

Add another view only if it isolates a named effect with an audited deployable response.
Uniform random thinning is a named information control, not a substitute for the survey
observation operator. Every view must save selected IDs, random seed, response fields,
topology/field manifests and source hashes. Hold out at least one response recipe—not
only a seed—from training.

Survey effects are not all treated as extra network features. Freeze this accounting
table with the view ladder:

| Effect/selection | Applied to mocks | Seen by encoder/SBI | External precedent and project reason |
| --- | --- | --- | --- |
| Footprint, imaging vetoes and holes | Apply the same angular support and HP-map cuts; derive binary `M` from full randoms | `M`/expected intensity and mask distance; U-PATCH zero-support cells are distinct from voids; G-PATCH also vetoes edges crossing `M=0` | DESI full randoms represent reachable angular support; ASTRA uses matched random points directly. Our field-first treatment is cheaper and preserves the established encoders. |
| Flux/magnitude/target selection and radial density | Apply the BGS selection in each registered view | Smooth training-fitted `ntilde(z)` or `log mu`; never infer the radial baseline independently per patch | DESI random/catalogue analyses match angular and radial selection. We retain an independently smoothed radial baseline because clustering-random redshifts can be data-linked. |
| Fibre assignment and collisions | Use the real Path1/alternate-MTL assignment lineage where available; random thinning is only a control | `C_fibre` or PIP/completeness response plus rebuilt graph/fields | DESI uses alternate assignment histories/PIP or completeness weighting. Lost neighbours are missing information, so conditioning cannot recreate them but should prevent false certainty. |
| Redshift success and final quality cuts | Apply the calibrated success/failure and final selection | `C_z`, quality/error summaries; SBI width and coverage must respond to them | DESI LSS catalogues explicitly correct varying redshift success and completeness. We need calibrated uncertainty, not merely an average correction. |
| Redshift error | Perturb observed redshift with the audited BGS error model | LOS and redshift-error scale; rebuild positions/topology per view | This changes radial geometry, so a weight alone is insufficient. |
| RSD | Present in the observed mock coordinates by construction | LOS is a physical channel; real-space positions/velocities remain privileged truth | RSD is signal+nuisance in the observed field, not a removable survey mask. |
| Sampling density/shot noise | Arises from every selection; add matched thinning severity controls | Raw counts remain separate from `mu`; posterior contraction/width is tested versus density | Randoms set the expected selection baseline but do not reconstruct missing galaxies. |
| Fiducial HOD/velocity bias | Keep the existing mock prescription fixed for baseline training | No HOD ID input; label inference as conditional on `H_fid` | Published DESI HOD ensembles show a useful later stress test, but their BGS suite is one-realization cubic data rather than production cut sky. |

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

Arm C contains a **bounded, ordered feature/structure ablation**, not an unrestricted
power-set search:

| Test | Change from preceding test | Required interpretation |
| --- | --- | --- |
| `R0` | Frozen P3a/P8 channels on `V_final` | Arm-A reference |
| `R1` | Replace only the occupancy-derived support/expected-count approximation with random-derived `M` and `mu`, preserving the frozen tensor width wherever possible | Isolates the random-reference correction |
| `R2` | `R1` plus audited continuous `C_fibre` and `C_z` (separate channels when identifiable; product plus flags otherwise) | Isolates explicit completeness conditioning |
| `R3-RF` | Replace the compressed random-response summaries with a high-S/N voxel-resolved random-field triplet, paired with the unchanged BRIGHT triplet in the same six-channel U-PATCH interface used by BRIGHT+FAINT | Tests whether the large BRIGHT+FAINT gain is recoverable from an uncompressed empirical response field rather than an additional tracer |
| `R4` | `R2` plus mask-boundary distance; retain the already-established LOS channels | Tests whether the model can identify reduced context near holes/footprint edges; deferred until the response-compression question is answered |
| `G-topology` | For G-PATCH finalists only, compare the same `R2/R4` node features with versus without the mandatory `M=0` edge-crossing veto | Isolates structural boundary leakage; the unsafe result is diagnostic and cannot ship |
| `G-random-node` | For a still-competitive G-PATCH only, merge a bounded random-node sample in the ASTRA style | Optional alternative representation, not a production prerequisite |

Run the registered response ladder `R0/R1 -> R2 -> R3-RF` for the leading U-PATCH
family. R2 and R3-RF may run concurrently once their separate technical canaries pass:
their purpose is a pre-registered representation ladder, not post-hoc selection based
on the unfinished R1 curve. Repeat only the
representation-specific minimum for another encoder that remains competitive. Keep
architecture, target rows, phase/core sampler, optimizer-update budget and seeds fixed.
Do not run all subsets of all response channels. A promoted `R4` must also pass a
leave-one-block-out diagnostic for `M/mu`, completeness and boundary distance to detect
shortcut dependence; this diagnostic is not a second architecture search.

#### R2 response-semantics preflight (2026-08-23)

- [x] Audit the nested `V_dense -> V_assign -> V_final` catalogue views for
  ph000/ph002--ph006 without opening ph001. All six phase reports pass registered row
  identity, target uniqueness, probability-range and view-nesting checks. The assigned
  fraction is `0.79983--0.80330` and the final-view fraction is
  `0.79793--0.80128` across phases.
- [x] Correct the response interpretation. `FRACZ_TILELOCID` is the local fibre-
  assignment completeness quantity, not a redshift-success probability; its ten-bin
  calibration error against `LOCATION_ASSIGNED` is only
  `1.33e-5--4.43e-5`. `FRAC_TLOBS_TILES` is a supplementary TILES-group
  completeness term. Their product approximates total assignment completeness but is
  not to be relabelled as an individual Bernoulli probability without the saved
  component channels.
- [x] Establish the mock redshift-success contract. Every assigned object in every
  visible phase satisfies the registered mock success definition and every clustering
  `WEIGHT_ZFAIL` is exactly one. The mocks therefore contain no continuous learnable
  `C_z` variation homologous to Loa `mod_success_rate`. For the fiducial mock ladder,
  store `C_z=1` with `C_z_informative=false`; do not imply that a model trained on this
  constant can respond to Loa redshift-success variation. That capability requires an
  explicitly degraded/simulated view in P11/P12 closure.
- [x] Close the spatial response-definition gap before writing
  `P10_VIEW_LADDER_READY.json`. Across all six visible phases the raw map covers
  `0.98477--0.98491` of random-supported pixels. The missing pixels are overwhelmingly
  boundary-localised: `0.9826--0.9827` lie within four nside-256 neighbour rings and
  `0.9988--0.9995` within eight. Because their cross-phase Jaccard is only `0.8605`,
  they are not a fixed hole mask. The frozen conservative policy therefore assigns the
  neutral no-competition value `C_fibre=1` inside random support and records
  `C_fibre_defined=0`; it never smooths from neighbouring galaxy density. Outside
  random support the response is zero.
- [x] Build all 12 hash-frozen ph000/ph002--ph006 cap overlays, preserve the two
  assignment components plus defined/identifiability flags, and pass the all-phase
  six-channel loader smoke. Aggregate overlay SHA-256: `6301e764...631cfe6`.
- [x] Pass a balanced 1,000-patch one-A100 canary at `6.260 patches s^-1`, versus
  `9.174 patches s^-1` for R1; record the `31.8%` throughput cost in scheduling.
- [x] Keep ph001 sealed throughout response construction and the technical canary.
- [x] Authorize full R2 on 2026-08-23 without waiting for R1 convergence. The purpose is
  now explicitly frozen as the middle row of the response-compression ladder. It must
  retain the same targets, sampler, optimizer-update budget and ph006 evaluator as
  R1/R3-RF and cannot be reconfigured after seeing either result.

#### R3-RF high-S/N random-field arm (priority 2026-08-23)

R3-RF is the immediate production-priority test. It does **not** add BGS_FAINT and it
does not use clustering-random `Z`. It reuses the frozen all-18 full-random P3b-R
response overlays on the exact P3a geometry. This is a high-S/N, voxel-resolved
empirical response field; it must not be described as a galaxy tracer or as a complete
fibre/redshift-response model.

The six input channels are arranged as two explicit three-channel blocks:

1. unchanged BRIGHT block: normalized `counts`, BRIGHT `log_count_ratio`, and frozen
   BRIGHT `exposure_apodized`;
2. random-response block: normalized `expected_counts_random`, clipped
   `angular_response`, and binary `support_random`.

This is deliberately capacity-matched to the existing BRIGHT+FAINT U-PATCH input width.
`expected_counts_random` receives a training-phase-only `log1p` plus z-score transform;
`angular_response` is centred on unity and clipped only at the frozen contract limits;
`support_random` is identity transformed. Immutable BRIGHT and P3b-R arrays are linked,
not duplicated.

- [x] Define the zero-copy R3-RF adapter and six-channel U-PATCH input contract.
- [x] Add focused tests for channel order, transforms, finite values and model width.
- [x] Freeze ph000/ph002--ph006 product manifests, source hashes, all-18 random IDs,
  training-only normalization and aggregate loader contract; keep ph001 sealed.
  Products: `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r3_random_field_v1/`;
  loader: `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract_r3_random_field/`.
- [x] Pass the all-phase/cap six-channel loader smoke and the 1,000-patch one-GPU
  throughput canary (`11.236 patches s^-1`). Technical marker:
  `training_contract_r3_random_field/P10_R3_RF_TECHNICAL_READY.json`.
- [x] Launch R2 and R3-RF concurrently as independent one-GPU tasks. The initial
  allocation `57475703` ended cleanly after 67 minutes when its non-persistent owner
  shell closed; all four checkpoints survived. Commit `300acba` adds the persistent
  tmux supervisor `workflows/sbi/run_p10_response_ladder_interactive.sh`. It resumed
  seeds 42/43 for both arms on allocation `57489518` and chains one four-GPU interactive
  allocation at a time until terminal markers exist. Run roots are
  `response_training/p10_r2_assignment_v1/` and `response_training/p10_r3_rf_v1/`;
  every task retains the frozen 84,446-patch epoch, 20-epoch cosine schedule,
  checkpoint every 250 updates and automatic resume.
- [x] Reach the registered matched epoch-15 response-ladder comparison with two seeds
  per arm. R2 has macro `0.58143/0.57839` (mean `0.57991`) and R3-RF
  `0.57513/0.56738` (mean `0.57126`), versus matched epoch-15 R0 `0.56003`.
  The gains are modest and reproducible (`+0.01988/+0.01123` in the two-seed means),
  occur mainly in the first three shells, and do not recover the BRIGHT+FAINT result.
  The configured 20-epoch terminal markers remain open; the two-seed means are already
  approximately flat from epoch 14 to 15, so completing them is subordinate to P12-A.
- [ ] Compare R0, R1, R2, R3-RF, FAINT Null and real FAINT at matched optimizer updates
  and frozen ph006 scoring. Report the full epoch histories as well as best checkpoints.
- [ ] Promote R3-RF as the deterministic response representation only if its gain is
  reproducible and its response/boundary diagnostics do not reveal a shortcut. A null
  result retains random response for posterior conditioning and closes the hypothesis
  that response compression explains the BRIGHT+FAINT gap.

A density-matched stochastic random-count arm is deliberately deferred. It becomes the
next diagnostic only if the high-S/N R3-RF field fails to approach the FAINT Null result;
that later arm would test whether FAINT Null gains arise from count-like sampling noise
rather than from the underlying response field.

#### R3-RF-DM and cross-phase FAINT controls (authorized 2026-08-25)

The high-S/N R3-RF result activates the previously deferred density-matched arm, but
the causal question is refined because the existing FAINT Null is not fully
structure-free. It preserves the real FAINT angular-direction multiset inside every
cap and `Delta-z=0.01` stratum; only the radius--direction pairing is permuted. The
projected angular clustering and some coarse three-dimensional structure can therefore
survive. The next controls must separate four distinct ingredients: deterministic
response, sparse point-process texture, realistic but independent tracer clustering,
and correct same-phase tracer information.

**U-PATCH output/target contract.** The canonical U-PATCH does not predict three
eigenvalues at every output voxel. The U-Net produces a latent field on the exact P3
lattice, `grid_sample` evaluates that field at authoritative BRIGHT-galaxy fractional
voxel coordinates, and a point head predicts three scaled linear eigenvalue increments
per galaxy. The 5-comoving-Mpc numerical lattice and the `R=7 Mpc/h` truth smoothing
scale are intentionally distinct. Required parity is:

- exact grid origin, cell size, axis order and cap ownership;
- exact parent/authoritative-galaxy fractional index and interpolation convention;
- exact target row and ordered-increment scaler;
- sufficient context relative to the U-Net receptive field and physical smoothing;
- no claim of voxelwise eigenvalue supervision unless a separate truth-grid contract
  is constructed and validated.

**`R3-RF-DM-v1` sparse response-only triplet.** For each visible phase, sample angular
directions from the frozen all-18 random angular response and restrict them to the
existing FAINT support. Within each cap and `Delta-z=0.01` stratum, match the exact
assigned-FAINT count and randomly pair the sampled directions with the exact FAINT
redshift/radius multiset. Deposit with the same CIC operator on the immutable P3 grid.
The second triplet must be processed exactly like FAINT:
`[sparse counts, sparse log-count-ratio, FAINT exposure]`, using the frozen
training-phase FAINT selection curve and count normalization. This is a diagnostic
one-point-selection match, not a deployable native random catalogue and not an
additional physical tracer.

**`U-BF-XPHASE-NULL-v1` independent-structure control.** Define a fixed donor
derangement over `ph000/ph002--ph005`; pair ph006 with a registered training-phase
donor. A target phase uses its own BRIGHT input, ownership, targets and weights but the
donor phase's complete real-FAINT triplet. This preserves realistic clustered texture,
shot noise, radial selection and network interface while removing correlation with the
target potential. The control is zero-copy only if donor/recipient cap grids are
exactly identical; otherwise it must be rebuilt in common survey coordinates rather
than silently reindexed.

- [x] Register the per-galaxy U-PATCH output and grid/target-alignment contract.
- [x] Register the R3-RF-DM and cross-phase scientific estimands and production guards.
- [x] Audit grid geometry and patch-index parity across every visible phase/cap. The
  phase grids are not byte-identical, so zero-copy donor reindexing is vetoed and every
  donor catalogue is deposited on the recipient P3 grid.
- [x] Build two deterministic R3-RF-DM catalogue-realisation seeds without reading
  tidal labels or ph001; preserve exact cap/fine-redshift counts and redshift multisets.
  Products: `multitracer/strict_controls/r3_rf_dm_seed{1701,2718}_v1/`.
- [x] Export CIC count overlays and multitracer-compatible shadow contracts with exact
  FAINT selection/normalization identity and immutable BRIGHT links.
- [x] Export the cross-phase donor contract, fixed forward derangement and reverse
  diagnostic; prove no phase donates to itself. Products:
  `multitracer/strict_controls/bf_xphase_{forward,reverse}_v1/`.
- [x] Add focused tests and product gates for count conservation, support containment,
  radial-multiset identity, angular independence by construction, channel order, grid
  ownership and sealed-phase access.
- [x] Pass all-phase extraction smokes for all four product roots. Every
  `STRICT_CONTROL_LOADER_SMOKE.json` records six phases, the identical selection hash,
  `sealed_phase_opened=false` and `targets_opened_by_validator=false`.
- [x] Pass a 1,000-update one-A100 throughput canary for the primary R3-RF-DM and
  cross-phase controls. All four seed/control canaries passed in interactive job
  `57583442` before the scientific runs began.
- [x] Complete the matched epoch-5 diagnostic for both strict controls and both model
  seeds. The two-seed macro means are `0.49388` for R3-RF-DM and `0.50295` for
  cross-phase FAINT, versus epoch-5 R0 `0.44071`. Their sparse-shell means are
  `0.35347/0.34859` versus R0 `0.35273`: the early gain is supported-shell
  optimization/representation, not added sparse-shell cosmological information.
- [ ] Train the primary R3-RF-DM realization and cross-phase Null with the frozen
  six-channel U-PATCH, seed, phase/core sampler, optimizer-update schedule and ph006
  evaluator. Extend the second random realization only after the first reaches the
  registered epoch-10 diagnostic or earlier technical failure. Recovered job
  `57670700` is running the four primary trajectories to epoch 10; persistent handoff
  `workflows/sbi/wait_then_run_p10_strict_epoch15.sh` then resumes the same checkpoints
  to epoch 15 without exceeding the two-allocation limit.
- [ ] Compare matched epoch 10 and epoch 15 histories for R0/R1/R2/R3-RF/R3-RF-DM,
  cross-phase Null, old FAINT Null and real FAINT. Report two-seed/random-realisation
  sensitivity, every shell, slopes, variance ratios and response/boundary strata.
- [ ] If a fixed sparse-random field helps, run refreshed-realisation training and
  multi-realisation inference before any posterior use; arbitrary random noise may not
  create narrower production posteriors.

Interpretation is frozen as follows:

| Contrast | Question |
| --- | --- |
| `R3-RF-DM - R3-RF` | Does sparse point-process texture help beyond the high-S/N response? |
| `old FAINT Null - cross-phase Null` | How much same-phase structure survives the old pairing null? |
| `real FAINT - cross-phase Null` | How much information comes from a correctly colocated second tracer? |
| `real FAINT - R3-RF-DM` | Does physical FAINT information beat a point-process- and selection-matched response null? |

BGS_FAINT remains context-only and non-production during this gate. Promote it only if
real FAINT produces a reproducible independent-phase improvement over both strict nulls,
with separate tracer response/HOD contracts and posterior coverage maintained. A gain
from an independent random or cross-phase field is an optimization/regularization
diagnosis, not new cosmological information.

Select on the final production-like ph006 view, while reporting every stage, worst
stage/effect, bins of mask distance/completeness, and at least one held-out degradation
recipe. Boundary residuals and posterior coverage must be reported separately for
footprint edges, holes and well-supported interiors. A curriculum is only an
order-controlled replay ablation using the identical Arm-C examples and optimizer
updates. After any dense-to-sparse warm-up, retain balanced replay of dense,
intermediate, and final views; a one-way curriculum that forgets earlier views is not an
admissible comparison. Paired consistency is optional. Cross-stage JEPA remains P11: it
may open as a bounded parallel experiment once the paired view ladder is frozen and the
P12 same-summary headroom/identity audit passes. The dense teacher remains advisory,
and P11 cannot delay Arm A, response work, or P12.

Minimal decision order:

1. [complete 2026-08-14] implement and pass the phase-balanced loader/transform
   contract;
2. [complete 2026-08-14] register and audit the existing ph000--ph006 plus Loa
   random/response sources and write `P10_RESPONSE_SOURCES_READY.json`;
3. [complete 2026-08-20] run freshly initialized Arm A using the frozen final-view R0
   contract and select U-PATCH on ph006;
4. [active 2026-08-23] freeze the FAINT Proxy/Null diagnostic at epoch 15 and retain it
   only as the matched six-channel comparator. Run the response-representation ladder:
   R1 compressed random reference, R2 audited completeness summaries, and priority
   R3-RF high-S/N voxel-resolved random response; do not wait for FAINT or JEPA before
   continuing P12;
5. continue the P12 baseline from frozen R0 artifacts in parallel; if R1 is promoted,
   regenerate response-conditioned cross-fits under a separately frozen contract;
6. [full R2 authorized 2026-08-23] the R2 missing-response policy, all-phase overlays,
   loader smoke and 1,000-patch GPU canary pass without opening ph001. Run it concurrently
   with R3-RF under frozen contracts; the matched sequence, rather than waiting for R1
   convergence, provides the test of increasing response detail. Defer boundary-distance
   R4 until this response-compression ladder is resolved;
7. [bounded v2 canary closed 2026-09-01] the exact-M JEPA arm passed its 500-update
   technical contract but failed the registered latent-content gate. Do not chain full
   JEPA-v2 training. All three bounded attribution controls are complete; require a
   separately frozen v3 contract before testing any altered objective, and keep the
   incomplete dense teacher as advisory evidence;
8. test curriculum or paired consistency only for a remaining observation-transfer
   failure;
9. open the bounded NEXUS+ auxiliary branch only for a diagnosed multiscale-morphology
   residual.

Four-GPU execution policy for these gates:

- retain one variable-size canonical patch per optimizer step for U-PATCH/G-PATCH;
  do not introduce four-GPU DDP merely to accelerate the current comparison, because
  averaging four patch gradients changes effective batch size, row weighting,
  optimizer-update semantics and the exact resume contract;
- request one four-GPU 80-GB interactive node when four independent tasks are ready,
  and allocate one GPU per task (for example two G schedule canaries plus two
  phase-parallel classical workers);
- treat genuine multi-GPU training as a separately validated future implementation
  requiring matched one-GPU gradient/update parity and measured scaling. Resource
  occupancy alone is not permission to alter the scientific estimator.

---

## 9. JEPA gate

### P11 — Representation pretraining

**Status:** FACTORIAL PRODUCTS AND EXACT-M V2 CONTRACT READY; CFS RUNTIME/R1
MIRROR RECOVERY PASSED; JEPA-V2 500-UPDATE TECHNICAL CANARY PASSED BUT ITS
REGISTERED LATENT-CONTENT GATE FAILED; FULL V2 CONTINUATION BLOCKED; ALL THREE
MATCHED CONTROLS COMPLETE; DENSE TEACHER ADVISORY; PH001 SEALED; BOUNDED AND
NON-BLOCKING; RANDOM-RESPONSE VIEWS ARE MANDATORY
**Duration:** 2–5 GPU days for bounded controls

#### P11.0 — Frozen factorial-view contract

The next representation experiment is a factorial rather than an undifferentiated
``more data'' run. Keep three axes separate:

1. observation stage: `V_dense -> V_assign -> V_final`;
2. tracer information: BRIGHT-only versus BRIGHT+FAINT context with separate channels;
3. response realization: native nested views, registered stochastic training
   degradations, and one held-out degradation recipe.

The frozen machine-readable definition is
`configs/p11_factorial_views_v1.json`. Every view of one latent core remains in the
same outer split and the total scientific weight of that core is one, divided among
the views sampled in an optimizer step. BRIGHT-only remains the production default;
FAINT is a separately testable information axis, not an implicit response channel.
The current mocks have `C_z=1` and do not license a learned redshift-success response.

The optional P11 branch uses `ph002--ph005` for training and `ph006` for selection.
It excludes `ph000` because its canonical final catalogue descends from the legacy
`path1_fiberassign` staging chain, whereas its available targetable and assigned
catalogues descend from the official `forFA0_nomask`/`altmtl0` chain. Those products
are not pointwise TARGETID-nested, so treating them as successive degraded views would
confound observation stage with catalogue lineage. This exclusion is local to P11:
`ph000` remains valid and retained in the frozen P10 training and P12 posterior
contracts.

Progress:

- [x] Freeze the observation-stage, tracer and stochastic-response axes, including
  `tileloc_correlated_thinning` as the held-out degradation recipe.
- [x] Freeze the visible-phase source manifest without reading ph001 or large data on
  the login node. Artifact:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/FACTORIAL_VIEW_SOURCES_READY.json`.
- [x] Materialize missing `V_dense` BRIGHT/FAINT and `V_assign` BRIGHT count fields on
  the exact canonical P3 lattices in a CPU allocation. Reuse the immutable final-view
  P3/P3b-R products. The existing FAINT overlay is assignment-stage context and is
  not silently relabelled as a final Loa catalogue; under the audited mock `C_z=1`,
  the final FAINT view is an explicit identity reference to the supported assignment
  field. Builder: `workflows/abacus_tweb/p11_build_factorial_view_counts.py`.
  Completed products and per-phase manifests are below
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/` for
  `ph002--ph006`.
- [x] Install a bounded persistent CPU supervisor for the heavy count build. It waits
  for `P12A_COMPLETE.json` and an allocation slot before running, so optional P11
  preparation cannot delay P12 or exceed the two-allocation policy. Artifact:
  `workflows/abacus_tweb/run_p11_factorial_view_counts_interactive.sh`.
- [x] Validate nesting, TARGETID identity, count conservation, common random support,
  per-view response tuples, and one-total-weight ownership before writing
  `FACTORIAL_VIEW_PRODUCTS_READY.json`. The aggregate marker passes with
  `sealed_phase_opened=false` and `truth_or_targets_read=false`; view-specific response
  transformations remain the next frozen-adapter gate rather than being inferred from
  count products.
- [x] Implement the label-free V_dense response fit, lazy stage-count overlay and
  capacity-matched three-channel U-PATCH adapter. Integrate it with the resumable
  phase-balanced trainer using ph002--ph005 only, retain ph006 as application-only,
  and add a compute-node worker. Artifacts:
  workflows/abacus_tweb/p11_factorial_training.py,
  workflows/abacus_tweb/run_p11_dense_teacher_interactive.sh, and
  tests/phase4/test_p11_factorial_training.py (10 combined P11 tests pass).
- [x] Materialize dense_response_adapter_v1/P11_DENSE_RESPONSE_ADAPTER_READY.json
  on a compute node, then run one extraction/normalization parity smoke and a bounded
  GPU update canary. The adapter gates and 10 tests pass; SHA-256
  `e2c2847a8f38a2ba46c38823c9facd86ad07e7c90b3dc6330251933c97fa56ed`.
  Training passed 500 finite optimizer updates with an atomic checkpoint. Active
  worker: allocation 57782878 on nid008477; runtime log
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/p11_dense_teacher_interactive_57782878.log`.
- [x] Resolve the NERSC interactive user-balance rejection. The DESI GPU balance
  accepted allocation 57782878 on 2026-08-31. Reuse interactive allocations for
  development gates; do not substitute sbatch.
- [x] Implement the first information-headroom sub-gate on the frozen P12-A ph006
  folds-2--4 posterior. The report uses 50,000 rows, 512 untempered draws per row,
  natural-volume weights and 500 cap+superblock bootstrap resamples. It explicitly
  estimates `R2_max(S_U)` for the seven-feature P12 summary, not `R2_max(X_final)` or
  `R2_max(X_dense)`. The global Bayes-identity gaps are
  `-0.00029/-0.00603/-0.00809`; the lambda1 97.5% upper bound on additional
  same-summary headroom is `0.01764`, below the registered `0.03` materiality scale.
  Artifact: `docs/evidence/p11/P11_INFORMATION_HEADROOM.json`; implementation:
  `workflows/sbi/p11_information_headroom.py`; three focused tests pass.
- [ ] Finish the dense teacher only as advisory evidence. The recovered epoch-4 run
  advanced from cursor `13,819` to `39,043` and global step `215,551` to `240,775`;
  a later altered-source resume correctly failed the content-addressed source guard.
  Restore the exact frozen source set before another resume; do not weaken the guard
  or use the historical `0.602` threshold as a JEPA veto.

JEPA is not GraphNet-only. Apply it to whichever graph, grid, or F-tier encoders remain
competitive. It is a parallel summary-learning challenger, not part of the critical
path to Arm A or P12. It need not wait for Arm C to prove that the failure is purely a
representation bottleneck, because paired dense/degraded pretraining can still improve
how efficiently the final-view encoder uses information that survives the observation
operator. It cannot recreate information that was removed:

```text
I(environment; student representation)
    <= I(environment; final observed view)
    <= I(environment; dense observed view).
```

Any gain must therefore be interpreted as better extraction or regularization of
surviving signal, never recovery of absent tracers or a substitute for posterior
uncertainty.

Use matched controls:

1. random initialization;
2. masked reconstruction/denoising;
3. JEPA latent prediction;
4. response-only prediction;
5. dense-teacher/final-view-student alignment.

For Graph-JEPA, prevent globally computed graph metrics leaking hidden targets: remove
target nodes/edges and exclude a feature-support guard. Target location may enter the
JEPA predictor but not production GraphNet node features.

For grid JEPA, mask compact three-dimensional regions of the signal channels while
retaining the response channel. The target-validity mask is exact binary
`support_random`; apodized exposure remains a model input but is not permission to
train on `M=0`. The registered v2 mask selects a fixed fraction of exact support in
four deterministic, spatially separated compact clusters and propagates the mask to
coarser U-Net layers with pooling-aligned logical-any operations. Reuse the U-Net
encoder before considering a new transformer.

Every P11 view carries its own deployable response tuple
`{N_s, M_s, C_s, mu_s, distance_to_boundary}`. Reuse the same base angular-random IDs
for `V_dense/V_assign/V_final`, then apply the audited stage response factors so paired
differences are not random-catalogue Monte Carlo noise. Mask JEPA targets only inside
common `M=1` support; never reconstruct `M=0` as a void. Condition the student and
predictor on response, but include a response-only control and stratify alignment by
response strength so trivial footprint reconstruction cannot masquerade as a cosmic
representation gain.

The preferred first variant uses paired observed views of the same latent field with
varied magnitude selection, fibre assignment, completeness, and redshift errors.
HOD/velocity-bias variation is a later nuisance-robustness extension, not a prerequisite
for the bounded JEPA test.

The deployable final-view adapter is frozen to the R1 random-response field contract,
not the legacy P3a occupancy exposure. `V_final` uses the immutable final BRIGHT counts
plus stored P3b-R log-ratio and random exposure, while `V_dense` uses its own dense
count/selection fit on the identical P3b-R support. A real-patch parity gate must prove
identical phase/cap/core/context/authoritative galaxies and a common underlying
response artifact before writing the step-0 checkpoint. Exact `support_random` is
loaded as mask-only M metadata; `exposure_apodized_random > 1e-4` is explicitly not an
M proxy because smoothing extends beyond binary support. Intersecting discrepant masks
is forbidden because it would hide a response-contract defect. The rejected preflight
that found 4,374 P3a/P3b-R exposure disagreements and the later infeasible-cuboid
preflight both performed zero optimizer updates and remain useful negative provenance.

Operationally, P11 uses the content-addressed recovery mirror at
`/global/homes/d/dkololgi/p11_contracts/training_contract_r1_random_repair_v2_20260901`.
It regenerates only small R1 manifests and the already-frozen transform, while all
phase/core/geometry arrays and fields remain symlinks to the canonical immutable
products. This is a storage-inode recovery, not a new scientific data version. The
loader must verify the mirror-local R1 inventory pointer, every phase adapter hash,
the unchanged target-scaler and R1 field-transform hashes, and sealed ph001 before use.

#### P11.1 — Information headroom and paired dense/degraded teacher--student gate

**Status:** BOUNDED MATCHED-CONTROL V2 CANARIES COMPLETE; JEPA-V2 LATENT-CONTENT
GATE FAILED; FULL V2 CONTINUATION FORBIDDEN; NOT A PRODUCTION PROMOTION

The historical T3 LUPI attempt is not evidence against this branch: it used a
true-density CNN teacher, never completed a valid GPU run, and was shelved without a
scientific result. The first P11 test instead uses the same observed-galaxy modality at
different known response levels so it isolates observation robustness rather than
distilling a teacher that directly sees the answer field.

The supervised dense teacher is an empirical privileged-view control, not an
information-theoretic upper bound and not a hard veto. Open the bounded
`PAIRED-DEGRADE-JEPA-v2` canary only if:

1. the frozen P12 posterior passes the Bayes-risk identity check closely enough to
   diagnose headroom inside the current deployable summary, while the report states
   explicitly that it does not estimate the raw final-view ceiling;
2. `V_dense`, `V_assign`, and `V_final` are correctly paired views of the same latent
   core under a frozen observation/response contract; and
3. all views of a latent core remain in one outer
   split with one total scientific weight.

The first sub-gate now passes. On the primary macro-shell statistic the posterior-
variance estimate is `R2_max(S_U)=0.57896` for lambda1, while the posterior mean and
frozen base achieve `0.58082` and `0.58188` on the same sampled rows. The small sign
reversal is a finite-posterior/calibration residual, not evidence of beating a Bayes
limit: the global lambda1 identity gap is `-0.00029` with a spatial-block 95% interval
`[-0.01764, +0.01764]`. There is therefore no material downstream-estimator headroom
inside `S_U` at the `0.03` scale. A material improvement must change what information
the summary retains, which is precisely the bounded question assigned to JEPA.

This does **not** establish `R2_max(X_final)` or prove a representation gap. The later
information ladder remains:

1. cross-fitted higher-capacity `q(lambda|X_final)` with held-out proper score and
   conditional variance;
2. the matched `q(lambda|X_dense)` control;
3. the held-out proper-score increment from adding the privileged view, treated as an
   operational rather than exact conditional-information estimate; and
4. the same headroom audit for `Z_JEPA` if the bounded canary trains.

Those estimates strengthen the diagnosis but need not delay one preregistered,
compute-bounded matched-control JEPA canary. The canary itself is now the direct test
of whether a different representation preserves useful final-view information that
the current supervised summary discards.

The final-view deficit may contain both representation loss and irreducible information
loss; that ambiguity does not block the bounded experiment. It does constrain the
claim: failure is unsurprising, and success must be demonstrated on the deployable
student alone rather than inferred from teacher quality.

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
JEPA supplies an encoder/summary, not a posterior, and alignment must never be presented
as calibrated uncertainty. Begin P12 after ph006 deterministic selection even if this
P11 experiment is still running.

Latent coordinates are neither ordered nor coordinate-wise bounded between views.
The registered bound is informational:

```text
I(environment; Z_student)
    <= I(environment; V_final)
    <= I(environment; V_dense).
```

Use fixed ph006 probe cores and one frozen teacher-space PCA basis to display steps
0/250/500. Scientific alignment evidence requires native/predictor CKA above both
ordinary and response-matched shuffles, cross-fitted Procrustes/CCA, paired retrieval,
non-collapsed variance/effective rank, response-quartile diagnostics, and spatially
disjoint target probes. The response-only encoder is a decisive shortcut control;
missing it caps the audit at advisory. Alignment that rises while rank or held-out
target-probe performance falls is a failure, not a visually pleasing result.

Because several density fields can be compatible with one final observed view, a
deterministic predictor may learn only a conditional-average teacher component. It
cannot identify a unique missing field and may encourage hallucination if alignment is
too strong. Therefore every promoted P11 encoder must be refit under P12 and pass SBC,
TARP, response-conditional, shell-conditional and local-information coverage. No
latent-alignment statistic may be substituted for posterior calibration.

The frozen v2 canary now demonstrates why this compound gate is necessary. By update
500 its native student/dense CKA reaches `0.87254` and retrieval MRR reaches `0.31349`,
but the held-out student target probe is `-0.08775`, versus `+0.01145` for the matched
plain-supervised control. Its effective-rank fraction is only `0.03328`; the supervised
control reaches a nearly identical `0.03342`, so low pooled rank is not uniquely a
JEPA pathology, but JEPA-v2 still preserves less target information. The registered
latent-content gate therefore blocks full v2 continuation. High cross-view alignment
is not accepted as evidence of a useful or uncertainty-aware representation.

Do not pretrain on DESI until truth-known sim-to-sim controls pass. DESI pretraining is
transductive domain adaptation, not zero-shot generalisation.

Adopt JEPA only for consistent fresh-graph or blind-phase deterministic improvement,
targeting at least +0.03 spatial-fold macro R²(λ1) or a comparably clear balanced-class
gain.

Progress checklist:

- [x] Freeze the paired view ladder and pass the P12 same-summary information-headroom
  sub-gate without opening ph001. License one bounded canary; do not equate this with
  scientific promotion.
- [ ] Finish the dense-teacher fit as advisory evidence; do not use its historical
  `+0.03` threshold as a mathematical JEPA veto.
- [x] Freeze the capacity-matched `supervised_masked`, `masked_reconstruction`,
  `response_only`, and `jepa` arm contract. All arms start from registered random
  initialization and use the same phase-balanced examples, masks, target weights,
  optimizer-update budget and three-channel U-PATCH capacity.
- [x] Reject the v1 interior-only mask before step 0. Four exact-M `8^3` cuboids are
  constructible in only 69.6--70.8% of the 500-per-phase audit, and the former
  `exposure_apodized_random > 1e-4` eligibility rule includes M=0 because apodization
  spills outside binary support. Do not repair this by dropping boundary cores,
  selecting a convenient parity core, or weakening M.
- [x] Specify the v2 leakage-safe mask as four compact, spatially separated clusters
  covering 25% of exact P3b-R `support_random`, with supported context left visible.
  M=0 is never an auxiliary target; all matched arms share the seeded mask schedule
  and retain every core's supervised eigenvalue loss. Propagate target masks through
  the U-Net pools by logical-any rather than nearest resampling. The production-identical
  audit finds 10 of 67,244 training cores (`1.487e-4`; 3/2/3/2 by phase and zero in
  ph006) auxiliary-invalid under the exact minima. They are explicit `aux_valid=false`
  examples with zero auxiliary weight, not omitted examples; the population rate must
  remain below the frozen `0.001` ceiling. The 500-update canary uses the separately
  registered finite-sample rule of at most one invalid update and fails at two.
- [x] Keep all paired views in one outer split with one latent-core scientific weight,
  and retain `ph002--ph005` for fit, `ph006` for selection and `ph001` sealed.
- [x] Freeze the forward-observation ladder, held-out response recipe, aligned layers,
  stop-gradient EMA teacher, predictor, spread/covariance regularization and loss
  weights. Preserve `configs/p11_paired_degrade_jepa_v1.json` as the rejected
  zero-update contract; use `configs/p11_paired_degrade_jepa_v2.json` for new work.
- [x] Implement exact checkpoint/resume and logging, a content-addressed frozen-data
  contract, atomic step-0 checkpointing, finite-gradient and checkpoint-reload gates,
  and fixed ph006 latent exports at steps 0/250/500. Controls export student encodings
  of both views and an explicit response-only latent; only the JEPA arm exports a
  trained predictor output.
- [x] Implement the fixed-probe latent audit: frozen teacher-space projection,
  CKA/shuffle controls, retrieval, spread/effective rank, response stratification and
  spatially disjoint target probes. A missing response-only control limits the result
  to advisory evidence.
- [x] Reject the first real-patch preflight before step 0 after it exposed 4,374
  support disagreements between legacy P3a `V_final` and P3b-R `V_dense`; replace the
  final-view adapter with the frozen R1/P3b-R contract rather than weakening the gate.
  The corrected v2 compute preflight passes all 37 registered implementation,
  loader, latent-diagnostic, supervisor and view-contract tests.
- [x] Recover the inaccessible ph002 R1 adapter-manifest inode with a 98-kB
  content-addressed home mirror; retain all large immutable products by symlink and
  archive exact pointers/hashes in
  `docs/evidence/p11/P11_R1_CONTRACT_MIRROR_RECOVERY.json`.
- [x] Complete and archive the production-identical, label-free, all-visible-core
  exact-M mask audit, including support/target/bottleneck quantiles and failures by
  phase. Across 84,040 ph002--ph006 cores, pooled mask propagation leaves 10 training
  cores (`10/67,244 = 1.487e-4`) auxiliary-invalid and zero in ph006, versus
  12.12--12.61% invalid under rejected nearest resizing. Evidence:
  `docs/evidence/p11/P11_JEPA_MASK_FEASIBILITY.json`; ph001 remains sealed.
- [x] Pass corrected R1/P3b-R exact-M real-view parity on ph002--ph006 and freeze
  aggregate data-contract SHA-256
  `004ef485b3773ded1639720aad2e2d634155000367f98ee663f8d0468e676f57`.
- [x] Complete all four 500-update technical canaries (`jepa`,
  `supervised_masked`, `masked_reconstruction`, and `response_only`): every arm has
  500 auxiliary-valid and zero auxiliary-invalid updates, finite state, a reloadable
  checkpoint, target fraction `0.249998`, and valid step-0/250/500 exports. ph001
  remained sealed.
- [x] Apply the registered JEPA latent-content gate and fail closed. Despite strong
  paired alignment, effective-rank fraction fell `0.22116 -> 0.03328` and the held-out
  student probe macro R2 fell `0.02201 -> -0.08775`; the shared-predictable-subspace
  gate is false. Do not launch full JEPA-v2.
- [x] Compare the bounded matched controls. Plain supervised training also compresses
  the pooled latent to rank fraction `0.03342`, showing that this symptom is not
  uniquely JEPA-induced, but retains a better target probe (`+0.01145`). Masked
  reconstruction is worse (`-0.65792`), while response-only input is non-predictive
  (`-2.40036`) and cannot explain the JEPA alignment as a response shortcut.
  Evidence: `docs/evidence/p11/P11_JEPA_V2_MATCHED_CANARY_RESULTS.json`.
- [ ] If P11 continues, freeze a distinct JEPA-v3 objective before inspecting new
  ph006 results and rerun the identical technical/content gate. Do not tune or resume
  v2 post hoc.
- [ ] Finish and report the dense teacher as advisory privileged-view evidence.
- [ ] Adopt only after a newly gated encoder gives reproducible deployable-view
  transfer gain and passes a fresh P12 posterior fit and calibration audit.

---

## 10. Posterior inference and VAC production

### P12 — Posterior calibration

**Status:** START AFTER PH006 DETERMINISTIC SELECTION; P11 MAY RUN IN PARALLEL

P12 is the binding gate for the intended posterior/class-probability VAC. Begin it from
scratch immediately after deterministic model selection on ph006 rather than waiting
for optional P11 or a long deterministic research tail. Deterministic protocol
selection and a deterministic canary do not require FMPE/NPE, but the production VAC
claim does.

1. Generate spatially out-of-fold embeddings or base predictions.
2. Fit FMPE/NPE on training phases.
3. Include ntilde(z) and response covariates directly in posterior conditioning.
4. Tune on ph006.
5. Evaluate once on ph001.

The baseline posterior contract is

~~~text
q(lambda_g | encoder(x_s), response_s, H_fid)
~~~

where ordered tidal eigenvalues are the current coordinates of environmental
information and `H_fid` is the fixed BGS galaxy--halo prescription used by the training
mocks. Do not describe this as HOD-marginalized. After in-domain P12 calibration passes,
run one held-out-HOD intervention if the published 11-model BGS cubic suite can be
located and joined to truth without changing the estimand. Compare the HOD-induced
shift in posterior location and coverage with the calibrated posterior width. If it is
small, release a conditional-robustness statement; if it is comparable to the width or
causes coverage failure, open a separate HOD-augmentation/marginalization project. This
stress test is not a prerequisite for the first conditional VAC.

Require SBC, TARP, coverage, conditional coverage, knot-probability reliability, Brier
skill, width-versus-error, posterior contraction, and prior-dominated flags. Condition
or stratify the diagnostics by phase, redshift, sampling density, `M`, `C_fibre`, `C_z`,
mask distance, hole versus footprint edge and held-out response recipe. Scalar
tempering that repairs average coverage while leaving shape failure is insufficient.

#### P12-A — Coordinate-aligned baseline posterior

Complete this before using independently trained raw fold latents. Fit

~~~text
q_A(lambda_g | lambda_hat_OOF,g, redshift, ntilde(z), cap,
                    random_support_boundary_distance, H_fid)
~~~

with the three physical OOF U-PATCH base predictions as the coordinate-aligned summary
and ordered softplus coordinates as the posterior target. Artificial fold-boundary
distance, fold ID, superblock ID and phase ID are not conditioning features. Fold and
superblock are retained only to keep ph006 width calibration (folds 0--1) spatially
disjoint from its selection report (folds 2--4). The first result must report proper
log score, SBC, TARP, marginal and shell-conditional coverage, width-versus-error,
knot reliability/Brier, posterior contraction and posterior-mean accuracy. A
technical completion marker is distinct from a calibration-pass marker.

Implementation status:

- [x] Implement and unit-test the phase/shell sampling, ordered target transform,
  deployable response feature contract, FMPE fit, disjoint ph006 calibration/evaluation
  and posterior diagnostics in
  `workflows/sbi/p12_prepare_base_response_dataset.py` and
  `workflows/sbi/p12_train_base_response_fmpe.py`.
- [x] Complete all five leave-one-phase-out encoders and export the six guarded OOF
  summaries (`ph000`, `ph002--ph006`). Every `OOF_SUMMARY_COMPLETE.json` passes its
  parent/hash, omitted-phase and sealed-ph001 gates.
- [x] Correct the P12 response covariate after the full builder exposed that P4
  `field_support_distance_mpc` is an intentionally unfilled NaN placeholder. P12 now
  samples `distance_to_support_boundary` and `support_random` from the canonical P3b-R
  overlays, excludes `M=0` rows before shell stratification, and caches the parent-level
  response under `p12a_random_support_parent_cache_v2/`. The six-phase 25k/5k canary is
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_contract_canary_v2/`.
- [x] Materialize the full two-million-row training and 600k-row ph006 selection
  dataset with `P12A_DATASET_READY.json` after all OOF, parent, random-support and
  finiteness gates pass:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/`.
- [x] Fit/evaluate the full FMPE on ph006 and write the technical completion marker:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/fmpe_seed42/P12A_COMPLETE.json`.
  The two-million-row fit converged after 143 epochs; evaluation uses 20k disjoint
  calibration rows and 50k selection rows, and ph001 remained sealed.
- [ ] Write `P12A_CALIBRATION_PASS.json` only after a frozen production acceptance
  contract is satisfied. The corrected audit shows that global physical-eigenvalue
  calibration is practically close to nominal, but the sparsest shell retains a
  lambda2/lambda3 conditional residual. Neither scalar temperature nor the tested
  affine map is an acceptable repair. Keep the pass marker absent rather than
  weakening the gate after seeing ph006.
  Correction programme:
  - [x] Preregister and unit-test randomized finite-rank diagnostics in both
    ordered-softplus and physical-eigenvalue coordinates, with matched SBC/TARP
    budgets and conditional response/shell audits:
    `workflows/sbi/p12_calibration_diagnostics.py`.
  - [x] Run the frozen 512-draw audit on the existing checkpoint; cache samples,
    row indices, plots, and a provenance-complete JSON report. The 50k-row report is
    `fmpe_seed42/calibration_audit_v1/P12A_CALIBRATION_AUDIT.json`; ph001 stayed
    sealed. It adds randomized ranks, physical-eigenvalue ranks, fold strata and a
    254-superblock spatial bootstrap.
  - [x] Render and archive the matched-row joint TARP curve from those cached draws.
    On 50,000 ph006 folds-2--4 galaxies with 512 draws each,
    `max|ECP-alpha|=0.00770`, passing the registered `0.05` gate. The plot uses the
    standardized ordered-softplus coordinates used by the frozen audit and labels the
    spatial-resampling envelope as a maximum-deviation reference, not a pointwise
    confidence interval. Evidence: `docs/evidence/p12/P12A_TARP_CURVE.json` and
    `docs/figures/p12_calibration_audit_20260830/p12a_tarp_curve.png`.
  - [x] Classify the defect before choosing a correction. Global physical rank
    distances are only `0.0075/0.0156/0.0078`, global coverage is near nominal and
    spatial TARP passes. The bounded residual is a sparse-shell lambda2/lambda3
    high-location/undercoverage tendency, while other shells are nominal or mildly
    overcovered.
  - [x] Run the bounded per-shell/per-coordinate location-scale canary in
    ordered-softplus space with
    `workflows/sbi/p12_affine_calibration_canary.py`. Fit ph006 fold 0 and test 1,
    reverse the roles, then fit folds 0+1 only if parameters are stable. Promotion
    additionally requires spatial proper-log-score improvement, preserved R2/TARP,
    no material shell degradation and improved sparse-shell lambda2/lambda3 coverage.
    The correction was rejected: both crossfit scores worsened, parameter stability
    failed and the folds-2--4 physical log-score delta was `-0.002653` with spatial
    95% interval `[-0.005009,-0.000334]`.
  - [x] Freeze the candidate map and evaluate exactly once on folds 2--4 using physical
    SBC/TARP, coverage, proper score, tail/class reliability and posterior-mean
    preservation. It improved sparse lambda2/lambda3 coverage but worsened other
    diagnostics; no corrected posterior was promoted. Evidence:
    `docs/evidence/p12/P12A_AFFINE_CALIBRATION_CANARY.json`.
  - [x] Retain uncorrected P12-A as the baseline and encode the sparse-shell limitation
    in the release/quality contract. Do not open a flexible conditional map merely to
    flatten ph006 ranks. A conditional recalibrator or richer P12-B summary is a new
    challenger and must improve crossfit and spatial proper score on data not used to
    fit it before it can replace P12-A.
  - [x] Verify that posterior width responds to available information and identifies
    difficult cases on the frozen 50k-row ph006 folds-2--4 sample. Median central-68%
    half-width grows by factors `2.21/2.33/2.25` from the densest to sparsest shell;
    widest-width-quartile RMSE is `3.45/3.33/3.38` times the narrowest. After control
    within shell and predicted-trace decile, local BRIGHT neighbour count still
    anticorrelates with log width (`-0.131/-0.151/-0.134`), while the explicit
    random-support boundary covariate has the stronger relation
    (`-0.481/-0.523/-0.551`). Sparse-shell coverage is
    `0.689/0.658/0.663`; the lambda2/lambda3 shortfall is a mild lower-tail/high-location
    residual, not failure to widen. Evidence:
    `docs/evidence/p12/P12A_WIDTH_INFORMATION_DIAGNOSTIC.json` and
    `docs/figures/p12_width_information_20260830/`.
- [x] Install the persistent `p12a_posterior` supervisor. It waits for all OOF exports
  and a free allocation slot, enforces the two-allocation limit, then performs a GPU
  canary and the full dataset/fit/evaluation chain using
  `workflows/sbi/run_p12a_posterior_interactive.sh`.

Execution order after the current strict controls is frozen:

1. [complete] complete P12-A without waiting for P11;
2. [complete] finish the factorial view products and P12 same-summary information-
   headroom gate; continue the dense teacher only as advisory evidence;
3. freeze and run matched supervised, masked-reconstruction, JEPA and curriculum
   controls under one bounded canary contract;
4. condition the same P12 head on the winning summary and compare proper posterior
   scores;
5. run end-to-end JEPA-initialized FMPE only after the frozen-summary challenger wins;
6. stress-test one held-out HOD only after in-domain calibration, and open ph001 once
   all choices are frozen.

Progress checklist:

- [x] Reopen after the five-phase U-PATCH representation passed ph006 deterministic
  selection at epoch 20. This opens P12 preparation; it does not unseal ph001.
- [x] Generate leakage-safe out-of-fold conditioning summaries.
  - [x] Implement the leave-one-phase-out contract builder and guarded latent exporter.
    The exporter refuses ph001 and refuses any phase listed in the source checkpoint's
    training phases; its validation phase must be exactly the exported phase.
  - [x] Materialize and validate five immutable contracts omitting, in turn,
    ph000/ph002/ph003/ph004/ph005. Each contract recomputes train-only selection and
    target/count transformations from the remaining four phases. The five contracts
    contain `67,244--67,678` complete training cores and `16,768--17,202` unique
    omitted-phase validation cores, and every omitted phase is absent from its training
    epoch. Evidence: `docs/evidence/p10/p12_crossfit_contracts_ready_20260820.json`.
  - [x] Train five fresh omitted-phase encoders and export the exact 32-dimensional
    latent, base prediction, truth and deployable response covariates for each omitted
    phase. Export ph006 only from the frozen all-five-phase epoch-20 checkpoint. The
    persistent four-GPU chain launched `omit_ph000` and `omit_ph002` after passed
    two-update canaries; both had completed eleven epochs and were active in epoch 12
    at the 2026-08-21 audit. The queue follows with ph003/ph004/ph005 and ph006 export.
    - [x] Complete the `omit_ph000` encoder at epoch 20: omitted-phase macro
      `0.56469`, first-three `0.62942`, shells
      `0.68091/0.63844/0.56891/0.37049`.
    - [x] Complete the `omit_ph002` encoder at epoch 20: omitted-phase macro
      `0.56844`, first-three `0.62796`, shells
      `0.68732/0.62367/0.57288/0.38987`.
    - [x] Fix and unit-test the P12 assignment-row lookup after the first full exports
      exposed that `len(NpzFile)` counts archive fields rather than galaxy rows.
      Commit `ad0925b` indexes by the length of `parent_node_id` and validates bounds
      and uniqueness; the bug did not affect encoder training or saved predictions.
    - [x] Run the corrected ph000/ph002 exports and stamp their
      `OOF_SUMMARY_COMPLETE.json` markers. The production-scale shards contain
      `5,026,863` and `4,929,962` parent-keyed rows respectively and pass their hash,
      parent-set and no-ph001 guards.
    - [x] Train and export the `omit_ph003`, `omit_ph004`, and `omit_ph005`
      encoders; their complete parent-keyed OOF markers now pass beside ph000/ph002.
    - [x] Export ph006 summaries from the frozen all-five-phase epoch-20 U-PATCH
      checkpoint and concatenate only the five leakage-safe training shards after all
      parent sets, hashes and response schemas pass.
      - [x] Export and validate the ph006 tuning shard: `4,908,831` parent-keyed rows,
        with latent/base/truth/response hashes and `sealed_phase_opened=false`.
      - [x] Concatenate after all remaining omitted-phase shards pass, producing
        the frozen `P12A_DATASET_READY.json` v2 response-conditioned dataset.
- [ ] Fit on training phases and tune on ph006 only.
- [ ] Pass marginal, multivariate, conditional, tail, and information gates.
- [ ] Record the `H_fid` conditional estimand and run the optional held-out-HOD stress
  test only after baseline calibration.
- [ ] Evaluate once on ph001 and freeze calibrated posterior artifacts.

Out-of-fold summary contract:

1. The all-five-phase epoch-20 U-PATCH checkpoint supplies ph006 selection summaries
   and the later frozen ph001 inference summary only. Its embeddings on its own five
   training phases are in-sample and must not train the posterior head.
2. Train five phase-cross-fitted U-PATCH encoders from scratch, each omitting one of
   ph000/ph002/ph003/ph004/ph005. Serialize the exact 32-dimensional per-galaxy latent
   consumed by the deterministic point head, the deterministic base prediction, and
   deployable response covariates only for the omitted phase.
3. Treat the three-dimensional base prediction plus deployable response covariates as
   the coordinate-aligned P12 baseline. Independently trained fold encoders do not
   guarantee a common basis for their raw 32-dimensional latents, even with the same
   seed. Before concatenating raw latents, use a truth-free common ph006 anchor to test
   cross-encoder alignment and fold identifiability; admit aligned latents only if they
   improve held-out ph006 likelihood/calibration over the base-prediction baseline
   without making the omitted fold recoverable.
4. Concatenate the five omitted-phase artifacts to form the P12 training population.
   Fit FMPE/NPE on those summaries, tune architecture/calibration only on ph006, and
   never condition on phase ID.
5. Freeze parent-row identity, checkpoint/source hashes, response schema and the
   no-ph001-access marker for every shard. A cheaper in-sample latent extraction is a
   technical diagnostic only and cannot support posterior coverage claims.

### P13 — DESI canary and scale-out

**Status:** GATED ON P10 AND A FROZEN DETERMINISTIC WINNER; P12 ONLY FOR POSTERIOR COLUMNS

Reproduce the winning representation exactly:

- GraphNet: canonical DESI graph/global metrics, then graph patch views.
- U-Net: canonical count/response fields, then field patches.
- F-tier: canonical graph/fields plus converged FFT overlap and trim.
- Hybrid: all components with out-of-fold-compatible fusion semantics.

Before the canary, retain the frozen superseding Loa DR2 family and freeze a source
manifest for its full randoms, clustering randoms, completeness/PIP, imaging/HP-map and redshift-quality
products. The mock-to-DESI schema crosswalk must reproduce `M`, `mu`, completeness,
mask distance and any quality/error channels without substituting a different catalogue
release. The P10 source contract is `DA2/loa-v1/LSScats/v2.1/PIP`; P13 may freeze a
newer exact Loa version after an explicit compatibility audit, but Kibo is superseded
and is not an admissible deployment fallback. The available SecondGen mocks remain
Kibo-derived, so the Kibo-to-Loa response crosswalk must remain visible in provenance.

Run a golden mock and one DESI canary before full scale-out. Every eligible galaxy is
authoritative core exactly once. Overlapping contexts are not independent evidence.
Use the pre-registered deterministic de-duplication rule; if P12 is added, never
multiply overlapping posteriors.

Required flags include redshift support, graph/field support, boundary, mask hole,
extreme edge, completeness, OOD, and overlap disagreement. Add prior-domination and
posterior-information flags only when P12 posterior columns exist.

Progress checklist:

- [ ] Reproduce frozen representation and schemas in GraphWeb_DESI.
- [ ] Freeze and hash the exact DESI data/random/response release and pass the
  mock-to-DESI response-schema crosswalk.
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
| P11 JEPA | after frozen view ladder + transferring teacher; parallel | 2–5 GPU d | optional JEPA decision |
| P12 posterior | immediately after ph006 deterministic selection | 2–5 GPU d | production calibration report |
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
