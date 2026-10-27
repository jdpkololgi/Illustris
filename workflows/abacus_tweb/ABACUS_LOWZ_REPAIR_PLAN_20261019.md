# Abacus low-z lookup repair: scope and next qualification

## Current status (supersedes the earlier file-0-only plan below)

The five-phase galaxy-position validation and versioned pipeline implementation
are complete within the scope documented in
`../../docs/galaxy_position_migration_20261019.md` and the latest `SCIENCE_LOG.md`.
The old host-index namespace bug is confirmed; own-galaxy coordinate sampling
avoids that lookup. Job 60003795 was explicitly approved and has been released.
The earlier approval/scope statements below describe the historical recovery run.

Unresolved satellites have usable RA/DEC/Z_COSMO coordinates. The outstanding
issue is absence of a separate original satellite XYZ reference, not inability
to locate them in the field. Separating coordinate availability from independent
validation is the recommended policy. The implemented v2 policy now labels them and records exact per-row independent
checks in INDEP_POS. See the migration report and product-lifecycle register for
the census and staged replacement status; production rebuilds remain pending.


## User-selected target: field at each galaxy position (2026-10-19)

The user explicitly selected the tidal field at each mock galaxy's own real-space
simulation position, including satellite offsets. Host-centre targets are no longer
the intended migration endpoint. Preserve host-link recovery as a validation tool;
it is not required to evaluate a field at a qualified galaxy coordinate. This changes
satellite targets above z=0.15 as well as repairing the low-z lookup. Independently
validate the inverse cut-sky transform for other slabs/phases, periodic boundaries,
and the unresolved population before a bulk migration; preserve Z_COSMO and distinguish
it from observed/redshift-space Z. No new production rewrite or compute budget follows
from this target choice.

The approved illustrative run 60002803 completed (4-second step, exit 0:0) and was
released. Four figures and sampled values are in
`outputs/lowz_mapping_explainer_20261019_job60002803/`. For resolved centrals in this
file-0 sample, trace agreement between reconstructed galaxy positions and direct raw
halo lookup is r=-0.150183 below z=0.15 (135 rows), versus r=1.0 above (217 rows).
The diagnostic uses x_L2com for the host reference, not the historical x_com field;
it isolates the index defect and does not reproduce every historical stored label.
This is label agreement, not a clustering or restored-learning-performance test.

## Central/satellite interpretation and property test

User asks whether shared host labels weaken property-environment relationships.
Shared labels erase within-host target variation, not retained CEN identity.
Galaxy-position labels restore spatial sampling but are not a central/satellite
classifier: R=7 Mpc/h smoothing can leave little within-halo variation. The recovered
39 satellite offsets have median 0.257755 Mpc/h and maximum 1.689873 Mpc/h; this is
one file sample, not a cluster-population result. Preserve CEN, stable host identity
where validated, host mass and periodic host-centric distance separately. Test the
property relationship on a fixed galaxy sample with old/new target definitions,
stratified by central/satellite status and redshift; do not claim the weak relationship
is explained until this comparison is measured. This test has not been run.

## Historical file-0 qualification result

Recovery allocation 60000317 completed once and was released. **67/67 heldout resolved centrals recover the exact raw halo row**, with no shared hosts between training and validation. All 135 centrals and 39 satellites match host mass. The inferred radius interval and evidence are in `ABACUS_LOWZ_ANNOTATION_AUDIT_20261019.md` and `lowz_recovery_validation_20261019.json`. The mapping passes this file-0 sample test, but remains unqualified for production or unresolved hosts.

The next scientific gate is independent-file/phase and boundary validation of the same selection convention, plus separate recovery of particle-host identity/positions or an explicitly versioned galaxy-position estimand. Do not refit the radius independently for each validation file and then describe those files as held out. No additional allocation or bulk migration is authorized by the completed one-run recovery approval.

## Established starting point

See `ABACUS_LOWZ_ANNOTATION_AUDIT_20261019.md` for the completed, approved one-slab audit. Direct raw/full-table HALO_INDEX is wrong for all 209 sampled low-z rows; the same lookup matches all 251 high-z halo IDs. Corrected geometry alone cannot make an unrelated halo row the right host. A cleaned mass/radius-filtered namespace recovers 62/135 resolved centrals exactly in mass and position, but is not yet a complete crosswalk. Particle-based rows require separate treatment.

Do not bulk-rewrite production catalogues using the partial candidate, switch everyone to x_L2com, or merely remove BOX_INDEX=-1 masks. The current masks contain the problem; extending the supported range requires a qualified replacement mapping and explicit target definitions.

## Recover the mapping before extending the pipelines

`recover_lowz_filtered_index.py` executed once under the separate recovery approval. It reads the already saved 460-row sample and the cleaned file-0 halo catalogue. For each resolved low-z central it independently searches for a halo within 0.01 Mpc/h of the reconstructed galaxy position and requires a matching cleaned mass at relative tolerance 1e-5. Multiple matches remain ambiguous; it does not force a nearest match.

The uniquely matched centrals are split deterministically by alternating original-row order. Only the training half constrains a strict selection-radius interval within a predeclared +/-0.05 Mpc/h bracket around the c000 z=0.16 distance. Membership changes at halo radii are enumerated exactly. The first feasible interval is selected without consulting the heldout half, and its midpoint is checked against heldout **raw halo-row identities**, all sampled central positions/masses, and sampled satellite masses. No mass-cut tuning, reordering, automatic widening or unresolved-host inference is permitted. No common interval is a useful rejection, not a reason to silently broaden the search.

This specifically tests whether tiny differences in the original cosmology/float arithmetic cause off-by-one-or-more offsets in the filtered table. A radius fitting one slab is not original production provenance. Even a successful heldout test must be followed by independent slabs/phases, boundary rows and direct grid-sampling validation before release.

Four unit tests check inferred intervals against explicit selections, tied boundary radii, incompatible constraints and empty training data. Syntax and no-allocation guards were checked separately before the approved run. Both one-run allocations have now been released; future large reads/tree construction require another user-approved allocation.

If this hypothesis fails, retain its negative result and obtain the creator's exact v0.1 exporter, halo-reader/cleaning version, selection radius/cosmology and pre-filter-to-raw crosswalk. Source history inspected so far is not that receipt. A small geometry-derived central crosswalk remains useful independent evidence, not permission to extrapolate to every satellite or unresolved host.

## Keep galaxy and host identity distinct

Use `(catalogue version/hash, phase, original source row)` as the durable **galaxy** identity, with a verified TARGETID-to-source-row bridge through fibre assignment. Preserve periodic replicas as distinct survey rows even where their periodic truth coordinates coincide. A validated host identity is separate from galaxy identity.

Retain at least `RES`, `CEN`, `Z_COSMO`, observed `Z`, FILE_NUM, original HALO_INDEX, BOX_INDEX and the original row alongside the derived mapping. A resolved raw host ID/row must be explicitly identified as such. Do not mix low/high branches or resolved/unresolved namespaces. Adding RES to an existing triple is still insufficient as a galaxy key: multiple low-z galaxies may share a host. Conflicting joins must raise an error rather than keep the first duplicate.

For a **host-centre target**, use the recovered raw host location for resolved haloes and generator host/particle positions for unresolved hosts. For a **galaxy-position target**, reconstruct RA/DEC/Z_COSMO with a pinned, validated transform and sample the periodic field there. This latter route can cover both populations without HALO_INDEX, but changes the target for satellites and requires reannotation/calibration as a separately versioned estimand. Never substitute observed Z for Z_COSMO in truth coordinates; observed geometry remains appropriate for observational inputs.

Use x_L2com to identify the upstream resolved host, but preserve the declared target-centre convention when resampling labels: a crosswalk to the correct raw halo row can still sample x_com if reproducing the existing host-x_com estimand. Changing centre field is a separate, explicitly versioned decision.

Publish explicit mapping status and source: validated raw host, validated filtered-host crosswalk, validated galaxy position, unresolved mapping, ambiguous match, or invalid metadata. Finite eigenvalues alone must not indicate valid truth. Record snapshot, cosmology/units, observer, coordinate convention, target location, grid origin/axes, smoothing and field checksum.

## Source-level repair inventory

The bounded repository scan found index references across annotation, catalogue, cache and diagnostic code; references alone are not proof that an artifact is affected. The important reviewed entrypoints are:

| Stage / files under workflows/abacus_tweb | Current behavior | Required repair / verification |
|---|---|---|
| `annotate_cutsky_with_tweb_eigs.py` | Applies FILE_NUM/HALO_INDEX to every row; bounds are the validity test | Shared validated mapping/status contract; reject unsupported low-z mappings; preserve per-row provenance |
| `export_cutsky_boxframe_points.py` | Direct host lookup without low-z branch validation | Consume the same mapping; distinguish host-position graph input from observed galaxy geometry |
| `export_cutsky_unique_host_halo_points.py` | Treats BOX_INDEX as halo row and host key | Retire or rewrite around real host IDs; BOX_INDEX is a cubic galaxy index, so this defect is not limited to low-z |
| `p10_build_bright_parent.py` | Preserves Z_COSMO but drops RES/CEN and original-row identity from compact schema | Preserve required truth metadata and a source-row bridge in a new schema/version |
| `secondgen_mocks/ph000/scripts/build_mock_bgs_maglim_catalog.py`, `write_fiberassign_mock_science_fits.py` | Default column subsets omit RES and Z_COSMO | Carry metadata or bind a lossless sidecar; do not infer lost fields from unrelated keys |
| `join_cutsky_eigs_to_fiberassign_catalog.py` | Masks BOX_INDEX=-1; builds a triple lookup and silently skips duplicate triples | Join by verified galaxy identity; validate duplicates and status; retain current containment until replacement qualifies |
| `write_stage3_postcollision_science_fits.py` | Excludes BOX_INDEX=-1, then deduplicates triples; omits RES | If low-z is admitted, deduplicate by galaxy identity, not shared host triple; preserve metadata |
| `build_abacus_graph.py`, `build_abacus_sbi_cache.py`, wedge/pooled/tiled cache builders | Filtering and row alignment depend on BOX_INDEX and catalog masks | Replace invalidity proxy with explicit validated status; fingerprint target mapping; ensure scalers/splits never consumed invalid targets |
| `p1_build_canonical_catalogue.py`, `p10_build_phase_index.py`, `p12a_coordinate_sample_audit.py` | Existing active-target guards exclude BOX_INDEX<0 and/or z<0.15; low-z may remain geometry-only context | Keep gates; prove any declared unaffected target sample satisfies them before avoiding a rebuild |
| `p10_build_observed_truth.py` | Uses sequential/index-aligned TARGETID parent binding and checks linkage | Preserve and extend this stronger identity contract; version receipts/schema if parent fields or labels change |
| `diagnose_shell0_boxindex.py`, older audit prose | Historical tests interpreted low predictability too strongly | Correct scientific interpretation; a low predictive score does not establish physical independence |

Current containment matters: the triple join and postcollision writer presently mask BOX_INDEX=-1, so their potential low-z duplicate-key failures must not be described as observed corruption in their current outputs. They become hazards if the mask is removed without a new identity contract.

## Regeneration order and scope

1. Qualify one shared mapping/identity implementation on a versioned representative fixture, including near-z=0.15 observed/cosmological mixing, centrals, satellites, particles, repeats and missing metadata.
2. Regenerate truth annotations into new paths with complete receipts; retain originals for comparison. Audit field values at reconstructed/crosswalk positions against existing grids before relying on learned correlations.
3. Rebuild joins, label-bearing truth NPZs and caches that consumed affected rows. Refit scalers/splits and retrain/recalibrate models only where provenance establishes that their training or validation inputs changed.
4. Rebuild graph topology/features only if galaxy membership or graph coordinates change. A label-only repair does not itself require recomputing unchanged observed-geometry graphs.
5. Do not regenerate matter density/T-Web grids merely because galaxy lookup was wrong. Those grids are upstream of this defect; separately validate grid provenance and coordinate conventions.
6. Check downstream GraphWeb_DESI inference against the affected checkpoint/selection lineage. An invalid low-z annotation file does not by itself invalidate every high-z trained model or observed catalogue. No sibling-repository code has been changed by this audit.
7. Retain the current z/validity science gates until coverage and calibration are demonstrated for the newly supported sample. Do not silently expand sealed evaluation or previously authorized compute scopes.

No bulk data rewrite, pipeline migration or scientific retraining has been launched. The two explicitly approved diagnostic jobs are complete.

## Additional source evidence recovered during repair investigation

An earlier local audit under `/global/homes/d/dkololgi/fmsu_tokyo_20261010/legacy_selection_tests_20261014/` tested slab 1 using a matter-only header cosmology and failed to recover masses (0.55% within 1% for the cleaned filtered candidate). It supports the need for exact selection reproduction; it is not an independent validation of our newer c000-neutrino candidate.

Its pinned newer source points to the **abacus branch**, not the default master branch, of [amjsmith/hodpy](https://github.com/amjsmith/hodpy/tree/14222dcf794b3bff68987bf8e271203db20f8730). `make_mocks/cut_sky_tools.py` assigns BOX_INDEX=-1 to the entire low-z branch, exports stable HALO_ID for resolved hosts, and includes explicit box/replica coordinates. Those scripts first appear in July 2024 in the inspected branch history, after the delivered v0.1 file's February 2024 timestamp; they are useful design evidence, not a certified reconstruction of that older export. Do not join a new HOD realization to v0.1 merely by adopting its new columns.
