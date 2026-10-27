# Galaxy-position T-Web migration — 19 October 2026

## 20 October consumer fixes and full-catalogue rollout

The readiness review's actionable points are adopted without changing the target
scale or introducing a low-z training extension. `p10_build_phase_index.py` now
defaults to corrected observed truth and a separate `galaxy_realspace_v2` P1
directory. Its label validity uses HAS_LABEL/TWEB_STATUS and finite ordered
eigenvalues, never BOX_INDEX or INDEP_POS. The existing observed-redshift science
shells and context interval remain explicit and unchanged.

The P10 training preparation refuses archived/unbound parent targets and requires
a replacement proof before using existing graph/patch products. The generic SBI
cache builder rejects its old BOX_INDEX filter on corrected catalogues. These
checks close specific silent-mixing paths; they are not an assertion that every
historical training entrypoint has been audited.

The user's requested phase-by-phase audit includes ph000's older single-phase
catalogue lineage. A different ph000 parent membership must be reported and
rebuilt, not assumed equivalent to the newer P10 LSS selection. ph001 truth
access remains separately controlled. Incorrect legacy labels stay archived in
place; retirement does not delete files or erase historical results.

`run_galaxy_truth_benchmark.py` prepares a metadata inventory of all six exposed
P10 phases, independently qualifies ph006 with the existing fixed tolerances,
then rebuilds ph002 through parent metadata, annotation, observed join, P1 and a
full-catalogue comparison. `audit_galaxy_truth_replacement.py` separates catalogue
validity from geometry/fold reuse and measures paired old/new label differences.
It checks host grouping without using the cubic galaxy BOX_INDEX as a host ID.
The benchmark does not fit models or automatically certify graph payload reuse.
Remaining full-phase resources are sized from the measured first-phase run.

Local validation: 41 focused synthetic tests pass. The prepared source snapshot
and tmux runner are under `outputs/galaxy_truth_benchmark_20261020/`; compute
execution and full-catalogue results are pending at this checkpoint.

## Current implementation: all-population v2

The user-approved policy now labels unresolved satellites using their own
RA/DEC/Z_COSMO, just like every other population. The HOD-prescribed satellite
position is used directly; locating a host is unnecessary. `TWEB_STATUS=1` and
`HAS_LABEL=1` describe finite valid field labels. `INDEP_POS` is independent:

| Code | Exact source-row validation evidence |
| --- | --- |
| 0 | This row has no separate original Cartesian cross-check; labels remain usable. |
| 1 | Matched original cubic-catalogue galaxy XYZ. |
| 2 | Matched native resolved central halo XYZ and mass. |
| 3 | Matched native unresolved central field-particle XYZ. |

Codes 1–3 apply only to exact audited SOURCE_ROW identities, never automatically
to an entire population. Sampled low-z satellites of both RES values can have
code 0. This avoids equating shared-transform validation with an independent
check on every galaxy. The coordinate transform is unchanged; the label-policy
version is `galaxy-realspace-all-populations-v2`.

Certificate: `workflows/abacus_tweb/galaxy_position_qualification_allpop_v2_20261019.json`,
with hashed per-phase source-row indices beside it. Default migration outputs
use `catalogues/galaxy_realspace_v2/`. Old v1 certificates, checks and smoke
outputs remain historical evidence and are not overwritten. Phase coverage is
still ph000/ph002/ph003/ph004/ph005; counting another phase does not qualify its
coordinate mapping. The observed join preserves INDEP_POS separately from labels.

The real five-phase CLI smoke now labels all 619 fixture galaxies, including all
14 unresolved satellites, with exact field values at the reconstructed voxels.
The old full-independent-population gate remains false; it is not a mask.

The earlier policy and receipts below are retained as history and are superseded
by this section where they describe masking unresolved satellites. The exact
full-catalogue census and product-lifecycle report are recorded separately.

## 2026-10-19 clarification: satellite coordinates versus independent checks

RA/DEC/Z_COSMO provide a galaxy position for unresolved satellites too. RES=0
does not mean these coordinates are missing. The same pinned transformation can
locate them in the density/T-Web grid without recovering a host halo or particle.
The absence of an independent original Cartesian reference is a validation gap,
not evidence that the reconstructed coordinates are wrong.

Independent checks used: high-redshift resolved galaxies were compared with
original cubic-catalogue galaxy XYZ (including satellite offsets); low-redshift
resolved centrals with native halo centres and mass; unresolved centrals with
native field-particle positions. The unresolved satellites sampled here belong
to the low-redshift branch, for which this audit did not recover a separate
original galaxy XYZ catalogue. A particle gives the pseudo-host centre, not the
satellite's added random offset. Matching that centre cannot reproduce the exact
satellite position. Supplemental host consistency passed for 9/14 sampled
satellites; five lacked a matching selected central. This is not a demonstrated
coordinate failure. Low-z resolved satellites also lack individual original XYZ
checks in this audit; avoid presenting the populations as equally cross-checked.

The recommendation is to use available galaxy coordinates while recording the
level of independent validation separately. The implementation/certificate from
job 60003795 still masks RES=0,CEN=0 (status 2, HAS_LABEL=0); revising that policy
and its tests is pending. Do not mistake this documentation clarification for an
implemented policy change or retroactively alter the saved validation receipts.

Fourteen of 160,000 sampled rows are unresolved satellites (0.00875% of this
sample). Sampling includes special endpoints: this is not an unbiased estimate
of their abundance. Full-catalogue and final-selected-sample counts are unmeasured.


## Scientific contract

Sample the existing periodic R=7 h^-1 Mpc, 2048^3 T-Web field at each galaxy's own **real-space** position reconstructed from RA, DEC and Z_COSMO. Use the pinned c000 distance relation and observer offset; observed redshift is never a substitute. This changes the target from host environment to galaxy environment. It does not establish improved learnability or a stronger galaxy-property correlation.

The high-redshift legacy HALO_INDEX referred to the original halo table. The low-redshift index referred to a filtered halo table: indexing the original table directly could move the lookup hundreds of Mpc. The independent cubic-box audit has a separate indexing convention: BOX_INDEX is local to FILE_NUM, requiring the corresponding block offset in the concatenated cubic catalogue.

## Qualification and limitations

32,000 spread-out source rows per phase were inspected for ph000, ph002, ph003, ph004 and ph005. Independent high-redshift cubic galaxy positions test resolved centrals and satellites. Native halo/field-particle checks in slabs 0 and 33 test low-redshift resolved and unresolved centrals. Shared branch conventions support resolved satellites at low redshift, but those satellites do not have independent original Cartesian position checks in this audit. This is bounded validation, not an exhaustive catalogue proof.

Unresolved satellites (RES=0,CEN=0) are **not independently qualified**. Fourteen occur in the 160,000-row audit sample; that count is not a prevalence estimate because sampling includes endpoint rows. Their generator can add satellite displacements to particle-based pseudo-host centres. A central-particle match therefore cannot certify a satellite position. These rows are retained with TWEB_STATUS=2, HAS_LABEL=0, NaN eigenvalues/derivatives and CWEB=255. Full-population qualification remains false. ph006 is inventoried but not coordinate-qualified; ph001 remains sealed.

## Pipeline changes

- `galaxy_position_mapping.py` defines the versioned coordinate transform; `tweb_grid_sampling.py` samples eigenvalues, classes, and periodic derivatives without loading whole grids.
- `annotate_cutsky_with_tweb_eigs.py` requires a phase-specific certificate bound to the transform hash and source/field metadata. Derived inputs require SOURCE_ROW and exact agreement with the original RA/DEC/Z_COSMO/RES/CEN/HALO_MASS values. Output includes source identity, positions, voxel indices, qualification status and a completion manifest. Existing files are never overwritten.
- Bright-parent metadata preserves original SOURCE_ROW before selection and keeps Z_COSMO, RES, CEN and HALO_MASS. The observed join preserves these columns and unsupported rows. Legacy host-triple joining refuses new-format annotations.
- `migrate_p10_galaxy_position_truth.py` prints a concrete rebuild plan by default; `--execute` requires compute and writes a new `catalogues/galaxy_realspace_v1/` directory. `run_p10_phase_stage.sh` exposes `galaxy-truth`. Production catalogues, graphs, caches and models have not been rebuilt by the validation run.
- Existing selection/redshift masks remain unchanged. A mapping repair alone does not authorize reopening a science gate.

## Rebuild decisions

| Product | Inventory count | Required action |
| --- | ---: | --- |
| P10 labelled catalogues | 12 | Rebuild into new versioned outputs after phase qualification. |
| Compact parent metadata | 5 | Rebuild with original galaxy identity and real-space metadata. |
| Legacy annotation FITS | 34 | Rebuild active dependencies; explicitly retire unused variants. Other smoothing scales need their own qualification. |
| P1 canonical arrays | 6 | Rebuild targets/validity; bind unchanged geometry to the new provenance. |
| P2 graph/union directories | 12 | Reuse only after exact galaxy identity, order and geometry equality. |
| P3 target-free fields | 6 | Reuse only after geometry and selection equality. |
| P3b/P3c response products | 12 | Reuse only if their selections and inputs are unchanged. |
| P4 patch/fold directories | 12 | Preserve split membership; replace/rebind all target-bearing arrays. |
| Cache/model roots | 11 | Trace 112 receipts; replace old targets, fitted scalers, trained models and derived evaluations where dependent. Directory membership alone does not prove contamination. |
| Native density/T-Web roots | 6 | Retain: the bug is in galaxy lookup, not the native field calculation. |
| P12F direct field targets | 6 | Not affected by host lookup; this is not a full physical recertification. |

Rebuilding labels changes target provenance even where numerical labels coincide. Never relabel an old checkpoint as trained on the new target. Geometry reuse requires an explicit equality receipt; metadata-only inventory does not supply that proof. Compare old/new labels on identical galaxies, separately by redshift, RES/CEN and host mass, before attributing weak property-environment relations to this bug. R=7 smoothing can make central/satellite labels similar despite distinct positions.

## Exact inventoried paths

The machine-readable inventory is `outputs/galaxy_position_rebuild_inventory_20261019.json`; it records sizes, timestamps, available metadata and lineage receipts. The bounded scan covers visible P10 phases and immediate legacy roots, not all scratch products. Source files are metadata-bound rather than fully rehashed.

### audit_lineage_before_retraining

- `/pscratch/sd/d/dkololgi/abacus/sbi_caches`
- `/pscratch/sd/d/dkololgi/abacus/SBI_cache files`
- `/pscratch/sd/d/dkololgi/abacus/SBI_cache_micro`
- `/pscratch/sd/d/dkololgi/abacus/phaseB_pooled_ntilde`
- `/pscratch/sd/d/dkololgi/abacus/phaseB_tiled_ntilde`
- `/pscratch/sd/d/dkololgi/abacus/jraph_runs`
- `/pscratch/sd/d/dkololgi/abacus/sbi_runs`
- `/pscratch/sd/d/dkololgi/abacus/regression_runs`
- `/pscratch/sd/d/dkololgi/abacus/e2e_field_v1`
- `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2`
- `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3`

### not_affected_by_host_lookup

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p12f_field_targets_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p12f_field_targets_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p12f_field_targets_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p12f_field_targets_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p12f_field_targets_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p12f_field_targets_v1`

### rebuild_galaxy_labels

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/catalogues/observed/ph000_bgs_bright_full_observed_with_tweb.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/annotated_parent/ph002_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/observed/ph002_bgs_bright_full_observed_with_tweb.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/observed/ph002_bgs_bright_observed_with_tweb.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/catalogues/annotated_parent/ph003_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/catalogues/observed/ph003_bgs_bright_full_observed_with_tweb.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/catalogues/annotated_parent/ph004_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/catalogues/observed/ph004_bgs_bright_full_observed_with_tweb.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/catalogues/annotated_parent/ph005_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/catalogues/observed/ph005_bgs_bright_full_observed_with_tweb.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/catalogues/annotated_parent/ph006_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/catalogues/observed/ph006_bgs_bright_full_observed_with_tweb.fits`

### rebuild_or_retire_legacy_annotations

- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_ng2048_rs4_v2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_ng512_rs16.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_ng512_rs24.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_ng512_rs4.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_ng512_rs8.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_x_L2com.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_ngrid2048_thr0p2_halo_xcom.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_12/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs12_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_12/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs12_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_16/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs16_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_16/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs16_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_20/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs20_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_20/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs20_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_24/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs24_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_8/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs8_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_07042026_rsmooth_8/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs8_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_20260427_rsmooth_10/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs10_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_20260427_rsmooth_11/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs11_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_20260427_rsmooth_6/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs6_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_20260427_rsmooth_7/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_20260427_rsmooth_9/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs9_ngrid2048_thr0p2_15d.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_23032026/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_10/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs10_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_11/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs11_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_6/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs6_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/stage3_postcollision_dedup_science_with_tweb_eigs_rs7_ngrid2048_thr0p2_halo_xcom.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/staged_mock_stage3_postcollision_full_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/staged_mock_stage3_postcollision_rs7_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_9/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs9_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_29042026_rsmooth_7/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits`
- `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_XXXXXX_rsmooth_6/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs6_ngrid2048_thr0p2.fits`

### rebuild_source_metadata

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/bright_parent/ph002_bgs_bright_parent_linkage.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/catalogues/bright_parent/ph003_bgs_bright_parent_linkage.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/catalogues/bright_parent/ph004_bgs_bright_parent_linkage.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/catalogues/bright_parent/ph005_bgs_bright_parent_linkage.fits`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/catalogues/bright_parent/ph006_bgs_bright_parent_linkage.fits`

### rebuild_target_arrays_rebind_geometry

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p1_canonical`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p1_canonical`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p1_canonical`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p1_canonical`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p1_canonical`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p1_canonical`

### retain_native_density_and_tweb

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/targets`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/targets`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/targets`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/targets`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/targets`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/targets`

### reuse_if_geometry_identical

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p2_graph`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p2_union`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p2_graph`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p2_union`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p2_graph`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p2_union`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p2_graph`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p2_union`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p2_graph`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p2_union`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p2_graph`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p2_union`

### reuse_if_geometry_selection_identical

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p3_fields`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p3_fields`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p3_fields`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p3_fields`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p3_fields`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p3_fields`

### reuse_if_selection_identical

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p3b_random_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p3c_assignment_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p3b_random_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p3c_assignment_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p3b_random_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p3c_assignment_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p3b_random_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p3c_assignment_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p3b_random_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p3c_assignment_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p3b_random_response_v1`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p3c_assignment_response_v1`

### review_target_bindings_preserve_folds

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p4_patches`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p4_rebuild`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p4_patches`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p4_rebuild`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p4_patches`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph003/p4_rebuild`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p4_patches`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph004/p4_rebuild`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p4_patches`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph005/p4_rebuild`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p4_patches`
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p4_rebuild`


## Validation receipt
Allocation 60003795; five phases, 32,000 rows each. All 18 focused regression tests passed. The real annotation CLI passed on 619 source-bound stratified rows (605 labelled, 14 preserved unlabelled), including independent voxel reads, metadata equality and overwrite rejection.
| Phase | Cubic centrals | Cubic satellites | Native resolved centrals | Native unresolved centrals | Max cubic residual (Mpc/h) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ph000 | 22151 | 3097 | 1082 | 273 | 0.00114617 |
| ph002 | 22187 | 3106 | 1063 | 276 | 0.00114716 |
| ph003 | 22154 | 3111 | 1053 | 268 | 0.00116183 |
| ph004 | 22083 | 3060 | 1095 | 268 | 0.00128799 |
| ph005 | 22126 | 3090 | 1055 | 280 | 0.00123736 |

All tested cubic identity/mass and native central-position checks passed; no web-class changes between reconstructed and independent positions in these samples. Floating-point position differences occasionally cross a voxel boundary, so eigenvalues are not everywhere bit-identical to independent reference positions. The smoke check instead requires exact equality at the reconstructed production voxels.

Supplemental unresolved-satellite checks found compatible central-particle hosts for 9/14 sampled satellites, with matching mass and offsets inside the adopted halo radius. Five lacked a matching central in the selected catalogue. This does not supply independent original satellite positions; all fourteen remain uncertified.

Evidence is retained in `docs/evidence/galaxy_position_20261019/`; the pipeline certificate is `workflows/abacus_tweb/galaxy_position_qualification_20261019.json`. Initial incorrect global BOX_INDEX reference results remain recorded within the coordinate audit, alongside the corrected file-local convention. Original native-audit script snapshot: `outputs/galaxy_position_validation_initial_source_60003795.py`.

Example: activate cosmic_env and run `python workflows/abacus_tweb/migrate_p10_galaxy_position_truth.py --phase ph002 --qualification workflows/abacus_tweb/galaxy_position_qualification_20261019.json` to print the concrete plan. Add `--execute` only in an approved compute step. The five qualified phases use this certificate; ph006 requires an additional audit. Full production rebuilds and matched-sample scientific comparisons remain future work.

Allocation 60003795 was released and accounted COMPLETED, elapsed 31:28, exit 0:0. Initial pytest and graph-refresh launch failures were resolved; exact steps are preserved in `execution_receipt.json`. Local AST graph and global references refreshed. Syntax, evidence hashes, certificate binding and patch whitespace checks passed.

Full census completed: 120,600/1,409,448,712 galaxies (0.008557%) across 22 exposed fiducial catalogues. See `docs/unresolved_satellite_census_20261019.md` for per-phase counts and selection limits.

Final v2 verification: 25 focused tests passed, the guarded cache entrypoint and versioned migration plan passed, and local/global code graphs were refreshed. Allocation 60005529 was released after 29:46, COMPLETED with all steps exit 0:0. Final execution and checksum receipts are in `docs/evidence/galaxy_position_allpop_v2_20261019/`.
