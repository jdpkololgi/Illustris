# P12-A handoff follow-up, 2026-09-24

These checks advance V0/V1; they do not authorize DESI inference. All numerical
work used the existing CPU allocation 58823514 (nid004174), except the small
401-point distance comparison and metadata checks. No fitting or reserved-phase
payload access was performed.

## Passed checks

- **ph000 import:** full SHA256 equality of original/imported catalogue and point
  files; original catalogue matches its P1 recorded hash. Direct original
  Planck18/SkyCoord replay is exact for 16,384 sampled rows. All 128 native
  x_com labels/classes match, with 16 rows in each cap/shell. Evidence:
  `evidence/p12/P12A_PH000_IMPORT_CLOSURE_20260924.json`.
- **Full-fit and five OOF encoders:** checkpoint, scaler, transform, loader and
  adapter/P3 receipt checks pass. Embedded normalizations/scalers equal their
  frozen JSON contracts and every selected checkpoint is epoch20. Cross-fit
  loader markers do not hash adapters themselves: current adapter bytes instead
  match the separately pinned full-fit inventory. Field payloads and full OOF
  summary arrays were not rehashed. Evidence: `P12A_SMALL_ENCODER_LINEAGE` and
  `P12A_EMBEDDED_TRANSFORMS` JSONs under `evidence/p12`, dated 20260924.
- **Historical source:** each checkpoint's six recorded source hashes can be
  recovered exactly from its launch Git revision. Three current source files
  differ from the full-fit receipt. The U-Net change extracts `sample_latent`
  from the unchanged forward computation; preserve a final-checkpoint replay
  requirement rather than treating source drift as automatic retraining.
- **Annotation provenance:** ph002-005 historical logs explicitly report x_com,
  zero skipped parent rows and their annotated output paths. ph000's original
  annotation log has not been located; its numerical/import checks above are
  separate evidence. See `P12A_ANNOTATION_SOURCE_LINEAGE_20260924.json`.
- **Loa live sources:** full and clustering data SHA256 match the recorded
  release. All 36 random files match size, row count and ordered column names;
  random content hashes are still unverified. An 8192-row full-data sample
  checks field availability and finite-value counts, not a selection policy.
  Evidence in GraphWeb_DESI: `docs/p12a_loa_source_audit_20260924.json`.

## Coordinate and response interpretation

The pinned native DESI/Abacus radius table differs from Planck18 times legacy
h=0.6766 by up to 2.178905 Mpc/h over z=0.15-0.55. The same-unit radial-volume
ratio is 1.001838-1.006097. This is a chart comparison, not a satellite/host
physical offset or the full Mpc-to-Mpc/h density conversion. P12 must retain
its Mpc coordinates and fitted selection together; corrected E2E products are
not interchangeable. See `P12A_NATIVE_DISTANCE_COMPARISON_20260924.json`.

The R0 encoder uses counts, exposure_apodized and log_count_ratio. Its loader
recomputes selection-dependent channels using the frozen per-contract selection
curve. The inherited adapter's old wedge-selection warning alone does not
establish that the frozen encoder consumed that old curve. The posterior's
support-boundary feature comes from the P3b random-support cache, replacing the
older artificial fold-boundary distance. Both paths still need numerical replay
in the production adapter.

The installed LSS `goodz_infull('BGS', ...)` requires ZWARN==0 and DELTACHI2>40.
GraphWeb's historical catalogue builder instead used DELTACHI2>=25 and GALAXY.
Do not silently reuse that legacy builder for P12. Pin the Loa release's success
policy and measure its count/response impact before adapter qualification.

## Population follow-up

An exact raw CutSky (RA,DEC,Z) join recovered all128 ph002 native-label rows,
with unique matches and identical FILE_NUM/HALO_INDEX/BOX_INDEX. Their raw flags
are CEN=1,RES=1 for119 rows and CEN=0,RES=1 for9 rows. Thus the existing exact
native-label agreement covers both central and satellite rows in this phase;
it does not cover RES=0. The compact parent discards CEN/RES, so resolving the
remaining population scope requires raw-source joins, not a guessed flag from
row order. Evidence: `P12A_POPULATION_SAMPLE_PH002_20260924.json`.

## Remaining gates

Recover and interpret CEN/RES population coverage, especially populations excluded
by the current BOX_INDEX>=0 native-sample rule. Complete response/selection replay,
Loa success-policy and random-content freeze, final-checkpoint golden mock and
phase exposure ledger. Current status remains coordinate_audit_status=unresolved,
ready_for_desi_canary=false. No evidence so far requires retraining.

## Execution and verification

Allocation58823514 completed and released after19m05s. ph000 audit16s; live Loa
source check37s; embedded transforms7s; raw population join3m17s. A first
embedded-transform launch failed because a login-node `/tmp` script is not
visible on the compute node. Moving the reproducible script into the repository
resolved it; the subsequent step passed. This was not a data/model failure.
Nine coordinate-helper and five observational-preflight tests passed. Repository
whitespace checks and required graphify refresh completed.
