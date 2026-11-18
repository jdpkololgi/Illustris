# P12-A low-z preparation checkpoint — 29 October 2026

**Subsequent execution:** CPU60477149 completed the23-phase raw-grid expansion
and separate geometry-only parent support/boundary arrays. Radial selection,
host/fold qualification, full/OOF transforms and loader integration remain open.
The allocation auto-logged out at30m18s; no replacement/retry was launched.
See `lowz_response_60477149.md/json` for current status. The earlier canary and
approval checkpoint below is retained as provenance, not a fresh launch budget.

## Request and fixed scope

Create the inputs needed for fresh P12-A training over **0.01 <= z < 0.55**,
then train one full and 20 leave-one-phase-out encoders, export halo48 summaries,
construct aligned posterior tables, and train/evaluate the ordered-increment
posterior. Retain Gaussian R7 and the shared 20/3/2 phase policy. ph018/ph019
remain sealed until the final protocol is frozen. No old-range fit substitutes
for the requested low-z model. The source simulation snapshot remains z=0.2.

The October27 bound input release has valid corrected labels but science shells
0.15–0.55 and context0.10–0.60. Its preparation receipts do not qualify low-z
training. P12-A parent-response sampling also had a literal0.10 lower bound.
CFM has the same domain mismatch, but its native Mpc/h grids are not interchangeable
with P12-A's observer-frame Planck18 Mpc representation.

## Implemented and verified

- `p10_redshift_contract.py` centralizes versioned science, context, sentinel,
  four-shell and fine-diagnostic metadata; legacy defaults remain unchanged.
- `p10_build_phase_index.py` consumes the registry's domain, preserves corrected
  label validity, writes range metadata and rejects incompatible reuse.
- `p12a_parent_response.py` consumes the registered context instead of a hardcoded
  0.10 floor, checks distance-table coverage, excludes the undefined observer
  direction, and checks P1/P3/selection domain agreement.
- `p3a_build_canonical_fields.py` supports verified cap-separated angular random
  maps and deferred selection. New raw products contain real CIC counts/support;
  selection-dependent channels are explicitly zero placeholders until a fresh
  training-only fit. They are not model-ready. No old spline is extrapolated.
- `prepare_p12a_lowz_inputs.py` implements the bounded two-phase raw build;
  `audit_p12a_lowz_groups.py` audits legacy duplicate keys.
- `configs/p12a_lowz_preparation_20261029.json` registers positive-z context below
  the0.01 supervision floor, ceiling0.60, sentinel exclusion0.585–0.595, unchanged
  5-Mpc cells and four training shells (first widened to0.01–0.25). Fine low-z
  diagnostic edges are0.01/0.03/0.05/0.075/0.10/0.15.

21 focused synthetic/regression tests pass using isolated `cosmic_env` unittest:
`tests.test_p12a_lowz_preparation`, `tests.test_p12a_lowz_canary`,
`tests.phase4.test_p12_parent_response`, `tests.phase4.test_p3_field_utils`,
`tests.phase4.test_p10_phase_index`. pytest is not installed. No environment was
modified. Patch whitespace checks passed.

## Executed products

Approved CPU allocation **60475597**, nid200076: one node,64CPU steps,
desi/interactive/Scratch, <=1h, no retries. Allocation released; sacct reports
COMPLETED0:0 after5m58s. Raw preparation step completed0:0 in1m59s; group audit
completed0:0 in27s. Native Shifter ownership was0:0.

Root: `/pscratch/sd/d/dkololgi/abacus/p12a_lowz_20261029_v1`.
Inspect `RAW_PREPARATION_COMPLETE.json`, `RUN_BINDING.json`, and each phase's
`RAW_PREPARATION_COMPLETE.json`, `INDEX_AUDIT.json`, `LOWZ_GROUP_AUDIT.json`,
`p1_canonical/`, `p3_fields/`, `contracts/`, and `logs/`.

| Quantity | ph001 | ph006 |
|---|---:|---:|
| Supervised rows0.01–0.55 | 7,321,047 | 7,321,104 |
| Context rows | 7,327,682 | 7,329,765 |
| Rows0.01–0.10 | 1,075,312 | 1,074,166 |
| Sampled low-z positions inside rebuilt grids | 4,096/4,096 | 4,096/4,096 |
| Random-supported sampled voxel centres | 4,039/4,096 | 4,052/4,096 |
| Raw build elapsed, parallel phase workers |111.7s|110.3s|

All8,192 sampled positions lie inside the grids;8,091 have random-map support.
Unsupported boundary samples are not silently promoted. Source/catalogue and
angular-map hashes were verified; raw-random provenance was inherited. This is
not independent tidal resampling, mass-completeness validation, a selection fit,
or model training. `model_inputs_ready=false` remains explicit.

## Newly exposed duplicate/fold blocker

The existing `p4_build_spatial_manifest.py` deduplicates by
`FILE_NUM,BOX_INDEX,HALO_INDEX`, retaining one row per tuple, then requires
repeated objects to be separated by at least1000Mpc. This is not valid for the
new low-z population: negative BOX_INDEX is not missing truth, and many distinct
satellites share host indices. RES namespaces can also collide.

For0.01 <= z <0.15, ph001 has2,334,047 distinct SOURCE_ROW identities, and ph006
has2,352,896. The old rule would discard291,223/291,339 rows respectively,
**582,562 distinct catalogue rows combined**. Every sampled adjacent repeated-key
pair (8,192 per phase) is closer than1000Mpc; median distances are4.999/4.815Mpc.
There are17,646/17,893 adjacent repeated pairs with differing RES values.
The old periodic-copy rule must not be applied or its gate simply relaxed.

Separate true galaxy periodic-copy identity from host grouping. High-z BOX_INDEX
is local cubic-galaxy identity within FILE_NUM; low-z HALO_INDEX belongs to a
filtered namespace. SOURCE_ROW gives exact CutSky row identity but does not by
itself prove periodic-copy equivalence. Resolved/unresolved namespaces also need
separation. Existing `audit_galaxy_truth_replacement.host_fold_audit` explicitly
does not qualify low/high namespace closure. Their real-space radial ranges
overlap (ph001 low max Z_COSMO0.159665, high min0.140225; ph0060.159548/0.135084).
Do not assume that adding a branch bit proves no same-host crossing.

Relevant prior evidence: `docs/galaxy_position_migration_20261019.md`,
`galaxy_position_mapping.py`, `audit_lowz_lookup_positions.py`,
`check_unresolved_satellite_hosts.py`. The old filtered-index hypothesis recovered
only62/135 sampled centrals; it is not a production host mapping. Own-position
tidal labels remain valid independently of these host identifiers.

## Next implementation and execution

1. Repair and qualify periodic-copy versus host-fold treatment, without dropping
   distinct low-z galaxies or silently weakening calibration/evaluation separation.
2. Complete field-only spatial/adaptor preparation; P12-A's U-Net does not need
   a fresh graph merely because the historical combined contract builds both.
   The current P4 probe uses P2 only for graph cost estimates; the spatial writer
   binds its metadata. Keep historical graph workflows intact.
3. Propagate range metadata into P10 selection/full transforms, all20 OOF
   selection/transforms, loader readiness, training shell labels and posterior
   inputs. Current P10/crossfit constants and p8/trainer shell labels still have
   old defaults. New metadata alone does not complete this integration.
4. Validate new full-context training-only selection, low-z closure and sampling,
   full/19-phase normalization, response alignment and actual loader behavior.
5. Then request a concrete GPU launch. No GPU allocation or training has yet
   occurred. Fixed encoder/posterior training epochs remain unset pending
   measured throughput. Historical full epoch336,579updates and all21 encoders
   6,731,580updates/epoch are old-range figures, not a new budget.

The user has now **explicitly approved a new preparation window**, not yet
allocated: one CPU node,64CPU step, desi/interactive/Scratch, <=2h, all23
training/development phases, duplicate/fold repair validation, selection/response
and full/20-OOF contracts, local AST graph refresh; no final-test access, no
automatic retries, release afterward. GPU training is a separate launch.
Prepare/test code before starting that clock. Existing raw products should be
reused with verified provenance, not overwritten or blindly rebuilt.

The graph refresh attempted under the first allocation was rejected by command
review as outside its narrower approval. It was not executed or bypassed. The
new explicit approval includes the graph refresh. The existing graph query
worked; no refreshed-index claim is made.

The worktree also contains the other chat's uncommitted preparation and CFM
changes. Preserve them. Home source/configuration/provenance stays in Git;
large products stay in Scratch. Freeze exact source hashes for each run.
