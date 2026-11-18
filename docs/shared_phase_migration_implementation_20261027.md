# Shared R7 phase migration: implementation and preparation handoff

The new campaign explicitly binds the adopted 20/3/2 policy. Historical defaults,
products and comparison results remain available. Historical results retain the
model and phase roles under which they were obtained; new-model comparisons can
include exposed/training phases, with that role stated. No extra historical-result
exclusion is introduced.

## Latest preparation evidence

The four-node follow-on run is recorded in
[shared_phase_preparation_60355333.md](shared_phase_preparation_60355333.md).
It completes the missing R7 fields and CFM observations, one new full P12 phase,
and one complete CFM target phase; the new campaign is still not training-ready.
The report records the ph017 nearest-cell qualification issue and resumable stages.

## Implemented

- `configs/p10_phase_registry_v3.json` routes P12 preparation to a new scratch root.
  Registry, full-loader and OOF-loader checks validate the shared policy hash,
  exact roles and R7. Every OOF contract excludes exactly its one training phase.
- `GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_20phase_20261027.json`
  selects the new field preparation campaign. Unset retains historical defaults.
  New normalization uses all 20 training phases with equal phase weights;
  loaders reject a transform fitted on the historical phase set. Whole-field
  roots and development roles follow the selected campaign.
- P12 response drivers accept explicit roots, registry and output directory.
  Source registries propagate the phase-policy binding. The ph001 exception is
  confined to the new campaign; historical blind guards remain intact.
- `shared_phase_migration inventory` generates the bounded metadata inventory;
  `verify` checks explicitly hash-bound candidate files on compute;
  `bind-p12` creates a separate runtime input config after a full contract exists.
  The checked-in P12 config is deliberately a template with a null full-contract
  hash, so it cannot be mistaken for a completed data release.

## Reuse and rebuild inventory

`shared_phase_reuse_inventory_20261027.json` records all 23 preparation phases.
It finds 18 corrected P12 completion markers and 19 CFM target completion markers.
These are existence/hash checks on metadata, not renewed payload certification.

| Product | Required action |
|---|---|
| Existing 18 corrected P12 phases | Verify source/label and geometry bindings; reuse immutable physical products with new campaign provenance |
| ph001 P12 truth | Existing density and 16-rank R7 T-Web receipts are candidates for reuse; verify payloads and qualify own-position annotation |
| ph014–017 P12 truth | Reuse native 2048-grid TSC densities after verification; build R7 T-Web and qualified own-position labels |
| Five newly training P12 phases | Complete corrected P1/label products, closure and phase proofs; reuse geometry only after identity checks |
| P12 transforms | Refit full-20 selection/target/field transforms and all 20 OOF transforms on 19 phases each |
| P12 random response | Reuse angular maps only after source/hash validation; rebuild parent response against the new selection |
| CFM ph001/004/005/006 | Qualify existing native density sources; prepare missing observations, geometry and targets |
| CFM ph014–017 | Expand from 16 to 128 pairs per phase, then 48 whole-field domains per phase |
| CFM selection | Retain the existing five-phase selection function; all five remain training phases |
| CFM model normalization | Refit on 20 phases; regenerate or explicitly re-express coarse/exterior interfaces under that chart |
| Models | Later: 21 fresh P12 encoders, halo48 exports and aligned posterior tables; fresh CFM fits |

Old CFM receipts bind a different configuration. Copying them under the new root
is not an adoption procedure. New readiness must bind verified physical source
products and regenerated geometry/normalization. New model training and whole-field
coarse/exterior deployment are not certified by these code changes.

## Bounded compute preparation

The user approved and completed the smoke in allocation 60354736 on nid200035.
The allocation was released after 6m49s; the preparation step passed in 6m12s.
The approved smoke used one CPU node, 64 CPUs, account `desi`, QOS `interactive`,
60 minutes, Scratch license, and releases its allocation on completion or failure.
No GPU, training, particle restore, cleanup of source data or production submission.

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=64 --constraint=cpu \
  --qos=interactive --account=desi --time=01:00:00 \
  --licenses=scratch --immediate=600
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores \
  bash workflows/sbi/run_shared_phase_migration_smoke.sh
```

The reviewed script verifies ph001 density/T-Web payloads, checks the ph000
corrected-phase bound receipts, creates a verified ph014 native-density adapter,
and runs density/T-Web input preflight without constructing the field. Outputs
and a log go to the new campaign's `preparation_smoke_JOBID` directory. This is
a representative reuse/preparation check; it does not complete all missing data.
The full 2048-grid T-Web construction needs a separately sized multi-node run.
Use the smoke measurements before requesting the remaining preparation budget.

## Validation and limits

46 focused unit tests passed, including legacy registry/field behavior, all 20
OOF role partitions, changed policy hashes, final-test refusal, expanded input
routing, parent-response identity, and normalization/loader regressions.
`pytest` is absent in cosmic_env; these are the repository's unittest tests,
run with unittest. The approved smoke now verifies the listed ph001 payloads, ph000 bound receipts,
and ph014 native-density payload. No new fit has run. Graph refresh was attempted
on compute and failed with fcntl.flock OSError 524; no locking bypass was used.
R7 is unchanged and smoothing investigations remain deferred.


## Completed smoke: allocation 60354736

`shared_phase_smoke_60354736.json` binds the exact source/configuration hashes,
output receipt hashes, log path and scheduler outcome. ph001 verifies 27 unique
files including completion receipts (146,029,081,192 bytes); all source hashes
match. ph000 checks 13 bound metadata files, not all underlying payloads.
ph014 verifies the 32 GiB native count array and writes a P10-compatible adapter;
T-Web input preflight passes 2048³ float32, all 136 particle files and relative
count error 9.505277764399594e-08. The native array is reused in place, not copied.

The initial verifier reread duplicate references in nested receipts. After the
successful smoke, this was optimized to hash each resolved file once while
rejecting conflicting hash/size records or changed mtime; a focused regression
test covers those cases. Source bindings in the smoke report identify the
original implementation actually run. The graph index remains stale because
both local update and subsequent global refresh require a successful lock.

These results establish reusable input bytes and a prepared density adapter.
They do not establish own-position annotation compatibility, full phase adoption,
completed new labels/field interfaces, new normalization or training readiness.
The next preparation stage needs qualified source adoption plus multi-node R7
T-Web construction for ph014–017, under a separately reviewed resource budget.

## Next representative construction: prepared, not authorized or launched

`workflows/sbi/run_shared_phase_ph014_tweb.sh` prepares one ph014 R7 T-Web
construction using the existing ph001 layout: four CPU nodes, 16 MPI ranks,
four ranks/node, 32 CPUs/rank, one-hour ceiling, account desi, QOS interactive,
Scratch license. The 16-rank layout keeps the 2048-grid transpose below the
2 GiB message-count boundary. It rechecks the native-density hash, writes a
new per-job adapter, and constructs only ph014 under the new campaign root.
Existing output directories cause refusal, not overwrite or automatic retry.

After explicit resource approval:

```bash
salloc --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=32 \
  --constraint=cpu --qos=interactive --account=desi --time=01:00:00 \
  --licenses=scratch --mem=0 --immediate=600
bash workflows/sbi/run_shared_phase_ph014_tweb.sh
exit
```

Release the allocation after completion/failure. Validate its rank coverage,
physical checks and completion receipt before sizing ph015–017 or launching
annotation work. This next allocation is separate from the released one-node
reuse smoke; it is not covered by that earlier approval.
