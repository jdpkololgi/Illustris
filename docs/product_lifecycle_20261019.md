# Product lifecycle after the galaxy-position annotation repair

## What this register does

The canonical registry is `configs/product_lifecycle_20261019.json`. It separates
confirmed legacy targets, unknown/review-required dependencies, native/direct-field
products, corrected replacements and historical products archived in place.
The scan is bounded to the named owned Scratch roots and small JSON receipts.
No checkpoint or density payload was opened for the dependency audit. Source
snapshots, log directories, symlinks and sealed phase directories were not traversed.
This is not an exhaustive proof of every consumer in both repositories.

**Archived in place** means the files remain at their original paths, while the
registry records that they are historical and must not supply new galaxy-position
science targets. It preserves reproducibility and running readers. It does not
free disk space or prevent tools that ignore the registry from reading old files.
The P10 observed-truth join and standard Abacus SBI cache builder reject
registered historical target inputs before building. Their narrow default guard
does not certify unknown inputs; normal provenance checks still apply. The
standalone strict consumption guard is intended for the new galaxy-label contract;
it is not a blanket block on target-free geometry or direct-density field work.

## The conditional-field distinction

The coupled R7 target builder reads native A+B particle density through
`matter/PHASE/DENSITY_COMPLETE.json`, then constructs density and tensor targets.
The whole-field pilot also reads that native density and samples rho directly.
Neither route obtains these targets from per-galaxy host-index eigenvalues.
Therefore the galaxy-annotation bug does **not** automatically invalidate their
checkpoints, density samples, coarse fields or field-target arrays.

Evidence: `workflows/sbi/e2e_coupled_target_products.py`,
`workflows/sbi/e2e_wholefield_pilot_data.py`, and
`docs/e2e_coupled_preparation_closeout_20260919.md`. The current source hashes are
recorded in the registry. Older coordinate conventions, partial generations and
failed calibration results have separate limitations; this audit does not
recertify them. Preserve scientific controls and negative-result evidence.

## Safe replacement sequence

1. Rebuild the source metadata and annotate into `galaxy_realspace_v2/` using
   `galaxy_position_qualification_allpop_v2_20261019.json`. Keep all populations;
   retain INDEP_POS separately. The five-phase certificate does not authorize
   untested phases or reopening existing selection gates.
2. Validate source-row identities, row order, coordinates, finite ordered labels,
   class consistency, periodic sampling and completion hashes. Keep the old files.
3. Replace dependent P1 targets, target-bearing caches, fitted scalers, model
   checkpoints and evaluations under new versioned paths. Reuse geometry only
   after exact equality checks; a catalogue reference alone does not prove that
   an observation-only consumer read its labels.
4. Register the corrected replacement and its receipt. The provided FITS adapter
   checks the v2 policy, output hash and exact shared identity/geometry columns,
   including galaxy order. Models/caches require their own validation adapter;
   a filename change is not a validated replacement.
5. After consumers have migrated, prepare an exact physical-move manifest. Check
   live jobs and unscanned consumers first. Never move a mixed product root merely
   because one generation is obsolete. Keep a path-to-archive mapping and rollback
   instructions; do not silently redirect a legacy filename to changed targets.

The native density and T-Web grids need no rebuild for this lookup defect.
No full production reannotation, cache rebuild or model retraining has been run
by this task. The v2 CLI is tested on the source-bound 619-row fixture.

## Commands

All commands run from the Illustris repository in cosmic_env. Full scans and
replacement hashing run in an approved compute allocation.

```bash
python workflows/abacus_tweb/product_lifecycle.py \
  --registry configs/product_lifecycle_20261019.json check /absolute/product/path

# Produces a NEW registry snapshot; leaves all product files in place.
python workflows/abacus_tweb/product_lifecycle.py \
  --registry configs/product_lifecycle_20261019.json archive-in-place \
  --paths /exact/registered/legacy/product --output /path/to/new_registry.json

# Only after the corrected FITS exists and its annotation checks pass.
python workflows/abacus_tweb/product_lifecycle.py \
  --registry configs/product_lifecycle_20261019.json register-replacement \
  --old /exact/legacy/catalogue.fits --receipt /corrected/catalogue.fits.manifest.json \
  --reason 'Validated same-galaxy migration to own-position targets' \
  --output /path/to/new_registry.json
```

## Redundancy review

A newer `_v2` or `_v3` name is a review cue, not proof that the older run is
redundant. The inventory also flags generation payloads without references in
the scanned receipts; external consumers may still exist. Neither cue permits
removal. Establish exact duplicate payload hashes, an explicit validated
successor, or a documented incomplete generation before physical retirement.
Scanned byte totals are logical file sizes, not measured reclaimable disk space.
The registry keeps these candidates distinct from confirmed legacy labels.
Scratch archival remains purgeable and is not a durable backup.

## Completed inventory and tags

The registry covers 300 registered products/groups and 875 grouped dependency links. 52 legacy target products/groups plus one documented incomplete field shard are archived **in place**. 10 further retirement candidates require review. No files were moved or deleted.

| Status | Products/groups |
| --- | ---: |
| archived_in_place | 53 |
| review_reuse_or_metadata | 47 |
| retain_native_or_direct_field | 12 |
| lineage_review_required | 188 |

The exact paths and reasons are in `configs/product_lifecycle_20261019.json` and `docs/evidence/galaxy_position_allpop_v2_20261019/product_review_queue.csv`. The full raw metadata evidence is retained at `outputs/product_lifecycle_optimized_20261019.json`.
