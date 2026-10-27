# Shared P12-A / CFM phase policy — 2026-10-27

User-adopted policy for new R7 training campaigns. Machine-readable authority:
`configs/shared_phase_policy_20261027.json`. This supersedes earlier phase-role
restrictions for new P12-A, coupled CFM and whole-field CFM fits, not historical
results, frozen run contracts or existing DESI products.

| Role | Phases |
| --- | --- |
| Training (20) | ph000–005, ph007–011, ph014–017, ph020–024 |
| Development/calibration (3) | ph006, ph012, ph013 |
| Final test (2) | ph018–019 |

Previously exposed ph001 and ph014–017 can be used for new training; they cannot
then be presented as independent tests. ph018–019 are excluded from fitting,
normalization, selection and calibration. Assess them only after freezing the
model and calibration procedure under a registered evaluation contract. Two
test phases provide limited phase-level uncertainty; report that limitation.

Keep Gaussian R=7 h^-1 Mpc, corrected own-position all-population P12-A labels,
native-density CFM targets, ordered P12-A eigenvalues and deterministic restart
requirements. Use fresh initialization and fixed-budget terminal checkpoints.
Fit normalization on training data only. Within development phases, separate
calibration, evaluation and selection spatial subsets; P12-A retains folds
0–1 / 2–4 / 15–17 with phase-cap-superblock identities. Register the corresponding
CFM spatial partition before its use; a common phase policy does not imply an
identical target or loader.

## Readiness at adoption

This is a metadata/record audit, not a new full-payload checksum or loader pass.
The October 26 numerical replay passes remain evidence for the repaired code;
they do not certify the changed data split. No compute or fits launched.

**P12-A:** the old 15 training + 3 development phases have corrected encoder
inputs and all 18 response products. The five newly admitted training phases
(ph001, ph014–017) have `BLIND_INPUT_COMPLETE.json` geometry receipts, but no
`catalogues/GALAXY_TRUTH_MIGRATION_COMPLETE.json` at the corrected v2 root.
Their corrected labels, target-bearing caches and response bindings must be
built/qualified. ph018–019 likewise remain geometry-only in this P12-A root;
their truth preparation/evaluation is separate from training readiness.

Then build a new full-20-phase contract and **20 leave-one-phase-out contracts**
(19 training phases each), fitting each fold's transforms without its omitted
phase. Reuse verified target-free geometry and compatible response products;
do not copy old train-fitted scalers. Fit 20 fresh OOF encoders plus one full
encoder; export halo48 predictions, align response/posterior tables, then fit
FMPE and perform fresh calibration/evaluation. Old 15-fold contracts and old
calibration certificates do not certify this campaign.

**CFM:** the coupled preparation has qualified products for 21 phases under the
old 13/2/6 split (see `docs/e2e_coupled_preparation_closeout_20260919.md`). Its
`cartesian_v2/targets` contains those 21 phase directories, but not ph001,
ph004, ph005 or ph006. Build/qualify those four phases in the CFM coordinate,
observation/response and native-density target conventions; P12-A readiness is
not interchangeable with CFM readiness. ph014–017 already have field products,
but their held-out geometry has 16 pairs/phase versus 128 for training in the
old coupled contract: expand training sampling or explicitly register a revised
balanced sampling design. Rebuild the 20-phase normalizer and role-bound
interfaces, and prepare the selected whole-field/domain representation. The
656-domain October 26 loader pass covers only the old selected dataset.

## Migration and handoff

The policy JSON is a decision manifest, not a trainer CLI configuration. Existing
runtime defaults still name historical contracts. Before a new launch, version
the P10 registry, P12-A full/OOF/response contracts and CFM data/training/evaluation
contracts to implement this policy; use new output roots, explicit source/policy
hash bindings, and focused role/normalization/loader checks. Do not edit old
hash-bound configs, completed reports, receipts or checkpoints in place, and do
not bypass their guards to admit new phases. The old configurations must not be
used as new-campaign launch authority.

Training budgets and Slurm resources need separate approval. No old resource
cap or job authorization transfers to the new split. Production VAC must consume
a new checkpoint/transform/phase-policy manifest with fresh calibration and
golden-mock parity; existing DESI predictions retain their original provenance.
