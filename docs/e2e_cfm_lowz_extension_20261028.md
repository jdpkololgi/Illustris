# CFM low-redshift extension: retraining preparation

## 2026-10-29 user correction — current authority

Low-z-covered **fresh training is required**, not contingent on frozen-checkpoint
accuracy or calibration below the original training range. Preserve saved models,
normalizers and results as historical references. The immediate task is to build
and qualify new training inputs over supervision **0.01–0.55**, with available
positive-redshift context below 0.01 and explicit missing-support/origin masks.
R7 and the shared 20/3/2 phase policy remain fixed. Evaluate new models by the
registered low-z bins after retraining; old-weight extrapolation is not the gate
for deciding whether to retrain, nor a prerequisite for preparing these inputs.

The other chat completed the ph001/ph006 label/support canary, not a low-z
selection/response release: see `docs/p12a_lowz_canary_60475000.md` and its JSON.
It verified 2,149,478 combined low-z labels and angular support, while old 3D
response support covered only 3,213/3,035 rows respectively. Its label checks are
not independent native-field resampling; magnitudes are not audited masses.
The October27 deterministic release remains an old-range release.

New implementation and exact limitations:
`docs/cfm_lowz_selection_response_20261029.md`,
`configs/cfm_lowz_selection_response_20261029.json`, and
`workflows/sbi/cfm_lowz_selection_response.py`. It prepares native-coordinate
training-only selection statistics/fits and response grids in a separate root.
No real low-z selection fit, response grid, training-domain manifest or matched
R7 target was generated in this editing turn. Separate compute approval is needed.

## Historical October28 audit (retained evidence; plan superseded where noted)

October28 status: **metadata/source audit completed; payload canary, low-z model inputs,
predictions and plots NOT completed**. No job, training, model replacement or
final-test payload access was performed. The current session cannot start
Python/Graphify because its managed sandbox fails to create a directory under
`/cray/tmp/tmp/.pam_namespace/dkololgi/codex-daemon-103008`. Do not change security
hooks, use an SSH relay, or modify session permissions to work around this.

## Finding

The former galaxy/halo-index failure is not evidence that Abacus is intrinsically
unusable below z=0.15. The field-model lineage is different: native 2048-cubed
matter density is normalized by its periodic box mean and Gaussian-smoothed once
at R=7 Mpc/h before the CFM target is constructed. This path does not require the
incorrect galaxy-to-host tidal-label lookup. See `workflows/sbi/e2e_coupled_operators.py`
(`smoothed_spectrum`), `CONTEXT.md`, and the current `SCIENCE_LOG.md`.

Nevertheless, **none of the audited models has established low-z conditional
performance**. Three different ranges must be distinguished:

| Quantity | Frozen coupled CFM | Expanded whole-field U-Net / transformer |
| --- | --- | --- |
| Training phases, each seed | 13 | Same 13 |
| Bound training domains, each seed | 1,664 | 624 |
| Actual domain-centre z range from metadata | 0.1501566–0.5494336 | 0.1537990–0.5494336 |
| Local field shape / cell | 64×48×48 / 6.766 Mpc/h | 96×64×64 / 6.766 Mpc/h |
| Local boxes intersecting z<0.1 | 145 | 115 |
| 48-cubed wide boxes containing observer | 513 | 192 |
| Observed-galaxy context | 0.10≤z<0.60, sentinel excluded | Same |

Both seeds have identical domain selections. For the coarse model, the local-box
column describes its associated paired fine domain, not the coarse output grid.
These are continuous box intersections, **not exact voxel exposure counts**.
Their minimum local-box radii are 151.735 and 65.009 Mpc/h respectively; the
48-cubed wide context has 27.064 Mpc/h cells. Fields can therefore extend into
low-z space while observations there remain masked. Seeing such a density
region during training is not the same as learning its conditional response to
low-z galaxies.

Evidence: `docs/e2e_cfm_lowz_metadata_audit_20261028.json`. Live deployment
binding: `/pscratch/sd/d/dkololgi/graphweb_desi/outputs/loa_shared_sixfit_20261026_v1/production_v1/NGC_seed17/BINDING.json`.
It names coarse checkpoints 013312, coupled fine checkpoints 026624 and expanded
checkpoints 009984, for seeds 17/29. The `BINDING.json` training budget is not
necessarily the saved step (notably coarse 013312 versus budget 26624).
This audit matched sidecar IDs to all 13 live geometry manifests; it has **not**
rehashed/reloaded the checkpoint binaries or certified current payload integrity.
The prepared compute canary performs those checks before staging points.

## Concrete missing dependencies

1. **Observation and response domain.** `e2e_coupled_observations.py` hardcodes
   z≥0.10 in both `observed_points` and `response_chunk`, including boundary
   distance. Changing JSON alone cannot extend it. Counts, random support,
   apodization, expected counts, radial selection and boundary distances must
   change consistently, in new products only. Missing support is not zero density.
2. **Selection/volume.** The inherited selection fit uses five historical
   training phases, fits at z=0.15–0.55, and tabulates a curve over z=0.10–0.60.
   Its inherited volume-Jacobian table starts at z=0.05. `np.interp` would clamp
   below the selection table, and `selection_volume_jacobian` rejects below its
   domain. Neither is a valid extension to 0.01. Build a label-free training-only
   selection/response model with the actual angular/effective-volume convention;
   test low-z volume discretization and observed/expected closure by phase and
   cap. Do not fit away development residuals. Freeze old curves/transforms for
   the reference comparison; quantify any interface change in the overlap.
3. **Geometry.** `e2e_coupled_geometry.py` hardcodes the four 0.15–0.55 shells,
   balanced interior/boundary counts, 108.256 Mpc/h owned-core width and periodic
   source non-overlap. Near the observer, old quotas/interior criteria may be
   infeasible. Make a separate random-support-selected low-z domain manifest;
   retain all exclusions and actual supported volumes. Do not silently relax
   gates, select by density/truth, or count overlapping tiles as independent.
   Test origin/LOS treatment, radial padding and cap boundaries explicitly.
4. **Reader interfaces.** Pair IDs are restricted to `s[0-3]` by
   `e2e_coupled_conditions.py`; whole-field selection expects the original 64
   training pairs/cap and eight evaluation pairs/cap. A versioned low-z loader
   and geometry contract are needed, not relabelled old pairs or global defaults.
5. **Targets and evaluation coordinates.** Reuse qualified native matter and
   sample new R7 fields/periodic tidal references with matching voxel windows.
   Keep native periodic truth distinct from finite-context reconstruction error.
   At galaxy locations, define whether the estimand is the tensor at observed
   redshift-space coordinates or the galaxy's own real-space location. Report
   that displacement effect explicitly; using true coordinates as conditions
   would leak information unavailable in DESI. Never revive HALO_INDEX/BOX_INDEX
   label validity cuts. Ordered-eigenvalue and own-position policies remain.
6. **Normalization.** New training must use a new training-only normalizer,
   channel contract and transforms covering the enlarged range. Preserve original
   transforms with historical weights; do not refit them in place. Any later
   historical replay uses its own original inputs and is reference-only.

The 2026-10-27 **20/3/2 data release does not supply these low-z dependencies**:
`configs/e2e_coupled_data_20phase_20261027.json` still specifies context 0.10–0.60,
science shells 0.15–0.55, and inherited selection. Its deterministic preparation
completion is not a newly trained checkpoint or low-z qualification. This task
requires new low-z-covered inputs before fresh training; the need for retraining
is settled by the user, while the appropriate geometry/conditioning remains to
be assessed during preparation.

## Historical prepared point/checkpoint canary — not the current prerequisite

- Contract: `configs/e2e_cfm_lowz_input_canary_20261028.json`.
- Runner: `workflows/sbi/prepare_cfm_lowz_inputs.py`.
- Seven bounded guard/geometry tests: `tests/test_cfm_lowz_input_contract.py`.
- Existing untracked `workflows/sbi/e2e_cfm_lowz_audit.py` remains untouched.

The runner checks every deployed model's sidecar and embedded checkpoint binding,
payload hashes, normalization and phase IDs. It inventories redshift counts from
the 13 original training catalogues plus development ph012/ph013. Existing
target-free successful-observation FITS files retain the full selected sample;
the ≥0.10 cut occurs later, so audit/reuse those before rebuilding from CFS.
It stages z=0.01–0.15 development points with exact TARGETID/sky/RSD parent parity:

- `observations/*_POINTS.npz`: observed RA/DEC/Z, cap, ID, observed Cartesian xyz only.
- `evaluation_only/*_POSITIONS.npz`: true Z_COSMO, own-position xyz and native-box
  coordinates, plus apparent magnitude for explicitly labelled diagnostics.

No model may read the evaluation-only directory. No new tidal targets, density
arrays, fitted selection or condition shards are claimed by these files.
`model_inputs_ready` remains false even after a successful point canary.
ph006 is retained as a later new-policy development check; ph018/ph019 stay sealed.

Under an approved CPU step in `cosmic_env`, with Python path contamination cleared:

```bash
python -m unittest discover -s tests -p test_cfm_lowz_input_contract.py
python -m workflows.sbi.prepare_cfm_lowz_inputs \
  --output /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_lowz_20261028_v1/input_canary
```

Output must not exist; partial runs are retained and must not be silently resumed.
The metadata-only mode is bounded JSON inspection; full mode requires a compute
node. Proposed first budget: one CPU node, 64 CPUs, desi/interactive, Scratch,
≤60 minutes, release afterward, no renewal/GPU/training. Approval was requested,
not assumed. Subsequent response/target preparation is sized from this canary;
it is not promised to fit within that first budget.

## Evaluation and plotting contract after low-z-covered retraining

Report 0.01–0.03, 0.03–0.05, 0.05–0.075, 0.075–0.10 separately, plus 0.10–0.15
as a bridge. Keep the four historical science shells and their outputs unchanged.
Preserve the old-range results as references. After fresh training, compare the
same overlap targets and separately assess input-context changes with explicitly
registered ablations. Changing both training and context is not a context-only
effect. Wide-context changes can alter higher-z predictions; new results never
replace historical ones in place.

Use preselected geometry and common targets/draw budgets for newly trained models.
Compare valid low-z conditions with matched prior/shuffled-condition controls;
register any input ablation against the new training distribution. Separate information gain from sharper but
miscalibrated distributions. Freeze development spatial selection/calibration/
evaluation groups before any tuning; both ph012/ph013 have prior evaluation
exposure. Reuse of development data does not create independent confirmation.

Required panels: selected counts and effective volume; expected-count closure;
support and boundary distance; standardized condition shifts; matched density/
tidal truth, draws, posterior mean and SD slices with fixed scales; shellwise
bias/RMSE and marginal coverage/PIT; derived class probabilities/reliability;
regional variance/covariance and joint tidal diagnostics; sampler and exterior-
context sensitivity. Include both caps, supported/boundary strata, true-position
versus observed-position comparisons and sample sizes. Keep phase/spatial
dependence in uncertainty estimates; do not bootstrap individual galaxies as
independent realizations. Reuse frozen historical thresholds where applicable;
register any new low-z decision thresholds before predictions are inspected.

For the low-mass DESI application, distinguish the **BGS_BRIGHT tracer sample**
from the galaxies at which the inferred field is queried. Do not impose a stellar-
mass cut on tracers by accident, or label apparent magnitude as stellar mass.
Check redshift/stellar-mass completeness in the actual DESI science sample and
mock-property availability before claiming validation for low-mass galaxies.

## Retraining decision

Retraining is **required**. Preliminary work establishes the correct selection,
geometry, conditioning, target and normalization contracts and measures resource
requirements. It does not apply old weights below their training range to decide
whether retraining is necessary. Accuracy, calibration and information gain are
post-training qualification criteria, not a reason to postpone range-correct
training. Existing regional/joint calibration failures remain unresolved.

All these mocks derive from the source epoch z=0.2. Observer redshift is a separate
coordinate; successful low-z mock evaluation cannot itself establish physical
evolution/galaxy-population fidelity at actual DESI z≈0.01. Treat that as a
separate simulation-to-data validity test, not a recurrence of the fixed label bug.

## Verification boundary

Live source and JSON metadata were inspected, and the geometric summary was
independently calculated from those JSON rows. `git diff --check` passed for
tracked changes. The new Python tests and canary could not start because of the
sandbox error; none is reported as passed. Graphify conda invocation failed at
the same startup boundary; direct targeted inspection was used transparently.
Graph refresh remains pending compute. No historical artifacts were overwritten.
