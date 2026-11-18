# Low-z CFM retraining: selection/response implementation

**Subsequent execution:** CPU60477149 completed all23 native statistics,
full+20OOF selections and23 full-response pairs. This remains an unqualified
preparation release: low-z volume sensitivity exceeds the1% numerical screen,
and shell auto-logout interrupted the independent payload audit. See
`lowz_response_60477149.md/json` for current evidence; the implementation-stage
status below is historical. No normalizer/model fit or target rebuild occurred.

## Decision and completed-work audit

Fresh low-z-covered training is required. Science scope is 0.01 <= z < 0.55,
with diagnostic edges 0.01/0.03/0.05/0.075/0.10/0.15/0.25/0.35/0.45/0.55.
Keep available positive-z context below 0.01, context ceiling 0.60 and the
0.585–0.595 sentinel exclusion. R7, source snapshot z=0.2, and shared 20/3/2
phase roles are unchanged. Existing weights/results remain references only.

The October27 preparation completed old-range inputs, not this extension.
The separate P12-A ph001/ph006 canary completed the low-z label and angular-map
audit (see `p12a_lowz_canary_60475000.md/json`), not selection, response grids,
domain/fold allocation or new transforms. Its all-population labels are not CFM
native-density targets. The old CFM point/checkpoint canary is not a prerequisite
for retraining preparation. The exact new output root did not exist at audit.

This turn implements code and a versioned contract; it does **not** claim a new
data release. Full inputs, geometry and matched R7 truth are still unbuilt.

## Implemented interface

Entry point: `workflows.sbi.cfm_lowz_selection_response`.
Contract: `configs/cfm_lowz_selection_response_20261029.json`.
Output root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_lowz_selection_20261029_v1`.
No historical default, checkpoint, transform, selection or payload is overwritten.

- `statistics --phase phNNN`: verify target-free observed and angular-map
  payloads against the current 20-phase source receipts; build a random-support-
  defined Cartesian box including the observer; accumulate CIC counts and matched
  effective volume by cap and radial bin. Development statistics are diagnostics,
  not fit inputs. Sealed phases fail before payload access.
- `statistics --phase phNNN --refinement 2`: independent half-cell statistics,
  with the same physical apodization width. This is potentially much more
  expensive than the native pass and needs an explicitly sized compute budget.
- `fit`: require all 20 training statistics; no development/test input accepted.
  `fit --omitted-phase phNNN` reads only the other 19 training statistics. This
  provides a leakage-safe selection component, not a complete P12 OOF contract.
- `response --phase phNNN --selection PATH`: rebuild count/response HDF5 grids
  under the bound fitted selection. Save explicit LOS and selection-valid masks,
  expected counts, angular/random support, apodization and boundary distances.
  The preparation receipt deliberately has `model_inputs_ready=false`; old
  loaders must not consume it as a qualified old-schema release.

All products bind the plan, phase policy, source configuration, coordinate table,
implementation dependencies and input receipts/payloads. Output directories must
be new; no overwrite, silent resume or automatic retry. Inherited raw-random
provenance stays inherited: this does not rescan all CFS random files.

## Estimator and units

For each cap and fixed delta-z=0.005 bin, pool training-phase **supported CIC
count weight**, divided by the corresponding sum of
`angular_response * apodization * cell_volume`. Both numerator and denominator
use the same voxel centres. Report raw catalogue counts and CIC weight outside
the response separately; do not hide boundary losses.

The density is piecewise constant within its registered bin and is fitted
directly in `(Mpc/h)^-3` using the pinned DESI/Abacus coordinate table. Do not apply
the old Planck18-Mpc selection Jacobian or an additional h conversion. There is
no interpolation-clamped extension of an old curve. Bins need positive effective
volume and at least 20 pooled count weight. Missing science bins fail the fit;
missing sub-0.01 context bins remain explicitly invalid. The origin has no LOS
or random support. A zero stored expectation with an invalid mask is **missing
information**, not evidence for an empty region.

This is a transparent **preparation estimator**, not a selected production
smoothing prescription. The pooled expected-count closure is true by construction
and is not evidence of accuracy. Before training, measure phase/cap and low-z
closure, native/half-cell volume convergence, and sensitivity to radial binning
and a training-only smooth selection alternative. Do not tune to development
residuals and then present those residuals as independent validation.

## Remaining training prerequisites

1. Execute source/response canary and predeclare numerical acceptance criteria
   before inspecting scientific outcomes; size the full 23-phase build from it.
2. Qualify the selection and response with the above independent diagnostics.
3. Register observer-crossing training-domain sampling, supported volumes,
   overlap accounting and disjoint development selection/calibration/evaluation
   spatial groups. A Cartesian support box is not that domain manifest.
4. Generate matched native R7 density/tidal targets with identical voxel windows;
   keep finite-context error distinct from periodic truth. Galaxy real-space
   positions/targets are evaluation-only, never observed condition channels.
5. Fit new training-only normalizers and build/validate new condition/target
   loaders; P12-A also needs its own grid/fold/transform and OOF integration.
6. Fresh training followed by separate low-z calibration and information-gain
   evaluation. Historical replay is optional reference work, not a retraining gate.

Lower-redshift coverage is not stellar-mass completeness. The ph001/ph006 canary
has magnitudes, not audited stellar masses. A source snapshot at z=0.2 does not
establish actual z=0.01 population/evolution fidelity.

## Redshift floor

Use 0.01 as the *coverage* floor, not a claim that peculiar velocities vanish.
At low z, illustrative v=300 km/s gives v/(cz) about 10%, 3.3% and 2% at
z=0.01, 0.03 and 0.05. The comoving radial displacement is approximately
`(1+z) v/H(z)`, about 3 Mpc/h at low z, appreciable compared with R7. Random
virial motions can be larger. The observation/target estimand and RSD treatment,
not a universal redshift cut, control environment-inference fidelity. Inspect
0.01–0.03 separately and use raised-floor sensitivity checks after retraining.
Relevant primary sources: [Kenworthy et al.](https://arxiv.org/abs/2204.10866),
[Davis & Scrimgeour](https://arxiv.org/abs/1405.0105).

## Execution boundary

Small synthetic tests run in isolated `cosmic_env`; no real FITS catalogue,
scientific target or Slurm allocation is opened by the tests. Real statistics
and response stages enforce compute-node execution. The already-completed
P12 canary approval is consumed and is not reused. A first new native-resolution
ph001/ph006 preparation smoke may be proposed for one CPU node, 64-CPU step,
desi/interactive, Scratch, at most 60 minutes, no retries and release on exit.
This is a request, not authorization, and not a promise that the full 23-phase
build or factor-two refinement fits that budget.

Verification: 19 focused tests pass (seven earlier guard/geometry tests plus
twelve new selection/response tests), including a 27-cell FITS/CIC/HDF5 writer
test. CLI help and whitespace checks pass. This is not payload or independent
physical validation. Repository-wide AST graph refresh is deferred to approved
compute under the NERSC graphify workflow; the existing graph was queried for
orientation, not claimed current after this edit.
