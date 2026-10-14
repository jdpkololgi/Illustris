# Matched Gaussian CFM verification checkpoint

The approved experiment completed its three fixed-budget neural fits, main
sampling panel, and initial five-size resource benchmark. Final verification
is incomplete. The subsequent direct-sweep timing and audit step exited with
code 1; the failing stage is not yet known because the approval service returned
HTTP 503 and prevented reading its log. Do not interpret this checkpoint as a
validated scientific closeout. No retraining or checkpoint selection is needed
to inspect the failure.

## Saved work

The output root is
`/global/u2/d/dkololgi/diagnostics/matched_gaussian_3d_20261009_v1`.

- `mean_covariance/COMPLETE.json` contains the 128 exact/learned endpoint
  interventions on the previous structured CFM checkpoint.
- `run/COMPLETE.json` records engineering completion of the main 3D comparison.
  Final checkpoints, losses, projected draws and evaluation receipts are saved.
- `audit_59596481.log` is the first file to inspect on resumption. The queued
  sequence was direct full-sweep timings, eight new focused tests, six existing
  exterior-CFM tests, then the independent numerical audit. Its exit code alone
  does not identify which stage failed.
- `actual_sweep_cost.json` and `VALIDATION.json` must be checked for existence,
  contents and hashes; neither is claimed complete here.

Five implementation tests passed before the main run. The later full audit is
not claimed to pass. The approximate four-standard-error screen falsely
rejected one exact-posterior control; retain that finding and use the disclosed
post-hoc Gaussian-null audit in the validation addendum. Do not silently
promote the original classification labels to calibrated decisions.

## Observations awaiting final independent reconciliation

For the previous structured fit, forward local updates have worst covariance
errors of 1.49 percent with exact mean mapping and learned innovation covariance,
5.80 percent with learned mean mapping and exact innovation covariance, and
6.01 percent with both learned. Exact/exact controls are at numerical precision.

The 3D fits use the inherited base-24, three-level coupled backbone with a
Gaussian conditioning interface: one shared whole/block fit at size 8, and
separate whole and block fits at size 16. Each uses 4,096 updates and 32,768
fresh fields. Block size is 8. After 16 block sweeps, the domain-mean estimates
are 6.06 and 5.19 exact posterior standard deviations below the two targets.
Whole-field fits also show substantial discrepancies. These are bounded
one-seed fits, not a convergence study or evidence that either method cannot
learn the target. No Abacus posterior qualification follows.

The direct-cost log already records whole-draw times of approximately 0.233,
0.252, 0.262, 0.288 and 0.458 seconds for grids of side 8, 16, 24, 32 and 48.
The corresponding measured block-sweep times through side 32 are approximately
0.233, 1.864, 6.296 and 14.947 seconds. The final side-48 sweep result remains
unread. Batch size is one, with 32 velocity evaluations per solve; a sweep is
not an independent or accuracy-qualified draw. Do not combine memory peaks
from this inference-only harness with the training-warmed microbenchmark.

## Resumption

Inspect the audit log, repair any demonstrated implementation error, rerun only
the affected verification, and reconcile the saved scientific and cost figures.
Keep the main run's bound sources and protocol unchanged. Then write the result
record and update SCIENCE_LOG.md and CONTEXT.md with verified conclusions.

GPU allocation 59596481 on nid200372 served this run. Its interactive shell was
asked to exit after the audit step terminated. The required graph refresh was
explicitly approved by the user but still timed out during Scratch-dependent
startup; its global registration was not refreshed. The normal environment
and shared installations were not modified.
