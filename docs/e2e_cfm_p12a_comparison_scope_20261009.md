# Coupled CFM and P12-A per-galaxy comparison scope

Status: scope note, 9 October 2026. No evaluation, data access, retraining or
resource allocation is authorized. This question is separate from the
[whole-field campaign](e2e_whole_field_comparison_plan_20261009.md).

## Current conclusion

P12-A has stronger blind-mock validation evidence. It is not established that
P12-A is better on every per-galaxy quantity, or that coupled CFM matches it:
there is no established like-for-like head-to-head comparison.

P12-A passed its registered blind mock gates. The CFM Stage B development
assessment finds informative galaxy class probabilities but remaining reliability
residuals. Stage B samples the field in the containing cell at observed galaxy
coordinates, whereas P12-A uses the host-halo environment target. Coordinate,
smoothing and representation differences make those scores non-interchangeable.
Real-data property trends do not substitute for matched mock-truth validation.

## Contract for a future comparison

Before scoring, register common galaxy IDs, selection, smoothing, coordinates,
truth convention, class threshold and phase-exposure accounting. If both frozen
models cannot predict the same physical estimand, report separate task performance
instead of silently relabeling one target or asserting dominance.

Compare eigenvalue error/proper scores and coverage plus class Brier/reliability,
retaining redshift/support strata, phase-level uncertainty and finite-posterior-
sample corrections. Freeze these choices before accessing a new confirmation
panel. Reuse compatible frozen products first; any new evaluation requires its
own bounded execution scope. P12-A's per-galaxy success does not establish a
coherent spatial joint posterior, and field coherence alone does not establish
per-galaxy calibration.

Evidence: [SCIENCE_LOG.md](../SCIENCE_LOG.md),
[Stage B protocol](e2e_stage_b_execution_20261006.md),
[Tier 1 findings](e2e_cfm_tier1_and_levers_20261006.md), and
[calibration investigation](e2e_cfm_calibration_investigation_plan_20261006.md).
