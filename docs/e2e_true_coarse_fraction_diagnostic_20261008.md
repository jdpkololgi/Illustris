# True-coarse interior-fraction diagnostic — 2026-10-08

## Frozen question and rule (before scoring)

Scope: **one parent seed, two reused phases, diagnostic only**. No training,
new model, selection, confirmation-phase access, or batch submission. Parent
fine checkpoint is seed17/step26624 (expected SHA256 prefix `88b2c7d1`).
The executable contract is
`configs/e2e_true_coarse_fraction_diagnostic_20261008_v1.json`.

Retain Stage C's galaxy weights, threshold0.2, interior/boundary assignments,
two cores per pair, first three class fractions per core, physical R7 truth,
and corrected density-only primary/original sensitivity operators. With
32 draws, Q=mean(ee^T)-mean(S)/32. Report phasewise trace ratios, full bias
vectors, centered ratios, classwise moments, and mean predictive variance.
Bootstrap 4096 whole-draw resamples with seed6100617, shared across all
cases/arms/phases; report 5th–95th percentiles as conditional-MC ranges,
not population or phase-resampling confidence intervals.

Apply the requested interpretation only to a matched full-cap A/B comparison:

1. B<=1.5 in both phases and upper(B)<lower(A) in both phases:
   **coarse-dominated**; subsequent investigation targets coarse uncertainty.
2. B>=2 in at least one phase with overlapping A/B ranges in that same phase:
   **coarse factor not implicated by this diagnostic**; subsequent investigation
   targets the fine conditional or observation/truth setup.
3. Otherwise: **inconclusive**. Never select a preferred mechanism afterward.

These are diagnostic routing labels, not proof that a component has no effect.
True conditioning does not remove finite-panel uncertainty or possible tidal
operator/exterior/truth mismatch. The requested difference
trace(mean S_A)-trace(mean S_B) is descriptive, not a causal decomposition.

## Inventory and matching constraint

The augmentation oracle receipts contain only power/width, but their upstream
parent fields **were saved** under
`coherent_exterior_20261002_v1/verification/sampler_factor_20261005_v1/results/`
in each pair's `*_DRAWS.h5`, group `oracle_coarse_local_fine` (32 draws).
They use the same addressed fine noise but an isolated production-window
integration and historical local-plus-wide operator. Arm A's registered Stage C
scores instead use shared full-cap integration and cap tides, including the
density-only correction. These saved local fields are **not an eligible B**
for direct application of the rule to Stage C A. Full-cap generation or an
explicitly separate matched-local diagnostic needs a resolved user choice.
The user subsequently approved the matched sequence: saved-field power,
class-fraction truth closure, then matched full-cap true-coarse sampling.
Report the original numerical rule alongside its appropriately qualified
meaning: a coarse-conditioning-pathway contribution versus a defect persisting
with true coarse. Neither branch certifies a unique causal component.

Independent saved-field work proceeds: reuse A; compute coarse-only regional
density moments; compute sampled-coarse residual power for parent, control,
and all five augmented strengths. Preserve the original decision receipt;
attach a separately hash-bound addendum. Primary spectral intervals are the
user's [.04,.08), [.08,.16); also retain the previous (lo,hi] convention for
parity and document any edge-mode differences. Do not infer missing spectra
from raw latent U, oracle power, or summary-only files.

## Results

Pending execution. No interpretation branch has been selected.
