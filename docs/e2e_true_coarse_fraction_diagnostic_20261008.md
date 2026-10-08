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

## Results in progress

Saved-field analysis is complete. Parent Stage C moments and their conditional-MC
ranges reconcile to the original scores. Independent power verification passes
154 comparisons with the historical spectrum products. The original decision
receipt remains unchanged. No A/B interpretation branch has been selected:
matched B generation and the full truth-closure panel are still running.
The byte-verified separate attachment is
`cfm_conditioning_augmentation_20261006_v3/decision/SAMPLED_COARSE_POWER_ADDENDUM_20261008.json`
under the same Scratch E2E root.

For the corrected interior parent, total ratios are 2.6958 and 2.1147;
removing the common six-coordinate mean-error vector leaves 2.3857 and
2.0420. A global mean-offset correction therefore does not account for most
of the mismatch. The centered statistic still includes region-dependent mean
errors; it does not establish that posterior variance alone is too small.

Physical sampled-coarse fine-residual power divided by truth power:

| Setting | ph012 [.04,.08) | ph012 [.08,.16) | ph013 [.04,.08) | ph013 [.08,.16) |
| --- | ---: | ---: | ---: | ---: |
| Parent | 0.8507 | 0.8004 | 0.8801 | 0.8463 |
| Control | 0.8508 | 0.7986 | 0.8799 | 0.8449 |
| Augmented, s=0 | 0.9245 | 0.7958 | 0.9555 | 0.8410 |
| Augmented, s=.025 | 0.8958 | 0.7930 | 0.9258 | 0.8382 |
| Augmented, s=.05 | 0.8679 | 0.7905 | 0.8968 | 0.8358 |
| Augmented, s=.1 | 0.8143 | 0.7868 | 0.8414 | 0.8322 |
| Augmented, s=.2 | 0.7169 | 0.7842 | 0.7409 | 0.8299 |

Thus the uncorrupted augmented model improves the lower band, but no setting
removes the higher-band deficit. The premise that augmentation fixed both
sampled-coarse power deficits while leaving fractions unchanged is unsupported.
These are equal-window aggregate ratios, not independent-mode measurements.

The secondary interior coarse-only regional-density joint ratios are 1.0208
[0.9106,1.3025] for ph012 and 1.0711 [0.8775,1.4697] for ph013; centered ratios
are 0.7208 and 1.0703. Brackets are conditional-MC sensitivity ranges. These
volume-weighted block-density summaries do not test the anisotropic/spatial
coarse information relevant to galaxy-weighted class fractions.

Both phases' new global true-coarse canvases match all 32 registered saved
wide-window truth crops exactly. First-stage truth closure and retained B
generation use interactive allocations only. Scripts, inputs and outputs are
hash-bound under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/true_coarse_fraction_diagnostic_20261008_v1`.
