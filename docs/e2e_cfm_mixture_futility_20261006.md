# Target-specific mixture feasibility screen

## Question and scope, frozen before target-summary extraction

Does model diversity produce a potentially useful change in the failing
regional quantities, large enough to justify further full-cap generation?
This is a resource-continuation screen, not a calibration pass or a replacement
for the registered full-panel investment gate.

Use the first eight completed, matched draw addresses (0–7) in SGC in BOTH
ph012/ph013. SGC is the pre-existing smaller-cap technical-pilot choice and has
completed batches in both phases; no summary values select the cap or cases.
Retain every SGC pair, all shells and both support strata. Primary summaries:
paired four tidal gaps over all SGC pairs, and paired six galaxy-weighted class
fractions over SGC interior pairs, original operator, threshold0.2. Boundary
fractions and interior/boundary gaps are secondary. NGC and the corrected
operator are outside this cheap screen: no full-panel qualification follows.
No new draws or training. Eight draws is a small Monte Carlo diagnostic.

Use Stage B's exact cell counts/selection and physical truth, not masses as
proxies. Extract original-operator eigenvalues from completed full-cap products.
Verify seed17 float32-product class labels against the saved float64 Stage B
labels at threshold0.2 and regional means against Stage A at2e-7. Any changed
class label stops the analysis; do not adjust thresholds or drop cells.

## Measures and safeguards

For each paired vector report component means, covariances, mixture mean,
within-component average covariance, and quarter outer product of mean
differences. Also report the paired-Monte-Carlo-noise-corrected between-model
term: subtract Cov(Y29-Y17)/(4M), retaining negative eigenvalues as sampling
noise, not projecting to positive. Compare total covariance and mean changes
in a fixed regularized seed17 32-draw covariance metric (20% diagonal shrinkage,
relative ridge1e-8). Keep residual-direction projections and full matrices.

Compute fair energy scores for both components and their equal-weight mixture
directly, without random component assignment noise. Exclude same-address pairs
from every pair-distance expectation, including the cross-component term:
matched noise correlates same-address draws. No claim of iid regions.

Use1024 paired draw-address bootstraps, fixed seed610061, with one common
resampling across all selected regions in a phase. These are conditional
Monte Carlo sensitivity bands, NOT population or cosmological confidence
intervals. Reference32-draw covariance is held fixed. Report raw and corrected
moments, point estimates, sensitivity bands, and half-bank stability.
Do not equate Stage C error/variance ratios with required variance inflation.

## Resource decision rule

Predeclared engineering materiality thresholds: mixture mean displacement
0.10 baseline SD (root mean squared whitened coordinates), total covariance
change0.10 (mean casewise spectral norm in the baseline metric), between-model
covariance0.05 in that metric, and absolute relative energy change0.02.
These are resource-screen thresholds, not derived guarantees about20% coverage
improvement. They deliberately do not infer futility from covariance alone.

If all four primary phase/summary groups have their95th bootstrap percentile
below ALL thresholds, label `futile_on_screened_support`: do not request another
generation segment. This means low measured effect on this declared support,
not mathematical impossibility elsewhere. Require at least8paired draws,
nondegenerate baseline metrics and positive score denominators; otherwise
`insufficient_evidence`. If any group's5th percentile exceeds a materiality
threshold, report `material_effect_detected`, which rejects the negligible-
effect premise but does NOT establish benefit or license automatic continuation.
Other outcomes are `insufficient_evidence`. In every case hold further resource
requests pending interpretation of both primary targets and direction of scores.

The current generation allocation is not cancelled or modified by this screen.
No automatic second segment or retry is installed. Preserve all outputs.

## Validation

Test identical components, analytic covariance identity, finite-draw debiasing,
mean-shift and covariance-change controls, fair-mixture energy including common-
noise exclusion, phase/geometry guards, and the three-way resource decision.
Bind source, protocol, all receipts and selected payload hashes before scoring;
write separate immutable configuration, extracted arrays, results and review.
