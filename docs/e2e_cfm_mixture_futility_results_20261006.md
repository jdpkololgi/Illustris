# Mixture feasibility screen: little added model-diversity spread, mixed utility

## Scientific answer and resource decision

Hold further full-cap generation requests. The cheap matched-target screen
does not show a consistently useful repair of the requested uncertainty defects.
It is NOT a completed32-draw/full-panel investment-gate failure, and it does
not demonstrate calibration or formally exclude benefits in NGC.

The predeclared **negligible-effect futility rule did not fire**: the label is
`material_effect_detected`, because mean/covariance-shape changes exceed the
engineering thresholds. That means the models are not interchangeable, not
that their mixture helps. The added between-model variance is nevertheless
small, supporting the user's concern about this particular spread mechanism.
The observed score direction—especially ph013 tidal gaps—argues against
automatically purchasing the remaining campaign. The protocol puts all three
screen outcomes on resource hold pending scientific interpretation.

The current GPU allocation59422691 is neither cancelled nor modified by this
check. No second generation allocation or automatic resubmission is created.
If a tighter conclusion is needed, use already-produced later batches under
an explicit follow-up analysis; this result does not authorize more generation.

## Matched evidence

The first8draw addresses of both models, original full-cap operator, ph012/ph013
SGC only. Every8SGC pair per phase is retained; interior fraction statistics
use4pairs per phase. Stage B galaxy weights and unchanged physical full-box R7
truth are preserved. The first-three-class simplex coordinates of both cores
form the6-vector; tidal gaps form the4-vector. Threshold0.2. No new draws,
training, corrected-operator evaluation, NGC scoring, or untouched-phase access.

Below, variance ratios use the trace (sum of coordinate variances) of the SAME
eight-draw seed17 covariance on the SAME pairs as denominator. They are not
generalized covariance eigenvalues, marginal interval-width changes, or a
universal variance budget. Signs in the score columns describe improvement
(positive) or deterioration (negative) in fair energy, lower being better.

| Primary target | Phase | Added between-model variance | Total mixture variance change | Mixture energy improvement | Seed29 energy improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| Tidal gaps, all SGC pairs |012|1.58%|−3.00%|+0.29%|−0.20%|
| Tidal gaps, all SGC pairs |013|1.21%|+0.58%|−3.41%|−7.69%|
| Galaxy fractions, SGC interiors |012|0.66%|+7.46%|+0.93%|+1.39%|
| Galaxy fractions, SGC interiors |013|3.79%|−4.41%|−11.59%|−24.58%|

Mixing can narrow total covariance when seed29's within-component covariance is
smaller: between-model variance is only one part of the exact decomposition.
Paired-mean-noise subtraction reduces the between-model trace estimates to
1.46/1.14% for gaps and0.09/1.87% for interior fractions. These adjusted estimates
are noisy eight-draw moments, not precise population estimates.

The small aggregate additions coexist with measurable direction-dependent
changes. In the fixed32-draw regularized baseline metric, mixture mean shifts
are0.129/0.117SD per coordinate for gaps and0.133/0.158 for interior fractions.
Mean casewise spectral norms of covariance changes are0.183/0.169 and
0.397/0.452, respectively; these absolute directional magnitudes must NOT be
reported as18–45%uniform variance inflation.

## What is and is not resolved

The ph013 gap energy deterioration persists in both4draw halves (2.96/3.83%)
and its paired-bootstrap5th–95th percentile improvement band is
−3.87 to−0.83%. That is a conditional Monte Carlo sensitivity statement on
these fixed regions, not a cosmological confidence interval.

The ph013 fraction result is much less stable: one half improves5.20%, the
other worsens21.77%. Its bootstrap improvement band is−19.06 to+4.33%.
Do not call the point estimate of11.59%deterioration a definitive failure.
The ph012 gap/fraction improvement bands also include zero.

Eight draws do not support re-scoring the original32-draw coverage gate. No
coverage-repair claim is made. The evidence does distinguish a small added-
diversity spread term from a proven no-effect result, and from a useful repair.
It cannot establish that a prescribed variance increase would fix Stage C's
error/variance mismatch, nor validate boundary uncertainty.

## Verification and provenance

Six focused tests pass: identical components, covariance identity, paired
finite-draw correction, mean/variance controls, independent direct fair-energy
arithmetic, and three-way resource decisions. All baseline class labels exactly
match saved Stage B; gap summaries match Stage A within2e-7. All selected
receipts, HDF5 payloads, compact inputs and actual script/protocol sources are
hash-bound. Eight-draw arrays are saved before effect estimates are computed.
An independent review using scalar all-pairs distances and NumPy covariance
passes8,440 arithmetic checks. Analysis22.11seconds on CPU59426886,
nid200036; no additional GPU work.

Artifacts under Scratch
`e2e_field_v3/cfm_model_mixture_20261006_v2/futility_v1`:
`CONFIG.json`, `PANEL.npz`, `MOMENTS.npz`, `COMPLETE.json`, `REVIEW.json`,
`analysis.log`, `review.log`. Config SHA256
`6054f8cd18230d106b79eed48a26dd3d2d0d29893b123b8d73dda6ae7cfbbb50`.
Registered screen: `docs/e2e_cfm_mixture_futility_20261006.md`.
Entrypoints: `workflows/sbi/e2e_cfm_mixture_futility.py` and
`workflows/sbi/e2e_cfm_mixture_futility_review.py`.
