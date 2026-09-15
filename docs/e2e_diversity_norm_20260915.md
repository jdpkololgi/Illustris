# Diversity and global normalization isolation

Registered 2026-09-15 before fitting; user approves CPU/GPU compute. Full E2E
and held-out phases remain paused. Allocation 58373492: two GPUs, two hours.

## Matched matrix

Near-identity FiLM U-Net, unchanged internal GroupNorm, tau .05, AdamW,
learning rate 1e-4, clipping 1, 3,072 updates, original continuous multi-noise
exposure including the pure-noise endpoint. Fresh fits at 3, 6, 9, 15 regions
times current / strict-global-train normalization, two training seeds: 16 fits,
49,152 updates. Evaluate at 0, 1,536, 3,072; preserve 32 checkpoints. Noise
tensors and physical noise/time exposure are paired across cells. Field/bin
scheduling avoids phase or field aliasing with the six noise bins. Presentations
per field decrease with diversity at this fixed update budget.

Three training phases (000/002/003), not independent new cosmologies. Metadata-only
selection retains the original three NGC fitted anchors and expands to 15 mutually
non-overlapping stored source footprints. Twelve SGC transfer anchors (four shells
per phase) have no stored source-footprint overlap with any training anchor.
Transfer regions can overlap each other; coarse conditioning domains can overlap.
This uses the existing source-coordinate mapping, not an independent audit of it.
Increasing diversity also changes environment/redshift/selection coverage.

## Numerical scale control

Current normalization is already global, fitted on all 96 training-role cutouts,
including these development transfer regions. Strict normalization uses only the
largest 15-region fitting partition and is FIXED across all four diversity levels.
The tiny strict fit therefore uses moments from additional eligible training
fields, but no transfer moments. Identity condition channels and nonlinear
transforms remain unchanged. Internal activation GroupNorm is NOT changed.

New module's endpoint-safe affine VP chart preserves the exact physical corrupted
field and original physical velocity objective, including both endpoints. Changing
target units must not silently change the physical denoising problem or loss.
Frozen original 3,072-update weights are evaluated with current, strict, tiny-only,
strict-target-only and strict-condition-only scalers. Direct swaps are sensitivity
interventions, not equivalent predictors or proof that new scaling is correct.
An inverse/forward compensated chart must preserve the frozen physical function.
Replica-zero tiny/current fit must replay the previous checkpoint to numerical
tolerance, a separate implementation and control-fidelity check.

## Outcomes and interpretation

Zero injected noise at nominal ratios 0, .001, .005, .01, .025, .05, .2;
two paired noise realizations at .05 and .2 on the fixed 27-region panel.
Clean VP input is a*y, corresponding to a clean input in the VE denoiser chart.
Exactly zero identity is algebraically forced and is NOT learned evidence.
At positive nominal noise, clean distortion is a diagnostic rather than a demand
that the optimal finite-noise conditional estimator be exactly identity.

Report physical clean RMS/bias/max error and spectral gains/error separately
from injected-noise leakage and physical noisy error. Preserve four spectral bands
and compare against the frozen 384 parent on the same panel/noise realizations.
Report common fitted-three, exposed fitted subset, unused eligible training fields,
and fixed transfer-twelve separately; do not confuse changing fitted-set medians
with a matched learning curve. Show both optimization checkpoints and seed spread.

Explore clean-distortion associations with field mean/std, skew/tails, density
distribution, redshift, support/selection and counts, including within-phase
centering. These are descriptive, correlated, small-sample comparisons, not causal
tests or independent-field confidence intervals. A decreasing transfer curve with
stable fitted performance supports coverage; a paired normalization benefit
supports scale sensitivity. Neither guarantees an asymptote at fifteen regions or
convergence at 3,072 updates. Mixed outcomes remain possible. No production or
posterior-calibration promotion follows from these capability diagnostics.

Implementation: `workflows/sbi/e2e_diversity_norm.py`; frozen settings:
`configs/e2e_diversity_norm_20260915.json`; tiny algebra/scheduling tests:
`tests/test_e2e_diversity_norm.py`. Full-size forward/backward smoke precedes fitting.
Output root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/diversity_norm_20260915_58373492`.

## Physical equivalence of normalization charts

Write the original normalized target as y, corrupted state as x=a*y+b*epsilon,
and the new target as y'=(y-m)/s. Define q=sqrt(s²*a²+b²). The new network sees
x'=(x-a*m)/q with a'=s*a/q and b'=b/q, so exactly the same physical noise
realization is used. Its time is t'=2*atan2(b,s*a)/pi. Convert its velocity back:

`v = a*b*(1-s²)*x/q² - b*m/q² + (s/q)*v'`.

This avoids division by a or b and remains finite at clean and pure-noise
endpoints. The original physical velocity loss is evaluated AFTER conversion.
Changing numerical units therefore does not change the corrupted physical field
or objective. It does change what the raw network sees and how its parameters
optimize, which is the intended intervention. Applying a matching inverse chart
inside a frozen model instead must preserve its complete physical function.

The measured pooled strict target mean/scale in original normalized units are
0.0011421151 and 0.9977481911. Thus the fine-target scale changes by only0.225%.
Conditioning changes are larger and remain a separately tested component.
