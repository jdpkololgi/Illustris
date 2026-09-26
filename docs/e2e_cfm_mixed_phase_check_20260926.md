# Mixed-checkpoint phase sensitivity: quick exposed-data check

2026-09-26. Exploratory, no training, allocation, or sealed-phase access.
Candidate: coarse13312/fine26624 EMA. Inputs: 32 posterior draws per
anchor, 16 anchor pairs per phase, seeds17/29, ph012–015 only.

Source: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_mixed_20260926_v1/COMPLETE.json`.
SHA256: `749a0d7b4a593bc5b2963b68b0fd07955f32f517bb1a9f6fb51f11b3e7c9caa3`.
All392 input hashes verified;384 rows across three arms. This check uses128
mixed rows. Equal numbers of regions/anchors/seeds permit equal-weight averaging.

|Phase|Core coverage|Block coverage|Fine-region coverage|Tidal coverage|Eigengap coverage|Fine RMSE/spread|Fine bias/spread|
|---|---:|---:|---:|---:|---:|---:|---:|
|012|84.38%|89.06%|79.69%|86.80%|86.74%|1.142|-0.081|
|013|92.19%|87.89%|96.09%|87.59%|87.32%|0.663|0.057|
|014|89.06%|91.41%|85.94%|87.77%|87.58%|0.944|0.026|
|015|89.06%|86.72%|83.98%|87.06%|86.07%|1.045|0.088|

Finite-draw attainable nominal90 coverage is29/33=87.88%. Tidal/eigengap
entries are marginal coverage, not joint-field or class-probability calibration.

Method: read per-case `scores`; average group `coverage['0.9']` within phase.
For fine regions use `region_mean`, `region_truth`, `region_std` indices10:18.
Let error=mean-truth and spread=sqrt(mean(std**2)); report
sqrt(mean(error**2))/spread and mean(error)/spread. Centered-RMSE/spread is
1.139,0.661,0.944,1.041 respectively. These estimates pool regional entries
descriptively; small average bias does not rule out conditional/local biases.
Posterior means estimated from32 draws add sampling error; no correction is
applied here. RMSE/spread close to1 is a diagnostic, not a sufficient calibration
criterion.

Seed17/29 fine-region coverage: ph01278.13/81.25%, ph01396.09/96.09%,
ph01488.28/83.59%, ph01580.47/87.50%. Removing each anchor pair in turn
(both seeds together) gives fine-coverage ranges: ph01278.33–81.67%,
ph01395.83–98.33%, ph01485.00–88.33%, ph01582.92–89.58%.
These are sensitivity ranges, NOT confidence intervals. The012/013 contrast
is not explained by one pair or one training seed.

Interpretation: mixed model retains encouraging marginal tidal calibration,
but fine-region coverage heterogeneity cannot yet be declared negligible.
Ph012 errors exceed predicted spread; ph013 errors are appreciably smaller
than spread. A single global widening is not supported. Four phases and
spatially correlated/possibly overlapping anchors cannot establish population
significance or distinguish cosmic variance from observation-regime shift.
No voxel/region-level independent-binomial p-values are justified here.

Keep this as the provisional development candidate. Before changing the model,
compare errors/spread across matched observational regimes within these exposed
phases, retaining phase/anchor dependence. Sealed ph016–019 remain untouched;
their use requires a frozen evaluation decision, not iterative tuning.
