# Learned reference: scientific conclusions, 2026-09-19

## Determination

The bounded reference is complete. It does **not** qualify a reproducibly accurate
learned Gaussian field posterior, so neither the nonlinear lognormal/Poisson rung
nor the large coupled Abacus campaign is promoted. This is a finite-budget result
for a small existing-family U-Net, not a proof against VDM, CFM or amortisation.

Twelve fits (VDM/CFM, two seeds, amortised plus two fixed-observation fits per
seed/objective), checkpoints1024/4096 and96ensembles of512draws are archived in
[the complete numerical report](evidence/e2e_conditional_reference_20260919/README.md).
The [registered protocol](e2e_conditional_reference_v1.md) was frozen before fitting.
No model, loss or acceptance threshold was changed after inspecting results.

Slurm58594516 completed with exit0:0 in18m33s; the GPU step took18m02s. Main
runner1067.17s, peak allocated tensor memory123,553,280bytes; total run tree220MiB
(main outputs0.213GiB). One shared A100, no other allocation, no extension. The
tmux-backed shell exited and released the allocation. Ten focused tests pass,
including independent Gaussian conditioning, sampler moments and optimiser replay.

## What the models learned

The following are averages across matched observations0/1 and both training seeds
at4096updates,256NFE. Mean RMS is normalized by the reference posterior RMS;
covariance error is relative Frobenius error of the fixed16-dimensional spatial
probe covariance. These are different quantities, not percentages of voxels failed.

| Model | Training regime | Mean RMS error | Covariance error |
|---|---|---:|---:|
| VDM | Amortised | 0.372 | 0.374 |
| VDM | Fixed observation | 0.153 | 0.402 |
| CFM | Amortised | 0.278 | 0.253 |
| CFM | Fixed observation | 0.094 | 0.248 |

The registered mean tolerance is0.1; covariance tolerance includes the finite-
ensemble reference99th percentile, approximately0.176-0.184 for these templates.
No regime passes across the two seeds and matched observations. One fixed CFM
cell passes at128NFE and narrowly crosses its covariance threshold at256NFE
(0.18335 versus0.18404; threshold0.183995). This boundary sensitivity is not
evidence for deterioration with more integration; the whole configuration has
not passed reproducibly regardless of that cell.

**Amortisation adds a mean-learning burden in this test.** Training directly on
one posterior substantially improves its mean. However, **it does not remove the
covariance error**: fixed VDM is not better than amortised VDM on the averaged
probe-covariance diagnostic; fixed/amortised CFM are very similar there. Hence
amortisation is not the sole demonstrated cause of the joint-distribution failure.
The fixed fits are an idealized diagnostic with exact posterior training draws,
not a practical DESI inference method or a perfectly equal-information control.

The highest Fourier shell of the fixed-CFM *posterior residual variance* is
1.51-1.72times its exact value despite average voxel variance ratios close to1.
This is uncertainty power about each ensemble mean, not total reconstructed-field
power. The small8^3 shell has few modes and is not a DESI physical-k tolerance.
Nevertheless the consistent distortion reinforces the need for scale-resolved
uncertainty checks rather than aggregate variance alone.

## What was isolated

1. **Numerical error is real for the current VDM sampler.** With an exact
   denoiser its worst-template covariance error is9.71% at128NFE,5.020% at256,
   2.553% at512 and1.288% at1024. The registered neural128/256 results remain
   numerically confounded under the5% gate. Their much larger mean errors are not
   explained by the exact sampler's ~4e-6 normalized mean error at256NFE. Do not
   subtract full-covariance numerical error from projected learned error: the
   diagnostics are not the same norm on the same object.
2. **CFM sampling is not the leading issue on this reference.** Exact Heun
   covariance error is at most0.0337% at128NFE and0.0083% at256. Learned results
   change negligibly when doubling NFE. Its residual errors survive this control.
3. **The metrics detect dependence failures.** All four exact-reference ensembles
   pass. Erasing cross-voxel covariance while retaining correct voxel marginal
   variances yields projected errors0.807-0.864 and octant coverage0.667-0.708.
   All four deliberately30%-underdispersed controls also fail. This validates
   these toy diagnostic controls, not the old Abacus10% effect-size requirement.
4. **Training has not demonstrated an asymptote.** From1024to4096updates, fixed
   CFM covariance error falls0.480to0.248; amortised CFM0.360to0.253; fixed VDM
   0.581to0.402; amortised VDM0.594to0.374 (256NFE). Optimization, target variance,
   time parameterization and representation remain entangled. Do not label the
   remaining error irreducible capacity failure or claim convergence.

## Objective and the amortisation critique

The exploratory estimand is a **joint** posterior over smoothed evolved matter
fields given observed galaxies and survey response, under stated simulation,
galaxy-population and selection assumptions. Desired draws must carry plausible
coherent alternatives and reliable regional-mass/tidal uncertainty, not merely
accurate per-voxel marginals. P12 remains a separate production per-galaxy product.
The present nominal Abacus programme does not marginalize all cosmology/HOD
uncertainties or establish real-DESI robustness.

Opus's concern about simulation efficiency and learning an observation-to-
posterior map was warranted. Its stronger suggestion that amortisation is the
established explanation, unnecessary in general, or grounds to abandon field
inference was not. Even the known Gaussian problem here retains covariance
errors when that map need not be learned. Conversely, a finite-budget failure of
the easier fixed fit is not a proof it cannot represent the target.

The [Doeser--Jasche reference study](https://arxiv.org/html/2606.10023v1) also finds
that concentrating training on reference-posterior samples helps without removing
all discrepancies, and that mean/cross-correlation agreement is insufficient.
Our comparison is an adaptation, not a reproduction: their inverse problem and
SI transport differ, and our VDM comparison is an additional control. The paper
supports careful validation, not a blanket impossibility claim.

Amortisation might be useful across many mocks, survey domains and systematic
variants. Whether it wins in total accuracy/cost for a single DESI survey remains
open. An explicit likelihood is not automatically simple: the smoothed matter
field alone does not determine galaxies, velocities or correlated survey losses.

## Next decision, not an automatic launch

Keep the prepared Abacus products and P12 unchanged. Use the cheap reference to
compare the frozen learned denoiser/velocity against its exact conditional affine
map over time/noise and spatial modes. A next bounded learning control should use
that exact conditional target to remove stochastic training-target variance,
with an affine Gaussian-capable learned baseline and a measured learning curve.
This can separate optimization/target noise from the U-Net representation; it is
a diagnostic teacher available only on the reference, not a proposed DESI oracle.
CFM is the cleaner first instrument here because its numerical floor is negligible.
Any VDM comparison should qualify at least512NFE on this problem or use an
independently validated improved sampler. No such extra neural fit was launched.

Only after a reproducible learned reference pass should the frozen nonlinear
lognormal/Poisson test be designed against demonstrably converged numerical
posterior chains. The coupled proposal's dependence tolerances still require
phase-aware power/effect-size validation on relevant conditional probes. A pass
on16projections of an8^3 Gaussian toy would not prove correct full-field DESI
posterior geometry, nor authorize a large campaign automatically.
