# Gaussian / log-Gaussian oracle and loss-conflict diagnostic

## Registered follow-on: frozen neural sampler and optimizer factorial

2026-09-16: configs/e2e_frozen_controls_v1.json and e2e_frozen_controls.py freeze
two30720 control seeds; first fitted/transfer field per phase; two paired draws;
DDIM32/128 versus Heun32/128/256/512 NFE. Reference256->512 relativeRMS <=1% is a
provisional numerical screen, not a posterior gate. Fine-only TRUE-coarse input.
216 independent one-step tests cross retained/zero FIRST moment and clipping
on/off, three objectives, three fitted phases and sigma.005/.01/.05. Preserve
variance/step age; compare against denoising-only within the same setting.
Full contract/paper-control status: docs/e2e_frozen_controls_v1.md. No new fit/E2E.

## Completed results (2026-09-16)

All tests completed on allocation58439522, released COMPLETED0:0 after18m18.
48 focused unit/regression tests pass. Frozen initial source38c656b and follow-on
dd38acc; source/result hashes reverified after execution. Checked-in provenance:
[RESULTS.json](evidence/e2e_field_v2/oracle_conflict_v1/RESULTS.json).

Gaussian/log-Gaussian oracle gates pass; Gaussian noisy MSE agrees with Bayes
risk within0.5%. Correct clean-input changes at positive sigma are nonzero;
they should not be confused with a neural modelling error. The separate scalar
lognormal density-noise calculation also converges. Its prior is iid, not the
correlated log-Gaussian field used in the log-space/sampler experiments.

| Model evaluations | DDIM Gaussian error | VP Heun Gaussian error | DDIM log-density error | VP Heun log-density error |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 4.691% | .4690% | 8.557% | .5327% |
| 128 | 1.194% | .02900% | 2.239% | .03274% |
| 512 | .3001% | .001806% | .5665% | .002037% |

These are relative RMS errors against exact transport of IDENTICAL initial
noise, not independent-draw differences. Original low-step DDIM suppresses
power here; this does not explain the earlier cosmic excess high-k power.
The new solver remains a separate candidate, not a production replacement.

Gradient diagnostics produce336 component records and1,008 cross-sigma pairs.
At .01, clean/denoising gradient cosine is+.980 for seed0 control and-.115 for
seed1 control; seed1 strong-preservation has-.160, negative for all6 examples.
Conflicts are not universal and a six-example diagnostic is not a population
estimate. Read-only model and Adam equality checks pass for all eight checkpoints.

The54 restored-checkpoint one-step interventions reject automatic adoption of
raw gradient projection: it enforces nonnegative raw dots but does not preserve
the actual Adam/clipped update or fresh-noise risk. Across9 interventions x3
evaluation sigmas per seed, projected identity is jointly no worse than the
denoising-only control in12/27 tests at seed0 and0/27 at seed1. Median paired
clean/noisy MSE ratios are .9977/1.00013 and .9143/1.01015 respectively. These are
small local changes, not training-convergence or transfer measurements. Some
projected/unprojected medians coincide because projection changes only conflicting
cases. No updated weights were saved; original states restored exactly.

Next actionable order: isolate actual optimizer moments/global clipping; test
the solver candidate on frozen neural scores without training; then register
an oracle-calibrated local learning objective. Full E2E remains paused. Min-SNR
weighting is algebra-tested but NOT trained, because its standard v-weight
strongly downweights the very low-noise regime currently under investigation.

Registered 2026-09-16 after the eight-fit preservation experiment failed its
joint gate. User authorizes implementation/testing, including log-Gaussian fields
and a focused literature check. Full E2E remains paused. No automatic repair fit,
weight sweep, held-out access or reinterpretation of a failed registered gate.

## Motivation and evidence

Array58407655 and report58407656 completed0:0. All eight fits reached30,720
updates. All32 probe hashes verify; summary SHA256
77adc5743c11826ed64330b1d362e6e57c3a7bf6b4ec1d4283331f266dc8ab47.
Weak/strong/response losses reduce final .05 transfer clean RMS by23/47/40%
at seed0 and23/54/47% at seed1, but seed1 .01 noisy MSE rises47/101/29%.
No modified arm passes across both seeds/checkpoints/groups. A positive-noise
clean-preservation preference is not necessarily the Bayes denoising target.

## Known-distribution tests

Synthetic periodic16^3 boxes,32 addressed draws, known positive spectrum P with
point variance.49; fixed spatial conditional mean. No empirical per-draw scaling.
Gaussian additive-noise oracle is m+P/(P+sigma^2)*(y-m) in Fourier coordinates.
Its finite-sigma clean distortion is legitimate shrinkage, not learning error.
Compare noisy MSE with exact Bayes risk and clean distortion acrosssigma.

Log-Gaussian: delta=exp(g-var(g)/2)-1. With Gaussian noise added to g, use the
exact posterior density MEAN exp(mu_post+var_post/2-var_prior/2)-1, not just the
exponential of the posterior log mean. Also test a separate iid lognormal prior
with additive DENSITY noise using adaptive scalar quadrature (tightened tolerance
cross-check). This is deliberately NOT a correlated-density posterior oracle.
These likelihoods are distinct; no direct inference about changing cosmic
residuals to log space. Fine residuals can be negative and need not obey delta>-1.

Adapt the known Gaussian conditional expectation to BOTH the existing cosine-VP
v/DDIM sampler and straight-path CFM/Heun sampler. Use the same initial white
noise and compare with the exact affine transport at32/128/512 steps; exponentiate
the same coupled draws for a log-Gaussian generative check. Record shell power,
relative field error and density support. Tests concern the local sampler
primitive, not the entire conditional coarse/fine posterior pipeline.

Preregistered numerical checks: noisy Gaussian MSE within5% of analytic risk;
512-step coupled relative RMS<2.5% Gaussian/<5% log-Gaussian for both samplers;
quadrature tolerance changes<1e-7. Record failures without adjusting thresholds.
No neural network is fitted to synthetic data in this phase: the exact oracle
separates numerical and diagnostic errors from learnability.

## Checkpoint gradient experiment

All eight final checkpoints, unchanged weights/Adam. First registered fitting
field per phase (three), two independently addressed noise/auxiliary draws,
sigma=.001/.005/.01/.05/.2/1/5. No transfer field is used to choose gradients.
Measure individual denoising, clean-anchor, even-drift and odd-response gradients,
cosines/norms and combined auxiliary weights; compare denoising gradients across
sigma. Auxiliary probes above.05 are explicitly extrapolative diagnostics, not
training exposures. Paired fields/noise across models. 336 component records,
1,008 cross-sigma comparisons. Hash-bound per-checkpoint output permits restart.

Implement stable Min-SNR-gamma=5 v-weight min(1,gamma*sigma^2)/(1+sigma^2) and
one-sided conflict projection as algebra-tested diagnostic candidates only.
The projection guarantees a nonnegative raw gradient dot product, not safe Adam
updates or transfer performance. Min-SNR suppresses low-sigma v-MSE; blindly
applying it here could worsen low-noise underlearning. Neither changes a checkpoint.
Only after these results justify a repair should a bounded paired neural fit be
registered; independent-seed success precedes any E2E sampling/training restart.

## Focused literature check

- Hang et al., [Min-SNR](https://arxiv.org/html/2303.09556v2): conflicting
  timestep gradients motivate noise-dependent weighting. Implement the correct
  v-chart factor and measure conflicts first; image FID gains are not our gates.
- Ono et al., [Debiasing with Diffusion](https://arxiv.org/html/2403.10648v1):
  galaxy-conditioned dark-matter reconstruction using 2D CAMELS projected maps.
  Appendix B compares attention, U-Net depth, time embedding and a learned versus
  fixed noise schedule. Relevant controls, but not evidence for our 3D selected
  lightcone/coarse-fine posterior or for stronger clean-identity penalties.
- [Conditional Diffusion-Flow 3D fields](https://arxiv.org/html/2502.17087v1):
  parameter-conditioned3D generation uses train-derived log scaling, clipping and
  Huber loss. Supports examining dynamic range and robust objectives, but clipping
  changes density tails and Huber changes the optimum; do not copy them silently.

Hugging Face paper markdown endpoints failed; regular paper page and primary
arXiv HTML were used. No source/data were uploaded and no packages installed.
Diffusers is absent in cosmic_env: this phase verifies analytic formulas and our
existing sampler directly, not a claimed Diffusers-library equivalence test.

## Execution

### Follow-on numerical candidate, registered after the oracle result

The initial oracle passes at512 steps, but32-step DDIM has4.69% Gaussian and
8.56% log-density coupled relative RMS error even with an exact denoiser.
Implement a separate cosine-VP probability-flow Heun integrator with derivative
(pi/2)*v, integrating1->0. Compare against unchanged DDIM at matched32/128/512
network evaluations (Heun uses half as many intervals). Same spectrum, draws,
condition and initial noise. Success requires lower coupled RMS for BOTH charts
at EVERY tested NFE. This follow-on is explicitly selected after the initial
oracle result; it is not part of the original preregistration. No E2E code is
switched over, and success with an oracle does not imply success with a neural
score, finite-time endpoint extrapolation or posterior calibration.

`e2e_oracle_solver.py --root <fresh-frozen-root>` writes SOLVER.json. Stage a
separate source snapshot after committing this candidate; preserve the original
oracle/gradient run unchanged. Do not rerun all gradients for this solver test.

### Follow-on optimizer intervention, registered after partial gradients

The initial measurements show seed/noise-dependent negative clean/denoising
gradient alignment. Test one-sided projection through the ACTUAL restored AdamW
state and clipping, not just Euclidean dot products. `e2e_gradient_step_probe.py`
uses the two control30,720 checkpoints, three first registered fitting fields,
and gradient sigma=.005/.01/.05. For EACH case independently restore model/Adam,
take one step with denoising only, denoising+identity(weight1), or denoising plus
projected identity. Evaluate clean MSE and two NEW noise draws at all three
sigmas on that field; also report actual parameter-displacement dot products.
54 single-step interventions, not54 sequential training steps. No weights saved,
no transfer use or generalization/convergence claim. All originals restored.
Projection is rejected as an automatic repair if it fails to improve the
fresh-noise clean/noisy tradeoff consistently versus the denoising-only control.
No full fit follows automatically, regardless of this local result.

Activate cosmic_env before graphify/Python. Commit, then stage a new
oracle_conflict_* child of the registered Scratch root with
`python -m workflows.sbi.e2e_oracle_conflict stage --root <root>`.
On one approved GPU, from <root>/source, run the same module with `run`.
New source is frozen; prior hash-bound experiments are unchanged. Logs and
ORACLES.json, gradients/*.json, SUMMARY.json stay on Scratch. Interrupted
incomplete checkpoint diagnostics can be recomputed; no optimizer updates occur.
