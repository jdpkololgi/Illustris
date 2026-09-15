# Fixed-noise fine-denoiser capability test — 2026-09-15

## Registered plan

User approved planning, implementing and testing the fixed-noise/residual-path/
high-resolution comparison recommended after the EDM-informed ablation. Full
E2E remains paused at384; this is a bounded training-only diagnostic, not a
production model change, held-out opening or full diffusion sampling run.

Use the same three NGC fit anchors (ph000/ph002/ph003), three SGC training-pool
transfer anchors, frozen normalization, teacher-forced coarse conditioning,
five evaluation noise ratios and two evaluation noise replicates as the earlier
tests. Transfer changes cap/conditioning but is NOT held-out validation. New
training-noise seed91529 is shared across arms and noise levels, separate from
the original evaluation-noise seed91526. Initial model seed91530 is fixed.

Train separately at sigma/alpha=.05 and .2,512 updates EACH for four arms:

| Arm | Initialization and local operation | What it tests |
| --- | --- | --- |
| vp_parent | Original384 U-Net, fresh AdamW, original v-MSE | Can the current trained backbone learn one fixed noise level? |
| vp_fresh | Fresh original U-Net, zero output head, v-MSE | Warm-start history versus starting anew |
| residual_unet | Same fresh backbone weights/zero head, normalized correction-MSE, explicit clean skip | Output parameterization/residual prediction, conditional on this fresh initialization |
| residual_highres | Fresh width24/four full-resolution residual blocks, same correction objective/clean skip | Combined resolution, residual architecture, capacity and normalization contrast |

All eight fits use AdamW lr1e-4, weight decay1e-4 and clip1; evaluations and
checkpoints at0/128/256/512. Hold noise level constant within each fit while
cycling the three fit anchors and drawing independent fresh Gaussian noise.
Keep local and wide conditions in every neural model. The full-resolution model
retains a wide summary encoder, has no local downsampling or activation
normalization, uses zero-padded nonperiodic convolutions and a zero output head.
It is NOT a pure receptive-field or GroupNorm ablation. Matched update counts
are not matched FLOPs; record parameter counts and branch wall time.

For raw noisy z=y+s*epsilon, normalized VP state x=a*z, b=s*a:

    correction model: D = z+r_hat, target r = -s*epsilon
    equivalent audit velocity: v = -b*x/a-r_hat/b
    objective = mean((r_hat+s*epsilon)^2)/s^2

The audit's clean estimator a*x-b*v therefore equals D. The correction objective
is clean-MSE divided by s^2; v-MSE is clean-MSE divided by b^2. Predicting the
correction directly in field units avoids asking the raw head to amplify a
small noise perturbation by1/s. This changes parameter-space gradients, despite
the fixed-sigma clean losses differing only by a scalar. A win is not solely
attributable to one extra identity connection. The fresh v zero head begins at
a*x, whereas both zero-headed correction networks begin at z. No inference-time truth input.
Tests verify the conversion, fixed objective, initialization and output gradients.

Two no-learning controls: noisy identity and a **predeclared nonperiodic linear
low-pass**. The latter applies an orthonormal DCT-II, preserves modes k<=.32
h/Mpc, uses a cosine taper .32--.40, and zeros higher modes. The DCT corresponds
to even boundary extension, not periodic opposite-edge wrapping. Cutoffs use
the already declared diagnostic band boundary, not a scan of evaluation results.
This is a constructive reference for attainable low-noise filtering, not a
learned posterior, optimized Wiener estimator or production-field correction.
It may discard genuine high-k signal; its full band metrics and noiseless
response are explicitly retained. Do not mistake a reference pass for a model
or scientific-validation pass. No empirical reference retuning is allowed.

## Gates and budget

Preserve the existing each-phase gate at the trained ratio: absolute residual
high-k noise projection<=.2, high-k error<=.25 of the paired original384 parent,
and lower-three-band gains in[.9,1.1]. Medians over two paired noise seeds within
each phase, fit/transfer assessed separately. An architecture clears the local
capability test only if its TWO independently trained noise-level models pass
all corresponding checks; they do not constitute one multi-noise model.
Retain the existing audit's VP-noise normalization (division by b) unchanged.
The identity reference leaves s/b=1/a of that noise amplitude, not exactly1.

Evaluate all five noise levels at every checkpoint, but off-trained-noise
results are stress tests, not a reason to treat a specialist as a sampler.
Save72 explicitly labelled controlled clean estimates:48 neural endpoints and
24 identity/linear-reference estimates, six anchors per trained ratio/model.
These are denoising reconstructions, NOT generated posterior draws. No DDIM
trajectory or joint-noise continuation is launched automatically.

Expected totals:4,096 optimizer updates;1,920 neural probes plus60 parent and
120 reference probes (2,100 total);72 reconstructions;24 diagnostic checkpoints.
One GPU, one-hour allocation,45-minute application cap; stop on nonfinite
loss/gradient, provenance mismatch or elapsed budget. No automatic restart or
extension. Original source, datasets, normalization and parent checkpoints are
immutable; new source/config hashes bind the independent experiment.

Interpretation: reference success with neural failure supports an optimization/
representation bottleneck under this budget, not an intrinsically impossible
threshold. Neural fixed-noise success narrows the next question to joint-noise
training, but cannot by itself causally prove gradient interference: earlier
joint runs have different schedules/initialization/training seeds. No success
establishes calibration or warrants restarting full E2E. Failure at512 is not
proof of architectural impossibility or convergence. Report successes,
negative results, signal loss, off-noise regressions and remaining confounders.

Sources: `configs/e2e_fixed_noise_20260915.json`,
`workflows/sbi/e2e_fixed_noise_models.py`, `workflows/sbi/e2e_fixed_noise_test.py`.
Execution and results are recorded below.

## Post-fit response localization (descriptive, no changed gate)

During the run, the .05 residual U-Net passed all fitted-field checks but failed
the transfer reconstruction-error checks despite strong noise cancellation.
Add a bounded read-only response check to distinguish distortion already
present on clean inputs from sensitivity to the injected noise. No model,
training budget, checkpoint-selection rule, threshold or reference cutoff changes.

For each of the eight final checkpoints and six anchors, hold the nominal noise
time and conditions fixed, evaluate the clean input and a sign-reversed version
of evaluation-noise replicate0, and reuse the saved positive-noise reconstruction.
There are96 new forwards, zero training updates and a ten-minute application cap.
For reconstruction errors e+ and e-, define even=(e++e-)/2 and odd=(e+-e-)/2.
Verify mean paired error power = even power + odd power in every spectral band.
Compare the even error with the clean-input error, including their correlation
and difference power. Noise-even error is not automatically a bias estimate;
agreement with the clean-input response is separate evidence for that attribution.
Neither this antithetic pair nor training-pool transfer is independent validation.
Source: `workflows/sbi/e2e_fixed_noise_response.py`.

## Results: fitted-field capability demonstrated, full transfer gate still fails

Job58363476 completed on one A100/nid001040 and was released. Main experiment
843.46s (14m03.46s, excluding startup/preflight), process14m31.09s; allocation
17m17s. Training and analysis steps both COMPLETED0:0. All4,096 updates,
2,100 probes,72 controlled reconstructions and24 checkpoints completed. The
post-fit check added96 forwards, zero updates,16.07s after preflight. Fourteen
focused tests pass, including residual-chart identity and antithetic power
closure. Source/checkpoint/reconstruction hashes and training/evaluation-noise
separation verify. No source drift, held-out reads or full diffusion sampling.

**The same-size residual U-Net passes all three fitted phases at BOTH noise
levels. No architecture yet passes both levels on fit AND transfer.** This is
the first positive fitted-field capability result in this diagnostic sequence.
It revises the blanket claim that the original backbone cannot learn the local
operation: it can, with the tested field-unit output parameterization and
fixed-noise exposure. It does not validate the original joint-noise checkpoint.

Median injected high-k VP noise amplitude remaining after512 updates:

| Arm | Fit .05 | Transfer .05 | Fit .2 | Transfer .2 | Full each-phase gate: .05 fit/transfer; .2 fit/transfer |
| --- | ---: | ---: | ---: | ---: | --- |
| vp_parent | 21.82% | 26.12% | 13.59% | 13.93% | fail/fail; pass/pass |
| vp_fresh | 67.69% | 71.41% | 71.62% | 72.32% | fail/fail; fail/fail |
| residual_unet | 13.09% | 13.17% | 4.28% | 3.91% | pass/fail; pass/pass |
| residual_highres | 49.95% | 51.98% | 3.14% | 5.69% | fail/fail; pass/pass |

The gate includes error reduction and signal preservation, not just <=20%
amplitude. Neural phase checks:21/48 pass, with9/12 for residual_unet,
6/12 for vp_parent,6/12 for residual_highres and0/12 for vp_fresh. Checks are
correlated, not independent trials or confidence estimates. No architecture
has a complete fit+transfer pass at both levels. All group-level passes first
occur at the registered512 endpoint, not an earlier selected checkpoint.

For residual_unet at .05, all fitted phase noise amplitudes lie .130--.147;
paired high-k error/parent .142--.144, and lower-three-band gains
.99943--1.00016. This meets all criteria, rather than hiding signal removal
behind a low noise projection. At .2, all fitted and transfer phases pass:
noise .039--.053 fit/.037--.043 transfer, error/parent .034--.038 fit and
.054--.077 transfer, with lower-band gains .9969--1.0025.

At .05 transfer, residual_unet still removes the injected noise (amplitudes
.105--.146), but error/parent is .314/.493/.428 for ph000/ph002/ph003, exceeding
.25 in every phase. Its lower-band gains remain .99945--1.00056. Group-median
high-k gain is.966 and power/truth1.071, so this is not wholesale erasure of the
high-k signal either. High-k error is about14% of truth-band power, versus
5.7% on fit anchors. About4.34% of transfer high-k error is projected onto the
injected noise, compared with11.90% on fit anchors. The dominant remaining
transfer error is not the original near-clean noise leakage.

### What the comparisons identify

**Output parameterization/optimization matters, without adding capacity.**
vp_fresh and residual_unet have identical initial hidden weights, zero output
heads,96,969 parameters, inputs/conditions,512 fresh-noise updates and optimizer
hyperparameters. The change to field-unit corrections plus a clean skip
reduces fitted .05 residual noise67.7%->13.1% and .2 noise71.6%->4.3%.
This is the strongest controlled result. Raw output scale, clean skip,
initial denoised prediction and parameter-space gradients change together;
do not attribute the entire benefit to a single identity connection or claim
that a new scalar loss alone solved it.

**The current backbone benefits from concentrated fixed-noise exposure.**
vp_parent reaches the .2 gate and gets close at .05. Its prior384 training
updates are additional experience, so comparison with vp_fresh is a warm-start
contrast, not equal total training compute. Earlier joint-noise experiments
also had different time schedules/seeds and far fewer updates near .05. We
have NOT separately proven harmful cross-noise gradient interference.

**More full-resolution capacity is not the near-clean remedy in this budget.**
The139,201-parameter high-resolution model needs about143.5s per branch versus
88--91s for the96,969-parameter U-Net branches; it fails .05 but passes .2.
It combines resolution, residual blocks and removal of normalization, and it
may need different optimization. This does not establish that GroupNorm helps
or hurts, or that all larger models fail. Curves are still moving at512; there
is no convergence/plateau claim or automatic extension.

### Linear reference: why cancellation alone was an unsafe criterion

Identity fails every gate and reproduces the expected unchanged noise. The
predeclared low-pass leaves only .8--1.2% high-k noise amplitude and preserves
lower-three-band gains very close to1. It passes all six .2 phase/group checks.
At .05 it fails all six reconstruction-error checks: error/parent .445--.651
on fit and .357--.592 on transfer. It removes genuine higher-frequency structure
along with noise. The noiseless reference metrics are archived explicitly.
No cutoff was retuned after inspecting these results. Its failure does not
prove all linear estimators fail; its .2 success does not make filtered fields
valid posterior samples. The residual U-Net's .05 fitted pass is stronger
than merely matching this aggressive filter's cancellation.

### Remaining transfer error: clean-input and sign-reversal localization

For residual_unet/.05 transfer, median noise-even high-k error accounts for
71.55% of the paired +/- error power, compared with27.28% on fit. Its correlation
with the clean-input error is.9862 (fit .9236). Clean-input error power is1.124x
the paired transfer error power; this ratio can exceed1 because adding noise
changes the network's response. It is not an additive attribution fraction.
The even-minus-clean difference power is about6.4% of the paired transfer error
(ratio of group medians), and the exact even+odd power closure passes per case.

This is evidence for a persistent reconstruction distortion on transfer
fields, with noise-dependent modulation, rather than
failure to cancel the original perturbation. It does not isolate local versus
wide conditioning, normalization, training-anchor coverage or overfitting as
the cause. The clean input is evaluated at nonzero nominal noise time: these
probes are a descriptive bias/sensitivity control, not a new demand that a
Bayesian denoiser always map a clean field exactly to itself.

The .2 specialists also have nontrivial clean-input responses despite passing
their noisy-field gate; all results are archived. It would be unsafe to infer
from the .05 correlation alone that an identity regularizer must improve
denoising or posterior calibration.

### Decision

Basic fixed-noise denoising is now demonstrated on fitted fields with the
original-size U-Net; transfer success is demonstrated at .2. Retain this positive
result, but **do not promote the specialists or restart full E2E**. The full
two-noise fit+transfer gate remains unmet, and none of these separate fixed-noise
models demonstrates a stable calibrated multi-noise sampler. In particular,
the raw z+r chart is singular at the pure-noise VP endpoint; off-trained-noise
probes are stress tests, not a supported generative interface.

The next priority is the .05 transfer distortion: isolate local/wide condition
sensitivity and training-anchor coverage with matched controls before prescribing
a regularizer or a larger network. Then test a single endpoint-safe multi-noise
residual parameterization against its fixed-noise controls, preserving the
same noise/error/gain gates plus broad-noise and generation checks. No such
training or held-out evaluation has been launched. More updates on the same
three fit anchors are not established as a cure for transfer bias.

## Evidence and provenance

- [Summary, curves, all phase gates and hashes](evidence/e2e_field_v2/fixed_noise_20260915/SUMMARY.json).
- [Learning curves](evidence/e2e_field_v2/fixed_noise_20260915/learning_curves.png).
- [Clean-input/antithetic response](evidence/e2e_field_v2/fixed_noise_20260915/RESPONSE.json).
- [Run receipt](evidence/e2e_field_v2/fixed_noise_20260915/RUN_RECEIPT.json).
- Training source commit `b75d44786ec922f6dc6675f444ab92958ea440bc`.
- Scratch root `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/fixed_noise_20260915_58363476`.
- Raw `FIXED_COMPLETE.json` SHA256 `1fa5efa8b873c96d953167e92743f21f6e2584548217901176020cc984dafaf1`.
- Logs `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/fixed_noise_58363476.log`
  and `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/fixed_noise_analysis_58363476.log`.

Four model arms at two fixed ratios are diagnostic branches only; saved
`update_000512.pt` is a diagnostic update number, not a new full E2E step512.
Original384 weights, normalization, sealed phases and training_ready=false /
r0_physics_pass=false remain unchanged. No P12/D2/P13 changes.
