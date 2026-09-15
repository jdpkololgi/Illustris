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
Results pending execution.
