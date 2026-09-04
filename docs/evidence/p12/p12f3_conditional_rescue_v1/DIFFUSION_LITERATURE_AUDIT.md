# P12-F3-L2d diffusion literature and adequacy audit

Date: 2026-09-04.  This audit was written while the frozen seed-42 F3-L2d
posterior was being evaluated.  It does not alter that checkpoint, its target,
conditioning, samples, or registered ph006 gates.  `ph001` remains sealed.

## What the bounded comparator does well

The current arm is a controlled objective swap against F3-L2c.  Both models see
the same seven-channel BRIGHT/random-response condition, the same train-only
conditional location/scale transform, and the same exactly represented and
whitened low-frequency Hermitian Fourier target.  F3-L2d uses a variance-
preserving cosine path, uniform continuous time, `v` prediction, and a
deterministic DDIM sampler.  The `v` parameterization is specifically motivated
for stable few-step deterministic sampling by Salimans & Ho
([arXiv:2202.00512](https://arxiv.org/abs/2202.00512)).

## What it is not

It is not yet a literature-grade cosmological diffusion architecture.  Its
denoiser is the capacity-matched `base=4` U-Net inherited from the flow control;
time is supplied as one constant spatial channel.  It has no sinusoidal/log-SNR
time embedding injected at every level, no diffusion residual blocks, no
attention, no exponential-moving-average evaluation weights, and no learned
reverse variance.  The frozen budget is 10,000 one-patch updates and the primary
sampler uses 24 network evaluations.

For comparison, Ono et al.
([arXiv:2403.10648](https://arxiv.org/abs/2403.10648)) use a four-block
hierarchical residual U-Net with group normalization, AdamW, cosine warm
restarts, 60,000 updates at batch size 12, and 250 refinement steps.  Their
ablation trains to 300,000 updates and finds a modest but persistent benefit
from attention.  Nichol & Dhariwal
([arXiv:2102.09672](https://arxiv.org/abs/2102.09672)) likewise show that
capacity, training compute, the noise schedule, and reverse-variance treatment
can materially change sample quality and the number of adequate sampling
steps.  EDM ([arXiv:2206.00364](https://arxiv.org/abs/2206.00364)) is a further
warning not to conflate network preconditioning, training schedule, and sampler
choice.

Therefore a negative result from the current arm means only that this matched
small-network diffusion objective did not beat the matched flow.  It cannot be
reported as evidence that conditional diffusion in general fails for the field
posterior.

## Practices to retain and mistakes not to import

1. Do not select a posterior using only denoising loss, MSE, images, power
   spectra, or cross-correlation.  The CAMELS reconstruction paper reports these
   useful summaries, but our primary gates remain block-aware TARP, conditional
   coverage, and joint proper scores after the fixed tidal-physics layer.
2. Do not copy periodic/circular padding from periodic-box field emulators into
   CutSky patches.  The random-derived support and boundary channels must remain
   explicit, and unrelated survey edges must never be joined.
3. Do not add classifier-free guidance or another sharpening heuristic: it
   changes the conditional law and can manufacture undercoverage.
4. Do not infer sampler adequacy from training loss.  Compare the frozen 24-step
   DDIM archive with a common-seed 50-step archive before diagnosing the
   denoiser.
5. Do not copy an update count mechanically.  A successor must use internal
   training-phase sample diagnostics and effective passes, with a frozen cap and
   no ph006-selected extension.
6. Preserve physical augmentations only.  Any rotation or reflection must act
   consistently on the field, survey/line-of-sight geometry, and every response
   channel; otherwise it creates an invalid observation pair.

## Registered interpretation ladder

The 24-versus-50 DDIM audit is checkpoint-only and uses the same 256 ph006 cores,
64 draws, seeds, physics, and evaluator.  Sampler convergence requires changes
of at most 0.01 in ordered/eigengap TARP and global coverage error, 0.05 in either
registered low-band power ratio, and 1% in every proper score.

If this audit passes, the 24-step result is an adequate measurement of the
frozen small diffusion arm.  If it fails, no sampler-family conclusion is
allowed until the 50-step result is interpreted.

The 24-to-50 comparison subsequently missed only the frozen low-band-power
change tolerance (`0.056 > 0.05`), while TARP, global coverage and all proper
scores were stable.  Because the 50-step sample also moved the longest two
bands closer to their physical target, one post-result checkpoint-only
NFE100 audit was authorized with unchanged cores, draws, seeds, checkpoint and
evaluator.  This does not refit or promote the model.  Sampler convergence is
now assessed by the same untouched tolerances on 50-to-100; failure there
blocks any claim about the denoiser or diffusion family from this arm.

A separate literature-grade diffusion successor is justified only as a newly
registered model, not a continuation.  Its minimum contract is:

- a 3-D residual U-Net with GroupNorm, multilevel sinusoidal/log-SNR time
  embeddings, and bottleneck/coarse-scale attention;
- a capacity canary against the current `base=4` model, plus effective-batch
  scaling through GroupNorm-safe accumulation if memory still forces one patch
  per device;
- EMA and raw-weight checkpoints evaluated side by side;
- a train-split-only comparison of the frozen cosine VP schedule against a
  learned monotone log-SNR schedule or EDM-style preconditioning; these are
  separate ablations rather than an untracked bundle of improvements;
- train-only validation sample diagnostics in addition to the denoising loss;
- 24/50-step sampler convergence and, if needed, a stochastic reverse-SDE
  control;
- power, cross-correlation and phase-sensitive higher-order summaries alongside
  the unchanged physical TARP/coverage/proper-score evaluation and sealed
  `ph001` boundary.

Legin et al. ([arXiv:2304.03788](https://arxiv.org/abs/2304.03788)) demonstrate
that score models can sample high-dimensional cosmological fields and correctly
stress posterior realizations rather than point reconstructions.  Their
standardized-residual coverage test is useful, but it checks only marginal first
and second moments; it does not replace our joint eigenvalue, eigengap, shear,
conditional, and proper-score ladder.

Riveros et al. ([arXiv:2502.17087](https://arxiv.org/abs/2502.17087)) provide a
useful target-matched warning about the sampler itself: their 3-D density-field
comparison finds a speed/quality tradeoff between DDIM and the much longer DDPM
reverse chain, and explicitly reports low-wavenumber limitations from finite
training volumes.  We therefore require the frozen NFE convergence audit and a
stochastic reverse-SDE/DDPM-style control before attributing a residual failure
to the learned denoiser.  Their power/PDF/bispectrum validation motivates adding
phase-sensitive higher-order summaries, but it still cannot replace calibrated
conditional-posterior tests after the tidal-physics layer.
