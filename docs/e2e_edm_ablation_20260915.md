# Actionable fine-denoiser ablations — 2026-09-15

User requested implementation and testing of the noise/time, preconditioning/loss
and capacity hypotheses from the linked *Pause E2E Training* discussion. The full
linked message was read as context; live repository evidence takes precedence.
This is a bounded training-only experiment, not authorization for full E2E
continuation, release, or opening sealed phases. CFM/coarse parents stay frozen;
the first causal ladder targets diffusion, where the EDM mapping is exact.

## Important algebraic control: EDM is not automatically a new loss here

The paper separates preconditioning, noise sampling and network conditioning.
Its scalar coefficients and log-normal noise distribution were verified against
[Table 1 and Section 5](https://proceedings.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf)
and the authors' [preconditioning](https://github.com/NVlabs/edm/blob/main/training/networks.py)
and [loss](https://github.com/NVlabs/edm/blob/main/training/loss.py) implementations.
The HF markdown/HTML routes failed; the published NeurIPS PDF and official code
were used. The implementation here is independent, based on the equations.

For normalized targets with sigma_data=1, let z=y+s*epsilon,
a=1/sqrt(1+s^2), b=s*a, x=a*z. Existing VP v=a*epsilon-b*y implies

    D = a*x - b*v = a^2*z + b*F, with F=-v.

EDM gives c_skip=a^2, c_out=b, c_in=a and loss weight1/b^2. Thus
weighted clean-field MSE equals F-target MSE equals the current v-MSE.
This is our algebraic inference, not a claim that every VP formulation equals
EDM. An EDM wrapper alone cannot fix this model. Stable F-target evaluation
avoids subtracting nearly equal clean fields at small noise. Unit tests compare
losses AND parameter gradients, not just coefficients.

## Registered intervention ladder

All arms start from identical diffusion fine384 weights, including the frozen
normalization. Same three NGC fit anchors and three SGC training-only transfer
anchors, evaluation-noise seeds, targets, conditions and five noise ratios as
the prior learnability test. Fresh common training-noise/time streams are used
across arms. Each fit stops at512 diagnostic updates; checkpoint/evaluation
points0/128/256/512. AdamW learning rate1e-4/weight decay1e-4 unchanged.

| Arm | Change | Causal comparison |
| --- | --- | --- |
| vp_inherit | Original uniform-time objective and inherited AdamW | Matched longer diagnostic control |
| vp_reset | Fresh AdamW moments | versus vp_inherit: optimizer history |
| edm_noise | log(sigma)~Normal(-1.2,1.2^2), stable EDM-equivalent loss | versus vp_reset: noise exposure; not a new v-loss weighting |
| edm_film | Zero-initialized log-noise scale/bias modulation at every encoder/decoder block | versus edm_noise: conditioning plus small adapter capacity |
| edm_film_clip10 | Clip threshold10 instead of1 | versus edm_film: clipping policy |
| edm_film_rf | Zero-output residual bottleneck with dilations2/4 | versus edm_film: capacity/convolutional receptive field combined |

The modulation uses log(sigma)/4 with sinusoidal features and an MLP; the
original input time channels remain to preserve the warm-started model.
New block scales/biases and the added residual output start at zero. Require
identity with parent predictions before fitting and identical initial panel
probes. This is an EDM-inspired conditional3-D adaptation, not reproduction of
the paper's image benchmark or original architecture. sigma_data=1 follows the
existing unit-variance normalization; the paper's image-specific .5 is not
blindly substituted. Its log-normal numerical parameters are a tested choice,
not assumed optimal for cosmological fields.

Only the log-noise embedding uses a [.002,80] finite range, including the
pure-noise VP endpoint. Actual VP states, targets, legacy time features and
training noise are not clipped. This finite endpoint proxy is a documented
limitation. The capacity arm preserves the nonperiodic cutout padding and adds
no Fourier wrapping. RF and parameter count effects are not separately identified.

Preserve the existing32-step DDIM sampler for all diagnostic draws, so a solver
change cannot masquerade as a denoiser improvement. Generate parent plus six
final-arm true-coarse draws on all six anchors:42 oracle controls. No clipping,
post-hoc smoothing, estimator change, or promotion of these fields is allowed.

## Required capability and interpretation

Retain the previous per-phase near-clean gate: at ratios.05/.2, residual high-k
noise amplitude<=.2, high-k error power<=.25 of matched parent, lower-three-band
gain .9--1.1. Fit and transfer checks stay separate. Inspect all five ratios and
the generated middle/high-k power for specialist regressions, whether or not
the primary gate passes. Training controls cannot establish calibration.

Record clipping frequency, pre-clip gradient norms by base/adapter/RF component,
sampled noise distribution, parameter counts, full learning curves and field
hashes. No automatic successful-arm selection or extension follows these tests.
A failure at512 does not establish architectural impossibility or a plateau.
Near-clean success with poor generation is not an E2E restart criterion.

One GPU, at most one hour, application work cap45 minutes. Stop on nonfinite
loss/gradient, initial mismatch, provenance drift, or budget expiry. Separate
experiment binding on every new checkpoint; original step384 remains the
full E2E endpoint. The six branch endpoints are not a new full-pipeline run.
Code/config: `e2e_edm_ablation_models.py`, `e2e_edm_ablation.py`,
`configs/e2e_edm_ablation_20260915.json`. Results will be appended after execution.
