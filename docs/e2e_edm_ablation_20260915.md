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
`configs/e2e_edm_ablation_20260915.json`.

## Completed results

Job58361744 used one A100 on nid001028; source66ba31d. The bounded experiment
completed in617.65s (10m17.65s, excluding startup/preflight), process10m45.99s,
allocation14m50s, now released. All six512-update fits,1,440 probes,42
true-coarse diagnostic draws and18 checkpoints completed. Fifteen focused
tests pass. Full-size initialization differences were exactly zero for every
arm; paired initial probes, separate evaluation noise, source/checkpoint/draw
hashes and the final original-pipeline preflight pass. No held-out payloads read.

**No arm passes. All72 correlated phase/ratio/group checks fail.** The table
reports median high-k noise amplitude remaining, as a percentage of the injected
noise amplitude; smaller is better. This is a signed projection, not an RMS
uncertainty or total reconstruction error. All listed values are positive.
Required: <=20% on each phase at BOTH ratios, with the error and signal gates.

| Arm | Fit .05 | Fit .2 | Transfer .05 | Transfer .2 | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent384 | 98.72% | 91.28% | — | — | 96,969 |
| vp_inherit | 86.33% | 50.80% | 86.96% | 52.38% | 96,969 |
| vp_reset | 87.65% | 53.76% | 87.81% | 54.79% | 96,969 |
| edm_noise | 74.30% | 33.33% | 75.01% | 36.28% | 96,969 |
| edm_film | 73.99% | 25.64% | 75.80% | 30.86% | 102,825 |
| edm_film_clip10 | 74.71% | 28.13% | 75.22% | 31.08% | 102,825 |
| edm_film_rf | 75.20% | 29.09% | 75.79% | 31.80% | 159,369 |

The modified arms' fitted near-clean lower-three-band median gains remain about
.982--1.004: noise removal at those levels is real, not wholesale removal of
the signal. But fitted high-k error power at .05 remains .684--.705 of parent
and at .2 remains .276--.293, rather than <=.25 (ratios of group medians here;
the archived gate uses paired ratios within each phase). Transfer errors remain
larger. Even edm_film, with the lowest fitted noise-amplitude projection, has
error/parent .705 at .05 and .293 at .2. Noise projection alone would therefore
overstate its improvement.

### What the interventions isolate

1. **Noise exposure matters under the tested optimizer/network.** Against
   vp_reset, log-normal exposure improves both near-clean ratios substantially
   without adding parameters. This is not evidence for a novel EDM scalar-loss
   fix: the loss/gradient equivalence control rules that interpretation out.
2. **Per-block log-noise conditioning has a limited, noise-dependent benefit.**
   Against edm_noise, edm_film improves fitted .2 residual amplitude .333->.256,
   transfer .363->.309, but essentially not .05. This combines conditioning with
   5,856 adapter parameters; it does not isolate embedding from adapter capacity.
3. **Inherited optimizer moments are not a sufficient explanation.** Resetting
   them does not improve near-clean removal in this paired run. This does not
   rule out different learning rates or other optimizers.
4. **Frequent clipping is not a sufficient explanation.** Raising the threshold
   from1 to10 reduces clipped updates from100% to17.38%, yet does not improve
   residual-noise amplitude. Median pre-clip norm falls5.95->5.01. Other arms
   clip99.80--100% of updates; clipping frequency alone was not causal evidence.
5. **The tested extra bottleneck capacity/receptive field is not a remedy.**
   Parameters rise102,825->159,369, but results do not improve over edm_film.
   The RF component receives nonzero gradients throughout (median norm.040),
   so it is not disconnected. This is one bottleneck adapter, not a test of
   wider high-resolution layers or every possible denoising architecture.

### Generation and all-noise regressions

The fixed-seed true-coarse draws improve dramatically in high-k power, but do
not acquire the correct spectral shape. These are diagnostic oracle-coarse
fields, not posterior samples usable for science. Band boundaries are
[0,.08,.16,.32,infinity] h/Mpc. Table entries are generated power / truth power,
medians over the three fitted anchors; ideal power agreement is1, not a
standalone calibration criterion.

| Arm | .08--.16 | .16--.32 | >.32 |
| --- | ---: | ---: | ---: |
| Parent384 | .547 | .755 | 80.80 |
| vp_inherit | .802 | 1.085 | 15.33 |
| vp_reset | 1.034 | 1.146 | 20.34 |
| edm_noise | .592 | .903 | 6.93 |
| edm_film | .550 | .858 | 9.08 |
| edm_film_clip10 | .701 | 1.033 | 9.35 |
| edm_film_rf | .584 | .969 | 9.08 |

Thus the EDM variants reduce graininess relative to both parent and uniform
controls, while the .08--.16 fitted band is weaker than in the matched uniform
controls. They do NOT all regress relative to parent: several bands improve.
Transfer EDM high-k ratios remain6.93--8.98 and middle-band ratios.679--.929.
Support violations also decrease relative to parent, but remain nonzero.

At ratio20, edm_noise high-k error power is4.95x parent on fit/4.57x transfer,
whereas edm_film reduces that error to.449/.809 but has fitted lower-band gains
only [.144,.173,.178], versus [.672,.742,.719] for vp_reset. High-noise clean
estimates need not have unit gain, but these paired changes and generated
spectra prohibit declaring a broad generative solution. There is no uniformly
best arm across noise levels, signal gains, error power and draw spectra.

Exposure is a concrete limitation: the common512 uniform-time draws place
[22,43,206,183,47,11] updates in sigma intervals
[0,.05,.2,1,5,20,infinity]; the log-normal draws place
[27,138,276,63,7,1]. There are only8 log-normal updates above5, versus58 for
uniform time. The image-derived distribution is not a proven fit to this
cosmological residual. Improved near-clean behavior does not imply high-noise
competence, and rare-high-noise exposure may contribute to the regressions;
that explanation is not separately established by this ladder.

### Decision and next isolating test

**Keep full E2E paused at384; promote none of these diagnostic branches.**
There is partial learnability, no required capability, no calibration claim and
no demonstrated convergence. Curves are still moving at512; failure in this
bounded test does not establish a plateau or architectural impossibility.
It also does not justify an unbounded extension or simply changing768 to a
larger update count. CFM was not retested in this diffusion-specific ladder.

The next informative experiment, requiring a separately bounded run, is a
fixed-sigma .05/.2 capacity/representation control: fit a small denoiser with
an explicit noisy-input residual/skip path and compare a linear train-only
reference, current backbone and stronger high-resolution residual blocks on
the SAME signal/noise/phase gates. Keep noise conditioning constant within
each fit to separate local denoising optimization from multi-noise interference.
Use the reference only as an explicitly labelled diagnostic, not post-hoc
smoothing of production fields. If a simple reference passes but the backbone
does not, investigate architecture/conditioning suppression; if fixed-sigma
networks pass but joint training fails, investigate noise conditioning and
cross-noise gradient conflict. Retain high-noise probes/generation as a
separate regression gate before any joint-training/E2E restart. This follow-on
has NOT been launched; no additional training is running.

## Evidence and operational provenance

- [Compact summary, every phase gate and hashes](evidence/e2e_field_v2/edm_ablation_20260915/SUMMARY.json).
- [Learning curves](evidence/e2e_field_v2/edm_ablation_20260915/learning_curves.png).
- [Scheduler receipt](evidence/e2e_field_v2/edm_ablation_20260915/RUN_RECEIPT.json).
- Scratch root:
  `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/edm_ablation_20260915_58361744`.
- Main raw artifact `ABLATION_COMPLETE.json`; SHA256
  `3eb1eacdd63a10f6d825de3ac0e79a12a8712c3abf59dbc3bd84d8c33c644193`.
- Main log `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/edm_ablation_58361744.log`.

Training step.0 and final report step.3 completed0:0. Initial report step.1
also completed; verification-enhanced report step.2 hit the intended exclusive
file-creation guard because the first report existed, after its checks passed.
The initial report was preserved under Scratch `initial_report/`; step.3
created the final verified archive successfully. No fit or draw was rerun,
no source binding changed, and no scientific failure was hidden by retrying.
