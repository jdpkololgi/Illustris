# Frozen neural sampler and optimizer controls

## Completed results

Allocation58442539/nid001012 released COMPLETED0:0 after39m44; frozen source
df9aa25, report a1f19e7.57 focused
tests pass.216 independent optimizer steps and144 saved fields (24 paired cases
x6 methods) complete. All54 retained-momentum/clipped controls exactly reproduce
the previous probe, including fresh-noise errors and displacement dots. Receipt:
`docs/evidence/e2e_field_v2/frozen_controls_v1/RESULTS.json`. Full arrays, metrics,
fixed slices and hashes are in the receipt's Scratch root. Original checkpoints
reloaded/hash-verified after execution. No updated weights saved or E2E restart.

All24 reference256->512 screens pass; maximum coupled relativeRMS0.1784%.
Heun128 beats DDIM128 numerically in24/24 cases. At32 NFE it wins only17/24,
and high-k power is inflated relative to its refined limit. No default switch.

| Seed/group | DDIM128 relativeRMS | Heun128 relativeRMS | Refined residual power ratios, four bands |
| --- | ---: | ---: | --- |
| 0 fitted | 1.660% | .473% | .954/.894/.978/1.156 |
| 0 transfer | 2.196% | .581% | 1.757/.967/.969/1.129 |
| 1 fitted | 1.828% | .830% | .892/.775/1.046/1.275 |
| 1 transfer | 2.017% | .855% | 1.348/.945/1.025/1.098 |

Bands are (0,.08],(.08,.16],(.16,.32],>.32 h/Mpc, Hann-windowed fine residual;
the lowest-band excess is NOT a35--76% claim about the total matter field.
Refined transfer median density<-1 fractions are1.75e-5 and5.65e-7; fitted medians
zero. True targets have no support violations (minimum -.795 to -.815). Paired
draw differences are49--93% of sample RMS in normalized residual units, ruling
out exact draw collapse, NOT establishing calibrated uncertainty. The images
are inspectable but spectral biases persist after numerical refinement.

Projected auxiliary joint clean/noisy non-regression versus matched denoising-only
controls, each denominator27 (9 interventions x3 evaluation sigmas):

| First moment / clipping | Seed0 | Seed1 |
| --- | ---: | ---: |
| retained / on | 12/27 | 0/27 |
| retained / off | 21/27 | 0/27 |
| zeroed / on | 12/27 | 1/27 |
| zeroed / off | 19/27 | 2/27 |

Clipping changes the update geometry materially: projected-gradient median scale
.0558/.0901 (seeds0/1). Removing clipping eliminates adverse incremental primary
dots for seed0, but leaves6/9 for seed1 even without historical first moments.
Clearing first moments removes seed0's absolute local-ascent cases, not the joint
tradeoff. For seed1, retained/no-clip clean/noisy ratios are .718/1.044 versus
.914/1.010 with clipping. Thus neither blanket unclipping, reset-first-moment,
nor raw Euclidean projection is a demonstrated reproducible auxiliary-loss repair.

Main scientific step24m28 and report10s complete0:0. Supplemental CPU-generator
reload failed because saved objective RNG was CUDA; corrected GPU reload passes.
This was a validation-device error, not a failed fit or a sampling rerun.

Decision: preserve the improved denoiser as a research reference, but stop the
sampler-only/identity-only repair loop. Direct frozen coarse-to-fine evaluation
can localize pipeline errors without a training extension. Before new training,
the faithful published-reference gap is now explicit in
`e2e_published_reference_gap_20260916.md`; no further ablation queue is launched.

Registered 2026-09-16 before execution. Contract:
`configs/e2e_frozen_controls_v1.json`; implementation:
`workflows/sbi/e2e_frozen_controls.py`. Full E2E remains paused at384.

Both control checkpoints at30720 updates. First registered fitted/transfer field
per ph000/ph002/ph003, two paired draws, both training seeds. DDIM32/128 NFE versus
VP-Heun32/128 NFE; Heun256/512 refinement. Heun uses two forwards per interval.
Provisional reference-resolution screen:256->512 relativeRMS <=1% per trajectory.
Same initial noise and conditioning; no clipping, churn or endpoint change.
Save all144 outputs and hashes, residual spectra/gain/correlation, total-density
quantiles/support. Total density adds TRUE interpolated coarse to physical fine
residual. No posterior calibration or sampled-coarse E2E claim.

216 independent one-step interventions cross retained/zero historical FIRST
moment and norm1 clipping on/off, two seeds, three fitted phases, gradient sigma
.005/.01/.05, and denoising-only / +identity1 / +projected-identity1 objectives.
Preserve second moment, step age, betas, LR and weight decay. Restore original
model/Adam/RNG every time; same gradient/evaluation seeds as preceding54 probes.
Evaluate clean/noisy MSE at three sigmas with two fresh noise draws. Pair against
denoising-only in the SAME factorial setting. Raw Euclidean projection is not a
guarantee through Adam's coordinate-wise preconditioner, even without momentum.
No saved updated weights, held-out access, auxiliary-loss fit or E2E restart.

## Paper-control integration

| Paper/control | Actual status | Staged test |
| --- | --- | --- |
| [EDM official loss/preconditioning](https://github.com/NVlabs/edm/tree/main/training): noise-dependent coefficients, log-normal exposure, log-noise conditioning | ALREADY implemented/tested in the six-arm512-update experiment58361744; see e2e_edm_ablation_20260915.md | sigma_data=1/F=-v loss AND gradients equal existing VP-v under our mapping; do not repeat the wrapper as a novel repair. Noise exposure helped but failed the old capability gate. Earlier reset-all-Adam/clip10/RF controls are distinct from today's first-moment-only factorial. |
| [Min-SNR](https://arxiv.org/html/2303.09556v2) timestep conflicts and weighting | Gradients measured; v-weight algebra-tested, NOT trained | Uniform-v versus gamma5 at matched exposure, all noise regimes; tiny sigma is strongly downweighted by the published v-weight. |
| [CAMELS Appendix B](https://arxiv.org/html/2403.10648v1): schedule, embedding48/24, depth4/2, bottom attention | Explicit recommendations, NOT implemented ablations here | Prioritize time/noise encoding and learned/fixed schedule with matched exposure, then individual depth/attention controls. |
| [3D diffusion-flow](https://arxiv.org/html/2502.17087v1): train-only log/minmax, Huber, PS/PDF/bispectrum | Train-only normalization and synthetic log-Gaussian tested; current panel has spectra/quantiles. Huber/log-cosmic training and new bispectrum tests NOT implemented | Separate Huber/MSE from representation; log requires valid positive full density, not signed residuals. Include unclipped-tail controls. |
| Solver/parameterization separation | Exact-oracle tested; frozen-neural matched-NFE test implemented | This is cosine-VP Heun, NOT an EDM recipe reproduction. No EDM schedule/churn/preconditioning silently adopted. |

The optimizer factorial is our causal diagnostic, not a paper replication. The
papers' projected2D maps and parameter-conditioned3D boxes differ from this
selection-aware observed-galaxy-conditioned posterior. Successful numerics do not
establish correct scores. If resolved frozen samples remain physically wrong,
do not spend another run merely changing sampler steps or stronger identity loss.

## Reproduction

Stage a clean Git archive with e2e_oracle_conflict stage into a new registered
oracle_conflict_* Scratch child. Source and JSON contract are manifest-hashed.
From ROOT/source, activate cosmic_env and use an approved GPU allocation:

```bash
python -m workflows.sbi.e2e_frozen_controls --root ROOT --mode optimizer
python -m workflows.sbi.e2e_frozen_controls --root ROOT --mode sampler
```

Atomic receipts and writer lock preserve partial diagnostics. No automatic retry
or budget extension. Prior failed science gates are not retrospectively relaxed.

## Decision discipline after the user's concern about diagnostic churn

The paper ablations are a menu of conditional hypotheses, NOT a mandatory queue
of further optimizations. Clean-input identity at positive sigma was a debugging
proxy, not the scientific estimand. Do not demand that every proxy be perfect
before inspecting generated fields. If fine-only frozen draws are credible,
the next meaningful test is frozen coarse-to-fine inference with true versus
sampled coarse conditioning, not another auxiliary-loss training run. If the
resolved fine sampler fails, isolate that demonstrated generative failure first.
Neither branch automatically authorizes restarting full E2E training.
