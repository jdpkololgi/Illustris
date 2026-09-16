# Frozen neural sampler and optimizer controls

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
