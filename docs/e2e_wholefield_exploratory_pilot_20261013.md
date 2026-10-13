# Exploratory train-and-test field pilot

User authority, 13 October: proceed with two interactive jobs toward a working,
approximately calibrated field model. Sensitivity is no longer a prerequisite.
Keep the failed sensitivity result as a limitation, not a stop on exploratory
training. No additional statistical prerequisite campaign. Essential numerical,
physical, leakage and scheduler checks remain. Keep coarse frozen; no ph018/019.

Candidates: whole-domain U-Net and transformer, each direct velocity or clean
endpoint converted to velocity. A bounded four-arm Gaussian diagnostic informs
parameterization choice, not permission to train. Retain at most two Abacus fine
configurations and seeds 17/29. This is not eight new Abacus fits.

Initial jobs: one CPU data-preparation allocation and one GPU diagnostic
allocation; never more than two pending/running interactive jobs. Use existing
explicit SSH-independent tmux ownership. No batch submissions. Preserve original
sources and outputs and write all generated data/checkpoints to Scratch.

## Common Abacus data

Fine domain: 96 x 64 x 64 cells at the existing 6.766 Mpc/h sampling, centered
on each original pair midpoint. This contains eight old 64 x 48 x 48 generation
windows at their existing 32 x 16 x 16 stride. No new resolution or smoothing.
Conditioning: the same twelve fine observation channels, now over the expanded
domain, plus the same 48-cubed wide observation and coarse fields. Preserve
physical positions, nonperiodic observation boundaries and positive block-mass
decoder. No targets enter observation products or inference readers.

Training: exactly the original thirteen phases. For each cap select 24 of its
64 geometry-defined training pairs using evenly spaced sorted addresses. This
gives 624 domains; all arms use identical addresses and normalized channels.
624*(96*64*64) == 1664*(64*48*48): one new epoch matches the old target voxel
exposure. Target 32 epochs, checkpoints 4/8/16/32 epochs, with the 24-epoch
checkpoint available for a neighboring-checkpoint comparison. A training domain
does not add an independent phase. Prepared data retain the existing train-only
normalizer and loss spectrum policy. The overlapping central target/observation
arrays must agree with the existing products before fitting.

Development and evaluation: all sixteen original pairs per phase, ph012--017.
Checkpoint selection uses ph012/013 only. Report ph014--017 separately and do
not call them fresh confirmation. Evaluate true-coarse and shared frozen-model
sampled-coarse conditioning, using matched region identities and decoder. The
new finite-domain law differs from full-cap generation; exterior/boundary
conditioning must be disclosed in comparisons with the completed reference.

Readouts: learning curves, sample plausibility, positive mass conservation,
regional density and tidal uncertainty, proper scores, coverage, and paired
NFE refinement. Report improvements and regressions without requiring a pass.
Approximately calibrated means uncertainty agrees reasonably with tested truths
on the reported support; neither low training loss nor attractive fields proves
it. Do not silently apply a post-hoc summary recalibration or redefine the target.

Output root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1`.

## Implementation and first checks

Both backbones use the expanded fine observation and unchanged wide observation
and coarse fields. U-Net keeps the previous base-24 three-level architecture
with rectangular bottleneck attention. Transformer uses eight width-240,
six-head blocks and nonoverlapping four-cell patches; all 6,144 fine and 1,728
wide tokens interact. Forward computation is BF16; loss, parameters and AdamW
state remain float32. The common physical bridge, projected block-zero residual,
positive mass decoder, alpha=.3 train-only spectral weighting, batch two,
AdamW cosine 1e-4 to 1e-5, clipping .5 and EMA .999 are retained. All runs have
9,984 updates and the same addressed data/noise order. No architecture or
posterior-calibration conclusion follows from matching voxel exposure alone.

Eleven focused tests pass (data selection/parity geometry, rectangular tokens,
owned-core/collar coordinates, noise replay, mass reconstruction, score formulas
and access guards). Full-shape GPU smoke and numerical restart replay also pass;
these are engineering checks, not a statistical gate. Checkpoint files include model, EMA, optimizer, loss weights
and CPU/CUDA RNG; the bridge RNG is addressed by seed/update. Development
velocity loss is recorded separately and is not called posterior calibration.

Primary tidal evaluation uses a shared completed reference exterior: retain its
full-cap uncorrected tensor, replace the 96x64x64 local density, apply the exact
free-space tensor of the density difference, and recompute the same conservative
linear reconstruction correction. Zero replacement must reproduce the bank's
corrected core summaries. This preserves the operator and exterior draw, not a
jointly reconditioned full-cap posterior. The unchanged exterior may retain
incompatible reference correlations; report this modular approximation. Also
record the operator evaluated on true local density under that exterior, so
closure errors remain visible. A local-plus-wide free-space operator is available
only as a separately labelled diagnostic; never pool it with full-cap scores.

For sampled conditioning, read the actual shared frozen coarse draw from the
completed bank, with physical-coordinate crop checks. Oracle conditioning reads
true coarse density explicitly; sampled inference does not open targets. Score
against original fullbox truths. Report density/shear CRPS, ranks, 90% coverage,
posterior widths and normalized errors; refinements use identical noise. The
sampled baseline is not a matched oracle-coarse baseline and is not labelled as
one. No post-hoc rescaling of fields or uncertainties.

## Gaussian advisory result and GPU smoke

The 4,096-update, three-seed comparison completed all twelve fits and their
128-draw paired NFE64/128 readouts. Mean diagnostic scores (max standardized
probe-mean error plus relative covariance error, averaged over observations
and seeds) are:

| Backbone | Velocity | Clean endpoint | Selected |
| --- | ---: | ---: | --- |
| U-Net | 1.734833 | 1.700022 | clean endpoint |
| Transformer | 1.403997 | 1.501201 | velocity |

The U-Net difference is small and is not an architecture qualification. Clean
endpoint draws show materially greater paired integration sensitivity than
velocity draws, retained as a diagnostic concern. No candidate passes a
calibration screen and none needed a pass to continue. Gaussian job59721421
completed in 28m21s (0.4725 allocated GPU-hours), exit0:0.

GPU59722038 then verifies real batch-two 96x64x64 forward/backward paths,
serialized optimizer restart, finite sampling and positive block-mass decoding.
U-Net has8,398,387 parameters, transformer8,939,824. Smoke peak allocations are
2.8242/0.7706 GiB; model restart differences are5.08e-7/0 (tolerance2e-6).
Decoder mass relative errors are4.19e-16/4.10e-16. Thus numerical replay is
tolerance-level for U-Net, not claimed bitwise exact. Both clean and velocity
backward paths are exercised; sampling an endpoint reinterpretation of velocity
weights is deliberately not treated as a model check.

The GPU worker remains bounded to one allocation and never requests another
job. It prepares shared frozen exterior tensors while CPU59721376 finishes
the training panel, then starts the selected U-Net seeds17/29. At most two
interactive allocations remain active; no batch submission or scheduler retry.
