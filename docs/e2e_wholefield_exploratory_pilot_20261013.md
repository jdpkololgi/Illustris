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
