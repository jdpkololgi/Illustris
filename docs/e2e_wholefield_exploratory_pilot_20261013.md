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

Here "oracle" names the fine model's true-coarse input, not an entire cap or
exterior replaced by truth. Oracle density scores isolate the fine factor;
oracle tidal scores still include the explicitly retained reference exterior
and are accompanied by its closure-on-truth diagnostic.

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
velocity draws, retained as a diagnostic concern. No posterior-calibration
qualification is claimed, and none needed a pass to continue. Gaussian job
59721421 completed in 28m21s (0.4725 allocated GPU-hours), exit 0:0.

GPU 59722038 then verifies real batch-two 96x64x64 forward/backward paths,
serialized optimizer restart, finite sampling and positive block-mass decoding.
U-Net has 8,398,387 parameters, transformer 8,939,824. Smoke peak allocations are
2.8242/0.7706 GiB; model restart differences are 5.08e-7/0 (tolerance 2e-6).
Decoder mass relative errors are 4.19e-16/4.10e-16. Thus numerical replay is
tolerance-level for U-Net, not claimed bitwise exact. Both clean and velocity
backward paths are exercised; sampling an endpoint reinterpretation of velocity
weights is deliberately not treated as a model check.

The GPU worker remains bounded to one allocation and never requests another
job. It prepares shared frozen exterior tensors while CPU 59721376 finishes
the training panel, then starts the selected U-Net seeds 17/29. At most two
interactive allocations remain active; no batch submission or scheduler retry.

Reproducibility snapshot: `source_20ee757` under the output root contains the
tracked source archive at commit `20ee757` plus the existing transformer-block
module, whose SHA256 is
`a2d5f434aea9732ccb6e2a50941c9ebd69bef62b3de352c13e1bc9cb66fd9492`.
All nine checkpoint-bound source hashes match the passed smoke receipt, and
all eleven focused tests pass from the snapshot. Running jobs remain on their
bound, unchanged live sources; the snapshot is available for later loading and
evaluation without depending on future worktree edits.

The bounded continuation worker proceeds from the selected fits into 32-draw
physical evaluations at the final 32-epoch checkpoint. It visits development
ph012/013 before the already exposed ph014--017, interleaves caps and model
seeds, and retains both sampled and true-coarse inputs. The first ph012/seed17
case also gets paired NFE128/256 evaluation in both modes. This is386 case/NFE
addresses per architecture (384 primary plus2 refinements), not386 new models.
Two focused controller tests pass. Each allocation stops with saved work before
its wall limit and does not submit a successor job. Intermediate checkpoint
files remain available for the separately reported stability comparison.

## Training checkpoint, 13 October 01:35 UTC

CPU59721376 completed all nineteen data phases in 1h19m35s, exit 0:0.
U-Net seed17 completed 9,984 updates in 1,644.64 seconds; its final fixed-noise
development velocity loss is 0.0135501. This is not a posterior score. The
final checkpoint is `fits/unet_x_17/CHECKPOINT_009984.pt` under the output root.
Seed29 continues on GPU59722038. Transformer seeds17/29 run on GPU59722838;
its bounded worker proceeds into posterior evaluation after both fits finish.
No posterior-calibration result is available at this checkpoint.

By 01:46 UTC transformer seed17 also completed 9,984 updates in 976.69 seconds,
with held-out velocity loss 0.0564073. Transformer seed29 then started on the
same allocation. These loss values are optimization diagnostics, not evidence
that either conditional distribution is calibrated.

The independent saved-score readout checks CRPS through a sorted-ensemble
identity and the joint energy score through pairwise scalar accumulation.
It reports observed denominators, missing panels as missing, development and
exposed-evaluation phases separately, and paired integration refinement.
Three focused readout tests pass. It adds no statistical stop condition or
post-hoc uncertainty adjustment and never labels oracle input as a matched
sampled-reference comparison.

## All fits complete; first posterior cases (02:07 UTC)

All four final 9,984-update checkpoints exist. U-Net seeds17/29 take
1,644.64/1,622.48 seconds, final held-out velocity losses0.0135501/0.0135932.
Transformer seeds17/29 take976.69/964.90 seconds, losses0.0564073/0.0563840.
GPU59722038 completed0:0 in1h21m20s, with all bound source hashes unchanged.
U-Net evaluation now uses GPU59723578; transformer continues evaluation in
GPU59722838. Only two allocations are active. Training completion is not
posterior qualification.

For the first ph012 NGC boundary00/seed17 case with sampled coarse input:

| Model | Density voxel CRPS | Voxel nominal90% coverage | Joint energy score |
| --- | ---: | ---: | ---: |
| U-Net clean endpoint, NFE128 | 0.11962 | 0.82849 | 0.08275 |
| Transformer velocity, NFE128 | 0.16747 | 0.68372 | 0.15848 |
| Shared frozen reference | not recomputed here | not recomputed here | 0.09336 |

These are **one case**, not panel averages. U-Net misses3/6 regional intervals;
transformer misses4/6. The six features and8,192 voxels are correlated, not
independent coverage trials. Interpolated nominal90% intervals from32 draws
also have finite-ensemble coverage bias; retained33-level rank counts accompany
the interval results. Do not equate voxel coverage with joint calibration.

For transformer, paired NFE128/256 summary changes are at most0.000372 posterior
SD with sampled coarse and0.001105 with true coarse. Refinement does not repair
its first-case mismatch. True coarse lowers its absolute density CRPS to0.14899
but narrows spread: voxel coverage0.58728, regional misses5/6. This does not
establish the cause across the panel. U-Net refinement/oracle results are still
pending at this checkpoint.

Both primary first-case evaluations pass corrected-bank zero-replacement parity
and density/tensor-trace consistency. Independent arithmetic verifies the saved
summary scores. The initial readout receipt is`READOUT_20261013T0204.json` in
the output root; it explicitly contains only one completed address. Later
readouts must report their own completed-case denominator. No stop, rescaling,
new training configuration or calibrated-posterior claim follows from these
initial cases; the authorized panel continues.
