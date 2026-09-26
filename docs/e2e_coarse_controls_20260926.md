# Coarse posterior train/generalization and classical controls

User authorizes both comparisons. No neural fitting, objective changes, fine
sampling, or new phase opening. Train13phases unchanged; development012-015;
016-019remain sealed. Checkpoints13312/26624, EMA, seeds17/29 fixed.

## Neural diagnostic

Choose four geometry00 anchors per training phase deterministically from the
sixteen cap x shell x boundary/interior strata, rotating the selection by phase
index.52anchors total; all16strata represented.32draws/anchor/factor,128NFE,
same addressed sampler as development.6656new coarse draws. Reuse saved
13312/26624 regional results for all64development anchors: no resampling.
Core16^3 and block24^3 masses align with coarse cells; fine zero-sum residuals
cannot change them. A closure test checks this, including arbitrary residuals.

Report per phase, observational regime and equal-stratum standardized train/
development results. Raw equal-phase summaries are descriptive and may have
slightly different stratum mixtures; standardized results must also be used.
Retain ranks, bias, centered RMSE, RMS spread, width, coverage and fair CRPS.
Subtract variance/32 from mean-estimation MSE as a finite-draw diagnostic, not
from coverage or the reported raw RMSE. No independent-voxel confidence bounds.
An increasing train/development gap supports overfitting but does not uniquely
prove its mechanism. Small pooled bias can hide conditional bias.

## Classical control: Gaussianized coarse Wiener posterior

Train-only52-anchor fit: log-density prior mean and radial DCT spectrum;
free log-count-ratio intercept/response separately for local/wide encodings;
empirical residual variance by fixed exposure bins. Sparse bins use pooled
training variance, explicitly counted. No held-out parameter fit.

Uses the same observation footprint and prepared local/wide log-ratio,
support and exposure fields. Local information is pooled to coarse cells and
REPLACES the overlapping wide likelihood, never counted independently twice.
This is footprint-matched, not full neural-information equivalence: other
channels and subcell detail are discarded. No claim of an exact Poisson
likelihood: averaging precedes inverse log transforms in these products.

Log-density Gaussian prior gives positive density draws by exponentiation.
48^3 latent observed domain padded eight cells on all sides; no observation
likelihood in padding, reflecting DCT boundary rather than periodic wrap.
Prior spectrum interpolated in physical frequency from train-only spectrum.
Matrix-free CG constrained realizations, tolerance1e-7, first case per shard
checked against1e-8 with identical noise. Dense8-variable Gaussian oracle tests
mean/covariance; zero-response test checks masking. The empirical diagonal
likelihood ignores correlated galaxy residuals: passing numerical tests does
not validate that physical approximation. Score32draws on all116train/dev
anchors. Classical train scores are in-sample, explicitly not cross-validated.

This is a classical COARSE control, not yet a classical+neural fine hybrid.
Do not couple it to the fine network until its coarse calibration and chart
compatibility are established. Do not repair held-out coverage by tuning noise.

## Execution

One four-GPU/four-hour interactive allocation under fixed tmux launcher.
Measured neural coarse cost suggests~9GPUh, plus CPU fit/classical sampling.
First full-size classical case is a fail-closed smoke before neural panel.
Atomic per-anchor summaries/draws; completed cases resumable, no automatic
renewal. If four hours insufficient report partial status, no false completion.
Output: /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coarse_controls_20260926_v1.
Automatic COMPLETE.json/SUMMARY.md require all52training neural cases per
seed/checkpoint plus116classical cases and matched development truth hashes.
