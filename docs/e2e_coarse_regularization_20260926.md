# Next bounded experiment: coarse regularization and classical failure audit

## Synthesis

The training/development gap supports coarse-scale overfitting, not a unique
proof of insufficient phase diversity or conditionally unbiased means. The
classical control is unusable (fails correlation/zero-predictor checks), not a
valid competitor. A fitted voxelwise log-space likelihood plus nonlinear
exponentiation is NOT direct regional least squares: in-sample regional failure
does not logically prove an indexing bug. Audit before replacing or simulating
galaxies with that likelihood. No general verdict on classical inference.

## Frozen experiment

Two seeds17/29, each resumed from original EMA/raw/optimizer/RNG checkpoint13312.
Four workers: baseline AdamW weight_decay1e-5 vs decay0.1. Only decay changes.
0.1 is a fixed exploratory intervention, not selected by development scores;
smaller1e-3 under the remaining small learning rates would barely shrink weights.
Same13312additional updates, original cosine schedule/alpha=.30/EMA=.999,
batch2, noise RNG and phase order. No dropout, capacity or data changes. Parent
states/source preserved. Baseline final parameter difference against original
26624 is recorded as a continuation-replay diagnostic.

Save checkpoints every1664updates from13312to26624. This is a continuation,
not a newly trained dense8k-onward curve. Evaluate19968and26624 automatically;
reuse13312saved draws.32draws/anchor,128NFE, matched addressed seeds. Evaluation
uses8training anchors (two existing anchors each from000/002/020/024) and
16development anchors (four per012-015, all16cap/shell/support strata pooled).
This small training panel is narrower than the earlier52-anchor diagnosis.
Report phase/seed and train/development error/spread, coverage, width and CRPS.
Do not use small-panel percentages to certify tight calibration.

Early-stop reference: candidates13312/19968/26624. Separately for each arm,
select on012/013 and assess014/015, then swap. Rank by maximum core/block
absolute coverage error against29/33, average across selection phases/seeds;
earlier checkpoint breaks ties. CRPS does not select, but CRPS/width are mandatory
guardrail readouts so broad intervals cannot be promoted solely for coverage.
All four phases were previously inspected: these swaps are exploratory
robustness checks, NOT fresh unbiased confirmation. No automatic winner/promotion.

## Classical and hybrid status

Audit existing classical predictions versus zero fluctuation, in-sample and
development, with positive correlation and lower RMSE as basic functional gates.
Check true-wide-to-regional alignment on all52training anchors; inspect local/
wide log-ratio-versus-log-density correlations and conditional residual offsets.
No held-out repair/tuning, and no claimed repair until the cause is identified.
Numerical dense-Gaussian tests passed previously but cannot establish likelihood
adequacy. Full-size audit runs before regularization workers.

Fine model remains frozen: its training condition used TRUE coarse fields
(coarse_source='training_truth'), while inference uses sampled coarse. Freezing
therefore does NOT make a new coarse distribution automatically safe. Any
coarse candidate first passes full-panel coarse assessment, then a bounded
fine-conditioned draw check for power/dependence/support before E2E promotion.

Synthetic lognormal/2LPT pretraining and galaxy resampling are deferred: both
need a verified observation-generation model; naive resampling of averaged
log-channels is not valid Poisson augmentation. Synthetic diversity also changes
the simulator/domain, so it cannot isolate real phase diversity by itself.

## Decision and execution

One4GPU/4h allocation (16GPUh cap), fixed tmux launcher, no automatic renewal.
Estimated~1.2h training +~2.1h evaluation per worker from measured coarse timings.
Source and checkpoints frozen; atomic checkpoints and per-anchor results.
No ph016-019access. If the screen shows benefit in both seeds/swap directions,
evaluate that candidate on full development before freezing a final configuration.
An eventual one-time016-019confirmation requires a separate explicit release
decision, frozen estimands/operator/sampler/selection, and realistic phase-level
uncertainty; it is not authorized by this run.

Output: /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coarse_regularization_20260926_v1.
CLASSICAL_AUDIT.json is separate from model SUMMARY.md/COMPLETE.json. Completion
requires all four workers and both new evaluation checkpoints, not Slurm exit alone.
