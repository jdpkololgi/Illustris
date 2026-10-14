# Whole-field transformer CFM: frozen pilot protocol

User requests transformer internals for the whole-field CFM comparison and
approves one shared-interactive A100 GPU for at most one hour, then release.
This is a synthetic Gaussian architecture test, not Abacus retraining or a
production change. Existing experiments and checkpoints remain unchanged.

## Architecture and matched quantities

Replace the convolutional U-Net/context encoders with a whole-domain transformer:
eight pre-normalized blocks, width240, six heads, MLP ratio4, nonoverlapping
2x2x2 voxel tokens, global noncausal attention in every block. All voxel values
are retained by patchification. Raw observation/mask/noise tokens enter through
a learned linear embedding; domain-relative Fourier positions and time AdaLN-zero
condition the blocks. Linear patch readout plus the original state residual
retains the baseline zero-head velocity v=z at initialization. No convolutional
feature hierarchy, truth/exterior inputs, analytic posterior, whitening, global
posterior summaries or oracle teacher enters the network. Parameter count must
be within3percent of the old8,398,387. This matches information and parameter
scale, not identical representation, initialization, optimizer suitability or FLOPs.

Use the exact existing Gaussian prior, observation masks/noise, training
generator, independent bridge RNG, whole-field CFM loss and Heun integrator.
Two separate fits at domain sides8 and16. Seed61009492,4096updates,batch8,
32768fresh fields per fit, AdamW lr1e-4->1e-5 cosine,weight decay1e-5,FP32,
TF32off. Identical field/bridge streams and supervised-voxel counts to the
previous whole-field model. No early stopping, checkpoint choice, rescue fit,
training extension or hyperparameter sweep. Smoke weights are discarded.

Compare against saved, hash-verified whole-field U-Net fits from
`/global/u2/d/dkololgi/diagnostics/matched_gaussian_3d_20261009_v1`.
The target and evaluation generator are unchanged. Reuse exactly two held-out
observations per size,256independent draws,30fixed probes,NFE32primary and
NFE64paired refinement. These observations were already used in earlier work;
this is a predeclared new architecture test on a reused development panel,
not a fresh blind generalization assessment. One seed is not an architecture ranking.

## Interpretation and checks

Use the corrected Gaussian Normal/chi-square null audit, declared before these
fits: familywise alpha0.01, bound28evaluations,30probes,two statistics,two tails,
with0.01normalized-mean and1percent variance slack. Include both architectures
at bothNFEs and four unique exact controls (20evaluations); use the same
conservative bounds as the previous comparison. The shared evaluator's older
four-SE labels are historical diagnostics only and do not control conclusions.
Non-rejection is inconclusive, never a full-joint-law certificate. Any claim
that one architecture is better is descriptive on this panel, not a significance
test of the difference or general model selection result.

Report normalized mean errors, probe variance ratios/covariance discrepancies,
empirical90percent intervals and NFE refinement. Tight probe refinement requires
maximum mean and variance changes<=0.01; this is not full-field ODE convergence.
Independently recompute metrics from saved projections, cross-check Gaussian
conditioning with FFT precision/CG, reconcile controls across architectures,
check quantile-vs-CDF decisions and verify source/artifact hashes.

Direct same-GPU untrained inference-only costs at sides8 and16,batch1,NFE32,
three repetitions, full conditioning included. Report both architectures'
wall times and peak allocated device memory, separately from training memory
and batch32 scientific evaluation. No larger-grid scaling study or speed at
matched posterior accuracy is inferred.

## Runtime gates

Scratch is currently reserved. Continue the previously approved bounded Home
diagnostic/read-only NERSC pytorch2.8.0 exception; do not modify cosmic_env or
shared installations. Output root
`/global/u2/d/dkololgi/diagnostics/gaussian_transformer_20261009_v1`.
Keep outputs below1GiB. Use fail-if-exists files and source hashes; no overwrite
of prior fits. Smoke tests include geometry, no leakage, remote dependence,
initial-flow equivalence, parameter count, gradients and reproducibility/RNG
consumption. Representative GPU smoke must estimate complete run+audit<2400s
and training peak<30GiB before launch. Stop on nonfinite output, failed checks,
source drift or inadequate remaining allocation time; preserve partial evidence.
