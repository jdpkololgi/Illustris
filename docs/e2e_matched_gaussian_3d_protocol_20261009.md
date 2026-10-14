# Matched Gaussian CFM comparison

This bounded experiment separates errors in the previous learned conditional
means and covariances, then compares learned whole-field and block sampling
with the intended 3D backbone. The user approved both experiments and compute
on 2026-10-09. No Abacus data, production model, or acceptance threshold changes.

## Endpoint interventions

Keep the existing 20000-update structured CFM checkpoint fixed. Cross exact
versus learned conditional mean response (including bias) with exact versus
learned innovation covariance. Evaluate all four combinations on the same
eight held-out observations, forward and reverse block order, with and without
the existing exact global update. Use NFE512 endpoints and analytic stationary
moments. These are diagnostic interventions, not deployable learned models.

## Neural comparison

Use the current CoupledBackbone layers unchanged: base24, levels3, residual
3D convolutions, bottleneck self-attention, two spatial context encoders and
cross-attention. A new Gaussian interface replaces galaxy/coarse inputs with
masked noisy observations, mask, noise standard deviation, exterior state,
exterior mask, coordinates and zero padding. Native and factor2 pooled full
context replace the two Abacus footprints. Position features use domain-relative
cell coordinates. The Gaussian field is unconstrained, so no fine residual
projection or density decoder applies. This tests the intended architecture
family with the real layers, not the unmodified Abacus observation model.

Prior: existing periodic fixed-cell-scale long-range Gaussian (20 percent
spectral covariance plus 80 percent common-mode covariance). Observations:
existing sparse regime1 with inhomogeneous Gaussian noise. Domain sizes8 and16;
fixed8-cell blocks. Whole and block arms have identical parameters and all
observations; block inference also sees the entire current sampled exterior.
Context compression is learned and may be insufficient. It is not an oracle
precision summary, a qualified finite collar, or constant-memory context.

At n8 the arms are identical and share one trained checkpoint after an explicit
paired-training equality test. At n16 train separate whole and block models.
One seed,4096 AdamW updates, batch8 fresh prior/likelihood pairs per update,
cosine learning rate1e-4 to1e-5, weight decay1e-5. The data RNG is separate
from bridge noise and reset identically across arms. Block locations cycle
through all eight locations. Thus field-draw and optimizer budgets match;
block supervision uses one eighth as many target voxels at n16. Wall time,
supervised-voxel count and memory are reported, not treated as equal budgets.
Only final checkpoints determine the outcome; no rescue fits or selection.

## Accuracy and resource criteria

Two fixed held-out observations per size;256 independent realizations each.
Whole-field draws at NFE32 and64 share initial noise. Block chains start at
zero, with snapshots after1,4,8,16 sweeps at NFE32. NFE64 repeats the first
four sweeps with matched innovations. An independent exact-posterior-initialized
one-sweep experiment tests invariance; its oracle initialization is QA only.
No global oracle update is added to the learned 3D comparison.

Use identical predeclared domain, octant, corner, contrast and Fourier probes.
Reference posterior moments come from Gaussian conditioning, with independent
precision checks at n8. Evaluate normalized mean, covariance, marginal90percent
intervals and cross-probe correlations. Independent chains at each snapshot
permit Monte Carlo uncertainty estimates; use conservative four-standard-error
lower bounds to flag errors above the existing1percent mean/variance tolerance.
These lower bounds are diagnostic approximations, not posterior certificates.
Exact independent posterior samples undergo identical finite-ensemble analysis.
Finite ensembles cannot certify the full-dimensional1percent worst-direction
gate or0.5percentage-point coverage gate. Outcomes are failed or inconclusive,
never qualified from noisy point estimates alone. Sweep changes and lag
correlations are mixing diagnostics, not proof of stationarity or reliable ESS.

Benchmark measured training-step and sampling time and peak allocated GPU
memory on8,16,24,32,48 grids at batch1, including full-context encoding. Record
whole draw, block update, and projected sweep cost (block update times block
count; explicitly an estimate). These larger-grid tests are untrained cost
benchmarks, not accuracy results. No time-to-qualified-draw or speedup at matched
accuracy can be claimed unless accuracy is actually established.

## Runtime and provenance

Use one approved GPU interactive allocation with a representative smoke gate.
Scratch remains reserved; continue the previously approved read-only NERSC
pytorch2.8.0 runtime and bounded Home diagnostics exception. Do not modify
cosmic_env or shared software. Record source hashes, runtime, GPU, node, Slurm
ID, seed, checkpoints, timings and engineering receipts. Keep total outputs
below1GiB. Stop on nonfinite training/sampling, parity failure, source mutation,
or an estimated total budget above the allocation. Preserve partial results.
