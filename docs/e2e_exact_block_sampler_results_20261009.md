# Exact Gaussian block sampler results

The exact block-conditional sampler preserves the intended joint Gaussian
posterior in all 48 registered configurations. Adding a conditional global-mode
move improves mixing substantially in the sparse long-range stress tests.
This supplies a constructive alternative to the patch-marginal canvas failure,
but depends on an exact global precision matrix unavailable to the learned
Abacus sampler. No cosmological posterior is newly qualified.

## Correctness and convergence

All 18 targets and 48 sampler/order configurations in the prospective protocol
completed without target selection or gate changes. Seven focused tests passed,
including covariance-versus-precision conditional identities, actual decoder
mass preservation, independent transition composition, and a deliberately wrong
independent-marginal negative control. The largest one-sweep relative covariance
invariance residual was 6.92e-15; the largest mean residual was 4.94e-16, both
below the frozen 1e-8 tolerance. Every sweep transition had spectral radius below
one. These numerical checks accompany the analytic Gaussian conditional-update
identity; innovation covariance was accumulated from actual update noise, not
defined by subtracting from the desired stationary covariance.

All configurations passed the registered finite-budget criteria by 64 sweeps
or earlier: worst-direction whitened mean contraction <=0.01, relative covariance
error <=0.01, and analytic target probability within each nominal 90% probe
interval within 0.005 of 0.90. The table reports the FIRST passing checkpoint
among 1, 2, 4, ..., 512, not an exact minimum burn-in time. Coverage is posterior
probability against an analytic target, not repeated-realization cosmological
coverage. There is one seeded observation realization per target.

On the n4 spectral/template0 target, actual sampled sweeps from stationary and
zero initial states passed for both arms: 16,384 independent draws per check,
maximum mean discrepancy 2.41 standard errors and covariance discrepancy 3.56,
below the prespecified seven-standard-error bound. Independent receipt, gate,
panel and source-hash reconciliation passed 1,376 checks; it is not a second
implementation of the numerical sampler.

## Mixing benefit and limits

Here tau is the largest stationary integrated autocorrelation time among the
registered probes. Smaller is better; effective samples per sweep for that
probe are approximately 1/tau. Comparisons below use forward sweeps.

| Target | Passing sweeps local / global extension | Worst tau local / global extension |
|---|---:|---:|
| n4 long-range sparse observations | 64 / 8 | 16.08 / 1.30 |
| n8 long-range sparse observations | 64 / 16 | 12.94 / 1.77 |
| n8 spectral sparse observations | 16 / 16 | 3.37 / 2.02 |
| n8 constrained long-range sparse observations | 32 / 32 | 7.10 / 2.54 |
| n8 constrained spectral sparse observations | 32 / 32 | 5.59 / 2.56 |
| Independent controls | 1 / 1 | 1.00 / 1.00 |

The n8 long-range/sparse worst-probe asymptotic variance per sweep falls by
about 7.3-fold. Measured sweep time multiplied by worst tau falls from 7.60 ms
to 1.06 ms in this tiny dense test. These timings exclude kernel construction
and analytic QA and do not demonstrate survey-scale performance. The additional
move costs time on already-independent controls. It need not improve every
probe, and the constrained examples show that better stationary sampling does
not necessarily improve the worst-direction burn-in checkpoint.

The constrained targets use 504 independent coordinates for the n8 factor4
block-zero-mean subspace. The largest measured physical block mean after repeated
updates was 1.39e-16. Global directions are projected into that subspace; the
constant direction vanishes. This demonstrates compatibility with the existing
fine-field constraint and decoder, not correction of uncertainty in coarse
block masses. The synthetic prior in that subspace is B-transpose C B; it is not
an assertion that projecting a prior equals conditioning the full cosmological
field on observed coarse masses.

## Matched comparison with the existing canvas

The unchanged actual canvas with all-observation patch-marginal oracles was
rerun at n4, side3, NFE512 on the same targets. Its relative joint-covariance
errors are 25.9% and 38.8% for spectral standard/sparse observations, and 31.7%
and 49.1% for long-range standard/sparse observations. Corresponding domain-mean
90% interval probabilities are 85.0%, 79.1%, 81.1%, and 76.5%. Full-domain
controls have covariance error <=3.81e-5 and essentially 90% probability;
independent side3 controls also pass at finite integration accuracy.

The exact block sampler uses the CURRENT full exterior state through the global
precision matrix. The canvas oracle instead receives exact patch marginals
given the observations. These are different information contracts: the result
does not imply that rearranging existing marginal-network calls implements an
exact Gibbs sampler. A deterministic check also constructs two positive-definite
joint covariances with identical available side3 patch marginals and opposite
unseen-corner covariance (+0.6 versus -0.6). Patch marginals alone therefore do
not determine the joint law in general.

The warranted next design question is how to supply a compatible exterior
conditional and a global update for the learned field. This experiment does
not establish bounded-halo sufficiency, neural conditional compatibility,
non-Gaussian mixing, physical resolution convergence, or Abacus calibration.
No production sampler change, training or sealed-phase access was performed.

## Runtime and reproducibility

The user explicitly approved a temporary exception during the Scratch outage.
The normal cosmic_env and shared installations were not modified. The read-only
NERSC pytorch/2.8.0 module supplied Python 3.12.11, NumPy 2.1.2, SciPy 1.16.2,
PyTorch 2.8.0+cu129 and h5py 3.14.0. This was CPU-only work. The scientific
protocol was amended prospectively only for runtime and output location; source
hashes and runtime versions are recorded in every target receipt.

CPU allocation 59593723 on nid200211 declared U2 rather than Scratch and was
released after 3m47s. Campaign step .1 and validation step .2 completed 0:0;
the numerical campaign took 27.08 seconds. A subsequent read-only display
command (.3) used a misspelled receipt key and exited 1; corrected display .4
completed 0:0. That error did not affect tests, campaign files or validation.
Shell syntax and git diff whitespace checks pass. Required graph maintenance
could not finish before its 15-second timeout in the Scratch-hosted executable;
no graph/global-index refresh is claimed.

Output root:
`/global/u2/d/dkololgi/diagnostics/exact_block_sampler_20261009_temporary_v1`.
COMPLETE.json SHA256:
`64bd1ce86d03650360f91d018bdcd562dbb0abe9131cef04dd7192dc6c8eb736`.
VALIDATION.json binds that receipt and the validator source. The original
Scratch output path and old numerical implementations were not overwritten.
All new source and documentation remain uncommitted; unrelated edits remain
preserved.
