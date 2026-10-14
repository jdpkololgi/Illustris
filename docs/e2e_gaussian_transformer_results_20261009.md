# Whole field transformer CFM results

The transformer improves worst-probe mean error, probe-covariance discrepancy
and interval-probability error on all four reused development observations at
the fixed training budget. It does not recover an acceptable posterior. Both
architectures reject the exact Gaussian reference at both tested integration
resolutions. This is a promising bounded architecture pilot, not a qualified
field posterior or a general transformer versus U-Net ranking.

## Matched experiment

The new model uses eight global-attention transformer blocks, width 240, six
heads, MLP ratio 4 and nonoverlapping 2 by 2 by 2 voxel tokens. Linear state and
observation embeddings replace the convolutional hierarchy and context encoders.
Time enters through adaptive layer normalization; Fourier features encode token
positions. There are 64 tokens at side 8 and 512 at side 16. All observations
are supplied, but no training truth, exterior state, analytic posterior,
whitening or teacher velocity enters the conditioning. Total parameter counts
are 8,518,568 versus the baseline's 8,398,387, a 1.43 percent difference.

Separate side-8 and side-16 fits use the same Gaussian prior and likelihood,
seed, fresh training-field and bridge-noise streams, full-voxel supervision,
4,096 AdamW updates, batch 8 and learning-rate schedule as the saved whole-field
baseline. Each fit sees 32,768 fields. Only the final checkpoint is evaluated.
Transformer training takes 60.0 and 90.2 seconds; previous U-Net training took
76.0 and 104.8 seconds on another allocation. These training times are not a
same-session controlled timing experiment. Final losses are 0.567/0.596 for
the transformer versus 0.569/0.566 for the U-Net; lower training loss is not
equivalent to better held-out posterior accuracy.

Two fixed observations per size, 256 draws each, the same 30 probes and paired
NFE32/NFE64 sampling are reused. The observations have already informed previous
diagnostics. This is a predeclared new model on a reused development panel, not
a new blind generalization test. Initialization, inductive bias and optimization
suitability are not matched merely by using similar parameter counts.

## Posterior accuracy

Primary comparison at 32 velocity evaluations per solve. Mean errors are in
exact posterior standard deviations of each probe, not Monte Carlo standard
errors. Covariance discrepancy is the relative Frobenius norm of the covariance
error in the same normalized 30-probe space, not a full-field worst-direction
error. Observation labels correspond directly to the retained zero-based cases.

| Grid | Observation | U-Net worst mean error | Transformer worst mean error | U-Net probe covariance discrepancy | Transformer probe covariance discrepancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 cubed | 0 | 0.864 | 0.333 | 0.282 | 0.203 |
| 8 cubed | 1 | 1.342 | 0.825 | 0.454 | 0.341 |
| 16 cubed | 0 | 1.098 | 0.627 | 0.503 | 0.453 |
| 16 cubed | 1 | 0.963 | 0.819 | 0.488 | 0.456 |

Probe variances remain distorted. Across side-16 cases, transformer variance
divided by exact variance ranges from 0.508 to 1.551, compared with 0.396 to
1.561 for the U-Net. Some uncertainty directions remain about half as variable
as required; others have roughly 55 percent excess variance. The corresponding
side-8 ranges are 0.600 to 1.379 and 0.428 to 1.338.

The maximum absolute difference between 0.9 and the exact reference probability
inside each empirical central 90 percent probe interval improves from
12.98/36.86 percentage points to 4.19/16.71 at side 8, and from 28.11/35.35 to
20.24/13.30 at side 16. This is a fixed-observation interval-content diagnostic,
not repeated-data frequentist coverage, and remains noisy with 256 draws.

Improvement is not uniform across individual probes. For side-16 observation 0,
the signed domain-mean error worsens from -0.013 to +0.627 posterior standard
deviations even while the worst error across all probes decreases. Observation 1
improves from -0.893 to +0.023 in that same domain-mean probe. Global attention
has not uniformly solved common-mode calibration.

All 16 learned evaluations (both architectures, two sizes, two observations,
two NFEs) reject the Gaussian reference under the predeclared simultaneous
Normal/chi-square audit. None of the four unique exact controls rejects.
Familywise alpha is 0.01 over a conservative bound of 28 evaluations, 30 probes,
two statistics and two tails, with 0.01 mean and 1 percent variance slack.
The mean bound is 0.2944 and variance bounds are 0.6422 to 1.4676. The legacy
four-SE flags in the shared evaluator are not the controlling statistical test.
Even non-rejection would not certify the full joint posterior.

Doubling NFE does not remove these failures. For the transformer, maximum probe
mean changes across observations are 0.0073/0.0880 at side 8 and 0.0162/0.0749
at side 16. Maximum probe-variance changes are 0.0033/0.0134 and 0.0293/0.0268.
Only one of four transformer cases passes the tight 0.01 mean-and-variance
refinement screen; no U-Net case does. Integration convergence is not established.

## Direct inference costs

Same A100 GPU and process, batch one, NFE32, three repetitions per entry with
conditioning included. These use untrained model instances after the fits and
cleanup, not the learned checkpoints. Peak allocated device memory includes
resident process allocations; it is not reserved memory or total GPU usage.

| Grid | Architecture | Seconds per draw | Peak allocated MiB |
| --- | --- | ---: | ---: |
| 8 cubed | U-Net | 0.2335 | 49.81 |
| 8 cubed | Transformer | 0.1253 | 50.62 |
| 16 cubed | U-Net | 0.2510 | 51.96 |
| 16 cubed | Transformer | 0.1234 | 57.06 |

The side-16 transformer draw takes about half the raw inference time and about
10 percent more peak memory in this paired benchmark. Do not mix these peaks
with the earlier standalone cost process's memory baseline. Training memory is
separate: the transformer's batch-8 peaks are approximately 228 and 791 MiB.
Global attention has different token-count scaling; no larger-domain runtime,
memory or posterior-accuracy result follows. Neither model has an established
cost per accuracy-qualified draw.

## Verification and provenance

All 15 focused tests and 508 independent numerical/provenance checks pass.
Checks cover token geometry, no truth leakage, remote token influence, parameter
count, initialization, gradients, RNG consumption, reproducibility, saved
projection calculations, CDF-versus-quantile null decisions and independent
FFT-precision/conjugate-gradient posterior checks. Hashes bind the old baseline,
new source, fixed protocol, model checkpoints and output files. Engineering
verification does not turn failed scientific gates into passes.

Protocol: `docs/e2e_gaussian_transformer_protocol_20261009.md`.
Source entrypoint: `workflows/sbi/e2e_gaussian_transformer_run.py`.
Output root: `/global/u2/d/dkololgi/diagnostics/gaussian_transformer_20261009_v1`.
Key records: `smoke/SMOKE.json`, `run/COMPLETE.json`, `run/VALIDATION.json`,
`run/cost.json`, training receipts, checkpoints and saved projection arrays.
The main run takes 198 seconds and records about 69.6 MB of files.

Job 59598276 on nid200372 and its smoke/run steps complete with exit 0. The
allocation is relinquished after 5 minutes 15 seconds. The approved temporary
read-only NERSC runtime and bounded Home-output exception are retained;
cosmic_env and shared installations are unchanged. The repository graph tool is
not on the current command path, so its required refresh could not run.

The next useful comparison would test whether the transformer advantage survives
multiple seeds and longer, predeclared matched learning curves, with a stronger
integration-refinement check. It is not evidence to replace a production model,
and no additional fit or Abacus run has been launched.
