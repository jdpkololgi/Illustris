# Whole field and block CFM on matched Gaussian domains

Whole-field CFM has much smaller mean errors and lower raw sampling cost than
the block chain in this fixed-budget experiment. Blocks reduce GPU memory, but
neither learned method recovers an acceptable posterior. This is not a comparison
of cost per accuracy-qualified independent draw, and it does not qualify an
Abacus model.

The experiment is complete: 14 focused tests and 726 independent numerical,
calculation and provenance checks pass. Scientific accuracy failures remain
failures. This result supersedes the verification checkpoint; it preserves the
original statistical-screen failure and the subsequent correction below.

## Matched setup and scope

The models inherit the actual CoupledBackbone layers: base width 24, three U-Net
levels, residual 3D convolutions, bottleneck self-attention, two spatial context
encoders and cross-attention; 8,398,387 parameters including the fixed schedule.
Gaussian observation channels, domain-relative positions and unconstrained
field output replace the Abacus interfaces. Native and pooled full-domain views
replace the two survey footprints. This is the intended architecture family,
not an unmodified galaxy likelihood, density decoder or complete Abacus model.

The prior combines 20 percent fixed-cell-scale spectral covariance and 80
percent common-mode covariance. Sparse, inhomogeneous Gaussian observations,
data seeds and model initialization match across methods. All observations are
available; block inference additionally receives the current full exterior.
The learned context encoder can compress that information imperfectly. No
analytic posterior or precision summary is given to either trained network.

Each fit uses one seed, 4,096 AdamW updates, batch 8 and 32,768 fresh fields.
At side 8, one block covers the whole domain: the methods are identical and
share one checkpoint, with paired-training equality tested. At side 16, the
whole and block models are separately trained; blocks have fixed side 8.
Field-draw and optimizer budgets match, but the block loss supervises only one
eighth as many target voxels. This is not equal supervision or a convergence
study. Training takes approximately 76, 105 and 76 seconds respectively.

There are two fixed held-out observations per size and 256 independent
realizations per configuration. The 30 fixed probes include the domain mean,
octants, corners, contrasts and Fourier modes. Results apply to this prior,
mask, seed and bounded training budget, not all CFM models.

## Posterior comparison

The primary side-16 comparison uses 32 velocity evaluations per flow solve for
both methods. Mean errors below are normalized by the exact posterior standard
deviation of the corresponding probe, not by the Monte Carlo standard error.

| Quantity across the two held-out observations | Whole-field draw | Block chain after 16 sweeps |
| --- | ---: | ---: |
| Largest absolute probe mean error | 0.96 to 1.10 | 5.19 to 6.06 |
| Range of probe variance divided by exact variance | 0.396 to 1.561 | 0.454 to 2.047 |
| Exact Gaussian reference rejected | Both observations | Both observations |

The block chain's largest errors are negative domain-mean offsets. They are not
stationary-error estimates: the chain has not demonstrated convergence. Late
domain-mean lag-one correlations are approximately 0.78 to 0.79, and no reliable
stationary effective sample size is claimed.

Starting the block sampler from exact posterior draws also fails after just
one learned sweep. Maximum probe mean errors are 1.36 and 1.58 standard
deviations; the corrected null test rejects both cases. Therefore slow mixing
from a poor initialization cannot be the sole explanation. The learned update
does not preserve the target in this test.

Whole-field CFM also fails at side 8, where block assembly is absent. Doubling
evaluations from 32 to 64 does not remove the failures. However, integration is
not certified converged: whole-field probe mean changes reach 0.048 standard
deviations, and the block four-sweep comparison changes by up to 0.249. These
are not passes of a tight refinement criterion.

## Direct resource measurements

One A100 GPU, batch one, 32 velocity evaluations per solve, three timed
repetitions. These are inference-only, untrained architecture benchmarks.
Memory is peak allocated GPU memory, not reserved memory, host memory or total
process memory. Input data are already on the GPU; context encoding is included.

| Grid | Whole draw seconds | Entire block sweep seconds | Whole peak MiB | Block peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 8 cubed | 0.233 | 0.233 | 41.48 | 41.97 |
| 16 cubed | 0.252 | 1.864 | 44.12 | 42.09 |
| 24 cubed | 0.262 | 6.296 | 52.96 | 42.79 |
| 32 cubed | 0.288 | 14.947 | 68.48 | 46.15 |
| 48 cubed | 0.458 | 50.766 | 134.39 | 58.25 |

At side 48, blocks use about 43 percent of the whole-field peak memory, but one
sweep takes about 111 times as long as one whole draw. A sweep contains 216
separate block solves and is not an independent draw. Additional sweeps and
autocorrelation would add cost. Full exterior encoding also means block memory
is not constant as the domain grows. The earlier training-warmed microbenchmark
has a different resident-memory baseline; its peaks must not be mixed into this
inference-only table. Larger grids were not tested for posterior accuracy.

## Conditional mean and covariance interventions

This separate experiment uses the previous side-4 structured Gaussian CFM's
fixed 20,000-update checkpoint, not the newly trained 3D models. Cross exact and
learned endpoint conditional mean mappings, including bias, with exact and
learned innovation covariances. The table shows forward local-chain stationary
worst-direction covariance errors; these are analytic, not Monte Carlo estimates.

| Conditional mean mapping | Innovation covariance | Worst covariance error |
| --- | --- | ---: |
| Exact | Exact | Numerical precision |
| Exact | Learned | 1.49 percent |
| Learned | Exact | 5.80 percent |
| Learned | Learned | 6.01 percent |

Errors in the mean response alter feedback between successive blocks, so they
change the assembled covariance as well as the mean. Replacing innovation
covariance alone therefore does little to repair this checkpoint. An exact
global update reduces the learned/learned worst error to 5.11 percent, still
above the 1 percent gate. All 128 combinations of intervention, observation,
order and global-update choice were independently reconciled.

## Statistical correction and numerical verification

The original approximate four-standard-error screen narrowly rejected one exact
control. Its saved labels are retained but are not calibrated decisions. The
disclosed post-hoc audit instead uses exact Gaussian Normal/chi-square null
distributions with Bonferroni correction: familywise alpha 0.01 over a bound of
28 evaluations, 30 probes, two statistics and two tails. Null bounds allow
normalized mean deviations of 0.01 and variance deviations of 1 percent.
The resulting sample-mean bound is 0.2944 and sample-variance bounds are
0.6422 to 1.4676. All 20 learned evaluations reject; none of the six exact-control
entries rejects. Non-rejection would not certify the full joint law.

Independent FFT precision actions and conjugate-gradient means check the
Gaussian reference. Separate dense row-update assembly and fixed-point iteration
check the structured interventions. The first audit stopped at its 1,000-step
iteration cap: an exact control had residual 1.12e-13 against a 1e-13 tolerance.
Increasing only the cap to 10,000 resolved that verifier limitation. The
tolerance, models, observations and scientific gates were unchanged.

## Evidence and operational closeout

Output root:
`/global/u2/d/dkololgi/diagnostics/matched_gaussian_3d_20261009_v1`.
Primary records are `run/COMPLETE.json`, `mean_covariance/COMPLETE.json`,
`actual_sweep_cost.json` and `VALIDATION.json`. Checkpoints, projection arrays,
source hashes, losses and runtime metadata are retained. The main run took
1,131 seconds and its recorded files occupy about 106 MB.

GPU job 59596481 completed the fits, sampling and costs. CPU job 59597879
completed the repaired independent audit. Both allocations were relinquished.
The previously approved read-only NERSC runtime/Home-output exception was used;
normal environments and shared software were unchanged. The separately approved
graph refresh still timed out during Scratch-dependent startup and remains
unrefreshed.

The immediate implication is to retain whole-field CFM as the simpler reference
while investigating learning accuracy before advancing either method. A longer,
predeclared matched-budget learning study or an analytic-velocity teacher
diagnostic could distinguish optimization/objective difficulties from context
and architecture limitations. Neither has been launched as a rescue fit.
