# Partial whitening and convergence: registered follow-up

Prior panel:32fits/160precision ensembles completed; no full-gate passes.
Partial whitening is a hypothesis, not an established optimum or a cure.
Opus's proposed alpha~0.30 uses already-examined Gaussian posterior structure:
therefore the old four cases are DEVELOPMENT, not fresh confirmation evidence.
The saved progression uses512draws at128NFE (not128draws).

## Comparisons

1. Fresh32fits: alpha in{.20,.25,.30,.35}, fixed case0/amortised,
   stochastic/exact targets, seeds17/29. Same U-Net, Adam3e-4, batch32 and
   32768updates as the prior panel. u=P_train^-alpha x; white unit base in u,
   matching exact teacher and inverse sampling. Reuse the prior-only8192-field
   power estimate unchanged. No posterior moments define the transform.
2. Continue ALL eight alpha=.5 white-bridge fits from32768 to65536updates,
   restoring optimizer and random streams. No other change. Check49152/65536.
   Compare fresh partial fits to OLD32768full-whitening endpoints; the65536
   fits test learning progress, not a compute-matched alpha advantage.

Development endpoints:2048draws,128/256NFE. Curves:512draws,128NFE.
All physical-coordinate gates remain unchanged, including DC. Report mean,
16-probe covariance, variance, exact octant-interval coverage, every power shell,
bridge-correct absorption and the99%single-real-mode chi-square sampling null.
Coverage here is NOT population TARP. Two seeds are limited replication.

## Selection and sealed confirmation

Choose ONE partial alpha using ONLY stochastic AMORTISED development results
pooled over both seeds and all four old observations. Minimize the worst
gate-normalized error (mean/.1, covariance/.15, |variance-1|/.1,
|coverage-.9|/.05, max_shell|power-1|/.1); tie-break average error then alpha.
Exact-target results do not choose alpha. A selected alpha is not a passing one.
Write SELECTION.json BEFORE any confirmation evaluation.

Confirm selected alpha and full whitening at BOTH32768and65536 on new cases4–11
of the deterministic12-case Gaussian reference. These share the same two
mask/noise templates: no unseen-operator generalization claim. Evaluate both
seeds and both target laws,2048draws at128/256NFE. Do not reselect after opening
confirmation results. Fixed fits are capability controls, not transfer models.

## Predictions and limits

Partial whitening should improve DC and high-k together if variance dynamic
range is a major cause. Continued full whitening should improve DC if the
earlier two-point trend reflects incomplete learning. Neither is guaranteed.
Fourier shell-averaged variances are not posterior eigenvalues under a mask;
the proposed scale explanation is not a demonstrated universal mechanism.
Absorption is a restricted-family diagnostic, not causal proof for a U-Net.
Do not infer a DESI-ready preconditioner from this Gaussian problem: nonlinear
selection, tracer stochasticity and operator shifts remain untested.

## Execution

Source/config/power/parent hashes frozen; no old output overwritten. Unit and
restart smoke tests precede fitting. Four independent GPU workers, checkpoint
every1024updates, fixed tmux/Slurm launcher, maximum2h30allocation. Previous
measured32-fit panel took~60minutes; this adds eight continuations and sealed
confirmation. At most two allocations; leave other project jobs untouched.
No Abacus, P12, VDM or nonlinear training. Commit design, implementation and
launch/closeout receipts incrementally.
