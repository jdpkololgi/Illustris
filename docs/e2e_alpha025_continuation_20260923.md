# Alpha=.25: plain continuation, not a new objective

Continue all eight selected-alpha checkpoints from32768to65536updates: seeds
17/29, fixed/amortised, stochastic/exact targets. Restore model, Adam and RNG;
same batch32, LR3e-4, transform, bridge and condition channels. No EMA, new
noise coupling, batch change, alpha reselection or additional architecture.
This is the missing plain-continuation control, not yet a variance-reduction
comparison. Any later such comparison must match compute and estimand explicitly.

Checkpoints49152/65536,512draws/128NFE development curves; final2048draws at
128/256NFE on the original four development observations and eight previously
held-out observations. The latter are now a FIXED longitudinal evaluation panel,
not newly untouched confirmation. Fixed fits evaluate only their fitted case0.
Retain all gates, including DC and every spectral shell. Do not stop at a lucky
checkpoint or average away a failed seed/case.

Primary extra readout: signed DC mean error divided by exact posterior DC sd,
its Monte Carlo standard error, and squared error with ensemble-mean sampling
variance subtracted (retain negatives; do not present clipping as unbiased).
Compare the same statistic on stored32768draws. Retain bridge-aware absorption
and the Gaussian single-real-mode99%null [~.921,~1.082] at2048draws. This null
is not a confidence interval for an arbitrary non-Gaussian learned sampler.

This tests whether more optimization at the selected scale improves the mean
and residual high-k error. It cannot establish permanent convergence or survey
calibration. Exact targets remain privileged. No loosening of gates, physical
units/smoothing invented for the toy, or automatic Abacus/P12 promotion.

Closeout of the previous experiment is recorded before launch. Source, parent,
power and evaluation hashes frozen; same four independent GPU topology;1hour
allocation maximum, at most two allocations. Technical deterministic restart
smoke is separate from unchanged scientific kernels. Fixed tmux launcher and
Slurm-owned work; completion/failure receipt ends the allocation holder.
No autonomous resubmission. Future downstream tolerance work and classical
nonlinear baselines are distinct proposals, not added to this run.
