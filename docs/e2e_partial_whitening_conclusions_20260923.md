# Partial whitening: completed scientific conclusions

All40fits and192confirmation ensembles completed. Development selected alpha=.25
using stochastic amortised results only. Neither target law at this selected
alpha qualifies across all eight confirmation observations and two seeds.

| Held-out amortised model | Mean RMS | 16-probe covariance error | DC ratio | Top-shell ratio | Full passes |
|---|---:|---:|---:|---:|---:|
| alpha=.25,32k,stochastic | .14564 | .12795 | 1.16773 | 1.17213 | 0/16 |
| alpha=.25,32k,exact | .08485 | .08324 | 1.04630 | 1.12563 | 0/16 |
| alpha=.50,32k,stochastic | .15093 | .29296 | 1.77983 | .95464 | 0/16 |
| alpha=.50,65k,stochastic | .11176 | .19634 | 1.53865 | .96543 | 0/16 |
| alpha=.50,32k,exact | .11670 | .12018 | 1.26979 | .94310 | 0/16 |
| alpha=.50,65k,exact | .08420 | .09536 | 1.18300 | .96276 | 2/16 |

At alpha=.25, stochastic covariance/variance/coverage pass16/16, while mean and
all-shell power pass0/16. Exact supervision passes all four non-spectral gates
14/16; two cases additionally fail mean accuracy. Thus 'only the top shell
fails' applies to14cells, not the whole panel. Development alpha=.30 exact
fits pass6/8amortised and2/2fixed, but were not selected for confirmation.
Do not retrospectively call them held-out successes or change the selection.

Partial whitening improves joint scale balance; longer full-whitening training
also improves mean and covariance. Neither proves convergence or an irreducible
trade-off. Under stochastic alpha=.25, DC absorption predicts1.12587 versus
observed1.16773; top-shell prediction1.00872 versus observed1.17213. This is
restricted-family diagnostic support, not causal proof for nonlinear U-Nets.
Maximum128/256NFE shell difference0.000933 is much smaller than failures.
The single-mode Gaussian99%power null at2048draws is[.92132,1.08235].

Slurm training/select/confirmation/collect steps58790556.4–.7 all completed0:0.
The enclosing allocation later timed out: its holder had already been cancelled,
so the intended completion-file exit did not release it. Scientific completion
is established by application receipts, not the parent allocation state. The
next fixed launcher writes a failure/success exit receipt as well as COMPLETE.

Programme conclusion: useful capability evidence on a512-voxel Gaussian problem,
not a calibrated DESI posterior. Its exact Gaussian solution is already the
reference, not an ML competitor that has never been considered. Classical
nonlinear baselines, cost/accuracy comparisons and downstream-derived tolerances
are sensible future work, but not added to the authorized plain continuation.
No claim about months/weeks to DESI readiness is supported by these experiments.
