# Gaussian null audit of the matched comparison

The frozen run's approximate four-standard-error screen narrowly rejected one
exact-posterior control at n=8, observation 0: the variance-error lower estimate
was 0.01105, versus a 0.01 tolerance. Therefore its classification labels are not
treated as calibrated significance statements. The frozen code, outputs and
training budgets remain unchanged.

This post-hoc audit independently recomputes every saved projection statistic
and adds exact Gaussian-null tests. It uses Normal sample-mean and chi-square
sample-variance distributions, Bonferroni correction over at most 28 evaluations,
30 fixed probes, 2 statistics and 2 tails, and familywise alpha=0.01. Null bounds
allow normalized mean deviations up to 0.01 and variance deviations up to 1 percent.
Both tails use alpha divided by 3,360. The mean bound uses the maximum permitted
variance 1.01; variance bounds use 0.99 and 1.01 times the corresponding chi-square
quantiles divided by 255. All snapshots have 256 independent realizations across
chains, even though successive snapshots of each chain are dependent.

This is a disclosed correction to statistical interpretation, not a new
training or checkpoint-selection pass. Rejecting the Gaussian reference law
is evidence against posterior correctness; non-rejection cannot establish
full-dimensional accuracy or exclude non-Gaussian errors. The original screen's
false-positive control remains visible in the final record. Precision-form
FFT actions and conjugate-gradient means independently check the Gaussian
reference, and direct fixed-point iteration checks the endpoint interventions.
