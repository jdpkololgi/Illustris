# Posterior behaviour at environmental transitions

The saved P12-A posteriors contain useful transition uncertainty that hard labels
discard. Their reported crossing probabilities are broadly reliable for knots
and filaments on the exposed mock sample. However, they do not identify every
true transition, and the knot boundary has a substantial asymmetric failure.
No model, posterior, class threshold, Loa selection or VAC value was changed.

## Evidence and definitions

[Three-page figure PDF](figures/p12a_transition_audit_20260929/transition_audit.pdf)
contains [probability reliability](figures/p12a_transition_audit_20260929/01_probability_reliability.png),
[behaviour through true boundaries](figures/p12a_transition_audit_20260929/02_transition_behaviour.png)
and [interval coverage](figures/p12a_transition_audit_20260929/03_transition_coverage.png).
[Full numerical results](figures/p12a_transition_audit_20260929/RESULTS.json),
[validation](figures/p12a_transition_audit_20260929/VALIDATION.json) and
[artifact hashes](figures/p12a_transition_audit_20260929/ARTIFACT_SHA256.json)
retain provenance and uncertainty intervals.

Eigenvalues are stored ascending, lambda1 <= lambda2 <= lambda3. The fixed
class threshold is 0.2, not zero. In the common descending convention the
smallest eigenvalue is instead called lambda3. Here the three events are:

| Interface | Crossing probability q | Equivalent class probability |
|---|---|---|
| Filament–knot | P(lambda1 > 0.2) | P(knot) |
| Wall–filament | P(lambda2 > 0.2) | P(filament) + P(knot) |
| Void–wall | P(lambda3 > 0.2) | 1 - P(void) |

These are integrated posterior probabilities. The existing hard VAC class is
the largest integrated class probability, not the class of a posterior peak.
Binary side decisions at q=0.5 and four-class argmax are different decisions.

Exactly the same already-exposed ph006 50,000 evaluation galaxies and their
512 saved draws were used. Transform/checkpoint/dataset hashes and cumulative
probability identities were checked. All fractions below use natural weights.
ph006 was a selection/development phase, not a fresh blind test; ph001 untouched.
The 128 cap/superblock bootstrap draws give 16–84% intervals conditional on this
phase and the saved Monte Carlo draws, not independent-phase uncertainty.

## What happens when the posterior reports a transition?

Define a transition candidate using only model output: 0.16 <= q <= 0.84,
so neither side receives less than 16% probability. This approximately matches
the central 68% interval spanning the boundary; discrete Monte Carlo probability
and interpolated quantile endpoints can differ slightly. These are uncertainty
candidates, not proven physically intermediate galaxies.

| Boundary | Candidate rows | Empirical coverage of 68% interval | Of 90% interval | Mean probability in adjacent two classes |
|---|---:|---:|---:|---:|
| Filament–knot | 5,986 | 68.9% | 90.0% | 94.0% |
| Wall–filament | 16,954 | 69.7% | 90.9% | 93.4% |
| Void–wall | 11,441 | 72.4% | 92.1% | 95.6% |

Continuous-eigenvalue intervals therefore generally retain appropriate coverage
in these observable selections, with mild overcoverage for void–wall. An
alternative observable selection, posterior median within 0.05 of the threshold,
gives 68% coverage of 68.0%,70.3%,72.5%, respectively. This is more informative
than checking only overall coverage or hard classification accuracy.

Reliability curves compare each reported q bin with its observed crossing rate
over all galaxies in the selected redshift range. Knot and wall–filament curves
are broadly close to the identity line, but not perfectly calibrated in every
bin. For example, wedge P(knot) around 0.65 corresponds to about0.58 observed
(166 rows). Void–wall shows a more systematic discrepancy: full-range q means
around 0.15,0.25,0.35 correspond to approximately 0.10,0.19,0.29 observed crossing
fractions. Those roughly 5–6 percentage-point residuals survive spatial
resampling. Do not describe every transition probability as calibrated.

## Does it find galaxies truly close to a boundary?

For this diagnostic, select only the two adjacent **true** classes and a true
crossing eigenvalue within +/-0.05 of 0.2. This numerical band is descriptive,
not a physical transition thickness or a new class definition. The +/-0.025
and +/-0.1 sensitivity bands are retained in the results.

| True near-boundary population | Rows | Posterior flags ambiguity | Confidently wrong side | Mean mass in adjacent classes |
|---|---:|---:|---:|---:|
| Filament–knot | 1,770 | 62.1% | 9.7% | 82.0% |
| Wall–filament | 7,439 | 72.4% | 4.1% | 90.4% |
| Void–wall | 7,548 | 69.7% | 4.4% | 92.9% |

Confidently wrong means q<=0.1 on a truly positive side or q>=0.9 on a truly
negative side. Its denominator is the entire true near-boundary cohort.
The ambiguity fraction intervals are 60.7–63.8%,71.7–72.9%,68.7–70.4%; the
confident-error intervals are 8.8–10.5%,3.9–4.4%,4.1–4.8%. These are 16–84%
spatial-bootstrap intervals. Excluding cases within 0.05 of another eigenvalue
threshold changes ambiguity fractions to 63.1%,73.2%,70.7%, so intersecting
class boundaries do not explain the knot result. Narrowing the band to 0.025
also retains the knot disadvantage (63.7% ambiguity versus74.2%,72.3%).

The knot failure is strongly asymmetric. Among 765 true knots with
0.2<lambda1<=0.25, mean reported knot probability is only 0.318, and **22.0% have
P(knot)<=0.1**. The mean posterior-median error for the two-sided near-knot
population is approximately -0.101. These are descriptive truth-conditioned
errors, consistent with shrinkage toward the much more common non-knot
population, but this audit does not isolate a causal explanation.

This does not contradict the reliability curve. A bin of reported low knot
probabilities can be mostly true non-knots and correctly calibrated overall,
while containing a sizeable fraction of the rare true borderline knots.
Probability reliability and recovery of a truth-selected population answer
different questions. Likewise, nominal Bayesian interval coverage is not
guaranteed after selecting a narrow band of the unknown truth. The measured
68% coverage in the true near-boundary cohorts (63.1%,74.5%,73.9%) is useful
descriptively but is not, by itself, a formal posterior-calibration rejection.

## Redshift and interpretation

All four redshift shells and the 0.2–0.3 wedge were checked, not just the pooled
sample. The high-z mock posterior becomes less specific: mean adjacent-pair
mass in posterior-selected candidates falls to 87.0%,84.4%,82.5% at 0.45–0.55.
In that shell the true near-void/wall cohort is also problematic: 250 rows,
51.6% flagged ambiguous and10.8% confidently wrong. Pooled results must not
be taken as uniform performance across redshift. Sparse reliability bins are
reported with counts and intervals; plotted bins require effective count >=30.

The posterior is more useful than a forced label for representing ambiguity,
but does not recover all true transitions. The knot interface remains the
largest pooled weakness, and void–wall probability calibration has a measurable
residual. A large mixed probability is uncertainty about membership; it is not
proof that an individual galaxy sits on a physical boundary. These tests do
not validate connected boundary surfaces or spatially coherent joint field
draws. They also do not establish real-DESI coverage or remove the known mock
population mismatch. The existing provisional release limitations remain.

## Reproduction

Script: `workflows/sbi/p12a_transition_audit.py --output <new-scratch-directory>`.
CPU allocation 59072569; both bounded steps completed. v2 only improves figure
uncertainty bands and labels; all numerical subset results equal v1 exactly.
Row cache: `/pscratch/sd/d/dkololgi/abacus/p12a_transition_audit_20260929_v2/transition_rows.npz`.
Code, source hashes, plots and numerical evidence are retained in Git; the
larger per-row diagnostic cache stays in Scratch. No posterior resampling or
training was performed. Ordered-transform, probability/class identities,
weighted-statistic and probability-bin endpoint sanity checks passed.
