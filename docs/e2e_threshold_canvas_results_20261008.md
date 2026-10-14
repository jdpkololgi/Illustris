# True-coarse, threshold and exact-assembly results — 2026-10-08

## Bottom line

The matched true-coarse diagnostic is complete: the interior fraction defect
persists with true coarse. Threshold choice substantially changes the magnitude,
especially in ph013, but does not explain away every tested discrepancy. An exact
Gaussian test establishes that the actual overlap-averaged canvas need not recover
the global posterior even with perfect patch oracles. None uniquely identifies
the dominant Abacus mechanism, nor establishes a resolution ceiling.

All results are development-only on the existing panels. No training, model or
threshold selection, new Abacus fields, sealed phases, Loa changes or gate changes.
The prospective diagnostic contract is `e2e_threshold_canvas_diagnostics_20261008.md`.

## Matched true coarse

All32 four-draw batches validate; the unchanged registered summary completes and
independent label/fraction/scalar reconciliation passes96checks. R is adjusted
squared mean error Q divided by predictive variance V, summed over the first
three class fractions in each of two cores and averaged over eight pairs.
It is NOT an empirical coverage percentage. Conditional-MC ranges resample whole
draws, not galaxies, regions or phases.

| Corrected interior | A R (5–95% MC) | B true-coarse R (5–95% MC) | Q A → B | V A → B |
|---|---|---|---|---|
| ph012 | 2.6958 (2.3573–3.3541) | 3.1462 (2.6378–4.1349) | .050107 → .027899 | .018587 → .008868 |
| ph013 | 2.1147 (1.8132–2.7252) | 2.1483 (1.8036–2.8441) | .071526 → .049400 | .033824 → .022995 |

The frozen branch is `coarse_factor_not_implicated`, meaning **the defect persists
with true coarse**, not that coarse inference has no effect. Absolute adjusted
error falls44.3%/30.9%, while variance falls52.3%/32.0%. Common-bias removal leaves
B ratios2.8795/2.1062. Boundary ratios fall .7501→.5258 and1.0216→.4204: neither
uniform broadening nor one global diagnosis follows. Original-operator interior
B ratios3.0828/2.4284 also remain high.

Changed-reference sensitivities remain separate from the frozen rule. B against
the ideal all-true padded cuboid gives1.8281/1.4504; against mean true-covered /
original-exterior labels gives2.4876/1.6295. These are changed targets, not causal
fractions of error or replacement acceptance criteria.

## Threshold experiment

Reclassified saved parent eigenvalues AND physical truth at every predeclared
threshold; all32pairs,32draws, two operators and both support strata retained.
The0.2 result passes32exact moment/bias/variance reconciliations. Primary corrected
interior R values:

| Threshold | ph012 | ph013 |
|---|---:|---:|
| 0.00 | 1.718 | 0.795 |
| 0.10 | 2.386 | 1.553 |
| 0.15 | 2.030 | 1.218 |
| 0.20 | 2.696 | 2.115 |
| 0.25 | 2.179 | 1.248 |
| 0.30 | 2.749 | 0.884 |
| 0.40 | 2.254 | 0.930 |

Thus0.2 makes ph013 look substantially worse than several nearby definitions.
ph012 remains above1 across the sweep; even its lowest conditional-MC lower
bound is1.452 at threshold0. This is sensitivity, not independent model selection
or a population significance claim. No threshold is promoted.

At0.2, corrected interior galaxy-weighted four-class Brier scores improve from
.3826→.2012 (ph012) and .3537→.1984 (ph013) with true coarse. Mean posterior-draw /
truth class disagreement improves35.96%→19.19% and34.49%→18.92%. Brier is a proper
predictive score, not a pure calibration statistic; posterior draws need not match
the realized truth even for an exact posterior. The improvement also uses extra
truth information and is NOT a deployable model improvement.

No-neural ideal closure changes1.63%/1.33% of interior galaxy labels;812/820 and
826/838 of these changes lie within0.025 of a physical-truth eigenvalue crossing.
Only17.46%/17.21% of galaxy weight lies in that margin. Production-exterior closure
has similarly concentrated flips. This specifically supports threshold amplification
of operator error; it does not interpret all posterior uncertainty as model error.

Regional and galaxy-weighted findings must not be conflated. At0.2, the highest
redshift shell supplies85.38%/93.56% of A's adjusted regional squared error and
96.17%/98.42% of B's. The smallest interior cores contain11 and5galaxies; one
galaxy changes a fraction by9.1 or20percentage points. These cores remain in the
registered equal-pair statistic. Count/redshift effects are confounded on this
small panel; this is descriptive leverage accounting, not causal attribution.

B/closure products save only0.2labels. Their margin/Brier diagnostics use matched
physical-truth eigenvalues, but no multi-threshold B/closure results are claimed.

## Exact Gaussian canvas experiment

Unmodified `e2e_global_canvas.integrate`, float64 repository ExactModel,4^3domain,
two masked heteroscedastic observation templates; independent and correlated
priors. Exact affine propagation eliminates Monte Carlo error. Full-window and
independent-cell controls have maximum covariance relative error7.63e-6 atNFE512.
All608 analytic-oracle/control checks pass;72settings are retained.

For correlated fields, even patch marginal oracles conditioned on ALL observations
fail when they see only the local evolving state:

| Patch side | Covariance relative error (two masks) | Domain-mean variance / exact | Actual probability inside nominal90% Gaussian interval |
|---|---|---|---|
| 2 | .2955 / .2874 | .7476 / .7006 | 84.50% / 83.14% |
| 3 | .2587 / .2631 | .7673 / .7175 | 85.04% / 83.65% |
| 4 (whole domain) | 6.69e-6 / 7.63e-6 | .999998 / .999996 | 90.00% / 90.00% |

At side3, voxel-variance errors are only1.50%/2.78%, while full covariance errors
are25.87%/26.31%. Half-domain cross covariance is .001031 vs exact .001563 for
mask0, and .001853 vs .004671 for mask1. Posterior means remain exact to numerical
precision in this all-observations control. This isolates a global-distribution
problem from missing observation information and from neural approximation.
Changing NFE128→512 changes side3 covariance relative errors by only6.6e-6/2.2e-5.

Local-observation patch oracles introduce another difference: at side3 normalized
mean errors are .2416/.2203 and domain-mean coverage85.83%/82.52%. These oracle
posteriors genuinely condition on less information; do not call that difference
a learned-model error. Each oracle also matches its own standalone analytic flow.

This is a structural counterexample, NOT a measured share of Abacus error. The
small periodic correlated prior, stride-one cubes, coarse-stage identity projector,
and two observation realizations do not reproduce the fine mass-constrained
hierarchy or real survey geometry. Window enlargement helps and the full-domain
limit succeeds: the result does not imply that patch-based methods are irreparable.

## Scientific interpretation and convergence discipline

Only one local density halving,6.766→3.383 Mpc/h, was tested on four selected
pairs; its73–75% width-deficit recovery is a direction, not saturation. Testing
collar16→32 is a domain test, not a second resolution. A proper next sequence
would retain physical smoothing/target and refine again, separating source
representation, interpolation, finite exterior and inference changes. Stop only
against a predeclared numerical tolerance or a resource bound, not an unexplained
nonzero residual. No such new compute is authorized by this diagnostic.

Likewise finite padding, finite NFE, a small augmentation grid, two checkpoint
seeds, a truncated mixture screen and simple bias/spread corrections constrain
their tested interventions, not entire explanation families. The new exact test
establishes a failure mechanism and an NFE plateau for this toy, not its Abacus
dominance or a patch-context convergence ceiling.

T-web is a threshold- and smoothing-dependent operational definition. The original
paper explicitly explores both, rather than identifying a unique physical boundary:
[Forero-Romero et al.2009](https://arxiv.org/abs/0809.4135).
An exact continuous joint posterior has an exact pushforward for any fixed
threshold; discontinuity can amplify approximation/target mismatch but cannot
by itself invalidate exact posterior calibration. Discrete finite-ensemble coverage
must still use the appropriate attainable/randomized reference. A better overall
assessment separates continuous density/tensor/eigenvalue inference, operator
closure, spatial dependence, and derived class sensitivity. Existing continuous
regional-density/eigenvalue failures prevent attributing all problems to classes.

The Loa BGS property–environment relationship is to be measured. It must not be
used as an expected relationship to tune the model, threshold or tolerances.
No acceptance rule has been changed here.

## Execution and provenance

User-approved CPU interactive59553598, nid200209,64logical CPUs, one-hour bound;
all science completed in the allocation and it was released early. Isolated
cosmic_env, no GPU. Unchanged summary/final check step0 and new diagnostics step1
exit0:0. Nine existing plus two new focused tests pass. Step2 tests pass before
`graphify update .` fails with filesystem `flock` errno524; global graph refresh
therefore does not run. Scientific artifacts are unaffected; no lock bypass.

Run entrypoints: `e2e_true_coarse_fraction_summary.py`,
`e2e_true_coarse_fraction_final_check.py`, `e2e_exact_canvas_diagnostic.py`,
`e2e_threshold_diagnostic.py` under `workflows/sbi/`.

Receipts under `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/`:

- `true_coarse_fraction_diagnostic_20261008_v1/COMPLETE.json`, `FINAL_CHECK.json`
  and `summary_59553598.log`, `final_check_59553598.log`;
- `threshold_canvas_diagnostics_20261008_v1/GAUSSIAN_COMPLETE.json` and
  `THRESHOLD_COMPLETE.json`, binding sources, plan, inputs and allocation.

The full threshold receipt includes all boundary/original-operator rows, margin
bins, per-case weights/errors/variances and conditional-MC ranges, not just the
interior summaries displayed here. Completion `passed` is engineering validity;
`posterior_qualified=false` remains explicit.
