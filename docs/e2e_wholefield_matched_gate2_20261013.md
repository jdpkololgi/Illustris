# Matched full-cap Gate 2: frozen execution binding

The 13 October request authorizes completing the remaining gated plan. The
192-task reference bank and final assembly pass, with 32 draws for each of
six phases, two seeds and 16 region pairs. This is the newly approved full-cap
law, not the historical pair-local bank. No scientific gate has passed yet.

Run the CPU-only sensitivity prerequisite first. The executable and configuration
are `e2e_wholefield_matched_screen.py` and
`e2e_wholefield_matched_gate2_20261013.json`. Commit both before simulations.
Preserve old artifacts and sealed phases. No model is fitted by this test.

## Full advancement rule, fixed before simulations

The six-dimensional identity-scale vector and corrected operator are unchanged.
Pool Q and V separately with equal phase/seed weights and take Q/V. Q subtracts
finite-draw mean noise V/32. Both model seeds remain nested in each phase.
The minimum point improvement is 10% in absolute distance from R=1; powered
alternatives are distinct, at 20% and 40%. Also require absolute gain at least
the frozen two-seed noise floor: the sample SD of the separately pooled seed
deviations, i.e. their absolute difference divided by sqrt(2). The analogous
fair-energy floor uses the two pooled seed scores. Two seeds do not estimate
population uncertainty; these are fixed empirical reproducibility guardrails.

Select each arm's checkpoint by minimum fair energy on ph012/013 under sampled
coarse only; ties choose earliest. Test the preceding adjacent checkpoint, or
the following one if first is selected. Both checkpoints and both coarse modes
must clear ratio and strict fair-energy improvement margins. All 6^6 whole-phase
bootstrap resamples preserve both seeds and all within-phase dependence. Require
positive lower percentile bounds for the two margins at 0.05/16, accounting for
two arms, two checkpoints, two modes and two metrics. This is a nominal rule,
NOT a calibrated false-positive claim: actual full-rule null calibration is
mandatory before a pass. Coordinatewise 90% interval-content error may not
worsen by >2 percentage points; fair CRPS may not worsen by >2%. Report all
phase/seed values and the ph014--017 pool; neither is an alternative pass route.

## Rejection-only test

The necessary event allows ANY of two arms at ANY of four checkpoints to meet
the pooled sampled-coarse ratio threshold and its fixed seed floor. It drops
the actual selector, adjacent stability, true-coarse requirement, proper score,
bootstrap significance and marginal guardrails. Full advancement implies this
event. Its probability is therefore an upper bound on full-rule power under
each declared synthetic law. It cannot pass Gate 2 or estimate its actual null
false-advancement rate. Null event probabilities are only upper bounds.

There are 4,096 independent repeated six-phase experiments for each of four
laws and three paired-noise correlations (0, 0.9, 1), with five effects
(0, 20, 40, 60, 100%). Simultaneous one-sided Hoeffding bounds use familywise
MC error 0.01 across all 120 finite-draw and exact-moment event probabilities.
Reject if any powered-effect finite-draw upper bound is below 80%. Otherwise
complete full-rule calibration, including calibrated/decorrelated controls,
before any Gaussian or fine-model advancement. No threshold adjustment after
inspecting results.

Covariance is estimated from both seed banks over the entire 96-coordinate
phase vector (16 pairs x six coordinates), with 10% diagonal shrinkage. It is
a nuisance fixture only, never an inference normalization. Predictive traces
and expected squared error traces match every reference phase/seed. Shared
phase errors are perfectly aligned across seeds, a favorable coupling that
does not invent extra independent universes. The Gaussian/common/heavy laws
change the common predictive variance scale to target the *pooled population*
ratio exactly. Biased laws reduce common-direction bias to the same target.
Common-mode transfers 25% of covariance trace to one direction; t5 laws use
independent truth and posterior radial factors, sharing posterior factors for
paired candidate comparisons. Candidate/checkpoint innovations have the original
0.8 same-arm correlation. Both seeds share phase truth and each realization
is shared across all candidate checkpoints. Report surrogate limits explicitly;
these are conditional sensitivity experiments, not measured cosmological power.

## Execution

Use one CPU interactive allocation, at most eight single-thread workers, one
hour maximum, desi account and Scratch license. Existing SSH-independent tmux
ownership remains explicitly approved; the allocation exits when its fixed
commands finish. No batch fallback, new scientific decisions in the controller,
or automatic failure retries. Output root:
`wholefield_comparison_20261009_v1/matched_gate2_20261013_v1`.
The eight-GPU-hour Gaussian subcap remains if Gate 2 passes. Coarse training,
full-chain changes and sealed confirmation remain separately authorized work.

Status: code/configuration binding prepared; simulations not yet run.
