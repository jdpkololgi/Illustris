# Shared-state CFM calibration on training-unseen mocks

## Results: pooled calibration largely retained, not global qualification

All 32 cases completed with 32 draws per sampler. The frozen shared-state
sampler retains the original model's pooled marginal calibration closely.
There is no large new deterioration from coupling in this panel, but regional
joint calibration remains phase-dependent. This is encouraging evidence for
continuing the coupling approach, not proof of a calibrated survey-wide
posterior and not a demonstrated need to retrain merely to remove seams.

Central-interval coverage, in percent. At M=32 the attainable reference for
the nominal90% interval is **87.8788%**, not90%.

| Quantity | Original local | Independent ownership tiles | Shared state |
| --- | ---: | ---: | ---: |
| Owned-cell density | 88.70 | 88.70 | 88.70 |
| Core mass | 85.94 | 85.94 | 85.94 |
| Block-region mass | 89.84 | 86.33 | 91.02 |
| Fine-region mass | 87.11 | 87.11 | 87.11 |
| Seam-slab density | 88.83 | 88.77 | 88.85 |
| Individual seam-side masses | 85.94 | 82.81 | 85.94 |
| Seam-average mass | 90.63 | 93.75 | 90.63 |
| Seam mass contrast | 87.50 | 87.50 | 87.50 |
| Tidal eigenvalues | 88.35 | Not evaluated | 88.28 |
| Eigenvalue gaps | 87.07 | Not evaluated | 87.13 |

Original and ownership-tiled core statistics coincide by construction: the
ownership boundary is outside those original cores. The independent tiled
field is a comparator for seam-crossing statistics, not a second independent
replication of original-core calibration. Tidal quantities use the same
finite-context operator on original/shared left windows, not an invented
tidal field from inconsistent independent coarse windows.

Relative to original local sampling, shared-state fair CRPS changes by +0.18%
for density, +0.24% for tidal eigenvalues, +1.31% for core mass and +1.13% for
fine-region mass (positive is worse). These are small descriptive changes,
not established equivalence or statistically significant regressions. Pooled
seam energy/variogram scores improve by2.15%/2.55% relative to independent
ownership tiles. Both scores improve against that comparator in each phase,
but with only two phases this is not a significance claim. Against the
original single-window joint field, seam energy is nearly unchanged.

The seam-contrast interval is18.51% narrower than independent tiling while
retaining the same87.5% observed central coverage. This is useful evidence
against simple variance inflation as the explanation for its coverage; it is
not a demonstration of correct conditional covariance everywhere.

### Joint regional checks and heterogeneity

TARP empirical coverage at credibility90%, using reference sd0.25. Unlike the
central intervals above, the randomized TARP null is **90%**.

| Joint summary | Original ph016 | Shared ph016 | Original ph017 | Shared ph017 |
| --- | ---: | ---: | ---: | ---: |
| Core masses | 87.70 | 88.48 | 82.62 | 81.54 |
| Block-region masses | 88.87 | 88.57 | 91.60 | 91.31 |
| Fine-region masses | 88.87 | 90.63 | 77.34 | 76.95 |
| Seam-side masses | 87.50 | 87.50 | 86.82 | 86.52 |

Increasing the predeclared reference width to1.0 changes fine-region TARP90
to91.80%→92.09% in ph016 and82.42%→82.23% in ph017 (original→shared).
Thus the magnitude is reference-sensitive, but the ph017 warning persists
under both widths and is not newly introduced by shared-state sampling.
References do not add independent simulations; no iid confidence bands or
hypothesis-test p-values are reported.

Shared-state central core/fine coverage is90.63%/89.84% in ph016 versus
81.25%/84.38% in ph017. In the highest redshift shell, pooled core/fine
coverage is75%/81.25% (only eight cases). These diagnostics prevent a blanket
calibration claim despite favourable pooled density/eigenvalue coverage.
The corresponding original high-shell values were81.25%/84.38%; the shared
sampler has a net one additional core and two additional fine-region interval
misses there. This small-panel worsening must not be hidden by the pooled
agreement, but it does not establish a statistically significant regression.
The full shell, cap, boundary and actual seam-support splits are retained
in `summary32/SUMMARY.json` rather than selecting only favourable subsets.

## Result receipts

Under the Scratch root below, `samples32/RUN_COMPLETE.json` confirms the
complete panel; `scores32/SCORES.json` contains per-case metrics and
`summary32/SUMMARY.json` contains all group summaries, rank histograms,
central coverage levels, and TARP curves at both reference widths.
Post-run precision/refinement and independent arithmetic verification passed
as recorded below. Weights, original atlas, website,
production VAC and sealed ph018/ph019 remain unchanged.

### Completed verification

- `numerical_checks/CHECKS.json`: original independent coarse/fine integrators
  exactly reproduce the established sampler for identical real-model noise
  tensors. Four fixed-case, two-draw NFE128→256 checks have maximum RMSE0.351%
  of the relevant32draw posterior spread. StrictFP32 versusTF32 has maximum
  RMSE0.860%. These are representative numerical checks, not full-panel
  strictFP32 or high-NFE reruns and not evidence of posterior calibration.
- `VERIFIED.json`:384 independently computed central90% coverage values
  match exactly. Direct all-pairs core-mass CRPS agrees to4.86e-17. Pooled
  coverage and TARP curves reconcile exactly with equal-weight phase means.
  Full-panel shared overlap disagreement is0; maximum relative block-mass
  error is4.78e-16. Main case generation totals3998.92s (66.65min).
- `scores32/CONTROLS.json`: M32 calibrated Gaussian control coverage87.88%
  versus attainable87.8788%; biased and underdispersed controls give55.73%
  and37.41%. The correlated-vector shuffle worsens the fair variogram score
  from0.1230to0.2075. These synthetic results validate diagnostic sensitivity,
  not the Abacus sampler.
- Seven new focused tests and three existing TARP/region tests pass;
  `git diff --check` passes. Graphify refresh remains unavailable because the
  command is not installed. No scientific claims depend on its index.

Verified summary SHA256:
`0668cb49eda05852f9baf571ad4ecd5532423b072005b60835b19ecfcaf31539`.

### Decision

Retain shared-state coupling as a promising exploratory candidate. This panel
does not demonstrate that retraining is required just to maintain calibration
while obtaining overlap coherence. It also does not qualify a full-survey
posterior: only two exposed training-unseen phases, one model seed, two
x-adjacent windows per case, and selected regional summaries were evaluated.
Phase/high-redshift regional calibration and larger connected-domain/junction
behaviour remain qualification gates. No further training or production
sampling was launched as part of this evaluation.

## Pre-result protocol, 2026-10-01

User requests coverage/calibration of the new shared-state sampler. Keep the
seed17 coarse13312/fine26624 weights and train-only normalization frozen. No
retraining, model selection, Loa output replacement, or phase018/019 access.
Use ph016/ph017: unseen in training, already exposed to development evaluation,
not fresh sealed confirmation. Every existing cap x four redshift shells x
interior/boundary geometry row is included: 16 cases per phase, 32 total.
The four mock shells span z=0.15–0.55; this is not coverage of the entire Loa
redshift range. Aggregate values equally weight the registered cases, not
individual survey galaxies. The seam average and left–right contrast will
also be reported separately, not only pooled as a two-component statistic.
Error/spread diagnostics retain both raw mean-squared error and the finite-M
correction MSE minus posterior variance/M. The corrected squared quantity is
not silently clipped; a negative estimate has no reported square-root ratio.

Each existing case is the left window. Add a right window shifted +64 raw x
cells (+216.512 Mpc/h), with conditions rebuilt from that phase's existing
count/response caches. Verify left-condition replay before generation. The
old held-out context-augmentation guard remains unchanged: this is a separately
registered evaluation geometry, not training augmentation. Inputs contain no
matter truth. All target access is guarded explicitly to ph016/ph017.

The existing left truth spans 64 fine cells. The two ownership domains meet at
left x=48; thus x40:56 straddles the NEW tile ownership seam while lying wholly
inside existing verified truth. It is unnecessary to rebuild a native2048 FFT
to test this region. No claim will be made about truth outside the existing
left target, about multi-axis junctions, or full-survey long-range calibration.

## Comparisons and measurements

Generate two ensembles with identical frozen networks/NFE128 and fixed draw IDs:

1. Original independently integrated windows with independent physical noise.
2. One physically addressed coarse/fine state shared through every Heun step,
   averaging overlapping velocities, as in the feasibility probe.

Save original left-window and independent ownership-stitched density statistics
separately. The stitched baseline takes left x<48 and right x>=16, restricted to
the original truth extent x<64. Shared-state predictions coincide on overlap.
This distinguishes ordinary original-window calibration from atlas-seam effects.

Use the existing finite-ensemble central order-statistic intervals and fair
CRPS at nominal50/68/90/95%, reporting their actual attainable coverage. Report
rank histograms, interval widths, bias and error/spread. Reuse original owned
core and regional mass definitions. Add two adjacent seam-side regional means
(x40:48 and48:56,y16:32,z16:32), their sum/difference, and fixed paired spatial
probes, with fair energy/variogram scores. Compute original-core tidal
eigenvalue/gap coverage and class Brier with the SAME established local+wide
operator for both samplers; this is not a test of a new global exterior solver.

Primary pilot:32 draws for all32cases. A larger draw count is a follow-on,
not selected after seeing which case or method looks better. Include per-phase,
cap, redshift and support-stratum results. Cases/probes/voxels within one phase
are correlated: do not attach voxel-binomial confidence intervals. With only
two independent phases, phase-level differences and ranges are descriptive;
tight calibration/equivalence claims are unavailable.

Repeat the existing summary-space TARP diagnostic for core/block/fine regional
vectors and add the two seam-side means:64 random references per case,
N(0,0.25^2 I), seed71, randomized finite-ensemble ranks; reference-width1.0
sensitivity is predeclared. Use the unchanged `tarp_curves` helper from the
September27 calibration figures. These references are not extra simulations,
and these stratified anchors are not a formal prior-predictive SBC ensemble.
Reference: [Lemos et al.2023](https://proceedings.mlr.press/v202/lemos23a.html).
The seam-probe permutation control here shuffles total draws and diagnoses
total cross-seam dependence; it does NOT isolate fine dependence at fixed
coarse field. That finer attribution is outside this frozen-sampler comparison.

Run oracle, biased and underdispersed synthetic metric controls and shuffled
cross-seam dependence controls before interpreting measured scores. Such
controls test diagnostics, not the Abacus posterior. Check coherent overlap,
positive density and block mass on actual draws. Benchmark a fixed first case
before deciding whether the full panel fits the approved compute allocation;
pause for a measured additional budget if necessary, without cherry-picking
an incomplete panel. Retain partial results as partial, never as completed.

## Authority and execution history

The user approved up to one2h CPU and one2h sharedGPU allocation. CPU preparation
used59160616/nid200012. At preparation time no calibration results existed.
Sources are the existing cartesian_v2 geometry, conditions,
targets and frozen pilot checkpoint receipts. Scientific authority is
SCIENCE_LOG.md and the original CFM protocol, with later exposure records
superseding its old ph014–019 all-sealed wording.

Preparation completed for all32cases. Every left condition passed all24channel
replay checks, physically overlapping conditions agreed, and truth and input
products are hash-bound. Seven focused tests passed on59160616. The factored
sampler reproduced both saved four-draw NGC shared-state density windows
exactly (float32 saved representation). GPU59160719/nid200345 measured15.87s
for a four-draw case including both samplers and tidal summaries, implying
about68min for the fixed32case x32draw panel. This fits the approved2h GPU
allocation. These launch timings preceded the completed results above.

Scratch root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/global_mock_calibration_20261001_v1`.
`inputs/PREPARED.json`, `smoke/PARITY.json`, and `samples32/BINDING.json`
record inputs, exact replay and frozen runtime configuration respectively.
Implementation: `e2e_global_mock_prepare.py`, `e2e_global_sampler.py`, and
`e2e_global_mock_evaluate.py` under `workflows/sbi/`.

The smoke scoring path completed successfully (one case/four draws, explicitly
not the science panel). All three pre-existing regional/TARP regression tests
passed, in addition to the seven new tests. CPU allocation59160616 completed
in13m19s and was released. Final lightweight scoring reused the sharedGPU
allocation after generation; no idle CPU allocation was retained. GPU59160719
completed in1h18m27s and was released. Both allocations are schedulerCOMPLETED.

Numerical controls after generation are fixed to two matched draws for
ph016NGCs0interior, ph016SGCs0boundary, ph017NGCs3boundary and
ph017SGCs3interior. Compare NFE256 versus128 and strictFP32 versusTF32,
normalizing errors to each case's32draw density/regional/seam spread. The1%
threshold is an engineering diagnostic, not a statistical calibration gate.
Also check exact original independent-factor sampler parity with identical
initial tensors. These controls read observations and generated draws only.
