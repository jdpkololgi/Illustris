# Wider-coarse/local-fine physics comparison: completed

Run 2026-09-09; result interpretation 2026-09-10. Job 58131992 on nid004156
completed with exit 0:0 in 36m03s. All 96 training anchors, both masks and all
three declared variants were evaluated. The full report checksum matches the
terminal receipt. No learned training or holdout payload access occurred.

**Conclusion:** the wider-coarse/local-fine representation substantially reduces
the finite-domain error. The 1299.072 Mpc/h domain at 13.532 Mpc/h coarse spacing
is a well-supported candidate for a bounded research pipeline. It is not a
strict joint-topology pass, a calibrated posterior, or automatic training
authorization. Historical all-anchor thresholds remain diagnostic, not a blanket
veto on exploratory training.

## Matched comparison

Median per-anchor eigenvalue RMSE divided by that anchor's full-box-truth
standard deviation, expressed as percent; ordered lambda1/lambda2/lambda3.
These are deterministic representation errors, not posterior coverage errors.

| Representation | Observed support (%) | Complete science core (%) |
| --- | --- | --- |
| Local 96-cubed alone | 6.150 / 4.759 / 3.618 | 6.132 / 4.701 / 3.526 |
| 649.536 Mpc/h, factor 4 | 2.058 / 1.652 / 1.276 | 2.028 / 1.579 / 1.274 |
| 1299.072 Mpc/h, factor 4 | 0.817 / 0.643 / 0.489 | 0.826 / 0.646 / 0.486 |
| 1299.072 Mpc/h, factor 2 control | 0.808 / 0.637 / 0.484 | 0.816 / 0.640 / 0.477 |
| Full-box-low oracle + local high | 0.681 / 0.542 / 0.405 | 0.683 / 0.525 / 0.416 |

The large factor-4 domain improves all three eigenvalue RMSEs on every one of
the 96 anchors under both masks. Paired median improvements on observed support
are 7.41x / 7.64x / 7.99x. The improvement repeats in each cosmological phase;
observed shell/support-stratum median errors range from 0.707--0.962%,
0.580--0.712%, and 0.453--0.587% respectively. Anchors and voxels are correlated;
the three phases are the experimental units, not millions of independent voxels.

## Science summaries must remain separate

For the large factor-4 domain, all 96 anchors and both masks satisfy the old
eigenvalue, class, filling-fraction, pair-probability and fixed-terminal
connection limits. Maximum observed class disagreement is 0.554%, filling
change 0.0582 percentage points, and pair-probability change 0.000418. No tested
axis-connection event changes. This does not imply arbitrary connectivity or
all topology is accurate.

The one remaining old-threshold failure in both masks is largest-void fraction
at `ph003_NGC_s3_interior_01`. The observed fraction changes from 0.181564 to
0.243672: **6.211 percentage points**, despite only 0.252% class disagreement
at that anchor. The full-core change is 6.171 points. This outlier must not be
hidden by median eigenvalue accuracy or removed from the panel.

The 649.536 Mpc/h variant retains seven failing observed anchors and six complete
cores under the old limits, with worst observed void error 12.013 points.
The old local-96 calculation has 95 failing anchors and worst observed void
error 17.435 points. At the specific remaining large-domain outlier, however,
the old local calculation's void error was only 0.310 points: wider domains
do not improve every nonlinear statistic on every anchor.

## Resolution and attribution

Halving coarse spacing at fixed large extent improves per-anchor eigenvalue
RMSE by only about 1.0--1.1% at the median. It leaves the same void outlier
(6.195 observed percentage points). Thus finer coarse resolution is not the
missing remedy for that event, and the cheaper factor-4 variant is a reasonable
pilot candidate. This is a target/operator resolution test, not proof that a
learned coarse model will recover equally informative conditioning at both grids.

The full-box-low oracle reduces the outlier's observed void error to 0.0491
percentage points. That supports a contribution from residual finite-wide-domain
tides, while not proving a unique causal decomposition of thresholded topology.
The oracle has no old-threshold failures on observed support, but one different
complete-core void failure of 1.205 points at `ph003_NGC_s2_boundary_00`.
It uses simulation truth and is not an inference input or eligible candidate.

The exact tensor attribution retained all cross terms. Median observed squared
Frobenius energies for wide-low error fall from 5.399e-5 (small extent) to
3.389e-6 (large extent), while local-high error remains 5.674e-6. These marginal
medians are not an additive total error budget; consult each anchor's full
Gram matrix. Maximum algebraic closure error is 9.24e-16. Maximum combined
tensor-trace residual is 2.359e-7. Density reconstruction, Fourier interpolation
and response-count conservation checks also passed as previously recorded.

## What this enables next

**September-10 follow-up:** the loader/normalizer, matched coarse/fine models,
trainer, checkpoint continuation, ancestral sampler and training diagnostics
are now implemented. See the [pipeline contract](e2e_wide_pipeline_20260910.md).
The list below records the work identified at physics closeout; full-size
compute verification, fitting and held-out calibration are still pending.

This completes the authorized truth-only experiment; no rerun is needed.
It supports proceeding to finish a bounded research training pipeline around
the large factor-4 representation, with the void limitation explicit. It does
not establish that a neural network can infer its coarse target from sparse
galaxies, nor any conditional calibration or posterior-science release.

Remaining implementation/contract work is concrete:

1. Freeze the pilot's large-domain data interface, target transforms and
   training-only normalization. The current products are raw and cover the
   training anchor panel; wider selection/confirmation products are not built.
2. Complete the shared coarse/fine conditional trainer and synchronized sampler:
   one coarse realization must govern its fine children, including overlap,
   crop-invariance and exact-resume checks. Do not revert to independent patches.
3. Freeze matched model/compute budgets and a small learnability/calibration
   canary with claim-appropriate criteria. Keep accurate-topology claims separate
   from permission to explore. Obtain explicit authorization for GPU jobs and
   any additional phase payload access.

No trainer or new job was launched in this closeout. Do not change the frozen
`training_ready=false` or `r0_physics_pass=false` receipts into passes. The
original 64/96-domain negative result also remains intact.

## Evidence

- [Run and product contract](e2e_field_wide_coarse_20260909.md)
- [Archived completion and matched summaries](evidence/e2e_field_v2/wide_coarse_20260909/PHYSICS_COMPLETION_SUMMARY.json)
- Full report: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_coarse_20260909/PHYSICS_TEST_REPORT.json`
- SHA256: `ea4196eda37470445d3e003e0708d43d8845c5715affc06f9b6a283e40da574c`
