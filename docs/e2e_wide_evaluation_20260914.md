# Trained-field evaluation and convergence check

User-authorized September 14: update the science log, generate and evaluate
trained-checkpoint draws, and assess whether training should continue. This
does not authorize new training, modified targets or held-out phase access.

## Fixed diagnostic design

Evaluator source: `204ebc2`, `workflows/sbi/e2e_wide_evaluate.py`.
All checkpoint/data/model/normalization bindings are checked against the passed
smoke and completed training. No training implementation changes are made.

- Use all 96 existing training anchors, both objectives, four independent parent
  sample identifiers each: 768 base draws. Same identifiers pair the objectives.
- Preserve the full shared wide/fine realization in HDF5, with per-file hashes.
  The generator sees only observations; truth is used separately for evaluation.
- Report four-draw posterior-mean eigenvalue errors, pointwise draw dispersion,
  fair marginal CRPS, density errors and physical-support violations. These
  are descriptive training-panel estimates, not held-out coverage/SBC/TARP.
- Retain per-draw eigenvalue/class errors, filling, pair-bin probabilities,
  fixed-axis connectivity and largest-void statistics under both masks.
  Aggregate by phase and shell/support without IID-voxel error bars.
- Evaluate checkpoints 96/144/192 on each same training anchor using four fixed
  time/noise realizations per objective and stage. Compare changes within each
  objective, never CFM loss against DIFF loss.
- On one anchor per phase/shell/support (24, alternating caps), pair draw zero
  at the registered sampler steps with doubled steps: CFM Heun 16->32 and DIFF
  DDIM 32->64. These are evaluation-only overrides, not training/checkpoint
  rebinding or a new sampler selected against truth. All refined arrays are
  retained. A single refinement is a stability check, not proof of the exact
  continuous-time limit.

The four-draw ensemble is too small for precise tail probabilities. Anchors are
correlated within only three training phases. Improvement of an objective is
not proof of condition use, generalization or calibrated uncertainty. Likewise,
a random posterior draw need not resemble the one simulator truth pointwise;
physical density support, training progress and sampler stability answer
different questions.

Diagnostic follow-up, specified while the main evaluation runs: reuse the same
24-anchor panel for paired update-144 versus update-192 draws, and compare their
eigenvalue and thresholded-functional changes with the sampler-refinement
changes. Also compare full local-parent density power in four fixed wavenumber
bands using identical Hann windows and weighted demeaning for generated and
target fields. Check target as well as generated density support. These are
attribution diagnostics, not a blind convergence gate, posterior correction,
additional R7 smoothing or truth-driven sampler selection. No anchors are
selected according to their outcomes. The extra 48 update-144 draws use the
same sample-zero seeds and remain separate from the 192-update ensemble.

## Execution

Job 58305867: nid001057, one shared GPU, 32 logical CPUs, two-hour cap,
shared_interactive/desi_g and Scratch. No other allocations were active at the
pre-launch check. The larger wall-time envelope covers 768 generated fields,
48 doubled-step fields and 4,608 fixed-noise objective evaluations, including
the physical statistics and saved arrays. No batch or additional fit is run.

Outputs:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/eval_20260914_58305867`.
Log: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_eval_58305867.log`.
The original four training fits completed successfully on September 11 and
their group receipt is archived in `docs/evidence/e2e_field_v2/wide_research_20260911/`.

Completed successfully. Slurm 58305867 and both steps are COMPLETED 0:0;
allocation 1h24m12s, evaluation step 1h16m57s, report step 5m22s. The allocation
was released normally. All 768 base, 48 refined and 48 update-144 fields are
saved (864 total); all 4,608 fixed-noise probes completed. The report rechecked
the base HDF5 checksums and original source/data/normalization provenance.
The NERSC allocation skill governed isolated compute execution and release.

Main receipts are `EVALUATION_COMPLETE.json` and `INTERPRETATION_SUMMARY.json`
under the output root. A durable compact summary, their SHA256 values, all 240
HDF5 file hashes, checkpoint bindings and scheduler receipt are archived in
`docs/evidence/e2e_field_v2/wide_eval_20260914/SUMMARY.json`. Full arrays remain
on Scratch and are subject to its retention policy. Report source: `bb2701b`,
`workflows/sbi/e2e_wide_evaluation_report.py`. Five focused evaluator/report
unit tests pass. No training or held-out payload opening occurred.

## Results: current run is not converged

### Optimization and paired-field stability

Each objective is compared only with itself. Probe times/noises and anchors
are fixed across checkpoints; all values below average the same 96 anchors.

| Fit | Loss at 96 | At 144 | At 192 | Last-48-update decrease | Anchors improving late |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFM coarse | 1.24751 | 1.18388 | 1.16592 | 1.52% | 75.0% |
| CFM fine | 1.20906 | 1.16061 | 1.09433 | 5.71% | 100% |
| DIFF coarse | 0.72590 | 0.68969 | 0.63633 | 7.74% | 100% |
| DIFF fine | 0.73263 | 0.65869 | 0.63107 | 4.19% | 92.7% |

All four phase-averaged losses fall in each of ph000/ph002/ph003. CFM coarse
has slowed, but this alone is not a plateau certificate. The other stages
remain visibly in motion after just two training passes.

On the balanced 24-anchor paired-seed panel, median componentwise eigenvalue
RMS changes / mean pointwise four-draw standard deviation are:

| Change | CFM, three ordered components | DIFF, three ordered components |
| --- | --- | --- |
| Checkpoint 144 -> 192 | 0.443, 0.387, 0.314 | 0.446, 0.466, 0.441 |
| Double sampler steps | 0.00180, 0.00154, 0.00159 | 0.0318, 0.0223, 0.0296 |

This strongly separates late weight evolution from numerical sampling error.
It is paired-seed drift, not proof that every changed field became more accurate.
The four-draw standard deviation is a noisy descriptive scale, not a calibrated
posterior standard deviation.

Topology is more sensitive than RMS. Under the observed mask, checkpoint
changes alter fixed-axis connection outcomes in 4/24 CFM and 9/24 DIFF cases;
median largest-void-fraction changes are 5.84 and 13.63 percentage points (pp),
with maxima 24.41 and 25.07 pp. Doubled steps change no observed connection
outcomes, but DIFF's largest-void change reaches 2.01 pp (median 0.553 pp).
CFM's observed maximum is 0.063 pp. Under the complete mask, refinement changes
one DIFF connection and CFM's largest-void change reaches 1.62 pp. Therefore
do not claim exact sampler/topology convergence from a small eigenvalue RMS.

### Generated field quality

These are training-panel descriptive scores, not held-out accuracy or coverage.
Ensemble quantities summarize four draws per anchor; topology/class summaries
are per-draw comparisons with the one simulator truth. Medians below use the
observed mask; complete-mask and phase/shell/support summaries are retained.

| Diagnostic | CFM | DIFF |
| --- | --- | --- |
| Posterior-mean ordered-eigenvalue RMSE | 0.140, 0.149, 0.193 | 0.141, 0.148, 0.179 |
| Fair marginal CRPS, ordered components | 0.0645, 0.0705, 0.0956 | 0.0655, 0.0680, 0.0895 |
| Fraction of generated density below -1 | 1.18% | 1.55% |
| Four-class disagreement with truth | 54.1% | 51.1% |
| Absolute filling-fraction difference | 6.89 pp | 5.74 pp |
| Absolute largest-void-fraction difference | 27.97 pp | 26.10 pp |
| Fixed-axis connection disagreement, all draws | 53.4% | 47.1% |

The target minimum across all local parents is -0.8400, and no target voxel is
below -1. Generated support violations therefore are not inherited from these
reference fields. Worst observed anchor fractions are 3.55% CFM / 3.72% DIFF.
They occur in all phases; shell-0 interior medians are 1.99% / 2.52%. Do not
clip generated density after sampling to conceal this defect.

Matched-window, full-parent density power ratios (mean of four draw powers /
target power, then median across anchors) are:

| k band [h/Mpc] | CFM | DIFF |
| --- | ---: | ---: |
| (0, 0.08] | 0.743 | 0.946 |
| (0.08, 0.16] | 0.496 | 0.623 |
| (0.16, 0.32] | 1.206 | 1.077 |
| >0.32 | 113.27 | 117.41 |

The high-k excess occurs in all three phases. R7 smoothing strongly suppresses
the target denominator at high k, so 113x is not 113x total field variance.
This is a tapered finite-parent diagnostic, not a survey-window-deconvolved
power spectrum or another application of R7. Nevertheless, the combined
intermediate-scale deficit and high-k excess do not reproduce the target shape.
Longer optimization might improve this; these data do not establish that
training alone will cure it.

Sparse shell 3 has worse CRPS than shell 0 for both methods. Shell-3 boundary
largest-void differences reach median 36.0 / 33.6 pp. DIFF has somewhat lower
training scores on several diagnostics, but CFM is slightly better on the
first-component CRPS and support violations. This does not select a production
winner. A posterior draw need not match a single truth realization, so class,
void and pointwise differences alone cannot distinguish posterior uncertainty
from underfitting. The independent support, spectral and optimization evidence
is essential. No unconditioned/permuted-observation baseline was run, so these
scores do not demonstrate use of the observation condition.

## Decision and proposed next gate (not executed)

The 192-update canary is sufficient to demonstrate a functioning, reproducible
training-to-draw pipeline. It is not sufficient for an optimization-convergence,
calibration or science-release claim. Recommend a separately authorized bounded
continuation, initially to 384 updates per stage with a planned checkpoint
assessment. A possible later 768-update ceiling is a new decision, not an
automatic extension. Preserve optimizer/RNG state and matched data order,
targets, normalization and observation-only sampling; first validate the resume
contract rather than silently relaxing the original 192-update cap.

Predeclare what constitutes a plateau across at least two checkpoint intervals:
paired fixed-noise losses, generated-field drift and topology must settle while
physical support and the power-shape discrepancy improve, not merely stop
changing. Register tolerances before the continuation; the present diagnostics
are exploratory, not retrospectively invented pass gates. If losses improve
but field physics does not, stop blind extension and diagnose conditioning,
coarse-to-fine distribution shift and representation/objective limitations.
Any transform or support-aware model revision must be a new matched experiment,
not a post-hoc modification of these samples.

Generalization requires separately approved held-out validation with independent
phase units, selection-aware diagnostics across redshift/support, condition-use
controls and enough draws for coverage/SBC/TARP. Ninety-six overlapping anchors
in only three training phases are not 96 independent realizations, and four
draws do not resolve tails. Do not open the sealed confirmation set for tuning.
Keep `training_ready=false`, `r0_physics_pass=false`, prior negative-domain
receipts and the earlier 6.211-pp truth-only domain/topology caveat unchanged.
The latter is a representation check, not the learned accuracy reported here.
