# Authorized wide-field continuation to 384 updates

The user approved the recommended bounded continuation and its diagnostics:
"Begin additional training. Complete your recommendations" (September 14).
This is four continued CFM/DIFF coarse/fine fits from update 192 to update 384,
not fresh starts, open-ended training, held-out access or production release.

## Frozen before launch

`configs/e2e_wide_continuation_20260914.json` is the separate extension contract.
The original config and frozen training/data/model source files remain intact.
New checkpoints retain the original binding plus a continuation binding with
the parent checkpoint/receipt hashes, extension source/config hashes and exact
budget. No broad relaxation of the original 192-update launcher is made.

`workflows/sbi/e2e_wide_continue.py` verifies full model/optimizer/RNG restoration,
history continuity, parent receipt hashes and the passed original smoke binding.
Before full continuation, all four actual 192-update checkpoints must pass exact
two-update versus one-plus-one serialization/resume parity, including optimizer,
RNG and history. Those engineering branches are separate from the fits. The
fits restart from their original update-192 checkpoints, not the smoke branches.
The same learning rate, AdamW settings, clipping, batch size, deterministic
per-epoch order, loss definitions, normalization, targets and observation-only
sampling are retained. Checkpoints remain immutable at 16-update intervals.

Repeat evaluation on all 96 training anchors, four same-seed draws per method,
and the unchanged 24-anchor balanced refinement/paired-checkpoint panel. Probe
updates 288, 336 and 384 with the same four fixed noise/time addresses as before;
compare against archived 192-update diagnostics. Generate the paired 336-update
fields for late-weight drift, doubled-step 384 fields for numerical sensitivity,
and retain both masks and phase/shell/support breakdowns. No extra smoothing,
clipping or truth-selected sampler is licensed.

## Predeclared diagnostic gates

These are exploratory stopping/diagnosis tolerances, not calibrated science
release gates. They are fixed before continuation outcomes are observed.

- Absolute relative objective-loss change <=1% in each of 288->336 and 336->384,
  for every method/stage and each training-phase mean.
- Paired 336->384 median eigenvalue RMS change <=0.1 of pointwise draw standard
  deviation in every component; median largest-void-fraction change <=1 pp;
  no changed fixed-axis connection outcomes on the 24-anchor panel. Apply
  topology checks under both masks. Four draws give only a noisy dispersion
  estimate, so this is a diagnostic, not proof of a limiting posterior.
- Median generated density fraction below -1 <=0.1% under both masks, and
  median matched-window parent power ratio in [0.8,1.2] in every registered
  wavenumber band, separately for both methods. Report absolute power context
  where available: high-k ratios have a small R7-suppressed target denominator.
- Stop at 384 regardless. If objectives improve but physics fails, diagnose
  field representation, condition use and coarse-to-fine distribution shift;
  do not automatically train to 768 or change outputs post hoc. Passing these
  gates still does not establish held-out generalization or calibration.

Held-out phases and the sealed confirmation set remain closed. Condition-use
controls are limited to failure attribution on the same 24 training anchors:
fixed-noise matched versus shuffled observations (partner matched by phase,
shell and support but opposite cap), and fine-stage true-coarse versus saved
generated-coarse conditioning. These are stress tests, not causal information
gain or calibration claims; fine shuffled controls retain the true coarse field.
The matched loss must reproduce the primary evaluation before a control is
accepted. These controls run only if a diagnostic gate fails. No model revision
is automatically licensed. Prior negative-domain
receipts and `training_ready=false` / `r0_physics_pass=false` stay unchanged.

## Execution and results

Launched source commit `8431bfd` as job 58309454 on nid001037 after nine focused
unit tests and unchanged original smoke preflight passed. One GPU/32 logical
CPUs under shared_interactive/desi_g, Scratch license, two-hour cap. Prior measured
training plus evaluation/report totals about 100 minutes. The NERSC allocation
skill requires the allocation-reuse check, isolated cosmic_env and prompt release.
No compatible allocation was present at preflight. Record terminal receipts and
actual results here before claiming completion.

Training root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/continue_20260914_58309454`.
Evaluation root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/eval384_20260914_58309454`.
Logs in `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/`:
`wide_continue_58309454.log`, `wide_eval384_58309454.log`,
`wide_report384_58309454.log`, `wide_assess384_58309454.log`.
The fixed sequential chain fails closed and releases the allocation after
success or failure; no automatic retry, additional training or adaptive job
submission is enabled.

Training completed successfully in 842.2 seconds, with all four update-384
checkpoint hashes verified and unchanged first-192 histories. All actual-state
resume parity checks passed before fitting. Training completion and smoke
receipts are archived in `docs/evidence/e2e_field_v2/wide_continuation_20260914/`.
Mean losses over passes 1/2/3/4 through the 96 training anchors:

| Fit | Pass 1 | Pass 2 | Pass 3 | Pass 4 |
| --- | ---: | ---: | ---: | ---: |
| CFM coarse | 1.49667 | 1.25583 | 1.12502 | 1.01183 |
| CFM fine | 1.51108 | 1.16929 | 1.03845 | 0.84800 |
| DIFF coarse | 0.84741 | 0.74732 | 0.64968 | 0.53762 |
| DIFF fine | 0.86891 | 0.69265 | 0.59242 | 0.53143 |

These use varying noise/times and are progress, not the registered fixed-probe
plateau assessment.

The entire chain completed 0:0 and released job 58309454 normally. Slurm elapsed:
allocation 1h40m01s, step 1h39m19s. All 864 evaluation fields, 4,608 fixed-noise
probes, and the 24-anchor conditioning checks completed; the latter have 96
method/stage records and 960 matched/control forward probes. Every matched
control reproduces its primary fixed-noise evaluation baseline. Full source,
checkpoint and payload bindings revalidate. No retry or extra fit was needed.

Durable evidence under `docs/evidence/e2e_field_v2/wide_continuation_20260914/`:
started/completed training receipts, exact resume-smoke receipt, full conditioning
assessment, and `SUMMARY.json` with both-run comparisons, report hashes, all 240
draw-file hashes and scheduler status. Arrays and plots remain on Scratch and
are subject to its retention policy. Nine focused unit tests pass; the local
and global code graphs were refreshed after the source changes.

## Completed assessment: useful improvement, not convergence

All 20 predeclared diagnostic checks fail. These are correlated exploratory
checks, not 20 independent hypothesis tests or held-out release/calibration
tests. The failures must be interpreted together with the improvements below.

### Fixed-noise optimization

| Fit | Loss at 288 | At 336 | At 384 | Last-48-update decrease |
| --- | ---: | ---: | ---: | ---: |
| CFM coarse | 1.03928 | 1.03027 | 0.91240 | 11.44% |
| CFM fine | 0.94002 | 0.86725 | 0.79860 | 7.92% |
| DIFF coarse | 0.55074 | 0.52256 | 0.47757 | 8.61% |
| DIFF fine | 0.58680 | 0.56342 | 0.51224 | 9.08% |

Every phase mean improves in the final interval. All anchors improve for the
first three fits; 95/96 improve for DIFF fine. Four passes through the training
panel have not produced an objective plateau. Compare losses within objectives
only; the different numerical objectives do not rank CFM against diffusion.

### Field quality and distribution shape

Observed-mask medians, with 192 -> 384 comparisons:

| Metric | CFM | DIFF |
| --- | --- | --- |
| Generated density below -1 | 1.177% -> 0.170% | 1.546% -> 0.625% |
| Absolute largest-void-fraction difference from truth | 27.97 -> 8.82 pp | 26.10 -> 10.35 pp |
| Absolute filament/cluster filling difference | 6.89 -> 3.18 pp | 5.74 -> 2.34 pp |
| Four-class disagreement with truth | 54.13% -> 45.87% | 51.10% -> 46.40% |
| Mean-field eigenvalue RMSE, ordered components | [0.140,0.149,0.193] -> [0.131,0.141,0.179] | [0.141,0.148,0.179] -> [0.129,0.143,0.181] |
| Fair marginal CRPS, ordered components | [0.06446,0.07050,0.09564] -> [0.06452,0.06984,0.09057] | [0.06551,0.06804,0.08947] -> [0.05998,0.06653,0.08729] |

Complete-mask density-support medians are 0.189% CFM and 0.593% DIFF, also above
the 0.1% diagnostic tolerance. Largest observed anchor fractions are 0.857%
and 1.544%. Targets have no below-minus-one voxels. Support improvements recur
across all phases, but individual posterior scores do not improve uniformly:
ph002 CFM third-component CRPS worsens 0.09522 -> 0.09792; DIFF second/third
components worsen 0.07003 -> 0.07128 and 0.08892 -> 0.09374. Sparse shell 3
retains larger CRPS than inner shells. No production winner is selected.

Pointwise draw dispersion contracts substantially: observed medians change
from [0.161,0.154,0.164] to [0.114,0.109,0.117] CFM, and from
[0.156,0.149,0.158] to [0.127,0.121,0.129] DIFF. This may reflect learning to
denoise, but narrowing is not calibration and can make an approximate posterior
overconfident. Per-draw truth discrepancies also mix posterior uncertainty
with model error. Four draws and overlapping anchors in three training phases
are not enough to establish tail coverage or generalization.

Matched-window parent power ratios (mean of four draw powers / target, then
anchor median) reveal the important countertrend:

| k band [h/Mpc] | CFM 192 -> 384 | DIFF 192 -> 384 |
| --- | --- | --- |
| (0,.08] | 0.743 -> 0.474 | 0.946 -> 0.710 |
| (.08,.16] | 0.496 -> 0.347 | 0.623 -> 0.559 |
| (.16,.32] | 1.206 -> 0.775 | 1.077 -> 0.779 |
| >.32 | 113.27 -> 43.93 | 117.41 -> 61.15 |

High-k excess decreased, but low-k suppression became worse in every phase.
This is not merely a harmless small denominator: the median fraction of
windowed fluctuation power above .32 is 29.81% CFM / 31.34% DIFF, versus 0.494%
for targets. At 192 it was 43.44% / 41.18%. Those baseline fractions are recovered
from archived per-anchor power ratios times the identical target band powers.
These are full-parent Hann-windowed/de-meaned diagnostics, not a survey-window-
deconvolved spectrum. The smaller residual high-k excess still coexists with
incorrect large-scale amplitude. No clipping or extra R7 smoothing is applied.

### Checkpoint versus sampler stability

Median 336->384 eigenvalue RMS / pointwise draw standard deviation:
CFM [0.312,0.378,0.453], DIFF [0.422,0.404,0.352]. Both miss the 0.1 tolerance.
Observed late-checkpoint largest-void changes have medians 13.63 pp CFM and
2.35 pp DIFF; complete-mask medians are 13.81 / 2.59 pp. Fixed-axis connection
outcomes change in 7/24 observed and 9/24 complete CFM pairs, and 5/24 DIFF
pairs under either mask. Late generated fields are not stable.

Doubling sampler steps changes component RMS by only 0.136--0.166% of draw
dispersion for CFM and 2.49--3.56% for DIFF. Thus increasing sampler steps is
not the main remedy for the much larger checkpoint drift. Topology remains
more sensitive: DIFF refinement changes largest-void fraction by up to 3.18 pp
observed / 3.57 pp complete and one complete-mask connection. CFM has no changed
connections and maximum observed void change 0.050 pp. Do not claim exact
sampler convergence, especially for thresholded DIFF topology.

### Bounded failure attribution

Mean matched/control fixed-noise losses on the registered 24-anchor panel:

| Fit | Matched observations | Shuffled observations | Relative increase | Generated-coarse fine loss |
| --- | ---: | ---: | ---: | ---: |
| CFM coarse | 0.92722 | 0.98508 | 6.24% | n/a |
| CFM fine | 0.80691 | 0.88132 | 9.22% | 0.81766 (+1.33%) |
| DIFF coarse | 0.47662 | 0.57028 | 19.65% | n/a |
| DIFF fine | 0.51661 | 0.56495 | 9.36% | 0.52167 (+0.98%) |

Shuffling raises phase-mean losses in every phase/stage/method. This supports
functional use of observation inputs, rather than completely disconnected or
ignored conditioning. It is a coherent opposite-cap stress test, not a clean
causal decomposition of individual channels or a held-out information-gain
measurement; the fine shuffle retains its true coarse condition.

The modest generated-coarse loss changes do not support blaming a large
coarse-to-fine conditioning mismatch as the dominant failure in these probes.
They also do not rule it out in the tails, at other noise times or in topology.
The generated coarse and the one fixed truth residual are not a jointly drawn
truth pair, so this is a sensitivity test, not a direct estimator of deployment
risk. No loss-only control identifies the precise cause of the spectral defect.

## Decision and next recommendation

Stop at the authorized 384 updates: complete and executed. Additional training
helped several diagnostics, but it is neither converged nor science-ready.
The premise that more epochs alone would ensure convergence is unsupported.
Resume corruption, wholly unused observations and dominant sampler error are
not supported by the checks; scale-dependent variance remains unresolved.

Do not blindly extend to 768. Before a larger continuation, register a focused
scale-resolved denoising/score-error diagnostic versus noise time, separating
coarse and fine contributions, and a validation phase distinct from sealed
confirmation. Distinguish ordinary incomplete optimization from an objective,
representation or conditional-variance problem. Those experiments require a
new authorization; no held-out opening or model revision occurred here. Any
later continuation should retain intermediate field diagnostics and stop if
large-scale power/calibration deteriorate despite falling training loss.

Keep the original truth-only domain/topology caveat (6.211 pp), frozen negative
receipts and all release flags separate and unchanged. Neither these training
scores nor the completed execution establish SBC/TARP or held-out calibration.
