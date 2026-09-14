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

Pending launch after focused tests and source commit. Request one GPU/32 logical
CPUs under shared_interactive/desi_g, Scratch license, two-hour cap. Prior measured
training plus evaluation/report totals about 100 minutes. The NERSC allocation
skill requires the allocation-reuse check, isolated cosmic_env and prompt release.
No compatible allocation was present at preflight. Record terminal receipts and
actual results here before claiming completion.
