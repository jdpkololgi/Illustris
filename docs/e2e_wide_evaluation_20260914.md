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

The evaluation is initially running. Require the full completion marker and
successful step termination before claiming panel-wide results. The NERSC
allocation skill governs isolated compute execution and release afterward.
