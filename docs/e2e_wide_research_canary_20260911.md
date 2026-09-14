# Wide384_f4 matched research canary — September 11

The user's instruction, “Begin training,” authorizes the existing bounded
research canary after the passed GPU smoke, not an expanded science/production
fit or a held-out phase opening.

## Frozen run

- Launcher commit: `832da07`; `workflows/sbi/e2e_wide_research_canary.py`.
- Job: 58196924, nid001001, one GPU, 32 logical CPUs, one hour,
  shared_interactive/desi_g, Scratch license. No other allocations were active
  at the pre-launch check. The passed smoke measured roughly 1.2 seconds/update
  after startup, supporting this short single-GPU interactive canary.
- Four sequential fresh fits: CFM coarse, CFM fine, DIFF coarse, DIFF fine.
  Each stops at 192 updates (two passes over 96 training parents), with immutable
  checkpoints every 16 updates. Seed 42 and the batch order are matched.
  Do not resume from the two-update smoke checkpoints.
- Training targets, architecture, optimizer, transforms and normalization are
  unchanged. The launcher rechecks source/config/scales against the successful
  smoke receipt before every fit and at completion.
- Only ph000/ph002/ph003. No held-out selection, confirmation or sealed payloads.
  No automatic additional epochs, seeds, hyperparameter search, retries or
  production submission. P12-A/D2/P13 and their frozen products are untouched.

Configuration remains `configs/e2e_wide_pipeline_v1.json`. Its historical
`training_ready=false`, `r0_physics_pass=false` and science-training authorization
flag are not retroactively changed: this run has a separate explicit
`research_canary_authorized=true` receipt. These are research fits, not merely
engineering updates, but also not a scientific release or production model.

Normalization is reused from
`wide_pipeline_v1/smoke_20260911_58196582/normalization.json`, SHA256
`c9c22e88af58c7adafac30b1aa67957c414e7206c5528428d34712452d84369b`.

## Outputs and termination

Root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/research_20260911_58196924`.
Log: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_research_58196924.log`.
The fixed runner uses `srun` with an explicit GPU and isolated cosmic_env.
The allocation shell exits automatically when the runner succeeds or fails;
there is no tmux supervisor, adaptive agent loop or automatic retry.

Require four `CANARY_COMPLETE.json` files at 192 updates, all final checkpoint
receipts, `RESEARCH_CANARY_COMPLETE.json`, and successful scheduler/step exits
before claiming completed training. Starting this run does not establish those
terminal conditions. Losses across CFM and DIFF use different objectives and
must not be ranked directly. Held-out calibration and physical/topology
validation remain separate subsequent decisions.

Verified September 14: all four fits reached 192 updates with finite histories
and matching checkpoint hashes. Job 58196924 and its step are COMPLETED 0:0,
step 15m34s, allocation 16m47s, released. The group completion receipt is
archived in `docs/evidence/e2e_field_v2/wide_research_20260911/`.
The subsequent authorized trained-draw evaluation also completed; see
`docs/e2e_wide_evaluation_20260914.md`. It finds continuing optimization and
field-physics discrepancies, not convergence. No continuation beyond this
original 192-update contract has been executed.
