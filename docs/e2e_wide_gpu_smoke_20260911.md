# Wide-field GPU engineering verification — September 11

The user approved one short interactive GPU allocation to fit training-only
normalization and run full-size engineering checks, not science training.

Allocation 58196582: nid001001, one A100 40 GB, 32 logical CPUs, one-hour
`shared_interactive`, `desi_g`, Scratch license. The allocation-status check found
zero other interactive or batch jobs. Source is committed as `4983dd1`.

Runner: `python -u -m workflows.sbi.e2e_wide_gpu_smoke --output
/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/smoke_20260911_58196582`.
It runs through `srun` with an explicit GPU and isolated `cosmic_env` Python.
Log: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_smoke_58196582.log`.

The bounded checks are:

- Hash every registered training payload and fit/reload normalization on the
  original 96 anchors in ph000/ph002/ph003.
- For each CFM/DIFF coarse/fine stage, compare an uninterrupted two-update run
  with a one-update run resumed to update two: exact model, optimizer, history,
  update position and RNG agreement.
- Export each objective's full-size ancestral draw twice with the same sample
  identifier; require exact replay and shared-child overlap.
- Apply the audited wide-plus-local tidal map and generate training-reference
  diagnostics, retaining physical and topology summaries separately.

These are engineering fits, not the 192-update research canary or held-out
posterior validation. No ph004/ph005/ph001/ph006 payload access is permitted.
Frozen `training_ready=false` and `r0_physics_pass=false` remain unchanged.

## Verified result

**Technical pass.** Normalization fit/reload completed for all 96 anchors with
all 15 training payload hashes verified. CFM and DIFF passed exact CUDA resume
at both coarse and fine stages: model, optimizer, complete RNG state, update
counter and loss history. All four fits remained finite. Both methods' repeated
full-size ancestral draws matched exactly in every exported array, including
tensor and eigenvalues, with exact sibling-overlap agreement.

The sampled training anchor was ph000_NGC_s3_interior_00; the two update examples
were ph000_NGC_s0_boundary_00 and ph002_SGC_s1_boundary_01. This is deliberately
a small execution check, not a 96-anchor learned evaluation. CFM/DIFF maximum
tensor trace residuals were 8.454e-8 / 8.342e-8. Independent-reference training
diagnostics completed under both masks. No physical-error, learnability or
calibration pass is inferred from these nearly untrained draws.

The runner took 6m16s wall time (363.69s after initialization). Peak PyTorch CUDA
allocation was 846,084,096 bytes (0.79 GiB), not total driver/device consumption.
`time -v` reports 1,695,656 KiB process RSS; Slurm reports 12,604,640 KiB step
MaxRSS, a different accounting measure. Do not substitute the smaller number
for the allocation's memory requirement.

Step 58196582.0 completed 0:0 in 6m17s. The allocation was released normally and
reports COMPLETED 0:0 in 7m35s. No additional allocation or batch submission was
made. No pipeline source fix was needed during the smoke.

Archived [completion receipt](evidence/e2e_field_v2/wide_gpu_smoke_20260911/SMOKE_COMPLETE.json)
and [fitted normalization](evidence/e2e_field_v2/wide_gpu_smoke_20260911/normalization.json).
The original normalization and diagnostic checksums were reverified at closeout:

- SMOKE_COMPLETE: `46b7a2b422cebb49dd5d6c6089f8e5b96c4466be3042e6b89a58dcfbb1b27a49`.
- normalization: `c9c22e88af58c7adafac30b1aa67957c414e7206c5528428d34712452d84369b`.
- TRAINING_DRAW_DIAGNOSTICS: `5f2d5cfc6917d63ef612f588ebe174d0bb78a813504366393cb0bf8155d6bb23`.

The next possible work is the separately authorized matched research canary;
it was not started. Existing scientific gates and production programmes remain
unchanged. The NERSC allocation skill governed reuse checks, the single shared
GPU allocation, explicit GPU step, environment isolation and prompt release.
