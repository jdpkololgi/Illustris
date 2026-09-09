# Separate E2E spectral products — 2026-09-09

The user authorized spare interactive capacity to start the recommended separate
regeneration. No model training or D2 changes are authorized by this build.

Contract: `configs/e2e_field_spectral_v2.json`. New output root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/spectral_20260909`.
Builder: `workflows/sbi/e2e_field_regenerate_spectral.py`.

Reuse hashed 2048-cubed TSC counts; apply one Gaussian R7 smoothing and the
full-box spectral tensor operator. Sample the scalar and six tensor components
with the same local cubic Lagrange interpolation on integer-centred native sites.
Gaussian density is reconstructed independently by inverse-transforming the
smoothed spectrum, not by relabelling the inherited finite-difference trace.

Preserve all 160 response-selected anchors, phase roles, 64/96-cell nested
parents, 128-cell observation context and separate masks. Copy observations and
masks into new self-contained shards and verify exact equality. Refit the new
target scaler on ph000/ph002/ph003 only; recompute unchanged observation scalers
and require equality to the previous train-only values. ph004/ph005 are packaged
under their existing selection/confirmation roles, with numerical checks only.
ph001/ph006 payloads are excluded.

Before results, the numerical gates are: finite products, density/tensor trace
max residual <=2e-6, agreement with the previous training-core cubic reference
<=2e-6, and cubic-versus-quintic component RMS <=3e-4 of component scatter on
training cores. The last is an interpolation engineering tolerance, not a
scientific topology or posterior-calibration criterion. Full-size nested-reader,
source/shard hashes and release/confirmation guards are also checked.

Native source counts and all inherited products remain unchanged. A separate
phase completion marker follows its payload checks and hashes. Resume skips only
exactly matching completed phases; partial shards are preserved and never
silently overwritten. REGENERATION_COMPLETE.json is the terminal build marker.

The new reader is `SpectralParentDataset`. Default training/confirmation access
remains guarded. Successful regeneration does not resolve exterior tides,
topology sensitivity, scientific error budgets, matched model contracts or
diagnostic power; `training_ready=false` and `r0_physics_pass=false` remain.

Four focused implementation tests pass for audited cubic equivalence, quintic
polynomial reproduction, periodic/integer sampling and payload access guards.
Started job **58115135** on **nid004150**, one foreground CPU interactive
allocation with 64 logical CPUs, account desi, Scratch license and a two-hour
limit. No interactive allocations existed at the pre-launch check. The previous
audit's approximately 264 GiB peak RSS motivates a full CPU node; no GPU is used.
The first phase's input hashes passed and spectral density generation began.
Completion is not yet claimed. The foreground launcher exits with the job step.

Log: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v1/spectral_regeneration_20260909_58115135.log`
(log location only; all regenerated arrays go to the separate v2 root).
`REGENERATION_STARTED.json` records base Git revision
`a438e20f02c9ba5eae1a6a0ecd18a57309bffb03`, all builder/import source hashes,
source-index hash and configuration hash
`28ecea86dc318ee408d9755161b71745519247a670a0a5b842df0bfcbb89352e`.
Only `REGENERATION_COMPLETE.json` plus clean scheduler/step exits establish
technical completion; neither releases model training.
