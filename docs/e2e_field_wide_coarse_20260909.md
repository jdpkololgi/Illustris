# Wider-coarse/local-fine truth-only prototype

Authorized 2026-09-09: build additional wide-context products on an interactive
CPU allocation and initiate the truth-only physics test. No learned training,
holdout access, production VAC change, or P12-A/D2/P13 change is authorized.

## Frozen experiment

Use the same 96 anchors in training phases ph000/ph002/ph003, both cap masks,
96-cubed local fields and 32-cubed science cores. Extend the physical domain:

| Variant | Extent (Mpc/h) | Coarse grid | Cell (Mpc/h) |
| --- | ---: | ---: | ---: |
| wide192_f4 | 649.536 | 48 cubed | 13.532 |
| wide384_f4 | 1299.072 | 96 cubed | 13.532 |
| wide384_f2 | 1299.072 | 192 cubed | 6.766 |

The third variant is a fixed-extent resolution control. Apply the fixed full-box
Fourier projector `|k| <= 0.08 h/Mpc` to the original once-Gaussian-R7 field.
Reconstruct retained coefficients on a compact 512-cubed full-box lattice with
exact FFT normalization, then quintic-sample coarse cell centers. This is a
low/high decomposition, not another R7 smoothing. Retained modes lie below all
coarse-grid Nyquist frequencies. Synthetic Fourier tests check amplitude/sign.

Store `fine_residual = delta_R7_local96 - U(coarse_delta)`, where U is quintic
interpolation with its stencil strictly inside the wider domain. The local
density is unchanged up to checked float32 rounding. The wide matter field is a
training target, not known matter supplied to inference.

Coarse cap-specific observations are sums of counts/expected counts, cell-volume
means of support/response/exposure/number density, and a recomputed
`log((counts+0.5)/(expected+0.5))`. Cache each cap at factors 2 and 4 and reference
parent windows. Explicit geometry fractions distinguish missing observations
from zero counts. Observations never wrap at cap edges. The raw reader derives
LOS and observer radius from coarse cell-center coordinates. Fine conditions
and masks retain the hashed v2 shards. No wide-context normalizer is fitted.

## Physics comparison

Test `T_wide(coarse_delta) + T_local96(fine_residual)`, with the declared periodic
spectral operator, mean/3 diagonal completion, and no new smoothing. Compare to
full-box truth and the original local-96 baseline. A full-box-low plus local-high
oracle isolates remaining local-high boundary error; never use it as a model
condition. Decompose total tensor error into wide-low error, local-high error,
and the compensating coarse-interpolation residual, retaining the full
Frobenius Gram matrix with cross terms, not quadrature-added RMS values.

Report eigenvalues, classes, filling fraction, pair probabilities, fixed-terminal
connectivity, largest void fraction, and resolution-control error separately,
for both masks and by phase/shell/support. Previous all-anchor tolerances are
historical diagnostics only, not an automatic pilot-training veto or a release.
The old domain verdict is not retroactively changed. All products retain
`training_ready=false` and `r0_physics_pass=false`.

## Execution

Entrypoint: `python -m workflows.sbi.e2e_field_wide_coarse`. Requires Slurm compute,
verifies only training payloads, writes exclusive new outputs, finishes all three
phase products, then automatically starts the physics test. Partial runs are
preserved, never silently overwritten/resumed. Source/config hashes and input
modification state are checked through the run; each anchor audit is checkpointed.

Output: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_coarse_20260909`.
The native 2048-cubed double-precision forward FFT needs a full-memory CPU node;
subsequent low-mode FFTs use the compact grid. Request one CPU node, 64 logical
CPUs, two hours, interactive QOS, desi account, Scratch license, through srun.
No batch submission or autonomous retry.

Launched as interactive CPU job 58131992 on nid004156, frozen source 5051f69.
All 50 focused E2E tests passed before launch. Log:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_coarse_20260909.log`.
All three phases are built: 96 anchors, 288 anchor-variant checks, nine payloads,
5,686,861,663 bytes. Maximum density reconstruction error is 3.070e-7, independent
direct-Fourier interpolation error 6.191e-7, and relative count-conservation
error 6.706e-10. Both numerical field errors are below the 2e-6 limit.
The physics-start marker and first 16 anchor reports are verified; the full
test is still running, with no scientific outcome or release yet recorded.
Durable product hashes and start receipt:
`docs/evidence/e2e_field_v2/wide_coarse_20260909/PRODUCTS_AND_TEST_START.json`.
