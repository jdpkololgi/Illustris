# Bounded CFM field gallery and exploratory Loa patch

Authorized 2026-09-30: one shared GPU allocation, at most 90 minutes and
10 GiB new Scratch. No training, selection refitting, VAC changes, or access to
sealed phases 018/019. All four pilot training factors completed 26,624 updates;
use the frozen mixed seed-17 candidate (coarse 13,312; fine 26,624), not a new fit.

Fixed before predictions: ph016/ph017, NGC shell-0 interior_00 and boundary_00;
central two-cell XY slabs, truth, first three draws, mean and sample SD of the
32 saved draws. Dots are actual observed mock galaxies in that same slab.
These phases are unseen in training, but previously exposed to evaluation.
Common log-density limits [-0.5, 0.5]; show unsupported regions and owned cores.

Loa: one NGC patch centered nearest the eight-raw-cell grid point to
RA=180 degrees, Dec=10 degrees, z=0.22. Recompute pinned DESI Mpc/h coordinates
from the existing verified VAC input cache, never reuse its Planck-Mpc xyz.
Keep successful-observation selection, random angular response and inherited
training selection fixed. Rebuild the original 12 local and 12 wide channels
with nonperiodic zero/missing padding; require channelwise mock replay parity,
CIC conservation, and at least 25% support in both owned cores before sampling.
Use 16 draws at NFE=128 and a two-draw NFE=256 sensitivity check; positivity and
block-mass closure are mandatory. Retain hashes, overlay coordinates, and receipt.

Loa images are exploratory simulation-trained conditional samples, not a
calibrated real-data posterior or a production field VAC. Known Loa/mock mismatch
above z=0.35 remains unresolved, and the wide context can include that range even
when the local patch is at low redshift. Finite-context FFT tidal closure is not
an exact open-boundary gravitational solution. The existing per-galaxy P12-A VAC
is a separate provisional product and remains untouched.

## Boundary interpretation

The observation is not a periodic box. The network receives explicit support,
angular response, apodized exposure, expected counts, boundary distance, radial
selection, and line-of-sight information alongside galaxy counts. This allows
an unobserved cell to differ from an observed cell containing zero galaxies.
Outside the cap's computational grid, observations are padded as missing/zero;
they are never wrapped from the opposite edge. The 1,299.072 Mpc/h wide context
conditions the 433.024 by 324.768 by 324.768 Mpc/h local field. Generated density
outside observational support is a model-dependent conditional extrapolation,
not a measurement. The hatching in these plots marks slab support below 50%,
not a calibrated science-quality threshold. Owned cores are dashed rectangles.

Density plots do not require a new gravitational boundary solve. For later
tidal tensors, the existing operator combines a finite wide-grid FFT contribution
with the local residual. It preserves the trace/block identities but cannot
recover unconstrained exterior modes or eliminate finite-context boundary error.

## Reproduction

Entry point: `python -m workflows.sbi.e2e_cfm_field_gallery --output NEW_DIRECTORY`
inside an authorized one-GPU Slurm step, with `cosmic_env`, isolated Python paths,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Output must not already exist. The runner
does not modify the source catalogue or the VAC. It verifies cached Loa input
hashes and model/normalization bindings, and writes new draws only to its output.
Three focused tests cover cropped CIC parity, exact slab membership, and checking
every observation channel. The end-to-end mock replay separately tests the full
adapter against the committed observation products.

## Completed results

Allocation 59137943, node nid200416, one shared 80-GB GPU. The first attempt
stopped before producing a Loa draw because the deterministic CuBLAS environment
was missing; its directory/log were retained. The corrected attempt completed
in the same allocation. Scratch outputs total 48,628,792 bytes across both
attempts (about 46.4 MiB, excluding tiny sibling logs), below the 10 GiB limit.
Authoritative output: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_gallery_20260930_v2`.
Slurm accounting: allocation COMPLETED 0:0 in 9m09s (0.1525 allocated GPU-hours);
first step FAILED 1:0 in 1m11s, corrected step COMPLETED 0:0 in 4m53s, final
test/VAC-integrity step COMPLETED 0:0 in 9s. Allocation released promptly.

- Mock replay: all 12 local channels exactly reproduced; all 12 wide channels
  passed, maximum absolute discrepancy 1.90735e-6 (floating-point pooling order).
- Loa owned-core support: 1.000000 and 0.9996643. The displayed slab has no cells
  below 50% support; it contains 15,166 actual successful-observation galaxies.
- 16 main draws, NFE=128; two matched-seed refinement draws, NFE=256.
  Refinement field RMSE 0.00055597 versus RMS pointwise posterior SD 0.187193
  (ratio 0.00297). This is a two-draw numerical sensitivity check only.
- Positive decoding passed; maximum relative coarse-block mass error 8.69e-16.
  Cropped CIC grid sums agree with deposited weights to <3e-4 galaxies.
  Weights falling outside the crop are explicitly accounted for, not missing
  catalogue objects or a failed whole-survey conservation test.
- Three focused tests passed both before and on the compute node. All five PNGs
  were visually inspected. Common color scales, exact galaxy overlays, support
  hatching and explicit uncertainty panels follow the visualization workflow.
- Original production VAC SHA256 rechecked on the compute node:
  `cf472cb4e8fb327620629f347115ad26c55a3f985320b293e6753e07f50ebfbc`, unchanged.
  No mock repair, fitting, VAC rewrite, or sealed-phase access.
- Graphify refresh was attempted but unavailable (`graphify: command not found`).

Figures: [ph016 interior](figures/cfm_gallery_20260930/ph016_NGC_s0_interior_00.png),
[ph016 boundary](figures/cfm_gallery_20260930/ph016_NGC_s0_boundary_00.png),
[ph017 interior](figures/cfm_gallery_20260930/ph017_NGC_s0_interior_00.png),
[ph017 boundary](figures/cfm_gallery_20260930/ph017_NGC_s0_boundary_00.png),
[real Loa](figures/cfm_gallery_20260930/loa_exploratory.png).
The neighbouring `COMPLETE.json`, `MOCK_ADAPTER_CHECK.json` and `run.log` retain
source hashes, draw hashes, grid origins, geometry and numerical checks.
Overlay coordinates and all Loa draws remain in the authoritative Scratch output.

The boundary examples visibly show greater variation outside observed support,
but this does not establish posterior coverage. The earlier held-out calibration
failures remain; none of these visualizations promotes the model to production.
