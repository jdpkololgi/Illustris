# 1299 Mpc/h wide-coarse research pipeline

## Current result and boundary

**September-11 follow-up:** the user-approved normalization and full-size GPU
smoke passed. Both objectives/stages have exact CUDA checkpoint continuation
and repeated-draw parity; the allocation is released. See the
[verified run record](e2e_wide_gpu_smoke_20260911.md). This supersedes the pending
engineering status below, but not the remaining science/calibration gates.

The raw-loader, train-only transforms, matched CFM/DIFF models, staged trainer,
checkpoint continuation, ancestral sampler and training-diagnostic entrypoints
are implemented. This is the bounded engineering pipeline for `wide384_f4`,
not a trained posterior, a topology pass, or a production VAC change.

The user requested pipeline completion following the
[completed truth-only comparison](e2e_field_wide_coarse_results_20260909.md).
The larger extent has the lowest tested typical physical error. At that extent,
factor-2 resolution was about 1% better in median eigenvalue RMS than factor-4,
but uses eight times as many coarse cells; the economical pilot fixes factor-4.
The known 6.211-point largest-void outlier remains visible. No tolerances were
changed to declare a pass, and the original negative domain receipts are intact.

No scheduler action, real-data normalization fit, GPU canary, or held-out payload
read has been performed for this implementation. Those are explicit pending
verification steps, not inferred successes. Source-only tests do not establish
full-size memory use, CUDA repeatability, learnability or calibration.

Verification: all **71 focused E2E tests pass**, including 21 new loader/model/
pipeline tests. The CLI help and production metadata-only constructor pass
(96 parents, 15 referenced training payload checksums; no array reads in that
check). Preprocessing dependencies are included in normalization provenance so
changed transforms invalidate prior scales. No scientific pass is inferred.

## How 1299 Mpc/h fits inside a 2 Gpc/h box

The Abacus periodic simulation has side 2000 Mpc/h. Each coarse target is a
1299.072 Mpc/h subvolume of its once-R7, globally low-pass-filtered density,
sampled on a 96-cubed grid at 13.532 Mpc/h spacing. A cutout crossing the box
boundary uses the simulation's periodic coordinates. Because its side is less
than 2000, the cube does not repeat a full box along any axis.

Its volume is `(1299.072/2000)^3 = 0.274`, approximately 27.4% of a box. The
96 anchors overlap substantially, share large-scale modes, and are distributed
over only three independent training phases: ph000, ph002 and ph003. They are
not 96 independent universes, nor does coarse interpolation create extra volume
or information. Whole-phase splitting remains mandatory. Modes longer than
2000 Mpc/h, super-box variance and cosmology/HOD variation are not tested.

Inside each wide cube, the retained fine parent has 96 cells at 3.383 Mpc/h,
or 324.768 Mpc/h side. The central science core has 32 cells, or 108.256 Mpc/h.
Equal array dimensions at coarse and fine scales do not mean equal physical
coordinates. The fine field covers the central quarter of the coarse extent.

## Implemented distribution and physics

The pilot generates

```text
wide galaxy/response observations -> one coarse density draw L
local observations + wide-observation features + U(L) -> one fine residual draw H
local density = U(L) + H
science-core tensor = U_core[T_wide(L)] + crop[T_local(H)]
```

`U` is the original centered quintic interpolation, not a resize of the whole
1299 Mpc/h cube onto the 325 Mpc/h local cube. Both stages are stochastic and
newly learned. No frozen G1 completion or full-box-low oracle enters inference.
The observation-only reader does not read target/oracle arrays; it reads only
alignment attributes from the field file and galaxy/response channels. Initial
integrity verification hashes the registered training files without using their
target values as conditions.

The fine model consumes the full wide observation tensor through a separate
learned compression branch, plus aligned local observations and the same coarse
draw. That compression is an approximation to the fine conditional, not a
conditional-independence theorem. During training the coarse conditioning draw
is the true coarse target (standard staged conditional training); at inference
the sampler replaces it with the generated coarse field. Distribution shift
from imperfect coarse predictions remains something to diagnose.

Important amendment to the original plan's section 3.2: the global source
low-pass is fixed at `|k| <= 0.08 h/Mpc`, but a crop of it is not an orthogonal
low-pass projector on a local periodic lattice. Here `H = delta_local - U(L)`
is the exact stored finite-window residual. It is not constrained to an exact
local Fourier complement. Projecting its noise, state or velocity with a local
high-pass would change the tested target. This pilot uses full-coordinate
Gaussian noise and no such projection. The earlier same-lattice orthogonal
construction is a different, unimplemented representation, not a claim about
these arrays. WCFM and MIRA are not activated.

Both tensor terms use the audited spectral operator, all six symmetric
components, and isotropic parent-mean/DC completion. There is no second R7
smoothing. Computing tides only from the assembled local density would discard
the benefit of the wide domain and is deliberately not the implementation.
The sampler checks density/tensor trace closure and exports ordered eigenvalues
from the symmetric tensor, never a learned unconstrained three-value head.

One sample identifier gives one full wide draw and one full fine draw. Child
crops are views of that fine realization, so sibling overlap is exact. Separate
anchor calls have distinct random streams and do **not** establish a coherent
full-cap field across independently generated wide parents. Wider multi-anchor
coordination, joint multi-child calibration and survey deployment remain later
work; changing a sample ID is not evidence of full-cap coherence.

## Frozen canary contract

Configuration: `configs/e2e_wide_pipeline_v1.json`.

- Three training phases and the original 96 anchors; no ph004/ph005/ph001/ph006
  access. The products-complete manifest checksum is pinned.
- Separate affine coarse/fine target scales; streaming training-only moments;
  fixed observation transforms with explicit support and geometric-validity
  channels. Binary/response/direction identity channels remain unstandardized.
  Do not mask latent matter out merely because galaxies are unobserved.
- A matched modest 3D U-Net, base width 8 and three levels. The fine network
  has a separate wide-context branch. Architecture is identical between CFM and
  DIFF at each stage. This is a small baseline, not an architecture search.
- CFM uses the straight Gaussian-to-data path; DIFF uses cosine VP noising and
  v-prediction. Both use scalar voxel-mean objective losses, AdamW at 1e-4,
  weight decay 1e-4, gradient clipping at 1, batch size 1, seed 42.
- At most 192 updates per stage per objective: two passes over 96 parents.
  The four runs share initialization convention and deterministic parent order.
  No EMA, adaptive stopping, best-checkpoint selection, loss weighting search,
  additional training seeds or automatic budget extensions.
- CFM Heun 16 steps and DIFF DDIM 32 steps each use 32 network evaluations per
  stage, 64 for a complete draw. Smaller matched budgets are allowed for an
  engineering check; no automatic inference sweep.
- Immutable numbered checkpoints bind config, code, payload checksums,
  normalization, objective, stage and PyTorch version. They retain Adam state,
  update position, Python/NumPy/Torch/CUDA and explicit objective RNG states.
  Resume uses a new output directory and the original total update cap.
  Exact continuation is tested on tiny CPU arrays; GPU parity is still pending.

## Entrypoints and authorized-run sequence

Use `cosmic_env` in a separately user-approved interactive allocation. This
document is not permission to request one or start a science fit. No launcher
submits jobs automatically. CPU normalization and GPU training/sampling reject
login-node execution. All new outputs go beneath the separate Scratch root
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1`.

From the Illustris repository, the command family is:

```bash
python -m workflows.sbi.e2e_wide_pipeline --help
python -m workflows.sbi.e2e_wide_pipeline prepare --output /pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/normalization.json
python -m workflows.sbi.e2e_wide_pipeline train --help
python -m workflows.sbi.e2e_wide_pipeline sample --help
python -m workflows.sbi.e2e_wide_pipeline diagnose --help
```

`train` requires `--normalization`, `--stage coarse|fine`, `--method cfm|diffusion`
and a new `--output` directory. `--stop-after 2` gives a two-update engineering
check. `--resume` points to a numbered `.pt` checkpoint with its companion JSON
receipt; the new total stop must exceed the saved update and remain <=192.
Coarse and fine stages can be trained independently because training uses true
coarse conditions. They cannot be mixed across objectives or update budgets at
sampling. No trained checkpoints currently exist for this pipeline.

`sample` requires both `--coarse-checkpoint` and `--fine-checkpoint`, a registered
training `--anchor`, `--sample-id`, normalization and a new HDF5 output path.
It exports coarse density, full fine residual, assembled local density and
central tensor/eigenvalues, with deterministic spatial identity and checksums.
Sampling reads observations, not the training targets used later for evaluation.

`diagnose` takes one or more `--samples` and a new JSON output. It retains
per-anchor/per-draw eigenvalue and class errors, filling fraction, pair bins,
fixed connections and largest-void statistics under both complete-core and
observed-core masks. Phase/cap/shell/support identities are retained. These
descriptive training checks are not SBC, TARP, conditional coverage, a proper
held-out score, or a calibrated-posterior pass. In particular a stochastic draw
is not expected to equal one simulator truth pointwise.

## What is required next

1. Approved compute smoke: fit/reload all training normalizers, exercise both
   objectives and stages at full 96-cubed shape, export a draw, evaluate its
   independent reference, record wall time/memory, and verify CUDA replay and
   split-versus-uninterrupted continuation. Nonfinite states/losses/gradients,
   failed source binding, oracle leakage, inconsistent overlap or trace closure
   are technical stop conditions. Fix defects before a learned pilot.
2. If separately authorized, run the matched 192-update training canaries.
   Loss behavior and sample diversity are learnability diagnostics, not release
   gates; do not tune against the known void outlier or interpret training
   metrics as calibration.
3. Freeze a genuinely held-out experiment, build authorized wide-context
   selection/confirmation products and extend the reader under explicit access
   controls. Register phase-level uncertainty, conditional strata, coherent
   functionals, sampler convergence and calibration/power checks before opening
   those roles. The current train-only reader intentionally cannot do this.

Full scientific training and field-posterior selection are therefore not
completed by this engineering implementation. Frozen `training_ready=false`
and `r0_physics_pass=false` receipts remain unchanged. P12-A, D2 and P13 are
untouched. The app-history issue is recorded separately in the
[recovery note](recovery/2026-09-09-recovery-summary.md); no successful display
repair is claimed.
