# P12-B: U-Net representation / FMPE pilot

## Status reconciliation — 2026-09-08

Completed on 2026-09-06: all three arms ran 3,000 updates and 216,138 row
presentations, with all registered numerical/replay checks passing. Neither
frozen features nor joint fine-tuning demonstrated a convincing proper-score
gain over the matched point control; no production change was licensed.
Evidence: [pilot comparison](evidence/p12/p12b_unet_representation_v1/P12B_COMPARISON.json).
The separately registered [investigation](plan_p12b_representation_followup_v1.md)
is also complete and did not demonstrate absolute improvement from continuation.
The original prospective design below is retained as registration history, not
an instruction to rerun the pilot. P12-A and coherent-field D2 remain separate.

Registered 2026-09-05 following the user's request to prepare and launch
interactive frozen-latent and jointly learned posterior tests. This is a new
per-galaxy research comparison. P12-A, D2, and all ph001 products remain unchanged.
The design follows the representation question, not ph001 outcomes. No ph001
inputs, labels, metrics, or tuning are used by this workflow.

## From feature grid to galaxy posterior

`[1,3,X,Y,Z] -> U-Net -> [1,32,X,Y,Z]` is a learned feature field. Trilinear
interpolation at N authoritative galaxy positions returns `[N,32]`. The existing
deterministic MLP maps those vectors to three increments. FMPE instead learns a
velocity in three-dimensional ordered-softplus target space, conditioned on each
interpolated vector and response. Posterior draws are `[N,draws,3]`, not voxel fields.
Shared encoder features do not make independent galaxy draws coherent field draws.

Frozen features: detach/cache `[N,32]`, train only FMPE. Joint features: FMPE loss
backpropagates through interpolation and U-Net. During inference the feature grid
is computed once per patch; the small conditional head is evaluated along each ODE.


A useful way to view the grid is as a lookup table of *learned descriptors*.
Its spatial dimensions match the input, but its channel dimension is 32, not the
three eigenvalues. Interpolation selects the descriptor at each galaxy without
losing the surrounding context already processed by the convolutions.

For a patch with N selected galaxies, the training computation is conceptually:

```python
z = encoder.sample_latent(values, galaxy_grid_coordinates)  # [N, 32]
z = z.detach() if arm == "frozen" else z
context = concatenate(cached_point3, response4, standardize(z))  # [N, 39]
loss = fmpe.loss(standardize(ordered_softplus_truth3), context).mean()
loss.backward()
```

The point arm substitutes zeros for the standardized feature block. For the
frozen arm, cached features eliminate this encoder computation during head
training. For joint training, the U-Net is evaluated once per patch/update;
gradients flow through trilinear interpolation and the conditional FMPE velocity
network. At posterior sampling time, even the joint model computes the feature
grid once, then holds these deterministic conditioning vectors fixed while the
small velocity network transports many independent 3-D noise draws. The U-Net
therefore does **not** run separately at every ODE step.

The installed SBI convention is data at t=0 and noise at t=1: training predicts
noise minus data along its registered interpolation path, and posterior sampling
integrates backward from t=1 to t=0. The first optimizer update starts from an
SBI zero-initialized output; conditioning gradients are checked after that shared
initial step. Smoke updates are discarded before the fixed training runs.

## One latent basis and matched examples

Use the **terminal epoch-20** omit-ph005 U-PATCH trained on ph000/ph002/ph003/ph004.
Do not select its best checkpoint using ph005 scores. Terminal weights come from
`arm_a_checkpoint.pt`; input normalization and deterministic target scaler come
from the frozen omit-ph005 transforms. Do not concatenate independent fold latents.

Posterior training uses ph005 P4 folds 0--3. Fold 4 supplies internal diagnostics.
Require selected train/internal input-context boxes to be disjoint within each cap,
preserving native superblock/fold ownership and exact parent identities. All arms
share cores, rows, exposure order, stochastic loss seeds, transforms and response.
This is an alignment-free pilot using a single posterior-training phase, not a
full multi-phase production comparison. Historical encoder selection decisions
remain provenance; this is not fresh training-phase confirmation.

ph006 supplies a fixed development/transfer diagnostic, **not a new blind test**.
Export it with the same omit-ph005 encoder/transforms, not P12-A's all-five-phase
encoder. No ph006 metric selects checkpoints or changes this pilot. Report terminal
registered budgets only. No ph001 path is admitted.

## Arms and budget

All heads have the same 39-input interface: three cached frozen physical predictions,
four response covariates, and 32 standardized feature channels. Covariates are z,
log ntilde, cap, and log1p random-support boundary distance. Only supported
authoritative galaxies are retained. Phase/fold/core IDs are never features.

| Arm | Feature block | Optimized weights |
| --- | --- | --- |
| point | 32 zeros | FMPE only |
| frozen | Frozen 32-d U-PATCH features | FMPE only |
| joint | Differentiable 32-d U-PATCH features | U-Net and FMPE; old point head frozen |

Retain identical cached point predictions in all arms, making the comparison nested.
This tests additional information in features and then posterior-driven feature
learning; it does not add JEPA or another density-estimator family. Equal update and
target-presentation exposure is primary; joint compute/trainable parameter counts
are greater and must be reported.

Config: 512 train cores, 128 guarded internal cores, 128 ph006 cores, at most 128
supported galaxies/core, 3,000 single-core updates/arm, seed 42, head LR 5e-4,
encoder LR 1e-5, clipping 5, and a common 256-wide three-layer installed SBI FMPE.
Select cores by cap and dominant observed redshift shell; uniformly sample rows
within a core. This is a balanced panel estimand, not full-footprint inference.
Log actual rows, unique cores/superblocks, updates, presentations, time and memory.

## Verification and evaluation

Before training: source hashes; exact row identity; real-patch cached/online feature
parity; ordered target roundtrip; finite loss/backward; frozen-weight invariance;
nonzero joint U-Net gradients; deterministic checkpoint replay. Use SBI's FMPE loss
directly to preserve condition gradients and avoid an IID internal DataLoader split.

Keep untempered samples. Report physical joint energy score, marginal CRPS,
68/90% coverage, widths, physical eigenvalue/eigengap TARP, posterior mean R2,
and physical log scores on a bounded common subset using exact three-dimensional
divergence. Heun64 sampling gets a common-noise Heun128 subset check. Arm-difference
uncertainty uses cap+superblock clusters, not IID galaxies. Truth-conditioned slices
are diagnostic, not assumed to have nominal Bayesian coverage.

Write technical completion and a comparison, never a production/calibration-pass
marker. Narrowness alone does not select a winner. A promising representation must
improve held-out proper scores without worsening observed-shell coverage and needs
phase replication and independent confirmation. No adaptive extensions, extra seeds,
ph006 recalibration or temperature fits are part of this launch.

Stop on changed contracts/hashes, ph001 paths, duplicate parents, train/internal
context overlap, nonfinite values, gradient or memory failure. Save atomic checkpoints
before the interactive soft deadline. Resume only with the same contract/budget;
no autonomous allocation chain or batch fallback is authorized.

## Products and execution

Source/config/scientific record stay in Illustris. Normalized raw-patch caches,
parent-keyed feature/target/response arrays, manifests, checkpoints and reports stay
under dedicated Scratch `p12b_unet_representation_v1`. Reuse response caches only
after parent and source-hash checks. Never mutate P12-A caches or native truth.
Large HDF5 payloads are bound by their existing source manifests, not rehashed in
full for this pilot; log this verification scope explicitly.

Use one shared-interactive GPU for two hours: focused tests, preparation, smoke,
then the fixed three-arm run if smoke passes. Respect the two-allocation limit and
leave D2's queue intact. Record allocation/node, source/config hashes, environment,
commands and exit/completion status. If the budget exceeds the allocation, leave a
verified partial checkpoint rather than silently changing exposure or launching batch.
