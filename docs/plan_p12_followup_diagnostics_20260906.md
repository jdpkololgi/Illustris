# D2 support geometry and P12-B representation investigation

## Status reconciliation — 2026-09-08

Both diagnostic branches completed. All 133,698 D2 galaxies satisfy native
footprint/radial support; the 736 nearest-voxel M=0 cases are mask discretization,
not patch truncation. The user approved the [native-support amendment](d2_native_support_amendment_20260906.md),
retaining all galaxies and leaving masks and field calculations unchanged.
D2 seed42 subsequently passed the amended registered ladder; the licensed
second training seed is submitted, with no combined decision claimed here.

P12-B replay, feature-use and gradient diagnostics completed, followed by the
separately registered [five-arm continuation](plan_p12b_representation_followup_v1.md).
That follow-up is complete with no demonstrated absolute continuation gain.
See its [results](evidence/p12/p12b_representation_followup_v1/RESULTS.md).
The initial prospective instructions below describe the original registration;
they do not authorize a repeat allocation or further variants.

Registered 2026-09-06 after explicit user authorization of both investigations.
The D2 evaluation amendment is conditional on geometric/construction evidence.
No original checkpoint, patch archive, mask, transform, evaluation or contract is
overwritten. No ph001 data are accessed. P12-B uses only ph005 payloads.

## D2: decide whether an amendment is scientifically justified

Audit all 133,698 archived ph006 galaxies, with detailed coordinates/parent IDs for
the 736 M=0 galaxies. Distinguish distances to authoritative-core faces, context
patch faces and nearest supported voxel centres. Distances use voxel units and
the manifest's 5 Mpc cell size, not the inherited misleading Mpc/h name.
Compare supported and unsupported row geometry. Face distances are signed and use
cell faces at index +/- 0.5, not distances to outer voxel centres.

Independently reconstruct native random-map angular/cap membership and radial
membership at both galaxy positions and nearest voxel centres. Use the selected
18-random map, cap/PHOTSYS domain and frozen cosmology interpolation. The stored
mask must match this rule at every sampled voxel centre. This distinguishes patch
cropping, grid/HEALPix discretization, and genuinely unsupported angular positions.
Stored boundary distances are zero at M=0 and cannot measure outside depth.

If unsupported galaxies lie well inside patches, investigate support construction
before amending evaluation, as the user requested. Do not treat survey-mask edges
as proof of a patch-edge explanation. No numerical pass threshold is invented
post hoc: report the complete distance distributions and row-level classifications.
An evaluation amendment, if justified, preserves original all-galaxy metrics,
uses identical supported-galaxy conditional estimands for all methods, separately
accounts for unsupported rows and explicitly records the changed claim boundary.
The frozen 467f442 D2 export recovery proceeds independently.

## P12-B: read-only first diagnostic stage

Reuse immutable point/frozen/joint terminal checkpoints at update 3000. Verify
their registered source/data/config hashes, old deterministic head invariance and
every ph005 patch used. Read no ph006 payloads and no ph001 data. Preserve all
initial training-only normalizations; do not refit them on diagnostics.

1. Summarize sparse logged loss and clipping trajectories without claiming that
   different logged minibatches establish convergence.
2. Compare terminal versus initial encoder weights per tensor. Report relative
   changes, standardized feature drift and channel spread on internal rows.
3. On 32 training cores selected by evenly spaced manifest indices, evaluate one
   fixed-noise loss/backward probe per arm. Record head and encoder gradient norms
   and the resulting common clipping multiplier. Perform no optimizer updates.
4. On all 128 original internal cores, use the original evenly spaced at-most-16
   rows/core and 128 posterior draws. Replay original Heun64 posteriors exactly
   within 1e-6. Then zero or permute the entire standardized 32-feature vector
   within cap/shell groups; never alter the cached point/response block. The
   point-only arm is an exact negative control for both interventions.
5. Report energy, CRPS, 68/90 coverage and widths, fixed-noise FM losses (8 repeats),
   and paired cap+superblock 2,000-bootstrap energy differences. Check Heun128
   refinement on every 16th core for all variants using common noise. Apply the
   original numerical tolerances: scaled mean draw difference <=0.01 and absolute
   pooled coverage differences <=0.01. Do not interpret a failed sampler arm.

Zeroing/permutation can create off-manifold contexts. These are reliance probes,
not tests of conditional sufficiency or proof that one representation is better.
Single-phase/seed limits remain. These outputs cannot promote a production model.

## Next-stage design, conditional on diagnosis

If loss is still improving or encoder/head updates are poorly balanced, register
a matched continuation versus warm-start joint study: retain the point/frozen
controls, match total examples, report inherited head exposure, and separate
encoder learning rate from head learning rate/global clipping. Do not select on
ph006. A promising result then requires phase/seed replication and independent
confirmation. If reliance is negligible, first examine conditional redundancy and
head capacity/feature scaling rather than assuming a longer joint run will help.
At this initial registration the launch covered diagnostics only. Subsequent
training was separately specified and completed under the follow-up plan linked
in the dated status reconciliation above.

## Resource and stop contract

One user-authorized shared-interactive A100 80GB, 32 logical CPUs, at most one hour,
Scratch licensed, shared between D2 geometry and P12-B diagnostics. Unit tests and
source checks precede real data. P12-B has an internal 3000-second limit; no chained
allocation, automatic retry or batch fallback. Release allocation promptly after
completion. Record actual job ID, exit codes, manifests and scientifically qualified
results in SCIENCE_LOG.md. Stop on identity/source changes or nonfinite values.
