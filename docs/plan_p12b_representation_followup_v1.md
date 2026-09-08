# P12-B matched continuation and optimization controls

## Status reconciliation — 2026-09-08

Completed on 2026-09-06. All five arms reached 6,000 total updates and 431,146
row presentations; all numerical gates passed. Joint-minus-frozen energy is
-0.000251 with descriptive clustered 95% interval [-0.001427, +0.000903].
No continuation demonstrates improvement over its 3,000-update parent;
separate clipping and warm-start did not reliably repair the pilot. Features
are connected and used, but additional conditional information beyond the
cached point summary remains unresolved. No model is promoted and no further
variant is automatically licensed. See the [completed interpretation and next
gate](evidence/p12/p12b_representation_followup_v1/RESULTS.md).
The registered design below is preserved unchanged as historical provenance.

Registered 2026-09-06, after the user requested planning and launching the
joint-FMPE investigation and after the read-only diagnostic stage completed.
This is an exploratory internal follow-up, not independent confirmation.

## Motivation and limits

All original internal posteriors replay exactly, all numerical gates pass, and
the point ablation negative control is exact. Frozen/joint feature permutation
increases energy by 0.08098/0.08693 on the internal panel. Thus the fitted heads
use the features, but these off-manifold interventions do not establish extra
conditional information beyond the cached point/response block.

The joint encoder is neither disconnected nor collapsed: standardized feature
RMS movement is 0.1713, channels retain spread, and the old point head is unchanged.
Joint global gradient clipping occurs on 90.9% of logged updates (84% in the last
25), versus 23.1% (28%) for point/frozen. Terminal mean head/encoder gradient norms
are 3.59/14.98 on 32 fixed training probes. Adam's adaptive scaling means these
numbers alone do not prove that clipping caused poor performance. Test the recipe.

## Frozen five-arm continuation

Each arm has 6,000 total FMPE updates: inherit a verified 3,000-update checkpoint,
then perform exactly 3,000 additional updates with the identical extension of the
original seed-42 core schedule and loss noise/time seeds. Original source/data,
training-only transforms, cached point predictions and the old deterministic
head remain unchanged. No ph006 payload or ph001 data are accessed.

| Arm | Parent checkpoint | New optimized weights | Clip rule |
| --- | --- | --- | --- |
| point_continue | point at 3000 | FMPE | global norm 5 |
| frozen_continue | frozen at 3000 | FMPE | global norm 5 |
| joint_continue | joint at 3000 | FMPE + encoder | global norm 5 |
| joint_separate_clip | same joint at 3000 | FMPE + encoder | norm 5 separately per optimizer group |
| joint_warm_start | frozen at 3000 | FMPE + initial encoder | global norm 5 |

Warm-start inherits both frozen-head parameters and its AdamW state, then adds
the encoder as a fresh optimizer group. Its starting context must equal the
frozen context within 1e-6. Encoder exposure differs by design: warm-start learns
the encoder only in the last 3,000 updates, versus 6,000 for continued joint arms.
All heads have the same total row presentations; inherited exposure is reported.
Keep head LR 5e-4, encoder LR 1e-5 and weight decay 1e-4. This is not a learning-rate
sweep. Separate versus global clipping is isolated by a common parent checkpoint,
optimizer state, examples and random draws. One-step real-patch replay and frozen
old-head invariance are required before training; smoke updates are discarded.

## Evaluation and interpretation

Evaluate only terminal 6,000-update models on the original 128 ph005 internal
cores, at most 16 fixed rows/core, 128 posterior draws and common Heun64 noise.
Use Heun128 refinement on every 16th core with original tolerances (scaled mean
draw difference <=0.01, pooled coverage differences <=0.01, physical log-score
difference <=0.01). Measure physical energy/CRPS, coverage/widths, shell diagnostics
and exact-3D-divergence physical log scores on two fixed rows/core. Report paired
cap+superblock 2,000-bootstrap energy differences against each parent at 3,000
and the five predeclared contrasts in the configuration. No target-based row or
checkpoint selection, posthoc calibration, or blind-data comparisons.

Longer point/frozen controls distinguish budget from joint representation effects.
Warm-start compares frozen-head training followed by joint training against both
matched frozen continuation and joint-from-the-start continuation. These are
training-recipe contrasts, not pure information-sufficiency claims. Do not treat
a selected internal winner as a validated production change. No further variants
or extra seeds are automatically triggered; phase/seed replication remains a
separately specified next step if these controls justify it.

## Bound and provenance

Reuse allocation 57986464, the same single shared A100 80GB allocation used for
the read-only stage. Do not extend it or request another allocation. Require at
least 32 minutes remaining before launch; the stage's own budget is 1,800 seconds
with a 120-second checkpoint reserve. Save partial checkpoints and exit 75 if
needed, with no automatic retry. Every artifact goes to a new dedicated
p12b_representation_followup_v1 Scratch root. Original pilot files are read-only.
Technical completion and sampler gates never create a calibration/production pass.
