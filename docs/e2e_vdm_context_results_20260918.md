# Controlled VDM field-posterior experiment: scientific closeout

Status at 2026-09-18 13:29 UTC: **analysis in progress, not a final release**.
All training and posterior draws are complete and payload-verified. A/B/C case
reports are complete; the frozen D report and final registered decisions are
still running. This draft records verified results without selecting a model,
checkpoint, or new training prescription. The active goal is not complete.

## Experiment and claim scope

This implements the approved [contract](e2e_vdm_context_diversity_v1.md) and
[audit clarifications](e2e_vdm_context_audit_proposal_20260917.md).
The desired object is a conditional density-field posterior given galaxy and
survey observations, not a deterministic density reconstruction. This remains
a fixed-cosmology/HOD/epoch, conditional-on-mock experiment: the P3b angular
response is not an audited DESI fibre/redshift-success response.

| Arm | Distinct training cores / source phases | Observation information | Generated matter |
| --- | --- | --- | --- |
| A | 32 / ph000, ph002 | Local spatial channels plus wide means | Fine parent |
| B | 384 / ph000, ph002, ph003 | Same as A | Fine parent |
| C | Same 384 / 3 | Adds spatial wide context | Fine parent |
| D | Same 384 / 3 | Spatial wide context; coarse factor sees wide channels only | Shared coarse plus conditional fine |

Two optimization seeds; 20,480 updates and 40,960 presentations per factor;
checkpoints 5,120/10,240/20,480. Ten factors total, with 8,371,907 parameters
per fine factor and an additional 8,188,395 per D coarse factor. Fixed full-VLB
VDM, gamma [-13.3,13.3], deterministic FP32, AdamW 1e-4, weight decay 1e-5,
gradient clip 0.5, batch 2. No post-result fitting or best-checkpoint selection.

All arms share the same 48-cubed, 324.768 Mpc/h density parent and owned
16-cubed, 108.256 Mpc/h central core, with 6.766 Mpc/h cells. The target is the
physical-density 2-cubed average of the cubic-sampled R7 field at z=0.2, not a
second R7 smoothing. Observer redshift 0.15--0.55 is a selection coordinate.
D additionally samples a 48-cubed coarse field over 1299.072 Mpc/h. The parent
and wide fundamental wavenumbers are 0.0193467 and 0.00483667 h/Mpc.

All shared normalization/scoring scales use A32 only. No patchwise target
mean/std fit or density-DC subtraction. Spectral windows remove their weighted
DC; regional masses are therefore reported separately. Train phases are
000/002/003, development 004, internal confirmation 005; no ph001/ph006 access.
The two evaluation phases have programme history, not globally blind status.
384 patches are not 384 independent universes; overlapping parent/context
volumes and seven offset augmentations do not increase the phase count.

D enforces positive density and exact block mass through its coarse/log-residual
chart. Adjacent owned cores reuse the same sampled coarse realization, with
distinct fine noise. Its restricted coarse conditioner and conditional fine-core
independence remain approximations. C:D also changes capacity, compute, chart
and physical tidal closure; it is a package contrast, not pure stochasticity.

## Verified A/B/C results while D finishes

Final-checkpoint summaries give equal weight to anchors, seeds and evaluation
phases. CRPS is the fair finite-ensemble score standardized by A32 scales.
Coverage is over correlated core probes, not independent full-field SBC trials.

| Arm | Density CRPS | Density-mean RMSE | Density C90 | Physical tidal energy |
| --- | ---: | ---: | ---: | ---: |
| A | 0.35214 | 0.33604 | 79.09% | 0.76876 |
| B | 0.31384 | 0.31177 | 89.32% | 0.69840 |
| C | 0.32443 | 0.32122 | 87.35% | 0.73209 |

The final attainable C90 target is 90.7513%, versus 87.8788% for the earlier
32-draw checkpoints. Compare gaps to these targets, not raw coverage alone.
B improves pooled density CRPS by 10.88% over A, with positive gains in all
four seed/phase cells. C is 3.37% worse than B. The final registered H1/H2 gate
decisions, including all nonregression and dependence requirements, are pending.

Marginal improvements do not establish spatial calibration. The eight
162.384 Mpc/h octants of each generated parent have regional-mass C90 of
56.64% / 46.88% / 31.64% for A/B/C, against 90.75% attainable coverage.
These regions include the auxiliary parent halo; they are not the smaller
owned science core. Regional CRPS is 0.03819 / 0.04368 / 0.05239 and RMS
spread is 0.02997 / 0.02647 / 0.02177 versus mean-prediction RMSE
0.06167 / 0.06720 / 0.07641. The common comparison therefore reveals substantial
large-region bias/underdispersion even when core voxel coverage looks better.

Parent one-point mean density contrast is -0.02114 / -0.04310 / -0.05670,
versus truth -0.00218; mean within-field standard deviation is
0.43855 / 0.40163 / 0.37519, versus truth 0.45237. B's core density bias improves
from A's -0.06518 to -0.03553, while its parent mass bias worsens. Distinguish
core reconstruction improvement from the wider generated-field distribution.

Pooled sample/truth power ratios in the five registered parent bands are
A [0.880,0.961,0.970,0.617,0.462], B [0.691,0.803,0.814,0.785,0.707], and
C [0.478,0.605,0.695,0.756,0.725]. These are individual-sample powers averaged
over the panel, not posterior-mean powers. Per-cell B ratios span 0.562--0.893.
Posterior-mean and residual spectra, with finite-M corrections, remain separate.
No requirement is imposed that each draw match its paired truth spectrum.

## Completion evidence already verified

Run root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1`.
Frozen scientific source: `16398511d4552bd041e8d1e97cebd3e168088064`.

- All ten fits finished; all 30 checkpoint payload hashes verified in
  `analysis/CHECKPOINT_INTEGRITY.json`, including source and data-receipt bindings.
- All sixteen coupled 250/500/1000-step sampler screens pass. Worst late density
  power/width changes are 1.399%/0.757%; worst late paired MC95 bound is 1.447%.
  This is numerical stability within registered tolerances, not calibration.
- Exact draw ledger: 688 cases, 33,280 fine and 7,872 distinct coarse draws.
  `DRAW_LEDGER_INTEGRITY.json` verifies identities, bindings and paired RNG
  receipts. `DRAW_INTEGRITY.json` independently hashes all 5,144 payloads,
  34,634,901,688 bytes, with no mismatch.
- Actual SIGUSR1/fresh-process replay and full-size scalar/batch sampling tests
  pass. Sampling survived four planned pause/resume handoffs; nonzero code75
  is the registered clean pause, not an unexpected failure.
- Completed GPU cost is 78.0075 allocated GPUh. CPU cost before the live final
  report is 1.72833 nodeh, including the preview. Audit steps share the report
  allocation and must not be added again as node-hour charges.

Still required: complete D scoring, all-case/preview reconciliation, checkpoint
and conditioning summaries, tidal/eigengap/dependence interpretation, figures,
terminal accounting/storage/deadline audit, H1/H2 decisions, one justified next
step, and final science-log/field-plan updates. No production claim or new fit.
