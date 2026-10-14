# Wholefield pilot scientific closeout

## Scope and current result

The final-checkpoint evaluation is complete: 768 primary cases and four paired
integration-refinement cases. All four Abacus fits reached 32 epochs and 9,984
updates. The U-Net clean-endpoint configuration outperforms the transformer
velocity configuration on density and regional scores across the exposed
evaluation phases, but improves the reference regional energy score by only
3.08%. Neither configuration is qualified as a calibrated field posterior.

The geometry-selected epoch-24 comparison is also complete: 64 paired cases per
architecture, under the user's explicit approval for two GPU interactive
allocations, each at most three hours. Both completed successfully and were
released. No training, optimizer change, new architecture, uncertainty
rescaling, checkpoint selection or sealed-phase access is part of this closeout.

All numeric outputs are under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_pilot_20261013_v1`.
The complete primary readout is `READOUT_CLOSEOUT_20261014.json`; the final
stratified analysis including checkpoint stability is
`CLOSEOUT_FINAL_20261014.json`. Earlier partial readouts are historical.

## Population and interpretation

Training uses the original thirteen simulation phases and 624 expanded domains.
Phases 012/013 are development; phases 014--017 were excluded from these fits but
were already exposed in previous analyses. They are not fresh blind confirmation.
Phases 018/019 remain untouched. The exposed-evaluation aggregate contains four
phases, sixteen region pairs per phase and two seeds: 128 cases per configuration
and coarse mode, representing 64 distinct region pairs, not 128 independent
simulations. The two cores, nearby regions, seeds and voxels are dependent.

Primary scores use 32 draws at NFE128, the same truth fields, physical operator,
region identities and frozen reference exterior. The field is a finite-domain
replacement, not a jointly reconditioned full-cap or full-survey posterior.
Equal-case means give equal phase weight because every phase has the same number
of cases. They are not a galaxy-weighted, volume-weighted or DESI-population metric.

Fine-model training conditions on simulated true coarse density. In sampled
evaluation, the coarse input is drawn from the frozen model given observations;
truth is opened afterwards for scoring. Oracle evaluation deliberately supplies
true coarse density and is diagnostic only, unavailable on real DESI. The oracle
does not replace the exterior with truth and has no matched oracle reference.

Density CRPS is averaged over 8,192 correlated core voxels. Regional energy uses
six registered physical-unit components: inner-core mean density and two tidal
eigenvalue combinations for each of two cores. It is not a whitened multivariate
calibration test. Tidal spread is mean(lambda3-lambda1)/sqrt(2); tidal curvature
is mean(lambda1-2*lambda2+lambda3)/sqrt(6). Smaller proper scores are better, but a
small energy-score improvement alone does not establish joint calibration.

## Final checkpoint results

### Exposed evaluation phases 014 through 017

| Configuration | Coarse input | Cases | Density CRPS | Voxel interval coverage | Regional energy |
| --- | --- | ---: | ---: | ---: | ---: |
| U-Net endpoint | Sampled | 128 | 0.134611 | 0.875940 | 0.063764 |
| Transformer velocity | Sampled | 128 | 0.177831 | 0.688046 | 0.111060 |
| Shared reference | Sampled | 128 | Not recomputed | Not recomputed | 0.065790 |
| U-Net endpoint | True coarse diagnostic | 128 | 0.082708 | 0.880129 | 0.013780 |
| Transformer velocity | True coarse diagnostic | 128 | 0.157939 | 0.549614 | 0.096361 |

The coverage column uses interpolated 5th/95th sample quantiles. With only 32
draws it is not an exact population-90% interval benchmark; do not interpret
87.6% as a calibrated-posterior pass or automatically as a 2.4-point deficit.

U-Net density CRPS is 24.30% lower than the transformer. Its regional energy
score improves on the reference by 0.002027 (3.08%), with lower scores in 86/128
matched cases. Transformer energy is worse than the reference by 0.045270
(68.81%), with lower scores in only 2/128 cases. These are descriptive comparisons,
not independent-case significance tests or population-power claims.

U-Net improves the reference's tidal-spread CRPS in both cores, but slightly
worsens tidal-curvature CRPS: 0.004377 versus 0.004269 in core 0, and 0.004671
versus 0.004538 in core 1. It is not an across-the-board scientific improvement.

### Development phases 012 and 013

These 64-case groups are reported separately, not pooled into the main evaluation.

| Configuration | Coarse input | Density CRPS | Voxel interval coverage | Regional energy |
| --- | --- | ---: | ---: | ---: |
| U-Net endpoint | Sampled | 0.136258 | 0.866333 | 0.059609 |
| Transformer velocity | Sampled | 0.178908 | 0.688465 | 0.107770 |
| U-Net endpoint | True coarse diagnostic | 0.084429 | 0.868933 | 0.015501 |
| Transformer velocity | True coarse diagnostic | 0.158902 | 0.552771 | 0.096790 |

The direction agrees with the exposed evaluation; this agreement does not make
the previously inspected phases a new confirmation sample.

## Regional uncertainty and tidal structure

The following probabilities concern ranks 2 through 30 among truth and 32
draws. Under a continuous, correct conditional posterior with exchangeable
draws, their reference probability is exactly 29/33 = 0.8788. This avoids
confusing interpolated sample quantiles with exact 90% intervals. Dependence and
the small number of phases still preclude treating the entries as independent
binomial trials; no new statistical pass rule is imposed.

| Configuration and coarse mode | Core 0 density | Core 0 spread | Core 0 curvature | Core 1 density | Core 1 spread | Core 1 curvature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net sampled | 0.8516 | 0.8281 | 0.7578 | 0.8594 | 0.7969 | 0.7266 |
| U-Net true coarse | 0.8750 | 0.8359 | 0.7813 | 0.9141 | 0.7969 | 0.7813 |
| Transformer sampled | 0.7266 | 0.0313 | 0.6016 | 0.6953 | 0.0781 | 0.5547 |
| Transformer true coarse | 0.2734 | 0 | 0.3516 | 0.3750 | 0 | 0.2500 |

U-Net has useful density reconstruction, but its regional tidal uncertainty
remains imperfect, especially curvature. Sampled-coarse mean-squared errors
divided by mean posterior variances range from 1.17 to 1.71 across the six
components. These ratios include finite-draw posterior-mean noise and are
descriptive, not proof of a uniquely variance-only defect.

Transformer posterior means underestimate tidal spread in every one of the 128
exposed-evaluation cases in both cores. Mean biases are -0.06922 and -0.07072,
against mean truths 0.32330 and 0.32970: approximately 21.4% low. With true coarse
inputs the biases remain -0.06612 and -0.06597, while uncertainty narrows enough
that no nominal-90% spread interval covers truth. This is systematic structural
bias plus inadequate uncertainty, not merely uncertain coarse conditioning.

The shared-exterior truth-through-operator spread discrepancy is about 0.00455
and 0.00463 RMS across cases of the exterior-averaged shift, much smaller than
the transformer's bias. That measured mean closure
discrepancy cannot explain the entire defect. It remains material relative to
the much smaller U-Net oracle errors, and does not measure all errors in the
modular joint-law approximation.

True coarse input reduces U-Net density CRPS by 38.56% and regional energy by
78.39%. These are changes under a different conditioning problem, not additive
fractions of total error attributable to the coarse model, nor attainable DESI
performance. Regional tidal discrepancies persist under the oracle.

## Phase, selection and seed dependence

### Per-phase sampled-coarse regional energy

| Phase | U-Net | Transformer | U-Net minus reference |
| --- | ---: | ---: | ---: |
| 014 | 0.060826 | 0.109376 | -0.002320 |
| 015 | 0.060116 | 0.107123 | -0.001410 |
| 016 | 0.051091 | 0.096350 | -0.001613 |
| 017 | 0.083023 | 0.131393 | -0.002763 |

Both seed-specific means favor U-Net over transformer. U-Net's reference-energy
differences are -0.001714 for seed 17 and -0.002339 for seed 29. The seeds share
training simulations and evaluation truths and are not independent universes.
Each seed also uses its corresponding frozen coarse/reference draws, so this
split is not an isolated estimate of fine-network initialization variance.

### Redshift shells

Ranges below are actual region-center redshifts, not assertions about complete
survey-volume coverage. Each exposed-evaluation shell contains 32 cases per
configuration and coarse mode.

| Shell center redshift range | U-Net density CRPS | Transformer density CRPS | U-Net energy minus reference |
| --- | ---: | ---: | ---: |
| 0.154--0.248 | 0.106830 | 0.152142 | -0.003873 |
| 0.257--0.341 | 0.110557 | 0.163356 | -0.003105 |
| 0.360--0.449 | 0.132700 | 0.182137 | +0.000315 |
| 0.452--0.545 | 0.188359 | 0.213688 | -0.001443 |

U-Net is better than transformer in every shell on these density scores, but
does not improve reference energy in every shell. The shell-2 regression is
small and descriptive. Higher-redshift absolute errors are larger; this is
consistent with a harder observational problem but does not isolate selection
from changes in field structure, region geometry or other covariates.

Boundary density CRPS is worse than interior for both models: U-Net
0.151788 versus 0.117435; transformer 0.188132 versus 0.167529. U-Net improves
reference energy in both strata and both caps. Transformer defects therefore
are not confined to boundaries or one cap. Full stratified values are retained
in the machine-readable analysis.

## Numerical and checkpoint stability

### Integration refinement

The pre-existing paired NFE128/256 diagnostic covers one development region,
phase 012 NGC shell-0 boundary, seed 17, in both coarse modes. Its largest
component-wise same-noise RMS change divided by posterior SD is:

| Configuration | Sampled coarse | True coarse |
| --- | ---: | ---: |
| U-Net endpoint | 0.16270 | 0.92863 |
| Transformer velocity | 0.000371 | 0.001105 |

Integration refinement does not repair the transformer's first-case defect.
U-Net numerical sensitivity is material for the narrow oracle shear posterior;
convergence has not been established. This single case cannot establish
panel-wide convergence for either model, and paired draw displacement alone
is not a distributional distance.

### Training losses and epoch 24 comparison

| Configuration | Seed | Epoch-24 development velocity loss | Epoch-32 loss |
| --- | ---: | ---: | ---: |
| U-Net endpoint | 17 | 0.0139693 | 0.0135501 |
| U-Net endpoint | 29 | 0.0139536 | 0.0135932 |
| Transformer velocity | 17 | 0.0578154 | 0.0564073 |
| Transformer velocity | 29 | 0.0577191 | 0.0563840 |

Losses continue to improve modestly over the final eight epochs. CFM loss is not
a posterior score and does not establish convergence. The missing U-Net seed-17
epoch-4 development loss is retained as missing: development-data preparation
had not finished. It is not filled with zero or extrapolated.

The stability subset uses phases 012/013, both caps and all four shells, with one
boundary/interior pair per phase/cap/shell chosen by fixed alternating parity.
Both seeds and coarse modes give 64 epoch-24 cases per architecture. Existing
epoch-32 outputs are reused with identical noise addressing, coarse draws,
exterior and operator. Selection does not use scores. The purpose is sensitivity
assessment, not choosing a checkpoint or altering the final-panel comparison.
Because U-Net integration is not established as converged, its fixed-NFE
checkpoint comparison can mix changes in the learned field with changes in
integration error; it is not a pure optimization-error attribution.

Transformer comparisons are complete (32 paired cases per coarse mode):

| Coarse input | Density CRPS epoch 24 | Epoch 32 | Energy epoch 24 | Epoch 32 |
| --- | ---: | ---: | ---: | ---: |
| Sampled | 0.182232 | 0.181997 | 0.110097 | 0.110229 |
| True coarse | 0.161840 | 0.161604 | 0.095681 | 0.095918 |

Density CRPS improves by only 0.129%/0.146%, while regional energy worsens by
0.120%/0.248%. Both phases and both seeds show this aggregate direction. Median
same-noise regional draw displacement is 0.0046--0.0171 posterior SD across the
six components with sampled coarse, and 0.0328--0.0702 with true coarse. Maximum
case/component displacement is 0.0666/0.2659 SD respectively. The large spread
bias persists; oracle spread rank coverage is zero at both checkpoints. This
is local checkpoint stability of an inadequate prediction, not evidence of an
adequate posterior, an optimizer optimum or an architecture capacity limit.

U-Net comparisons are also complete (32 paired cases per coarse mode):

| Coarse input | Density CRPS epoch 24 | Epoch 32 | Energy epoch 24 | Epoch 32 |
| --- | ---: | ---: | ---: | ---: |
| Sampled | 0.140334 | 0.139365 | 0.064261 | 0.065016 |
| True coarse | 0.088642 | 0.086770 | 0.018177 | 0.016940 |

Density CRPS improves in every matched case: 0.690% on average with sampled
coarse and 2.111% with true coarse. However, sampled-coarse regional energy
worsens by 1.174%, in 21/32 cases; true-coarse regional energy improves by 6.807%,
in 25/32 cases. Both phases and both seeds share these aggregate directions.
This is not uniform late-training improvement across the scientific targets.

Median same-noise regional draw displacement ranges from 0.0283 to 0.1389
epoch-32 posterior SD with sampled coarse and 0.1223 to 0.4562 with true coarse.
Maximum case/component displacement is 0.6120/2.1616 SD respectively, both in
tidal spread. This sensitivity, together with unresolved endpoint integration,
precludes declaring the regional posterior converged. It does not imply every
case moves by that maximum, nor does a coupled-draw shift alone establish a
distributional distance. The sampled-coarse mean spread biases become more
negative between checkpoints, whereas the oracle spread biases move closer to
zero. A single density loss or oracle-only score would miss that tradeoff.

These numbers describe the geometry-selected half of the development regions,
not the full development table above or a new held-out panel. Epoch 32 remains
the reported final checkpoint; no retrospective checkpoint selection is made.

## Scientific disposition

U-Net endpoint is the stronger exploratory configuration on the tested support,
with a modest, nonuniform gain over the frozen reference and remaining regional
uncertainty, checkpoint sensitivity and numerical concerns. Transformer velocity has a repeatable,
large tidal-anisotropy defect and is not an adequate field-posterior candidate
in its present form. Neither result establishes full-survey joint calibration,
real-DESI validity, or a best architecture.

Backbone and output parameterization differ simultaneously; equal parameter
counts and updates are not equal optimized performance. Data limitation,
optimization and representation cannot be uniquely separated by these scores.
The evidence does weaken explanations based solely on coarse uncertainty,
the measured operator closure discrepancy, or first-case transformer integration.
No Muon, Set Transformer, alternative loss or further training experiment is
authorized or launched by this closeout.

## Reproducibility and verification

- Final training checkpoints: `fits/{unet_x,transformer_v}_{17,29}/CHECKPOINT_009984.pt`.
- Epoch-24 checkpoints: the corresponding `CHECKPOINT_007488.pt` files.
- Primary scores: `evaluation/<model>/step9984/<phase>/<pair>/<coarse>/fullcap/nfe128/COMPLETE.json`.
- Stability scores: the same addressed layout under `step7488`.
- Complete main-panel controller: `continuation_v1/COMPLETE.json`, both allocations
  completed 0:0 with source checks passing; final release 13 October 18:59:53 UTC.
- Stability jobs: U-Net 59758187 and transformer 59758193, each one GPU and at
  most three hours, started 13 October 23:42 UTC (14 October 00:42 BST).
  Transformer completed 0:0 in 57m12s; source checks passed and the allocation
  was released at 14 October 00:39:54 UTC.
  U-Net completed 0:0 in 2h19m27s; source checks passed and the allocation was
  released at 14 October 02:02:05 UTC. No jobs remain active at closeout.
- Analysis: `workflows/sbi/e2e_wholefield_pilot_readout.py` and
  `workflows/sbi/e2e_wholefield_pilot_closeout.py`; geometry-only stability worker:
  `workflows/sbi/e2e_wholefield_pilot_stability.py` and its bounded GPU launcher.
- All final regional summary scores are independently checked from saved 32-by-6
  draws using sorted-ensemble CRPS and a separate pairwise energy calculation.
  Exact matched truth and reference scores are checked across architectures.
  Density-voxel numbers are the saved outputs of the tested scoring code; this
  closeout does not claim an independent re-read of every raw density draw.
- Twelve focused readout, selection and closeout tests pass. No new pass gate
  or independent-voxel significance calculation is introduced.
- An independent scalar weighted-sum reconciliation passes 176 checks of
  phase-to-split denominators, scores, coverage and component biases.
- Initial `graphify` calls were unavailable in the unactivated shell, so scoped
  direct inspection was used. The final AST-only refresh succeeded after
  activating `cosmic_env` (14,126 nodes, 30,930 edges); large-graph HTML rendering
  was skipped automatically. Unrelated pre-existing edits are preserved.
