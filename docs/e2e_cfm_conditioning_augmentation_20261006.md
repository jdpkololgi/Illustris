# Tier 2: paired conditioning augmentation frozen protocol

## 2026-10-06 execution amendment (supersedes proposal-only wording below)

User explicitly instructs: state/freeze the remaining tolerances and start
Tier 2. The saved-draw non-truncated variant and matched experiment below are
now selected. Fixed machine profile:
`configs/e2e_cfm_conditioning_augmentation_v1.json`. No retrospective retuning
after new metrics. No additional seeds, alpha sweep, higher resolution,
production inference or new-phase access is included.

**Oracle gate, each phase, clean and every candidate s:** physical residual
power/truth must lie in [0.95,1.05] in EACH (.04,.08], (.08,.16] h/Mpc band
(the existing spectral helper's exact endpoint convention).
Signed matched-local tidal-gap width deficit D=100*(1-mean_width/truth_width)
must lie in [-0.5,+0.5] percentage points. Against EACH frozen parent and
matched unaugmented continuation require abs(D_new)-abs(D_reference)<=0.2
percentage points. Using magnitude prevents an excess-width overshoot from
being counted as improvement. These margins permit at most a small loss from
the historical .247/.330% oracle deficits while bounding the total distortion;
they are an explicit scientific tolerance choice, not fitted error bars.

**Uncertainty rule:** 4,096 paired whole-draw bootstrap replicates, NumPy
default_rng seed6100617; resample the32draw addresses with replacement and
preserve every region, phase, cap and paired arm within each selected draw.
Truth fields and region weights remain fixed. Recompute the complete statistic
for each replicate. Each lower-bounded oracle statistic must have its5th
percentile above its lower limit, and each upper-bounded statistic its95th
percentile below its upper limit. Regression uses its95th-percentile upper
bound <=0.2pp. These are pointwise one-sided95% bounds, NOT familywise bounds
after choosing s and NOT phase/population confidence intervals.

Any point-estimate oracle breach is fail; a point estimate inside bounds with
an MC bound crossing a limit is inconclusive; only all point/bound checks pass
is oracle pass. No extra draws automatically follow an inconclusive result.
Proper-score deterioration also uses a paired upper95% bound <=0 against the
continuation control; point estimate worse is fail, otherwise crossing is
inconclusive. Retain the existing point-estimate marginal and interior-moment
screens with MC sensitivities reported, rather than silently claiming coverage
equivalence from bootstrap precision. A candidate needs all gates to pass;
one inconclusive gate prevents promotion. Full regional/joint curves remain
mandatory reports, not an unannounced full-posterior qualification gate.

Engineering before scientific fitting: exact parent s=0 forward parity,
deterministic serialized update/EMA restart, paired fine RNG, noised crop
alignment and unchanged decoder mass; unchanged cached-coarse fine sampler
replay before full training launch. Keep the backbone file unmodified.
Initial Slurm scope is a bounded single-GPU interactive smoke/throughput/replay
allocation under the user instruction to start. Size the two continuation arms
from measured throughput, no automatic resubmission or speculative large
generation sweep. A resource estimate/launch record must precede full scoring.

The dated proposal text below preserves the motivation and estimands. Its
old statements that launch/tolerances await approval are superseded ONLY by
this amendment; no successful run is implied by the freeze.

## Decision and status

The next proposed model intervention is fine-model conditioning augmentation,
with an equal-budget unaugmented continuation. This is a bounded test of one
possible coarse-to-fine remedy, not proof of the cause of the current deficits.
The user requested repository/science-log closeout and supplied this proposal.
No training, sampling, allocation request, or phase opening is authorized by
this document. Implementation, a compute-node smoke test, explicit resource
approval and a hash-bound final run manifest remain prerequisites.

The user approved saved-draw, non-truncated conditioning noise after the
distinction below was raised. Proposed numerical implementation choices are
written down now, not represented as an
executed or fully frozen experiment. They must be fixed before any new scores.

## Clarify the inference rule before implementing

[Ho et al., sections 3.2 and 3.3](https://arxiv.org/html/2106.15282v3)
distinguish stopping the coarse sampler at an intermediate noisy state
(truncated augmentation) from completing the coarse draw and then applying
forward corruption (non-truncated augmentation). Their training procedures
share the extra noise-level conditioning. These are different inference laws.

User-selected first experiment: **non-truncated, conditioning-only CFM bridge
augmentation**, reusing completed coarse draws. This preserves the proposed
mass decoder and avoids regenerating coarse trajectories. It is inspired by
Ho et al.; the diffusion ELBO argument is not a posterior guarantee for this
CFM application. If true truncation is preferred, intermediate states must be
regenerated/saved from the same trajectory as its endpoint. Do not approximate
those states by adding noise to endpoints while calling that truncation.

The deployed coarse CFM uses standardized log-density coordinates and the
independent-base linear bridge z(t)=(1-t)*epsilon+t*C. Define augmentation
strength s=1-t, so C_s=(1-s)*C+s*eta and s=0 means clean conditioning. eta is
standard normal in this SAME coordinate system, not physical-density noise,
not train-spectrum-whitened noise, and not an estimated model-error law.

Training retains the paired true fine log-shape U and its current projected
CFM velocity target. Replace only the wide coarse conditioning tensor and
its aligned replicated joint crop by C_s. Give the network a separate scalar
s embedding; do not conflate this with the fine integration time t. Keep the
observed-galaxy channels, normalizer, fine noise law, loss weights and decoder.
One wide noise field supplies both conditioning views; never draw them apart.

At inference, retain completed C_hat for physical block masses. Make the
conditioning input (1-s)*C_hat+s*eta, fixed for the entire fine solve. Draw
eta once per global coarse canvas and draw address; overlapping windows must
see identical values. Use common fine noise and augmentation noise addresses
across s and arms. Decode with the UNCORRUPTED C_hat and generated U:
rho=C_mass*exp(U)/mean_block(exp(U)). No condition noise enters C_mass.

This keeps positivity and block mass, but for s>0 the learned conditional law
is p(U|C_s,G), not p(U|C,G). The association with the endpoint used for mass
is still an empirical calibration question. A flow intermediate and endpoint
also need their joint law considered. Neither variant inherits correctness
from exact mass conservation alone.

## Saved coarse inputs: current evidence

Read-only metadata verification on 2026-10-06 found all 32 complete seed-17
four-draw batches: ph012/ph013 x NGC/SGC x d000,d004,...,d028. Each FIELD.h5
contains float32 `coarse_logrho_standardized`, four draws and the expected
first-draw attribute. Thus 128 cap-level coarse realizations are present.

Root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1/fullcap_density`.
The producer is `workflows/sbi/e2e_exterior_fullcap.py`. This check read HDF5
metadata and completion records only; it did NOT reread/hash all field data.
Before reuse, verify receipt/payload hashes, seed/checkpoints, NFE128,
normalizer, support, origin, coarse/fine canvas geometry and draw addresses.
Require unchanged-model fine replay against existing fields before new fits.
There is not yet a fine-only cached-coarse production entrypoint; its existence
must not be inferred from the presence of the dataset.

## Proposed matched experiment

- Two arms, one seed (17): augmented continuation and s=0 continuation.
  Frozen parent is deployed fine CHECKPOINT_026624 EMA; coarse parent stays
  CHECKPOINT_013312 EMA. Warm-start both fine arms from the same EMA weights,
  with identically reset optimizer state and EMA initialized to those weights.
  This is fine-tuning the deployed model, not an exact optimizer-state resume.
- Initial bounded budget: 3,328 updates per arm, batch 2, identical training
  pairs/order and fine bridge RNG. AdamW, weight decay 1e-5, grad clipping .5,
  fixed LR 1e-5, EMA .999 and inherited alpha .30/spectral weights. These are
  proposed fixed choices, not an alpha or optimizer sweep.
- Augmented arm: probability .25 at s=0, otherwise s uniform on (0,.20).
  Independent addressed augmentation RNG must not change the shared data,
  fine-time or fine-noise sequence. Control always uses s=0.
- Separate s embedding must contribute exactly zero at s=0. Verify parent
  output parity at initialization; after fine-tuning, s=0 is clean conditioning,
  NOT a claim that the trained network still equals the frozen parent.
  Preserve existing backbone source/checkpoint-loader hash contracts: use a
  separate new implementation rather than editing the frozen backbone file.
- Inference grid proposed now: s in {0,.025,.05,.10,.20}; control at zero;
  retain frozen parent as the no-training anchor. Use all existing 32 pairs,
  64 cores and 32 draws in ph012/ph013. Training uses existing train phases
  only. No 014--017 payload access, no 018/019/001 opening, no Loa deployment.
- Select the smallest s satisfying ALL registered phasewise gates; if none
  qualifies, no selected success. Report every s, not only the winner. This
  development selection is not independent confirmation.

The cost is two fine-model training arms, plus several fine-generation passes,
tidal replay and scoring. Cached coarse draws remove coarse-generation cost
only. Measure one representative training step/batch and fine-only sampler
batch on a user-approved GPU allocation before estimating the full budget.
No automatic continuation, retry or full-panel launch follows a smoke test.

## Gate definitions to freeze in the implementation manifest

The user-specified requirements are adopted in direction. The following makes
their estimands and outstanding tolerance decisions explicit. Do not run a
selection sweep until the remaining choices below have been resolved.

1. Fine-residual power: use the existing physical density residual and
   rectangular pseudo-spectrum from `e2e_sampler_scale_diagnostic.py`, NOT
   raw standardized U power. Proposed well-resolved bands are [.04,.08) and
   [.08,.16) h/Mpc, checked separately in each phase: draw-averaged power/truth
   in [0.95,1.05]. Report all original bins and the coarse-boundary spectrum;
   do not let a pooled band or phase conceal opposite deviations.
2. True-coarse nonregression: same oracle conditioning, fine-noise addresses,
   decoder and matched operator as the saved control. Evaluate clean s=0 AND
   selected-s corruption of true coarse. Report power and signed tidal-gap
   width deficit against both frozen parent and continuation control. The
   historical approximately .3% deficit is a measured baseline, not a tolerance.
   A numerical noninferiority margin and uncertainty rule remain to be frozen.
3. Cell density AND all three individual tidal marginals retain the existing
   phase-level full-coverage-curve screen (0.05 discrepancy tolerance), using
   finite-draw attainable coverage rather than nominal percentages. No iid-cell
   standard errors or claim of joint calibration from marginal passes.
4. Interior fraction error/variance: use the identical Stage C interior
   selection, galaxy weights, threshold .2 and finite-draw correction
   Q=mean(error error^T)-mean(S)/32. Require trace(Q)/trace(mean(S)) in
   [0.67,1.5] separately for each phase; also report centered, classwise,
   boundary and directional results. This is a descriptive screen, not proof
   that any variance change is the correction of missing uncertainty.
5. Proper scores: predeclare fair energy for regional tidal gaps and interior
   class fractions and fair Brier for cell/galaxy classes, same weighting and
   support as Stage A/B. No phasewise deterioration against the continuation
   control in the primary point-estimate screen. Report paired whole-draw
   sensitivities and conditional Monte Carlo uncertainty; overlapping regions
   are not independent replications. Statistical noninferiority, if claimed,
   needs an explicit precision/margin plan; a noisy result is inconclusive.
6. Report regional density, tidal-gap, fraction and paired-joint coverage curves
   at every original credibility level, with bias/width/proper scores. These
   are required outputs, not silently converted into a full-qualification gate.
   Report corrected primary tidal operator AND original comparator, with
   matching true-field operators for local power/mechanism diagnostics and
   unchanged physical R7 truth for full-cap scientific scoring. Do not transfer
   the .3% matched-local oracle number into a physical full-cap tolerance.
7. Hard engineering checks: finite fields, positivity, exact block mass within
   existing double-precision 2e-6 tolerance, correct shared-canvas noise/crops,
   s=0 parity, no overwritten source products, deterministic restart checks,
   and provenance/phase guards. Any failure prevents scientific interpretation.

A pass supports further bounded investigation of this intervention; it does
not qualify the field posterior or the selected s on new data. A failure rules
out this particular design/budget, not all conditioning augmentation and not
the entire coarse/fine factorization. Joint flow is not an automatic fallback.

## Closeout verification in this turn

The preceding Tier 1, mixture, operator, closure and conditioning diagnostic
work was reviewed for repository closeout. All 36 pending Python files parsed;
43 focused tests completed in 2.903 s, with one GPU-only control skipped on
the login node. This is 42 passes plus one skip, not a new scientific run.
The queue was empty at the check. Existing complete and partial mixture outputs
remain preserved. No new Tier 2 implementation or result is claimed here.
