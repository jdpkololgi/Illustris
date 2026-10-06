# Tier 1 adoption and assessment of the next CFM levers

## Completed now

The question is what can be used honestly from the existing CFM while deciding
whether another model experiment has enough scientific value.

1. Adopt the density-only positive central-linear correction as the default
   **development-core analysis operator**, using the completed Stage A/B replay.
   Original remains an explicit comparison. This is not a whole-cap/Loa
   production replacement or posterior qualification.
2. Build a machine-readable, source-bound evidence inventory and usable analysis
   interface, with explicit tested/failing/untested claim boundaries below.
3. Do not fit the optional summary recalibrator in this turn. It is a separate
   statistical estimator, needs a frozen family/conditioning set/score and
   precision plan, and cannot turn two development phases into qualification.
   Ph014–017 are already exposed to earlier evaluations: they could provide
   out-of-fit replication, not blind confirmation. Ph018/019 and001 stay sealed.

User authorized cancellation of unnecessary work. Mixture allocation59422691
was still running and was cancelled after3h16m41s; Slurm confirms cancellation.
Five complete seed29 four-draw batches and four partial batches are preserved.
Partial batches are not usable completed posterior evidence. No output was
deleted and no automatic retry or further generation allocation was submitted.

## What changed operationally

Profile: `configs/e2e_cfm_analysis_v1.json`.
Interface: `workflows/sbi/e2e_cfm_analysis_products.py`.
Built manifest and evidence: Scratch
`e2e_field_v3/cfm_tier1_analysis_20261006_v1/{MANIFEST,EVIDENCE,COMPLETE}.json`.

`load_pair(manifest_path, pair_id)` now defaults to the corrected operator for
each of the32registered pairs/64cores in ph012/ph013, with32draws. It returns
regional eigenvalue means/gaps, consistently derived cell-class probabilities,
galaxy- and volume-weighted fraction draws, unchanged physical truth and exact
Stage B cell counts. `operator="original"` explicitly retrieves the comparator;
threshold0.2 is primary and0 is an explicit sensitivity. It fails on unregistered
phases/thresholds, changed source hashes, invalid classes, mismatched shapes,
changed trace or disagreement with saved volume fractions.

This reuses the already computed per-draw correction; no new model, FFT solve,
posterior draw, density rescaling or recalibration was performed. Existing
immutable HDF5 fields, generation/scoring contracts and production defaults are
unchanged. Consumers must use this new analysis interface/profile to obtain the
corrected default; no claim that older scripts silently changed behaviour.
Both original and corrected derived classes remain marked unqualified, since
operator improvement is not uniform improvement in class-fraction inference.

Five focused tests and160real-panel operator/threshold/default checks pass.
Assembly/verification took1.77s over compact saved products, with no new Slurm
allocation or intensive login-node work. Existing physical/replay tests are
inherited through hash-bound Stage A/B receipts, not claimed as new experiments.
The broad repository graph refresh remains unperformed after its earlier
safety-review rejection; direct source checks and tests are the verification.

## Evidence inventory and talk-safe statements

All percentages below are development-panel observations, not estimates from
independent voxels or a claim of population equivalence. Keep sampler, support,
physical R7 target,32draw reference, and metric-specific rank convention attached.

| Quantity | Current evidence | Permitted interpretation |
| --- | --- | --- |
| Positivity, block mass, trace, matched replay | Engineering checks pass | The tested numerical construction is consistent; not statistical calibration |
| Cellwise density marginals | Both phase-level50/68/90/95curves pass the0.05screen;90coverage87.24/88.61% vs87.879%reference | Encouraging development marginal calibration, not joint-field qualification |
| Cellwise individual tidal eigenvalue marginals | All3eigenvalues pass the same phase-level curve screen under both operators; corrected90coverage84.70–87.28% | Positive marginal evidence at tested tolerance, not conditional/spatial calibration |
| Core/regional density | Core-mass90coverage78.125/93.75%; both full curves fail; paired trace90looks close but middle levels fail | Phase-dependent regional uncertainty, not uniformly validated density summaries |
| Corrected regional anisotropy | Mean width deficit7.65/7.68%→5.22/5.24%; proper scores improve | Useful numerical correction with residual bias |
| Corrected paired tidal gaps |90coverage56.25/68.75% vs87.879%reference | Still undercovered |
| Galaxy class probabilities | Fair Brier strongly improves over train prevalence; fixed-bin residual failures remain | Informative, not reliably calibrated probabilities |
| Galaxy-weighted regional fractions | Corrected joint90coverage71.25/68.75%; some marginal scores worsen | Joint calibration fails; correction is not a universal downstream improvement |
| Interior/boundary fraction moments | Interior residual error/variance excess persists; boundary aggregate moments nearer unity | Localisation of a discrepancy, not proof boundaries are calibrated |
| Property–environment relationships | No matched downstream estimator evaluated | Untested; not automatically known biased and not validated |

Suggested talk statement:

> The CFM gives informative environment predictions and passes development
> checks for cellwise density and individual tidal-eigenvalue marginals.
> Regional and joint uncertainty is not yet reliable. A density-only numerical
> correction reduces tidal anisotropy bias without resolving joint coverage.

The statement “each component works alone” is too strong: coarse trace and
middle-credibility shear tests fail, and even true-coarse-conditioned joint
gap summaries are imperfect. The handoff is implicated, not the sole proven cause.

## Check that toy lessons were actually transferred

Verified the actual `BINDING.json` files for both stages and seeds17/29, not just
an old plan. All use alpha0.30, EMA0.999, and learning rate1e-4→1e-5. The runner
implements cosine decay and training-only spectral profiles from26pairs.
Fourier velocity errors receive amplitude weightP^-alpha before squaring, so
the squared-error objective usesP^(-2alpha), with the recorded floor and
normalization. Alpha0.30 remains provisional, not an Abacus-selected optimum.
An alpha experiment is tuning an existing lever, not adding whitening anew.

## Tier 2: useful hypotheses, but amend the proposed causal claims

### 1. Coarse-to-fine robustness: a candidate experiment, not a known cheap fix

Prioritize a bounded, explicitly specified test of conditioning/composition
over a large architecture or resolution rewrite. However, “physically valid
shape” and “statistically correct conditional target” are different.

Let C be block mass, U the block-centred log-density shape and G observations.
The desired law is p(C,U|G)=p(C|G)p(U|C,G). The decoder
rho=C exp(U)/mean_block(exp(U)) conserves whichever C is supplied. Keeping U
while replacing C by an independently drawn C' creates a positive field with
mass C', but does not preserve the original joint law. With conditional
independence of C' and the training truth, its training pairing tends toward
p(U|G)q(C'|G), erasing the dependence that the experiment is meant to repair.

Perturbing the *paired true* coarse field is different and retains some joint
structure. It trains a corrupted-conditioning conditional distribution; the
augmentation and inference law still need to be declared and tested. Do not
present it as automatically posterior-correct. Fit perturbations using training
data only; consider out-of-fit coarse errors where an error model is estimated.
Keep a zero-augmentation, equal-update continuation and untouched inference
settings as the matched control. Declare whether the noisy signal is merely a
conditioning channel or also the mass used by the decoder.

The [Ho et al. cascaded-diffusion paper](https://arxiv.org/html/2106.15282v3)
supports conditioning augmentation as an empirical image-generation technique.
Its noise/blur constructions and latent-variable discussion motivate a test;
ImageNet sample-quality results do not establish calibrated, mass-constrained
cosmological posteriors. The paper-reading check therefore strengthens the
distinction between a valid candidate surrogate objective and a guaranteed fix.

The existing equal-size directional perturbations also show that scalar coarse
error magnitude alone does not explain fine suppression. “Unlike training
inputs” remains a plausible hypothesis, not a measured density-ratio diagnosis.

### Pass criteria need a calibration guard, not just amplitude restoration

Fine power>=0.95 is one-sided: excessive power would pass it. Freeze a two-sided
well-resolved-band criterion with numerical/null precision, plus posterior-mean,
draw-power and residual-power separation. Power across the boundary scale is
useful but does not suffice to establish conditional dependence or coverage.
Keep the matched sampler/context contrast; do not assign it purely to averaging.
Interior error/variance<1.5 alone also admits severe overdispersion. Require
two-sided diagnostic accounting, bias, the original coverage curves, proper
scores, mass/positivity and nonregression. Proposed1-point/1.5thresholds are
engineering goals until their precision and downstream relevance are justified.

### Other levers

| Lever | Assessment |
| --- | --- |
| Joint coarse/fine flow | Potentially useful, but one failed augmentation does not prove factorisation structurally inadequate or license this rewrite automatically |
| Alpha tuning | Existing, bounded lever; evaluate calibration and proper scores as well as spectra. CFM does not universally underweight every small-variance mode, and tidal anisotropy is not exclusively fine-scale |
|3.383Mpc/h training | Attacks a demonstrated representation error, but no theorem says exactly half of the remaining posterior deficit disappears; eightfold voxel count is not a measured eightfold wall-time estimate |
| More independent volumes/coarse generalisation | Sensible hypothesis if phase generalisation is limiting; later-checkpoint deterioration does not prove data volume is the only remedy |
| Outpainting | Changes the sampler's probability law. The~1.3point local-versus-canvas contrast also changes context and both factors; it is not a clean velocity-averaging attribution |
| Observation stochasticity | Worth separating from cosmology/HOD uncertainty; preserve paired matter truth and survey selection and audit existing observation realisations before constructing more |

Do not repeat the failed exact mixture/global-shift/global-spread recipes or
extend the deteriorating training trajectory without a new hypothesis. But a
blanket ban on all spatial energy losses is not a consequence of these results:
the historical per-galaxy3D joint-score objectives are not equivalent to a
spatial field-law objective, and tested auxiliaries do not exhaust that class.
No loss change or Tier 2 training is launched here.

## Recommended next decision

First specify a statistically explicit, train-only paired-conditioning
experiment and its matched zero-strength control. Proceed only if its target
law and discriminating tests are clear; numerical mass conservation alone is
insufficient. Success would justify considering alpha tuning and resolution
as separate follow-ons, not automatically authorize them. Failure would close
that candidate objective, not prove a structural composition impossibility.
