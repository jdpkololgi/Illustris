# Mass-conserving coarse/fine conditioning ablation

## Authorized scope and first-stage protocol

Follow-up authorized on October 5, including necessary compute and allocation
release. Keep ph018/ph019 and ph001 sealed; use the same ph012/ph013 development
windows, frozen checkpoints, observations, draw addresses and local diagnostic
operator. No production replacement, target redefinition or model promotion.

Start with all 32 pairs and 32 existing draws, without training or sampling.
The suggested error/amplitude correlation is descriptive, not a causal test:
distance between a posterior draw and the single realized truth includes
legitimate posterior variation even for an exact model. Correlation cannot
confirm amplification, and absence of a scalar correlation cannot rule out
direction-dependent conditioning errors. Remove pair means before pooling
within-phase draw relationships; do not treat overlapping windows as independent.

The implemented latent is u = log(rho) minus its block mean, not an additive
zero-mean density fluctuation. The decoder is rho = C * exp(u)/mean_block(exp(u)).
Thus retaining u while changing C preserves the NEW block masses exactly, but
does not retain the original physical density target. Training on an independent
coarse draw paired with the original residual also changes the joint conditional
training distribution; mass conservation alone does not establish a valid
posterior-training objective. Noise matched to posterior-versus-truth distance
would include irreducible uncertainty, not just model error.

## Saved-draw crossed controls

Recover the unit-block-mean shape q from each saved field using its actual block
mean (eliminates float32 storage roundoff). Cross three shapes with two mass
fields: true shape, fine shape generated with true coarse conditioning, and fine
shape generated with sampled canvas coarse conditioning; each is decoded with
either true or sampled coarse masses. The last two shapes used identical fine
noise and observations in the previous experiment. This separates physical
mass weighting from the network's conditioning response without conflating the
two. Crossed combinations are privileged diagnostics, not posterior candidates.

Record fine-residual traceless tensor RMS on the same two cores, rectangular
power by the same five bands, latent-u and dimensionless-q variance, and coarse
log-density distance over both wide and joint crops. Correlate coarse distance
with native fine RMS/power and normalized shape variance within each pair and
phase, including fixed-mass shape responses. Check exact reconstructed parent
mass, positivity, and replay against the prior saved fine-RMS summaries.

A second, frozen-weight conditioning intervention will be specified after the
first-stage decomposition if needed. Any augmentation-training design must have
a matched unaugmented continuation, train-only perturbation calibration, and
an explicit account of its changed conditional target. No augmentation fitting
on these development truths is allowed.

## Frozen-weight intervention protocol

Use the same four geometry-selected pairs as the earlier operator diagnostic,
one per phase/cap, with all 32 original fine-noise addresses. The previous
oracle and sampled-coarse/local-fine draws provide the endpoints. Without
selecting a best perturbation, evaluate three additional log-coarse fields:
halfway from truth to the actual sampled field; the same displacement reflected
about truth; and a phase-randomized displacement preserving its Fourier
amplitudes, mean and RMS. The latter is an oracle stress control, not fitted
augmentation noise or a calibrated model-error distribution. All observations,
normalization, seed-17 weights and NFE128 stay fixed.

Decode each new fine shape with native, true, and original sampled masses.
The cross-decoded outputs isolate shape response; they are not deployable
posteriors. Native decoding always obeys the conditioning block-mass contract.
Independently replay one original sampled and oracle endpoint per selected pair
to check the inherited noise and conditioning conventions. Preserve positive
density and record per-band power and fine shear, normalized shape/latent
variance, and native regional eigenvalues. No scalar perturbation strength is
chosen to repair coverage, and no confirmation fields enter these tests.

## External precedent

[Ho et al., Cascaded Diffusion Models](https://arxiv.org/html/2106.15282v3)
study conditioning augmentation for image-generation cascades and support it as
a hypothesis to test. Their fidelity improvements do not establish calibrated
cosmological posterior inference or the cause of this experiment's deficit.

## Completed saved-draw results

All 32 pairs / 32 draws completed. Positive parent masses and unit block means
pass; maximum relative replay discrepancy in fine shear is 5.29e-9. Three
initial controls and two intervention controls pass, covering conserved shape
under changed masses, pure decoder-amplitude scaling, within/between-pair
separation, randomized spectral amplitudes, and intervention distances.

Fine-residual traceless tensor RMS relative to truth, pooled descriptively:

| Fine shape source | Decode with true masses | Decode with sampled masses |
| --- | ---: | ---: |
| True fine shape | 1.0000 | 0.9628 |
| Oracle-coarse-conditioned fine model | 0.9931 | 0.9581 |
| Sampled-coarse-conditioned fine model | 0.9505 | 0.9426 |

Both generated-shape rows use the single-window fine sampler and identical
fine-noise addresses. These are NOT the raw-canvas and oracle rows from the
previous headline table; the sampler confound has been removed. The true-shape
row contains no fine-model response at all, yet changing masses changes shear.

Starting from oracle-conditioned shape and true masses, changing masses first
changes the ratio by -0.0350, then changing shape by -0.0155. In the reverse
order, the changes are -0.0426 then -0.0079. Their 0.0271 interaction makes a
unique additive 'coarse error amplification' allocation inappropriate. Mixing
a true fine shape with a different sampled coarse realization also breaks its
physical cross-scale association; this privileged cross is not a valid draw
from the joint posterior even if the coarse marginal were exact.

The requested saved-draw relationship check does not show the proposed negative
trend. Correlations below are between joint-crop coarse log-density distance
and log response, after removing each pair's mean, for the fixed-canvas-coarse
single-window fine arm:

| Response | ph012 | ph013 |
| --- | ---: | ---: |
| Fine shear RMS | +0.0636 | +0.0107 |
| Integrated physical fine power | +0.1143 | +0.2501 |
| Dimensionless fine-shape variance | +0.0583 | +0.1351 |
| Block-centered log-shape variance | +0.0863 | +0.1396 |

Within-pair rank correlations support the same qualitative result (fine shear
+0.0462/-0.0034). Raw-canvas and local-both arms also have weak/nonnegative
shear associations. This does not prove the absence of a directional effect;
it fails to confirm the suggested scalar-distance explanation. The average
joint log-density RMS distance is approximately 0.260, not a 2% field error.
A 2% bias in ensemble shear RMS is a different quantity, and neither distance
nor that aggregate moment isolates approximate-posterior error from uncertainty.

## Statistical meaning of augmentation

For ideal conditional joint density p(U,C|G), independently sampling a coarse
conditioning field C' from p_model(C'|G) while keeping the original residual U
creates training pairs with law p(U|G) p_model(C'|G), conditional on G. It can
therefore discard the true coarse/fine dependence. Reusing a stochastic sample
from the correct coarse posterior does not make it the paired true coarse
field for that training example. Noise applied around the paired true C is a
different intervention: it smooths the conditional training distribution rather
than factorizing it outright. Neither construction inherits posterior validity
merely because its output satisfies a mass constraint.

Consequently these controls should precede, not automatically trigger, the two
suggested training arms. Any later experiment must explicitly label its
augmented objective, retain an equal-budget unaugmented continuation, use
train-only augmentation construction, and judge joint/conditional calibration
as well as power. Image-generation fidelity is not the acceptance criterion.

## Completed fixed-weight interventions

All four geometry-selected pairs, 32 shared-noise draws and three new
conditioning arms completed. This is an eight-core subset, not the all-pair
saved-draw panel. Keeping the decoder masses fixed at truth isolates the
generated shape response:

| Coarse condition | Wide log-distance from truth | Fine shear RMS / truth, fixed true masses |
| --- | ---: | ---: |
| True coarse, oracle endpoint | 0 | 0.9967 |
| Halfway to actual sample | 0.1808 | 0.9457 |
| Actual sampled coarse endpoint | 0.3617 | 0.9449 |
| Reflected displacement | 0.3617 | 1.2003 |
| Fourier-phase-randomized displacement | 0.3617 | 1.1286 |

The sign/direction effect repeats by phase: fixed-true-mass ratios for actual
samples are 0.9550/0.9357, reflected controls 1.2605/1.1458, and randomized
controls 1.1545/1.1051. Against each draw's own oracle endpoint with identical
fine noise, mean log-shear responses are -0.0540 halfway, -0.0535 sampled,
+0.1873 reflected, and +0.1248 randomized, pooled descriptively. Every tested
reflected/randomized draw raises fine shear relative to its paired oracle
endpoint; no independent-trial significance claim is attached to that count.

Native decoding adds the mass-weighting response: pooled RMS ratios become
0.9258 halfway, 0.9566 sampled, 1.4988 reflected and 1.3571 randomized.
These synthetic conditions are deliberately out-of-distribution stress controls,
not alternative posterior samplers. They establish that distance magnitude and
coarse-displacement power alone do not determine the response. They do NOT
establish which perturbation distribution would improve training or that these
unphysical counterfactuals should be made invariant by a robust predictor.

Maximum endpoint density replay error is 5.96e-8 relative, consistent with
float32 saved payloads; maximum parent-mass error is 4.45e-16. Interventions
preserve their specified displacement norms; the randomized arm preserves the
full displacement Fourier amplitudes and DC. Native eigenvalue traces agree
with each intervention's own density, not artificially with the true mass.

## Decision and reproducibility

The proposed scalar 'small coarse error makes the fine model hedge' explanation
is not confirmed. The evidence instead shows direction-dependent conditioning
response, decoder mass effects and a substantial interaction. Do not interpret
near-correct marginal coarse power as a validated joint coarse/galaxy/fine law.

No augmentation training was launched. The suggested independent coarse-draw
training arm would change the conditional target and can erase dependence;
the simple saved-draw test did not supply its proposed justification. Keep
conditioning augmentation as an empirical hypothesis, not an established fix.
A later training comparison needs a paired/noisy-conditioning objective whose
joint target is explicit, a train-only perturbation construction, a matched
unaugmented continuation and joint/conditional calibration gates. No target,
production operator, trained checkpoint or website was changed; ph001/018/019
remain sealed. The earlier density reconstruction remains a candidate.

Code under `workflows/sbi`: `e2e_conditioning_ablation.py`,
`e2e_conditioning_intervention.py`, `e2e_conditioning_summary.py`; focused tests
in `tests/phase4/test_e2e_conditioning_ablation.py` (five passed).
Immutable artifacts under campaign `verification/conditioning_ablation_20261005_v1`
and `conditioning_intervention_20261005_v1`, with separate pilot directories.
Both complete receipts and the reviewed intervention summary passed. CPU
59384745 and GPU59384986 were released after their tasks completed; unfinished
production jobs were left untouched. Source/receipt hashes, exact endpoint
replays and mathematical controls validate this diagnostic scope, not posterior
qualification. The initial tiny-grid unit test's empty physical spectral bands
were replaced by an all-mode test mask; scientific runs use the unchanged,
nonempty registered bins.
