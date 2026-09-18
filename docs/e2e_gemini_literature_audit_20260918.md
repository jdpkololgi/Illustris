# Gemini literature sanity check: useful mechanisms, stronger claims than evidence

Reviewed 2026-09-18 against primary papers and the completed A/B/C/D experiment.
The supplied conversation and attachment are questions/claims to verify, not
scientific authority. No model, sampler, target panel or resource request was
changed during this review. Companion decision:
[joint-field follow-up, CFM and data availability](e2e_vdm_context_joint_decision_literature_20260918.md).

## Bottom line

Gemini identified real, relevant papers. They do not collectively establish
a statistically optimal, jointly calibrated 3D matter posterior from a
DESI-like galaxy catalogue. The most useful additional mechanism is
noise-dependent spatial covariance in a posterior sampler. The claims that
calibration requires an unconditional prior, DPS, or stochastic rather than
deterministic integration are incorrect. We should retain the matched
joint-field VDM/CFM recommendation, not replace it with an unvalidated
galaxy-likelihood DPS implementation.

## What each cited paper actually demonstrates

| Paper | Actual inverse/forward problem | Calibration evidence and useful lesson |
| --- | --- | --- |
| [DES-Y3 diffusion prior, 2511.14667](https://arxiv.org/html/2511.14667v1) | Noisy shear to projected convergence | TARP improves with time-dependent likelihood scaling. The schedule is selected using TARP; a separate final tuning-independent panel is not clearly documented. Gaussian shape noise and independent sky patches remain approximations. Useful evidence for controlling approximate guidance, not proof of real-survey joint calibration. |
| [Galaxy inpainting, 2408.00839](https://arxiv.org/html/2408.00839v1) | Matter to galaxy counts, the opposite direction | Direct conditional score model; TARP tests power-spectrum vectors, not full fields. The authors explicitly report residual low-credibility bias. Useful stochastic forward modelling, not a ready inverse matter posterior. |
| [Conicus3D, 2606.00803](https://arxiv.org/html/2606.00803v1) | Shear to matter lightcones | Modified DAPS uses spectral covariance. Its prior factorizes into independent thick lens planes with a redshift-conditioned 2D U-Net. Calibration evidence is error-versus-spread correlation, not TARP or joint coverage. |
| [Cluster mapping, 2603.14503v2](https://arxiv.org/html/2603.14503v2) | Photometry and lensing to 2D cluster surface mass | Photometry-conditioned prior plus DAPS and lensing constraints; reports marginal reliability/coverage. Useful hybrid construction, not an unconditional-prior requirement or joint 3D BGS validation. |

For the original [Park workshop paper](https://openreview.net/pdf?id=7k2Eh7OCoz),
the existing review likewise found reconstruction/PDF/spectral evidence, not
a demonstrated TARP-calibrated galaxy-survey field posterior. Its reported
small-scale power deficit is pertinent to our experience. Current direct PDF
access intermittently presents a CAPTCHA; the earlier indexed primary-paper
evidence is retained in the companion review rather than bypassing access.

### The strongest new technical lead

Conicus3D replaces the diagonal Gaussian approximation to the clean-given-noisy
field with

\[
\Sigma_t^{-1}=F^{-1}\operatorname{diag}(\sigma_t^{-2}+P_k^{-1})F,
\]

using training-estimated power. DAPS alternates clean-field refinement and
re-noising. This is an approximate sampler, not an exact-posterior guarantee.
[Section 5.3 and supplement](https://arxiv.org/html/2606.00803v1).

Our interpretation: this addresses the covariance of possible corrections,
not merely the average power of generated images. In a stationary Gaussian
example, the variance of a clean Fourier coefficient after a noisy observation
is P*sigma^2/(P+sigma^2). Using raw P at every noise level would be wrong.
The same formula is not an exact non-Gaussian or masked-survey conditional
covariance. Nor does it restore dependencies excluded by independent spatial
factors. This motivates a train-only spectral reference/preconditioner with
an explicit boundary contract, not arbitrary recolouring of posterior draws.

An uncertainty-error correlation cannot establish calibrated interval widths:
multiplying every predicted standard deviation by two leaves that correlation
unchanged. We must still measure coverage, joint scores and informativeness.

### Likelihood scaling is a targeted correction, not a universal remedy

The DES-Y3 paper approximates an intractable noisy-state likelihood with a
likelihood evaluated at a denoised estimate. Early guidance can push samples
out of distribution; a sigmoid scale reduces this effect. The paper also
acknowledges non-Gaussian real noise and unresolved simulation-to-data shift.
[Methods and limitations](https://arxiv.org/html/2511.14667v1).

Our conditional VDM does not currently add this external likelihood gradient.
There is no corresponding scalar knob to transplant. If we build a guided
reference later, schedule tuning must use a development panel and the frozen
choice must pass independent confirmation. Passing the diagnostic used to
tune it is not independent validation.

## Corrections to Gemini's proposed mechanics

1. **Deterministic sampling does not imply MAP.** Probability-flow ODEs can
   have the same time-marginal distributions as their reverse SDEs with exact
   scores. Random initial states remain random final draws; a deterministic
   map of a random vector need not be constant. Approximation and solver error
   matter, but injecting fresh noise is neither necessary nor sufficient for
   calibration. [Song et al., Section 4.3](https://arxiv.org/html/2011.13456).
   This is also why conditional flow matching remains a legitimate posterior
   method, rather than a point-estimation fallback.
2. **Direct conditional models are not intrinsically un-Bayesian.** With
   representative simulator pairs and sufficient learning, they can estimate
   p(delta|observations) without an evaluable differentiable likelihood.
   FMPE provides a conditional-flow construction, with stated limitations on
   what finite training loss guarantees.
   [Dax et al.](https://arxiv.org/html/2305.17161).
3. **A realistic simulator is not automatically a differentiable likelihood.**
   AbacusHOD populates halos/particles stochastically and supports environmental
   and velocity-bias extensions. It is not an exact log p(galaxies|R7-smoothed
   density) evaluator. [AbacusHOD](https://arxiv.org/abs/2110.11412).
   Unresolved halos, velocities and nuisance parameters must be retained or
   marginalized. A learned galaxy generator alone does not provide the density
   gradient needed for a DPS likelihood.
4. **Poisson is an assumption, not the definition of a galaxy catalogue.**
   Independent Poisson counts may be a useful approximation at an explicitly
   validated scale. HOD stochasticity, exclusion and fibre assignment need not
   follow it; a negative-binomial marginal does not supply missing spatial
   dependence. Also, denoising/velocity MSE is a generative training objective,
   not an assertion of a Gaussian observational count likelihood. Replacing
   that MSE with Poisson loss is not the implied remedy.
5. **Do not count the same observations twice.** Starting with p(delta|g)
   and multiplying by p(g|delta) again generally squares the data contribution.
   Adding another probe h instead requires p(h|delta,g), with any dependence
   accounted for. The cluster paper motivates combining learned and physical
   information; it does not license reusing BGS as both prior conditioning and
   an independent likelihood.
6. **TARP is a diagnostic of a specified random object.** Testing P(k), local
   eigenvalues, or an entire field are different experiments. Finite empirical
   agreement is evidence, not a universal optimality certificate; goodness of
   coverage must accompany sharpness/proper scores and condition-stratified
   checks. [Original TARP method](https://proceedings.mlr.press/v202/lemos23a.html).

## What this changes for our DESI programme

The actual completed experiment has localized a dependence problem: sampled
shared coarse fields beat fixed means in all eight adjacent-core control
cases, yet pooled difference C90 is only 33.3%. True coarse information strongly
improves the variogram, but its pooled coverage mixes stochastic tidal/gap
components with an exactly determined density mean; that raw coverage must not
be treated as a continuous-posterior calibration result. D's regional-mass C90
is 74.8% versus a 90.75% target. Good marginals are not the stopping criterion.
See the [decision note](e2e_vdm_context_joint_decision_literature_20260918.md)
for score definitions and important panel-size/conditioning qualifications.

My recommendation remains one bounded test of a **coupled spatial state**,
with matched VDM and CFM objectives, fixed train-only scale conditioning and
additional genuinely independent paired training phases if the inventory
audit supports them. Retain the old D baseline, two seeds, prior-declared
joint power/coverage/tidal gates, and a stop rule. The new literature strengthens
the covariance motivation; it does not justify an optimizer sweep or resuming
unchanged training. CFM earns a fair comparison through a different transport
path and potentially cheaper draws, not a presumed calibration advantage.

A likelihood-based reference is a separate, conditional option: first prove
the forward/selection model and its covariance approximation on a bounded
problem with known or independently sampled truth. Only then compare guided
posterior sampling. Do not silently turn this literature review into that
larger implementation project.

For eventual DESI use, require verified matter/BGS phase and epoch pairing,
real/redshift-space coordinates, velocity/HOD nuisance treatment, lightcone
evolution, angular/radial completeness and actual fibre/redshift-success
response. Complete and altmtl mock products and additional raw Abacus phase
directories exist, but have not yet been established as a ready matched
training set. The current fixed-HOD/epoch P3b experiment cannot certify real
DESI calibration. Real observations supply no direct dark-matter truth;
independent mock tests and external-probe checks remain indispensable.

**Decision: scientifically worth one targeted joint-field follow-up; not ready
for production field inference, and no claim that these papers have already
solved our exact objective.** No new compute was launched by this review.
