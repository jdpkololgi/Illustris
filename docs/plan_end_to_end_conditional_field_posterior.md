# Plan — End-to-end conditional density-field posterior

**Programme name:** `E2E-FIELD-v1`
**Variants:** `E2E-CFM` (conditional flow matching), `E2E-DIFF`
(conditional diffusion) and the gated `E2E-WCFM` (Fourier-anchored wavelet
conditional flow matching)
**Status:** research-planning draft; no training or Slurm work is authorized by
this document
**Written:** 2026-09-04
**Scientific review:** 2026-09-05; data preparation authorized, allocation deferred
**Authority:** current `SCIENCE_LOG.md` and frozen evidence supersede this plan
when they disagree

**Status reconciliation, 2026-09-08:** metadata preparation and the small analytic
fixture are complete; the 96 parent proposals remain support-unscreened,
split-unassigned and `training_ready=false`. R0-PHYSICS, payload verification,
parent extraction and diagnostic-power checks remain ahead of learned training.
D2 seed42 has passed its registered amended science/sampler ladder, but its
licensed second training seed and combined decision remain outstanding in the
record. This is not authorization to inherit a provisional winner or start E2E
compute. P12-A's registered blind release is green; no P13 action is authorized.

**Subsequent user authorization, 2026-09-08:** one concurrent interactive slot
was authorized to construct E2E input products, with model fits deferred and D2
untouched. Response screening, parent arrays, all independent native labels and
full-size numerical checks are complete, superseding the metadata-only status
above. Evidence is in `docs/e2e_field_data_build_20260908.md`. Packaging is not a
science-training release; R0-PHYSICS and the final matched contract remain gates.

**Subsequent physical audit, 2026-09-08:** the user-authorized three-training-phase
error-budget comparison is complete. FD2 is sub-percent in typical eigenvalue
RMS but can change void connectivity materially. FD8 nearly matches spectral
derivatives; floor sampling and finite-domain tides dominate typical errors.
Separate matched spectral E2E target regeneration is recommended, not executed
as a training-product change. Even the diagnostic 128-cell parent retains
topology-sensitive outliers. See `docs/e2e_field_error_budget_20260908.md`.
Scientific tolerance/domain release and `training_ready` remain unresolved.

## 0. Executive decision and boundary

This is a new, research-only programme for a response-conditioned posterior over
the already-smoothed late-time density field. It is not part of the first P12-A
VAC, is not a prerequisite for P13, and must never delay, reopen, recalibrate or
otherwise alter the frozen P12-A production candidate.

The programme has two primary matched-objective variants and one licensed,
gated representation challenger:

1. `E2E-CFM`: hierarchical conditional flow matching;
2. `E2E-DIFF`: the target-, conditioner-, capacity- and evaluation-matched
   hierarchical conditional diffusion model;
3. `E2E-WCFM`: the same conditional-flow objective and explicit Fourier coarse
   posterior as `E2E-CFM`, but with the complementary fine residual generated in
   an orthonormal wavelet basis. It reaches a full fit only if the no-training
   transform audit and a paired learned canary both pass.

The third model is scientifically justified because it adds a second controlled
comparison with a distinct scientific question:

```text
E2E-CFM  versus E2E-DIFF  -> path/weighting/sampler recipe, representation fixed
E2E-CFM  versus E2E-WCFM -> wavelet coordinates and network bias, CFM path fixed
```

`E2E-WCFM` does not replace the explicit Fourier long-mode component that the
project evidence already supports. There is no wavelet-diffusion interaction
arm, no fourth model, no automatic successor after failure and no same-objective
training extension. The design therefore does not estimate an
objective-by-representation interaction. Gaussian, independent-patch and
deterministic reconstructions appear only as frozen controls.

The scientific change relative to existing P12-F work is not simply a larger
patch or a different loss. One posterior draw must have a persistent spatial
identity across a parent superpatch and all of its child regions. Long modes are
sampled once and shared. Fine stochastic innovations are addressed in global
coordinates so that duplicated computations of the same physical voxel refer to
the same random realization. The result is evaluated jointly across sibling
patches, not as a bag of locally calibrated draws.

The user authorized scientific review, data preparation and small diagnostic
scaffolding on 2026-09-05. That work can proceed while D2 runs. Learned canaries
and substantial field processing require a frozen experiment contract and a
later compute session. D2 may inform the diffusion recipe after its decision is
immutable. The 2026-09-08 reconciliation records the completed base4 science fit
and seed42 pass; two-seed completion remains outstanding. The final immutable
D2 decision, not a provisional score or architecture label, controls inheritance.

## 1. How this differs from `plan_field_level_multimodal.md`

`docs/plan_field_level_multimodal.md` is a valuable July 2026 historical plan,
but it answers a broader and earlier set of questions. It owns the
output-representation and multimodal discussion that led to the deterministic
F-tier. Its main branches include:

- classical density reconstruction;
- CNN-versus-graph representation controls;
- privileged-information/LUPI distillation;
- sim-to-real latent alignment;
- deterministic graph -> density field -> fixed tidal physics;
- an early local generative-field proposal.

This document does not replace those results. It supersedes only the old plan's
prospective claim that a local generative F3 head is sufficient to establish a
coherent field posterior.

| Question | July multimodal plan | `E2E-FIELD-v1` |
| --- | --- | --- |
| Primary purpose | Establish whether field-shaped targets and physics improve deterministic recovery; coordinate multimodal ideas | Infer a calibrated joint distribution over spatial fields |
| Main input question | Graph, grid, privileged field and latent-alignment alternatives | Frozen deployable `V_final` galaxies and random-derived response only |
| Main output | Usually a point density field or per-galaxy eigenvalue prediction | Correlated posterior realizations of `delta_R7` |
| Spatial unit | Wedge/grid or one local patch | Parent superpatch with uniquely owned child cores and overlapping evaluation halos |
| Long modes | Present in a deterministic/transductive field, or implicit in a local generator | Sampled once per posterior realization and reused by all children |
| Fine stochastic identity | Not specified across independently evaluated patches | Global-coordinate random field and canonical voxel ownership |
| Calibration | Early per-galaxy or local-field diagnostics | Block/mode-aware field, tidal, cross-sibling and conditional calibration |
| Model families | Broad: deterministic F-tier, FlowJAX, FNO/U-Net, distillation, alignment, flow/diffusion ideas | Two matched generative objectives plus one gated CFM representation challenger |
| Production relationship | Historically considered a route into the VAC | Explicitly research-only and firewalled from P12-A/P13 |
| Evidence regime | Early wedge/nzharm experiments | Multi-phase Abacus CutSky evidence after P12-F/P12-F3; fresh post-ph006 confirmation required |

The old plan's deterministic graph -> field -> fixed-physics result remains an
important representation proof. It does not establish the distribution

```text
q(delta_R7(x) | X_final, S_random, H_fid)
```

or the cross-location covariance required for a field posterior. A deterministic
global map is coherent but has no posterior spread. A locally stochastic map has
spread but need not be globally coherent. `E2E-FIELD-v1` requires both.

## 2. Scientific motivation and non-motivation

### 2.1 What P12-A already solves

P12-A approximates

```text
q_A(lambda_g | lambda_hat_OOF,g, response_g, H_fid)
```

for one galaxy `g`. It learns the joint three-dimensional distribution of the
ordered tidal eigenvalues at that galaxy. On held-out ph006, its physical
eigenvalue and eigengap TARP deviations are about `0.00684` and `0.01002`, with
an explicit mild sparse-shell lambda2/lambda3 caveat. That is a strong and useful
per-galaxy posterior result.

P12-A does not attempt to learn

```text
Cov(lambda_g, lambda_h | X, S),  g != h,
```

and independently drawing its rows does not create a possible density/tidal
field. This is not a defect in P12-A. It is outside its estimand.

### 2.2 Science that requires the harder estimand

A coherent field posterior is justified only for analyses that materially need
correlated multi-location uncertainty, for example:

- posterior uncertainty in environment-dependent clustering;
- void, filament, topology or Minkowski-functional distributions;
- coherent maps rather than posterior-mean maps;
- tidal eigenvector and intrinsic-alignment analyses;
- uncertainty in spatially aggregated environmental statistics;
- field-level posterior-predictive re-observation.

If the intended product is only per-galaxy eigenvalue/class probability and
quality information, P12-A is the appropriate endpoint and this programme should
not run.

The user's preferred science directions are environment correlations and cosmic
web connectivity (2026-09-05). The primary benchmark is therefore the posterior
of regional web filling fractions and environment pair statistics on fixed
interior voxel locations. For example, with ordered eigenvalues define
`m(x) = 1[lambda2(x) > 0.2]`, indicating at least two collapsed directions, and
evaluate `C_m(r) = mean_pairs_at_r m(x)m(y)` per parent draw. Start with fixed
10--20, 20--40 and 40--80 Mpc/h separation bins; verify supported pair counts
before freezing them. Draw-wise regional fractions and pair statistics require
the covariance that independent per-galaxy intervals cannot supply.

The secondary benchmark is the probability that this same excursion set connects
two specified interior regions, together with void (`lambda3 <= 0.2`) component
volumes. Freeze 6-neighbour connectivity, interior anchors, masks and threshold;
report sensitivity to one alternate connectivity convention only as supplementary.
These are R7- and grid-defined web statistics, not a claim to resolve a unique
continuous filament skeleton. Near-zero/one event probabilities require Brier
scores and reliability checks rather than continuous rank tests without ties.

Galaxy-marked clustering is a downstream application after the coordinate and
galaxy-position target audits pass: a statistic at observed redshift-space
positions differs from the true real-space environment at the source galaxy.
The initial voxel benchmark avoids silently equating those two estimands.
Tidal orientations remain supplementary and require an independently validated
tensor reference. No causal effect of environment on galaxy evolution is inferred
from these conditional correlations alone.

### 2.3 What the existing field experiments establish

The current evidence supports a specific model-design hypothesis:

1. P12-F1b proved that a proper conditional rectified flow can generate an
   informative complete local `delta_R7` patch. It did not pass strict
   conditional calibration.
2. The correlated Gaussian G1 control preserved joint structure better than the
   first local flow and diffusion challengers, but retained an eigengap and
   conditional-coverage defect.
3. The causal autopsy localized much of that defect to under-dispersed
   long-wavelength traceless-shear amplitude, not generic scalar-density error.
4. A truth-assisted low-mode scatter intervention improved eigengap TARP from
   about `0.061` to `0.020`, while a better posterior mean did not. Posterior
   covariance, rather than point reconstruction alone, is causal.
5. Wider 120-Mpc/h observational context improved the learned low-mode eigengap
   result relative to 40 Mpc/h, proving that information outside the fine target
   matters. It also over-dispersed another direction and worsened proper scores;
   larger context is useful information, not a sufficient cure.
6. Direct conditional Fourier generation repaired the two registered low-mode
   powers and difficult shear/eigengap dependence. It still failed simultaneous
   conditional-coverage gates.
7. The small diffusion comparator was encouraging on joint physics, but was not
   promoted. D2 is the bounded test of whether a better diffusion primitive can
   exploit the same conditional signal.

These results motivate

```text
explicit scale structure
+ wide observational context
+ observation-conditional stochastic dependence
+ shared spatial sample identity.
```

They do not establish that a monolithic larger U-Net, a generic full-field
diffusion model, or arbitrary patch enlargement will succeed.

### 2.4 Remaining causal uncertainty

The following explanations are not yet cleanly separated:

- finite observation context;
- finite generated domain;
- poor long-mode coordinates in a local real-space network;
- sampler/objective limitations;
- observation-conditional anisotropic covariance;
- survey-window and boundary coupling;
- independent-patch factorization;
- finite training exposure and capacity.

The programme must therefore remain diagnostically factorized even though its
final sampling path is end to end.

### 2.5 Literature position and novelty claim

The novelty claim must be narrow. Conditional flows, diffusion models, wavelet
generators and field-level Bayesian inference each already exist. The proposed
contribution is the combination of the scientific estimand, survey condition,
spatial sampling contract and calibration standard. In the closest literature
identified for this plan:

| Work | What it establishes | What it does not establish for this programme |
| --- | --- | --- |
| [BORG: Jasche & Wandelt](https://arxiv.org/abs/1203.3639), [Leclercq et al.](https://arxiv.org/abs/1502.02690) | Survey-conditioned initial and evolved 3-D density realizations, observational uncertainty propagation and probabilistic tidal-web maps | They already establish the broad field-posterior scientific idea; the question here is an amortized approximation with measured accuracy, survey-response robustness and cost |
| [CosmoFlow](https://arxiv.org/abs/2507.11842) ([code](https://github.com/sidk2/lambda-cfm)) | Scale-aware flow-matching representation learning on 2-D CAMELS density maps; progressive scale masking is relevant to multiscale design | Its encoder observes the target field, reconstruction is lossy at high frequency, and the downstream parameter example predicts a posterior mean rather than a catalog-conditioned field distribution |
| [Cosmo3DFlow](https://arxiv.org/abs/2602.10172) ([OpenReview](https://openreview.net/forum?id=7k2Eh7OCoz)) | Conditional 3-D wavelet flow matching for initial conditions given evolved matter or gridded halos in complete periodic simulation boxes | It targets an initial field rather than the late-time `delta_R7`/tidal posterior, has no CutSky response, and does not test shared-superpatch identity. Its reported spatial PICP is not a repeated-simulation joint-field calibration proof |
| [Kostic et al.](https://arxiv.org/abs/2212.07875) | High-dimensional initial-condition posterior sampling and controlled field-level consistency tests for an EFT likelihood | Analytic/linear consistency and field summaries are not the same as amortized response-conditioned calibration across held-out survey realizations |
| [JADE](https://arxiv.org/abs/2606.31988) and [MIRA](https://arxiv.org/abs/2605.02014) | Joint diffusion inference for weak-lensing maps and cosmology, with a sample-only multivariate calibration diagnostic | This is 2-D lensing rather than 3-D tracer-to-density/tidal inference; TARP is applied to cosmology, while full-field MIRA remains metric-dependent and only a necessary scalar diagnostic in its basic form |
| [TARP](https://arxiv.org/abs/2302.03026) | Coverage testing for multivariate simulation-based posteriors | It supplies a diagnostic, not a solution to spatial coherence, survey response or field representation |
| [Generative Diffusion Priors for 3D Mapping](https://arxiv.org/abs/2606.00803) | Learned 3-D cosmological priors combined with a weak-lensing forward model | It further rules out a broad novelty claim for 3-D diffusion posterior reconstruction; tracer conditioning and our calibration/geometry contract differ |
| [Patched Flow Matching](https://arxiv.org/abs/2606.22084) | Patch-based flow vector fields for reconstruction on larger fluid domains | Shared spatial generative computation is also being studied outside cosmology; it motivates a global-state implementation audit rather than an assumption that shared noise suffices |

Subject to a full literature review before publication, the candidate contribution
is therefore an empirically validated approximation and benchmark:

> amortized inference of late-time 3-D density realizations from a DESI-like
> observed galaxy and response condition, with measured accuracy and cost for
> environment correlations and connectivity, and simultaneous tests of field,
> long-mode, tidal and cross-sibling posterior dependence on fresh simulations.

Survey-conditioned cosmic-web realizations, uncertainty propagation and field
inference are established science. No first-ever claim is made for them, for
wavelet generation, or for posterior calibration. Coherence is a necessary
property of the proposed product, not by itself an architectural novelty.

This would be scientifically interesting for four reasons:

1. it turns uncertainty in a sparse, masked redshift-survey observation into
   correlated possible fields rather than independent local error bars;
2. it tests whether non-local tidal uncertainty can be learned without either
   periodic-box assumptions or a deterministic low-mode patch;
3. it makes posterior dependence falsifiable through overlap, mode, shear and
   sibling calibration, rather than judging samples primarily by visual quality
   or an average power spectrum;
4. its bounded comparisons test a path/weighting/sampler recipe (`CFM` versus
   `DIFF`) and a wavelet/network inductive bias (`CFM` versus `WCFM`) while
   retaining the low-mode representation supported by the project's evidence.

No novelty claim is earned by writing this plan or by obtaining plausible
samples. It requires independent-simulation calibration, shared-superpatch
closure and a successful frozen comparison. Failure would still be informative:
it would bound how far current amortized generators can support coherent
survey-conditioned field inference beyond the already successful P12-A
per-galaxy estimand.

## 3. The estimand and what “end to end” means

### 3.1 Scientific estimand

The target posterior remains

```text
p(delta_R7 | X_final, S_random, H_fid),
```

approximated by `q_theta`. Here:

- `X_final` is the BRIGHT-only final observed galaxy view;
- `S_random` is the deployable random-derived survey response;
- `H_fid` is the fixed fiducial mock galaxy--halo prescription;
- `delta_R7` is the matter-density contrast already smoothed at 7 Mpc/h.

The September-8 native-source audit clarifies a finite-resolution qualification:
the inherited P12-F scalar is actually `delta_R7_trace`, the trace of a tensor
formed by centred finite differences of the smoothed potential, sampled at
nearest native cells. It is not identically continuum Gaussian-smoothed density.
New arrays retain that matched source quantity under an explicit name. The
transfer-function distinction and independent references are recorded in
`docs/e2e_field_data_build_20260908.md`. R0-PHYSICS must budget this discretization
as well as boundary error before a tidal claim. No P12-A/D2 target is amended.

This is not an unsmoothed density posterior, an initial-condition posterior, a
cosmology posterior, or an HOD-marginalized posterior. Any future change to one
of those objects is a new estimand and a new programme.

Every field draw must use the unchanged fixed physics map

```text
delta_R7(k)
  -> T_ij(k) = (k_i k_j / k^2) delta_R7(k)
  -> eigvalsh(T),
```

with the DC convention frozen and no second `W_7` smoothing.

This formula is exact for a declared complete domain and boundary convention.
It is not an exact map from a masked or finite density crop to the cosmological
tidal tensor. Section 3.4 supplies the missing boundary/zero-mode contract.

### 3.2 Spatial factorization

Let `P` be a parent superpatch and `C_j` be child target cores inside it. Define
fixed, training-independent analysis operators `A_L`, `A_H` and synthesis
operators `S_L`, `S_H` satisfying, on the declared target support,

```text
S_L A_L delta + S_H A_H delta = delta
```

to a frozen numerical tolerance. Write

```text
delta_L = A_L delta_R7
delta_H = A_H delta_R7.
```

The coarse operator `A_L` is an explicit frozen low-`k` Fourier projection for
all variants. `E2E-CFM` and `E2E-DIFF` generate its complementary fine
real-space residual under the same baseline transform. `E2E-WCFM` keeps that
identical Fourier coarse target but applies a frozen orthonormal, exactly
invertible wavelet transform `W_H` to the complementary fine residual:

```text
w_H = W_H A_H delta_R7.
```

Only the fine target coordinates and the wavelet-aware fine-network coupling
differ between `E2E-CFM` and `E2E-WCFM`. A fully wavelet coarse field is an
analysis-only Stage-0 diagnostic, not an eligible trained model, because it
would discard the direct Fourier long-mode intervention already supported by
the project's experiments.

Use orthogonal projectors `P_L` and `P_H=I-P_L` on one complete padded parent
lattice, with `P_L P_H=0`. The stored fine array is an embedding in the
complementary subspace: its noise, noising path and predicted velocity must also
lie in that subspace. For wavelets the covariance of `W_H P_H epsilon` is
`W_H P_H W_H^T`; independent noise in all wavelet coefficients would populate
forbidden low modes. Apply the projection to the parent state/velocity or use an
explicit orthonormal basis of the complement, and test both training and sampling.
Masks do not multiply the field before this split; missing observations still
correspond to latent matter in the stochastic domain.

The learned approximation is

```text
q(delta_L^P, delta_H^P | X, S)
  = q_L(delta_L^P | X_wide^P, S_wide^P)
    q_H(delta_H^P | delta_L^P, X_wide^P, S_wide^P).
```

The factorization is exact only when the fine conditional retains the full
declared observation condition. Replacing wide information by a local crop or
compressed feature is an approximation to test, not a conditional-independence
identity. Child crops evaluate one parent vector field using a shared evolving
state and shared wide-context features. They do not define a product of separately
trained/sampled child posteriors.

### 3.3 Distribution-level end-to-end requirement

“End to end” means that one call beginning from the deployable observation and
one global sample identifier produces a complete parent realization:

```text
(X_final, S_random, sample_id)
  -> sampled parent coarse field
  -> sampled fine field with the same spatial identity
  -> assembled delta_R7 realization
  -> fixed tidal tensor
  -> eigenvalues/eigenvectors and derived summaries.
```

It does not require one opaque neural network. A modular probabilistic
factorization is preferred because the coarse and fine distributions can be
validated separately. What is forbidden is frozen G1 high-frequency completion:
both scale components must be generated by the new variant. Frozen G1 products
may be used as deployable conditioner channels or controls, but never as an
unacknowledged piece of the output draw.

### 3.4 Boundary physics, source resolution and the parent mean

This is the first hard scientific feasibility gate, `R0-PHYSICS`. Interior density
does not fix the external harmonic potential: `T_true = T_parent + Hess(phi_ext)`
with `Laplacian(phi_ext)=0` in the interior. External traceless shear can therefore
change eigenvalues without changing the local density trace. Numerical trace
closure and ordered eigenvalues are necessary but cannot detect this missing
physical contribution.

Construct independent native/full-box tensor truth and compare it with the tidal
operator applied to the *true* sampled parent density, at the same locations.
Use nested parent sizes and one fixed central science region to separate:

- observer/box coordinates and real-space versus observed-redshift positions;
- native-grid sampling, interpolation and coarse-grid resolution error;
- finite-domain and exterior tidal error;
- zero-mode/parent-mean treatment; and
- learned posterior error, only after the preceding errors are quantified.

The current target builder samples the trace at nearest native-grid cells.
It supplies a scalar field, not independent six-component tensor truth. Re-solving
that same scalar crop and calling the result truth would make the check circular.
Eigenvalues alone cannot reconstruct tensor orientations; use saved eigenvectors
if independently validated, or construct six tensor components from the native
density on a compute node.

Parent Fourier DC is the region's fluctuating mean and must be modelled. The
global periodic-box zero mode may be fixed to zero; the crop mean cannot be
silently removed on that basis. Adding `mean(delta)/3` to the tensor diagonal is
one declared completion convention, not recovery of exterior shear. Measure its
error against native truth. No Fourier projection or periodic FFT on a bounded
array should be interpreted as evidence that CutSky is physically periodic.

Freeze a training-only error budget requiring boundary/discretization bias and
RMS in the primary tidal summaries to remain below a declared fraction of the
posterior uncertainty, and no material distortion of the primary science
functional. The fraction and tests must be chosen before learned fits. If neither
candidate domain passes, stop the *tidal-science* claim at this stage. Do not add
an unlicensed external-tide model or flatten calibration to compensate. A local
density benchmark may still be reported with its narrower estimand.

## 4. Superpatch, child and observation geometry

### 4.1 Prediction region is not context region

The geometry must distinguish four domains:

1. **parent stochastic domain:** carries the shared coarse realization;
2. **child generation halo:** supplies fine-model receptive-field context;
3. **child authoritative core:** owns output voxels in the assembled field;
4. **observation context:** the larger region of galaxies and response seen by
   the conditioner.

The observation context may extend beyond the fine target. High-resolution
generation is not required over the full context volume. This is the principal
way to supply non-local tidal information without allocating a dense fine grid
over an entire cap.

The initial metadata proposals preserve the inherited P3 observer lattice:

```text
5 Mpc cells = 3.383 Mpc/h, using observer-coordinate h = 0.6766
64 or 96 cells per parent = 216.512 or 324.768 Mpc/h
32-cell child cores = 108.256 Mpc/h
8-cell child evaluation halos = 27.064 Mpc/h
16-cell observation halos beyond the parent = 54.128 Mpc/h per side.
```

These are proposals for the numerical audit, not frozen training dimensions.
The previously written "5 Mpc/h voxels" was a unit error relative to the source
products. Explicitly changing to 5 Mpc/h is a different resolution experiment.
The observer-coordinate conversion is part of the existing P3 mapping and must
not silently be replaced by the simulation's cosmological h. Use the same
central target and anchors across domain/context comparisons.

### 4.2 Parent selection and spatial split

The dataset builder must:

- select parent centres with a truth-free, response-based rule;
- group all overlapping children under one immutable `parent_id`;
- assign `phase_id`, cap, shell and response summaries at parent level;
- retain global voxel origins and physical coordinate transforms;
- materialize `M_observed`, `M_target`, `M_core` and `M_overlap` separately;
- record every source and transformation hash;
- split phases and parent groups before generating child crops;
- forbid siblings or overlapping halos from crossing train/validation roles.

Grouping must also follow the underlying simulation box after the observer-to-box
periodic mapping. Different caps, spatially separated observer regions, repeated
boxes, rotations and HOD/response re-observations can share the same matter
realization. Hash and group those source identities; include stochastic, physics
and observation halos in the overlap audit. Nearby non-overlapping parents remain
correlated. Parent count is not independent-phase count, and bootstrap groups
must reflect source superblocks or phases, not only parent IDs.

One parent, not one child or voxel, is the primary statistical and sampling
weight. Otherwise parents with more supported children would silently dominate
the objective and uncertainty estimates.

### 4.3 Boundary and support contract

The survey window is part of the condition, not missing-at-random noise.

- The hard `support_random` mask is metadata for support and attention.
- Apodized exposure may be a learned continuous channel but must not be used as
  a substitute for the exact support mask.
- Periodic/circular padding is forbidden for CutSky inference.
- Zero/valid padding, crop halos and support rules must be identical across the
  learned variants.
- Loss, physics and evaluation masks must be distinct and versioned.
- Parent regions with inadequate physics halos must be excluded or explicitly
  labelled prior-dominated; they may not be hidden by core averaging.

Store `M_observed`, `M_latent_domain`, `M_truth_available`, `M_loss`, and
`M_science_core` separately. Observational holes do not mean zero density or
missing simulator truth. Learning only supported voxels cannot establish the
posterior completion required for tides across holes: train the joint latent
domain wherever simulator truth exists, while keeping science-support scoring
separate and phase/parent weights fixed.

### 4.4 Globally addressed stochastic state

Sharing only `delta_L` is insufficient if two independently sampled fine models
assign different values to the same voxel. For each posterior `sample_id`, random
innovations must be a deterministic function of

```text
(programme_digest, sample_id, scale_id, global_voxel_coordinate, diffusion_step)
```

where `diffusion_step` is omitted for flow matching and deterministic diffusion.
Counter-based or otherwise crop-invariant random generation is required.

Two child evaluations that include the same physical voxel and the same required
halo must therefore receive identical stochastic input at that voxel. The final
assembled field uses canonical voxel ownership; it is never produced by averaging
incompatible independent posterior draws.

Shared input noise is only a necessary implementation ingredient. It does not
make independently integrated crops identical: repeated convolutional updates
propagate different crop boundary conditions inward. Use one parent state
`u_t^P`; at **every** velocity/denoising evaluation, extract all tile halos from
that state, compute updates, assemble the unique core updates, apply any global
spectral projection, and advance the parent state synchronously. Heun predictor
and corrector each require this synchronization. Pad only at the declared parent
boundary. Sampling must pass a full-parent versus tiled-velocity test, including
multiple time steps and shifted legal tilings.

The identity test concerns alternative computations of the same parent draw
under the same observation condition. Independently conditioned parent regions
need not match sample by sample. Extending coherent sampling between parents
requires an additional shared-domain construction, which is outside the current
superpatch claim. A reused sample ID alone cannot provide that construction.

## 5. Common conditioning and architecture contract

All variants must share every non-objective choice that can reasonably be
matched; the two registered contrasts below define the licensed differences.

### 5.1 Deployable conditioner

The base multiscale condition is constructed only from inference-time quantities:

- normalized BRIGHT counts;
- density proxy from the count/random ratio;
- apodized exposure/selection response;
- exact `support_random` metadata;
- random-support boundary distance;
- redshift/radial coordinate and line-of-sight geometry;
- optionally, frozen G1 mean, log-width and predicted shear amplitude as
  deterministic functions of the same `X_final,S_random`.

The optional frozen G1 summaries require one checkpoint-free, training-only
information audit and a single decision before any generative model is fit.
If retained, all variants receive the same summaries. G1 truth residuals,
ph006-fitted corrections and true environment are forbidden inputs.

### 5.2 Multiscale observation encoder

One shared architectural specification maps the conditioner onto:

- a coarse parent representation for `q_L`;
- local fine representations for `q_H`;
- explicit response/support features at every scale.

The encoder may share weights across variants only through identical
initialization; no model may be pretrained on another's held-out decisions.
The same parameter/FLOP tolerance, receptive field and support treatment are
required.

Patch-safe normalization is mandatory. Per-voxel channel LayerNorm or RMSNorm
is eligible. Spatial GroupNorm and InstanceNorm are excluded from this version
given the P6/P8 subdivision evidence. There is no normalization ablation.

### 5.3 Scale-balanced learning

Voxel count must not allow the fine level to erase the coarse objective. The
training contract must freeze level weights using training-set mode counts and
target variances so that:

- coarse and fine components each carry an explicit scientific weight;
- the two registered long bands remain visible in loss and diagnostics;
- large supported parents do not dominate small supported parents;
- dense shells do not dominate sparse shells;
- reweighting cannot be changed after external validation is inspected.

All target scaling is fitted on training parents only. Conditional
location/scale heads, if used, are part of the learned scale-specific
distribution. A spatially varying standard deviation must not be multiplied
onto an already filtered residual in a way that remixes the frozen low/high
subspaces.

## 6. The three-arm causal design

### 6.1 Variant A — `E2E-CFM`

`E2E-CFM` learns conditional velocity fields for both `delta_L` and `delta_H`.
The initial registered path is a Gaussian-to-data conditional flow-matching
path, with the exact path, time sampling and any endpoint weighting frozen before
the first science fit.

For level `ell` in `{L,H}`:

```text
y_t,ell = (1 - t) epsilon_ell + t y_ell
v*_ell = y_ell - epsilon_ell
v_theta,ell = v_theta(y_t,ell, t, condition_ell).
```

The fine velocity is additionally conditioned on the truth coarse component
during supervised training, as required by the exact factorization. At inference
it receives the sampled coarse component. This is not an invalid teacher-forcing
shortcut: fitting `q_H(delta_H | delta_L,X,S)` on true joint pairs is the correct
conditional objective. The finite-model rollout gap must nevertheless be
measured explicitly by evaluating fine draws under sampled rather than true
coarse states.

Sampling uses a deterministic ODE for a fixed globally addressed initial noise
field. The contract must freeze:

- integrator family;
- NFE ladder and convergence pair;
- absolute/relative tolerances if adaptive integration is used;
- crop batching and sample batching;
- raw versus EMA policy, if EMA is licensed for all relevant variants.

The primary implementation should prefer a fixed-step Heun/RK method for exact
replay and crop invariance. An adaptive solver is a diagnostic only unless its
batch- and crop-invariance behavior is demonstrated.

### 6.2 Variant B — `E2E-DIFF`

`E2E-DIFF` uses the same target components, observation encoder, parent/child
network widths, masks, parameter budget and training presentations. Its registered
differences are the probability path, time/loss weighting, prediction
parameterization and sampler; these form one recipe-level comparison.

The initial implementation should inherit the immutable D2 engineering result
where scientifically applicable:

- patch-safe 3-D residual blocks;
- multilevel sinusoidal/log-SNR time embedding;
- per-voxel channel normalization;
- non-periodic padding;
- EMA handling;
- frozen noise schedule and prediction parameterization;
- an explicit sampler-convergence ladder.

If D2 closes without a technically valid architecture/sampler, no silent repair
is allowed. The diffusion arm remains planned but requires a new explicit user
authorization that names the D2 failure and explains why the end-to-end change
addresses it.

The default primary sampler is deterministic for reproducible sample identity.
If a stochastic reverse process is required, every step's innovation must be
globally coordinate-addressed so crop subdivision does not change the draw.

### 6.3 Variant C — `E2E-WCFM`

`E2E-WCFM` is a fine-representation challenger, not a third generative
objective. It reuses the exact trained `q_L` checkpoint from `E2E-CFM` and
learns

```text
q_WH(w_H | delta_L, X_local, S_local)
```

with the same conditional flow-matching path, time distribution, optimizer,
training parents and presentation cap as the baseline CFM fine model. Synthesis
applies `W_H^-1` and the unchanged `S_H`; all scientific evaluation is performed
on the reconstructed common `delta_R7` field, never only in coefficient space.

Here local features retain the shared wide-condition encoding in Section 3.2.
An orthonormal change of coordinates alone preserves isotropic Gaussian noise
and Euclidean CFM squared loss. It cannot provide an intrinsic statistical
advantage. The test is of wavelet organization **and the resulting network
inductive bias**; coefficient scaling, subband weights and network couplings
must be recorded as possible changes in that intervention. For the initial
contrast, match the field-space loss metric and use all coefficients without
thresholding. A conjugated-network algebraic control is checkpoint-free and
verifies this equivalence before the learned canary.

One orthonormal wavelet family, decomposition depth and boundary rule may be
licensed by Stage 0. Wavelet-specific cross-scale skips or subband conditioning
are allowed, but the fine-model parameter count, FLOPs and receptive field must
remain inside the registered parity tolerance. Coefficient-lattice noise must
map deterministically from the same global `sample_id` and coordinates so that
overlapping crops replay the same physical realization.

The arm exists to test whether localized multiresolution fine coordinates improve
calibration, high-`k` structure or compute efficiency after the long modes have
already been represented explicitly. It cannot support a claim that wavelets are
generally superior, and it cannot proceed to a full fit without both `R0-WAVELET`
and `R1-WAVELET` passing.

### 6.4 Matched comparison rules

The common parts of the comparisons are:

- training/validation parent IDs and ordering;
- the explicit Fourier `A_L` target and low/high split in all variants;
- the baseline fine target in `E2E-CFM` versus `E2E-DIFF`;
- all non-representation choices in `E2E-CFM` versus `E2E-WCFM`, except the
  explicitly licensed wavelet/network coupling;
- condition channels and normalization;
- architecture, capacity band and receptive fields;
- optimizer, effective batch and patch presentations where compatible;
- augmentation sequence and seeds;
- posterior sample IDs and evaluation rows;
- fixed physics and all statistical gates.

Equal NFE is reported but is not automatically the scientifically fairest
primary comparison. Each sampler must first reach its frozen convergence
tolerance. Report quality at convergence, quality at matched NFE, wall time,
GPU-hours, memory and storage. A sampler that reaches the same posterior quality
more cheaply wins on operational efficiency; a cheap unconverged sampler does
not.

Flow matching includes diffusion probability paths
([Lipman et al.](https://arxiv.org/abs/2210.02747)). These labels are not disjoint
hypothesis classes. Freeze `alpha(t)`, `sigma(t)`, log-SNR/time sampling, target
parameterization, induced loss weight and converged solver for each arm. The
registered comparison estimates the effect of those complete recipes; it does
not isolate a universal "flow versus diffusion" mechanism. Verify analytic
Gaussian recovery and target/velocity conversions without training extra models.

### 6.5 Literature-informed choices that are not licensed arms

Cosmo3DFlow motivates the wavelet-coordinate canary; it does not license copying
its full recipe. In particular:

- no auxiliary power-spectrum loss is allowed in v1. It would change the
  generative objective and directly optimize a registered diagnostic, so a good
  spectrum could conceal a miscalibrated joint posterior;
- periodic/circular padding and spatial GroupNorm are not allowed under the
  CutSky and patch-invariance contract;
- the explicit Fourier coarse posterior is not replaced by an all-wavelet model;
- no wavelet diffusion arm is trained, because it would add the unestimated
  representation-by-objective interaction and turn the bounded design into a
  factorial architecture search;
- no encoder may observe the target density field at inference time. The
  autoencoding setup used for CosmoFlow representation learning is informative
  about scale organization but is not a deployable conditioner here.

Any future auxiliary spectral-loss or wavelet-diffusion experiment is a new
version requiring a new causal question, evidence split and user authorization.

## 7. Training protocol redesign

### 7.1 Phase roles and evidence hygiene

The existing ph006 panel has informed many P12-F design decisions. It is a legacy
development benchmark for this programme, not untouched confirmation.

Before training, freeze:

- training phases/parents;
- an internal selection split grouped by phase and parent;
- a disjoint internal confirmation split opened once;
- one fresh external simulation phase or ensemble not used to design this plan;
- a separate future blind phase/ensemble if a v2 blind claim is desired.

`ph001` belongs to the P12-A one-open claim and is forbidden everywhere in
`E2E-FIELD-v1`. No field result may reuse it as independent evidence.

If no fresh external phase/ensemble can be designated, the programme may produce
engineering and ph006 research diagnostics but cannot promote a field-posterior
research finalist.

The present registry contains only ph000--ph006. ph000/ph002--ph005 are available
development phases, but have historical training exposure; ph006 is used legacy
evidence, and ph001 remains excluded from this programme even after P12-A opens
it. The September-8 internal canary build assigns ph000/ph002/ph003 to training,
ph004 to internal selection and ph005 to internal confirmation, with entire
phases (all caps and replicas) grouped together. These roles supersede the
September-5 unassigned proposals; they do not establish fresh external evidence
or freeze the final science-training population.
New field phases require an exposure ledger and source-asset audit before being
called fresh. Independent HOD draws from one box do not create new cosmological
phases. A second training seed establishes optimization repeatability, not new
independent-universe evidence.

### 7.2 Stage 0 / `R0-PHYSICS` and `R0-WAVELET` — no-training gates

Before either objective is optimized, `R0-PHYSICS` in Section 3.4 must establish
the scientific target's finite-domain adequacy. Then:

1. build parent/child manifests and prove role disjointness;
2. validate global coordinate and mask parity under crop/subdivision;
3. freeze the explicit low-`k` Fourier `A_L` and baseline complementary
   real-space fine transform;
4. test at most one preregistered orthonormal wavelet family at at most two
   decomposition depths for the complementary fine residual;
5. measure exact analysis/synthesis round-trip error for the baseline and
   wavelet representations;
6. quantify DC/low-mode leakage, spectral leakage and CutSky boundary
   sensitivity;
7. verify that scale-specific standardization does not remix components;
8. reconstruct `delta_R7`, run the fixed tidal map and prove trace/order closure;
9. generate globally addressed real-space and coefficient-lattice random fields
   and prove crop-invariant replay;
10. measure transform memory, throughput and numerical conditioning;
11. construct truth and deterministic-reconstruction overlap references; and
12. freeze all data, transform and response hashes.

`R0-WAVELET` passes only if the candidate is exactly invertible to the registered
tolerance, preserves the Fourier coarse subspace and DC convention, closes the
`delta_R7`/tidal physics tests, is invariant to legal crop subdivision, has no
unresolved non-periodic boundary artifact and fits the frozen resource envelope.
Wavelet family, depth and boundary handling are chosen on these checkpoint-free
criteria only.

A fully wavelet decomposition may appear only as a numerical diagnostic. If no
candidate passes, `E2E-WCFM` is removed before any learned canary; the baseline
`E2E-CFM` and `E2E-DIFF` programme continues unchanged.

### 7.3 Stage 1 — bounded context audit

Hold fixed the authoritative fine target, generative architecture and training
parents. Compare at most two predeclared observation-context radii in short
training-only canaries. The smaller radius should be the strongest already
supported scale; the larger should test the next physically plausible parent
extent.

The decision uses:

- low-band residual power and cross-correlation;
- coarse linear-projection ranks;
- joint shear/eigengap diagnostics;
- boundary- and response-stratified behavior;
- memory and throughput.

This stage chooses one common context for all retained variants. It cannot
select the final model and never reads the fresh external phase.

### 7.4 Stage 2 — coarse parent canaries

Train `q_L` once for each generative objective under the same small presentation
cap: one CFM coarse model shared exactly by `E2E-CFM` and `E2E-WCFM`, and one
matched diffusion coarse model for `E2E-DIFF`. Each must
pass:

- finite loss, gradients and samples;
- exact checkpoint/resume and sample replay;
- sampler convergence;
- non-degenerate posterior spread;
- recovery of registered long-mode power/cross-power;
- coarse proper-score non-regression relative to frozen controls;
- no train/internal-selection contradiction; the confirmation split stays closed.

An objective failing the coarse gate stops. A failed CFM coarse gate also stops
`E2E-WCFM`; fine coordinates cannot rescue a failed shared coarse posterior.

### 7.5 Stage 3a / `R1-WAVELET` — paired learned representation canary

After the CFM coarse gate passes, train small from-scratch `q_H` canaries for
`E2E-CFM` and `E2E-WCFM`. They reuse the same frozen CFM `q_L` checkpoint and
match parent order, sample seed, conditioner, CFM path, optimizer, presentation
count, effective receptive field and capacity tolerance. The canary checkpoints
are discarded; neither can be warm-started into a full science fit.

`R1-WAVELET` requires all of the following on the training-only selection split:

- finite, replayable sampling and exact transform/physics closure;
- no registered regression beyond tolerance in parent energy/variogram score,
  projected ranks/TARP, derived shear/eigengap calibration, long-mode
  non-regression or overlap identity;
- sampler convergence in the reconstructed common field; and
- a preregistered practical improvement, with a paired parent-level uncertainty
  interval excluding zero, in either a fine-scale variogram/wavelet proper score
  or converged sampling cost at equal posterior quality.

If the arm merely matches the baseline, the simpler baseline wins and
`E2E-WCFM` stops. If it fails any gate, it stops. No alternative wavelet, depth,
network, loss or longer canary may be substituted.

### 7.6 Stage 3b — full fine conditional training

For each surviving objective, fit its coarse model to the preregistered full
coarse budget, then freeze it before full fine training. Both CFM representations
reuse this exact coarse checkpoint and sampled states. Attach and fit each fine
model from scratch under its own fixed budget. `E2E-WCFM` survives only through
`R1-WAVELET`. The data
loader presents parent units containing several sibling children; a training
step must not silently treat those children as independent scientific examples.

The fine loss is conditioned on the matching true coarse component. Evaluation
at every frozen milestone includes two modes:

1. **oracle-condition diagnostic:** fine samples conditioned on true `delta_L`;
2. **posterior rollout:** fine samples conditioned on sampled `delta_L`.

The difference isolates coarse-to-fine error propagation. Only posterior rollout
is eligible for model selection or promotion.

Training milestones and the maximum presentation budget are frozen before the
first full fit. Select the earliest feasible milestone within one standard error
of the best internal parent-level proper score. No post-cap continuation is
allowed.

### 7.7 Stage 4 — one-open internal confirmation

All architecture, conditioning, checkpoint age, sampler and sample-count choices,
and the provisional finalist ranking, are frozen on the internal selection
split. The disjoint internal confirmation split is opened once.

The confirmation stage can reject the prespecified finalist or weaken a contrast
claim. It cannot choose a new checkpoint or promote a runner-up. Cost-based ties
and simplicity rules are resolved on internal selection before opening it.

### 7.8 Stage 5 — external evaluation and replication

Only internally confirmed, frozen variants are evaluated once on fresh external
evidence to test the registered contrasts. The provisional finalist has already
been chosen on internal selection; external results confirm or reject that
choice. Only it receives a second seed, and only after every simultaneous
calibration, physics, coherence and proper-score gate passes. Report contrasts
among all predeclared survivors, including negative results, without using this
external panel as a second model-selection set.

Replication reuses the exact data, target, architecture, checkpoint-age rule and
sampler contract. It is not an opportunity to tune. A replication failure closes
the finalist claim; it does not promote an unreplicated runner-up.

## 8. Shared-superpatch overlap and coherence tests

### 8.1 Two questions that must not be conflated

The overlap programme contains a hard computational identity test and a
statistical posterior test.

**Computational identity:** if the same posterior realization and same physical
voxel are evaluated through two sufficiently haloed crops, the shared coarse
state and globally addressed stochastic input must agree exactly to numerical
tolerance. A disagreement is an implementation failure.

**Statistical coherence:** across independent held-out simulated parents, the
joint distribution of fields and tidal summaries spanning two children must be
calibrated. Mechanical equality of duplicated voxels does not prove this.

### 8.2 Canonical assembly test

For sample `m`, sibling crops `A` and `B`, and their valid duplicated region:

```text
max |delta_A^(m)(x) - delta_B^(m)(x)|
```

is measured after matching the required context halo. The tolerance is frozen
from float precision, transform round trips and deterministic subdivision tests,
not chosen after seeing a trained model.

The same test is repeated for:

- `delta_L` and `delta_H` separately;
- assembled `delta_R7`;
- five independent tidal-tensor components;
- ordered eigenvalues and eigengaps away from declared degeneracy tolerance.

The final parent field uses canonical ownership, never overlap averaging. A
blended field may be displayed only as a diagnostic and cannot enter posterior
scores.

### 8.3 Cross-sibling posterior calibration

There is generally one held-out truth field per simulated observation, not a
direct Monte Carlo sample from the exact conditional posterior. Therefore do not
invent a target correlation coefficient or demand arbitrary `r=1` calibration.
Use simulation-based calibration across held-out parent examples.

Freeze cross-sibling projections before evaluation, including:

- Fourier coefficients and phases supported across both children;
- coarse wavelet coefficients spanning the boundary;
- sums and differences of matched regional density/shear summaries;
- paired tidal-tensor components at registered separations;
- concatenated eigenvalue/eigengap vectors at paired locations;
- boundary-crossing variograms and structure functions.

For each projection, rank the held-out truth among posterior draws. Use
multivariate TARP on concatenated sibling summaries and clustered SBC with
parent/phase resampling. This tests the joint law without pretending repeated
truths exist for one observation.

Also test the primary regional fractions/pair statistics and secondary
connectivity events from Section 2.2, evaluated on each coherent draw. The
independent-sibling control must preserve single-region marginal samples while
destroying their coupling, so a detected change can be attributed to dependence.

### 8.4 Seam and boundary tests

Measure, as functions of distance to the child boundary:

- bias and normalized residual;
- posterior width and empirical error;
- local power and cross-power;
- variogram/structure-function residual;
- tidal trace and traceless-shear error;
- web-class probability reliability;
- disagreement between duplicate crop computations.

Compare boundary-crossing pairs with matched interior pairs of the same
separation, shell, response and target environment. A visible seam is a failure,
but absence of a visible seam is not a pass without the statistical tests.

### 8.5 Required negative control

At evaluation only, use two different controls: independently evolve duplicated
crops to test computational identity; independently permute posterior draw IDs
between disjoint regions to preserve their marginals while removing covariance.
The latter tests statistical dependence without relying on an obvious duplicated
voxel mismatch. It should be detectable for registered, nonzero-covariance
alternatives; a true nearly independent pair need not degrade. These are
evaluation-only controls, with no additional trained models.

## 9. Validation hierarchy

### 9.1 Numerical and physical correctness

Every variant and every draw must pass:

- finite, non-degenerate samples;
- immutable checkpoint reload and deterministic sample-prefix replay;
- exact mask, coordinate and crop alignment;
- multiscale analysis/synthesis closure;
- no double smoothing;
- Hermitian/real-field closure where Fourier operations are used;
- symmetric tidal tensors;
- `trace(T) = delta_R7` to numerical tolerance under the frozen DC convention;
- ordered finite eigenvalues;
- stable handling/flagging of near-degenerate eigenvectors.

### 9.2 Marginal field diagnostics

Report, but never select on alone:

- voxelwise SBC/PIT/ranks;
- 50/68/90% central coverage;
- marginal CRPS;
- posterior width versus empirical error;
- posterior mean `R2` and cross-correlation;
- density-tail and support behavior.

### 9.3 Joint field diagnostics

The primary field suite includes:

- parent-level energy score;
- coarse-scale energy score;
- variogram score at frozen separations;
- Fourier/wavelet ranks and amplitude/phase diagnostics;
- auto- and cross-power spectra with posterior uncertainty;
- covariance of registered spatial projections;
- peak, void and topology summaries as descriptive diagnostics;
- the registered regional fraction and environment pair functionals as primary
  science checks, and the preregistered connectivity event as a secondary check;
- cross-sibling tests from Section 8.

Energy score alone can be insensitive to dependence errors. It is a primary
paired score only inside simultaneous variogram, scale and calibration gates.

### 9.4 Derived tidal diagnostics

Apply the fixed physics map to every draw and evaluate:

- the five independent traceless-shear components;
- trace and shear amplitude/shape;
- ordered eigenvalues;
- both positive eigengaps;
- threshold/web-class probabilities;
- eigenvector orientation where gaps exceed the frozen degeneracy threshold;
- the same quantities jointly across sibling locations.

Per-galaxy physical diagnostics must use the exact supported galaxy rows and
interpolation convention. P12-A is shown as an external per-galaxy reference,
not as a teacher or truth target. A research field finalist does not replace
P12-A. Any replacement claim requires a separate non-inferiority contract and
user authorization.

### 9.5 Conditional calibration

Hard conditional gates use only inference-time/deployable strata:

- shell/redshift;
- cap;
- random response and sampling intensity;
- boundary distance and mask/hole type;
- tracer density/occupancy;
- predicted density, predicted shear and posterior width;
- parent support fraction and child position within parent.

Truth-density/environment strata remain important descriptive diagnostics but
are not trainable correction variables and are not used to flatten coverage.
Coverage conditional on realized truth need not be flat even for a correct
Bayesian posterior.

Uncertainty intervals and paired comparisons resample parents, spatial blocks,
phases or modes as appropriate. Galaxies and voxels are never treated as IID
replicates.

Pooled scalar ranks can pass even if the sampler ignores observations and draws
from the prior. [Modrak et al.](https://arxiv.org/abs/2211.02383) establish why the
choice of data-dependent test quantities matters. Require an observation-ignored
prior control and shuffled-condition control in addition to width/covariance
controls. In the analytic fixture use the known observation likelihood as a test
quantity; for the field use frozen deployable-condition strata and independent
observation-sensitive discrepancies justified by the available simulator. Neither
block resampling nor many Fourier projections alone guarantees conditional
correctness for every possible observation.

Freeze a small primary family of summaries and strata; use simultaneous
parent/source-block uncertainty bands and explicit minimum effective sample
counts. A gate passes when its uncertainty bound is inside a scientific
tolerance, fails when incompatibility is resolved, and is otherwise
**inconclusive**. Sparse strata or too few independent phases cannot silently
count as a pass. Thresholds such as 0.05 are scientific tolerances, not p-values.
Exploratory combinations remain supplementary to control multiplicity.

### 9.6 Decision status of experiments and posterior checks

The implementation contract must label every result before training. The
following hierarchy is binding:

| Status | Items | Decision authority |
| --- | --- | --- |
| **Hard qualification gates** | `R0-PHYSICS`; numerical and sampler correctness; source-block coverage/SBC/TARP in a small registered field/tidal/sibling family; deployable-stratum calibration; long-band power; proper scores; synchronous parent identity; primary environment-functional uncertainty | Resolved failure disqualifies; inconclusive evidence does not promote |
| **Gated experiments** | `R0-WAVELET`; bounded context selection; objective-specific coarse canaries; `R1-WAVELET`; diagnostic sensitivity study including `MIRA-ACTIVATION` | License a frozen component or test before model evaluation |
| **Secondary science checks** | The registered connectivity event, void component volume and their posterior calibration | Required for a connectivity claim; failure does not imply the primary environmental-correlation claim passed or failed |
| **Supplementary posterior checks** | Additional pooled PICP; relative variance; diversity spectra; posterior-mean `R2`; exploratory PDFs/topology; truth-environment strata; visual samples; posterior-predictive re-observation without an adequate forward model | Describe limitations; cannot rescue a hard-gate failure |

Marginal coverage is hard only in its preregistered parent/block-resampled form
and together with the joint gates. A Cosmo3DFlow-style PICP pooled over correlated
spatial locations is useful for comparison but remains supplementary; voxels are
not IID calibration replicates. Likewise, agreement of predictive variance or
sample power with truth is not sufficient evidence for the conditional joint law.

#### `MIRA-ACTIVATION`

MIRA is promising because it consumes a truth and posterior draws without
requiring a tractable density, but its scalar form is necessary rather than
sufficient and depends on distance, whitening, centers and dimension. Before it
is applied to a science model, run one checkpoint-free diagnostic study using a
calibrated analytic correlated-Gaussian conditional field (tiny synthetic fixture
first; realistic survey masks in the later compute session)
and the following registered alternatives:

- mean displacement;
- a sampler ignoring observations, and a shuffled-condition sampler;
- uniform and response-dependent width inflation/deflation;
- deletion of registered long-mode traceless-shear covariance;
- deletion of cross-sibling covariance; and
- the independent-child sample-ID negative control.

Freeze training-only whitening, distance metrics, center distributions, projection
spaces, draw counts and a power criterion. The required spaces are the explicit
low Fourier coefficients, blocks of fine wavelet coefficients, derived
shear/eigengap projections and concatenated sibling summaries. A raw
million-dimensional Euclidean field score is not sufficient on its own.

If the calibrated null agrees with its finite-sample reference and every
registered alternative is detected at the frozen power, MIRA becomes a
co-primary hard joint-calibration gate alongside projected TARP/SBC. If not, it
remains a supplementary sensitivity diagnostic and cannot rank or rescue a
candidate. No metric or center retuning is allowed after model results are seen.

Use an independent draw to define the random region boundary; exclude it from
the N samples counting its mass. The finite-N reference is
`E[MIRA]=(2*N+3)/(3*(N+2))`, not exactly `2/3`
([MIRA, Appendix A.4](https://arxiv.org/html/2605.02014)). Regions and projections
of one parent do not multiply the number of independent examples. A scalar score
passing its reference remains a necessary check, never a certificate of the full
conditional law. The initial analytic fixture does not establish power against
lost sibling covariance; MIRA remains supplementary until a preregistered study
demonstrates that sensitivity.

### 9.7 Posterior-predictive re-observation

After direct simulation calibration, re-observation is possible only if the
required conditional forward model is available. A smoothed `delta_R7` field does
not specify halos, velocities, luminosities or unresolved density. An HOD is a
galaxy model conditioned on halos, not a complete `p(X | delta_R7)`. Formally the
required operation is:

```text
field draw
  -> unresolved matter / halos / velocities conditional on that field
  -> galaxy/HOD realization conditional on halos
  -> RSD
  -> angular/radial selection
  -> fibre/redshift response
  -> replicated X_final.
```

Compare held-out diagnostics not used as model losses: `n(z)`, counts/occupancy,
two-point clustering, marked/environment clustering, void statistics, graph
statistics and shell/boundary behavior. A Poisson painting shortcut is an
engineering diagnostic, not field-posterior closure.

No such completion model is assumed to exist in v1 and no fourth learned model
is authorized to create it. Use existing jointly simulated catalog/field pairs
for direct calibration and response degradations. Keep the forward-model gap
explicit in any proposed DESI posterior-predictive claim.

Real-DESI truth-free closure is a later research stage. It cannot establish
posterior calibration by itself.

## 10. Frozen comparators and claims

The evaluation table includes:

- deterministic conditional mean;
- P12-F G1 correlated-Gaussian field control;
- the final immutable D2 low-mode result, whatever its decision;
- `E2E-CFM`;
- `E2E-DIFF`;
- `E2E-WCFM`, only if `R1-WAVELET` licensed its full fit;
- independent-child sampling as an evaluation-only negative control;
- P12-A only for matched per-galaxy marginal context.

Comparisons must match target, supported rows/voxels, response, physics window,
draw count and scoring subsets. Historical scores are not copied into a new
table unless they are recomputed on the exact matched evaluation objects.

Allowed claims are layered:

1. **local field posterior:** direct local calibration passes, coherence does not;
2. **superpatch-coherent research posterior:** direct, derived and overlap gates
   pass on independent simulations;
3. **full-cap coherent posterior:** requires a later full-cap inference and
   stitching/shared-global-mode closure not authorized here;
4. **DESI posterior:** requires simulator adequacy and truth-free closure; never
   inferred from in-domain calibration alone.

No result from this plan changes the P12-A VAC claim.

## 11. Promotion, selection and terminal stop contract

Exact numerical thresholds must be frozen in the implementation contract before
science training. The starting scientific tolerances inherit the P12-F/D2
discipline:

- joint TARP maximum deviation at most `0.05`;
- global 68/90% coverage error at most `0.05`;
- maximum deployable-proxy conditional coverage error at most `0.10`;
- registered long-band power ratios within `10%` of unity;
- no registered proper-score regression above `1%`;
- positive paired parent-level primary-score improvement with an interval
  excluding zero;
- exact physics, finite-sample and overlap-identity closure;
- second-seed replication for a research finalist.

These are necessary, not sufficient. The contract must additionally freeze
cross-sibling TARP/SBC, science-functional, seam and context-convergence thresholds
from training-only truth/subdivision references, with source-block uncertainty
and an explicit inconclusive outcome. Power ratios refer to matched ensembles
under the same window, not a demand that each draw match its paired truth power.

### 11.1 Qualification before ranking

A model is eligible for provisional ranking only after passing internal-selection
gates. Its subsequent confirmation and external tests cannot be used to refit
that ranking. A visually attractive or lower-loss failure is not a finalist.

### 11.2 Registered contrasts and finalist selection

Report the two causal contrasts before selecting a finalist:

1. `E2E-CFM - E2E-DIFF` estimates the registered path/weighting/sampler-recipe
   effect under the baseline representation;
2. `E2E-WCFM - E2E-CFM` estimates the wavelet-coordinate/network-bias intervention
   under the same CFM path and field-space loss, if `R1-WAVELET` passed.

There is no registered `E2E-WCFM - E2E-DIFF` causal interpretation because both
objective and fine representation change. It may inform an operational finalist
choice, but not a flow-versus-diffusion or wavelet-versus-real-space claim.

On internal selection, if exactly one model qualifies it is the provisional
research finalist. If multiple models qualify:

1. compare paired parent-level energy score under hard variogram, spectral,
   conditional-calibration and overlap non-regression gates;
2. require a frozen practically meaningful improvement and parent-bootstrap
   interval excluding zero;
3. if the difference is inconclusive, choose the lower-cost sampler at
   convergence; if cost is also inconclusive, use the frozen simplicity order
   among eligible candidates: `E2E-CFM`, `E2E-DIFF`, `E2E-WCFM`;
4. replicate only that one provisional finalist. Failure of its second seed
   closes the programme without promoting a runner-up;
5. never describe the result as a generic model-family ranking.

### 11.3 Hard stop

The full programme permits:

- one `R0-PHYSICS` audit, one `R0-WAVELET` audit and one bounded context audit;
- one CFM and one diffusion coarse canary, with the CFM coarse checkpoint shared
  by both CFM fine representations;
- one paired, discarded `R1-WAVELET` fine canary;
- one full seed-42 hierarchy for `E2E-CFM` and `E2E-DIFF` if their coarse gates
  pass, plus `E2E-WCFM` only if both wavelet gates pass;
- one fresh external evaluation per frozen first-seed survivor;
- one second seed for the single provisional finalist only.

It permits no:

- fourth learned model, wavelet-diffusion arm or fully wavelet coarse arm;
- auxiliary power-spectrum loss;
- alternative wavelet/depth/network after a wavelet-gate failure;
- replacement runner-up after confirmation or replication;
- ph006-fitted recalibration;
- truth-environment correction;
- extra architecture or schedule sweep;
- same-objective presentation/update extension;
- successor automatically registered by a negative result.

If no model passes, write `E2E_FIELD_NO_FINALIST.json`, archive the scientifically
useful failure decomposition and close the programme. If one model passes both
first-seed selection and its licensed replication, write a research-finalist
marker that explicitly says `not_a_production_vac` and records which higher
claim layers remain closed.

## 12. P12-A, D2 and P13 firewall

### 12.1 P12-A

This programme must not:

- modify P12-A checkpoints, transforms, datasets, gates or release wording;
- read, regenerate or recalibrate P12-A ph001 truth/predictions;
- consume the P12-A one-open phase as field evidence;
- train against P12-A posterior draws;
- delay or cancel P12-A compute;
- reinterpret a field-model failure as a P12-A defect.

P12-A artifacts are read-only scientific references. All new code, configs,
scratch roots and evidence namespaces are separate.

### 12.2 D2

D2 is owned by its active workflow until its decision marker is frozen. This
plan performs no competing edits, launches, allocations or evaluation. After
closure it may ingest only immutable D2 reports/checkpoints named in the new
contract.

### 12.3 P13 and production

No P13/Loa handoff, production export, VAC schema or DESI release decision is
authorized. A successful result is a v2 research candidate requiring a separate
production plan, full-cap scaling study, simulator-adequacy programme and user
authorization.

## 13. Proposed implementation layout

Names are provisional until the contract is approved.

```text
configs/e2e_field_posterior_v1.json

workflows/sbi/e2e_field_build_superpatches.py
workflows/sbi/e2e_field_transforms.py
workflows/sbi/e2e_field_models.py
workflows/sbi/e2e_field_train.py
workflows/sbi/e2e_field_sample.py
workflows/sbi/e2e_field_evaluate.py
workflows/sbi/e2e_field_overlap.py
workflows/sbi/e2e_field_decide.py

tests/phase4/test_e2e_field_contract.py
tests/phase4/test_e2e_field_transforms.py
tests/phase4/test_e2e_field_overlap.py
tests/phase4/test_e2e_field_sampling.py

docs/evidence/e2e_field_v1/
docs/figures/e2e_field_v1/
```

Source code and scientific contracts live in the repository. Large targets,
caches, checkpoints and draw archives live under a separately named Scratch
root and are content-addressed. CFS remains read-only.

Required immutable artifacts include:

- parent/child/source manifest;
- phase-role and leakage audit;
- target-transform and multiscale round-trip report;
- conditioner/response contract;
- architecture/capacity parity report;
- RNG/subdivision invariance report;
- checkpoint and sampler-convergence markers;
- per-parent draw manifest;
- fixed-physics validation report;
- joint calibration/proper-score report;
- shared-superpatch overlap report;
- `R0-WAVELET`, `R1-WAVELET` and `MIRA-ACTIVATION` decision reports;
- registered causal-contrast and finalist-selection report;
- visual audit marker;
- terminal decision JSON.

All long GPU training is submitted only after bounded interactive import,
data-read, memory, exact-resume and sample-replay smoke tests. Slurm submission
requires separate explicit user approval and follows the repository/NERSC
resource policy.

## 14. Decisions required before science training

The following are deliberately unresolved and must be answered without external
validation leakage:

1. Freeze exact regional/environment-pair summaries and the secondary
   connectivity event from the user's chosen science directions (Section 2.2).
2. What fresh phase/ensemble will supply external confirmation and a later blind
   claim?
3. What parent, observation-context, child-core and halo sizes pass the bounded
   context audit?
4. Which baseline complementary transform and, conditionally, which single
   orthonormal wavelet/depth/boundary rule pass `R0-WAVELET`?
5. Are frozen G1 summaries retained as conditioner channels only?
6. What D2 architecture/sampler result is immutable and eligible to inherit?
7. What parameter/FLOP and presentation budgets give credible matched
   `CFM-DIFF` and `WCFM-CFM` contrasts?
8. What practical improvement and non-regression tolerances govern
   `R1-WAVELET`?
9. Which cross-sibling projections and practical thresholds define coherence?
10. Which MIRA distances, whitenings, centers and sensitivity/power thresholds
    govern `MIRA-ACTIVATION`?
11. Which frozen field-to-galaxy forward model can support posterior-predictive
    re-observation?
12. What result is sufficient to stop at a superpatch research posterior rather
    than pursue full-cap inference?

No answer may be chosen using ph001 or by fitting ph006 failures.

## 15. Initial execution checklist

- [ ] Wait for and review the immutable D2 result without modifying its branch.
- [x] Choose environment correlations as primary and connectivity as secondary;
  specify initial functionals (Section 2.2), pending geometry/power audits.
- [ ] Designate fresh external and future blind simulation evidence.
- [x] Freeze the internal data-build geometry candidates: paired 64/96-cell
  parents, 96/128-cell context, central 32-cell science region. Scientific
  adequacy and the final training population remain open.
- [ ] Implement the checkpoint-free superpatch/transform/RNG audit and decide
  `R0-WAVELET`.
- [ ] Freeze the Fourier coarse transform, baseline fine transform, at most one
  eligible wavelet transform and one common conditioner.
- [ ] Freeze the matched three-arm architecture, compute and causal-contrast
  contracts.
- [ ] Run `MIRA-ACTIVATION` and preregister whether MIRA is hard or
  supplementary.
- [ ] Preregister calibration, proper-score, overlap and terminal gates.
- [ ] Pass unit, exact-resume, crop-invariance and bounded GPU smoke tests.
- [ ] Request explicit authorization before any Slurm training submission.
- [ ] Run the bounded internal funnel, including at most one `R1-WAVELET`,
  without ph006/ph001 tuning.
- [ ] Evaluate frozen survivors once on fresh external evidence.
- [ ] Replicate only the single provisional finalist.
- [ ] Write exactly one terminal finalist or no-finalist marker.

The research question is complete when that terminal marker is written. A
negative result is an answer, not an invitation to append a fourth model.

## 16. Data preparation started on 2026-09-05

`configs/e2e_field_data_prep_v1.json` is a preparation specification, not a frozen
training contract. `workflows/sbi/e2e_field_prepare_data.py` reads only bounded
source JSON metadata, hashes that metadata, checks response/target grid identity
and source-file presence, and proposes matched parent/child geometry. It rejects
ph001 before opening any phase-specific source, including symlink aliases.

The initial products in `docs/evidence/e2e_field_v1/preparation_20260905/` contain
12 phase/cap source records and 96 parent proposals across six visible phases,
with child core ownership, context extents, explicit Mpc/Mpc-h units and
provisional periodic source-box overlap pairs. The response adapter is the
repaired R1 contract, not a reconstructed galaxy-occupancy response. All proposals
have `support_screened=false`, `split_assigned=false`, and `training_ready=false`.
Stored source hashes are provenance references; large payload hashes have not
been recomputed. Historical normalizers and target values were not loaded.

`workflows/sbi/e2e_field_calibration_fixture.py` also creates a small exact Gaussian
field/observation/posterior product under
`docs/evidence/e2e_field_v1/analytic_fixture_20260905/`. It includes correct,
observation-ignored, independent-sibling and under-width draws, plus raw per-parent
diagnostic quantities. It is synthetic test data, not an Abacus training sample.
It illustrates that removing sibling covariance changes regional uncertainty
even when local marginals remain correct. Eleven focused tests cover phase and
unit rejection, periodic overlap, ownership, Gaussian conditioning, MIRA sample
separation, orthogonal-loss equivalence and synchronous versus independent crop
evolution.

The next compute session is specified in
`docs/e2e_field_research_review_20260905.md`. It must first screen proposals using
random response, validate source-box grouping and coordinates, build native
tensor reference patches, and run `R0-PHYSICS` before constructing a science-ready
parent dataset. No salloc, Slurm job, HDF5 target extraction, full-volume FFT or
learned training was run in this preparation session.

### 16.1 Authorized input build on 2026-09-08

The subsequent user instruction authorizes one concurrent interactive slot for
data preparation, with D2 untouched and no model fits. The build contract is
`configs/e2e_field_build_v1.json`; the operational and scientific handoff is
`docs/e2e_field_data_build_20260908.md`. It uses 160 response-screened anchors
across five historically exposed phases, with 320 nested views, whole-phase
roles and train-only normalization. Neither ph001 nor ph006 payloads are used.

Raw arrays may be packaged in quarantine to enable the remaining physical
audit; this is not a relaxation of R0-PHYSICS before science training. Default
dataset access rejects unreleased products and separately guards confirmation.
Native-reference and full-size numerical completion markers determine technical
readiness. The physical error budget, final common conditioner/transforms,
diagnostic-power study and model contracts remain unresolved; `training_ready`
must remain false until those gates are satisfied.

### 16.2 Physical error attribution completed on 2026-09-08

`configs/e2e_field_error_budget_v1.json` freezes the no-fit comparison over all
96 training anchors. Independent full-box spectral references, FD2/4/8 controls,
floor/nearest/linear/cubic sampling and 64/96/128-cell parent reconstructions
were compared on the same 32-cell central science region. Regional filling,
pair statistics, fixed-terminal six-neighbour connectivity and largest-void
fractions are reported separately for complete and observed support, per phase
and shell/support stratum. No selection/confirmation targets were used to choose
these outcomes, and no ph001/ph006 payloads were read.

Evidence is in `docs/evidence/e2e_field_v1/error_budget_20260908/`. Typical FD2
RMS is small, but an interior void-component example changes by 8.2 percentage
points. Floor sampling and exterior tides are larger typical effects; the
128-cell diagnostic does not establish topology safety. Thus the completed
attribution audit is not a scientific R0-PHYSICS pass. The recommended separate
spectral target/interpolation contract and final domain still need an explicit
decision and acceptable science-functional tolerances before training release.

### 16.3 Spectral v2 completion and finite-domain stop on 2026-09-09

Separate Gaussian R7/spectral-tensor products are technically complete on all
160 anchors (job 58115135). The user then authorized the matching cubic-target
64/96-cell domain audit and downstream contracts before a canary. The contract
in `configs/e2e_field_domain_gate_v2.json` was committed before that run, with
the previous audit already known: its thresholds are internal design criteria,
not independent validation or a measured posterior-scaled error budget.

Job 58120066 tested all 96 training anchors; no holdout payloads were opened.
Neither domain passes. At 96 cells all observed-support filling/pair statistics
pass the chosen limits, but 95/96 anchors fail the combined eigenvalue/class/
topology screen. Worst observed largest-void error remains 17.43 percentage
points; a truth-assisted constant traceless correction does not resolve every
topology failure. See `docs/e2e_field_domain_gate_20260909.md` and its receipt.

This settles the registered domain choice as **neither for the joint tidal and
topology purpose**. The final model/transform freeze and learned canary stop at
R0; do not reinterpret the common target/diagnostic requirements as a released
training recipe. Wider-domain or exterior-tide-aware modelling, or a narrower
density-only claim, require an explicit subsequent scope decision. No learned
fit, MIRA activation, posterior-scaled pass or training release occurred.
