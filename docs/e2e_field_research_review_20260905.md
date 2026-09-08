# E2E field posterior: scientific review and data preparation, 2026-09-05

This review records the evidence behind revisions to
[the programme plan](plan_end_to_end_conditional_field_posterior.md). The user
prioritized environment correlations and filament/void connectivity, and requested
research and data preparation while leaving allocations for a later day.

## Scientific decision

Retain the two primary CFM/diffusion recipes and the gated wavelet-CFM challenger.
Their purpose is to determine whether an amortized conditional field distribution
can propagate uncertainty into regional environment correlations and connectivity.
The first experiment is a target/domain audit with true fields. It determines
whether the chosen finite stochastic domain contains enough information to recover
the physical tidal statistics, before attributing an error to a trained generator.

The primary science functionals are regional web fractions and pair probabilities
of at least two collapsed directions, using `lambda2 > 0.2`, on fixed interior
voxels and physical separation bins. The secondary functionals are a registered
connectivity event of that excursion set and void component volumes. Draw-wise
statistics preserve spatial covariance; statistics formed from independent
per-location posterior rows need not. Galaxy-marked clustering and continuous
filament catalogues require additional coordinate, selection and topology choices.

## Literature and its implications

| Source inspected | Evidence relevant to this study | Decision |
| --- | --- | --- |
| [Jasche & Wandelt, BORG](https://arxiv.org/abs/1203.3639) | Physical Bayesian inference from realistic galaxy surveys already produces present density/velocity reconstructions and propagates observational uncertainty | Do not claim novelty for the general survey-conditioned field posterior |
| [Leclercq, Jasche & Wandelt](https://arxiv.org/abs/1502.02690) | Posterior density realizations from SDSS have already been used for tidal-web probabilities and global web fractions | Compare our contribution through amortization, calibration, response handling and measured science-functional accuracy |
| [CosmoFlow](https://arxiv.org/abs/2507.11842) | Scale-organized flow representation learning for 2-D density maps | Useful inspiration; an encoder that sees its reconstruction target cannot be our deployment condition |
| [Cosmo3DFlow v2](https://arxiv.org/html/2602.10172v2) | 3-D wavelet CFM for initial conditions from evolved matter/halos; periodic domains; spatial PICP/variance diagnostics | Retain a bounded wavelet/network experiment; do not transfer periodic padding or interpret high pooled coverage as full conditional calibration |
| [Lipman et al., Flow Matching](https://arxiv.org/abs/2210.02747) | Gaussian flow-matching paths include diffusion paths | Compare the full path/weighting/parameterization/sampler recipes; family labels alone are not a causal variable |
| [Kostic et al.](https://arxiv.org/abs/2212.07875) | Controlled field-inference consistency tests against known limits | Require analytic known-posterior tests before cosmological inference |
| [JADE](https://arxiv.org/abs/2606.31988) | Joint lensing-map/cosmology diffusion inference uses full-joint MIRA and cosmology TARP | Field-level multivariate posterior checks have precedent; dimension and estimand still matter |
| [MIRA](https://arxiv.org/html/2605.02014) | Sample-only score, necessary scalar calibration condition, finite-sample null expression | Keep supplementary until sensitivity to our registered conditional/dependence failures is demonstrated; exclude the boundary draw from the mass count |
| [Modrak et al., SBC test quantities](https://arxiv.org/html/2211.02383) | Parameter-only ranks can miss ignored data; data-dependent test quantities change sensitivity | Add observation-ignored and shuffled-condition controls; clustered uncertainty alone does not repair this blind spot |
| [Generative Diffusion Priors for 3D Mapping](https://arxiv.org/abs/2606.00803) | Learned 3-D matter priors support a weak-lensing posterior sampler | Further limits broad claims of first 3-D diffusion field inference |
| [Patched Flow Matching](https://arxiv.org/abs/2606.22084) | A fluid reconstruction method assembles patch contributions to a flow vector field over larger domains | Use one parent evolving state; distinguish computational consistency from calibration of the resulting distribution |

This is a targeted review, not an exhaustive priority search. The likely useful
contribution is a reproducible benchmark and an accurate, efficient approximation
for the specified DESI-like problem. A prior-based or response-shuffled generator
with realistic power spectra is an explicit negative control, not evidence of
successful conditioning. Training cost, inference cost and the break-even number
of scientific analyses should accompany an eventual speed claim.

## Corrections established by repository inspection and reasoning

1. **Units and epoch.** `docs/evidence/p3/p3_field_schema_v1.json` and all inspected
   target/response metadata use 5 Mpc observer cells, equal to 3.383 Mpc/h under
   the existing `h=0.6766` coordinate conversion. The target snapshot is z=0.2,
   even though observed-redshift shells span 0.15--0.55. This is a fixed-epoch
   mock experiment, not training on cosmological evolution over that range.
   The simulation cosmology and observer conversion must remain separately named.
2. **What the target contains.** `p12f_build_field_targets.py` samples the trace
   of native R7 eigenvalues at nearest native-grid cells, on the P3 lattice. The
   full cap target includes matter outside the observation mask. Existing loss
   masks encode old P4 coverage; they cannot define a new parent posterior domain.
3. **Exterior tides and crop DC.** The finite-domain density-to-tensor map depends
   on boundary conditions. A harmonic exterior potential has a traceless Hessian,
   invisible to the density trace. Parent DC carries a fluctuating regional mean;
   deleting it because the source periodic box has zero mean is not justified.
4. **Coherence.** Equal initial noise cannot prevent independently integrated
   crops from diverging. Receptive-field effects propagate on every sampler step.
   A tiny finite-difference test demonstrates this; synchronous evaluation of
   tiled updates from the same parent state matches whole-parent evolution.
5. **Representation.** Orthonormal transforms preserve Gaussian noise and the
   Euclidean CFM loss if transformed consistently. Wavelet coordinates alone do
   not define a better probability model. The actual intervention includes the
   network bias, with common field-space loss weights and complementary-subspace
   noise. Projecting only the final output is not sufficient to preserve the
   specified training path.
6. **Evidence and selection.** Observer-space separation does not prove source-box
   independence. Different caps or replicated box regions can alias the same
   truth. No new independent phase is registered for E2E yet. Provisional model
   selection must precede confirmation/external evaluation; a second seed on the
   same phase tests training repeatability rather than independent-universe
   replication.
7. **Forward-model sufficiency.** A smooth R7 field does not specify halos or
   velocities. Re-observation needs unresolved variables conditioned on that
   field; an HOD alone cannot fill this gap. This remains a deferred science
   capability and does not authorize another learned model.

## Products built in this session

Preparation specification: `configs/e2e_field_data_prep_v1.json`.

Metadata preparer: `workflows/sbi/e2e_field_prepare_data.py`. Its output is
`docs/evidence/e2e_field_v1/preparation_20260905/`:

- `source_inventory.json`: 12 phase/cap records, target and repaired R1 response
  metadata hashes, file presence, lattice units and source references;
- `parent_geometry_proposals.json`: 96 matched parent proposals (64 and 96 cells
  per side), all child core/halo coordinates, shared anchor IDs and context bounds;
- `source_box_overlap_proposals.json`: provisional source-box conflicts from the
  inherited mapping, including periodic edge crossings; and
- `PREPARATION_COMPLETE.json`: configuration/code/product hashes, read scope,
  unresolved compute tasks and explicit `training_ready=false`.

No target values, observation arrays or normalizers were loaded. Large payload
checksums were recorded from existing metadata, not recomputed. Candidate centers
are lattice proposals, not a representative or support-selected science sample.
The parent proposals cannot be split for training until the full support and
source-identity audit passes.

Analytic fixture builder: `workflows/sbi/e2e_field_calibration_fixture.py`.
`docs/evidence/e2e_field_v1/analytic_fixture_20260905/` contains a small compressed
NPZ and JSON report for 128 independent 12-cell Gaussian examples with four noisy
observations and known posterior mean/covariance. Four candidate sample sets are
included: exact posterior, observation-ignored prior, independent siblings with
correct local marginals, and half-width posterior.

For this fixture the exact mean conditional variance of the sum of all cells is
20.370. Removing covariance between the two six-cell regions reduces it to 14.273
(about 30%) without changing either region's marginal posterior. The scalar MIRA
means are 0.650 +/- 0.021 and 0.638 +/- 0.022 respectively; this small panel cannot
establish sensitivity to the covariance defect. MIRA therefore remains
supplementary. With 31 mass-count draws plus one independent boundary draw, the
finite-sample null mean is 0.6566 rather than 0.6667. The raw per-example scores
are retained for later validation; these standard errors describe independent
synthetic examples only and are not applicable to correlated Abacus parents.

The observation-ignored prior's data-dependent discrepancy rank has mean fraction
0.047, versus 0.500 for the exact posterior. Its scalar sum-rank mean is close to
one half, illustrating why that mean is uninformative by itself; a full rank
distribution and condition-sensitive checks are required. None of these toy
results is a cosmological calibration pass or a completed power study.

## Concrete next compute session (planned, not launched)

Start with a bounded CPU allocation; no GPU training is needed for this stage.
Choose resources after inspecting the native tensor storage format. Source
eigenvalue slabs are hundreds of MB each, so native tensor regeneration or
full-volume FFTs must stay off the login node. Do not request or submit an
allocation during the present session.

| Stage | Inputs and operation | Output / scientific gate |
| --- | --- | --- |
| A: response-only screening | Existing response/count HDF5 chunks at proposed parents; exact support plus shell/boundary summaries; at most the preregistered geometry candidates | Supported parent manifest, rejected-parent reasons and masked/boundary/interior strata; no truth-based center selection |
| B: source identity and coordinates | Parent/context footprints, periodic source mapping, native IDs and catalogue real-/observed-redshift coordinates | Source-superblock connected components and explicit role ledger; reserve fresh external evidence; unit/frame audit |
| C: native target reference | Native periodic matter/T-web sources; same registered interpolation and central locations for both parent extents | Scalar and six-component tensor references, native-grid/interpolation error, parent mean, source hashes; eigenvalues alone cannot provide orientations |
| D: truth-only domain comparison | Same central truth, varying parent/physics halo only; measure full-box versus finite-domain tensor and science functionals | `R0-PHYSICS`: boundary/DC/discretization error budget; stop if the proposed domain cannot support the tidal claim |
| E: parent extraction | Frozen accepted manifests and masks after A--D; target values across the whole latent domain, observed channels only as conditioner | Chunked parent products and separate observed/latent/truth/loss/science masks; source hashes and provenance |
| F: representation and sampler geometry | Complete padded parents, complementary projection, real/wavelet noise; dummy local vector field synchronized per step | `R0-WAVELET`, subspace closure and full-parent/tiled-sampler identity; retain coarse mean |
| G: diagnostic power | Exact analytic conditional fields with registered survey masks, projected summaries and controlled errors | Family-wise null/power study, observation-sensitivity and sibling-covariance detection; explicit MIRA activation or supplementary-only result |

The first allocation should finish A/B and estimate C/D resource requirements if
the native full tensor is unavailable; it must not assume that reading the scalar
target already provides the independent physical reference. Subsequent extraction
is licensed only by the scientific adequacy tests. Full training awaits D2's
immutable result and a separate frozen training contract.

## Validation and handoff

Eleven focused tests passed in `cosmic_env` for phase/symlink rejection, unit
consistency, metadata-only operation, periodic overlap, core ownership, Gaussian
posterior algebra, MIRA boundary-sample exclusion, orthogonal-loss equivalence,
and shared-state versus independent-crop evolution. The metadata products and
small synthetic fixture were generated without an allocation. P12-A/D2 code,
jobs, gates, data and ph001 were not modified by this work.
