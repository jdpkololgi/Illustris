# BGS sparsity and smoothing investigation after fixed scale retraining

Date: 27 October 2026. Audience: GraphWeb scientific and implementation team.

The immediate priority is to retrain both model classes at the existing Gaussian width **R = 7 h^-1 Mpc**: the conditional density-field model and the P12-A per-galaxy eigenvalue posterior pipeline. Preserve the physical target definition while incorporating the repaired numerical implementation and the applicable corrected data lineage. The investigations below are deferred follow-up work. They are not additional prerequisites for this retraining, and the proposed alternative scales are not adopted targets.

Sparsity can reduce the information available about an environment without changing the physical environment we wish to measure. At fixed R, the scientific question is whether posterior uncertainty correctly reflects that loss of information. A redshift-dependent target would answer a different question, with a different definition of environment at each redshift.

## Present evidence and model scope

The two model classes have different targets and dependencies.

| Model class | Existing target and conditioning | Retraining implications |
| --- | --- | --- |
| Conditional density-field model | Native matter field smoothed at R = 7, represented on a fine grid with cell width 6.766 h^-1 Mpc. Observation channels include counts, expected counts, sampling density, angular response, boundary distance and observer redshift. | Keep the target and selection conditioning. Use the repaired deterministic training path. The native matter target was not automatically invalidated by the galaxy label lookup error. |
| P12-A per-galaxy posterior | Ordered tidal eigenvalues at galaxy positions. U-PATCH base predictions and response covariates feed flow-matching posterior estimation; covariates include redshift, sampling density, cap and distance to the support boundary. | Retrain against corrected own-position galaxy labels. Build fresh out-of-fold and full-training encoder predictions before fitting the downstream posterior. Preserve R = 7 and the ordered posterior representation. |

Sources: [field target configuration](../configs/e2e_coupled_data_v1.json), [field conditioning implementation](../workflows/sbi/e2e_coupled_condition_products.py), [expanded P12-A configuration](../configs/p12a_corrected_expanded_inputs_20261026.json), and [readiness report](../outputs/field_restart_repair_20261026/REPORT.md).

Both training paths passed numerical restart checks, with zero parameter replay difference in the recorded checks. All fifteen P12-A cross-fit contracts and eighteen response products are prepared. The P12-A downstream tables still depend on fifteen fresh out-of-fold encoder fits, one full-training encoder, halo48 prediction exports and aligned table construction. Numerical readiness is not posterior calibration. Training budgets and scheduler resources remain separate run-planning decisions.

The earlier whole-field pilot gives a reason to study sparsity, but does not isolate its effect. U-Net density CRPS rose from 0.106830 at region-center redshifts 0.154–0.248 to 0.188359 at 0.452–0.545, approximately 76%. Geometry, field structure and other covariates also differ between shells. Neither evaluated field configuration was qualified as a calibrated field posterior. These historical results describe the previous fits, not the outcome of the proposed retraining. See the [pilot closeout](e2e_wholefield_pilot_closeout_20261014.md).

## What the existing smoothing guide measures

The earlier guide used 63,897,781 rows from an unmasked mock parent and full-sky shell volumes. Its BGS Bright illustration applied the magnitude cut r < 19.5. It did not measure the final footprint, fibre-assignment or redshift-success selection. Its [saved measurement manifest](../outputs/bgs_scale_support_redshift_20261019/manifest.json) explicitly labels the curves as non-optimal sampling guides.

For homogeneous unweighted Poisson sampling, a normalized three-dimensional Gaussian kernel gives

\[
N_{\rm eff}=8\pi^{3/2}\,\bar n\,R^3.
\]

Holding this reference effective count fixed implies R proportional to nbar^(-1/3). Thresholds of 10 and 30 were illustrative choices, not accuracy or calibration requirements. The steep high-redshift rise therefore flags declining sampling support; it does not establish an optimal Gaussian width or justify smoothing at tens of Mpc/h.

A useful reconstruction scale also depends on clustering signal, tracer bias, redshift-space distortions, weights, boundaries and correlated missingness. Flux limitation changes the luminosity distribution as well as the number of observed galaxies. Random thinning alone cannot establish robustness to that changing tracer population. DESI also documents spatially varying spectroscopic success and corresponding corrections; successful correction of clustering statistics does not by itself validate a local-field likelihood. [DESI spectroscopic systematics](https://arxiv.org/abs/2405.17208)

## Consequences for conditional flow models

A conditional flow model can represent uncertainty in a fixed-scale target even when the observations become sparse. In a well-calibrated field posterior, individual draws may contain plausible structure while disagreeing about its position and amplitude. Their average can be smooth. Realistic-looking draws or realistic auto-power do not prove recovery of the particular underlying field.

The desirable response to weak observations is increasing uncertainty and, in the limit of little information, behaviour approaching the appropriate conditional prior. Conditioning on selection variables enables that response but does not guarantee it. Selection mismatch, changing tracer bias, insufficient context or an inadequate learned distribution can instead yield confident errors. Bayesian field reconstruction provides a precedent for propagating survey selection and observational uncertainty, not a validation of these learned models. [Jasche and Wandelt](https://arxiv.org/abs/1203.3639)

For P12-A, test the joint ordered eigenvalue posterior and derived class probabilities at galaxy positions. Its per-galaxy draws are not automatically a coherent joint field posterior: independently drawing triplets and smoothing them cannot establish regional or cross-galaxy uncertainty. For the field model, test spatial dependence and regional tidal quantities in addition to voxel marginals. A direct architecture ranking does not answer either calibration question.

Distinguish three scales throughout: target smoothing, observation preprocessing and numerical grid resolution. Adaptive observation processing can preserve a fixed physical target. Adaptive target smoothing changes the estimand. Numerical resolution limits what either approach can represent.

## Deferred investigation sequence

Carry out these investigations in sequence after the fixed-scale retraining and its standard validation. Start with the R = 7 result; expand scope only when the preceding evidence motivates it. Register numerical gates and compute budgets before running a diagnostic campaign.

### 1 Measure the selected observation constraint

Use the actual selected catalogue and registered survey-response products to measure expected counts, support and sampling variation across redshift, cap and boundary distance. Distinguish structural mask zeros from low expected counts. Record the selection function, weighting convention and coordinate-volume convention beside each measurement.

Retain homogeneous Poisson effective counts as a reference. Estimate additional noise or covariance from forward observational realizations where missingness is correlated. An inverse-selection weight can correct a mean while amplifying variance; it does not restore missing information.

Deliverable: a support and noise assessment at R = 7, with the magnitude-only guide shown as a separate reference. This identifies which strata need focused tests without choosing a new target scale.

### 2 Separate sparsity from tracer and selection effects

Hold mock matter truth and geometry fixed. Apply several controlled thinning levels to a sufficiently dense observable sample, with independent realizations at each level, and propagate the changed selection into model conditioning. Identify levels outside the training distribution explicitly. This isolates sensitivity to reduced counts within that mock setup.

Run luminosity-selection and realistic observational-selection comparisons separately. Random thinning does not reproduce luminosity-dependent bias, correlated fibre incompleteness or redshift failures. Matched luminosity or volume-limited controls can help distinguish population changes from count changes, but their conclusions apply only where the matching is feasible.

Deliverable: paired changes in predictive accuracy, uncertainty and information at fixed R. Quantify uncertainty with simulation phases and spatial dependence in mind; thinning realizations, voxels and posterior draws are not independent cosmological volumes.

### 3 Diagnose the fixed scale posteriors

Freeze the retrained checkpoints and evaluator before examining the diagnostic panel. Compare truths and predictions with identical physical definitions and interpolation. Stratify by redshift, cap, support, boundary distance and luminosity where sample sizes permit.

For both classes, assess bias, proper scores, coverage and sharpness together. For P12-A, include joint eigenvalue calibration and class-probability reliability at registered tidal thresholds. For the field model, add regional density and tidal summaries, spatial covariance, and correlation with the actual truth. Check sampling/integration stability when it could explain a discrepancy. Conservation alone is insufficient.

Deliverable: a diagnosis distinguishing broad calibrated predictions, confident errors, numerical sampling problems and failure to exploit informative observations. Keep historical pilot results separate from the retrained model evaluation.

### 4 Measure information beyond the prior

Use a prior baseline matched to the target population, geometry and selection conditioning. For galaxy-position targets, a prior at random volume positions is not a matched baseline. Define exactly which observation information is removed from the baseline; do not call an unrelated reference reconstruction a prior.

Compare posterior and prior using held-out proper scores at the same R, together with coverage and sharpness. If useful, add a selection-aware classical reconstruction as a separate comparator. Wiener filtering illustrates how the balance depends on signal relative to noise rather than abundance alone. [Lahav et al.](https://arxiv.org/abs/astro-ph/9311059)

Deliverable: identify strata where the observations improve inference beyond prior knowledge. Avoid ranking different smoothing targets by raw error: stronger smoothing can reduce target variance and make errors smaller without increasing recovered information.

### 5 Consider scale or architecture changes only if justified

After the fixed-scale diagnosis, choose the smallest follow-up that addresses the identified limitation. Coarser-scale field diagnostics may be useful before any additional fits. For ideal Gaussian smoothing, an R = 7 field can be smoothed to a larger R using an additional width sqrt(R^2 - 7^2): approximately 9.75 for R = 12 and 18.73 for R = 20, in h^-1 Mpc.

Validate this operation against full-box truth with matched grid windows, boundary handling and sufficient context. Smooth density or tensor components before eigendecomposition; smoothing eigenvalues is not equivalent. Do not treat masked zero padding as physical density. Spatially varying smoothing does not generally commute with the tidal operator.

The current coarse, already-smoothed outputs cannot supply reliable R = 4 predictions through simple post-processing. That would require a separately justified target and resolution setup. P12-A per-galaxy posterior samples cannot substitute for coherent field draws in these coarser-scale tests.

Deliverable: a decision on whether any new scale, preprocessing or model experiment is warranted, with its own target definition and validation protocol.

## Pathways to consider after the diagnosis

| Pathway | When it is useful | Scientific cost or limitation |
| --- | --- | --- |
| Retain fixed R and publish uncertainty and information flags | Predictions are calibrated but some strata are weakly constrained | Some catalogue entries carry little information beyond the prior |
| Improve selection modelling or observation processing at fixed R | Informative observations are being used inefficiently or selection mismatch is evident | Requires controlled evidence that the change improves inference |
| Publish a bank of fixed physical scales | Coarser environments remain informative where finer environments do not | Requires scale-specific validation and consistent cross-scale dependence |
| Define an explicitly adaptive target R(z) | The science requires approximately uniform statistical resolution | Environment has a different physical definition at each redshift |
| Restrict the useful sample or introduce additional tracers | The available BGS observations impose an information limit | Reduced coverage, or additional tracer-bias and selection assumptions |

The first diagnostic should distinguish an information limit from model failure. Broad calibrated posteriors with little improvement over the prior suggest the former. Overconfident errors suggest a posterior or observation-model problem. Poor recovery despite substantial available information motivates an encoder or training change. These are interpretation criteria, not measured outcomes.

## Evaluation boundaries and immediate next work

Use the registered development/selection roles for exploratory work: the current field development panel uses ph012/ph013; the expanded P12-A contract uses ph006/ph012/ph013 with separate calibration, evaluation and selection subsets. Preserve the current reserved-data restrictions. Historically exposed phases are not fresh blind confirmation merely because a later contract excludes them from training.

The fixed-snapshot mock setup does not validate genuine cosmic evolution across the observed redshift interval. Nor does the existing finite-domain field result establish a coherent full-survey posterior. Keep those limits attached to later survey claims.

The immediate next work is the R = 7 retraining of both classes under registered budgets, followed by their existing validation. P12-A retains its encoder-to-export-to-posterior dependencies. No new smoothing target, adaptive kernel, multiscale training campaign or deferred diagnostic is required by this report before those fits.
