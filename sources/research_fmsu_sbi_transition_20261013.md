# Primary references for the Tokyo SBI transition

Checked with web search/open on 13 October 2026. The research-lookup skill's preferred parallel-cli backend was unavailable; used the available web tool instead. Notes are paraphrases, not quotations.

- Cranmer, Brehmer & Louppe (2020), *The frontier of simulation-based inference*. https://arxiv.org/abs/1911.01429 ; https://doi.org/10.1073/pnas.1912789117 . Supports the distinction between sampling a simulator and evaluating its likelihood, and neural approaches to inference. Does not validate this project's fitted posterior.
- Maksimova et al. (2021), *AbacusSummit: A Massive Set of High-Accuracy, High-Resolution N-Body Simulations*. https://arxiv.org/abs/2110.11398 ; https://doi.org/10.1093/mnras/stab2484 . Documents the N-body suite and matter/halo products. The suite spanning cosmologies is not evidence that P12-A marginalizes cosmology.
- Leclercq, Jasche & Wandelt (2015), *Bayesian analysis of the dynamic cosmic web in the SDSS galaxy survey*. https://arxiv.org/abs/1502.02690 ; https://doi.org/10.1088/1475-7516/2015/06/015 . Established precedent for propagating Bayesian structure uncertainty into tidal-web probabilities. Their field reconstruction is not identical to the local summary-conditioned P12-A method.
- Yuan et al. (2022), *AbacusHOD: a highly efficient extended multitracer HOD framework and its application to BOSS and eBOSS data*. https://arxiv.org/abs/2110.11412 ; https://doi.org/10.1093/mnras/stab3355 . General galaxy-population framework reference; does not by itself certify the precise BGS mock/observational recipe used here.

Local implementation interpretation was checked against docs/paper2_inference_framework_revised_20261010.md and SCIENCE_LOG.md (latest local entry 10 October), rather than inferred from these references.
