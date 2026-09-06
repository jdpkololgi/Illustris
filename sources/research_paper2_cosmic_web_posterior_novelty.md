# Literature check: uncertainty and non-uniqueness in cosmic-web inference

**Search date:** 2026-08-21  
**Purpose:** fact-check the proposed Paper 2 claim that a DESI-like galaxy catalogue is compatible with multiple underlying density and tidal fields, and identify the defensible novelty of a calibrated SBI catalogue.

## Finding

The physical and statistical premise is correct, but it is not itself novel. Inferring a continuous matter field from a finite, biased, noisy and incompletely observed galaxy point process is an ill-posed Bayesian inverse problem. The posterior represents a probability measure over latent fields compatible with the observations and the assumed forward model, rather than selecting a unique field.

There is also a direct cosmic-web precedent. Leclercq, Jasche & Wandelt (2015) propagated BORG posterior samples of SDSS-compatible density fields through a tidal classifier, obtaining posterior distributions for the tidal eigenvalues and voxel-wise probabilities for void, sheet, filament and cluster environments. Leclercq, Jasche & Wandelt (2015b) then used these probabilities for uncertainty-aware decisions, including abstention where the data constraints were weak. Consequently, Paper 2 should not claim the first recognition of latent-field ambiguity, the first probabilistic cosmic-web map, or the first posterior interpretation of cosmic-web type.

The DESI priority claim also needs care. Zapata-Zuluaga et al. (2026) currently claim the first public cosmic-web environment catalogue built on DESI data, using ASTRA to produce per-object class probabilities and entropies in DESI EDR. ASTRA's probabilities arise from repeated stochastic topological rankings of data and random points and do not constitute a posterior over the continuous matter density or tidal-eigenvalue field. This leaves a distinct methodological space for Paper 2, but rules out a broad "first probabilistic DESI cosmic-web catalogue" claim.

The strongest defensible contribution is the operationalization of the inverse-problem view in a scalable DESI BGS setting: an amortized, selection-aware joint posterior over continuous ordered T-web eigenvalues at galaxy locations or voxels; realistic fibre-assignment, redshift-success, mask and sampling response; independent-phase, spatially inductive and leakage-safe training; conditional calibration and posterior-contraction tests; and public-DESI inference. A priority statement such as "the first calibrated per-galaxy posterior over continuous T-web eigenvalues for DESI BGS" appears plausible from this targeted search, but should remain "to our knowledge" and be rechecked systematically at submission.

## Primary sources

1. Stuart, A. M. (2010), *Inverse problems: A Bayesian perspective*, Acta Numerica 19, 451. Bayesian inversion characterizes possible solutions and their relative probabilities and supplies uncertainty quantification. [DOI](https://doi.org/10.1017/S0962492910000061)
2. Jasche, J. & Wandelt, B. D. (2013), *Bayesian physical reconstruction of initial conditions from large-scale structure surveys*, MNRAS 432, 894. BORG formulates physical field reconstruction from galaxy surveys as posterior inference. [DOI](https://doi.org/10.1093/mnras/stt449)
3. Leclercq, F., Jasche, J. & Wandelt, B. D. (2015), *Bayesian analysis of the dynamic cosmic web in the SDSS galaxy survey*, JCAP 06, 015. Produces posterior tidal-eigenvalue and web-type maps from data-constrained density realizations. [DOI](https://doi.org/10.1088/1475-7516/2015/06/015) | [arXiv](https://arxiv.org/abs/1502.02690)
4. Leclercq, F., Jasche, J. & Wandelt, B. D. (2015), *Cosmic web-type classification using decision theory*, A&A 576, L17. Uses posterior class probabilities and permits no-decision outcomes where evidence is insufficient. [DOI](https://doi.org/10.1051/0004-6361/201526006) | [arXiv](https://arxiv.org/abs/1503.00730)
5. Cranmer, K., Brehmer, J. & Louppe, G. (2020), *The frontier of simulation-based inference*, PNAS 117, 30055. Reviews SBI as posterior inference for simulator-defined inverse problems. [DOI](https://doi.org/10.1073/pnas.1912789117)
6. Talts, S. et al. (2018), *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*. Defines SBC for checking posterior algorithms against a specified joint distribution. [arXiv](https://arxiv.org/abs/1804.06788)
7. Lemos, P. et al. (2023), *Sampling-Based Accuracy Testing of Posterior Estimators for General Inference*, ICML/PMLR 202. Introduces TARP coverage tests for sampleable posterior estimators. [PMLR](https://proceedings.mlr.press/v202/lemos23a.html) | [arXiv](https://arxiv.org/abs/2302.03026)
8. Ross, A. J. et al. (2025), *The construction of large-scale structure catalogs for the Dark Energy Spectroscopic Instrument*, JCAP. Defines DESI LSS catalogues and matched randoms as a sampling of the observational probability. [arXiv](https://arxiv.org/abs/2405.16593)
9. Lasker, J. et al. (2024), *Production of Alternate Realizations of DESI Fiber Assignment for Unbiased Clustering Measurement in Data and Simulations*. Describes fibre-assignment incompleteness and alternate-realization/PIP modelling. [arXiv](https://arxiv.org/abs/2404.03006)
10. Forero-Romero, J. E. et al. (2025), *Cosmic Web Classification through Stochastic Topological Ranking*, RASTI 4, rzaf032. Introduces ASTRA and probabilistic topological class assignments using galaxies and random catalogues. [DOI](https://doi.org/10.1093/rasti/rzaf032) | [arXiv](https://arxiv.org/abs/2404.01124)
11. Zapata-Zuluaga, D. C. et al. (2026), *The Cosmic Web in the DESI Early Data Release: A Probabilistic Environment Catalog*. Claims the first public DESI cosmic-web environment catalogue and reports ASTRA class probabilities and entropies. [arXiv](https://arxiv.org/abs/2604.01456)
12. Kololgi, D. et al. (2026), *Learning the cosmic web: graph-based classification of simulated galaxies by their dark matter environments*, RASTI 5, rzag025. Paper 1 establishes the graph-classification starting point but does not provide a selection-aware posterior over continuous eigenvalues. [DOI](https://doi.org/10.1093/rasti/rzag025)

## Interpretation guardrails for Paper 2

- Use "non-unique finite-data inverse" or "partial observability" unless strict statistical non-identifiability has been proved. Overlap of posterior support for multiple fields is not the same as two latent fields inducing identical likelihoods for all possible catalogues.
- Describe a credible region as a posterior-probability region, not as the literal range of every physically possible value.
- A posterior conditioned on a learned local summary is a posterior given that summary. It equals the posterior given the full catalogue only if the summary is sufficient for the target.
- Independent per-galaxy or per-voxel posterior samples are local marginals. They are not samples of globally coherent density fields unless the model constructs a joint field posterior.
- Mock calibration establishes calibration under the simulated joint distribution. Applying the posterior to DESI additionally requires simulator adequacy, response closure and OOD checks.
- The baseline project posterior is conditional on the fiducial galaxy--halo prescription. It does not include HOD uncertainty unless HODs are explicitly varied and marginalized.
