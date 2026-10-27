# Paper II: inference framework and statistical methodology

Revised 10 October 2026 to follow `paper2_abstract_introduction_revised_20261009.md`. This supersedes the theoretical exposition in the older drafts for current writing purposes, without deleting those drafts. Manuscript prose is below; editorial recommendations, source checks and implementation notes are kept separate. Equations are Markdown/LaTeX for transfer to Overleaf, not a standalone compilable document.

## Editorial recommendation

Use **Section 2: Physical target and inference framework**, followed by **Section 3: Simulations, observations and implementation**. Unify the theory with the statistical methodology that gives it meaning—compression, cross-fitting, response conditioning and validation—but do not put every catalogue cut, architecture setting or execution detail into the theory section.

Suggested paper structure:

1. Introduction.
2. Physical target and inference framework: the text below.
3. Simulations, observations and implementation: target construction, selected populations, coordinate/response recipes, exact phase splits, networks and sampling.
4. Validation on simulations: original blinded model versus deployed successor, marginal/joint/conditional diagnostics and limits.
5. DESI catalogue and galaxy-property application: full selected redshift range, flags, cross-matching and explicitly defined analysis subsets.
6. Discussion and conclusions.

This follows the useful separation in Jeffrey's SBI papers rather than copying their estimator. [Jeffrey, Alsing & Lanusse (2021)](https://academic.oup.com/mnras/article/501/1/954/6006276) introduce inference and compression in Section 2, then lensing, summaries and data/forward modelling in Sections 3–5, with validation/results later. [Jeffrey et al., DES Y3](https://academic.oup.com/mnras/article/536/2/1303/7911840) likewise place SBI principles, compression, posterior validation and marginalisation in Section 2; simulations, survey modelling and implementation follow. They use neural likelihood estimation and vary nuisance parameters in the mock generation. P12-A instead uses direct posterior estimation for local eigenvalues under a substantially fixed model. Their marginalisation claims cannot simply be imported into this paper.

### Drafting outline

- Define the host-linked physical estimand and nonlocal tidal operator before introducing ML.
- Define the forward survey operator and distinguish fixed assumptions, observed response covariates and random latent variables.
- Derive the local posterior as a pushforward, then distinguish full-data and compressed-data posteriors.
- Give the actual P12-A conditioning vector and ordered target coordinates.
- Explain cross-fitting as protection against in-sample compression optimism, not a calibration theorem or model-uncertainty integral.
- Interpret posterior width using total covariance and separate that interpretation from finite-model validation.
- Derive class probabilities and delimit spatial/population uses.

---

## 2. Physical target and inference framework

### 2.1 The tidal environment associated with a selected galaxy

Our inference target is a property of the underlying matter distribution, not a direct geometric label of the observed galaxy catalogue. At a fixed target epoch \(z_\star\), let \(\delta(\boldsymbol x)\) denote the matter-density contrast and \(\delta_R\) its Gaussian-smoothed counterpart. With the rescaled potential convention \(\nabla^2\Phi_R=\delta_R\), the tidal tensor is

\[
T_{ab,R}(\boldsymbol x)=\partial_a\partial_b\Phi_R(\boldsymbol x),
\qquad
\widetilde T_{ab,R}(\boldsymbol k)
=\frac{k_a k_b}{k^2}W_R(k)\widetilde\delta(\boldsymbol k),
\quad k\ne0,
\tag{1}
\]

where \(W_R(k)=\exp(-k^2R^2/2)\) and the zero mode is fixed by the density-contrast convention. The ordered eigenvalues \(\boldsymbol\Lambda=(\lambda_1,\lambda_2,\lambda_3)\), with \(\lambda_1\leq\lambda_2\leq\lambda_3\), characterise the T-web environment [Forero-Romero et al. 2009]. This is the Hessian of the rescaled gravitational potential, not the Hessian of density. Although its trace is local in the smoothed density, \(\operatorname{tr}T_R=\delta_R\), its anisotropic components depend on surrounding matter through the inverse Laplacian. The smoothing scale is therefore not a strict radius beyond which structure becomes irrelevant.

For P12-A, the target convention is \(R=7\,h^{-1}\mathrm{Mpc}\) and \(z_\star=0.2\). The mock label assigned to an observed galaxy is sampled at its linked host-halo location in the native simulation coordinates. This is distinct from evaluating a reconstructed field at the galaxy's observed redshift-space coordinate. Let \(U\) collect the latent matter, halo and galaxy realisation, including the host association. The target for object \(g\) is

\[
\boldsymbol\Lambda_g=F_R(U;\boldsymbol x_{h(g)}(U)).
\tag{2}
\]

The observed coordinate determines where the galaxy enters the inference inputs; equation (2) defines the physical label being inferred. Host linking, position sampling, smoothing and numerical normalisation are part of the estimand, not interchangeable implementation choices. In particular, the fixed target epoch does not turn observed-redshift trends across the VAC into measurements of tidal-field evolution.

The training population consists of selected, supported mock galaxies. Its distribution of environments is consequently galaxy-weighted and selection-dependent, not the volume-weighted distribution of random spatial voxels. Throughout this section, probabilities refer to this registered population; conditioning on sample eligibility is suppressed in the notation.

### 2.2 The observational operator and the assumed joint distribution

Let \(\mathcal M\) denote fixed modelling choices, including cosmology, the galaxy–halo prescription, simulation resolution and target definition. Let \(r\) denote a specified survey-response description. A schematic generative model is

\[
U\sim p_{\mathcal M}(U),\qquad
X\sim p_{\mathcal M}(X\mid U,r),\qquad
\boldsymbol\Lambda_g=F_R(U;\boldsymbol x_{h(g)}(U)).
\tag{3}
\]

The conditional law for \(X\) is the probabilistic observational operator. In a complete forward description it can include flux selection, redshift-space mapping, angular coverage, fibre assignment, redshift success and measurement errors. Naming these processes does not assert that all are accurately modelled in the current mock products. Nor does it imply independent Bernoulli selection of galaxies: effects such as fibre assignment can couple the inclusion of neighbouring targets. The implemented catalogue recipe, rather than a schematic factorisation, determines the training distribution.

Two distinctions are essential. First, the survey observation operator in equation (3) is not the tidal operator in equation (1): the former generates incomplete observations, while the latter defines the physical quantity to be inferred. Secondly, known or estimated response descriptors are not the same as uncertain nuisance parameters. Passing a selection estimate to a network conditions the prediction on that estimate. It does not automatically integrate over errors in the estimate.

If uncertain response parameters \(\eta\) were assigned a distribution constrained by auxiliary calibration data \(A\), the relevant likelihood would instead involve

\[
p(X\mid U,A,\mathcal M)
=\int p(X\mid U,\eta,\mathcal M)
       p(\eta\mid U,A,\mathcal M)\,d\eta.
\tag{4}
\]

When justified by the calibration design, the second factor may simplify to \(p(\eta\mid A,\mathcal M)\). By contrast, substituting one fitted response \(\widehat\eta\) uses \(p(X\mid U,\widehat\eta,\mathcal M)\). The two procedures are not equivalent. Likewise, repeating cosmological phases at a fixed galaxy–halo prescription samples structure variation, not uncertainty over alternative prescriptions. P12-A's present interpretation is conditional on its frozen mock and response recipes; a general nuisance-marginalised posterior is not claimed.

### 2.3 From compatible latent structure to a local posterior

A finite, selected tracer catalogue generally leaves multiple latent explanations with appreciable posterior probability. These explanations need not have the same tidal eigenvalues. Bayes' rule and the deterministic target map give

\[
p_{\mathcal M}(U\mid X,r)
\propto p_{\mathcal M}(X\mid U,r)p_{\mathcal M}(U),
\tag{5}
\]

\[
p_{\mathcal M}(\boldsymbol\lambda_g\mid X,r)
=\int \delta_{\rm D}^{(3)}
  [\boldsymbol\lambda_g-F_R(U;\boldsymbol x_{h(g)}(U))]
  p_{\mathcal M}(U\mid X,r)\,dU.
\tag{6}
\]

Equation (6) is a pushforward of latent-structure uncertainty. It explains why environmental inference calls for a distribution even when the target is deterministic in a fully specified simulation. Compatibility is weighted by the prior and forward model, not defined solely by whether a field can be adjusted to resemble the observations. Non-uniqueness of the field does not prove uncertainty in every functional of it: nonzero environmental width arises when the remaining alternatives also differ in the target of equation (2). This finite-information interpretation is consistent with established Bayesian cosmic-web reconstruction [Jasche & Wandelt 2013; Leclercq et al. 2015].

P12-A does not explicitly sample \(U\) or reconstruct the field posterior of equation (5). It learns a direct conditional distribution from paired mock observations and labels. Equation (6) defines the physical interpretation towards which this approximation is directed; it does not establish that independently generated local samples correspond to one shared latent universe.

### 2.4 What P12-A actually conditions on

The observed galaxy configuration is converted into a voxelised input for a frozen residual U-Net. In the production R0 path, the source fields are counts, apodised exposure and a log count-ratio field. The actual network tensor contains normalised counts, a clipped density proxy obtained from the count-ratio field, and apodised exposure. These selection-dependent quantities are rebuilt with the pinned transformations and selection curves. They encode aspects of observing conditions but are not a complete representation of the survey likelihood.

The compressor supplies a three-component physical prediction \(\boldsymbol b_g\). The conditional density estimator receives the seven-dimensional vector

\[
S_g=\left(
b_{g,1},b_{g,2},b_{g,3},
z_g,\log[\widetilde n_g/(1\,\mathrm{Mpc}^{-3})],
c_g,\log[1+d_g/(1\,\mathrm{Mpc})]
\right),
\tag{7}
\]

where \(c_g\) identifies the North or South Galactic Cap, \(\widetilde n_g\) is the fitted expected tracer density, and \(d_g\) is distance to the random-catalogue support boundary. The unit factors make explicit the numerical conventions used by the stored features. The full vector is subsequently standardised using training-set transformations. Simulation-phase identifiers, spatial-fold identifiers and artificial fold-boundary distances are excluded. Independently trained 32-dimensional encoder latents are also excluded from P12-A because their coordinates are not guaranteed to align across folds; the physical predictions provide a common coordinate system.

The estimator thus targets

\[
q_\phi(\boldsymbol\lambda_g\mid S_g)
\simeq p_{\mathcal M}(\boldsymbol\lambda_g\mid S_g),
\tag{8}
\]

not automatically the full-catalogue posterior in equation (6). A three-number compression, finite context and reduced response descriptors can discard relevant information. Even an exact approximation in equation (8) would not establish sufficiency of \(S_g\). The distinction also prevents the phrase ``conditioned on survey systematics'' from being read as a claim that every systematic uncertainty has been integrated out.

### 2.5 Ordered posterior estimation and out-of-fold compression

We model the target in unconstrained coordinates \(\boldsymbol u\), related to physical eigenvalues by

\[
\lambda_1=u_1,\qquad
\lambda_2=u_1+\operatorname{softplus}(u_2),\qquad
\lambda_3=\lambda_2+\operatorname{softplus}(u_3).
\tag{9}
\]

This parameterisation enforces strict ordering; exact label degeneracies require the recorded numerical gap-floor convention. Flow-matching posterior estimation learns a conditional transport from a reference distribution to the standardised target distribution [Dax et al. 2023]. Schematically, its velocity model is trained with a conditional flow-matching objective

\[
\mathcal L_{\rm FM}(\phi)
=\mathbb E\left[
\|v_\phi(\boldsymbol u_t,t;S_g)
      -v_{\rm path}(\boldsymbol u_t,t\mid\boldsymbol u_1)\|^2
\right],
\tag{10}
\]

where the expectation follows the implemented bridge and reference-noise law. Equation (10) states the method rather than specifying a different bridge from the pinned software implementation. The flow-time coordinate is a generative sampling coordinate, not cosmological time. Physical-space density scores require both the target-standardisation Jacobian and the ordered-map determinant \(\sigma(u_2)\sigma(u_3)\). Sampling and decoding alone do not require explicit density evaluation.

Crucially, the density estimator is not fitted only to predictions made by an encoder on its own training examples. For phase \(k\), let \(h_{\widehat\psi_{-k}}\) denote a compressor trained on the other registered training phases. Cross-fitted examples take the form

\[
\left(\boldsymbol\Lambda_{kg},S_{kg}^{\rm OOF}\right),
\qquad
S_{kg}^{\rm OOF}
=\left(h_{\widehat\psi_{-k}}(X_k)_g,r_{kg}^{\rm deployable}\right).
\tag{11}
\]

The omitted phase is also excluded from the corresponding fitted selection, input-normalisation and target-scaling transformations. This matters because a compressor can predict its training labels unusually well; a posterior fitted around those predictions may learn an unrealistically narrow distribution for deployment. Cross-fitting exposes the second stage to errors on structure that did not contribute to that encoder's parameter fitting. It addresses compression optimism without requiring a particular additive or Gaussian residual model: FMPE learns the conditional target distribution, not merely a fixed error bar added to \(\boldsymbol b_g\).

This procedure is not a calibration theorem. Finite training samples, inadequate conditioning information, density-estimator approximation and differences between leave-one-phase-out compressors and the final full-fit compressor can still produce erroneous posteriors. The distribution learned from pooled folds is approximately a mixture over fold-specific compression laws; applying it to a full-fit compressor requires separate validation of that handoff. Cross-fitting is also not Bayesian integration over network weights or observing-condition parameters. Fold-to-fold variability may affect the empirical training distribution, but it is not automatically a correctly weighted posterior over those uncertainties.

The implementation additionally distinguishes exclusion from optimisation from exclusion from all monitoring. The omitted phases were monitored, with retained summaries using terminal epoch 20. They should therefore be described as out-of-training-phase summaries rather than as independent blinded confirmation samples. The separate selection and confirmation phases serve a different inferential role. Context extent is part of this contract: when halo24 summaries were replaced by halo48 summaries, the frozen encoder weights were retained but the affected summaries were regenerated and the posterior refitted. Validation of the earlier composite pipeline does not automatically transfer to this successor.

### 2.6 What the posterior width means

For a fixed realised galaxy, the target in equation (2) has one value under the adopted definition. A posterior width describes uncertainty about that value, not physical fluctuation of the galaxy among several environments. If equation (8) is accurate, its covariance describes the spread of model-weighted latent explanations remaining after conditioning on the supplied summary.

This spread is not necessarily an irreducible property of the full observed catalogue. For a fixed trained compressor, let \(D\) denote the catalogue, response information and query definition, with \(S=h(D)\). Under the same joint model and finite second moments,

\[
\begin{split}
\operatorname{Cov}(\boldsymbol\Lambda\mid S)
={}&\mathbb E[\operatorname{Cov}(\boldsymbol\Lambda\mid D)\mid S]\\
&+\operatorname{Cov}(\mathbb E[\boldsymbol\Lambda\mid D]\mid S).
\end{split}
\tag{12}
\]

The first term is residual uncertainty given the complete specified observations, averaged over catalogues sharing the summary. The second is uncertainty arising from distinctions lost in compression. Both belong to a correct summary-conditioned posterior. The identity does not provide an empirical decomposition of the measured P12-A widths; those terms have not been separately estimated. Neural approximation error is a discrepancy between \(q_\phi\) and its intended conditional law, not an additional physical variance that the estimator is guaranteed to report.

The same reasoning distinguishes response-conditioned from response-marginalised uncertainty. If a nuisance parameter \(\eta\) were actually included in the inferential model, then

\[
\begin{split}
\operatorname{Cov}(\boldsymbol\Lambda\mid D)
={}&\mathbb E_{\eta\mid D}
 [\operatorname{Cov}(\boldsymbol\Lambda\mid D,\eta)]\\
&+\operatorname{Cov}_{\eta\mid D}
 [\mathbb E(\boldsymbol\Lambda\mid D,\eta)].
\end{split}
\tag{13}
\]

A calculation at fixed \(\widehat\eta\) is not this integral. In particular, uncertainty about the galaxy–halo prescription, fitted selection curve, simulator discrepancy or unmodelled redshift-success behaviour is not guaranteed to appear in P12-A's reported widths. A wrong observational model can yield a narrow, confidently wrong posterior; it need not reveal its inadequacy through increased width.

For intuition, an ideal scalar predictor minimising unweighted squared error in physical coordinates returns \(m(S)=\mathbb E[Y\mid S]\). Total variance gives

\[
\operatorname{Var}(Y)=\operatorname{Var}[m(S)]
                     +\mathbb E[\operatorname{Var}(Y\mid S)].
\tag{14}
\]

Compressed tails in a point-prediction histogram can therefore reflect missing information rather than solely failed optimisation. However, P12-A's deterministic precursor uses transformed targets; decoding a conditional mean in increment space does not generally give the conditional mean of physical eigenvalues. Equation (14) explains the motivation for distributional inference, not a proof that the actual network reaches the Bayes limit. Neither a performance plateau nor cross-fitting measures a fundamental information ceiling.

A credible interval should be described by its posterior probability content, not as the full range of physically possible eigenvalues. Marginal widths additionally conceal correlations, asymmetry and possible multimodality. Useful uncertainty requires accuracy as well as concentration: a broad distribution can be appropriate and informative relative to its response-conditioned prior, while a narrow distribution can be miscalibrated. Comparisons with a response-only baseline help distinguish information recovered from the galaxy configuration from information already supplied by redshift and survey selection.

### 2.7 Validation of the interpretation

For an exact conditional posterior, a region \(A_\alpha(S)\) assigned probability \(\alpha\) satisfies

\[
\Pr_{\mathcal M}\{\boldsymbol\Lambda\in A_\alpha(S)\mid S\}=\alpha.
\tag{15}
\]

Finite mock tests assess selected consequences of this statement, not its validity for every configuration. Marginal interval coverage, conditional coverage in response strata, multivariate diagnostics such as TARP, proper scores and class-probability reliability probe different failure modes [Talts et al. 2018; Lemos et al. 2023]. Population-averaged coverage may conceal compensating errors; individual eigenvalue calibration does not establish their joint dependence. A distribution returning the prior can pass some aggregate checks while extracting little observational information. Validation must therefore test both uncertainty accuracy and useful conditioning.

Spatial and phase dependence also matter. Millions of correlated galaxies are not millions of independent universes. Selection tests, spatial-block uncertainty estimates and independent-phase confirmation answer different questions and should not be combined into a single calibration percentage. P12-A's original blinded evaluation and the deployed halo48 successor's exposed-phase audits must be reported separately. The successor's tested modest overcoverage is compatible with passing specified tolerances but is not evidence of exact calibration.

Finally, numerical replay of the observation adapter establishes consistency of a transformation, not adequacy of the mock population for DESI. Even an exact mock conditional law can fail if \(p_{\rm DESI}(\boldsymbol\Lambda\mid S)\ne p_{\mathcal M}(\boldsymbol\Lambda\mid S)\). The known redshift-dependent mock–data mismatch therefore limits observational interpretation independently of cross-fitting and mock coverage. The provisional catalogue spans \(0.15\leq z<0.55\); its extent must not be confused with uniform scientific qualification over that range.

### 2.8 Environmental probabilities and the boundary of local inference

For the ordered convention, the T-web class is

\[
C(\boldsymbol\lambda)=\sum_{a=1}^{3}
\mathbf1\{\lambda_a>\lambda_{\rm th}\},
\qquad
P_{g,c}=\int\mathbf1\{C(\boldsymbol\lambda)=c\}
q_\phi(\boldsymbol\lambda\mid S_g)\,d^3\lambda.
\tag{16}
\]

Classes \(c=0,1,2,3\) correspond to voids, sheets, filaments and knots. The primary catalogue uses \(\lambda_{\rm th}=0.2\), applied after continuous posterior inference. Joint eigenvalue draws preserve the learned within-object dependence; independently combining marginal eigenvalue draws would not. Thresholding the posterior mean is not equivalent to equation (16).

These probabilities support explicitly defined posterior-weighted descriptions of galaxy properties. They do not establish that weighting by \(P_{g,c}\) recovers latent true-class property means or causal environmental effects. Those claims require additional assumptions about the joint distribution of galaxy properties, environments, selection and supplied summaries.

Nor does the catalogue provide \(p(\boldsymbol\Lambda_1,\ldots,\boldsymbol\Lambda_N\mid X)\). Independent draws from its rows are not a spatially coherent field posterior and cannot supply the missing covariance for regional web fractions or connected structures. Conditional-field models address this distinct target but require their own operator closure and spatial-joint validation. Their scientific motivation is a natural extension of local environmental inference, not an automatic consequence of successful per-galaxy calibration.

---

## Implementation and evidence checks for Section 3

| Item | Verified local source | What to report |
| --- | --- | --- |
| Actual encoder tensor | `workflows/abacus_tweb/p8_train_unet_patch.py`, `model_inputs` | Normalised counts, clipped `expm1(log_count_ratio)` proxy, apodised exposure; distinguish source-field order from network-channel order. |
| Posterior context | `workflows/sbi/p12_prepare_base_response_dataset.py`, `FEATURE_NAMES` and context construction | Exactly seven components in equation (7); training standardisation; no phase/fold/superblock input. |
| Cross-fit transformations | `workflows/sbi/p12_prepare_crossfit_contracts.py` | Omitted phase excluded from the fitted selection, normalisation and target scaler; full-fit transfer is still a validation question. |
| OOF export and epoch caveat | `workflows/sbi/p12_export_unet_summaries.py`; `docs/p12a_handoff_followup_results_20260924.md`; `workflows/abacus_tweb/p12a_handoff_followup.py` | Five omitted-phase compressors; monitored omissions; terminal epoch 20 retained. Do not call the omission entirely unseen during development. |
| Phase roles | `p12_prepare_base_response_dataset.py` | Training phases 000,002,003,004,005; development 006; original blind 001. ph006 spatial folds 0–1 and 2–4 separate diagnostic roles, not independent cosmological phases. |
| FMPE training | `workflows/sbi/p12_train_base_response_fmpe.py` | `sbi` FMPE, external standardisation, internal 10% validation split; this internal split is not the external phase test. Record exact pinned software bridge, sampler, solver and draw count. |
| Statistical prior versus software support | Same trainer | The physical prior comes from selected simulation pairs. The expanded `BoxUniform` supplied to the sampling interface is not evidence of uniform physical-eigenvalue training or HOD marginalisation; document any effective support truncation. |
| Target location | `docs/p12a_coordinate_sample_results_20260924.md`; `docs/e2e_cfm_p12a_comparison_scope_20261009.md` | Native host-halo label versus observed-coordinate query; not an interchangeable field-at-galaxy target. |
| Coordinate chart | `docs/p12a_population_response_replay_results_20260924.md` | Preserve P12's observer Mpc chart and selection calibration together. Do not transplant the newer field branch's native Mpc/h inputs. |
| Halo48 handoff | `docs/p12a_halo48_readiness_20260925.md`; `docs/p12a_golden_conditional_results_20260925.md` | Context changed, encoder weights frozen, OOF/full-fit summaries regenerated, FMPE refit; exposed-phase checks do not inherit original blind certification. |
| DESI response replay | `GraphWeb_DESI/workflows/p12a_vac/production_inference.py` | Same seven features rebuilt with pinned selection/response; source hashes and exact arithmetic checked. Replay tests fidelity, not model adequacy. |
| Observational success policy | `docs/p12a_golden_conditional_results_20260925.md` | Loa finite positive `Z_not4clus`, `ZWARN==0`, `DELTACHI2>=25`, `GALAXY`; mock noiseless success is not equivalent spectroscopic-success marginalisation. |
| Provisional scope | `GraphWeb_DESI/docs/p12a_provisional_release_20260928.md` | Full selected range retained; real-data coverage unverified; known high-z population mismatch is flagged, not marginalised. |

### Plain-language explanation of cross-fitting versus marginalisation

Imagine two otherwise comparable observed configurations, one seen under an accurately known completeness and the other under a completeness estimate that might be wrong. The first problem asks for environmental uncertainty **given that response**. The second additionally asks how the answer changes across plausible responses. P12-A's response features help with the first problem. To solve the second, one needs a justified distribution over response uncertainty and its propagation into the inference.

Cross-fitting addresses a different issue: whether the posterior is being taught using unrealistically accurate compressor outputs from the compressor's own training examples. It makes the second-stage task closer to prediction on new structure. It neither proves the resulting conditional density is right nor introduces response variations absent from the simulation law. Testing out-of-fold residuals is therefore useful evidence about generalisation, but it is not automatically a survey-systematics error budget.

## What the latest repository changes alter

Reviewed local Illustris HEAD `b5144c9` and GraphWeb_DESI HEAD `d959540`, with the science log through its 10 October entry. This was a read-only review of code, documents and committed receipts; no live NERSC jobs, large payloads or sealed phases were inspected. The existing 9 October introduction and 29 September outline were untracked and have been preserved.

The recent work is predominantly the conditional-field branch, not a new P12-A model. Full-cap density/operator work has advanced beyond the September independent-tile status. However, true-density closure shows a material tidal-operator discrepancy that explains only part of the joint deficit; changing to operator-matched truth does not rescue qualification. The 8 October class-fraction analysis additionally shows target closure and sparse/high-redshift sensitivity. These motivate explicit estimand/operator definitions in equations (1)–(3), not rewriting P12-A's validated target to match the field branch.

The early 9 October two-phase sensitivity rejection is superseded as the decision for the requested six-phase panel: the corrected panel requires missing full-cap tidal inputs. Subsequent generation is rebuilding the required evidence. On 10 October, one new four-draw generation task and corrected extraction passed technical checks; the log records bounded parallel continuation, not a completed Gate 2 or posterior qualification. The paper should not turn these technical completions into a field-calibration claim.

The 9 October comparison-scope note explicitly rules out a current like-for-like dominance claim: field Stage B evaluates at observed coordinates, whereas P12-A uses native host-linked targets. P12-A has stronger original blind-mock evidence, not established superiority for every quantity. The main Paper II framework should remain local-posterior inference; include these field lessons as limits and future directions unless a separately validated matched result becomes available.

### Small companion correction to the introduction

Replace “at an individual galaxy location” with “at the native host-halo location associated with each mock galaxy, inferred from its observed galaxy configuration.” Keep the observational VAC phrased as host-environment inference under this mock convention, not as direct measurement of a tidal tensor at an observed redshift-space coordinate. The next section should open with this distinction rather than leaving it to an appendix.

## References and manuscript key suggestions

- `forero_romero2009`: [A dynamical classification of the cosmic web](https://doi.org/10.1111/j.1365-2966.2009.14885.x).
- `jasche_wandelt2013`: [Bayesian physical reconstruction of initial conditions from large-scale structure surveys](https://doi.org/10.1093/mnras/stt449).
- `leclercq2015`: [Bayesian analysis of the dynamic cosmic web in the SDSS galaxy survey](https://arxiv.org/abs/1502.02690).
- `cranmer2020`: [The frontier of simulation-based inference](https://arxiv.org/abs/1911.01429).
- `jeffrey2021`: [Likelihood-free inference with neural compression of DES SV weak lensing map statistics](https://academic.oup.com/mnras/article/501/1/954/6006276).
- `jeffrey_desy3`: [DES Y3: likelihood-free, simulation-based wCDM inference with neural compression](https://academic.oup.com/mnras/article/536/2/1303/7911840), preprint [2403.02314](https://arxiv.org/abs/2403.02314); use the journal year from the final bibliography, not a guessed year from the arXiv identifier.
- `dax2023`: [Flow Matching for Scalable Simulation-Based Inference](https://arxiv.org/abs/2305.17161).
- `talts2018`: [Validating Bayesian Inference Algorithms with Simulation-Based Calibration](https://arxiv.org/abs/1804.06788).
- `lemos2023`: [Sampling-Based Accuracy Testing of Posterior Estimators for General Inference](https://arxiv.org/abs/2302.03026).

The Jeffrey papers' structures and distinction between learned likelihoods and P12-A direct posterior estimation were checked against the primary full texts during this revision. Other methodological references and physical definitions are retained from the earlier theory drafts. Citation keys above must be reconciled with the existing Overleaf bibliography.
