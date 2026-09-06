# Paper 2 theory draft: cosmic-web inference as a probabilistic inverse problem

> **Drafting note (remove before submission).** The non-uniqueness of the density-field reconstruction is not itself a new result: Bayesian large-scale-structure methods, most directly the BORG cosmic-web analysis of Leclercq, Jasche & Wandelt (2015), already propagated an ensemble of galaxy-compatible fields into voxel-wise tidal-eigenvalue and web-type posteriors. A 2026 preprint also presents a probabilistic DESI EDR environment catalogue based on ASTRA. The defensible novelty here is the calibrated, amortized inference of *continuous* local T-web eigenvalues for DESI BGS under a realistic observation operator, together with independent-phase validation and an explicit connection between deterministic shrinkage, posterior width and recoverable information. Any priority claim should remain "to our knowledge" pending a final systematic literature review.

## 2. Cosmic-web inference as a probabilistic inverse problem

### 2.1 From a matter field to an observed galaxy catalogue

The dark-matter environment of a galaxy is not observed directly. A redshift survey records a finite and selected point process of galaxies, whereas the T-web environment is a functional of an underlying continuous matter field. The inference therefore runs opposite to the physical data-generating process and is naturally an inverse problem. This distinction is central to the interpretation of both deterministic predictions and posterior uncertainties.

Let \(\delta_{\mathrm{ini}}(\boldsymbol{q})\) denote the initial matter-density contrast and let \(\mathcal{F}_{\Omega}\) denote gravitational evolution under cosmological parameters \(\Omega\). An evolved field at the target epoch is

\[
\delta_{\mathrm{m}}(\boldsymbol{x},z_\star)
=\mathcal{F}_{\Omega}[\delta_{\mathrm{ini}}](\boldsymbol{x},z_\star).
\]

Galaxies are a stochastic and biased realization of this matter field. Writing \(H\) for the galaxy--halo connection and associated nuisance parameters, a latent target catalogue \(G\) is drawn from \(p(G\mid\delta_{\mathrm{m}},H)\). The survey then maps \(G\) to an observed catalogue \(X_s\) through a stage-dependent observation operator \(\mathcal{O}_s\),

\[
X_s\sim p_s\!\left(X_s\mid G,\mathcal{R}_s\right),
\qquad
\delta_{\mathrm{ini}}\sim p(\delta_{\mathrm{ini}}\mid\Omega),
\qquad
G\sim p(G\mid\delta_{\mathrm{m}},H).
\tag{1}
\]

Here \(s\) can distinguish a targetable, fibre-assigned or final successful-redshift view. The response \(\mathcal{R}_s\) includes the angular footprint and imaging vetoes, radial selection, fibre assignment, redshift success, redshift-space distortions and measurement errors. A useful schematic decomposition is \(p_s(\boldsymbol{x})=M(\boldsymbol{x})C_s(\boldsymbol{x})\), where the binary field \(M\) encodes geometrical support and \(C_s\) encodes the conditional probability that an eligible target appears in view \(s\). This separation matters physically: a voxel with \(M=0\) contains no survey information and must not be interpreted as a matter void. DESI random catalogues and alternate fibre-assignment realizations provide empirical descriptions of parts of this response [Ross et al. 2025; Lasker et al. 2024].

For a fixed Gaussian smoothing scale \(R_\star\), the rescaled gravitational potential satisfies \(\nabla^2\Phi_{R_\star}=\delta_{\mathrm{m},R_\star}\). The T-web tidal tensor is

\[
T_{ij}(\boldsymbol{x};R_\star)
=\frac{\partial^2\Phi_{R_\star}}{\partial x_i\partial x_j},
\qquad
\widetilde{T}_{ij}(\boldsymbol{k};R_\star)
=\frac{k_i k_j}{k^2}\,W_{R_\star}(k)\,\widetilde{\delta}_{\mathrm{m}}(\boldsymbol{k}),
\tag{2}
\]

under the Fourier and Poisson conventions used here. We take \(R_\star=7\,h^{-1}\mathrm{Mpc}\) and order the eigenvalues as

\[
\boldsymbol{\Lambda}(\boldsymbol{x})
=\bigl(\lambda_1,\lambda_2,\lambda_3\bigr)(\boldsymbol{x}),
\qquad \lambda_1\leq\lambda_2\leq\lambda_3.
\tag{3}
\]

The inferential target at a galaxy position or voxel centre \(\boldsymbol{x}_\star\) is therefore the deterministic functional \(\boldsymbol{\Lambda}_\star=\mathcal{T}_{R_\star}[\delta_{\mathrm{m}}](\boldsymbol{x}_\star)\), but the matter field on which it depends is latent.

### 2.2 Why a galaxy catalogue does not determine a unique tidal field

Equation (1) is not uniquely invertible for a finite survey. The observed catalogue contains a finite set of biased tracers, while the latent matter field contains vastly more degrees of freedom. Poisson or non-Poisson sampling, stochastic galaxy formation, peculiar velocities, magnitude selection, fibre collisions, redshift failures, masks and unobserved long-wavelength modes all remove or mix information. Consequently, distinct fields \(\delta_{\mathrm{m}}^{(1)}\neq\delta_{\mathrm{m}}^{(2)}\), possibly accompanied by different latent catalogues or nuisance realizations, can both assign appreciable likelihood to the same observed \(X_s\):

\[
p(X_s\mid\delta_{\mathrm{m}}^{(1)},H,\mathcal{R}_s)>0,
\qquad
p(X_s\mid\delta_{\mathrm{m}}^{(2)},H,\mathcal{R}_s)>0.
\tag{4}
\]

Those fields need not have identical tidal tensors at \(\boldsymbol{x}_\star\). The catalogue can therefore constrain the local eigenvalues without fixing them exactly. This is a non-unique finite-data inverse problem, even if the assumed statistical model might be identifiable in an asymptotic sense. We use this more precise statement instead of asserting strict non-identifiability, which would require proving that different latent models induce exactly the same distribution for every possible catalogue.

Bayesian inference turns the set of compatible latent explanations into a probability measure. Under a fiducial galaxy--halo prescription \(H_{\mathrm{fid}}\), the latent-field posterior is

\[
p(\delta_{\mathrm{ini}},G\mid X_s,\mathcal{R}_s,H_{\mathrm{fid}},\Omega)
\propto
p_s(X_s\mid G,\mathcal{R}_s)\,
p(G\mid\mathcal{F}_{\Omega}[\delta_{\mathrm{ini}}],H_{\mathrm{fid}})\,
p(\delta_{\mathrm{ini}}\mid\Omega).
\tag{5}
\]

The prior and the forward model are therefore part of the meaning of compatibility. A field is not counted merely because it can be hand-adjusted to pass through the observed galaxies; it is weighted by its cosmological plausibility, its probability of producing the latent galaxy population and the probability that the survey would produce the observed catalogue from that population. This is the standard Bayesian resolution of an ill-posed inverse problem [Stuart 2010] and underlies field-level reconstructions such as BORG [Jasche & Wandelt 2013].

### 2.3 The local eigenvalue posterior is a pushforward of latent-field uncertainty

The posterior for the local tidal eigenvalues is obtained by pushing the field posterior through the deterministic T-web operator:

\[
\begin{split}
p(\boldsymbol{\Lambda}_\star\mid X_s,\mathcal{R}_s,H_{\mathrm{fid}},\Omega)
=\int &\delta_{\mathrm{D}}\!\left[
\boldsymbol{\Lambda}_\star-
\mathcal{T}_{R_\star}\!\left(\mathcal{F}_{\Omega}[\delta_{\mathrm{ini}}]\right)(\boldsymbol{x}_\star)
\right]\\
&\times p(\delta_{\mathrm{ini}},G\mid X_s,\mathcal{R}_s,H_{\mathrm{fid}},\Omega)
\,\mathrm{d}\delta_{\mathrm{ini}}\,\mathrm{d}G.
\end{split}
\tag{6}
\]

Equation (6) gives the precise content of the statement that several tidal environments may be compatible with one observed galaxy configuration. Each compatible latent field maps to one ordered eigenvalue triplet at \(\boldsymbol{x}_\star\); their posterior weights induce a distribution over those triplets. The result is not merely an error bar attached to an otherwise exact reconstruction. It is the inferential object required by the non-unique inverse.

Direct sampling of the full posterior in equation (5) is computationally expensive. We instead learn an amortized neural posterior estimator

\[
q_{\phi}\!\left(
\boldsymbol{\Lambda}_\star
\mid
\boldsymbol{s}_{\psi}(X_s;\boldsymbol{x}_\star),
\boldsymbol{r}_s(\boldsymbol{x}_\star),
H_{\mathrm{fid}}
\right)
\simeq
p\!\left(
\boldsymbol{\Lambda}_\star
\mid
\boldsymbol{s}_{\psi},\boldsymbol{r}_s,H_{\mathrm{fid}}
\right),
\tag{7}
\]

where \(\boldsymbol{s}_{\psi}\) is a leakage-safe learned summary of the observed galaxy configuration and \(\boldsymbol{r}_s\) contains deployable response covariates. Simulation-based inference is appropriate because the forward process can be sampled even when its likelihood is impractical to evaluate [Cranmer, Brehmer & Louppe 2020]. In the present baseline, the aligned three-dimensional deterministic prediction and response variables form the conditioning summary; higher-dimensional cross-fitted latents are admitted only if they improve held-out likelihood and calibration.

The posterior is parameterized in coordinates that enforce the physical ordering rather than learning three unconstrained eigenvalues. For unconstrained network outputs \((u_1,u_2,u_3)\), we use

\[
\lambda_1=u_1,
\qquad
\lambda_2=\lambda_1+\operatorname{softplus}(u_2),
\qquad
\lambda_3=\lambda_2+\operatorname{softplus}(u_3).
\]

The flow is trained in this ordered-increment space and transformed to physical eigenvalues only for posterior interpretation. This prevents the posterior estimator from allocating probability to ordering-violating triplets while retaining a continuous joint density.

There are two important qualifications to equation (7). First, the learned distribution is a posterior conditional on the supplied summary, not automatically on every detail of the full catalogue. By the data-processing inequality,

\[
I(\boldsymbol{\Lambda}_\star;\boldsymbol{s}_{\psi}\mid\boldsymbol{r}_s)
\leq
I(\boldsymbol{\Lambda}_\star;X_s\mid\boldsymbol{r}_s),
\tag{8}
\]

with equality only if the summary is sufficient for this target. Its width can therefore include information discarded by the representation as well as ambiguity already present in the observed catalogue. Secondly, per-galaxy or per-voxel posteriors are local marginal distributions. Independent samples from different rows need not assemble into a single globally coherent density or tidal field. The present method quantifies local environmental ambiguity at catalogue scale; it does not claim to sample complete matter-field realizations as a field-level posterior method would.

### 2.4 What a calibrated posterior width means

For eigenvalue component \(a\), define the conditional posterior variance

\[
\sigma^2_{a,\star}(X_s)
=\operatorname{Var}\!\left[
\lambda_{a,\star}\mid
\boldsymbol{s}_{\psi}(X_s),\boldsymbol{r}_s,H_{\mathrm{fid}}
\right].
\tag{9}
\]

If the simulator, observation model and neural posterior are adequate, this width measures how broadly the model-weighted ensemble of latent universes compatible with the supplied observations is distributed in \(\lambda_{a,\star}\). A narrow posterior means that the surviving galaxy configuration and response strongly constrain that local tidal component. A broad posterior means that materially different local eigenvalues remain compatible with those inputs. This is an uncertainty about which latent field generated a fixed observed catalogue, not a claim that the true eigenvalue of our Universe fluctuates physically between repeated measurements.

The phrase "range of possible eigenvalues" is useful intuition but is mathematically too strong. Continuous posteriors can have long tails or formal support over a large domain. We therefore report credible regions, posterior covariance and, where necessary, multimodality rather than a minimum-to-maximum range. The full joint distribution matters: a scalar width for each eigenvalue does not encode correlations among the ordered eigenvalues and can conceal distinct environmental modes.

The conditioning statement is equally important. The baseline posterior is conditional on \(\Omega\), the smoothing and threshold conventions, the simulation family and \(H_{\mathrm{fid}}\). It includes only sources of variation represented in the training joint distribution, such as independent density phases and whatever catalogue or observation stochasticity is explicitly varied by the forward model. It does not automatically include cosmological uncertainty, galaxy--halo prescriptions that were not varied, simulator discrepancy, unmodelled DESI systematics or neural-estimator uncertainty. If a nuisance variable \(H\) were explicitly marginalized, the law of total covariance would give

\[
\operatorname{Cov}(\boldsymbol{\Lambda}_\star\mid X_s)
=\mathbb{E}_{H\mid X_s}\!\left[
\operatorname{Cov}(\boldsymbol{\Lambda}_\star\mid X_s,H)
\right]
+\operatorname{Cov}_{H\mid X_s}\!\left[
\mathbb{E}(\boldsymbol{\Lambda}_\star\mid X_s,H)
\right].
\tag{10}
\]

The current conditional VAC estimates the within-\(H\) covariance at \(H=H_{\mathrm{fid}}\); it does not contain the between-HOD term. A held-out-HOD intervention can test whether the omitted shift is small relative to the reported width, but only explicit HOD variation and marginalization would promote that contribution into the posterior itself.

### 2.5 Deterministic shrinkage is the expected conditional-mean solution

This probabilistic formulation also explains the behaviour of the deterministic model. Suppressing the fixed response and model conditions for clarity, the population minimizer of mean-squared error for one eigenvalue is

\[
m_a(X_s)=\mathbb{E}[\lambda_a\mid X_s].
\tag{11}
\]

Writing \(\lambda_a=m_a(X_s)+\epsilon_a\), with \(\mathbb{E}[\epsilon_a\mid X_s]=0\), gives the variance decomposition

\[
\operatorname{Var}(\lambda_a)
=\operatorname{Var}[m_a(X_s)]
+\mathbb{E}\!\left[\operatorname{Var}(\lambda_a\mid X_s)\right].
\tag{12}
\]

Unless the observations determine \(\lambda_a\) exactly, the second term is positive and the conditional mean has a smaller population variance than the truth. Extreme true eigenvalues associated with ambiguous observations are consequently mapped towards the centre of the conditional distribution. This tail compression is not, by itself, evidence of bias or failed optimization; it is the Bayes-optimal behaviour of a squared-error point estimator under partial information.

The direction of the calibration regression follows directly. Because \(m_a\) is a function of \(X_s\),

\[
\mathbb{E}[\lambda_a\mid m_a]=m_a.
\tag{13}
\]

Thus an ideal conditional-mean predictor has unit slope when truth is regressed on prediction. Conversely, for population linear regressions with intercepts,

\[
\beta_{\,\lambda_a\mid m_a}=1,
\qquad
\beta_{\,m_a\mid\lambda_a}
=\frac{\operatorname{Var}(m_a)}{\operatorname{Var}(\lambda_a)},
\qquad
\frac{\sigma(m_a)}{\sigma(\lambda_a)}
=\sqrt{\frac{\operatorname{Var}(m_a)}{\operatorname{Var}(\lambda_a)}}.
\tag{14}
\]

For the Bayes predictor, the variance ratio in equation (14) is the population \(R^2\). A prediction-on-truth slope below unity and a compressed predicted variance are therefore expected even when equation (13) is satisfied. The appropriate response is not an affine stretching of a calibrated conditional mean, which would make point predictions more extreme without recovering the missing information. It is to infer the conditional distribution around that mean. The posterior restores the scientifically relevant tail probability by representing uncertainty, rather than by pretending that the point estimate knows which tail value occurred.

### 2.6 From eigenvalue posteriors to probabilistic environments

For the ordered convention in equation (3), the T-web class at threshold \(\lambda_{\mathrm{th}}\) is determined by

\[
N_{\mathrm{coll}}(\boldsymbol{\Lambda}_\star)
=\sum_{a=1}^{3}\mathbb{I}(\lambda_{a,\star}>\lambda_{\mathrm{th}}),
\tag{15}
\]

with \(N_{\mathrm{coll}}=0,1,2,3\) corresponding to void, wall, filament and knot, respectively. The posterior probability of class \(c\) is the pushforward

\[
P(C_\star=c\mid X_s)
=\int
\mathbb{I}\!\left[C(\boldsymbol{\Lambda})=c\right]
p(\boldsymbol{\Lambda}\mid X_s)\,
\mathrm{d}\boldsymbol{\Lambda}.
\tag{16}
\]

In particular, because \(\lambda_1\) is the smallest eigenvalue,

\[
P(C_\star=\mathrm{knot}\mid X_s)
=P(\lambda_{1,\star}>\lambda_{\mathrm{th}}\mid X_s).
\tag{17}
\]

Equations (16)--(17) show why thresholding a posterior mean is not equivalent to probabilistic classification: in general \(C(\mathbb{E}[\boldsymbol{\Lambda}\mid X_s])\) is neither the posterior modal class nor a summary of boundary uncertainty. A galaxy whose posterior straddles \(\lambda_{\mathrm{th}}\) should carry that ambiguity into environmental analyses. Class labels, abstention rules or science-sample cuts can then be chosen by posterior expected utility rather than by treating every catalogue row as equally certain [Leclercq, Jasche & Wandelt 2015b].

### 2.7 Calibration, contraction and the limits of interpretation

Posterior width has the interpretation above only if the posterior estimator is calibrated for the relevant data-generating distribution. If \(\mathcal{C}_{\alpha}(X_s)\) is an \(\alpha\)-credible region, marginal coverage requires

\[
\Pr\!\left[
\boldsymbol{\Lambda}_\star\in\mathcal{C}_{\alpha}(X_s)
\right]=\alpha
\tag{18}
\]

over repeated draws from the specified simulation-and-observation joint distribution. Stronger conditional calibration requires the same statement within scientifically relevant strata \(B\),

\[
\Pr\!\left[
\boldsymbol{\Lambda}_\star\in\mathcal{C}_{\alpha}(X_s)
\mid B
\right]=\alpha,
\tag{19}
\]

where \(B\) may index phase, redshift, tracer density, completeness, mask distance, holes versus footprint edges, or a held-out response recipe. Simulation-based calibration and TARP test complementary aspects of this requirement [Talts et al. 2018; Lemos et al. 2023]. Average coverage alone is insufficient because over-coverage in well-observed regions can conceal under-coverage in sparse or boundary regions, and a scalar rescaling can repair interval width while leaving the posterior shape or dependence structure incorrect.

Calibration is an ensemble property, not a guarantee that a particular 68 per cent interval contains the truth. For one DESI galaxy, a 68 per cent credible region means that the fitted conditional model assigns 0.68 posterior probability to that region. Across mock galaxies drawn under the validated joint distribution, intervals constructed in the same way should contain the simulated truth at the stated rate. Because DESI has no observed tidal-field truth, calibration on data is inherited from the adequacy of the mocks and observation model; it must be supported by closure tests, held-out phases and response recipes, posterior predictive diagnostics and explicit OOD flags.

Width should also be interpreted relative to the prior uncertainty. For a local response-conditioned prior \(p_0(\boldsymbol{\Lambda}_\star\mid\boldsymbol{r}_s,H_{\mathrm{fid}})\), the information gained from one observed configuration can be summarized by

\[
\mathcal{I}_\star(X_s)
=D_{\mathrm{KL}}\!\left[
p(\boldsymbol{\Lambda}_\star\mid X_s,\boldsymbol{r}_s,H_{\mathrm{fid}})
\,\Vert\,
p_0(\boldsymbol{\Lambda}_\star\mid\boldsymbol{r}_s,H_{\mathrm{fid}})
\right].
\tag{20}
\]

Its expectation over catalogues is the conditional mutual information between the local environment and the observations. Posterior-to-prior covariance contraction, entropy reduction and prior-dominated flags provide complementary, more interpretable catalogue diagnostics. A broad posterior can still represent meaningful contraction from a very broad prior, while a narrow but multimodal or miscalibrated posterior can be misleading. We therefore treat width, calibration and information gain as separate properties of the inference.

### 2.8 Relation to previous probabilistic cosmic-web inference

The premise of this section has clear precedents. Bayesian field-level methods already describe galaxy surveys through ensembles of compatible density fields. Most directly, Leclercq, Jasche & Wandelt (2015a) propagated BORG density-field samples inferred from SDSS galaxies into posterior tidal-eigenvalue fields and voxel-wise cosmic-web probabilities, while Leclercq, Jasche & Wandelt (2015b) used those probabilities for uncertainty-aware classification. More recently, ASTRA introduced stochastic, random-catalogue-assisted cosmic-web class probabilities [Forero-Romero et al. 2025], and Zapata-Zuluaga et al. (2026) applied ASTRA to the DESI Early Data Release. The existence of environmental ambiguity, probabilistic cosmic-web maps and uncertainty-aware class decisions is therefore established rather than newly discovered here.

Our contribution is a different operational realization of that principle. We target the joint distribution of continuous, ordered T-web eigenvalues at the positions of DESI BGS galaxies, amortize the inference so that it can be evaluated at catalogue scale, condition explicitly on the survey response, and validate the estimator across independent cosmological phases using spatially inductive and cross-fitted summaries. This connects deterministic conditional-mean calibration to posterior tail recovery and makes posterior contraction, response dependence and prior domination explicit catalogue quantities. Unlike BORG, the present local estimator does not generate globally coherent density-field realizations; unlike ASTRA, its probabilities are derived from a calibrated posterior over a specified dark-matter tidal-field target. The scientific advance is therefore not the observation that many latent fields can match one galaxy catalogue, but a scalable and testable posterior for what that ambiguity implies for each local DESI cosmic-web environment.

## References for this section

Forero-Romero, J. E., Hoffman, Y., Gottlöber, S., Klypin, A. & Yepes, G. 2009, *A dynamical classification of the cosmic web*, MNRAS, 396, 1815. [doi:10.1111/j.1365-2966.2009.14885.x](https://doi.org/10.1111/j.1365-2966.2009.14885.x)

Forero-Romero, J. E., Palomino, A., Gómez-Cortés, F. L. & Li, X.-D. 2025, *Cosmic Web Classification through Stochastic Topological Ranking*, RASTI, 4, rzaf032. [doi:10.1093/rasti/rzaf032](https://doi.org/10.1093/rasti/rzaf032)

Cranmer, K., Brehmer, J. & Louppe, G. 2020, *The frontier of simulation-based inference*, PNAS, 117, 30055. [doi:10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117)

Jasche, J. & Wandelt, B. D. 2013, *Bayesian physical reconstruction of initial conditions from large-scale structure surveys*, MNRAS, 432, 894. [doi:10.1093/mnras/stt449](https://doi.org/10.1093/mnras/stt449)

Kololgi, D., Naidoo, K., Saintonge, A. & Lahav, O. 2026, *Learning the cosmic web: graph-based classification of simulated galaxies by their dark matter environments*, RASTI, 5, rzag025. [doi:10.1093/rasti/rzag025](https://doi.org/10.1093/rasti/rzag025)

Lasker, J. et al. 2024, *Production of Alternate Realizations of DESI Fiber Assignment for Unbiased Clustering Measurement in Data and Simulations*. [arXiv:2404.03006](https://arxiv.org/abs/2404.03006)

Leclercq, F., Jasche, J. & Wandelt, B. D. 2015a, *Bayesian analysis of the dynamic cosmic web in the SDSS galaxy survey*, JCAP, 06, 015. [doi:10.1088/1475-7516/2015/06/015](https://doi.org/10.1088/1475-7516/2015/06/015)

Leclercq, F., Jasche, J. & Wandelt, B. D. 2015b, *Cosmic web-type classification using decision theory*, A&A, 576, L17. [doi:10.1051/0004-6361/201526006](https://doi.org/10.1051/0004-6361/201526006)

Lemos, P., Coogan, A., Hezaveh, Y. & Perreault-Levasseur, L. 2023, *Sampling-Based Accuracy Testing of Posterior Estimators for General Inference*, Proceedings of Machine Learning Research, 202. [arXiv:2302.03026](https://arxiv.org/abs/2302.03026)

Ross, A. J. et al. 2025, *The construction of large-scale structure catalogs for the Dark Energy Spectroscopic Instrument*, JCAP. [arXiv:2405.16593](https://arxiv.org/abs/2405.16593)

Stuart, A. M. 2010, *Inverse problems: A Bayesian perspective*, Acta Numerica, 19, 451. [doi:10.1017/S0962492910000061](https://doi.org/10.1017/S0962492910000061)

Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. 2018, *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*. [arXiv:1804.06788](https://arxiv.org/abs/1804.06788)

Zapata-Zuluaga, D. C., Guevara-Montoya, S., Torres-Gomez, V., Hernandez, J. & Forero-Romero, J. E. 2026, *The Cosmic Web in the DESI Early Data Release: A Probabilistic Environment Catalog*. [arXiv:2604.01456](https://arxiv.org/abs/2604.01456)
