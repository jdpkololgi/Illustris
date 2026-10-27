# Paper II: cosmic-web SBI for DESI — outline and theory

**Working document, 29 September 2026.** Reconciled against the science log through 28 September, the generalisable-VAC roadmap, the current P12-A deployment plan, and linked NERSC evidence summaries. This is a manuscript plan, not a declaration that the provisional VAC is qualified or publicly released. Bullet points describe what each section should say; Section 2 is a fuller mathematical draft. Proposed analyses are distinguished from completed work.

## Editorial recommendation and paper boundary

- **Keep principled training, posterior validation, DESI inference, and the bounded galaxy-property application together in Paper II.** They answer one question: what can a sparse, selected galaxy survey tell us about a galaxy's dark-matter tidal environment, and how should that information be reported and used?
- Suggested title: **“Cosmic-web inference for DESI: simulation-based tidal-environment posteriors and a provisional galaxy catalogue.”** Avoid “calibrated DESI catalogue”: simulation validation and real-survey coverage are different claims.
- Central argument: **a galaxy configuration constrains, but generally does not uniquely determine, the underlying tidal environment. Posterior width is part of the scientific result, conditional on the information supplied and the assumed simulator.** This is the right conceptual centre, but non-uniqueness itself is established prior art. Bayesian cosmic-web reconstruction already propagates galaxy-compatible matter-field uncertainty into environment probabilities ([Jasche & Wandelt 2013](https://doi.org/10.1093/mnras/stt449); [Leclercq, Jasche & Wandelt 2015](https://arxiv.org/abs/1502.02690)).
- Defensible contribution: the specific combination of continuous ordered tidal-eigenvalue inference, phase-separated and out-of-fold training, explicit observation/context contracts, posterior diagnostics, survey-scale amortized deployment, and a careful account of downstream environment-weighted summaries. Present this as a contribution, not an established priority claim.
- Relative to [Paper I, Kololgi et al. 2026](https://doi.org/10.1093/rasti/rzag025): move from simulated-galaxy graph classification to continuous probabilistic inference and observational deployment. The production anchor is now U-PATCH compression plus FMPE; do not describe it as simply the previous GNN with a posterior attached.
- Reserve **coherent spatial field inference** for a distinct later paper if ready. That boundary is scientific—joint field distributions versus local three-eigenvalue distributions—not an artificial separation of methodology from the DESI application. Current hierarchical CFM results are genuine research progress, but their joint-calibration failures do not support presenting them as the validated engine of this VAC.

## Evidence ledger: keep these results separate

| Result / product | Evidence available now | Supported claim | Unsupported extension |
| --- | --- | --- | --- |
| Original frozen P12-A | All ten registered ph001 blind-release gates passed on 4,897,905 rows; one opening | Independent-phase evidence for that frozen model and those criteria | Exact conditional calibration; confirmation of later refits |
| Deployed halo48 P12-A | Context correction, re-exported summaries, FMPE refit; exposed-ph006 audit; conditional checks; golden replay | Audited successor with modest mock overcoverage and implementation checks | Fresh independent blind confirmation; transferred ph001 certification |
| Loa provisional VAC | 5,436,413 unique IDs; 5,404,568 supported; 31,845 null/flagged | Completed survey-scale inference and reproducible provisional packaging | Verified DESI coverage or a qualified public release |
| CIGALE wedge | Current probabilities; controlled property comparisons on 94,028 common-support rows | Descriptive residual property–environment association | Causal tidal effects, unbiased latent-class means, superiority to the old model |
| Hierarchical field CFM | Regional joint diagnostics on phases 012–017; fine-scale failures | Evidence motivating stronger spatial-joint validation | Validated coherent-field VAC or full-field calibration pass |

## Abstract — points to include

- Motivate environment inference as a probabilistic inverse problem, not a competition to recover one exact label from sparse tracers.
- State the target: the joint distribution of three ordered tidal eigenvalues at mock-defined galaxy/host-linked locations, Gaussian smoothing \(R=7\,h^{-1}\mathrm{Mpc}\), fixed target snapshot \(z_\star=0.2\), and T-web probabilities at threshold \(\lambda_{\rm th}=0.2\).
- Name the actual method: frozen deterministic U-PATCH summaries, out-of-fold residual learning, response conditioning, and flow-matching posterior estimation.
- Distinguish original blind-test evidence from the deployed successor's exposed-phase audit; describe the latter as modestly conservative on tested mocks, not exactly calibrated.
- Report the completed provisional Loa catalogue, its supported-row count, and the known mock–data mismatch/unverified real-survey coverage.
- Include the bounded application: after mass/redshift standardization, knot-minus-void mean log sSFR differs by approximately \(-0.205\) dex and the low-sSFR fraction by \(+7.77\) percentage points. Call these posterior-weighted descriptive associations.
- Abstracts normally omit citations; supporting sources appear below. Do not present confidence fractions as accuracy.

## 1. Introduction — points to include

- Explain the physical inference problem: galaxies are biased, incomplete matter tracers; tidal environments belong to the underlying matter field. Distinguish T-web classes from geometric filament distance, group membership and local galaxy density ([Forero-Romero et al. 2009](https://doi.org/10.1111/j.1365-2966.2009.14885.x); [Libeskind et al. 2018](https://arxiv.org/abs/1705.03021)).
- Explain why hard labels conceal continuous tidal information and uncertainty near class boundaries. Joint eigenvalue samples permit derived thresholds without retraining, for an unchanged target definition.
- Position Paper I as a classifier proof of concept, not a like-for-like numerical baseline: simulation, smoothing scale, selection, architecture and evaluation have changed ([Kololgi et al. 2026](https://doi.org/10.1093/rasti/rzag025)).
- Acknowledge Bayesian field reconstruction explicitly. **Do not claim the first recognition that several density fields fit the same galaxies**, or the first probabilistic DESI environment catalogue ([Leclercq et al. 2015](https://arxiv.org/abs/1502.02690); [2026 ASTRA DESI EDR catalogue preprint](https://arxiv.org/abs/2604.01456)). ASTRA is priority context, not an identical continuous tidal-eigenvalue estimand.
- Introduce amortized SBI as a scalable route to model-conditional local posteriors ([Cranmer, Brehmer & Louppe 2020](https://arxiv.org/abs/1911.01429); [Dax et al. 2023](https://arxiv.org/abs/2305.17161)).
- State three contributions: a precise uncertainty interpretation; separation of training, selection, blind evaluation and deployment checks; and a provisional observational application with a bounded property analysis.
- State early that mock-posterior accuracy does not establish real-Universe validity. The observed high-redshift mismatch belongs in the scientific narrative, not only the last paragraph.

## 2. Theory: what a local cosmic-web posterior means

### 2.1 From a latent matter field to an observed point process

A galaxy redshift catalogue is not a direct measurement of the continuous matter or tidal field. It is a finite, biased and selected point process. We formulate environmental inference through the forward process connecting latent cosmological structure to the observations. Let \(U\) collect the latent initial conditions, evolved matter, halo and galaxy realizations, peculiar velocities, and positions needed to define the target. Let \(\mathcal M\) denote the adopted cosmology, galaxy–halo prescription, resolution and other fixed modelling choices; let \(r\) specify the survey response. Then

\[
U\sim p_{\mathcal M}(U),\qquad
X\sim p_{\mathcal M}(X\mid U,r),\qquad
p_{\mathcal M}(U\mid X,r)
\propto p_{\mathcal M}(X\mid U,r)p_{\mathcal M}(U).
\tag{1}
\]

Here \(X\) contains the accepted observed galaxies and their attributes. The response may include footprint, targeting, fibre assignment, redshift success, measurement errors and redshift-space mapping. Writing these processes in equation (1) does not assert that the present mocks model each faithfully. In particular, an unobserved region is not an observed empty region. DESI catalogue and fibre-assignment methods supply the observational framework ([Ross et al.](https://arxiv.org/abs/2405.16593); [Lasker et al.](https://arxiv.org/abs/2404.03006)).

At target epoch \(z_\star\), let the rescaled potential satisfy \(\nabla^2\Phi_R=\delta_R\), where \(\delta_R\) is the Gaussian-smoothed matter contrast. The tidal tensor and ordered eigenvalues are

\[
T_{ij,R}(\boldsymbol x)=\partial_i\partial_j\Phi_R(\boldsymbol x),
\qquad
\widetilde T_{ij,R}(\boldsymbol k)
=\frac{k_i k_j}{k^2}W_R(k)\widetilde\delta(\boldsymbol k),
\qquad
\boldsymbol\Lambda=(\lambda_1,\lambda_2,\lambda_3),
\quad \lambda_1\leq\lambda_2\leq\lambda_3.
\tag{2}
\]

The Fourier expression applies to nonzero modes, with the mean-density mode fixed by convention. This is the Hessian of the rescaled gravitational potential, not the Hessian of density. Its inverse-Laplacian dependence means that smoothing at \(R\) does not impose a strict information horizon of radius \(R\): exterior structure can influence the local tide ([Forero-Romero et al. 2009](https://doi.org/10.1111/j.1365-2966.2009.14885.x)).

For a fixed voxel, the target is \(F_R(U;\boldsymbol x_\star)\). For a galaxy, its real-space or host-linked evaluation location can depend on the latent realization, giving \(\boldsymbol\Lambda_g=F_R(U;\boldsymbol x_g(U))\). This latter form accommodates the current halo-linked mock labels; it must not silently be replaced by evaluation at the observed redshift-space position. Our production target is defined at \(R=7\,h^{-1}\mathrm{Mpc}\) and \(z_\star=0.2\), not by independently trained matter fields at every observed redshift.

### 2.2 Non-unique reconstruction and the posterior pushforward

Several latent realizations can give appreciable likelihood density to the same finite galaxy catalogue. Stochastic occupation, unobserved matter, velocities, selection and incomplete coverage leave multiple explanations even for a perfectly implemented inference procedure. Compatibility means weighting these explanations by both likelihood and cosmological prior, not simply finding a field that can pass through the galaxy positions.

The environmental posterior is the pushforward of latent uncertainty:

\[
p_{\mathcal M}(\boldsymbol\lambda_g\mid X,r)
=\int dU\,\delta_{\rm D}^{(3)}
\!\left[\boldsymbol\lambda_g-F_R(U;\boldsymbol x_g(U))\right]
p_{\mathcal M}(U\mid X,r).
\tag{3}
\]

Field non-uniqueness does not guarantee uncertainty in every functional: distinct fields may share particular local eigenvalues. Nonzero width arises when residual latent alternatives also change the target. This is a finite-information statement, not a proof of strict statistical non-identifiability of every forward model.

This interpretation has direct precedent in Bayesian large-scale-structure and cosmic-web inference ([Jasche & Wandelt 2013](https://doi.org/10.1093/mnras/stt449); [Leclercq et al. 2015](https://arxiv.org/abs/1502.02690)). Its consequence for local ML is that the objective is to represent the weighted range of plausible environments, rather than force every realization onto a point estimate. A particular galaxy has one realized environment under the chosen definition; the posterior represents uncertainty about it, not simultaneous occupation of several physical environments.

### 2.3 Full-data versus summary-conditioned uncertainty

The network does not use every aspect of \(X\). It receives \(S_g=(h_g(X,r),r_g)\), comprising a learned representation and selected response variables. Its intended target is

\[
q_\phi(\boldsymbol\lambda_g\mid S_g)
\simeq p_{\mathcal M}(\boldsymbol\lambda_g\mid S_g),
\tag{4}
\]

not automatically \(p_{\mathcal M}(\boldsymbol\lambda_g\mid X,r)\). Equality requires the summary to preserve all full-data information relevant to the target. A finite context, a three-component deterministic compressor, or omitted response variables can discard useful information even when posterior estimation is accurate.

Write \(D=(X,r)\), suppress the galaxy index, and assume finite second moments. Since \(S\) is a function of \(D\), total covariance gives

\[
\operatorname{Cov}_{\mathcal M}(\boldsymbol\Lambda\mid S)
=\mathbb E_{\mathcal M}\!\left[
\operatorname{Cov}_{\mathcal M}(\boldsymbol\Lambda\mid D)\mid S\right]
+\operatorname{Cov}_{\mathcal M}\!\left(
\mathbb E_{\mathcal M}[\boldsymbol\Lambda\mid D]\mid S\right).
\tag{5}
\]

The first term is residual full-data uncertainty averaged over catalogues represented by the same summary. The second is information lost by compression. Both contribute to a correct summary-conditioned posterior. Neither measures neural approximation error, which is a discrepancy between \(q_\phi\) and its intended distribution, not an independently additive physical variance.

A validated width is therefore **model-conditional uncertainty given the supplied summary and response**. Calling all of it “fundamental uncertainty of the observed galaxy configuration” is too strong. Better observations may reduce one component, more informative representations another. Equation (5) is a conceptual decomposition; these components have not been separately measured by the current experiments. Under nested information sets, additional information reduces covariance in the corresponding conditional-average sense; it need not narrow every individual posterior interval for every newly observed catalogue. A credible interval contains a specified posterior probability, not the entire set of physically possible eigenvalues; widths alone also hide asymmetry, multiple modes and dependence between eigenvalues.

### 2.4 Why deterministic predictions can compress the tails

For a scalar physical target \(Y\), an unrestricted predictor minimizing unweighted population squared error satisfies

\[
m(S)=\mathbb E[Y\mid S],\qquad
\mathbb E[(Y-f(S))^2]
=\mathbb E[\operatorname{Var}(Y\mid S)]
+\mathbb E[(m(S)-f(S))^2].
\tag{6}
\]

The conditional mean summarizes plausible values but cannot recover the realization-specific residual \(Y-m(S)\). Consequently,

\[
\operatorname{Var}(Y)=\operatorname{Var}(m(S))
+\mathbb E[\operatorname{Var}(Y\mid S)],\qquad
R^2_{\rm Bayes}
=1-\frac{\mathbb E[\operatorname{Var}(Y\mid S)]}{\operatorname{Var}(Y)}.
\tag{7}
\]

The second identity uses ordinary population \(R^2\) under one sampling measure. A narrower prediction distribution can thus be consistent with optimal squared-error prediction. Artificially stretching it to match the target histogram does not recover the missing realization-specific information.

For this project, the loss and target coordinates matter. Squared error on \(u=t(Y)\) yields \(\mathbb E[u\mid S]\), while generally \(t^{-1}(\mathbb E[u\mid S])\ne\mathbb E[Y\mid S]\). Target-dependent weights change the physical squared-error optimum to \(\mathbb E[wY\mid S]/\mathbb E[w\mid S]\). Equations (6)–(7) explain a mechanism, not a theorem that every decoded network is the physical conditional mean. Approximate truth-on-prediction calibration is also insufficient to establish Bayes optimality: even the constant population mean has a weak mean-calibration property. A metric plateau alone does not identify an irreducible information ceiling.

The principled response is to estimate and test the conditional distribution, including its tails, rather than assess the model solely by how well its point predictions recover extreme realizations.

### 2.5 Ordered eigenvalues and environmental probabilities

We represent the posterior in unconstrained increment coordinates \(\boldsymbol u\), with decoding

\[
\lambda_1=u_1,\qquad
\lambda_2=u_1+\operatorname{softplus}(u_2),\qquad
\lambda_3=\lambda_2+\operatorname{softplus}(u_3).
\tag{8}
\]

This covers strictly ordered triples; exact degeneracies lie on the boundary and require an explicit numerical convention if present in cached labels. Samples can be decoded directly, but physical-space density scores require the Jacobian, \(\det(\partial\boldsymbol\lambda/\partial\boldsymbol u)=\sigma(u_2)\sigma(u_3)\), and any standardization factors. Ordering is a support constraint, not a calibration guarantee. FMPE supplies the conditional generative model ([Dax et al. 2023](https://arxiv.org/abs/2305.17161)).

Define \(C(\boldsymbol\lambda)=\sum_{j=1}^3\mathbf1\{\lambda_j>\lambda_{\rm th}\}\), giving void, sheet, filament and knot for \(C=0,1,2,3\). Then

\[
P_{g,c}=\int\mathbf1\{C(\boldsymbol\lambda)=c\}
q_\phi(\boldsymbol\lambda\mid S_g)\,d^3\lambda
\simeq\frac1B\sum_{b=1}^B
\mathbf1\{C(\boldsymbol\lambda_g^{(b)})=c\}.
\tag{9}
\]

Draw eigenvalues jointly: combining marginal draws destroys dependence and can violate ordering. Thresholding the posterior mean does not reproduce equation (9). At production threshold 0.2, a knot requires \(\lambda_1>0.2\). These are smoothed tidal classes, not bound-cluster membership or proximity to a narrow geometric filament.

### 2.6 Calibration, information and model dependence

For an exact posterior, a region \(A_\alpha(S)\) assigned mass \(\alpha\) satisfies

\[
\Pr_{\mathcal M}\{\boldsymbol\Lambda\in A_\alpha(S)\mid S\}=\alpha.
\tag{10}
\]

Finite validation samples test only selected projections and strata. Aggregate coverage cannot establish equation (10) for each configuration; marginal calibration does not establish the joint distribution. Standard parameter-rank checks can even miss an estimator returning the prior. Calibration therefore needs conditional diagnostics and proper scores measuring useful predictive information ([Talts et al. 2018](https://arxiv.org/abs/1804.06788); [Modrák et al.](https://arxiv.org/abs/2211.02383); [Lemos et al. 2023](https://arxiv.org/abs/2302.03026)).

Millions of correlated galaxies are not millions of independent cosmological realizations. Phase-separated tests and spatial-block summaries address different aspects of validation. Passing predeclared tolerances is useful evidence, not proof of exact calibration. The current halo48 mock tests resolve modest overcoverage, so the exact-posterior interpretation is an inferential ideal rather than a measured identity for the deployed model.

Conditioning on \(\mathcal M\) is essential. Were nuisance parameters \(\eta\) explicitly marginalized, total covariance would include

\[
\operatorname{Cov}(\boldsymbol\Lambda\mid S)
=\mathbb E_{\eta\mid S}[\operatorname{Cov}(\boldsymbol\Lambda\mid S,\eta)]
+\operatorname{Cov}_{\eta\mid S}(\mathbb E[\boldsymbol\Lambda\mid S,\eta]).
\tag{11}
\]

Training at a fixed galaxy–halo prescription does not automatically include the second term for uncertainty in that prescription. Unmodelled selection mismatch does not automatically yield appropriately wider posteriors either. A perfectly accurate mock posterior can fail on observations when \(p_{\rm DESI}(\boldsymbol\Lambda\mid S)\ne p_{\mathcal M}(\boldsymbol\Lambda\mid S)\). Feature-envelope membership is a support check, not proof of conditional-distribution equality. The high-redshift Loa–mock discrepancy limits transfer claims independently of mock-posterior validation.

### 2.7 Local posteriors are not a coherent field posterior

The catalogue supplies a joint distribution over three eigenvalues per galaxy, not the joint distribution across galaxies:

\[
p(\boldsymbol\Lambda_1,\ldots,\boldsymbol\Lambda_N\mid X)
\ne\prod_g p(\boldsymbol\Lambda_g\mid S_g)
\quad\hbox{in general}.
\tag{12}
\]

Independent local draws need not correspond to any physically consistent matter field. They do not encode eigenvector orientations or all inter-galaxy correlations. Variance of a linear population statistic includes cross-object covariances unavailable from local marginals. Similarly, summing genuine full-data class probabilities gives an expected class count, but its uncertainty requires dependence information, and summary-conditioned probabilities require additional care. Coherent posterior fields are a distinct, stronger inferential object.

### 2.8 What probability-weighted property trends estimate

Let \(Y\) be a measured galaxy property and \(C\) the latent class. A natural summary is

\[
\widehat\mu_c^{\rm soft}
=\frac{\sum_g P_{g,c}Y_g}{\sum_g P_{g,c}}.
\tag{13}
\]

This describes a probability-weighted sample, but is not generally an unbiased true-class mean even if the probabilities are calibrated. Under the stronger ideal \(P_c(S)=\Pr(C=c\mid S)\), the population identity

\[
\mathbb E[Y\mathbf1\{C=c\}]
=\mathbb E[Y P_c(S)]
+\mathbb E[\operatorname{Cov}(Y,\mathbf1\{C=c\}\mid S)]
\tag{14}
\]

exhibits the missing term. Dividing by \(\Pr(C=c)\) relates the true class mean to the population soft-weighted mean. Equality requires, for example, conditional mean independence of property and class given \(S\), which is not guaranteed when the scientific question concerns residual property–environment relations. Conditioning class probabilities additionally on \(Y\), under a validated joint model, would be a different inference procedure. Finite-sample ratio bias and observational selection add further issues.

Mass/redshift standardization improves comparability without eliminating this limitation. For common strata \(b\) and a reference mixture \(\pi_b\) shared by all classes, define

\[
\widehat\mu_c^{\rm std}
=\sum_b\pi_b
\frac{\sum_{g\in b}P_{g,c}Y_g}{\sum_{g\in b}P_{g,c}},
\qquad \sum_b\pi_b=1.
\tag{15}
\]

For \(Y=\mathbf1\{\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})<-11\}\), this is a standardized low-sSFR fraction. It controls the specified mixture on retained support, not local density, halo mass, satellite status, property errors or all selection effects. The CIGALE result is therefore a residual association in a selected probability-weighted sample, not a causal tidal effect or fully deconvolved latent-class contrast.

Local widths and population trends answer related but different questions. Widths quantify remaining model-conditional uncertainty in the supplied information; population conclusions need their own estimands, selection assumptions and dependence model. This distinction is central to moving beyond interpreting every imperfect point prediction as an ML failure.

## 3. Simulations, observational data and targets — points to include

- Describe Abacus mock construction and the galaxy–halo prescription; distinguish fixed from randomized cosmology/HOD quantities. Independent phases probe realization generalization, not automatic model-family marginalization ([Maksimova et al. 2021](https://arxiv.org/abs/2110.11398)).
- Specify target epoch, smoothing, threshold, host linkage, units, mesh/interpolation, coordinate conventions and label provenance. Do not implicitly inherit Paper I's \(R=2\,h^{-1}\mathrm{Mpc}\) target.
- Identify the actual observational input: **DA2/loa-v1/LSScats/v2.1/PIP**. Quality rules include finite positive Z_not4clus, ZWARN=0, DELTACHI2>=25 and GALAXY. Mock redshift success is noiseless; document this asymmetry.
- Distinguish local accessibility from a verified publicly redistributable release. Include observational DESI inference, but check the precise Loa release/access/citation contract before claiming “public DESI data.” Do not relabel Loa as DR1/DR2 for convenience ([Ross et al.](https://arxiv.org/abs/2405.16593); [Lasker et al.](https://arxiv.org/abs/2404.03006)).
- Describe the CIGALE cache separately: CG15/CG5 mass/SFR and positions, exact TARGETID joins, no historical environment labels. Original upstream access is presently permission-denied; the surviving cache is pinned. Recoverable provenance is required for reproducibility ([Boquien et al. 2019](https://arxiv.org/abs/1811.03094)).
- **Table:** phases, catalogue versions, selections, target conventions, property support and access status.

## 4. Training and inference — points to include

- Present the current pipeline rather than the entire exploration: U-PATCH R0 epoch-20 frozen compressor; five leave-one-phase-out encoders; aligned three-dimensional physical predictions; FMPE conditioned on those predictions plus redshift, radial-density response, cap and random-supported boundary distance.
- Explain out-of-fold summaries: the posterior head should learn residual uncertainty from encoder outputs not generated by training on those same target realizations. Document alignment and final full-fit deployment. Phase/fold identities must not become shortcuts.
- Give the split contract: ph000 and ph002–005 training; ph006 selection/exposed checks; ph001 original one-shot blind test. “Fresh” is determined by exposure history, not phase name.
- Specify ordered-softplus target coordinates, normalization, FMPE objective, architecture, sampler, draw count and score Jacobians ([Dax et al. 2023](https://arxiv.org/abs/2305.17161); [Cranmer et al. 2020](https://arxiv.org/abs/1911.01429)).
- Treat context as a deployment contract. Final-weight halo24 failed 4/8 context-growth and 2/8 subdivision checks; halo48 passed the tested 24 dense/sparse/edge cores. Frozen encoders re-exported summaries and FMPE was refitted: this is a successor posterior, not the identical original checkpoint.
- State that the reported routes are untempered. Do not imply retrospective width correction.
- Move abandoned partitioned training and historical sweeps to an appendix unless they substantiate a central methodological claim.
- **Figure:** phase ownership → out-of-fold summaries → response-conditioned FMPE → ordered samples → class probabilities; clearly distinguish original/successor provenance.

## 5. Validation — points to include

- Separate implementation parity, point predictions, marginal/joint posterior checks, conditional strata, independent phases and observational transfer. No one coverage number replaces the others ([Talts et al. 2018](https://arxiv.org/abs/1804.06788); [Modrák et al.](https://arxiv.org/abs/2211.02383); [Lemos et al. 2023](https://arxiv.org/abs/2302.03026)).
- **Original frozen model:** ph001, 4,897,905 rows, ten registered gates passed. Eigenvalue-wise \(C_{68}=0.66661/0.67075/0.66480\), \(C_{90}=0.89115/0.89481/0.89001\); maximum conditional error 0.03429. Give registered tolerances. These gates are not the stricter historical SBC marker, which remained false.
- **Deployed halo48 model:** exposed ph006 natural audit, 50,000 rows, 512 draws. \(C_{68}=0.69469/0.69870/0.70975\), \(C_{90}=0.91012/0.91041/0.91765\). Largest C68 excess: 2.975 percentage points; spatial-block 95% intervals resolve modest overcoverage. TARP maximum error 0.0146 is below the existing 0.05 gate.
- Keep the **separate conditional acceptance audit** separate: \(C_{68}=0.69418/0.69624/0.70714\), \(C_{90}=0.90910/0.90922/0.91596\); global error 0.02714 versus 0.03 tolerance; non-sparse 0.03576 and sparse 0.01928 versus 0.06. Selection/weighting and quartile definitions differ from the natural audit.
- Matched knot Brier changes \(0.0342303\rightarrow0.0341958\): a small descriptive difference, not demonstrated significance or prior-baseline skill.
- **Do not transfer ph001 confirmation to halo48.** Current successor evidence is not a fresh full independent confirmation. Stronger language requires resolving eligibility before any new phase access, not retuning on the original blind phase.
- Golden replay: eight cap/shell cores, 14,149 galaxies and 64 response-identity checks. Describe exact coordinate/response parity and native label consistency as implementation evidence, not real-Universe validation.
- Do not require exact nominal coverage within every true-class-selected subset as an automatic Bayesian identity: selection on the target differs from conditioning on observed covariates. Such plots need an appropriate diagnostic reference.
- **Figure/table:** original and successor side by side, with exposure status, row weighting, draw count, intervals and thresholds. Proposed proper-score/prior-baseline tests are not completed results.

## 6. DESI inference and provisional VAC — points to include

- Report 5,436,413 unique targets, 5,404,568 supported, 31,845 null/flagged; define eigenvalue quantiles, class probabilities, quality/support metadata, model IDs and provenance.
- All rows carry provisional flag 32; 847,421 carry high-redshift warning flag 64; 125 lie outside the tested training-feature envelope. Flag counts need not be disjoint. Few envelope failures do not establish absence of conditional domain shift.
- Median maximum probability 0.744 and 40.01% with maximum at least 0.8 describe concentration, not accuracy/purity.
- Use the existing same-position atlas: fixed 20-Mpc equatorial slab, 48,798 galaxies; low-redshift close-up, 6,044. No class-dependent selection or smoothing. Visual coherence is descriptive.
- Show the transfer issue: high-shell \(0.45<z<0.55\) Loa/mock ratios are roughly 1.9 in the original comparison; boundary-eroded controls remain 1.823/1.855 SGC/NGC, and NTILE>=3 controls 1.664/1.740. Specify masks for each ratio; initial comparisons did not all have exact area parity.
- Summarize DR1/Loa shared-ID and interior-footprint controls as evidence against a simple release-change explanation, not a unique diagnosis of HOD or selection. Exact subpixel/tile-mask parity remains open.
- Mock repair was stopped on 28 September; model, mocks, intended selection and posteriors were preserved. Report this freeze transparently. This outline does not authorize restarting repair.
- Do not interpret class changes with observed redshift as measured physical evolution: selection/transfer and the fixed target epoch both limit that interpretation. Smaller low-redshift discrepancies do not certify low-redshift coverage.
- Cite [Ross et al.](https://arxiv.org/abs/2405.16593) and [Lasker et al.](https://arxiv.org/abs/2404.03006) for survey methods; use project counts/masks for actual mismatch claims.
- **Figures:** atlas plus cap/redshift mock–data comparisons with warning regions and explicit provisional captions.

## 7. CIGALE galaxy-property application — points to include

- Frame this as a bounded application, not validation of environmental truth. Use CIGALE properties, not the abandoned incomplete full-survey FastSpecFit exploration ([Boquien et al. 2019](https://arxiv.org/abs/1811.03094)).
- Account for samples: 111,503 cache rows, 103,814 exact-ID matches; three redshift-discrepant matches excluded and six outside the current-coordinate wedge. There are 94,415 supported positive-finite property rows at RA \(120^\circ\)–\(160^\circ\), Dec \(14.5^\circ\)–\(30.6^\circ\), \(0.20\le z<0.30\).
- Define the estimand by equation (15), not an unbiased physical class mean. Use common \(\Delta z=0.025\), \(\Delta\log_{10}M_\star=0.25\) strata, registered range \(8\leq\log_{10}(M_\star/M_\odot)<13\), effective support at least 20 per class, one pooled reference mixture. Define \(N_{\rm eff}=(\sum w)^2/\sum w^2\). Retained: 32 strata/94,028 rows; mass after redshift-only control retains 94,415, so raw and adjusted comparisons differ slightly in support too.
- Give standardized values in void/sheet/filament/knot order: mean log sSFR \(-10.6447/-10.7110/-10.7830/-10.8498\); low-sSFR fractions \(0.4348/0.4588/0.4863/0.5125\). Define the \(-11\) log-sSFR threshold as descriptive, not universal quenching.
- Lead with raw versus adjusted knot–void contrasts: mean log sSFR \(-0.4687\rightarrow-0.2051\) dex; low-sSFR fraction \(+16.401\rightarrow+7.774\) percentage points. Mass/redshift mixture explains part, not all, of the descriptive difference. After redshift control, the mass contrast remains \(+0.1134\) dex.
- Quote **16–84% block-bootstrap intervals**, not 95% intervals: \([-0.2171,-0.1943]\) dex and \([7.325,8.347]\) percentage points. There were 96 resamples of 22 occupied nside-8 sky blocks, holding the reference mixture fixed. These omit property and transfer systematics; they are not a complete uncertainty budget.
- Direction survives hard labels (\(-0.270\) dex/\(+10.23\) pp; 91,902 rows), confident labels (\(-0.344\) dex/\(+14.06\) pp; 37,260 rows) and CG5 (\(-0.186\) dex/\(+7.93\) pp; 94,027 rows). Different supports prevent isolation of soft-weight attenuation. No matched old-model comparison establishes improvement over earlier environments.
- Explain overlapping heatmaps: galaxies recur with different weights, logarithmic normalization emphasizes broad support, and modest mixture changes do not require visibly distinct loci.
- Compare cautiously with weak/controlled associations and detected residuals in [Alpaslan et al. 2015](https://arxiv.org/abs/1505.05518), [Eardley et al. 2015](https://arxiv.org/abs/1412.2141), [Hoosain et al. 2024](https://arxiv.org/abs/2401.09114), [O'Kane et al. 2024](https://arxiv.org/abs/2409.09028), and [Kraljic et al. 2018](https://arxiv.org/abs/1710.02676). Their estimands, selections and controls differ; there is no universal expected dex effect for our R7 tidal classes.
- Conclude “moderate residual association in a selected wedge,” not “tidal quenching independent of density/halo mass.” Similar property availability (90.8–91.1%) across inferred classes does not exclude selection bias.
- **Figure:** raw and standardized class summaries with block intervals; existing heatmaps as complementary views rather than the main inferential evidence.

## 8. Discussion — points to include

- Distinguish observation-limited uncertainty, compression loss, estimator error and simulator mismatch. Wider is not automatically more correct; sharper is not automatically more informative on shifted data.
- Summarize progress since Paper I: continuous targets, support constraints, phase separation, out-of-fold residual learning, context/response parity, survey deployment and explicit downstream estimands.
- Coverage is not discrimination, and modest property contrasts do not diagnose poor inference. Baseline-relative proper scores and controlled contrast-retention tests address utility more directly ([Modrák et al.](https://arxiv.org/abs/2211.02383)).
- **Optional, not executed:** use exposed mocks to test retention of known property–environment contrasts, with physical property truth where available or clearly labelled synthetic injection otherwise. Compare soft/hard/confident estimators on identical support. These would inform interpretation, not certify DESI transfer or justify retuning the frozen catalogue.
- Explain why this is not a BORG replacement: local amortized posteriors do not reconstruct an ensemble of globally coherent matter fields. Compare inferential objects, not unsupported speed/accuracy superiority ([Jasche & Wandelt 2013](https://doi.org/10.1093/mnras/stt449); [Leclercq et al. 2015](https://arxiv.org/abs/1502.02690)).
- Brief field-level outlook: mixed-resolution hierarchical CFM has trained results on phases 012–017, not merely a proposed architecture. Fine-region TARP ECP90 examples are 0.780/0.801 on ph012 and 0.767/0.782 on ph017, with reference-scale-sensitive magnitudes. Tests use 2-core/8-block/8-fine regional vectors and 32 draws, not full-field TARP. Correlated, stratified anchors produce SBC-style ranks, not an iid prior-predictive SBC ensemble. Latest evidence keeps 018–019 sealed.
- Mention true-coarse training versus sampled-coarse inference without claiming it is the proven sole cause. Global widening is not a demonstrated remedy. Keep this as motivation for a distinct spatial-joint programme, not validated results supporting this VAC.
- End with scope: field coherence, model-family uncertainty, selection transfer and true-class population inference require additional evidence or different models.

## 9. Conclusions — points to include

- The product is a distribution over continuous tidal environments, not a claim that sparse galaxies determine a unique matter field.
- Original and halo48 validation histories are distinct; preserve that distinction.
- Provisional Loa inference is complete; real-data coverage and high-redshift transfer remain unresolved.
- The controlled CIGALE wedge shows a moderate residual association, with limited sampling intervals and a descriptive soft-weighted estimand.
- The contribution is a reproducible, evidence-bounded inference framework and application—not perfect recovery or discovery of inverse-problem non-uniqueness.

## Availability, appendices and assembly

- Publish exact input IDs, model/configuration hashes, split/exposure ledger, coordinate/response contract, draw conventions, schema, flags and evaluation scripts subject to rights. FITS/draw products still require durable archival beyond Scratch; do not promise public access before it exists.
- Preserved catalogue SHA256: cf472cb4e8fb327620629f347115ad26c55a3f985320b293e6753e07f50ebfbc. Atlas/property work did not alter its posteriors.
- **Appendix A:** target operator, host linkage, units/coordinates, ordered transformation, degeneracy handling and score Jacobians.
- **Appendix B:** splits/exposure, original/successor provenance, conditional diagnostics, spatial blocks and finite-draw conventions.
- **Appendix C:** context tests, golden replay, selection crosswalk, mock–Loa/DR1–Loa controls and flags.
- **Appendix D:** property provenance, sample accounting, effective support, standardization, estimator sanity checks and bootstrap settings.
- Historical architecture sweeps belong in an optional appendix only when needed. Keep the field-CFM campaign out of this paper's main results.
- **Before submission:** resolve observational access/citation rights; decide how to present the successor's lack of fresh blind confirmation; retain or explicitly narrow provisional claims; secure durable products/property provenance; freeze figure/table manifests. Stronger release claims need stronger evidence, not relaxed wording.

## Internal evidence and literature audit trail

- Primary status: [SCIENCE_LOG.md](/Users/dakshesh/Developer/Illustris/SCIENCE_LOG.md), especially 24–28 September. New entries supersede dated readiness checklists where explicitly closed.
- Programme: [generalisable-VAC plan](/Users/dakshesh/Developer/Illustris/docs/plan_generalisable_graphweb_vac.md); deployment: [P12-A DESI plan](/Users/dakshesh/Developer/Illustris/docs/plan_desi_p12a_vac_20260924.md).
- Successor: [halo48 readiness](/Users/dakshesh/Developer/Illustris/docs/p12a_halo48_readiness_20260925.md) and [golden/conditional results](/Users/dakshesh/Developer/Illustris/docs/p12a_golden_conditional_results_20260925.md). Readiness is a dated checkpoint, not the final status of every subsequent step.
- Observational product: [provisional release](/Users/dakshesh/Developer/GraphWeb_DESI/docs/p12a_provisional_release_20260928.md); [CIGALE application](/Users/dakshesh/Developer/GraphWeb_DESI/docs/p12a_cigale_environment_20260928.md); [property literature assessment](/Users/dakshesh/Developer/GraphWeb_DESI/docs/p12a_property_literature_assessment_20260928.md).
- The [earlier theory draft](/Users/dakshesh/Developer/Illustris/docs/paper2_theory_cosmic_web_sbi.md) is retained. This document strengthens summary/full-data distinctions, the transformed-loss caveat and soft-weight interpretation; it supersedes broader deployment/novelty wording for this manuscript plan.
- Literature links were checked against primary publisher/arXiv records on 29 September 2026. This is a focused novelty/interpretation check, not an exhaustive priority review. No literal quotations are used. The mathematical identities are derivations for this outline; citations indicate established methodological context, not novelty for those identities.
