# Tokyo talk: uncertainty → Abacus mocks → SBI

Review date: 13 October 2026. Inspected the open, edited Keynote presentation, especially slides 8–13. No Keynote content was changed or saved. Scientific context comes from the local SCIENCE_LOG (latest entry 10 October), and the revised Paper II inference framework; this is not a fresh NERSC status check. Numerical values visible in A02 were read from the slide, not independently recomputed from its source draws.

This brief supersedes the *main-talk placement* of A03/A04 in the 10 October plan. It follows the subsequent decision to prioritize the physical definition, uncertainty and why SBI over training details.

## Recommendation

Use the existing slots, not a longer methods section:

| Current slide | Recommended role | Time |
|---|---|---:|
| 9: first uncertainty slide | Sparse tracers do not uniquely determine tidal environment | 35 s |
| 10: second uncertainty slide | Replace with “From uncertain environments to Bayesian inference” | 45 s |
| 11: A02 | Move after the SBI explanation; regenerate as “Uncertain eigenvalues, probabilistic environments” | 50 s |
| 12: A03 | Move training/split details to backup | — |
| 13: A04 | Replace with “Abacus mocks let us learn the inverse problem”; place before A02 | 55 s |

Thus the main sequence is current 9 → rewritten 10 → simplified A04 → redesigned A02 → calibration. Allow about three minutes total, including transitions. These are recommendations, not edits already applied to the deck.

## What the present slides do and do not establish

- **Slide 8:** classification-boundary ambiguity motivates a richer environmental representation. It does not by itself demonstrate a calibrated posterior or prove that errors are irreducible. A probabilistic classifier can also represent class uncertainty; continuous eigenvalues preserve information beyond a thresholded class.
- **Slide 9:** its schematic makes sparsity and survey support clear. However, the left-hand lines look like a filament skeleton. They are not the density or tidal field. For this audience, either label them explicitly as an illustrative sketch, or preferably replace them with a smooth matter-density illustration and a marked target location. Do not imply the project reconstructs filament spines.
- **Slide 10:** the current isolated expression q_phi(lambda_g | S_g(G_obs), H_fid) is a useful *destination*, but does not yet explain Bayes, the simulator, or the likelihood obstacle. Its presenter notes are still inherited text about galaxy properties and should be replaced.
- **A02 / slide 11:** the marginal intervals and probability bars are useful. There is no prominent takeaway title. “Lower/higher class concentration” and percentile labels require unnecessary explanation. Crucially, the right-hand example has wider intervals but greater class concentration: keep this distinction, rather than relabelling it “more precise.”
- **A03 / slide 12:** scientifically careful protocol documentation, but it interrupts the question-to-answer narrative. Preserve for questions about leakage, independent phases and model lineage.
- **A04 / slide 13:** correct high-level deployment story but too many boxes, implementation labels and version details. Also distinguish stored source fields from their normalized/clipped network transformations if this diagram is retained as backup. The main talk needs the simulator-to-inference story, not the tensor contract.

## Slide 10: From uncertain environments to Bayesian inference

### On-screen content

Subtitle: **Which tidal environments remain plausible given these galaxies?**

Use one large equation, with three short labels beneath its factors:

\[
p(\boldsymbol\lambda_g\mid D,\mathcal M)
\propto
\underbrace{p(D\mid\boldsymbol\lambda_g,\mathcal M)}_{\text{likelihood}}
\underbrace{p(\boldsymbol\lambda_g\mid\mathcal M)}_{\text{prior}}.
\]

Define beneath it, in plain language:

- D: observed galaxies and survey information, for a specified target galaxy.
- lambda_g: its three ordered, host-linked tidal eigenvalues.
- M: assumed cosmology, galaxy–halo model and observational recipe.

Bottom takeaway: **One realised environment; uncertainty about its value.**

Use a small visual of one marked galaxy and a distribution, not several apparently measured “true” fields. If multiple compatible fields are drawn, label them “conceptual alternatives—not reconstructed field samples.”

### Speaker notes (~45 seconds)

“For a given matter realization, the tidal tensor and its eigenvalues are well defined. But our catalogue is a sparse and selected view of that realization. Different underlying structures can remain plausible, and they can imply different tidal environments at this galaxy. We therefore ask for a posterior distribution: which environments are supported by the observations, weighted by our physical and observational model? The galaxy does not physically occupy several environments; this distribution expresses our uncertainty about its environment.”

Transition: **“The difficulty is evaluating this likelihood. The advantage is that we can simulate paired examples.”**

### Mathematical qualification (notes/backup, not another main-slide equation)

Let U contain the latent matter, halo and galaxy realization, including the host association; r specifies the survey response, and G is the observed catalogue. With M fixed,

\[
U\sim p_{\mathcal M}(U),\quad
G\sim p_{\mathcal M}(G\mid U,r),\quad
\boldsymbol\Lambda_g=F_R(U;\boldsymbol x_{h(g)}).
\]

The local posterior is the pushforward

\[
p_{\mathcal M}(\boldsymbol\lambda_g\mid G,r)
=\int\delta_{\rm D}^{(3)}\!\left[\boldsymbol\lambda_g-F_R(U;\boldsymbol x_{h(g)})\right]
p_{\mathcal M}(U\mid G,r)\,dU.
\]

This is the physical interpretation, not a claim that P12-A explicitly reconstructs or samples U. Non-unique fields cause target uncertainty only insofar as the remaining alternatives differ in the target eigenvalues. The prior is induced by the simulated, selected galaxy population; it is not an arbitrary flat prior over three independently chosen eigenvalues. Selection/eligibility and the target query are implicit in the compact main-slide notation.

Reference: Leclercq, Jasche & Wandelt (2015), https://arxiv.org/abs/1502.02690 . Bayesian web uncertainty is established prior work; the talk should claim the specific inference method/application, not discovery of non-uniqueness itself.

## Simplified A04: Abacus mocks let us learn the inverse problem

### On-screen visual

Draw a *fork*, not a chain that sends eigenvalues into a galaxy simulator:

```
                     Abacus matter + halos
                      /                 \
             tidal operator       galaxy population
                   |              + survey observation
          host-linked eigenvalues          |
                   |                 mock catalogue
                   |                       |
                   +------ paired with ----+
                                |
                     learn conditional posterior
```

Condense to three visual groups: **simulation**, **paired truth + mock observations**, **posterior inference**. Use no more than six short labels. On a second build, show the observed-catalogue input entering the already trained inference stage; never draw a DESI-to-known-truth arrow.

One equation below:

\[
q_\phi(\boldsymbol\lambda_g\mid S_g;\mathcal M)
\approx p_{\mathcal M}(\boldsymbol\lambda_g\mid S_g),
\qquad S_g=h(D)_g.
\]

Labels: **S_g: compressed galaxy context + survey descriptors**; **q_phi: learned conditional distribution**.

Bottom takeaway: **Simulate the joint distribution; learn the conditional distribution.**

### Speaker notes (~55 seconds)

“Abacus gives us matter and halo realizations. From these we obtain the tidal-eigenvalue labels and, through the galaxy and survey recipe, mock observations. We can generate these pairs, but evaluating the likelihood of a whole observed galaxy configuration given only three local eigenvalues requires accounting for a huge range of unobserved structure. SBI learns the conditional distribution from simulated pairs without explicitly evaluating that likelihood. Here we learn it conditional on a compressed representation of the galaxy context and survey information. Once trained, the same estimator can be applied to many galaxies. Our implementation uses a neural compressor and flow-matching posterior estimation; the physical output is a distribution of three eigenvalues.”

Say briefly: **“These are model-conditional posteriors, which we then test on mocks.”**

### Why SBI, rather than merely why uncertainty?

- Non-uniqueness motivates a probability distribution, not uniquely SBI.
- The forward model provides samples of truth and observations, while the required marginal likelihood is not tractably evaluated by this pipeline.
- Neural posterior estimation amortizes the inference over many galaxies and can represent correlated, non-Gaussian eigenvalue distributions.
- This is not “ordinary Bayes fails,” “the likelihood does not exist,” or “SBI is assumption-free.” Explicit-likelihood Bayesian reconstruction remains a legitimate alternative.
- AbacusSummit as a suite spans cosmologies; the current training recipe does not therefore marginalize over that suite. Repeated phases at fixed cosmology/HOD sample structure variation, not cosmology/HOD uncertainty.
- Conditioning on estimated survey descriptors is not marginalizing their errors. Nor is S_g demonstrated sufficient for the full catalogue.

References: Maksimova et al. (2021), https://arxiv.org/abs/2110.11398 ; Cranmer, Brehmer & Louppe (2020), https://arxiv.org/abs/1911.01429 . The AbacusHOD framework reference is Yuan et al. (2022), https://arxiv.org/abs/2110.11412 ; cite the actual BGS mock recipe additionally when its provenance is bound, rather than treating the framework paper as verification of every survey effect.

## Redesigned A02: Uncertain eigenvalues, probabilistic environments

### Recommended main visual

Retain two genuine mock examples from the same current model, but give them physical captions:

1. **Near a class boundary:** the existing example has about 41% void / 59% wall.
2. **Predominantly filament:** the existing example has about 87% filament / 10% cluster / 3% wall.

These are rounded values displayed in the deck, not new measurements. Preserve exact source IDs and fractions in the regenerated manifest. Do not imply either example was selected independently of the posterior being illustrated.

Show large marginal PDFs or horizontal half-violins for lambda_1, lambda_2, lambda_3, using the saved joint draws. Include the threshold at 0.2, dimensionless axes, and a directly labelled class-probability bar. Retain identical eigenvalue scales across the two examples. Add mock truth markers if available, clearly distinct from the posterior median. A truth marker in an interval for one object is not a calibration test.

If the full PDFs make the slide too busy, the current interval plot is an acceptable minimal alternative: enlarge it, explain 68%/95% once, add truth, and remove percentile-ranking text. Do not force a new corner plot into this short talk.

Takeaway: **Class ambiguity depends on both posterior spread and proximity to the threshold.**

Optional second build / notes equation:

\[
P(C_g=k\mid S_g,\mathcal M)
\approx\frac1M\sum_{m=1}^{M}
\mathbf 1\!\left[\sum_{a=1}^{3}\mathbf 1(\lambda_{g,a}^{(m)}>0.2)=k\right],
\quad\boldsymbol\lambda_g^{(m)}\sim q_\phi(\cdot\mid S_g;\mathcal M).
\]

The approximation includes finite-draw and learned-posterior error. Count threshold crossings within each *joint triplet*, not independent marginal draws or products of marginal probabilities. k=0,1,2,3 means void, sheet/wall, filament, knot/cluster; choose one consistent vocabulary throughout the deck.

Speaker notes (~50 seconds): “These intervals express what remains uncertain after conditioning on the information supplied to our model. We sample all three eigenvalues together and classify each draw. That gives probabilities for the environments, not just one hard label. Notice that this example can have wider continuous intervals yet a more concentrated class probability because most of its probability stays on the same side of the relevant thresholds. Narrow intervals and confident class labels are not the same thing. The next question is whether these distributions are reliable on mocks.”

## What posterior widths mean: short answer and backup

Main-talk sentence: **“The spread represents the model-weighted range of tidal environments left uncertain by the supplied galaxy information.”**

Qualification in notes: a fixed galaxy has one target under the chosen definition. Width is not time evolution, not the complete range of physically possible values, and not a measured universal uncertainty floor. It can reflect both observational ambiguity and information lost in compression. Model misspecification and neural approximation error need not appear as wider intervals.

For a fixed compressor S=h(D), the exact model-conditional identity is

\[
\operatorname{Cov}(\boldsymbol\Lambda\mid S)
=\mathbb E[\operatorname{Cov}(\boldsymbol\Lambda\mid D)\mid S]
+\operatorname{Cov}(\mathbb E[\boldsymbol\Lambda\mid D]\mid S).
\]

This is an interpretation, not an empirical decomposition already measured in the project. Calibration diagnostics test consequences on specified mock populations; they do not establish accuracy for every real DESI galaxy. Local posterior draws also do not constitute a coherent joint spatial field draw.

## Regeneration brief for NERSC-side work

No new training, simulation sampling, sealed-phase access, or posterior refit is needed for this presentation revision. Replot existing saved outputs; report a missing source rather than substitute illustrative numerical results.

| Asset | Action | Evidence requirement |
|---|---|---|
| A02-v2 | Main-slide posterior examples as above | Bind model version, phase, galaxy IDs, draws, truth lookup, selection rule; recompute intervals and class fractions from the same draws |
| A03 | Preserve as backup; no urgent regeneration | Confirm split/phase statements from manifests before final talk; do not transfer original blind qualification to halo48 |
| A04-v2 | Replace main version with forked forward-model/SBI schematic | Source-verified target and mock recipe; mark information arrows, not gravitational dynamics |
| A04-detail | Retain original detailed pipeline as backup | Distinguish stored fields from actual normalized/clipped input channels |
| A01-v2, optional | Improve slide 9's skeleton-like sketch | Smooth density illustration or actual matched simulation/mocks; if schematic, label it and do not claim conditional field samples |

### Style and acceptance criteria

- Match the deck: black background, white sans-serif text, existing mint/blue/magenta/gold class mapping. Keep the same meaning for colors across all slides.
- Deliver transparent or black-background plot assets, editable SVG/PDF, PNG at 1920×1080 or higher, and the generating source plus a provenance manifest. Do not bake every title into a full-slide raster: keep titles and equations editable in Keynote where practical.
- At 1920×1080: approximately 64–76 px title, 36–44 px body/axis labels, at least 28–32 px essential footnotes. Verify actual placement at 1280×720. Do not confuse these canvas pixels with plotting-library point sizes.
- Main figures need a takeaway title, labelled axes/units, clear threshold, defined intervals, and mock/model status. Long version strings and file paths belong in notes/manifests.
- Round visible class percentages sensibly; preserve exact values in the receipt. A displayed zero from 512 samples is not proof of zero physical probability.
- No invented posterior curves. If a density estimate is smoothed, record bandwidth and check that smoothing does not introduce artificial support/order violations; calculate probabilities from raw joint samples.
- No new animation is required. Two static builds are enough: paired simulation products, then learned inverse inference. If animation is used, flow time must not resemble cosmological evolution.

## Final narrative check

The audience should be able to answer four questions before seeing calibration:

1. What physical quantity is being inferred? Host-linked tidal eigenvalues at the stated smoothing scale/target epoch.
2. Why a distribution? Sparse selected tracers leave uncertainty in that quantity.
3. Why SBI? Simulations provide paired samples of a complicated joint law whose marginal likelihood is not tractably evaluated here.
4. What does the distribution condition on? A compressed galaxy-context summary and survey descriptors, under the registered mock model.

The short sequence should not require the audience to learn spatial folds, halo48, softplus transforms, FMPE bridge losses, or seven-component feature vectors.
