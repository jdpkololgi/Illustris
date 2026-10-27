# Tokyo: the next slides after your current slides 9 and 10

Prepared 13 October 2026 from the live Keynote slides and your presenter notes. This is copy-ready material, not an edit to the Keynote file. Proposed slide numbers below refer to the revised sequence. Local scientific status checked against SCIENCE_LOG.md, whose newest entry is 10 October; no live NERSC status inspection was performed.

## First: keep the progression you already have

Slide 9 now shows smoothed matter density and selected mock galaxies, rather than the earlier skeleton illustration. That is the right visual for your target. Its caption says these are true mock positions in a real-space slab: verbally call this an illustration using selected mock galaxies, not a direct picture of what DESI observes in redshift space. A single matched pair illustrates incomplete tracing; it is not itself a demonstration of multiple conditional field realizations.

Your conversational notes work. Only small edits are needed:

- Slide 9: “an observed distribution of galaxies **does not** uniquely determine the underlying field.”
- Slide 10: change “Which in our case is the DESI survey” to “In our case, that means our assumed cosmology, galaxy–halo model, and description of the DESI observations.”
- Slide 10: “Different plausible underlying structures **can imply** different tidal environments” is more precise than saying every different structure necessarily does.
- Keep your last line exactly as the handoff: **“The difficulty is evaluating this likelihood. The advantage is that we can simulate paired examples.”**

For slide 10 itself, keep the question and Bayes equation. Shorten the three definitions to:

> D: galaxy catalogue + survey information  
> lambda_g: the galaxy's three tidal eigenvalues  
> M: assumed physical + observational model

Explain host linkage verbally or in the target-definition slide. Enlarge the equation; avoid another diagram on this slide. Your current slide 11 repeats Bayes and adds q_phi without explaining the intervening steps. Replace that repeated equation slide with the following sequence.

## Recommended sequence and timing

| Proposed slide | Its one job | Time |
|---|---|---:|
| 11 — Simulations give us both sides of the problem | Explain what the paired examples actually are | 40–45 s |
| 12 — Learning the posterior from simulations | Explain SBI and the actual conditional distribution | 50–55 s |
| 13 — What does the posterior tell us? | Interpret width and turn joint draws into class probabilities | 45–50 s |
| 14 — Are these uncertainties reliable? | Introduce mock validation, then transition to DESI | 35–40 s |

About three minutes in total. These replace the duplicated Bayes and detailed training/pipeline sequence; they are not four additional methods slides. Move current A03 and detailed A04 to backup. Reuse the A02 data for proposed slide 13. If rehearsal runs long, combine 11 and 12 as two builds of one slide and use the short script at the end.

## Slide 11 — Simulations give us both sides of the problem

### Put this on the slide

**Title:** Simulations give us both sides of the problem

One large forked diagram:

```
                        Abacus matter + halos
                         /                 \
                Tidal calculation      Galaxies + survey recipe
                       |                        |
              Known eigenvalues            Mock observations
                         \                 /
                           Paired examples
```

**Bottom line:** Known environment + mock observation

Use two small, genuine matched thumbnails if they are available; otherwise use a clearly labelled schematic with no pseudo-data. Highlight the same target galaxy/host association on both branches where the coordinate mapping supports it. No equation is necessary on this slide. Cite Maksimova et al. (2021) in a small footer; add the actual mock-catalogue reference from its provenance.

### Speaker notes — copy into Keynote

So this is where the AbacusSummit simulations come in.

In a simulation, we know the underlying matter distribution. So we can calculate the tidal tensor and assign the eigenvalues at the host halo associated with each mock galaxy.

At the same time, the galaxy population and survey recipe give us a mock catalogue: the information we would have available for inference.

So we have both sides of the problem. We have the mock observation, and we have the environment we're trying to infer. And importantly, those two come from the same simulated realization.

The question now is how we use these pairs to learn the inverse relationship.

### Figure prompt — forward-model diagram

```text
Create an editable scientific schematic for a 16:9 Keynote talk about local tidal-environment inference from DESI-like BGS galaxies. This is a conceptual information-flow diagram, not a new numerical result. Use a black background, IBM Plex Sans or the deck's matching sans-serif, white text, light-grey arrows, and restrained blue/mint accents. Leave a clear title band at the top and a one-line takeaway band at the bottom. Main labels must be readable in a 1280x720 slide preview.

Place one parent group, “Abacus matter + halos”, at upper centre. Branch to two equally prominent groups: left “Tidal calculation” leading to “Known eigenvalues”; right “Galaxies + survey recipe” leading to “Mock observations”. Join the two products with a bracket labelled “Paired examples”. The eigenvalues are host-linked tidal labels, not eigenvalues inferred from the mock counts. Do not draw an arrow from three eigenvalues into a galaxy simulator; both products share the latent realization.

Prefer verified matched simulation/mocks thumbnails if supplied. If none are supplied, use simple geometric icons labelled “schematic”; do not fabricate numerical density maps, galaxy positions, or eigenvalues. Do not imply a real-space illustration is an observed redshift-space catalogue. Do not list fibre assignment, redshift failures or other effects unless their implementation is verified in this mock recipe. No network layers, loss functions, fold IDs, halo48 jargon, or parameter counts.

Deliver editable SVG and vector PDF plus PNG, with separate thumbnail assets if used. Keep the slide title and takeaway as separate Keynote text, not baked into the image. Include a source manifest for any actual data thumbnails. Arrows show information flow, not time evolution.
```

## Slide 12 — Learning the posterior from simulations

### Put this on the slide

**Title:** Learning the posterior from simulations

**Small subtitle:** Simulation-based inference (SBI)

Main visual: **Galaxy context + survey information → learned summary S_g → joint eigenvalue posterior**.

Below the visual, one large equation:

\[
q_\phi(\boldsymbol\lambda_g\mid S_g;\mathcal M)
\approx p_{\mathcal M}(\boldsymbol\lambda_g\mid S_g).
\]

**Bottom line:** Learn the distribution, not just a single estimate.

Small definition: “S_g: compressed galaxy context + survey descriptors.” Use the same M notation as slide 10 instead of introducing H_fid as a second name. Do not repeat Bayes' theorem. Do not add the flow-matching loss or the seven-feature list. The approximation targets the summary-conditioned posterior, not a proven full-catalogue posterior.

Reference footer: Cranmer, Brehmer & Louppe (2020).

### Speaker notes — copy into Keynote

This is the idea behind simulation-based inference.

We can produce examples from our model, even though evaluating the likelihood directly is difficult. Three local eigenvalues don't specify all the surrounding matter, the galaxy population, or what makes it into the catalogue. Those possibilities would all have to be accounted for.

Instead, we use the simulated pairs to learn a conditional distribution.

In our case, we first compress the galaxy context, and combine that with information about the survey. We then use flow-matching posterior estimation to predict a joint distribution of the three eigenvalues, rather than one best estimate.

Once trained, we can apply the same estimator to many galaxies. But the interpretation is still conditional on our simulation model and on the information retained in that summary.

### Figure prompt — inference diagram, simplified A04

```text
Create an editable, minimal scientific information-flow figure matching the previous black-background Abacus paired-examples schematic. The figure explains deployment of a learned conditional posterior, not gravitational evolution and not density-field reconstruction.

Use three left-to-right groups: (1) “Galaxy context + survey information”; (2) “Summary S_g”; (3) “Joint eigenvalue posterior”. A small label under group 2 can read “Neural compression”; a small label under group 3 can read “Conditional flow”. The current implementation uses a residual U-PATCH compressor and FMPE; do not insert a GNN or message-passing architecture. The summary includes compressed galaxy context and survey descriptors. Keep implementation acronyms out of the main boxes.

For the posterior icon, use a simple labelled “schematic” distribution icon, or an explicitly supplied source-verified posterior plot. If showing multiple eigenvalue marginals, label them as marginals of one joint posterior, not independent distributions. Do not use an invented numerical curve as a measured result. Do not show independent local draws forming a coherent cosmic-web map.

Leave clear space below for an editable Keynote equation and takeaway. Do not render the equation into an AI-generated image. Match fonts and line widths to the preceding figure. Use a small muted annotation “learned from simulated pairs” rather than a second training diagram. Deliver SVG, PDF and PNG; include a full-slide placement preview at 1280x720.
```

## Slide 13 — What does the posterior tell us?

### Put this on the slide

**Title:** What does the posterior tell us?

Use redesigned A02: two examples with large, directly labelled marginal intervals for lambda_1, lambda_2 and lambda_3. Keep the dashed lambda_th=0.2 line and class-probability bars. Panels:

- **Near an environmental boundary**
- **Predominantly filament**

The current A02 examples are suitable for explaining class ambiguity, not a clean narrow-versus-broad information comparison. The more class-confident example has wider continuous intervals. Preserve that fact.

**Bottom line:** Uncertain eigenvalues → probabilities for environments

No new main-slide equation. Explain the class rule verbally; keep the indicator-sum formula in backup. Give the interval levels, dimensionless x axis and “Mock examples” status legibly. Put long provenance in the notes.

### Speaker notes — copy into Keynote

So what does this distribution actually tell us?

For this galaxy, it gives us a range of tidal eigenvalues that remain plausible given the information supplied to the model. The width tells us what is still uncertain under that model. It isn't the galaxy physically changing between different environments.

We draw the three eigenvalues together, and apply the same threshold definition to each draw. The fraction classified as a filament, for example, gives us its filament probability.

Here, probability is split across a boundary. Here, most of it remains in the filament class. Notice that wider eigenvalue intervals don't necessarily mean a more uncertain class: where they sit relative to the threshold also matters.

But producing a distribution isn't enough. We also need to check whether those uncertainties are reliable.

### Figure prompt — numerical replot for NERSC, not an image generator

```text
Regenerate the current A02 posterior-example figure for the Tokyo talk using existing saved joint eigenvalue draws. No new training, refitting, simulation generation, sealed-phase access or new posterior sampling. First resolve the original figure's source receipt, exact checkpoint/model lineage, phase, target galaxy IDs, selection rule and saved draws. Stop and report if these cannot be bound; do not substitute illustrative numerical posteriors.

Retain the current two examples if their sources verify. Create two panels with the same eigenvalue x limits, titled “Near an environmental boundary” and “Predominantly filament”. Plot three horizontal marginal credible intervals per example, one row each for lambda_1, lambda_2 and lambda_3. Use thick 68% and thin 95% intervals with one clear legend, a distinct median marker and, if available, a separately labelled mock-truth marker. State whether intervals are equal-tailed or another construction. Show the dashed lambda_th=0.2 line. X axis: “Tidal eigenvalue (dimensionless)”. Do not crop posterior tails to exaggerate precision.

Under each example, show a large stacked probability bar, calculated by thresholding all three eigenvalues within every saved JOINT triplet. Use the existing class colors: mint void, blue wall/sheet, magenta filament, gold cluster/knot. Choose terminology consistent with the final deck. Direct-label meaningful percentages; round to whole percentages on screen, retain exact values and Monte Carlo counts in the receipt. Do not infer joint class probabilities by multiplying marginal probabilities. Zero observed draws is not proof of zero physical probability.

Black background, white text, ample space, no percentile-ranking labels and no “more/less precise” panel labels. Essential axis labels should render at roughly 36–44 pixels on a 1920x1080 canvas; verify the actual full-slide placement at 1280x720. Reserve title/takeaway bands for editable Keynote text. Keep “Mock examples” and the model identity legible. This plot illustrates posterior interpretation, not a calibration result.

Deliver vector PDF/SVG, PNG, generating script, exact plotted arrays and a compact JSON provenance/QA receipt. Recompute all displayed intervals and fractions from the same saved draws and check agreement with the old figure before changing its presentation.
```

## Slide 14 — Are these uncertainties reliable?

### Put this on the slide

**Title:** Are these uncertainties reliable?

One mock coverage plot. X: “Nominal credible probability”; Y: “Fraction containing mock truth”; identity reference line. Three clearly distinguished eigenvalue series at most.

**Bottom line:** Test against known truth in mocks.

**Visible qualification:** Mock validation does not establish real-DESI coverage.

Use the lineage corresponding to the posterior examples, preferably the deployed halo48 model's explicitly labelled exposed-ph006 diagnostic. Do not place the original blind ph001 badge on halo48 results. Keep a separate original-P12-A blind-validation panel in backup if useful. Do not use a diagram of training folds as the main validation result.

### Speaker notes — copy into Keynote

This is where the simulations give us another important advantage: we know the answer, so we can test the uncertainty.

For example, across a suitable mock test sample, a 68 per cent credible interval should contain the true eigenvalue about 68 per cent of the time. We check this at different probability levels, and across different observing conditions.

That tests whether the distributions are too narrow or too broad. We also need to check that they're informative, rather than simply returning the prior.

The important limitation is that these are tests within the mock model. They don't, on their own, establish the same reliability on real DESI data. With that distinction in mind, we can look at the survey application.

### Figure prompt — numerical replot for NERSC

```text
Prepare one presentation-scale coverage figure from existing, source-verified posterior-evaluation outputs. No new scientific evaluation or fitting is authorized by this figure request. Use the same model lineage as the A02 examples and state its phase, sample size, weighting and exposure status. In particular, P12-A halo48 exposed-ph006 diagnostics must not inherit the original P12-A blind-ph001 qualification.

Plot nominal credible probability on x and empirical fraction containing mock truth on y, both spanning 0–1, with a thin identity line. Show at most three labelled eigenvalue curves/series, using distinct line styles as well as colors. Use actual saved levels only: if only 68% and 90% measurements are available, plot those points rather than inventing a smooth coverage curve. Use dependence-aware uncertainty only if it is supplied by the registered evaluation; do not add iid galaxy-binomial error bands. Keep marginal and joint coverage distinct.

Use the existing black theme, white readable labels and large plotting area. Include a concise status caption with model, phase, sample size and whether already exposed; detailed weighting/protocol goes in the receipt and speaker notes. Add the visible statement “Mock validation; real-DESI coverage unverified”. Do not title it “Perfect calibration”. Include in the handoff the matching proper-score or informative-baseline evidence for speaker notes if it already exists, but do not invent such a comparison for this model version.

Deliver editable/vector output and PNG, exact plotted numbers, source paths/hashes, and a 1280x720 placement preview. If the required source is unavailable, report the missing item and retain the slide as a clearly marked placeholder, not a fabricated result.
```

## Short fallback: merge slides 11 and 12

Use the forward-model fork as build 1; reveal paired examples feeding the posterior estimator as build 2. Retain only the q_phi approximately p equation at the bottom. Use this ~70-second script:

So this is where the AbacusSummit simulations come in. We know the matter distribution, which gives us the tidal eigenvalues, and the galaxy and survey recipe gives us the mock observations. So we have paired examples of what we can observe and what we're trying to infer.

We can generate these pairs, but evaluating the likelihood directly would mean accounting for all the unobserved structure and galaxy populations compatible with those local eigenvalues.

Instead, simulation-based inference learns the conditional distribution from the pairs. In our case, we compress the galaxy context, include survey information, and use flow-matching posterior estimation to predict the three eigenvalues together.

That gives us a distribution rather than one best estimate, which we can obtain for many galaxies once the model is trained. It remains conditional on our assumed model and on the information retained in the summary.

## References and production boundaries

Primary references were verified in the preceding review and recorded in sources/research_fmsu_sbi_transition_20261013.md:

- AbacusSummit: Maksimova et al. (2021), https://arxiv.org/abs/2110.11398 .
- SBI: Cranmer, Brehmer & Louppe (2020), https://arxiv.org/abs/1911.01429 .
- Bayesian tidal-web precedent: Leclercq, Jasche & Wandelt (2015), https://arxiv.org/abs/1502.02690 .

Use the actual mock-product citation once bound. Neither an Abacus reference nor a DESI label establishes that every survey effect was modelled or marginalized. The general uncertainty argument is established Bayesian reasoning; the specific local inference implementation/application is the contribution here.

The schematic prompts can be executed locally as editable vector diagrams. The numerical prompts are for a plotting agent with access to existing NERSC outputs, not a text-to-image model. This document requests no automatic messages to another chat and no NERSC jobs. No graphics were generated or deck edits applied in preparing it.
