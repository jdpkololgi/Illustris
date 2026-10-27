# Tokyo deck: narrative, slide review and replacement speaking cues

## Integration completed — revised assets and full scripts

The following supersedes the earlier same-day status sections. The revised assets from `Working Files/FMSU-2026/revisions_20261018` are now installed in the saved Keynote deck.

- **Slide 8:** the branching slide has been moved from the end of the deck. Its new `A_body.png` ends at paired examples. The obsolete exact-posterior equation and arrow were removed, not merely covered.
- **Slide 9:** installed `E_survey_selection_body.png`.
- **Slide 10:** installed the simplified `B_body.png`; the previous publication banner, six-panel embedding and other body objects were removed.
- **Slide 11:** installed `C_probability_bar_body.png`. The separate bar distinguishes class probabilities from the example triplets and visually connects to the later posterior slide. The alternative C composition was not added as a redundant slide.
- **Slide 15:** installed `D_body.png`, retaining the saved diagnostic data but using the revised plain-language axes and legend. The old plot/build objects were removed.
- **Slide 20:** retained the four-map slide as backup and installed `E_tidal_four_panel_body.png`.

All six replacement bodies are placed at x=0, y=180, width=1920, height=900 on the 1920×1080 slide. Original editable Keynote titles are retained above them. The installed figures were visually checked for fit. This uses their designed internal margins rather than stretching or cropping the charts.

**All 20 presenter-note fields were replaced and their text verified**, not appended to the old notes. Each has one intended script, with pacing cues; technical provenance is separated into optional “if asked” paragraphs. The two backup scripts are explicitly marked. The main story ends at slide 18. Slides 19 and 20 are now marked skipped, and playback confirms slide 18 is the last of 18 presented slides. The summary's single outlook build was tested: one advance reveals it, with the next state at the end of the show. A complete timed spoken rehearsal is still required.

### Source-audit corrections incorporated into scripts

The delivered coordinate audit resolves the earlier uncertainty: **the ph002 four-map and matter/selection illustrations really use Z_COSMO and true real-space galaxy positions**, with pinned DESI/c000 distances in h^-1 Mpc. The ph006 ambiguity context instead uses observed Z with **Planck18** distances in Mpc and retains RSD. Neither “all panels are real space” nor “all panels use DESI fiducial redshift-space positions” is correct. Notes now describe each separately.

The Paper I receipt identifies **saved predicted-class density contours**, not a true-class scatter; no matching pointwise coordinates were available for entropy recolouring. The new script no longer refers to absent entropy panels. The ambiguity receipt confirms 209/303/0/0 classes in 512 joint draws (40.8%/59.2%) for the same saved galaxy. The TARP script retains the previously exposed evaluation status and describes the envelope as a descriptive maximum-deviation reference, not a calibrated confidence band.

### Transition check

7 → 8: the unobserved matter field motivates simulations that supply both targets and observations.

8 → 9: paired examples motivate a brief concrete explanation of the observing branch; the true-position illustration is identified explicitly.

9 → 10: the observation model defines what can be learned from galaxy-derived information; Paper I supplies the historical classification step.

10 → 11 → 12: class uncertainty leads to continuous environmental ambiguity, then the Bayesian formulation—not a claim that classification cannot have probabilities.

12 → 13 → 14: difficult likelihood but available simulated pairs → a learned summary-conditioned joint density → interpretation of widths and class probabilities.

14 → 15 → 16: meaningful uncertainty must be checked in mocks before discussing the provisional real-survey application.

16 → 17 → 18: probabilistic catalogue → descriptive galaxy science → central takeaway and bounded field outlook. No extra main-talk slide was added.

The script timing labels retain the 11:20 target plus 40 seconds of buffer. They are rehearsal budgets, not measured durations; do not read the optional provenance paragraphs in the main delivery. Six figures were imported; no training, inference or scientific recomputation was performed locally.

---

## Updated review after your edits — 18 October 2026

This section supersedes the initial review below. The live deck now has **20 slides: the talk concludes on 18, with two further slides intended as backup**. Those last two slides are not currently marked skipped. The old selected slides 9, 10, 17 and 20 are now **10, 11, 15 and 18**, respectively. The old branching slide 13 is now 20. Review was through the live Keynote editor; a timed rehearsal of all builds remains necessary.

### Changes actually made and saved in Keynote

- **10, Paper I:** replaced the input-diagram wording with “Galaxy graph metrics → Class probabilities / Tidal field: labels only.” Replaced the bottom claim with “Class probabilities describe uncertainty in a web label / Next: infer a joint distribution of continuous tidal eigenvalues.” Replaced the presenter notes, removing the entropy/error and physical-boundary conflations. The publication banner and six UMAP panels remain; their optional replacement is specified in Prompt B.
- **11, ambiguity:** replaced the presenter notes to distinguish example triplets from class probabilities and a posterior illustration from a demonstration of multiple complete matter fields. The raster figure itself remains unchanged; Prompt C is the priority regeneration request.
- **15, TARP:** kept the plot unchanged and replaced the duplicated presenter notes with a plain-language explanation of probabilities versus repeated success rates, with random-reference regions specified. The 80% example is explanatory, not an additional measured result.
- **18, summary:** created a small workflow above a large, bold, two-line central question and a smaller outlook beneath. Added a 0.4-second Dissolve build-in to the outlook and replaced the notes with a reveal cue. Preserved contact details and logos. The central question is “Which environments remain plausible, given what we observe?” The model conditioning is stated verbally.

The figures have not been recomputed, the branch slide has not yet been moved, and the remaining old notes have not all been overwritten. No NERSC jobs or messages were sent. A copy of the saved deck before these edits was retained at `/private/tmp/FMSU-Kavli-IPMU-2026-before-codex-layout-20261018.key`; this is temporary storage, not a durable archive, and may not include user edits that were still unsaved at that instant.

### Does the revised story work?

Yes. Your observing-model move and combined calibration slide improve the progression. Keep the emphasis on **definition → incomplete observations → distributions of environments → mock checks → galaxy science**. Do not add more training detail. The three-slide ending remains DESI probabilities, galaxy properties, then summary.

| Current slide | One or two ideas to communicate | Current recommendation |
|---|---|---|
| 1 | Infer local tidal environments; retain uncertainty | Keep; 15 seconds |
| 2 | Environment matters for galaxy science | Discuss association, not demonstrated causation |
| 3 | DESI gives the observational opportunity | Distinguish observed tracers from matter; remove unnecessary headline counts |
| 4–5 | T-web is a chosen local tidal definition; scale and threshold matter | Count threshold crossings, not a monotonic density sequence; glyph directions are schematic |
| 6–7 | We want tidal information but observe selected galaxies | Treat as one short reveal |
| 8 | Simulations provide paired observations and known targets | **Replace this slot with the branching diagram**, corrected as below; preserve the four maps as backup |
| 9 | Galaxy population, selection and RSD define the observation model | Keep briefly as the concrete illustration of the observational branch; no repeat of the whole training argument |
| 10 | Paper I established informative graph features; now infer continuous joint uncertainty | Text corrected; simplify UMAP/publication layout only if time permits |
| 11 | A single context can support more than one local environment | Regenerate the figure; exact instructions below and in Prompt C |
| 12 | Bayes states the inverse problem; simulation replaces an impractical explicit likelihood | Define D as context/data, not just the target galaxy; keep M |
| 13 | Learn a conditional joint density from paired examples | One method slide is sufficient; keep q distinct from ideal p |
| 14 | Read widths and threshold-crossing probabilities | Give this slide time; joint draws, not independent marginal draws |
| 15 | Do assigned probabilities agree with repeated truth checks? | **Keep the TARP plot**; use the new notes |
| 16 | Same DESI galaxies, four different probabilities | Provisional application; not four disjoint samples or a reconstructed field |
| 17 | A first galaxy-property use of the catalogue | Trace is density-related; descriptive association, not an isolated geometry effect |
| 18 | Plausible environments are the product; connected structures are the outlook | Redesigned and animated; stop here |
| 19 | Filament-probable objects are not a connected filament catalogue | Backup only |
| 20 | Simulation branches produce paired examples | Move into slot 8 after correction; do not present it again at the end |

Timing target: slides 1–3 1:10; 4–7 1:50; 8–9 0:55; 10–11 1:20; 12–13 1:15; 14 1:05; 15 0:55; 16 1:00; 17 1:10; 18 0:40. Total **11:20**, leaving 40 seconds for transitions. These are budgets, not measured delivery times.

### Should the branching diagram replace “Simulations provide known tidal environments”?

**Yes.** It better explains why simulations make this inverse problem tractable. Keep the four eigenvalue/class maps as backup. The branch diagram should terminate at **“Paired examples: observed context + known local tidal target”**, not an arrow straight into an exact posterior. The later learning slide supplies that missing estimation step.

Suggested narration: “In each simulated realisation we have both sides. From the matter field we calculate a known local tidal target. From the galaxy and observing model we obtain the mock information that would be available to us. Pairing them lets us learn a conditional distribution—and then test it where we know the answer.”

Slide 9 can remain, but spend only 15–20 seconds showing what the galaxy/observing branch means. If this feels repetitive in rehearsal, omit 9 rather than restore the four-map explanation to the main talk. Prompt A supplies a standalone replacement specification.

### Exactly what was wrong on old slides 9/10?

#### Current 10: Paper I

1. **Inputs versus labels:** “IllustrisTNG300 tidal field + graph/geometric metrics” in one input stream suggests inaccessible tidal information goes into the predictor. It belongs on the label/truth side. The text is now corrected; a clean replacement diagram should show separate input and supervision arrows.
2. **Entropy is not error:** the lower UMAP panels are Shannon-entropy panels. High entropy means the class distribution is diffuse; it does not establish that the predicted class is wrong. An embedding boundary is also not automatically a physical boundary in the cosmic web.
3. **Classifiers already have probabilities:** do not motivate SBI as “classification has no uncertainty.” The advance is a joint continuous environmental distribution, from which different threshold-defined classifications can be derived.

The new bottom text in Keynote states exactly that distinction. The next visual simplification would be one class-coloured UMAP and, optionally, the same embedding coloured by entropy—not six small panels. Retain a small paper citation rather than a large publication screenshot/QR. This optional change is specified in Prompt B; it is not necessary to delay the talk for it.

#### Current 11: ambiguity example

The current image has candidate triplets on the left pointing into the galaxy context on the right. That reads as a **forward generator**, but three local eigenvalues do not generate a whole galaxy catalogue. The bottom context label also risks locating the posterior on the point map.

Regenerate with the **observed mock context on the left → inferred local possibilities on the right**. Label the arrow “Infer under model M.” Put “Illustrative posterior draws” over the two triplet cards. Put **P(void) = 40.8%** and **P(wall) = 59.2%** beside the corresponding class summaries, explicitly separate from the displayed example triplets. Verify these numbers against the saved draws before printing them. Do not say each exact triplet has that probability.

Keep “Mock posterior example” visible. Suggested on-slide sentence: **“One observed context can support more than one tidal class.”** Suggested footer: “Triplets are example draws; percentages summarise classes.” Prompt C provides the full specification, including source checks and coordinate labels.

### Coordinate correction: comoving does not mean real space

You are right. Converting observed redshift to distance with a fiducial cosmology gives `s = chi_fid(z_obs) n_hat`; it does **not** remove peculiar-velocity contributions. These remain redshift-space positions, expressed in comoving units.

My earlier categorical descriptions of slides 8/9 as real-space displays were too strong. Their embedded captions say “real space” / “matched real-space slice”, which conflicts with your account of the galaxy coordinates. The correct action is to check the plotting inputs and fix the captions—not infer coordinate provenance from Mpc axes.

Distinguish the two objects: the simulated matter/tidal grid can genuinely be real space while the galaxy context uses redshift-space coordinates. Host-linked targets are not automatically the tidal field evaluated at the observed redshift-space location. Prompt E asks NERSC to resolve that distinction and return precise labels; do not relabel a real-space matter grid as redshift space just to match its neighbouring panel.

### TARP: keep the figure, simplify the language

The current plot is suitable. No mandatory regeneration is needed. Say:

> “Do the probabilities mean what they say? If a region is assigned 80% probability, the true answer should fall inside such regions about 80% of the time across many mock examples. The diagonal is that agreement. TARP checks the joint distribution using regions around random reference points. This curve is encouraging in mocks; applying the model to DESI is a separate transfer question.”

This avoids requiring the audience to know “nominal” and “empirical coverage”. It is not a claim that this exact dataset has a newly measured 80% result. Keep the 50,000-galaxy sample and mock-test scope available. The previously inspected evaluation sample is not a new blind final test. Keep the bootstrap maximum-deviation shading explanation in notes; do not describe it as a pointwise band. Prompt D is **optional cosmetic relabelling only**.

### NERSC handoff

Detailed, presentation-independent requests are in [fmsu_nersc_figure_revision_prompts_20261018.md](fmsu_nersc_figure_revision_prompts_20261018.md). Priority: **C (ambiguity), E (coordinate audit), A (branch diagram)**. B is optional simplification; D is optional and the existing TARP figure can remain. Existing plots should be regenerated from their source data, never recreated with invented scientific marks.

---

## Initial review — historical numbering, retained for reference

**The material below records the earlier deck state. Its numbering, suggested order, unedited-status statement and categorical real-space descriptions are superseded by the revision above.** It remains useful for the earlier slide-by-slide speaking cues not replaced in this pass.

Initially reviewed 18 October 2026: `FMSU-Kavli-IPMU-2026.key`, then 20 active slides and one skipped slide. The initial review itself did not edit Keynote.

Scientific framing was checked against the latest local SCIENCE_LOG entry (14 October). This was not a fresh NERSC run audit or an independent recomputation of figure statistics. In particular, slide 19's 59,774-object wedge is the sample stated on the current figure; its estimator and source receipt should be checked before adding numerical claims.

## Overall judgement

Yes: the deck now has a coherent and scientifically interesting story. It is about **inferring a physically specified environment, retaining uncertainty, and using that information in galaxy science**, rather than about optimising a neural network. The strongest material is slides 10, 15, 17 and 18. The closing application gives the talk a purpose beyond methodological validation.

The weakness is repetition in the middle: slides 8, 11 and 13 explain related aspects of simulation pairs, while slides 16 and 17 repeat the calibration setup. Slide 9 has too many competing visual elements. These slow the story just before the most interesting results.

The intended audience takeaway should be:

> We infer a distribution of local tidal environments from galaxy observations, conditional on a specified simulation model. We test those distributions in mocks, then provisionally use them to study DESI galaxies.

### Highest-priority corrections

1. **Slides 4–5:** replace the “Density” ordering arrow. T-web class counts eigenvalues above a threshold; it is not simply a monotonic density sequence. Also, eigenvalues do not specify directions—the eigenvectors do.
2. **Slide 9:** separate observables used as inputs from simulation tidal information used as training labels. The current diagram and notes can sound as though the tidal field is supplied to the classifier at inference.
3. **Slide 12:** `D` is not merely the specified target galaxy. Define it as the observed galaxy data/context and relevant survey information; `g` identifies the target.
4. **Slide 13:** simulation pairs do not directly yield an exact posterior. Remove the direct arrow to `p`, or explicitly insert posterior estimation and use `q`.
5. **Slides 18–19:** replace the stale notes. Slide 18 now has four probability maps. Slide 19 now shows a wedge/trace analysis, not the older result underlying the −0.19 dex script.
6. **Slide 19:** a trend with tidal trace is a density-related association, not evidence by itself for an additional anisotropic-web effect at fixed density.

## Recommended order and timing

Minimal rearrangement: move current slide **11 immediately after 8**. Then the observed-context ambiguity on 10 leads directly into Bayes on 12. Put **13 in backup** if using both 8 and 11; the paired-example idea can be said over those figures. Omit **16**, using its question as the transition into 17. Treat 4–5 and 6–7 as short builds, not four separate explanations.

| Current slide numbers, in speaking order | Purpose | Budget |
|---|---|---:|
| 1 | Question and scope | 0:15 |
| 2–3 | Why environment; why DESI | 0:55 |
| 4–5 | Physical definition and chosen scale | 1:10 |
| 6–7 | Target versus what is observed | 0:40 |
| 8, 11 | Known simulation targets and survey-like observations | 0:55 |
| 9 | Paper I → new scientific question | 0:35 |
| 10 | Concrete ambiguity | 0:45 |
| 12, 14 | Why Bayes; why SBI | 1:15 |
| 15 | Read the posterior and class probabilities | 1:05 |
| 17 | Joint-posterior check in mocks | 0:55 |
| 18 | DESI probability maps | 1:00 |
| 19 | Galaxy-property application | 1:10 |
| 20 | Conclusion and bounded outlook | 0:40 |
| — | Total / remaining buffer | **11:20 / 0:40** |

This keeps the ending to three slides. Do not add a field-results slide. If running late, shorten Paper I and the class-fraction comparison, not the uncertainty interpretation or conclusion.

## Slide-by-slide review

### 1 — Learning the Cosmic Web

**Communicate**

- The target is the local tidal environment of a galaxy, not cosmological parameters.
- The advance is a distribution of plausible environments rather than just a label.

**Content/layout:** Keep the survey image and subtitle. The footer is busy, but not a major problem; the social QR can live only on the final slide. Do not spend time reading affiliations and supervisors aloud.

**Replacement speaking cue**

> I’m Dakshesh, a PhD student at UCL. I’ll talk about what DESI galaxies can tell us about their local tidal environments—and how we can retain the uncertainty in that inference rather than reduce everything to one web label.

### 2 — Cosmic Web Inference in Stage IV Surveys: Dressler

**Communicate**

- Galaxy properties and environment are associated: this motivates measuring environment well.
- Local density is a useful starting point, but not the whole tidal description.

**Content/layout:** Rename to **“Why infer environment?”** Keep the historical figure, but foreground the morphology–density trend with direct labels. Avoid explaining every curve and histogram. Replace “might influence” with “is associated with” unless explicitly discussing a causal hypothesis. Remove the old note saying environmental studies have until now been restricted to small samples; that skips decades of survey work.

**Replacement speaking cue**

> The motivation is galaxy evolution. The morphology–density relation is a classic example: the mix of galaxy morphologies changes with local density. That is an association, not by itself a causal explanation. Here I want to extend the environmental description to the local tidal field.

### 3 — Cosmic Web Inference in Stage IV Surveys: DESI

**Communicate**

- DESI makes environmental inference useful at large survey scale.
- Distinguish the observed galaxy distribution from the underlying environment being inferred.

**Content/layout:** Suggested title: **“Environmental inference at DESI scale.”** Remove the visible “Caption” placeholder. Drop the equation involving `T = −H` here: `H` is undefined and the sign convention distracts before the definition slide. Fix “Galaxies properties” to “Galaxy properties.” If retaining the 50-million spectra figure, supply a dated official source and distinguish spectra from unique galaxies and from the BGS analysis sample. The exact number is unnecessary for this argument.

**Replacement speaking cue**

> DESI gives us a much larger and richer opportunity to connect galaxy properties with environment. But a galaxy catalogue is not a direct map of matter. The question is what environmental information we can infer from those observations, and how certain that inference should be.

### 4 — How do we define the Cosmic Web?

**Communicate**

- This is a local, dynamical T-web definition—not a topological filament skeleton.
- The eigenvalues describe the principal tidal responses; their threshold crossings define the classes.

**Content/layout:** Keep the four glyphs and ordered eigenvalues. Replace the **“Density” arrow** with **“Number above threshold: 0 → 3”**, or remove it. Show `T_ij = ∂_i∂_j φ_R`, making the smoothing explicit. Explain that eigenvectors give directions and eigenvalues give signed strengths. The glyphs are schematic tidal responses, not measured velocity arrows. This is consistent with the threshold-count definition in [Forero-Romero et al.](https://arxiv.org/abs/0809.4135).

**Replacement speaking cue**

> There are several useful definitions of the cosmic web. Here I use a local tidal definition, rather than extracting a filament skeleton. We take the Hessian of the smoothed gravitational potential. Its eigenvalues describe the tidal response along three principal directions. Counting how many exceed a chosen threshold gives voids, walls, filaments and knots—the cluster class in these figures.

### 5 — Definition: threshold build

**Communicate**

- Smoothing scale and threshold are part of the definition, not incidental settings.
- This talk uses one specified environmental convention.

**Content/layout:** Make this a brief build of slide 4. Replace “incomplete collapse” alone with **“Here: R = 7 h⁻¹ Mpc; λ_th = 0.2.”** The positive-threshold motivation can remain in notes. Do not claim this threshold is a universal exact collapse boundary. Remove “in increasing order of density” from the old notes.

**Replacement speaking cue**

> Both the smoothing scale and the threshold matter. Here the environment is defined at seven inverse-h megaparsecs, using a threshold of 0.2. So when I say “filament,” I mean this particular local tidal classification, not every possible definition of a filament.

### 6 — Aim: infer tidal eigenvalues

**Communicate**

- Infer continuous local tidal quantities from galaxy observations.
- Web-class probabilities will be derived from those quantities.

**Content/layout:** Shorten title to **“Infer tidal environments from galaxy observations.”** Label the two layers “Observed galaxy context” and “Latent matter/tidal environment.” The heatmap has no clear scalar legend and should not look like a direct image of all three eigenvalues: identify it as schematic. Do not describe this illustration as a measured Abacus eigenvalue map; that is slide 8.

**Replacement speaking cue**

> Our target is the three tidal eigenvalues associated with a galaxy’s environment. They retain more information than one class. In the mock training data these targets are attached through the galaxy’s host halo. Once we infer the eigenvalues, we can apply the same web definition afterwards.

### 7 — Density field not directly accessible

**Communicate**

- We observe sparse, biased tracers rather than the matter field.
- The hidden field is the reason this is an inference problem.

**Content/layout:** Keep as a quick reveal. Prefer **“We observe galaxies, not the matter field.”** A red cross can mean “this field is wrong”; fading the lower layer and labelling it “latent—not directly observed” is clearer. The old notes duplicate slide 6 and do not explain the cross.

**Replacement speaking cue**

> The difficulty is that the lower layer is not observed. We have galaxies, selected by a survey, rather than the underlying matter field itself. Recovering environmental information from them is therefore an inverse problem.

### 8 — Forward model observation with simulations

**Communicate**

- Abacus supplies a known matter field and hence known tidal targets.
- Continuous eigenvalues contain information discarded by the class map.

**Content/layout:** Retitle **“Simulations provide known tidal environments.”** The four-panel figure is useful; remove the repeated equation to enlarge it. Highlight one location across all panels. The footer identifies a real-space slice: do not present it as an observed redshift-space reconstruction. Replace the bidirectional catalogue/field arrow with “paired observations and targets”; the inverse is not unique.

**Replacement speaking cue**

> In Abacus we know the simulated matter distribution, so we can calculate the tidal field. These panels show its three eigenvalues and the corresponding class map. The continuous fields contain structure that the four-class description compresses away. This gives us known targets with which to train and test inference.

### 9 — Kololgi+2026: graph neural network classifier

**Communicate**

- Paper I established that galaxy-derived graph information is informative about simulated web labels.
- The new question is which continuous environments remain plausible, not simply which class wins.

**Content/layout:** This is the most crowded slide. Reduce the publication banner and QR to a small citation. Use one selected UMAP panel, not six tiny panels. Most importantly, separate **galaxy-derived inputs → classifier** from **simulation tidal field → training labels**. The current diagram and notes conflate these. UMAP overlap alone does not establish irreducible uncertainty or prove that errors occur at physical web boundaries; soften that claim unless showing the relevant diagnostic. Classifiers can output probabilities, so do not frame the advance as “classification cannot express uncertainty.”

**Replacement speaking cue**

> My first paper showed that graph information derived from simulated galaxies can predict their web labels. The tidal field supplies the labels, rather than being an input we would have for observed galaxies. The next question is richer: can we infer a joint distribution of tidal eigenvalues and derive the environmental probabilities from it?

### 10 — Many tidal environments are compatible with observations

**Communicate**

- One observed galaxy context need not identify a unique local environment.
- A split class probability can be an appropriate inference, not simply a failure.

**Content/layout:** One of the strongest slides. Shorten title to **“One galaxy context, several plausible environments.”** The arrows currently point from eigenvalue examples into the catalogue; reverse them or use a bracket labelled “plausible local targets.” Three local eigenvalues are not a forward generator for an entire catalogue. Label the 40.8%/59.2% numbers as class probabilities, not probabilities of the individual displayed triplets. Keep “mock posterior example” visible: this illustrates the model output, not an independent demonstration of two full matter fields producing exactly the same catalogue.

**Replacement speaking cue**

> Here is the kind of ambiguity we want to retain. For this mock galaxy context, the inferred distribution supports both void-like and wall-like environments. The two triplets are examples, while the percentages refer to the classes. Sparse galaxy observations do not uniquely fix the matter field, so forcing a single answer can conceal meaningful uncertainty.

### 11 — Train on what the survey can observe

**Communicate**

- The mock must connect matter to galaxies and then to survey-selected observations.
- The posterior is conditional on this modelling recipe.

**Content/layout:** Move after slide 8. Rename the middle box **“Galaxy + observing model”**: galaxy population modelling is not merely a selection cut. Keep matched scales between the matter and galaxy panels. The displayed slice is real-space even if selection is survey-like; do not narrate this image as a demonstration of redshift-space displacement. Replace the duplicated slide-10 notes.

**Replacement speaking cue**

> Knowing the matter field is only half the story. We also need to model the galaxies and what the survey retains. That includes the galaxy population and observational selection. These choices define the model under which our probabilities are meaningful; they are not uncertainties that disappear simply because we use SBI.

### 12 — Uncertain environments to Bayesian inference

**Communicate**

- The posterior combines compatibility with the data and the model’s distribution of environments.
- SBI is useful because simulation is available while direct likelihood evaluation is difficult.

**Content/layout:** Keep the equation and generous spacing. Correct the legend to **“D: observed galaxy data + survey information; g: target galaxy.”** Use bold λ for the three-dimensional target if convenient. Keep `M` visible. The ideal full-data posterior here and the actual summary-conditioned estimator on slide 14 must not silently become interchangeable.

**Replacement speaking cue**

> Bayesian inference expresses the question directly: which tidal eigenvalues are plausible for this galaxy, given the observed catalogue and our model? The likelihood is difficult because the three local eigenvalues do not specify all the surrounding matter, galaxies and selection effects. But we can simulate examples of both the observations and their targets. That is what makes simulation-based inference useful here.

### 13 — Simulations give us both sides of the problem

**Communicate**

- Each training pair comes from the same simulated realisation.
- Pairs enable posterior estimation; they are not the exact posterior themselves.

**Content/layout:** The branching diagram is clear, but redundant if 8 and 11 remain. Recommended backup. If retained, end at **“paired examples”** and remove the large arrow to exact `p`; slide 14 supplies the estimator. No regeneration is needed for a text/arrow edit.

**Replacement speaking cue, if retained**

> Crucially, each mock observation and its tidal target come from the same realisation. These paired examples let us train a conditional distribution. They do not give us the posterior directly—we still have to estimate it and test that estimate.

### 14 — Learning the posterior from simulations

**Communicate**

- A learned summary plus survey information conditions a joint three-eigenvalue distribution.
- The result is conditional on the simulation model and the retained information.

**Content/layout:** Keep this as the only substantive neural-method slide. The displayed `q_phi(λ_g | S_g; M) ≈ p_M(λ_g | S_g)` is appropriately qualified. Enlarge the three-box flow; reduce the prominence of the SBI citation heading. Do not add architecture, loss, folds or training schedules. Do not automatically relabel the current compressor as a GNN merely because Paper I used one.

**Replacement speaking cue**

> We compress the galaxy context and combine it with survey information. A conditional density model then learns a joint distribution of the three eigenvalues from the simulated pairs. The important word is joint: their correlations matter. This is a posterior estimate conditional on the information retained in our summary and on our simulation model.

### 15 — What does the posterior tell us?

**Communicate**

- Width measures remaining uncertainty conditional on the supplied information and model, assuming a reliable estimator.
- Class certainty depends on how the joint distribution sits relative to the threshold, not just its width.

**Content/layout:** Keep and give it time. Connect the left example explicitly to slide 10. Enlarge the histograms by simplifying the header/legend; use unmistakably different truth, median and threshold styles. The three marginals are a display, not independent distributions to multiply. Derive class probabilities from joint triplets. The existing notes are largely good and need shortening, not conceptual replacement.

**Replacement speaking cue**

> These are two mock galaxies. On the left, the same example as before, probability straddles an environmental boundary. On the right, most probability remains in the filament class. Width describes uncertainty under the supplied information and model—it is not the galaxy changing environments. We sample the three eigenvalues together and classify each triplet. The resulting fractions give the class probabilities. So broad eigenvalue distributions need not imply an equally uncertain class.

### 16 — Are these uncertainties reliable? Empty setup

**Communicate**

- Producing a distribution is not enough; it must be checked against known truth.

**Content/layout:** Omit, or use as a five-second reveal before 17. It currently has a title/subtitle and a largely empty body. Its notes duplicate 17 and already transition to DESI before the TARP plot appears. There is no need to commission a new figure for it.

**Replacement speaking cue, if retained**

> But producing a distribution is not enough. Are its uncertainty estimates reliable when we know the answer?

### 17 — Are these uncertainties reliable? TARP

**Communicate**

- This checks the joint posterior using mock truth across a test population.
- A near-diagonal finite-sample diagnostic is encouraging, not a guarantee of exact or real-DESI calibration.

**Content/layout:** Keep the large plot. Suggested subtitle: **“Joint-posterior check in mocks.”** State the sample size, 50,000 galaxies, in readable text. Translate “already exposed” into notes as “held out from training, but previously inspected during evaluation”—not a fresh blind final test. Simplify the legend, but preserve that the shading is a bootstrap maximum-deviation envelope, **not a pointwise confidence band**. The old 68%-marginal-interval explanation is useful intuition but does not describe TARP precisely; [Lemos et al.](https://arxiv.org/abs/2302.03026) introduce a random-reference-point coverage test.

**Replacement speaking cue**

> Here we test the joint eigenvalue posterior using TARP on 50,000 mock galaxies held out from training. The method compares nominal and empirical coverage using regions defined relative to reference points. The curve lies close to the diagonal in this evaluated sample, which is encouraging. It is not proof of correctness for every galaxy or observing condition, and it does not establish calibration on real DESI data.

### 18 — Probabilistic environments for DESI BGS

**Communicate**

- Every panel shows the same galaxies, coloured by a different environmental probability.
- This is provisional local environmental inference, not a reconstructed field or filament network.

**Content/layout:** Keep the four maps: they make the probability-vector idea tangible. Use the same 0–1 colour scale, as now. Low-probability purple points are dark against black: slightly lift the colour floor or add a subdued neutral context layer without changing the probability scaling. Highlight one common region in all panels if helpful. Replace the technical “provisional P12-A” alone with **“Provisional DESI application; calibration tested in mocks.”** Keep coordinate convention, slice depth and sample count accessible.

**Replacement speaking cue**

> These four panels show exactly the same 4,459 DESI galaxies. What changes is the probability being displayed: void, wall, filament or cluster. They are not four disjoint galaxy samples. We retain uncertainty instead of forcing a label. These are provisional, model-dependent estimates: the calibration check was in mocks, and transfer to real DESI remains a limitation. This is local environmental information, not a reconstructed matter field.

### 19 — Connecting tidal environments to galaxy properties

**Communicate**

- The inferred environments can be used in a galaxy-property comparison with mass/redshift matching.
- The displayed association is descriptive; neither it nor class-fraction agreement validates the real environments.

**Content/layout:** This is now a different analysis from the old notes. The footer identifies a **59,774-galaxy wedge**, with a trace-binned mean-log-sSFR curve, class-average diamonds and 16–84% spatial-bootstrap intervals. Remove the old **−0.19 dex** quotation unless re-established for this exact figure and estimator.

Recommended layout: give the right-hand property plot approximately two-thirds of the width. Keep the left fraction comparison as a small contextual panel or backup. Shorten title to **“Galaxy properties vary with inferred environment.”** Clearly label “Line: trace bins” and “Diamonds: class averages.” Add visible CIGALE attribution if that is the source for this specific wedge. Verify whether the fraction bars and class averages use soft probabilities or hard labels; do not infer the weighting rule from the colours alone.

The critical interpretation is that `tr(T_R) = δ_R` under the normalized Poisson convention. Thus a trace trend is a density-related result. The class diamonds overlap in trace, as the slide correctly notes: they are not four successive non-overlapping trace bins. This plot does not isolate an anisotropy effect at fixed density. Likewise, small bootstrap intervals do not include all SED and mock-to-data systematics.

**Replacement speaking cue**

> Here is an initial galaxy-property application in the lower-redshift sample. The left panel compares inferred class fractions, but that is only an aggregate comparison—not a truth test for DESI. On the right, specific star formation decreases with inferred tidal trace in a mass- and redshift-matched comparison. The line shows trace bins; the diamonds summarise the classes, which overlap in trace. Since trace is density-related, this does not yet isolate a separate geometric effect. It demonstrates a scientific use of the catalogue, with transfer and galaxy-property systematics still to assess.

### 20 — From labels to environmental uncertainties

**Communicate**

- The achieved progression is physical definition → inferred uncertainty → galaxy science.
- Connected-structure uncertainty additionally requires a validated spatial joint posterior.

**Content/layout:** Keep the three-slide ending and contact details. Give the central takeaway hierarchy: a small three-stage workflow above, one large two-line question in the middle, and a smaller outlook below. The current similarly sized long lines compete. Suggested central text: **“Which environments remain plausible, given what we observe?”** Add “under our model” verbally. No new field-model performance plot: the latest local log still treats the field models as exploratory, not qualified calibrated field posteriors.

**Replacement speaking cue**

> The main message is that environmental inference is more than assigning a label. We define a tidal environment, infer its uncertainty, and use that information in galaxy science. The next step is uncertainty in connected structures, which needs a spatial joint posterior—not independent draws from local distributions. For now, the question is: which environments remain plausible, given what we observe and the model we use?

### Skipped slide — Filament-probable galaxies

**Communicate if asked**

- The highlighted objects have `P(filament) ≥ 0.5`; they are not an extracted connected filament network.
- The map is a crop: 626 highlighted objects within 1,539 context galaxies, not the 4,459-object full slice.

**Content/layout:** Useful backup for this conference. Its explicit “not an extracted filament network” line is valuable. Keep skipped rather than adding a fourth closing slide. The notes still describe an unrelated two-panel map and should be replaced. Mixed Mpc axes and an `h⁻¹ Mpc` smoothing-circle annotation are acceptable only if the plotted circle has been converted consistently; verify the figure-generation receipt.

**Optional speaking cue**

> This crop highlights galaxies whose inferred filament probability is at least one half, while retaining the others as context. These are local tidal classifications. They do not, by themselves, define the connections, branches or uncertainty of a filament network.

## Minimal figure/edit worklist

- **No wholesale redesign:** preserve black backgrounds, white text and the mint/blue/pink/gold class colours. Keep continuous probability colour scales visually distinct from categorical class colours.
- **Keynote edits:** titles, slide-4 arrow label, slide-12 data definition, slide-13 arrow, final-slide hierarchy, notes, and optional reordering/removal.
- **Slide 9:** recompose the Paper I visual around one diagnostic and an unambiguous input/label diagram. This is the highest-value visual replacement.
- **Slide 10:** fix arrow semantics and class-probability labels; retain the actual saved mock example and draws.
- **Slides 15/17:** simplify legends and increase plot area. Preserve all numerical data and the exact meaning of the TARP envelope.
- **Slide 18:** improve low-probability visibility; do not change scales independently between panels.
- **Slide 19:** verify the current wedge receipt, weights, mass/redshift-matching procedure and uncertainty calculation; then re-layout the existing result. Do not substitute an older full-footprint statistic.

Any NERSC figure refresh should reuse frozen data, not train new models for aesthetic reasons. Require legible labels at actual presentation size, explicit units, shared scales for comparisons, sample/selection identification, model-status caveats, and a receipt pointing to inputs and plotting settings. Use plotted scientific data, not generative imagery, for numerical results.

## Rehearsal checks

- Say what is observed, what is inferred and what supplies truth before introducing Bayes.
- Let slide 10 lead into 12 without another long methodological detour.
- Do not read pipeline identifiers or file provenance aloud.
- Do not describe mock calibration as real-DESI calibration.
- Do not call posterior width exclusively irreducible physical uncertainty: summary information loss and estimator error also matter.
- Do not imply that classification probabilities constitute full propagation of all uncertainties into a galaxy-property relation.
- Reach slide 18 around 8:30; retain the conclusion even if the earlier discussion runs long.

The scientific-slides guidance informed the recommendation to simplify the middle, enlarge result graphics, and make each slide support one spoken claim. The science, examples and revised notes remain specific to the inspected deck.
