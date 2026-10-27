# Tokyo FMSU talk: 12-minute structure and slide-by-slide review

Draft prepared 10 October 2026. Scientific status is the local `SCIENCE_LOG.md` snapshot, including its latest NERSC-side updates; no new remote job inspection or scientific evaluation was performed for this review.

Companion: [NERSC figure and media brief](fmsu_tokyo_nersc_assets_20261010.md).

## Recommendation

Use **10 main slides, rehearsed to 11:20**, leaving 40 seconds for pauses and transitions. Retain the current deck's visual identity, but rebuild the scientific results around the current model and validation history. This is a focused Paper-II talk with a short field-inference outlook, not a history of every model tried.

The central question is:

> Given the galaxies we can observe, what can we infer about their underlying tidal environments—and which uncertainties have we actually validated?

The story is **continuous environmental inference → principled validation → provisional DESI application → why coherent spatial uncertainty is a separate challenge**. The important progress since Paper I is not just a better predictor: it is a clearer estimand, spatial/phase separation, out-of-fold posterior training, and an explicit boundary between information, calibration and deployment.

## Conference positioning

The [8 October programme](https://drive.google.com/file/d/1_C6Hndw4oT53IBn4h-3NPx93-TKWgcGV/view), pp. 1–3, places Dakshesh at **11:07–11:22 JST on Monday 26 October**, in **Defining and reconstructing the web**. The format is **12 minutes + 3 minutes Q&A**, not 15 minutes of slides. Wendy Wallace's AbacusSummit filament talk precedes yours; Farhanul Hasan's MCPM/DisPerSE talk follows. Leander Thiele speaks at 14:15–14:22, followed later by the session discussion at 14:29–14:59.

The [conference objectives](https://indico.ipmu.jp/event/522/overview) emphasize robust observables, tracer/scale dependence, reconstruction and connections to galaxy evolution. Accordingly:

- Skip a long generic introduction to why filaments matter. This audience already cares about that.
- State early that the primary output is a **local tidal-environment posterior**, not a filament-spine catalogue, connectivity graph or direct measurement of the dark matter field.
- State the smoothing scale. “Multiscale conference” does not justify implying that a fixed-scale target is a multiscale web finder.
- Use one galaxy-property result to connect to the rest of the meeting, without treating an expected trend as a calibration test.
- Position Thiele/Tastet as complementary: stochastic filament-catalogue geometry versus local tidal eigenvalues. The programme abstract describes the filament work; [Thiele 2026](https://arxiv.org/abs/2602.14630) is the related **void** paper, not the filament paper.
- Do not claim that ambiguity from sparse tracers, Bayesian web reconstruction, or probabilistic classification is newly discovered here. [Leclercq, Jasche & Wandelt 2015](https://arxiv.org/abs/1502.02690) already propagate uncertainty into dynamical web classifications.

### Resolve the submitted-title mismatch

The programme still says “Probabilistic Inference of the Cosmic Web in Galaxy Surveys: A Graph-Based Approach” and describes a GNN. The current VAC is **U-PATCH plus a conditional flow-matching posterior estimator**, not the old GraphNet stack. Do not present the current results under an unchanged GNN architecture diagram.

Suggested slide title: **Learning the cosmic web: posterior inference for DESI galaxies**.

Retain the registered title in small text or ask the organisers whether its subtitle can be updated. In either case say: “We began with graph-based classification; the current work compares representations under a common inference and validation protocol.” No organiser message has been sent.

Paper I should now be cited as **Kololgi et al. (2026), RASTI, rzag025**. The [published article](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzag025/8554208) is dated 28 March 2026; the old deck repeatedly says 2025.

## Scientific status to freeze into the talk

| Evidence | What can be said | Essential boundary |
| --- | --- | --- |
| Original P12-A blind ph001 evaluation | Ten registered gates passed on 4,897,905 mock-galaxy rows; one opening, no post-open refit. C90 = 89.115/89.481/89.001% for the three ordered eigenvalues. | Conditional on the registered simulator/HOD and tested diagnostics. Millions of correlated galaxies are not millions of independent universes. Not exact conditional calibration. |
| Halo48 deployment successor | Frozen encoder, regenerated out-of-fold summaries, refitted untempered posterior; exposed-ph006 conditional checks and eight-core observer golden replay passed. | Not the original blind-tested fit. Natural-weight C68 = 69.469/69.870/70.975%, C90 = 91.012/91.041/91.765%; modest resolved overcoverage. Do not attach the old blind badge to the new model. |
| Provisional Loa VAC | 5,436,413 unique IDs; 5,404,568 supported, 31,845 flagged nulls. Full-footprint technical production is complete. | Not a science-qualified public release. Real-DESI coverage unverified; mock-population mismatch especially above z≈0.35, severe at 0.45–0.55. Low z is not automatically certified. |
| Current CIGALE application | 4,989,187 usable supported CG15 galaxies overall. Primary talk comparison: 2,327,331 at 0.20≤z<0.30, full available footprint. Mass/z-standardized knot-minus-void mean log sSFR contrast −0.186 dex; low-sSFR fraction +7.13 percentage points. | Probability-weighted descriptive associations, not recovered true-class populations or causal quenching. Small sky-bootstrap intervals exclude SED and transfer systematics. |
| Coupled field work, 5–8 October | Operator mismatch explains part, not all, of regional tidal errors. Informative class probabilities coexist with failed regional/joint calibration. True-coarse conditioning does not remove the class-fraction defect. | Field-at-observed-cell targets and P12 host-halo targets are not interchangeable; no head-to-head superiority claim. |
| Gaussian controls, 8–9 October | A local-canvas counterexample exposes covariance errors despite near-correct voxel variance. Learned whole-field U-Net/transformer pilots improve some errors but do not pass qualification. | Neither “whole-field fixes it” nor “the transformer wins” is established. Analytic toy results are not a quantitative decomposition of Abacus errors. |
| Latest, 10 October | Full-cap regeneration and corrected extraction have passed technical checks for available products; the new four-draw task brought the recorded total to 57/192 task products. | Full panel, revised sensitivity gate and scientific qualification remain incomplete. This is not a new calibrated field result. Omit job IDs and GPU accounting from the talk. |

Sources: [VAC plan](plan_desi_p12a_vac_20260924.md), [halo48 review](p12a_halo48_readiness_20260925.md), [conditional/golden checks](p12a_golden_conditional_results_20260925.md), [provisional release notes](../../GraphWeb_DESI/docs/p12a_provisional_release_20260928.md), [CIGALE report](../../GraphWeb_DESI/docs/p12a_loa_cigale_20260929.md), [field comparison scope](e2e_cfm_p12a_comparison_scope_20261009.md), [latest full-cap record](e2e_wholefield_fullcap_regeneration_20261009.md), and the dated entries in [SCIENCE_LOG.md](../SCIENCE_LOG.md). Older plan status paragraphs are not current authority.

## Timed main deck

Asset IDs refer to the companion production brief. Each slide has one principal visual and one spoken conclusion; notes below are prompts, not slide text.

| New slide | Time | Assertion-style title / purpose | Visual and spoken emphasis |
| --- | --- | --- | --- |
| S1 | 0:00–0:25 | **Learning the cosmic web** | Reuse a restrained version of the title image, name/affiliation and conference/date. Open: “The same sparse galaxy catalogue need not identify a unique underlying matter field. I want to infer the range of tidal environments compatible with what we observe.” |
| S2 | 0:25–1:15 | **We observe galaxies, not their dark-matter environments** | A01: mock matter/tracer comparison or clearly labelled schematic. One sentence on galaxy evolution. Small Paper-I citation/graph thumbnail, not its full confusion matrix. Distinguish the observation from the hidden target. |
| S3 | 1:15–2:30 | **Infer eigenvalues first; derive environmental probabilities** | A02: ordered triplet, threshold-count strip and two posterior examples. R=7 h⁻¹ Mpc, target epoch z=0.2, λth=0.2 visible. Explain width as model-conditional ambiguity, not simply model failure. Class probabilities come from joint triplet draws. |
| S4 | 2:30–3:50 | **Independent regions and phases change what “generalisation” means** | A03: spatial core/context and phase/OOF split diagram. Explain how random-node splits can reward interpolation within one field; overlapping context is not independent evidence. Freeze transforms/model/criteria before confirmation. This is the methodological advance to emphasize. |
| S5 | 3:50–5:00 | **Compress the observed field, then learn its conditional posterior** | A04: observed count/response fields → residual U-PATCH → out-of-fold summaries + deployable covariates → conditional FMPE → ordered eigenvalue draws. Show a small “Paper I: GAT classifier → current: protocol-selected estimator” lineage, not a GraphNet diagram relabelled U-Net. |
| S6 | 5:00–6:35 | **Mock validation supports a bounded per-galaxy claim** | A05: legible coverage diagnostic with the identity reference. Separate badges for original blind P12-A and exposed halo48 deployment checks. Headline 4.90 million rows / 10 registered gates for the original only; say “within registered tolerances”, not “perfectly calibrated”. Include one proper-score/information statement if the plot has room. |
| S7 | 6:35–8:05 | **DESI now has a provisional environmental-posterior catalogue** | A06: same positions shown as tracers, P(filament), and uncertainty/concentration. 5.44 million rows, 5.40 million supported. Identify observed/redshift-space plotting coordinates. Permanent “Provisional; real-data coverage unverified” label. State the high-z mismatch aloud. No live website demonstration. |
| S8 | 8:05–9:35 | **Environmental probabilities retain modest galaxy-property contrasts** | A07: two CIGALE panels, mass/z-standardized log sSFR and low-sSFR fraction, full footprint 0.20–0.30. Highlight −0.186 dex and +7.13 pp, with definitions and sampling-only intervals. “An application of the inferred probabilities, not evidence that they are calibrated on DESI.” |
| S9 | 9:35–10:45 | **Local uncertainty is not uncertainty in the whole web** | A08: marginal-versus-joint schematic plus at most one field diagnostic. Explain why regional fractions/connectivity require a spatial joint law. Current coupled-field models are informative but not qualified for these claims; full-cap tests are ongoing. Link naturally to the session's probabilistic filament work. |
| S10 | 10:45–11:20 | **Infer the environment—and validate the uncertainty you use** | Three short takeaways: continuous per-galaxy inference; independent mock validation and provisional DESI application; joint-field uncertainty remains a distinct validation problem. Keep contact/QR and a quiet A06 crop. Thank collaborators. Remain here for Q&A. |

Total scripted time: **680 seconds = 11:20**. Buffer: **40 seconds**. The chair's remaining three minutes are for questions. Checkpoints: start S5 by 3:50, S7 by 6:35, S9 by 9:35.

If behind: remove the optional movie, shorten S9 to its single distinction, then go to S10. Do not rush or omit the calibration/deployment caveat. If the field campaign has not produced qualified new results by the deck freeze, the default S9 already works unchanged.

## Theory to say, and theory to keep in backup

Use one central probability expression on S3:

\[
q_\phi(\boldsymbol\lambda_g\mid S_g(\mathcal G_{\rm obs}),\mathcal H_{\rm fid}),
\qquad \lambda_1\leq\lambda_2\leq\lambda_3.
\]

Here \(S_g\) is the actual information retained by the compressor and deployable covariates, not automatically the entire information content of the catalogue. \(\mathcal H_{\rm fid}\) includes the simulator, galaxy population, survey response, cosmology and target definition.

Suggested spoken explanation:

> “The widths describe the range of target environments compatible with the retained observations under our model. A broad posterior can be the correct answer to an underconstrained problem. But width is meaningful only after checking the inference procedure: it does not automatically include uncertainty in an incorrect galaxy-population model, or information lost by our compression.”

Show a simple 0/1/2/3 threshold-count strip. Keep the integral in backup:

\[
P(C_g=k\mid S_g)=\int\mathbf1\!\left[\sum_{a=1}^{3}\mathbf1(\lambda_a>\lambda_{\rm th})=k\right]
q_\phi(\boldsymbol\lambda_g\mid S_g)\,d^3\lambda.
\]

Do not classify the posterior mean or multiply the three marginal crossing probabilities. Knot/cluster here means the three-axis T-web category, not an independently identified halo catalogue. A gravitational-potential Hessian and a density Hessian are different definitions. Use the positive-Hessian T-web convention consistently; the old slide 4 minus sign conflicts with slide 5. With normalized potential \(\nabla^2\Phi_R=\delta_R\), \(T_{ij}=\partial_i\partial_j\Phi_R\) and \(\mathrm{tr}\,T=\delta_R\). Otherwise carry the normalization explicitly. Do not label trace simply “density” without defining overdensity and units.

The field-outlook backup can use:

\[
p(\{\boldsymbol\lambda_g\}_g\mid\mathcal G)\ne\prod_g p(\boldsymbol\lambda_g\mid\mathcal G),
\qquad
\mathrm{Var}\!\left(\sum_g w_g I_g\mid\mathcal G\right)
=\sum_{g,h}w_gw_h\mathrm{Cov}(I_g,I_h\mid\mathcal G).
\]

The covariance terms explain why independent galaxy draws cannot provide uncertainty on regional fractions or connected structures. A coherent-looking generated field is also not sufficient: its joint distribution must be tested. This is the natural connection to the conditional-field project, without absorbing all its unresolved methodology into a 12-minute Paper-II presentation.

If asked about Zel'dovich: flow-matching time is a generative transport coordinate, not cosmological time; learned transport is not automatically a Lagrangian gravitational displacement. Save a physical-dynamics comparison for discussion, not the main talk. Cite [Lipman et al.](https://arxiv.org/abs/2210.02747) for flow matching.

## Slide-by-slide audit of the existing Keynote

Reviewed all **37 slides** as rendered images, including the untitled backups, and the available presenter notes. Progressive slides are counted individually below. Review PDF was exported to temporary storage; **the Keynote was not edited**. PDF snapshots do not validate movie playback or every animation build; playback testing remains an assembly-stage check. In particular, slide 23's exported media frame is effectively blank.

Overall: the black background, white headings and mint/blue/magenta/gold class palette are reusable. The neon title glow is less legible than the body slides. Many old plots have journal-sized labels, large unused margins or several competing panels. Re-render from numerical sources; do not solve this by enlarging screenshots. The evidence base, not only the date, has changed.

| Old # | Current content observed | Disposition | Specific change / destination |
| --- | --- | --- | --- |
| 1 | Neon title, wedge imagery, portrait, QR and logos | **Update → S1** | Keep visual identity and one strong web image. Remove glow/italic title treatment; add Kavli IPMU and 26 Oct 2026. Larger name, fewer footer objects; QR can wait until S10. |
| 2 | TNG100 matter image, four environment labels, galaxy-evolution motivation | **Keep visual, compress → S2** | One motivating sentence is enough. Retain image credit and scale bar; distinguish illustrative TNG image from Abacus training/DESI application. |
| 3 | Adds Stage-IV surveys/ML motivation | **Merge into S2** | No separate click for generic survey motivation. Replace long text with “Sparse tracers → uncertain environment”. |
| 4 | Adds galaxy-properties/tidal-field connection and negative Hessian | **Replace/merge → S3** | Remove inconsistent sign and implication of a one-to-one property–tidal relation. Use normalized positive potential Hessian and explicit target scale. |
| 5 | Tidal equation, ordered eigenvalues, collapse-axis icons | **Keep/redraw → S3** | Best reusable explanatory element. Enlarge icons and labels; replace vertical “Density” ordering arrow with “number of eigenvalues above threshold”. The classes are not strictly non-overlapping density bins. |
| 6 | Adds threshold rationale | **Merge into S3** | Keep λth=0.2 and R=7 h⁻¹ Mpc visible. Avoid interpreting this phenomenological threshold as an exact collapse-time measurement. Cite Forero-Romero et al. 2009. |
| 7 | Adds claim other definitions are “less physical” | **Remove that claim; merge remaining diagram** | Different estimands are not a hierarchy of physical legitimacy. Particularly unsuitable in this session of topology, dynamics and reconstruction specialists. |
| 8 | Idealised skewer, three PDFs and class bar; discrete-information argument | **Update → S3 / A02** | Retain explanatory device but enlarge it. Label an idealised animation as a schematic. For numerical examples use current saved joint draws; no synthetic curves disguised as calibrated posteriors. |
| 9 | Adds “least certain, most informative” transition claim | **Merge; rewrite** | Say class assignments can be sensitive near the chosen threshold. “Most informative” and greatest posterior width at boundaries are not generally established. |
| 10 | Adds continuous-posterior conclusion; build | **Keep message, collapse → S3** | One slide with at most two reveals. Emphasize joint triplet draws → class probabilities; continuous inference does not make all discrete web definitions obsolete. |
| 11 | Galaxies rather than DM; inference expression | **Keep concept → S2–S3** | Use conditional-on-summary/model notation. This deserves more prominence than the subsequent architecture history. |
| 12 | Paper-I confusion matrix, article screenshot, QR | **Shrink to lineage inset / backup B1** | Update citation year to 2026. Keep matrix only in backup, with TNG300 mass selection and normalization. Article screenshot plus QR occupies too much space for the main talk. |
| 13 | Boundary-error explanation added to Paper I | **Merge into S3; matrix backup** | If retained quantitatively, support with a boundary-distance diagnostic; adjacency in a confusion matrix alone does not prove location at physical boundaries. |
| 14 | Classifier → graph-conditioned posterior | **Rewrite → S5** | Preserve classification-to-inference transition, but do not imply current production is the same GNN with a new head. |
| 15 | Galaxy-node/edge graph cartoon | **Backup B1 / small S2 inset** | Clear cartoon, but “why graphs?” is no longer a three-slide main section. Graph construction is history/context, not the current VAC input diagram. |
| 16 | Graph-metric/edge-length distributions by class | **Remove from main; optional B1** | Thin curves/tiny axes cannot be read at current size. Only regenerate if specifically needed for Paper-I questions. Correct “geometrc”. |
| 17 | Same graph metrics plus mock-training statement | **Replace → S4** | Use space for spatial/phase separation and out-of-fold summaries—the actual methodological advance. |
| 18 | GraphNet message passing → conditional normalising flow | **Replace → S5 / A04** | Draw actual U-PATCH + FMPE path with current channels/covariates. Remove irrelevant JAX logo and galaxy-image icon suggesting photometric image input. |
| 19 | Adds δ=(λ1, λ2−λ1, λ3−λ1) | **Replace; transform in B2** | This is not the canonical consecutive softplus-increment parameterisation. Use λ2=λ1+softplus(u2), λ3=λ2+softplus(u3); distinguish target support transform from a prior. |
| 20 | Adds amortisation statement to architecture | **Merge into S5** | Keep the useful “train once; condition separately for each galaxy” explanation. No third architecture build. |
| 21 | Truth versus posterior-mean histograms | **Replace → S6** | Matching population histograms is not calibration; posterior means can legitimately be narrower through shrinkage. Use coverage and a proper score, not distribution matching as proof of fidelity. |
| 22 | Old TARP and three rank histograms | **Regenerate → S6 / A05** | Preserve the idea, replace old model evidence. Separate original blind model and refitted halo48. Rank panels are too small; put detailed diagnostics in B3. Retain real-DESI caveat. |
| 23 | “Zero-shot” DESI fly-through; blank PDF media frame | **Replace → S7 / A06** | Current full-footprint map with static fallback. Say “simulation-trained inference applied to DESI”, not “all DESI systematics forward-modelled”. Revalidate playback; no essential result should depend on video. |
| 24 | Abacus/DESI posterior-mean histograms; “Abacus≈DESI≈True” | **Remove from main; replace by A09 in backup** | Similar histograms neither establish transfer nor supply DESI truth. Show actual selection/support mismatch by redshift if discussing domain shift. |
| 25 | Class-fraction bars titled “Validation on DESI” | **Replace → S7 caveat / B4** | Rename any retained plot “descriptive distribution comparison”. Predicted fractions matching a simulation do not validate real class assignments or calibration. Do not tune DESI to simulation fractions. |
| 26 | UMAP overlap and class-coloured embedding; “Validates forward-modelling” | **Remove** | Neither embedding overlap nor classes predicted from that same representation is independent validation. Also wrong encoder vintage. Not worth refreshing for a short talk. |
| 27 | Three old property panels versus tidal trace | **Replace → S8 / A07** | Use current CIGALE product and probability-weighted common-cell analysis, not the legacy wedge/property pipeline. Two panels maximum; define trace as overdensity if shown. |
| 28 | Same panels plus redder/trend assertions; problematic notes | **Merge into S8; rewrite notes completely** | Delete “errors are conservative because misclassification only dilutes signal”. Bias need not be simple attenuation, and omitted uncertainties do not make intervals conservative. Use descriptive associations, not DESI calibration or evolution. |
| 29 | Three old categorical property panels | **Merge into S8** | Replace hard-label presentation with probability-weighted current CIGALE summaries. Uniform class vocabulary/colour key. Keep log-sSFR and low-sSFR fraction only. |
| 30 | Broad mass-bin quenching curves | **Replace → S8 / backup B5** | Use common fine mass/redshift cells and matched support; show exact estimator and spatial-bootstrap caveats. Persistence within broad mass bins is not a clean causal environmental effect. |
| 31 | Summary with “future scale to larger areas”, Y5 statement, 3-D scatter | **Rewrite → S10** | Scaling is now technically done. Say provisional full-footprint product, bounded mock evidence, joint-field challenge. Do not promise a Y5 paper/release without current collaboration approval. Keep contact and uncluttered map. |
| 32 | Extra-slides divider | **Keep as backup divider** | Not spoken or timed. Organize backups by scientific question. |
| 33 | “Preliminary Science” repeats old property panels | **Replace → B5** | Current CG15/CG5 sensitivity, completeness and estimator, not another copy of superseded results. |
| 34 | Untidily packed corner plot and threshold sweep | **Update → B2, only if needed** | Split into a single threshold-sensitivity plot or one posterior example; identify old versus current model and fixed selection. Do not choose λth to force fraction agreement. |
| 35 | TARP parameterisation comparison | **Archive; replace by B3 if needed** | Old comparison is not a current head-selection result. Its “above diagonal = over-confident” annotation needs correction for the displayed empirical-coverage convention: above nominal ordinarily means overcoverage. Verify statistic before relabelling. |
| 36 | Real-DESI skewer/PDF/class-probability animation frame | **Regenerate, optional A02 movie** | Good reusable format if rebuilt from current saved draws, with frame identity, units, threshold and legible labels. Per-galaxy samples along a skewer are not a coherent field draw. |
| 37 | “DESI cluster-fraction recovery toward truth”; staged domain adjustments | **Remove entirely from scientific talk** | Abacus fraction is not measured DESI truth; closing that fraction gap is not demonstrated recovery. Replace with evidence-based input-mismatch diagnostic if asked. |

## Backup deck: six questions, not six old results

- **B1 — What changed since Paper I?** Published TNG GAT classification, then common-protocol encoder comparison, then current U-PATCH/FMPE. Historical confusion matrix clearly dated.
- **B2 — What exactly is the target?** Potential-Hessian convention, smoothing/epoch, host-halo versus observed-cell coordinates, increment map, joint-draw threshold probabilities. Eigenvalues do not supply directions/spines.
- **B3 — What does calibrated mean here?** Original blind contract and full diagnostics; halo48 lineage and separate exposed-phase evidence; proper scores, conditional strata, dependence-aware uncertainty.
- **B4 — What limits DESI deployment?** Selection n(z), support/nulls, redshift caveats and immutable model/version metadata. No claim all inputs are public/releasable without checking actual release provenance.
- **B5 — How robust are the property trends?** CG15/CG5 on identical rows/cells, class-weighting formula, bootstrap scope, matching completeness, SED caveats. Include distribution overlap rather than only tiny-error trend lines.
- **B6 — Why conditional fields / how does this relate to Thiele?** Local marginals versus joint spatial quantities; matched-estimand requirement; field diagnostics; catalogue geometry versus tidal environments. New field result only if its stated gate has genuinely passed.

## Assembly and freeze checklist

- Recheck SCIENCE_LOG and NERSC artifact receipts before inserting final figures. Current local snapshot is not a claim about future conference-day status.
- Freeze the primary S6/S7/S8 figures before chasing the latest field result. A failed or unfinished field experiment must not make the talk impossible to deliver.
- Each result panel carries model/version, mock versus DESI status, selection and target convention. A short visible badge plus detailed notes is sufficient; avoid tiny caveat paragraphs.
- Confirm collaboration permission for displaying Loa/CIGALE results and for distributing slides. “DESI” does not by itself mean every source file is publicly released.
- Rehearse three times with actual builds/video. At most one short optional movie in the main deck; static fallback embedded next to it in the presenter file.
- Export a slide-only PDF with useful poster frames, verify all ten main slides at 1280×720, and test actual Keynote media playback. Do not use a browser-dependent live visualization.
- Keep the original 37-slide duplicate unchanged until the new outline and assets are approved; this review itself did not modify it.

## Compact reference set for slide footers / notes

- [Kololgi et al. 2026, Paper I](https://doi.org/10.1093/rasti/rzag025): S2 or B1.
- [Forero-Romero et al. 2009, T-web threshold definition](https://arxiv.org/abs/0809.4135): S3/B2.
- [Leclercq, Jasche & Wandelt 2015, Bayesian dynamical web](https://arxiv.org/abs/1502.02690): S3/B6; credit probabilistic-web precedent.
- [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): S5/B6.
- [Lemos et al. 2023, TARP](https://proceedings.mlr.press/v202/lemos23a.html): S6/B3. A finite diagnostic pass is not a proof of universal conditional accuracy.
- [Thiele 2026, Bayesian Cosmic Void Finding with Graph Flows](https://arxiv.org/abs/2602.14630), and Tastet-led filament work as described in the [conference programme](https://drive.google.com/file/d/1_C6Hndw4oT53IBn4h-3NPx93-TKWgcGV/view): B6. Do not conflate the two works.

Use short author/year citations visibly and full links in notes. The timing and one-visual-per-message structure follow the scientific-slides guidance; this document is a plan and review, not a newly generated slide deck.
