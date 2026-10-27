# Tokyo figure revisions — self-contained NERSC prompts

Prepared 18 October 2026. These prompts describe the desired figures without requiring access to Keynote. They are requests, not claims that figures have been generated. Reuse the frozen source artifacts for the current FMSU talk. Do not retrain, silently change the sample, or fabricate illustrative numerical results. If provenance cannot be established, report the missing source rather than guess.

## Shared delivery and styling contract

For each requested figure deliver: a 1920×900 black-background body graphic (the slide title is added in Keynote), a vector PDF or SVG where practical, the plotting script, and a short receipt listing input paths, sample/filter, target definition, coordinate convention, units, model/checkpoint and any changes from the previous figure. Provide a full 1920×1080 preview with the suggested title separately. Do not bake two titles into the body graphic.

Use the existing FMSU palette, sampled from its plotting theme: void mint, wall blue, filament pink, cluster gold; white main text, light-grey secondary text, black background. Use IBM Plex Sans if installed, otherwise report the fallback. At the 1920-pixel output size aim for labels 32–38 px, ticks at least 28 px, secondary receipts at least 24 px. Keep 70 px margins and generous panel gaps. Use proper h^-1 Mpc versus Mpc labels, dimensionless eigenvalue labels, and colourbars with readable units/ranges. Inspect the output at half size. No overlapping labels or legends covering data. Export optional build layers with identical canvas size and positions, without recomputing data between layers.

## A — Paired simulation branches (priority: useful replacement)

Purpose: replace the four eigenvalue/class maps in the main talk with a clearer answer to “Why do simulations enable SBI?” The maps can remain backup. Suggested slide title: “Simulations give us both sides of the problem”.

Create a clean diagram, no scientific scatter data needed. Top centre, x≈960/y≈100: **Abacus matter + halos**. Split down two branches with thin grey arrows:

- Left, centre x≈480: **Tidal calculation** → **Known local eigenvalues**. Small subtitle: “Smoothed matter field; host-linked target”. Show λ_g = (λ_1, λ_2, λ_3). Use the actual target scale R = 7 h^-1 Mpc only after confirming the current artifact convention. Targets are associated through host linkage; do not draw an arrow implying a field measured directly at the observed redshift-space position.
- Right, centre x≈1440: **Galaxy + observing model** → **Mock observed context**. Small subtitle: “Galaxy population, selection and redshift-space positions”. Include fiducial-distance conversion in the receipt; avoid implying that conversion removes RSD.
- Bottom centre y≈740: join the branches with a bracket labelled **Paired examples: context + known local target**. Underneath, small text: “From the same simulated realisation”.

**Stop at paired examples.** Do not draw a direct arrow into an exact posterior p(λ|D). The next talk slide introduces learning q_phi. A small “Used to train and test inference” footer is sufficient. Use modest mint/blue outlines, no decorative icons necessary. Return a single static graphic and, optionally, three layers: parent/branches, branch outputs, paired-example bracket.

## B — Paper I: separate inputs, labels and probabilistic classification (optional)

Purpose: summarise Paper I in about 35 seconds, then motivate continuous environmental inference. Suggested title: “From web labels to continuous environments”. The current slide contains a large publication banner/QR, a graph-network diagram and six small UMAP panels. Replace those with a clean input/label diagram plus at most two larger embedding panels.

Left third: **Galaxy-derived graph information → classifier → class probabilities**. A separate small box **Simulation tidal field → training labels** connects to a “supervision” annotation, not the observed input stream. Do not imply tidal information is available at DESI inference time. Check the actual Paper I input definition before calling features purely geometric or adding galaxy properties.

Right two-thirds: reuse the published/saved Paper I embedding, not an Abacus posterior embedding presented as Paper I. Prefer one projection coloured by true T-web class. Optionally pair it with the **same coordinates and same objects** coloured by class-probability Shannon entropy. If using entropy, state whether it is in nats, bits, or normalised by log(4); label the colourbar accordingly. Do not call entropy an error rate. Do not fit a new UMAP just for appearance. Use simple “UMAP 1” and “UMAP 2” axes and avoid assigning physical meaning to these coordinates. If no matching embedding and entropy arrays exist, deliver only the verified class panel and note the limitation.

Bottom text (two short lines):

“Class probabilities describe uncertainty in a web label”

“Next: a joint distribution of continuous tidal eigenvalues”

Small citation: Kololgi et al. (2026), RASTI, doi:10.1093/rasti/rzag025. No large article screenshot or QR. No claim that overlap proves irreducible uncertainty, errors, or physical web boundaries. Deliver panels separately as well as the composition so Keynote text can remain editable.

## C — One context, several plausible environments (priority: essential)

Purpose: illustrate a model posterior supporting more than one class, without implying that three local eigenvalues generate a whole observed catalogue. Suggested title: “One galaxy context, several plausible environments”. Current figure has two triplet cards on the left with arrows into a galaxy map on the right; reverse the conceptual reading by rebuilding as below.

Source: recover the same evaluated mock galaxy used in the current ambiguity figure and the left example of the two-galaxy posterior plot. The current talk describes a ph006 example with 512 saved joint posterior triplets, approximately P(void)=40.8% and P(wall)=59.2%. Verify the exact object identifier, phase, model, draws, target ordering, and threshold before rendering. Do not select a different example merely to recreate the percentages. If unavailable, report that fact and propose a clearly identified replacement with its own probabilities.

Layout:

1. **Left 45%:** the selected galaxy context as a spatial scatter plot. Retain the original verified selection/window. Surrounding galaxies subdued grey; target white or gold with a thin ring. Header “Observed mock context”. Label both axes and units; state whether this is redshift space from z_obs and a fiducial cosmology. Do not label it real space just because coordinates are comoving. Show a scale bar only if physically converted correctly. No posterior label beneath this point map.
2. **Middle 10%:** one left-to-right arrow labelled “Infer under model M”. This is an inference arrow, not a causal simulation arrow.
3. **Right 45%:** header “Plausible local tidal environments”. Two cards, mint void and blue wall. Each shows one **actual saved joint draw** with λ_1≤λ_2≤λ_3, labelled “Example posterior draw”; print sensible precision, not excessive decimals. Show the class name and, separately, **P(void)=40.8%** / **P(wall)=59.2%**, only if confirmed by the exact saved draws. A compact stacked class-probability bar can replace repeated percentages. Include other classes if nonzero rather than silently renormalising to two. Annotate λ_th=0.2 if confirmed. Derive each draw's class by counting its eigenvalues above threshold, never by combining independently selected marginals.
4. Bottom sentence: **“One observed context can support more than one tidal class.”** Footer: **“Mock posterior example · triplets are example draws; percentages summarise classes.”**

The probabilities refer to class regions of a continuous joint distribution, not probability masses on the displayed triplets. Do not label this “two universes producing exactly the same catalogue”; it is an illustration of the fitted posterior, not an independently simulated degeneracy experiment. Do not add a reconstructed matter-field image without a genuine compatible field sample.

Deliver static full figure and optional two build layers: observed context first, arrow/cards second. Also deliver each component separately, with shared positioning, for editable Keynote assembly. In the receipt supply the exact class counts out of 512 and the two chosen draw indices. Match the later posterior-example slide's rounding or explain the difference between 40.8/59.2 and rounded 41/59.

## D — TARP wording only (optional; current figure is acceptable)

Do not recompute or improve the curve. Preserve the saved P12-A halo48 ph006 diagnostic with 50,000 mock galaxies, its unweighted joint-parameter convention and its existing bootstrap maximum-deviation envelope. This sample was held out from training but already inspected in evaluation; do not relabel it as a new blind test.

Suggested title outside figure: “Do the probabilities mean what they say?” Preserve a small “Joint-posterior check in mocks · TARP” subtitle. The existing near-diagonal plot can otherwise remain unchanged.

If cosmetic revision is requested, first check the implementation's exact x/y convention. Where x is assigned probability and y is the observed fraction of truths inside the corresponding random-reference regions, use:

- x: **“Probability assigned to test region”**
- y: **“Fraction of mock truths inside”**
- ticks: 0%, 20%, 40%, 60%, 80%, 100%
- legend: “Mock test”, “Perfect agreement”, “Bootstrap reference envelope”.

If the implementation instead plots complements, transform both consistently or retain the original convention and explain it—do not merely replace labels incorrectly. Keep the identity diagonal and data limits identical. Preserve the envelope's actual construction: 95th-percentile maximum deviation across the curve, **not a pointwise confidence band**. Put its technical explanation in a short receipt/notes caption rather than a large warning block. Do not claim that a shaded region guarantees calibration.

Readable footer: “50,000 mock galaxies · held out from training”. Receipt: ph006 already exposed during evaluation; joint test; weighting; reference-point and bootstrap protocol; source paths. Citation: Lemos et al. (2023), https://arxiv.org/abs/2302.03026.

An 80% explanatory example in narration is fine, but do not mark an empirical point as exactly 80% unless the saved result supports it. No new numbers or qualitative claim of exact calibration on DESI.

## E — Coordinate audit and corrected captions (priority: accuracy)

The user confirms that some displayed galaxy panels have RSD and were converted from observed redshift to comoving distance using DESI fiducial cosmology. Existing simulation figures nevertheless contain embedded text “real space” and “matched real-space slice”. Resolve this from plotting inputs; do not infer it from axis units.

Locate the current FMSU figures: (i) four panels of the three tidal eigenvalues plus T-web class, (ii) matter panel → galaxy/observing-model box → selected mock-galaxy panel, (iii) ambiguity-example context. For each identify:

- coordinate columns actually used (z_obs/RSD redshift versus cosmological redshift or simulation Cartesian positions);
- whether peculiar velocities were added or removed, along which line of sight;
- fiducial cosmology and distance conversion, including h convention;
- field-grid coordinates versus galaxy coordinates, projection, slice centre/depth and selection;
- whether target assignment is host linked and whether paired panels really show the same spatial locations or only corresponding populations.

If the galaxy panel uses s=chi_fid(z_obs)n_hat, label **“Redshift-space galaxy positions (fiducial comoving coordinates)”**. If its neighbouring field is evaluated on an unshifted simulation grid, label that panel **“Real-space simulated tidal field”**. Do not call their overlay perfectly spatially matched without verifying the mapping. Distinguish paired galaxies/targets from a tidal field reconstructed in observed coordinates.

Regenerate only captions/labels that the audit shows to be wrong; preserve data and styling. Return a compact provenance table and corrected figures. If field and galaxy panels cannot support the current matched-slice comparison, say so and use the branching schematic A in the main talk rather than manufacturing apparent alignment.

## Acceptance checklist

- Every numerical mark comes from verified saved data; cosmetic changes are distinguished from new analyses.
- Input/label and inference/forward-generation arrows cannot be confused.
- Coordinate and unit labels describe actual inputs, not assumptions based on appearance.
- Class probabilities come from joint draws; entropy is not called an error map.
- Posterior estimate, mock validation and real-DESI transfer remain distinct.
- Text remains legible at half-size; title space, margins and colours match the black Keynote theme.
- The current three-slide ending needs no new field-performance figure or additional science claim.
