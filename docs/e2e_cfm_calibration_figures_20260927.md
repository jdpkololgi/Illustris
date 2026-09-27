# Mixed-model calibration figures and architecture

Source model: coarse13312/fine26624 EMA, seeds17/29 separately. Six exposed
non-training phases012–017,16anchor pairs per phase,32draws per anchor.
018/019 remain sealed. 012–015 influenced development;016/017 were evaluated
with the mixed choice frozen. None of these are real DESI data.

## Figures

- `figures/cfm_calibration_20260927/sbc_ranks.png` and `.pdf`:
  saved randomized truth-rank histograms for core/block/fine-sensitive region
  mass, density, tidal eigenvalues and eigengaps. Each phase and seed is shown.
  Histogram height is relative to a uniform rank distribution (ideal=1).
- `figures/cfm_calibration_20260927/tarp_regions.png` and `.pdf`:
  TARP random-point distance coverage for the2core,8block and8fine-sensitive
  regional mean-density vectors. A diagonal curve is the calibrated null.
  These are joint summary-space tests, not full-field TARP or T-Web class tests.
- `figures/cfm_calibration_20260927/model_dataflow.png` and `.pdf`:
  code-derived data flow, including observation context, stochastic hierarchy,
  joint fine generation, training/inference conditioning difference, and exact
  positive block-mass-conserving decode. Pilot classes use threshold0, not the
  production VAC threshold0.2. Spectral alpha0.30 weights the training loss;
  it is not an additional whitening of coordinates at generation time.

## TARP method and limitations

Reference: Lemos et al.(2023),
https://proceedings.mlr.press/v202/lemos23a.html;
https://tarp.readthedocs.io/en/latest/.
For each anchor and random reference, compare Euclidean distances from that
reference to all32posterior vectors and to the true vector. In physical mean
delta units, references are independently N(0,0.25^2 I),64reference repetitions
per anchor, seed71. No scale/reference fit to held-out truths. Randomization
uses (number of strictly smaller distances + U*(ties+1))/33. This finite-draw
version gives a uniform null rather than treating an empirical discrete CDF as
continuous. The plotted ECP is the fraction of these ranks <= credibility,
averaged over references and anchors. Random references do not add simulations.

No iid confidence bands, hypothesis-test p-values, or pass/fail claims:
only6phases,16selected correlated anchors each; fine regions strongly overlap;
training seeds share truths. Reference families, Euclidean metrics, selected
summaries and finite sample size limit power. Passing these curves cannot prove
the whole conditional field posterior correct. Marginal ranks alone cannot
establish joint dependence; TARP on these summaries likewise does not cover all
spatial dependence. Formal prior-predictive SBC would require an appropriately
sampled independent simulator test ensemble, not these stratified mock anchors.

## Provenance and checks

Script: `workflows/sbi/e2e_cfm_calibration_figures.py`.
Source summary hashes are recorded beside figures in `sources.json`.
Extraction verifies every selected case and chunk checksum and reproduces saved
regional means and standard deviations. It reads posterior draws only; truth
vectors come from the already exposed case scores. The compact cache lives at
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_additional_20260926_v1/regional_calibration_cache.npz`.
Figure curves are also exported to `tarp_curves.json`.
Regional implementation matches authoritative evaluator at1e-14 tolerance.
Gaussian calibrated toy maximum ECP deviation0.0082 on2048test cases; explicit
underdispersion and tie cases are included in unit tests.

User approved CPU-only30minute extraction; no new posterior draws or training.

Completed as58949424 on nid004154 in51seconds, exit0; allocation released.
All192cases extracted and saved regional moments reproduced. Three unit tests
passed in0.597s. All three PNGs visually inspected; diagram connector overlap
fixed. Graphify refresh exceeded5second login cap; index refresh incomplete.

## Scientific reading

Density rank histograms are near uniform, while regional ranks are noisy and
phase-dependent; ph017 has excess high truth ranks for core/fine-region masses.
Fine-region TARP at credibility.90 is.780/.801(ph012),.952/.946(ph013),
.909/.857(ph014),.871/.889(ph015),.949/.949(ph016),.767/.782(ph017), seeds17/29.
This is random-point credible-region coverage, NOT the central interval coverage
previously tabulated. Block-vector curves are visually closer to the diagonal.
No significance claim: these are16anchors per phase, not independent references.

Changing only reference sd from.25to1.0 gives fine-region ECP90
.822/.841 for012 and.821/.854 for017. These remain below.90 but magnitudes
are reference-dependent; no unique scalar calibration error is inferred.
The max deviations for014 are particularly reference-sensitive (.094/.132
versus.035/.047). Results in `reference_sensitivity.json`, produced with the
same `tarp_curves` function, seed71, fine-region indices10:18, width1.0.
No tuning or new confirmation opening followed these exploratory diagnostics.
