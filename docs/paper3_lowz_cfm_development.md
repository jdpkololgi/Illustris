# Inferring the matter field and cosmic web from BGS galaxy positions

## Methods and development results — working manuscript

Galaxy surveys measure an incomplete set of luminous tracers of the underlying
matter field. Even with a known selection function, several distributions of
matter can account for the same observed galaxies. We ask whether draws from
a conditional reconstruction assign realistic uncertainty to the local density
and tidal environment of a cell, rather than whether one reconstruction
reproduces every filament of a simulation.

We condition a two-stage conditional flow matching (CFM) model on an observed
mock bright-galaxy catalogue and its survey response. A wide-field model draws
a coarse density; a fine 3D U-Net draws a residual conditional on that same
coarse realization. Both stages start from fresh noise for each ensemble member.
Windows within a cap share a spatial state. The two caps are sampled separately.
The fine field is positive and preserves the sampled coarse-cell masses. The
target is matter density in units of the cosmic mean at Gaussian smoothing
radius 7 Mpc/h, derived from the Abacus z=0.2 snapshot; this is not a test of
cosmic evolution along a past light cone. Each draw is extended using the
frozen training-only exterior model and transformed to three ordered tidal
eigenvalues. The number above 0.2 gives the void, sheet, filament or knot class.
Mock matter is used for supervision in training and as truth in scoring, never
as a conditioning input during inference.

Our development ensemble has 32 draws per cap and phase (192 cap fields).
We score registered, supported cells against independent full-box targets.
The simulation phase, not the voxel, is the replication unit. These cases have
been used to develop the method and do not constitute blind confirmation.
We compare truth with individual draws for morphology; the ensemble mean is a
point estimate and the ensemble SD measures location-wise uncertainty.

| Metric | Split type | Phase or sky | Observed redshift | Artifact |
|---|---|---|---|---|
| Cell-weighted fair CRPS, density and ascending eigenvalues: 0.1196, 0.0474, 0.0508, 0.0620 | Development, supported evaluated cells | ph006/012/013, both caps | [0.01,0.55) | A |
| Median draw/truth density contrast ratio: 0.996 (95th minus 5th percentile); 0.983 (spatial SD) | Development, median over draws within each registered core pair, then median of pair ratios | Same three phases, both caps, 48 pairs | [0.01,0.55) | B |
| Mean reconstructed minus true density, by gridded count [0,1), [1,2), [2,4), [4,∞): +0.0029, −0.0066, −0.0066, −0.0096 | Development, pooled supported cells | Same three phases, both caps | [0.01,0.55) | B |
| Loa/mock counts: 0.962 SGC, 1.030 NGC | Input transfer check, common supported pixels | Loa v2.1 vs Kibo altMTL ph006 | [0.01,0.15) | C |
| Loa/mock counts: 1.964 SGC, 1.918 NGC | Same input check | Same | [0.45,0.55) | C |

The contrast test asks whether the draws fluctuate enough, not whether the
right structures occur at the right positions. A median near one does not
establish a joint posterior or survey-scale power-spectrum recovery. Gridded
counts are CIC-deposited and can be fractional: these are not exact galaxy
positions. Equal-area cells and galaxy-weighted samples answer different questions.

The Loa check applies the observed science cut to the full LSS BGS_BRIGHT
HPmapcut product, not a clustering sample or the FastSpecFit low-z extract.
We select the BRIGHT bit, finite observed redshift and ZWARN=0 in both data
sets. The observed sample additionally requires GALAXY and DELTACHI2≥25;
these are not mock observables. We count unweighted galaxies on the common
NSIDE=256 random-occupied footprint in each cap and photometric system using
random index 0 and GOODHARDLOC. This is a check of delivered counts, not an
absolute completeness measurement. It does not correct subpixel area, prove
the same galaxy–matter relation, or establish coverage on real data.

Evidence: Scratch `abacus/e2e_field_v3/` products
A: `lowz_calibration_61647276/CALIBRATION.json`;
B: `lowz_finish_dev_61811977/FINISH_COMPLETE.json` (also paired slice PNGs);
C: `lowz_loa_input_check_61811977/INPUT_CHECK.json` (source paths, selected
row traces, common pixels, counts). [Development calibration figures](figures/lowz_cfm_calibration_61647276/calibration.pdf).

### A matched local count benchmark

We compared the CFM with a probabilistic, single-cell regression. The only
predictors are Gaussian-smoothed CIC galaxy counts (7 Mpc/h), random-catalogue
support and radial redshift. A fixed ridge model predicts log density, the
central eigenvalue and the two log gaps. Resampling their *joint* residual
vectors preserves positivity and eigenvalue ordering. The fit uses 16 fixed
domains in each of the 20 training phases. Residuals come from predicting each
training phase with a fit to the other 19; the final predictor and residual bank
were fixed before development scoring. No matter truth is used for prediction.
This is an observation-based benchmark for **marginal** forecasts, not a
classical solver that generates coherent cap-wide fields.

Results below are cell-weighted over exactly 344,302 supported locations,
ph006/012/013, NGC and SGC, 0.01≤z<0.55. Columns are density and the three
ascending eigenvalues. Each method provides 32 draws. Artifacts D and E.

| Metric | Density | λ1 | λ2 | λ3 |
|---|---:|---:|---:|---:|
| CFM fair CRPS | .11962 | .04736 | .05077 | .06196 |
| Smoothed-count benchmark fair CRPS | .14381 | .06073 | .06367 | .08386 |
| CFM improvement, 1 − CRPS(CFM)/CRPS(benchmark) | 16.8% | 22.0% | 20.3% | 26.1% |
| CFM empirical 5th–95th coverage | .869 | .850 | .855 | .847 |
| Benchmark empirical 5th–95th coverage | .857 | .865 | .853 | .857 |
| CFM mean interval width | .665 | .258 | .273 | .319 |
| Benchmark mean interval width | .793 | .339 | .349 | .462 |

The CFM gains CRPS for all four fields in each phase and each redshift shell,
and has lower pooled fair Brier scores for all four web classes. Its intervals
are narrower; it does not uniformly give higher coverage. The 5th and 95th
percentiles are linearly interpolated from only 32 draws, so 90% is the target
*posterior probability*, not an exact finite-ensemble repeated-sample
expectation. Rank-based intervals in the calibration pack retain their own
nominal levels. The three phases are the replicates; no voxel-based significance
claim is made.

### Draw morphology and regional tidal shear

The local density distributions are measured on exactly the same pixels in
truth and every draw. In all three phases, the highest measured local Fourier
bin (0.363–0.464 h/Mpc) has a median pair-level draw/truth power ratio above
one: 1.22, 1.22, 1.11 respectively. Thus near-unity integrated contrast does
not establish agreement at every scale. See [PDFs and masked local spectra](figures/lowz_cfm_finish_61848915/DENSITY_SHAPES.pdf)
and [matched single-cell score plots](figures/lowz_cfm_finish_61848915/MATCHED.pdf).

As a secondary check we reproduced the historical six-component regional
statistic in the original geometry: inner 12³ density and mean shear spread
and curvature within each 16³ core of a pair. This tests modeled core cells,
not the supported-voxel single-cell estimand. For regional shear spread
(mean λ3 minus mean λ1)/√2, mean signed error in posterior SD units was
−0.97, −0.96 and −1.32 on ph006/012/013. Empirical 5th–95th coverage was
24/32, 25/32 and 14/32 cores. This is a real limitation of the development
ensemble, not evidence for or against a calibrated joint field on DESI.

Evidence under the same Scratch root: D: `lowz_count_reference_61848915/`
(`FIT.json`, `TRAIN_ONLY.npz`, `REFERENCE_COMPLETE.json` and
`MATCHED_PERCENTILE90.json`); E: `lowz_finish_scorecard_61848915/SCORECARD.json`
(paired phase, shell, cap and support summaries, regional draws and truth);
F: `lowz_dev_density_shapes_61848915/SHAPES.json` (includes overflow counts).

### What remains when the density is known?

For each development phase and cap, we replaced the modeled density by true
R7 density on exactly the finite production cells. We kept the original
support, learned exterior prescription, 32-cell padding and T-web operator;
we did not supply true matter in the unmodeled exterior. All 48 registered
core pairs matched the bound target density on supported cells before applying
the operator (maximum absolute discrepancy zero).

| Signed bias in posterior SD units | λ1 | λ2 | λ3 |
|---|---:|---:|---:|
| CFM ensemble mean minus full-box truth | +.161 | −.008 | −.143 |
| True-density control minus full-box truth | +.107 | −.001 | −.087 |
| CFM ensemble mean minus true-density control | +.054 | −.007 | −.056 |

The denominator is the same production posterior SD at each cell in all three
rows; means cover the 344,302 supported registered cells at 0.01≤z<0.55 in
ph006/012/013, both caps. The regional shear-spread bias likewise decomposes
as −1.084 = −0.723 − 0.361 SD (average over 96 historical cores). Thus a
substantial part of the compressed shear remains with true input density.
This diagnostic combines the finite survey geometry, exterior draw and tidal
operator, with one exterior seed per cap/phase. It neither identifies a
unique source nor licenses subtracting a bias from posterior draws.
Evidence: `lowz_dev_oracle_density_v1/CLOSURE_READOUT.json` and its six
phase/cap receipts under the same Scratch root.

### Frozen blind confirmation on ph018 and ph019

The two independently held-out simulation phases contain 109,144 and 121,818
supported evaluation cells, respectively, in the 16 preregistered core pairs
per phase. Predictions were frozen before opening matter truth: 32 shared-canvas
draws per phase and cap (128 total), and 32 draws from the training-only
smoothed-count benchmark per cell. The comparison below is cell-weighted
over both caps. The targets are the R7 density and ordered tidal eigenvalues
from the full periodic simulation box, not the tides of a truncated truth patch.

| Blind metric | Density | λ1 | λ2 | λ3 |
|---|---:|---:|---:|---:|
| CFM / benchmark fair CRPS (lower is better) | .1182 / .1465 | .0461 / .0604 | .0502 / .0634 | .0610 / .0836 |
| Improvement over benchmark | 19.3% | 23.6% | 20.7% | 27.1% |
| CFM empirical 5th–95th coverage | 87.83% | 86.93% | 86.21% | 85.22% |
| Mean 5th–95th width, CFM / benchmark | .679 / .794 | .264 / .342 | .278 / .349 | .325 / .462 |
| CFM order-statistic coverage (nominal 27/33 = 81.82%) | 84.89% | 83.99% | 83.16% | 82.05% |

The CFM improved all four CRPS values and all four class-specific fair Brier
scores **on each phase**. Pooled Brier reductions relative to the same count
benchmark were 29.5%, 24.2%, 19.9%, and 16.6% for void, sheet, filament
and knot. In the 54,522 cells at 0.01 ≤ z < 0.15, CRPS reductions were
10.5%, 37.2%, 29.2%, and 33.4%; density rank-interval coverage was 86.23%
against 81.82%, consistent with cautious low-z intervals on development.
The empirical interpolated 5th–95th interval from 32 draws does not have
exact 90% frequentist coverage even under exchangeability. Report its coverage
and width alongside the finite-ensemble rank diagnostic; do not interpret
the gap to 90% alone as underdispersion. Neither phase is used for tuning.
These are marginal results from **two independent phases**, not 230,962
independent simulations, proof of joint calibration, or coverage on DESI.

Two differences matter for interpreting a catalogue. First, pooled mean
predicted versus true class fractions were 53.01/52.28% (void),
35.34/36.34% (sheet), 10.57/10.53% (filament) and 1.07/0.85% (knot).
Thus a better Brier score does not imply exact class abundance, particularly
for rare knots. Second, below z=0.15 the mean density bias (in units of
cosmic mean density) was −0.0012 with equal cell weights but −0.0196 when
weighted by gridded galaxy counts. The latter gives more weight to populated
cells and is not an exact-galaxy-position test.

[Phase/shell tables, including galaxy-count weighting](evidence/lowz_cfm_blind_score_table.json) ·
[CRPS, coverage and widths by shell](figures/lowz_blind_61911489/marginals_by_shell.pdf) ·
[Ranks and T-web reliability](figures/lowz_blind_61911489/ranks_and_web_reliability.pdf).
Full stratification: `lowz_cfm_blind_confirmation_v1/blind_score_61911489/READOUT.json`
under the campaign Scratch root.

### Real DESI application (not a validation against real matter truth)

The Paper 3 deliverable includes an application of the frozen two-stage CFM
to the Loa BGS_BRIGHT observations. Use the v2.1 full HPmapcut catalogue,
not a clustering catalogue or the Paper 2 VAC, and the matching 18 full
randoms. The registry is `configs/p10_response_sources_v1.json`
(`desi_candidate`). All 19 paths and sizes matched the registry at inventory;
this was not a fresh payload hash or a validated production response. Subsequent
preparation verified all 19 source hashes and produced 7,835,655 context rows.
Of these, 7,813,715 lie in the science range below; all fell on supported
pixels of the 18-random map in the appropriate cap. This is **not** a measure
of spectroscopic completeness. Positive-redshift context extends to z < 0.60
with the registered sentinel exclusion; it is not the science sample.

| Science redshift | Loa NGC galaxies | Loa SGC galaxies |
|---|---:|---:|
| [0.01,0.15) | 1,715,380 | 661,922 |
| [0.15,0.25) | 1,981,867 | 799,947 |
| [0.25,0.35) | 1,286,073 | 521,105 |
| [0.35,0.45) | 504,806 | 205,287 |
| [0.45,0.55) | 95,587 | 41,741 |

Source: `lowz_loa_preparation_61911547/OBSERVATIONS_PREPARED.json` under
the campaign Scratch root. Transfer comparisons restrict both catalogues
to the same angular domain; the totals above are for Loa alone.

Both Loa responses passed development parity and count conservation. Across
ph006/012/013, Loa/mock ratios in the five successive shells are
1.024–1.035, 1.024–1.037, 1.093–1.113, 1.274–1.303, 1.876–1.918 (NGC)
and 0.917–0.962, 1.002–1.035, 1.068–1.091, 1.244–1.279, 1.963–2.047
(SGC). These are phase ranges on matched cap/photometric-domain pixels
(184,726 NGC and 81,922 SGC), not error bars. The high-z excess recurs
against all three mocks; its cause remains unresolved. Prioritize the
lowest two shells for real-data interpretation without changing conditioning.
[Counts by phase and cap](figures/lowz_loa_transfer_61911856/counts.json) ·
[Count-ratio plot](figures/lowz_loa_transfer_61911856/counts.pdf).

Keep the observed quality cut above, both caps, the training-only selection
and normalizer, the learned exterior and the same R7 operator. Apply the
model to each cap independently. Compare observed counts, angular support
and expected counts with the corrected mocks by cap and redshift shell
before interpreting a real-data posterior. Keep high-z discrepancies visible;
do not thin Loa or refit on its counts to force agreement. A low-z science
mask restricts the **reported locations**, not the surrounding observations
used by the trained model, unless a different conditioning scheme is tested.

Show representative density draws plus density uncertainty and T-web class
probabilities, with support and redshift masks. Label empirical coverage and
reliability as **mock** results. The real galaxy catalogue supplies conditioning,
not an observed matter field with which to establish DESI coverage.
