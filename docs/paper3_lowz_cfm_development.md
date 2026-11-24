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

### Real DESI application (not a validation against real matter truth)

The Paper 3 deliverable includes an application of the frozen two-stage CFM
to the Loa BGS_BRIGHT observations. Use the v2.1 full HPmapcut catalogue,
not a clustering catalogue or the Paper 2 VAC, and the matching 18 full
randoms. The registry is `configs/p10_response_sources_v1.json`
(`desi_candidate`). All 19 paths and sizes matched the registry at inventory;
this was not a fresh payload hash or a validated production response.

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
