# Paper 3: what a coherent CFM draw recovers

Frozen seed-17 observation-conditioned coarse and fine CFM. Both caps; held-out ph018 and ph019; 32 draws per cap per phase; matched registered cells, 0.01 ≤ z < 0.55. Truth is for scoring only. R7 T-web threshold 0.20. Two independent phases, not 230,962 independent simulations.

## Density power and spatial alignment

Matched 32×16×16-cell windows, 6.766 Mpc/h spacing. Subtract each field's supported-cell mean and zero unsupported fluctuations. Compute FFT auto-power ratios and normalized real cross-power against truth for each draw, then mean over 32 draws and median over registered pairs by phase. Not a deconvolved full-cap spectrum, not power of the posterior mean. Identical windows mix modes in all fields.

| k (h/Mpc) | ph018 power ratio | ph019 power ratio | ph018 cross correlation | ph019 cross correlation |
|---|---:|---:|---:|---:|
| .025–.100 |1.111|.970|.841|.868|
| .100–.200 |.990|1.002|.724|.761|
| .200–.300 |1.054|1.050|.662|.636|
| .300–.363 |1.029|.987|.646|.619|
| .363–.464 |.983|.939|.664|.627|

## T-web confusion matrices (%)

Rows condition on true class; columns are mean posterior votes, not majority-map predictions. All supported cells weighted equally within each phase and truth class.

| Phase | True | Cells | Void | Wall | Filament | Knot |
|---|---|---:|---:|---:|---:|---:|
|ph018|Void|57238|76.81|20.12|2.88|0.19|
|ph018|Wall|39817|31.90|56.11|11.30|0.69|
|ph018|Filament|11219|13.50|40.23|41.60|4.66|
|ph018|Knot|870|7.25|20.84|43.86|28.05|
|ph019|Void|63510|78.36|19.26|2.26|0.12|
|ph019|Wall|44106|29.61|57.90|11.84|0.64|
|ph019|Filament|13103|10.01|38.96|46.31|4.72|
|ph019|Knot|1099|4.92|18.80|44.72|31.56|

## Interpretation and other evidence

Across ph018/ph019, true filament fractions are 10.28/10.76%, versus draw fractions 10.26/10.85%; true knot fractions 0.80/0.90%, versus 1.06/1.09%. Density at true knots is underpredicted by about 0.70 in stored density units on both phases. Conditioning on true peaks induces regression toward the mean; this is not a calibration test conditional on observed galaxy counts. At z<0.15 true filament → wall probabilities are 38.0/34.7%, knot → filament 45.1/41.6%.

Blind single-cell fair CRPS improved over the matched count baseline by 19.3% for density and 23.6/20.7/27.1% for ascending tidal eigenvalues. Empirical 5th–95th coverage was 87.83/86.93/86.21/85.22%, respectively. These tests support marginal skill on mocks, not joint-field calibration or DESI accuracy.

Individual draws retain much of the local power but do not reproduce exact truth morphology. Location, amplitude and shape errors can all reduce cross-correlation. The model samples possible smoothed matter fields conditional on the observed tracers and learned mock relation, not mass maps pinned to individual galaxies. No measurement here proves galaxy-position conditional calibration. Loa's high-redshift number-density mismatch remains unresolved.

## Evidence paths

- Per-draw spectra and cross correlations: /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/blind_local_power_62103650/{ph018,ph019}.json
- Truth-class tallies and conditional density errors: /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_points_blind_62103339/{ph018,ph019}.json
- Repository: /global/u2/d/dkololgi/TNG/Illustris. Scoring scripts: workflows/sbi/blind_local_power.py and workflows/sbi/loa_points_blind_web.py. Calibration: docs/evidence/lowz_cfm_blind_score_table.json. Matched slice: docs/figures/lowz_blind_61911489/ph018_first_pair.png.
- Loa: https://loa-lowz-32draw-lightcone.astrodak99.chatgpt.site (32 draws per cap for class probabilities; 16 selectable density draws).
