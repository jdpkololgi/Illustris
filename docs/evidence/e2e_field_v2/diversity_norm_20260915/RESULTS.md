# Diversity / normalization results

All rows: median across two seeds of each model's median over the same 12 transfer regions; no confidence intervals. Separate within-field paired effects and all seed results are in SUMMARY.json.

| Fields | Normalization | Clean RMS at .05 | .05 noise left | .05 error / parent | .2 noise left |
|---:|---|---:|---:|---:|---:|
| 3 | current | 0.00951191 | 0.11454 | 0.273281 | 0.0373099 |
| 3 | strict_global_train | 0.0106522 | 0.10425 | 0.334755 | 0.155361 |
| 6 | current | 0.00790839 | 0.263432 | 0.255397 | 0.0894125 |
| 6 | strict_global_train | 0.00993469 | 0.0955244 | 0.29669 | 0.0558652 |
| 9 | current | 0.00951782 | 0.161731 | 0.262803 | 0.113195 |
| 9 | strict_global_train | 0.00782778 | 0.280078 | 0.282973 | 0.067848 |
| 15 | current | 0.00816714 | 0.152407 | 0.230607 | 0.0495104 |
| 15 | strict_global_train | 0.00777398 | 0.248008 | 0.226376 | 0.0913101 |

## Frozen model: same weights, direct scaler interventions

| Scaler | Transfer clean RMS .05 | Transfer noise left .05 | Transfer error / parent .05 |
|---|---:|---:|---:|
| current | 0.0103055 | 0.0657669 | 0.311083 |
| strict_conditions_only | 0.0101557 | 0.075931 | 0.303372 |
| strict_global_train | 0.0101062 | 0.0798969 | 0.30042 |
| strict_target_only | 0.010259 | 0.0697962 | 0.307958 |
| tiny_train | 0.0550328 | 0.825457 | 0.772618 |

Compensated frozen coordinate roundtrip: maximum absolute normalized difference 3.33786e-06.

## Caveats

- Three phases only; transfer regions and wide contexts can be correlated.
- Strict scaler uses fixed largest training pool even for tiny fit; no transfer moments.
- Finite nominal noise clean identity is diagnostic, not required Bayes identity.
- Two seeds describe run variability, not independent-cosmology uncertainty.
- Fixed updates reduce exposures per field as diversity grows; no convergence guarantee.
- Direct frozen scaler swaps change the predictor; only compensated roundtrip is function-equivalent.
- Raw RMS naturally covaries with signal amplitude; field-relative RMS associations are also reported.
