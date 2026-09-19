# Learned Gaussian reference results

Allocation 58594516; 17.79 minutes in the main runner; 0.213 GiB main outputs. 96 registered evaluation ensembles (512 draws each).

These are 8^3 synthetic field diagnostics, not Abacus/DESI validation. Fixed fits use exact posterior training draws. All tolerances were frozen before fitting.

## Exact-sampler controls

Maximum relative covariance error across the two observation-mask templates:

| NFE | VDM ancestral | CFM Heun |
|---:|---:|---:|
| 32 | 0.320908 | 0.005889 |
| 64 | 0.181781 | 0.001391 |
| 128 | 0.097068 | 0.000337 |
| 256 | 0.050200 | 0.000083 |
| 512 | 0.025533 | 0.000021 |
| 1024 | 0.012877 | 0.000005 |

Numerical gate is <=0.05. A failure prevents attributing the corresponding neural result exclusively to learning. No learned NFE512 run is implied by the exact control.

## Metric controls

| Case | Control | Mean RMS | Covariance error | Variance ratio | Octant coverage | Pass |
|---:|---|---:|---:|---:|---:|:---:|
| 0 | exact | 0.0498 | 0.1301 | 0.9956 | 0.8896 | True |
| 0 | independent | 0.0484 | 0.8116 | 0.9978 | 0.6666 | False |
| 0 | shrunk | 0.0416 | 0.3309 | 0.6969 | 0.8187 | False |
| 1 | exact | 0.0437 | 0.1306 | 1.0074 | 0.9074 | True |
| 1 | independent | 0.0435 | 0.8640 | 1.0054 | 0.7077 | False |
| 1 | shrunk | 0.0365 | 0.2895 | 0.7052 | 0.8411 | False |
| 2 | exact | 0.0429 | 0.1321 | 1.0081 | 0.8962 | True |
| 2 | independent | 0.0421 | 0.8075 | 0.9995 | 0.6738 | False |
| 2 | shrunk | 0.0359 | 0.2860 | 0.7057 | 0.8271 | False |
| 3 | exact | 0.0458 | 0.0845 | 1.0051 | 0.9003 | True |
| 3 | independent | 0.0432 | 0.8607 | 0.9986 | 0.7013 | False |
| 3 | shrunk | 0.0384 | 0.2842 | 0.7036 | 0.8325 | False |

## Matched-case learning progression

Means across cases 0/1 and the two seeds; pass counts retain all four cells. These cells are not treated as four independent cosmological universes. Covariance refers to the fixed 16-dimensional probe covariance.

| Objective | Regime | Update | NFE | Mean RMS | Covariance error | Variance ratio | Coverage | Pass cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cfm | amortised | 1024 | 128 | 0.3441 | 0.3602 | 1.2299 | 0.9324 | 0/4 |
| cfm | amortised | 1024 | 256 | 0.3441 | 0.3602 | 1.2292 | 0.9324 | 0/4 |
| cfm | amortised | 4096 | 128 | 0.2784 | 0.2532 | 1.0853 | 0.9208 | 0/4 |
| cfm | amortised | 4096 | 256 | 0.2784 | 0.2535 | 1.0853 | 0.9207 | 0/4 |
| cfm | fixed | 1024 | 128 | 0.1125 | 0.4794 | 1.0722 | 0.9284 | 0/4 |
| cfm | fixed | 1024 | 256 | 0.1126 | 0.4795 | 1.0715 | 0.9283 | 0/4 |
| cfm | fixed | 4096 | 128 | 0.0938 | 0.2480 | 1.0260 | 0.9183 | 1/4 |
| cfm | fixed | 4096 | 256 | 0.0938 | 0.2483 | 1.0258 | 0.9183 | 0/4 |
| vdm | amortised | 1024 | 128 | 0.5648 | 0.6098 | 1.5160 | 0.8570 | 0/4 |
| vdm | amortised | 1024 | 256 | 0.5669 | 0.5942 | 1.5602 | 0.8625 | 0/4 |
| vdm | amortised | 4096 | 128 | 0.3670 | 0.4128 | 1.0338 | 0.8758 | 0/4 |
| vdm | amortised | 4096 | 256 | 0.3718 | 0.3745 | 1.0797 | 0.8790 | 0/4 |
| vdm | fixed | 1024 | 128 | 0.2120 | 0.6135 | 1.1219 | 0.8566 | 0/4 |
| vdm | fixed | 1024 | 256 | 0.2116 | 0.5809 | 1.1556 | 0.8628 | 0/4 |
| vdm | fixed | 4096 | 128 | 0.1517 | 0.4504 | 1.0264 | 0.8795 | 0/4 |
| vdm | fixed | 4096 | 256 | 0.1525 | 0.4021 | 1.0665 | 0.8874 | 0/4 |

## Final per-seed/per-case evidence at NFE256

| Fit | Case | Mean RMS | Covariance error | Variance ratio | Coverage | Failing gates |
|---|---:|---:|---:|---:|---:|---|
| cfm_seed17_amortised | 0 | 0.2265 | 0.2783 | 1.0438 | 0.9253 | mean, covariance |
| cfm_seed17_amortised | 1 | 0.3146 | 0.3245 | 1.1209 | 0.9500 | mean, covariance, variance, coverage |
| cfm_seed17_amortised | 2 | 0.2391 | 0.2850 | 1.0571 | 0.9247 | mean, covariance |
| cfm_seed17_amortised | 3 | 0.3155 | 0.3715 | 1.1421 | 0.9566 | mean, covariance, variance, coverage |
| cfm_seed17_fixed0 | 0 | 0.0860 | 0.3309 | 1.0697 | 0.9442 | covariance |
| cfm_seed17_fixed1 | 1 | 0.1043 | 0.2040 | 1.0149 | 0.9155 | mean, covariance |
| cfm_seed29_amortised | 0 | 0.2587 | 0.2105 | 1.0574 | 0.8909 | mean, covariance |
| cfm_seed29_amortised | 1 | 0.3139 | 0.2006 | 1.1192 | 0.9165 | mean, covariance, variance |
| cfm_seed29_amortised | 2 | 0.2446 | 0.2681 | 1.1054 | 0.9247 | mean, covariance, variance |
| cfm_seed29_amortised | 3 | 0.3522 | 0.1782 | 1.1031 | 0.9142 | mean, covariance, variance |
| cfm_seed29_fixed0 | 0 | 0.0958 | 0.1840 | 1.0151 | 0.8920 | covariance |
| cfm_seed29_fixed1 | 1 | 0.0894 | 0.2744 | 1.0036 | 0.9215 | covariance |
| vdm_seed17_amortised | 0 | 0.3388 | 0.3310 | 1.0475 | 0.8988 | mean, covariance |
| vdm_seed17_amortised | 1 | 0.3410 | 0.3626 | 1.0351 | 0.9211 | mean, covariance |
| vdm_seed17_amortised | 2 | 0.3545 | 0.3199 | 1.0308 | 0.8650 | mean, covariance |
| vdm_seed17_amortised | 3 | 0.3699 | 0.3639 | 1.0534 | 0.9184 | mean, covariance |
| vdm_seed17_fixed0 | 0 | 0.1265 | 0.3520 | 1.0716 | 0.8956 | mean, covariance |
| vdm_seed17_fixed1 | 1 | 0.1642 | 0.4506 | 1.0273 | 0.8927 | mean, covariance |
| vdm_seed29_amortised | 0 | 0.4216 | 0.3268 | 1.1144 | 0.8265 | mean, covariance, variance, coverage |
| vdm_seed29_amortised | 1 | 0.3857 | 0.4775 | 1.1219 | 0.8697 | mean, covariance, variance |
| vdm_seed29_amortised | 2 | 0.3223 | 0.3491 | 1.0675 | 0.9292 | mean, covariance |
| vdm_seed29_amortised | 3 | 0.4278 | 0.4578 | 1.1368 | 0.8858 | mean, covariance, variance |
| vdm_seed29_fixed0 | 0 | 0.1648 | 0.3501 | 1.1019 | 0.8735 | mean, covariance, variance |
| vdm_seed29_fixed1 | 1 | 0.1547 | 0.4556 | 1.0652 | 0.8877 | mean, covariance |

Coverage here is the exact Gaussian posterior probability inside the generated 5%-95% octant intervals. It is not TARP, a test on thousands of independent truths, or proof of correct full 512-dimensional dependence. The complete receipts also retain all shell power ratios and voxel variance errors.

No lognormal/Poisson or Abacus training was run. Numerical correctness, finite-budget neural accuracy and real-survey robustness remain separate judgments.
