# Toy downstream sensitivity (not a DESI requirement validation)

R is in cells, with no assigned physical box size. Classes use threshold zero.
Linear coverage integrates the exact Gaussian reference; nonlinear class probabilities have finite Monte Carlo error.

| Arm | R/cell | Smoothed variance ratio | Class probability RMS error | Oracle-null RMS |
|---|---:|---:|---:|---:|
| oracle_null | 0.0 | 0.99962 | 0.01020 | 0.01020 |
| oracle_null | 0.5 | 1.00051 | 0.00923 | 0.00923 |
| oracle_null | 1.0 | 1.00340 | 0.00845 | 0.00845 |
| oracle_null | 2.0 | 1.01157 | 0.00866 | 0.00866 |
| top13 | 0.0 | 1.00047 | 0.01019 | 0.01020 |
| top13 | 0.5 | 1.00052 | 0.00924 | 0.00923 |
| top13 | 1.0 | 1.00340 | 0.00845 | 0.00845 |
| top13 | 2.0 | 1.01157 | 0.00866 | 0.00866 |
| all13 | 0.0 | 1.12957 | 0.01395 | 0.01020 |
| all13 | 0.5 | 1.13057 | 0.01340 | 0.00923 |
| all13 | 1.0 | 1.13384 | 0.01340 | 0.00845 |
| all13 | 2.0 | 1.14308 | 0.01313 | 0.00866 |
| erased_covariance | 0.0 | 1.00057 | 0.01441 | 0.01020 |
| erased_covariance | 0.5 | 0.61195 | 0.05172 | 0.00923 |
| erased_covariance | 1.0 | 0.39983 | 0.08763 | 0.00845 |
| erased_covariance | 2.0 | 0.21664 | 0.10982 | 0.00866 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | 0.0 | 1.02652 | 0.02029 | 0.01020 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | 0.5 | 1.06419 | 0.01922 | 0.00923 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | 1.0 | 1.10528 | 0.01788 | 0.00845 |
| alpha0.25_white_bridge_stochastic_seed17_amortised_decay | 2.0 | 1.05441 | 0.01963 | 0.00866 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | 0.0 | 1.01725 | 0.01613 | 0.01020 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | 0.5 | 1.03970 | 0.01455 | 0.00923 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | 1.0 | 1.05535 | 0.01342 | 0.00845 |
| alpha0.25_white_bridge_exact_seed17_amortised_decay | 2.0 | 1.01979 | 0.01572 | 0.00866 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | 0.0 | 1.02228 | 0.01931 | 0.01020 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | 0.5 | 1.06508 | 0.01860 | 0.00923 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | 1.0 | 1.11235 | 0.01740 | 0.00845 |
| alpha0.25_white_bridge_stochastic_seed29_amortised_decay | 2.0 | 1.06152 | 0.01941 | 0.00866 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | 0.0 | 1.01076 | 0.01527 | 0.01020 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | 0.5 | 1.03486 | 0.01354 | 0.00923 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | 1.0 | 1.05504 | 0.01210 | 0.00845 |
| alpha0.25_white_bridge_exact_seed29_amortised_decay | 2.0 | 1.04990 | 0.01588 | 0.00866 |

All original unsmoothed gates remain unchanged. This assay cannot certify 7 Mpc/h or DESI T-Web accuracy.
