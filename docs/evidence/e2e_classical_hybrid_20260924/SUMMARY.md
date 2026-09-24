# Classical / conditional-hybrid comparison

Gaussian development only. All models use the same prior/observation law. R in cells; threshold-zero toy tidal classes.
Independent splice is a negative control, not a calibrated construction. Classical draws are a competitor, not merely a label oracle.

| Arm | R | Field mean error | Covariance | Smoothed variance | Class RMS | Regional width4 coverage |
|---|---:|---:|---:|---:|---:|---:|
| classical | 0.0 | 0.02176 | 0.06479 | 1.00065 | 0.01058 | 0.90089 |
| classical | 1.0 | 0.02176 | 0.06479 | 1.00744 | 0.00877 | 0.90089 |
| classical | 2.0 | 0.02176 | 0.06479 | 1.01682 | 0.00934 | 0.90089 |
| oracle_hybrid_control | 0.0 | 0.02181 | 0.06206 | 0.99959 | 0.00410 | 0.89975 |
| oracle_hybrid_control | 1.0 | 0.02181 | 0.06206 | 1.00266 | 0.00590 | 0.89975 |
| oracle_hybrid_control | 2.0 | 0.02181 | 0.06206 | 1.00188 | 0.00877 | 0.89975 |
| neural_17 | 0.0 | 0.08270 | 0.08208 | 1.03400 | 0.01975 | 0.90848 |
| neural_17 | 1.0 | 0.08270 | 0.08208 | 1.10949 | 0.01857 | 0.90848 |
| neural_17 | 2.0 | 0.08270 | 0.08208 | 1.03969 | 0.02446 | 0.90848 |
| conditional_hybrid_17 | 0.0 | 0.08121 | 0.07230 | 1.02731 | 0.01936 | 0.89913 |
| conditional_hybrid_17 | 1.0 | 0.08121 | 0.07230 | 1.04176 | 0.01430 | 0.89913 |
| conditional_hybrid_17 | 2.0 | 0.08121 | 0.07230 | 0.98935 | 0.01396 | 0.89913 |
| independent_splice_17 | 0.0 | 0.07998 | 0.48241 | 1.10569 | 0.02143 | 0.93104 |
| independent_splice_17 | 1.0 | 0.07998 | 0.48241 | 1.60285 | 0.03277 | 0.93104 |
| independent_splice_17 | 2.0 | 0.07998 | 0.48241 | 1.06556 | 0.01576 | 0.93104 |
| neural_29 | 0.0 | 0.07987 | 0.08093 | 1.02569 | 0.01944 | 0.90627 |
| neural_29 | 1.0 | 0.07987 | 0.08093 | 1.10054 | 0.01736 | 0.90627 |
| neural_29 | 2.0 | 0.07987 | 0.08093 | 1.06336 | 0.01890 | 0.90627 |
| conditional_hybrid_29 | 0.0 | 0.07964 | 0.07144 | 1.01924 | 0.01889 | 0.89677 |
| conditional_hybrid_29 | 1.0 | 0.07964 | 0.07144 | 1.03432 | 0.01510 | 0.89677 |
| conditional_hybrid_29 | 2.0 | 0.07964 | 0.07144 | 0.97120 | 0.01755 | 0.89677 |
| independent_splice_29 | 0.0 | 0.07762 | 0.48776 | 1.09927 | 0.02080 | 0.93089 |
| independent_splice_29 | 1.0 | 0.07762 | 0.48776 | 1.60971 | 0.03295 | 0.93089 |
| independent_splice_29 | 2.0 | 0.07762 | 0.48776 | 1.05816 | 0.01599 | 0.93089 |

CPU seconds for 2048 classical draws per case: [0.045052721980027854, 0.049520414002472535, 0.05160538700874895, 0.04484358700574376]
No neural training required for classical draws; hybrid retains the neural generation cost. No cross-hardware speedup claim.
A Gaussian result does not validate a nonlinear lognormal/Poisson reference or DESI robustness.
