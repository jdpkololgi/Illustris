# Lognormal-Poisson: classical reference versus Laplace

4^3 development toy, known prior and selection, no bias/RSD/cosmology uncertainty. No neural training.
Convergence checks are evidence, not proof; interval mass is conditional reference mass, not survey-wide SBC.

| Case | Rate | Rhat max | Bulk ESS min | Tail ESS min | Pass |
|---|---:|---:|---:|---:|---|
| 0 | 0.5 | 1.0007 | 21771 | 32398 | True |
| 1 | 0.5 | 1.0009 | 22293 | 32892 | True |
| 2 | 5.0 | 1.0007 | 20462 | 32294 | True |
| 3 | 5.0 | 1.0007 | 20308 | 32037 | True |

| Case | R/cells | Laplace mean error/sd | Variance ratio | 90% interval ref mass | Tidal probability RMS | Reference replication RMS |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.2294 | 1.6465 | 0.8936 | 0.0383 | 0.0041 |
| 0 | 0.5 | 0.4545 | 1.5800 | 0.8757 | 0.0556 | 0.0043 |
| 0 | 1.0 | 0.8980 | 1.5252 | 0.7918 | 0.1220 | 0.0047 |
| 1 | 0.0 | 0.2313 | 1.6926 | 0.8949 | 0.0268 | 0.0030 |
| 1 | 0.5 | 0.4306 | 1.5785 | 0.8838 | 0.0442 | 0.0026 |
| 1 | 1.0 | 0.7776 | 1.4578 | 0.8212 | 0.1051 | 0.0029 |
| 2 | 0.0 | 0.2062 | 1.3947 | 0.8914 | 0.0545 | 0.0040 |
| 2 | 0.5 | 0.3926 | 1.3130 | 0.8733 | 0.0745 | 0.0040 |
| 2 | 1.0 | 0.6927 | 1.2629 | 0.7988 | 0.0205 | 0.0008 |
| 3 | 0.0 | 0.1835 | 1.2737 | 0.8931 | 0.0275 | 0.0035 |
| 3 | 0.5 | 0.3324 | 1.1942 | 0.8767 | 0.0527 | 0.0040 |
| 3 | 1.0 | 0.5380 | 1.1477 | 0.8198 | 0.1122 | 0.0045 |
