# Classical/hybrid result and decision

All96arm/case/smoothing comparisons completed on CPU58824793/nid004194;
assay116.5seconds, exit0. Twelve focused tests pass, including constrained-draw
covariance identity, Schur covariance preservation and unchanged retained modes.
Source/input hashes are in manifest/COMPLETE; no training occurred.

At R=1cell (four development observations; neural/hybrid averaged over two seeds):

| Method | Smoothed variance / exact | Tidal-class probability RMS | Width4 regional90% coverage |
|---|---:|---:|---:|
| Classical constrained realizations |1.00744|.00877|.90089|
| Selected alpha=.30 neural |1.10502|.01797|.90738|
| Conditional low-mode hybrid |1.03804|.01470|.89795|
| Independent low-mode splice |1.60628|.03286|.93097|

Classical is an independent exact-target ensemble; its finite-draw deviations
provide the relevant Monte Carlo scale, not evidence of analytic bias. The
oracle-through-hybrid control shares high modes with the reference and therefore
is NOT an independent two-ensemble noise floor. Hybrid improvement does not
repair the retained high-mode marginal or establish universal qualification.
The negative control's seemingly reasonable regional coverage coexists with
16-probe covariance error~.485 and strongly excessive smoothed variance.

Classical generation of2048draws takes .045-.052seconds on four CPU threads,
excluding prior factorization/reference problem construction. Setup includes
the observation gain and hybrid conditional matrices (~.011-.013seconds/case).
No network fitting is required. Hybrid correction itself is cheap (~.016seconds)
but still requires expensive neural generation; no cross-device speedup ratio
is claimed. Dense512-dimensional algebra does not establish survey-scale cost.

## Scientific conclusion

Stop tuning the Gaussian neural model. A classical method already supplies the
correct posterior more simply; the hybrid teaches us that maintaining cross-mode
dependence is essential, but does not earn its extra machinery here. This is
expected on a linear-Gaussian rung, not a proof against neural nonlinear inference.

The next meaningful value test is nonlinear lognormal/Poisson: converged
multichain classical sampling as BOTH reference and competitor, cheap Laplace
approximation, and neural/hybrid only when the reference is trustworthy. It is
written into the plan but not implemented/launched in this bounded comparison.
Reference convergence, scientific downstream tolerances and robustness to survey
selection remain separate questions. No Abacus or production-P12 change follows.

Tidal classes here use threshold zero and R is in grid cells, not7Mpc/h. There
are only four development observations. Do not claim DESI environment calibration.

Outputs: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/reference_classical_hybrid_20260924_v1`.
Archived report and complete measurements: `docs/evidence/e2e_classical_hybrid_20260924/`.
