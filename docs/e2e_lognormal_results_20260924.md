# Lognormal–Poisson classical comparison: conclusions

Completed on CPU allocation 58825943/nid004164. Four unit tests passed;
the four-case workflow exited successfully with source hashes unchanged.
Protocol: `e2e_lognormal_reference_20260924.md`. Receipts:
`evidence/e2e_lognormal_reference_20260924/COMPLETE.json`.
Full chains: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/reference_lognormal_20260924_v1`.

## What was established

Eight Metropolis-corrected, Laplace-preconditioned MALA chains per observation
pass the registered diagnostics: maximum rank/folded Rhat 1.00087, minimum bulk
ESS 20,308, minimum tail ESS 32,037. Independent step-size groups agree within
3.08 combined mean MCSE across 138 diagnostics. This is strong numerical evidence
for the tested reference, not a proof of full joint convergence or survey calibration.

Laplace uses the identical likelihood and prior, approximates log-density at its
MAP with the inverse Hessian, then transforms draws correctly to density.

| Case / mean count scale | Global mass bias / reference sd | Global mass variance ratio | Mass in nominal 90% interval | R=1 tidal probability RMS |
|---|---:|---:|---:|---:|
| 0 / 0.5 | +1.048 | 1.508 | 0.762 | 0.1220 |
| 1 / 0.5 | +0.866 | 1.417 | 0.808 | 0.1051 |
| 2 / 5 | +0.718 | 1.238 | 0.780 | 0.0205 |
| 3 / 5 | +0.545 | 1.138 | 0.800 | 0.1122 |

Independent MCMC replication gives global interval mass 0.9001–0.9030,
global mean differences -0.0248 to +0.0049 sd, and R=1 tidal probability RMS
0.00083–0.00467. The Laplace discrepancy is therefore well above replication noise.
Unsmoothed voxel interval mass looks healthy (0.891–0.895), despite variance
ratios 1.27–1.69. After R=1-cell smoothing, interval mass drops to 0.792–0.821,
and mean error is 0.538–0.898 sd. Smoothing does not make this error irrelevant:
coherent mean bias survives while local fluctuations average down.

These are conditional posterior-mass checks against reference draws, NOT coverage
frequencies across many independently generated observations. The four cases
have different truths and masks; differences cannot isolate count-density effects.
R is in grid cells, not physical Mpc/h. Tidal classes use threshold zero and the
registered periodic/DC convention. No DESI T-Web tolerance is inferred.

## Cost and decision

MAP fitting took 0.003–0.009 s per case; 16,384 Laplace draws took 0.034–0.039 s.
The eight-chain sampling loop (warmup included) took 3.66–3.81 s per case;
these times EXCLUDE compilation/import, convergence diagnostics, and tidal scoring.
Dense matrices and a 64-cell grid make this a capability test, not a survey-scale
throughput benchmark. No neural training was needed.

The inexpensive Laplace approximation is NOT an adequate substitute on the
tested regional-mass and tidal quantities. Classical nonlinear sampling is already
a practical competitor here, not merely an oracle used to grade neural models.
The next useful neural/hybrid experiment must earn an accuracy-versus-total-cost
advantage against it. A bounded comparison should use the same nonlinear generator,
compare amortised CFM and a learned residual/transport around Laplace, include
Metropolis-corrected classical draws, and account for training plus per-observation
cost. A learned correction is not presumed calibrated. Test on new development
observations before reserving a larger confirmation panel; include regional mass,
tidal probabilities and joint-dependence diagnostics. That training is NOT launched
or implied completed by this report. No return to Gaussian tuning or Abacus
promotion is justified by these four cases.

The statistical model follows the lognormal–Poisson tradition of
[Jasche et al.](https://arxiv.org/abs/0911.2498), but this implementation is MALA,
not HADES/HMC. Diagnostics follow
[Vehtari et al.](https://avehtari.github.io/rhat_ess/rhat_ess.html).
Known bias, selection and cosmology, no RSD, tiny periodic volume: model adequacy
for real DESI data remains a separate, untested question.
