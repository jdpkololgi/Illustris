# Threshold sensitivity and exact canvas reference — 2026-10-08

## Frozen diagnostic scope (before new results)

User authorizes saved-field threshold tests and an exact Gaussian test of the
actual patch assembly; one CPU interactive allocation, at most one hour. No new
Abacus fields, training, GPU work, sealed phases, model selection or gate changes.
Existing true-coarse summary and independent final check run unchanged first.

Thresholds: 0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4. The registered 0.2 physical truth
remains primary. Retain every ph012/013 pair, both operators and support strata.
Reclassify saved parent A eigenvalues and physical truth at each threshold;
report the existing finite-32-draw corrected fraction moment ratio, mean error
and predictive variance. Reconcile 0.2 exactly against the registered summary.
Report galaxy-weighted class Brier score and expected draw/truth disagreement
by nearest truth-eigenvalue distance in [0,.01,.025,.05,.1,infinity). These are
descriptive, not new acceptance tests or claims that Brier measures calibration.
Include galaxy counts and the one-galaxy fraction increment in case accounting.
B and no-neural closure retain labels only: their margin diagnostics are possible
at 0.2, but a multi-threshold B/closure sweep is not possible from saved products.
Thresholds will not be chosen based on which passes.

Gaussian reference: reuse the repository's float64 Gaussian prior/conditioning
and ExactModel. Test an independent-cell and correlated 4^3 prior, two existing
masked heteroscedastic observation templates, and all stride-one overlapping
cubes of side 2, 3, 4. Compare the full-domain conditional flow, cropped marginals
conditioned on all observations, and patch posteriors conditioned on local
observations. Every flow passes through the unmodified global canvas integrator.
Propagate zero and a complete basis to recover its affine mean/covariance without
Monte Carlo; NFE 128/256/512 checks numerical convergence. Report full covariance,
voxel variance, half-domain covariance, and exact Gaussian coverage/variance of
the domain mean. Independent-cell and full-window controls must converge to the
reference; each patch oracle is independently reconciled with oracle_moments.
This tests coarse-stage overlap/conditioning sufficiency, NOT fine-stage mass
projection, Abacus quantitative attribution, or neural-model adequacy.

Outputs: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/threshold_canvas_diagnostics_20261008_v1`.
Exclusive-create receipts bind source and input hashes and allocation. All results
remain development-only; scientific qualification is separate from completion.
