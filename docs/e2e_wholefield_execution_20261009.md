# Whole-field campaign execution record

## Authority and budget

Approved plan/scope/log commit: `6d3b993`. Sensitivity runs first. All gates must
be reported before advancement; an insufficient panel stops the campaign.
Execution begins with CPU allocation 59602162 at 2026-10-09 19:17:04 UTC.
The 14-day deadline is 2026-10-23 19:17:04 UTC. GPU ceiling is 32 allocated
hours, including at most 8 for Gate 1. GPU use so far: zero.

Allocation 59602162: interactive CPU, desi, nid200076, one hour maximum,
Scratch license, 64 requested logical CPUs (exclusive node allocation exposes
256). No other allocation was present at start. Gate 2 output root is
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/wholefield_comparison_20261009_v1/gate2`.
No new fields, training or sealed-phase access is needed for preparation.

## Gate 2 preparation

`configs/e2e_wholefield_gate2_v1.json` fixes the panel, physical coordinates,
32-draw budget, effect and power requirements before synthetic sensitivity runs.
The six-dimensional fine-sensitive pair vector consists, per core, of density
averaged in the fixed offset subcube [2:14]^3 and two orthonormal trace-free
contrasts of volume-mean ordered tidal eigenvalues. Identity scaling uses the
existing dimensionless physical units; no development-fitted whitening is used.
Full-core mass and all three regional eigenvalues remain separately recorded.
The offset is geometry-defined, not selected by residuals. Full-core masses
aligned to coarse blocks remain controls, not fine improvement targets.

Preparation verifies saved parent receipts and hashes on the compute node,
checks all 32 pair IDs across draw batches, and reconciles regional eigenvalues
and physical truth against the earlier Stage A2 summaries. Six focused tests
pass. The saved PREPARED.json and PANEL.npz bind exact inputs and coordinates.
Preparation took 3.9 seconds for the four caps plus summary/receipt writing.
This is not a sensitivity pass.

All-pair continuous ratios are 1.4687444108 (ph012) and 0.7665127685 (ph013).
Interior-only ratios are 1.4697373317 and 1.3046031679. Both strata are retained;
the primary all-pair panel is not replaced by the more favorable interior subset.
The class-fraction targets 2.5 to 1.5 are secondary and not these baselines.

Historical true-coarse saved products lack regional continuous tidal draws.
Do not relabel the local-oracle products as full-cap true-coarse evidence.
A necessary-condition screen on the available sampled-coarse panel may reject
the complete gate, but cannot pass it: full advancement implies the necessary
event. If a simultaneously valid upper confidence bound for that relaxed event's
detection probability is below 80%, additional stricter criteria cannot rescue
the complete rule under that synthetic law. Otherwise continue to the complete
rule and obtain any missing matched summaries within the approved budget.
