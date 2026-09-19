# Coupled-field physical-reference gate: PASS

Completed2026-09-18 in Slurm step58540511.13, COMPLETED0:0. Evaluation took
348.64s after waiting for the requiredph008 targets (total step28m59s).
Peak step RSS1.29GiB. This is truth-only representation qualification, **not
neural reconstruction performance, posterior calibration or training launch**.

## Registered panel and decision

The predeclared `configs/e2e_coupled_physics_v1.json` uses training phases007/008,
one pair in each cap x four redshift shells x interior/boundary stratum. That is
32pairs/64owned cores. The primary context offset is zero. Six additional context
translations are diagnostic only, making224pair-offset cases in the saved report.
No held-out predictions, checkpoint selection or gate adjustments were involved.

For both the independent-parent and joint-rectangle wide-plus-fine operators,
each ordered tidal eigenvalue must improve median per-core RMSE by at least25%
over the same parent-only baseline, with no median regression in either phase.

| Operator | Median improvement λ1 | λ2 | λ3 | Worst primary core/component improvement | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Independent parent + wide context | 88.44% | 88.44% | 88.53% | 76.79% | Pass |
| Joint rectangle + wide context | 88.20% | 88.30% | 88.30% | 76.23% | Pass |

Per-phase medians are87.86–88.18%/88.68–88.93% for the independent operator and
87.69–88.23%/88.52–89.00% for the joint operator (007/008 respectively).
Every tested primary core/component improves. Across all seven context offsets,
even the worst recorded improvement remains71.64%/71.72% respectively. The extra
translations were not substituted for the primary test.

Absolute median per-core eigenvalue RMSE, in physical dimensionless δ units:

| Operator | λ1 | λ2 | λ3 |
| --- | ---: | ---: | ---: |
| Parent only | 0.00869355 | 0.00872745 | 0.00863526 |
| Independent + wide | 0.00097568 | 0.00096016 | 0.00093985 |
| Joint + wide | 0.00095744 | 0.00096842 | 0.00095964 |

Median gap RMSEs are[0.00171328,0.00164004] for the independent operator and
[0.00168900,0.00168579] for the joint operator. These are diagnostic errors, not
gap posterior coverage. Maximum numerical trace discrepancy across all cases
and tested operators is5.33e-15.

## What this permits

The tested multiscale representation retains the full-box R7 tidal information
much better than the finite parent alone. Both independent-parent and joint
physical operators qualify against the frozen representation criterion. Their
similar truth errors are not evidence for either neural generative factorization.
The later primary I/J comparison must still use the same rectangular operator
and block-owned assembly, avoiding estimator and overlap-averaging confounds.

This clears one preparation gate. The separate all21phase products/audits,
strict13-phase normalization, actual targetless/normalized interfaces and
technical GPU throughput/restart checks subsequently passed on2026-09-19; see
the [complete preparation closeout](e2e_coupled_preparation_closeout_20260919.md).
No scientific model training is authorized by those checks. Whole-panel readiness
is established by the separate evidence, not inferred from this two-phase test.

## Reproducible evidence

Full case-level receipt:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/cartesian_v2/physical_gate/PHYSICAL_GATE_COMPLETE.json`

- Receipt SHA256:`0b53bea9cbfcb4760247e556f3195a24f5810189d056141154c16d51643c68a7`
- Physics config SHA256:`6e0a5854663047ec1e4bc558765393a2aa92fab8249c1a043452819e3fd4f9c9`
- Layout SHA256:`810a078eae98101f59ea380330c827f4603fb403ae7e166ce5e3dbae00a9c941`
- Operator SHA256:`54934aa3c69bad36d0e4d9b5135106387fe01f44efdef24b4166caf5fcedc38d`
- Gate builder SHA256:`63bd35fb68e84d561a8e04d9d317567120f1cfeb3add6ce7a2c44aa04ac1c9a9`

All referenced geometry/target receipt hashes were rechecked after completion.
Frozen executable source:`source_snapshots/physical_gate_v1`; log:
`physical_gate_58540511.log` under the registered preparation root. Small result
summaries are in this repository; bulk Scratch products remain subject to purge.
