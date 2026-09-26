# P12-A population, response and frozen-model replay

## Coordinate conclusion

The native/Planck18 comparison does not establish a P12 input/label error.
P12 uses a consistent fiducial observer chart, while its T-Web labels are
sampled in native host coordinates. Sampled native labels and observer points
reproduce. Keep that P12 convention for this VAC; E2E product substitution is
not required. This conclusion does not waive the separate tiling gate below.

## Actual DESI quality cuts

The existing GraphWeb catalogue satisfies ZWARN==0, DELTACHI2>=25 and
SPECTYPE==GALAXY for every row. In the VAC range, tightening DELTACHI2 to>40
would remove16,752/6,466,730 rows (0.259%). The installed LSS helper additionally
omits the GALAXY requirement; its Loa selection is therefore not a nested
version of the historical cut. This is a scientific sample/response choice,
not proof that the catalogue is broken. Detailed counts and explicit mock
success semantics: GraphWeb_DESI/docs/p12a_quality_cut_results_20260924.md.
No quality cut has been silently changed.

## Frozen replay: matched predictions and response, failed halo24 invariance

GPU allocation58824466, training phase ph002 only. Eight geometry-selected
cores span both caps and four shells,14,230 galaxies. No ph001 or reserved
confirmation payload was opened; no fitting or real-DESI inference ran.

The first strict stored-prediction replay disabled cuDNN TF32, unlike the
original exporter. Maximum discrepancy was3.72e-4. Retrying with original
precision (cuDNN TF32 on, matmul TF32 off) reproduces every stored prediction
**exactly**, with unchanged1e-5 numerical acceptance criteria. Retain both
reports and both hash-verified source snapshots. This discrepancy was an audit
execution-setting difference, not a training-product inconsistency.

All eight cores also reproduce ntilde exactly. Sixteen sampled rows/core
reproduce P3b random-support flags and boundary distances exactly. Separately,
eight archived training contexts yield finite ordered FMPE draws and exact
same-seed repeatability (32 draws/context). These checks do not establish
fresh calibration or Loa adapter qualification.

The production full-fit encoder with halo24 fails four of eight context-growth
checks and two of eight subdivision checks. Worst core halo24->48 normalized
RMSE is about0.0595 versus the pre-existing worst-core limit0.04; two core
subdivision NRMSEs exceed0.02. This is far larger than the resolved precision
difference. Do not weaken the gates or describe halo24 as context-converged.
The current posterior was calibrated under that fixed halo24 pipeline; this
failure does not erase that mock evidence, but prevents deployment qualification
under the registered context/subdivision requirements.

## Bounded candidate correction: frozen weights, halo48

On the same eight cores, halo48->64 passes the existing convergence criteria:
aggregate NRMSE=2.59e-5. Aligned subdivision at halo48 also passes, aggregate
NRMSE=1.41e-4. This is a diagnostic candidate, not a promoted model. It supports
trying a context-size correction before replacing the U-Net or training new
encoder weights. It does not justify feeding halo48 summaries into the old
halo24-calibrated FMPE without validation.

Sparse/edge follow-up also passes on16 additional preselected cores (eight
sparse and eight edge, all caps/shells). Every selected edge core has100% of its
rows within the registered support-boundary distance. Aggregate halo48->64
NRMSE=4.93e-4; subdivision NRMSE=1.49e-4. Worst individual growth NRMSE=0.00857.
Evidence: P12A_HALO48_STRESS_20260924.json. Across dense and stress controls,
24 distinct cores and19,281 galaxies were tested. This is still one training
phase, not cross-phase calibration or independent confirmation.

Before a replacement can deploy: finish input-rebuild controls;
regenerate all affected out-of-fold and full-fit summaries under one pinned
context policy; refit the posterior; rerun registered calibration and independent
confirmation, preserving phase reservations. Retrain encoder weights only if
these controls show it is necessary. No new E2E programme is part of this fix.

## Population census status

The completed census covers all five training phases ph000, ph002-005.
Every raw R_MAG_APP<19.5 galaxy in0.15<=Z<0.55 has RES=1. RES=0 objects exist
at other redshifts, so the absent unresolved target sample is a consequence of
the target population, not a reason to drop additional rows or invent a test.
This closes the RES=0 target-range coverage question for these raw bright
parents; it does not claim arbitrary low-redshift context buffers are resolved.

| Phase | Resolved centrals in raw bright target range | Resolved satellites | Unresolved |
|---|---:|---:|---:|
| ph000 | 18,397,320 | 2,439,497 | 0 |
| ph002 | 18,444,414 | 2,457,974 | 0 |
| ph003 | 18,377,593 | 2,439,864 | 0 |
| ph004 | 18,324,507 | 2,416,690 | 0 |
| ph005 | 18,429,119 | 2,450,880 | 0 |

## Evidence

All files below are under docs/evidence/p12:

- P12A_FROZEN_MOCK_REPLAY_20260924.json (first precision configuration).
- P12A_FROZEN_MOCK_REPLAY_20260924_v2.json (exact stored prediction replay;
  halo24 context/subdivision failures remain).
- P12A_HALO48_CONTROL_20260924.json (bounded dense-region candidate control).
- P12A_POSTERIOR_REPLAY_SMOKE_20260924.json (sampling repeatability only).
- P12A_POPULATION_CENSUS_PH002_20260924.json and subsequent training-phase
  census reports (raw magnitude/redshift census; no reserved phases).

No DESI inference is ready on the basis of these partial passes. The decisive
new blocker is halo24 context/subdivision dependence, not the distance chart
or the existence of a different DESI quality-cut convention.

## Execution and verification

CPU job58824338 completed the actual-catalogue cut census and all five raw
population censuses. GPU58824466 completed reference and candidate controls
and was released. Scheduler completion is not scientific success: the original
halo24 report remains failed. A GPU allocation request with64GiB host memory
was rejected by the scheduler's cores/GPU rule;48GiB succeeded. An untyped GPU
step rejection was resolved by specifying the allocation's actual a100 type.
No additional user compute approval was necessary.

Nine coordinate-helper tests and five observational-preflight tests pass;
new audit entrypoints compile and repository whitespace checks pass. Illustris
code/global graphs refreshed. GraphWeb graphify refused a smaller rebuilt graph
(586 versus592 nodes); its previous graph was preserved, without forcing away
potential semantic entries. This indexing limitation does not alter the audit
outputs or scientific gates.
