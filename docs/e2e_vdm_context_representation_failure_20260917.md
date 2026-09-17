# VDM context experiment: preserved physical-representation failure

**Update2026-09-17:** the explicitly approved operator-only v2 correction passes
the unchanged A32 physical gate. The original failed result below remains intact.
No GPU fit or new posterior draw has started; this is a representation result.

## Paired correction result

Frozen evaluator2a2d837, `source_physics_v2`; job58472098/nid004225
COMPLETED0:0,44s allocated,21.333s gate evaluation. Exactly the same32 anchor IDs
and parent-only metrics as v1 (verified exact equality). No held-out fields.

| Ordered eigenvalue | Parent median normalized RMSE | v2 median normalized RMSE | Median per-anchor relative reduction |
| --- | ---: | ---: | ---: |
| lambda1 |0.0645172|0.0104776|84.1744%|
| lambda2 |0.0512690|0.0080755|84.9633%|
| lambda3 |0.0398007|0.0064029|84.8787%|

All32 anchors improve for all3eigenvalues; minimum per-anchor reduction64.35%.
The unchanged>=25%median gate passes separately for each eigenvalue. Exact
mass/roundtrip/trace errors remain<=4.45e-15. Analytic DC/axis/diagonal relative
tensor errors are0,1.93e-16,2.60e-16,3.57e-16. This validates removing the
block-lift commutator; finite exterior/boundary and coarse-density errors remain.
It is not a neural-training result, calibrated posterior or production release.

New immutable `data/REPRESENTATION_GATE_V2.json` SHA256:
`7b438a46515652d7f3711bb7e3105ca3d434f959654a4731c39256cf1c3f291c`.
Original failedv1 SHA verified unchanged. `data/REPRESENTATION_RELEASE.json`
binds failedv1, passedv2, PHYSICS_V2_SOURCE and A32-only NORMALIZATION hashes.
Full training still requires the source/data manifest and GPU smoke/replay/cost
gates. Normalization uses the unchanged data chart, not a representation repair.

V2 gate evaluation costs21.333s vs1.404s for v1 (about15.2x, or0.67s/anchor),
because its wide192^3 FFT is larger. Rebenchmark full draw scoring on compute;
CPU-node cost can be parallelized within the unchanged8CPU-nodeh ceiling.
Cumulative allocated time is5900s=1.6389CPU-nodeh,0GPUh; no active allocation.
Full A/B/C/D completion remains the goal, with unchanged elapsed/resource caps.

The pre-fit physical gate failed on 2026-09-17. **No GPU fit or new posterior draw
has been launched.** This is a failure of the proposed tidal representation with
known matter fields, not a neural learning result or evidence against VDM.

## Verified execution

Frozen physical evaluator: Git `e1bde72`, Scratch `source_physics`.
Job58470989/nid004215: FAILED1:0 after18s; the application intentionally raised
after publishing the failed gate. Application evaluation took1.404s. It was not
an infrastructure failure and does not qualify for an automatic replacement.

Immutable receipt:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1/data/REPRESENTATION_GATE.json`
SHA256 `c72e7d9a6169299aa8beab877f156316bf6d6d56701aedefdc87fd2591e61581`.
It records `training_launch_allowed=false`, `heldout_used=false`, and all32 A-panel
training anchors. Preserve this receipt and its source; never overwrite it with a
repaired result or report a later success as though this failure did not occur.

| Ordered eigenvalue | Parent-only median normalized RMSE | Proposed composite median normalized RMSE | Median per-anchor relative reduction |
| --- | ---: | ---: | ---: |
| lambda1 | 0.06452 | 0.50695 | -702.9% |
| lambda2 | 0.05127 | 0.33943 | -551.2% |
| lambda3 | 0.03980 | 0.29750 | -629.1% |

The final column is the median of per-anchor ratios, not a ratio of the first two
columns. The required improvement was >=25% for each eigenvalue. All32 anchors
worsen for all three eigenvalues. The composite is typically6.5--8.0 times worse,
not marginally below tolerance.

Bookkeeping remains numerically correct: maximum blockmass error1.33e-15,
roundtrip3.55e-15 and trace4.44e-15, all far below2e-6. The scalar block-average
plane/DC transfer tests also pass. These checks are necessary but insufficient
for an accurate tensor field.

## Isolated operator defect

The registered reconstruction uses block replication U:

    T_candidate = U T_coarse(delta_c) + T_parent(delta_f - U delta_c).

But block replication and the tidal operator do not commute. Even on a common
periodic physical domain, where there is no unknown exterior or survey boundary,

    T_candidate - T_fine(delta_f)
      = U T_coarse(delta_c) - T_fine(U delta_c).

This difference is traceless: it can alter tidal shear and eigenvalues while all
the mass/trace checks remain exact. Expanding a coarse tensor into constant blocks
is not the tide of the block-expanded density used in the residual subtraction.

A tiny analytic fixture in
`workflows/sbi/e2e_vdm_context_operator_audit.py` uses16^3 fine/4^3 coarse grids on
the same periodic domain, no data or training, and exactly the unchanged operators:

| Plane-wave mode | Tensor RMS error / reference RMS |
| --- | ---: |
| DC (0,0,0) | 0 |
| Axis-aligned (1,0,0) | 1.25e-16 |
| Diagonal (1,1,0) | 0.48198 |
| Diagonal (1,1,1) | 0.60043 |

Density reconstruction, tensor trace and the commutator identity remain within
1.4e-16. Thus the previous axis-aligned/scalar transfer controls missed a genuine
anisotropic tensor error. The corresponding regression test deliberately records
this defect; it is not a replacement acceptance gate.

This isolates a mechanism, not a complete quantitative decomposition of the
cosmological error. Coarse aliasing and exterior/boundary approximation may still
matter after correcting the inconsistent tensor lift.

## Approved single correction test

User approved this action on2026-09-17. It is now being implemented; no v2
cosmological result is implied by authorization or the analytic tests. Publish
`data/REPRESENTATION_GATE_V2.json` from frozen `source_physics_v2`; the original
gate/source must remain unchanged. A passing new gate and A32 normalization
produce `data/REPRESENTATION_RELEASE.json`, which binds both results and the new
source. All downstream training checks require this release and the GPU smoke.

Keep the density target, block-mass factorization, data, resolution, model family
and >=25% training-only physical gate unchanged. Test a **consistent-density tidal
operator**: expand the wide coarse density onto the same fine lattice used in the
local residual, calculate its wide-domain tide there, crop that tensor, and add
the local residual tide. This replaces `U T_coarse(delta_c)` with the central
crop of `T_wide_fine(U delta_c)`; it does not change the generated density chart.

Requested bound: one CPU allocation up to30minutes, within the existing8CPU-nodeh
and elapsed ceilings, on the same A32 training panel. No GPU work, extra neural
fits, held-out predictive scores or relaxed gate. First require exact tensor
recovery on the matched-periodic analytic fixtures above. Then evaluate the same
cosmic gate and measure the increased CPU scoring cost. A new separately named
receipt/source must retain the failed v1 provenance. Passing is not guaranteed.

This is a material operator amendment to the registered formula, so it has been
approved explicitly rather than silently applied. The full matrix remains
incomplete pending a valid representation release and all downstream work.

## Data and resource state

All five phase products are now complete:384 training anchors,32 primary
evaluation anchors and4 adjacent-core companions. Packaging evaluated no learned
scores; ph005 is not opened for predictive confirmation. Product build58470146
completed0:0 in46m. No ph001/ph006 access. All allocations have ended.
Allocated CPU time through the gate is5856s =1.6267 CPU-nodeh across four jobs;
new-experiment GPU time is zero. Products total4,810,778,219 bytes (~4.48GiB).
The112GPUh/8CPU-nodeh/300GiB limits and2026-09-19T14:51:16Z deadline are unchanged.
