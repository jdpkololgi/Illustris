# VDM context experiment: preserved physical-representation failure

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

## Single next proposed action; approval requested

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
asked of the user rather than silently applied. The full matrix remains
incomplete pending that direction and a valid representation release.

## Data and resource state

All five phase products are now complete:384 training anchors,32 primary
evaluation anchors and4 adjacent-core companions. Packaging evaluated no learned
scores; ph005 is not opened for predictive confirmation. Product build58470146
completed0:0 in46m. No ph001/ph006 access. All allocations have ended.
Allocated CPU time through the gate is5856s =1.6267 CPU-nodeh across four jobs;
new-experiment GPU time is zero. Products total4,810,778,219 bytes (~4.48GiB).
The112GPUh/8CPU-nodeh/300GiB limits and2026-09-19T14:51:16Z deadline are unchanged.
