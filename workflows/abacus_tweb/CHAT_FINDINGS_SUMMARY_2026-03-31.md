## Abacus CutSky ↔ T-Web ↔ Graph-metrics validation (chat findings, 2026-03-31)

### Goal
Validate that our graph construction + graph-metric pipeline on AbacusSummit DESI BGS CutSky mocks is consistent with the T-Web eigenvalue labels, and diagnose why early “feature ↔ eigenvalue” correlations were weak / inconsistent.

This summary focuses on **label alignment** (IDs + coordinates + T-Web product) and **what was/was not actually being tested** when we computed Pearson correlations.

---

## Key artifacts produced / used

### Graph construction outputs (unique host halos)
- **Alpha graph artifacts** (built with `build_abacus_graph.py` on unique host halos):
  - `host_halos_unique_alpha_edges_combined_idx.npy` (edges)
  - `host_halos_unique_alpha_tetrahedra_idx.npy` + `host_halos_unique_alpha_tetrahedra_volumes.npy`
  - `host_halos_unique_alpha_points.npy` (+ `_points_xyz.npy`)
  - `host_halos_unique_alpha_metadata.json`

### Graph metrics (cuGraph / RAPIDS)
- Computed with `abacus_graph_features_cugraph.py` (rapids-gnn env):
  - `host_halos_unique_alpha_cugraph_node_features.parquet`
  - `host_halos_unique_alpha_cugraph_edge_features.parquet`
  - `host_halos_unique_alpha_cugraph_gnn_arrays.npz`
  - `host_halos_unique_alpha_cugraph_gnn_metadata.json`

### Unique-halo export key file (critical for alignment)
- `host_halos_unique_keys.npy` shape `(17,320,272, 2)` dtype `int64`:
  - column 0: `FILE_NUM`
  - column 1: `BOX_INDEX`
  - row order == points order used for the original unique-halo export.

### T-Web slab directories that were compared
We discovered multiple slab products with different metadata:

- `/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs`
  - `ngrid=3414`, `Rsmooth=2.0`, `threshold=0.2`, `boxsize=2000.0`

- `/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid_v2/backend_optimized_ngrid_2048_rsmooth_4/`
  - `ngrid=2048`, `Rsmooth=4.0`, `threshold=0.2`, `boxsize=2000.0`
  - includes explicit `x_start/x_end` metadata per slab.

- `/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid/backend_optimized_ngrid_2048_rs4`
  - appears numerically identical to the `fullgrid_v2` product, but missing some slab metadata fields (notably `x_start/x_end`) and slightly different file sizes due to metadata.

**Conclusion:** Prefer the `fullgrid_v2/...rsmooth_4` directory going forward because it is self-describing.

---

## The main conceptual pitfall: which “Pearson correlation” are we computing?

We computed three different kinds of “Pearson r” during the debugging:

1) **Graph metrics vs eigenvalues** (halo-level):  
   \(\mathrm{corr}( \text{metric}(halo), \lambda_i(halo) )\)  
   - This is a *predictiveness* check. Low Pearson here can be “real” (metrics may be lossy / nonlinear relationship).

2) **FITS eigenvalues vs slab eigenvalues** (label-vs-label, halo-level):  
   \(\mathrm{corr}( \lambda_i^{FITS}(halo), \lambda_i^{slabs}(halo) )\)  
   - This is a *label alignment* check. Pearson should be ~1 if we are sampling the same T-Web field at the same positions.

3) **BOX_INDEX labels vs x_com labels** on a **sample** (diagnostic):  
   Compare eigenvalues assigned via two methods to test mapping assumptions.

The most important result in this chat came from (2): **label-vs-label was ~0**, which indicates a mapping/ID/convention mismatch.

---

## What we learned (high signal)

### 1) `BOX_INDEX` is not a halo index into `halo_info_XXX.asdf`
Email from mock creator (paraphrased):
- Halo catalog coordinates live in \([-1000,1000]\) Mpc/h.
- Observer at \((-1000,-1000,-1000)\) Mpc/h; shift observer to origin; periodic replications used.
- `BOX_INDEX` is an index of the **galaxy in the cubic box mock**.
- `BOX_INDEX = -1` marks galaxies in unresolved halos (placed using DM particles).

**Implication:** you cannot do `halo_info[BOX_INDEX]` and expect `x_com`.

### 2) The “slabbing” artifact in the box-frame plot was caused by mis-indexing
In `test_mock2cube.ipynb`, the “slabbed” cube appearance came from mapping:
`FILE_NUM + BOX_INDEX → halo_info_*.asdf → x_com`.

After switching to:
`FILE_NUM + HALO_INDEX → halo_info_*.asdf → x_com`,
the point cloud became much more physically plausible (no voxel-aligned planes).

### 3) Label-vs-label validation: FITS λ vs slab-sampled λ was ~uncorrelated
We wrote and ran:
- `validate_unique_halo_eigs_fits_vs_slabs.py`

For the same unique halos, it compared:
- `LAMBDA1/2/3` stored in the annotated FITS
vs
- `eig_vals` sampled directly from T-Web slabs at the halo positions used.

Result (for both `ngrid=3414, Rsmooth=2` and `ngrid=2048, Rsmooth=4` slab dirs):
- Pearson \(r\) ≈ 0
- MAE/RMSE large (order unity and larger)

**Interpretation:** At least one of the “position+key” choices used in that comparison was not the same mapping used to generate the FITS labels. The dominant mismatch we identified was using a `keys.npy` that encodes `(FILE_NUM, BOX_INDEX)` while the halo-linkage annotation pipeline uses `(FILE_NUM, HALO_INDEX)`.

### 4) The two “fullgrid 2048 rs=4” slab dirs are effectively the same data
We inspected `rank0000` and other ranks and found:
- identical shapes/dtypes and sampled values
- `fullgrid_v2` includes `x_start/x_end`

So differences between those two dirs are packaging/metadata, not physics.

---

## What the images show (and how they were used)

### A) “Slabbed” appearance when using the wrong indexing (BOX_INDEX→halo_info)
This is the symptom of using a galaxy index as if it were a halo index.

### B) Non-slabbed “chunky” appearance when using HALO_INDEX→halo_info
This is expected when you take a **sky-footprint / lightcone selection** and map it into a periodic box; it is not a uniform cube, but it should not show thin quantized planes.

### C) RA/DEC/Z_COSMO → Cartesian shows survey footprint
This is expected in observer-centric coordinates (and can extend beyond a single fundamental domain unless you wrap).

### D) kNN local-continuity diagnostic (coords → slab-sampled eigenvalues)
We clarified an important statistical point:
- It is **expected** that `corr(x, λ)` (absolute coordinate vs eigenvalue) is ~0 in a homogeneous periodic field.
- The correct “nearby points have similar λ” test is a **local continuity** test (e.g. kNN or distance-binned correlogram).

We ran a kNN continuity check on a random subsample:
- `N=200,000`, `K=16`
- mean neighbor distance ≈ **26.77 Mpc/h**
- kNN vs random pair differences (MAE ratios):
  - λ1: MAE ratio ≈ **0.950** (weak improvement)
  - λ2: MAE ratio ≈ **0.882**
  - λ3: MAE ratio ≈ **0.831**

**Interpretation:** eigenvalues show **some** local continuity at ~27 Mpc/h scales (especially λ2/λ3), but it is not extremely strong at that neighbor distance. If you want a stronger continuity signal, reduce the neighbor scale (smaller k / use a radius cut) or explicitly test continuity vs distance bins.

You shared screenshots during the debugging; the key ones are saved locally at:
- `/.cursor/projects/global-homes-d-dkololgi/assets/image-673d2f60-39ef-476c-bd91-6c328c35593c.png`
- `/.cursor/projects/global-homes-d-dkololgi/assets/Screenshot_2026-03-31_at_17.38.54-166493b8-1f2f-4f07-874a-b5300b862768.png`
- `/.cursor/projects/global-homes-d-dkololgi/assets/Screenshot_2026-03-31_at_18.17.28-d6271ba2-50b6-4960-a21b-bff94d13ae47.png`

If you view this markdown in Cursor (not GitHub), you can embed them like:

```text
![](/global/homes/d/dkololgi/.cursor/projects/global-homes-d-dkololgi/assets/Screenshot_2026-03-31_at_17.38.54-166493b8-1f2f-4f07-874a-b5300b862768.png)
```

---

## Practical conclusions / next steps

### A) Stop using `BOX_INDEX` as a halo row index
For halo `x_com` in `halo_info_XXX.asdf`, use:
- `(FILE_NUM, HALO_INDEX)`

Keep using `BOX_INDEX = -1` only as a filter for “unresolved halo” galaxies.

### B) Keep your label path and graph node ID path consistent
If you build a unique-halo graph on a set of halos keyed by `(FILE_NUM, BOX_INDEX)`,
but your labels were produced by halo linkage `(FILE_NUM, HALO_INDEX)`,
then any “FITS-vs-slabs” or “metrics-vs-FITS” comparisons are not guaranteed to align.

**Recommended:** rebuild the unique halo export keyed by `(FILE_NUM, HALO_INDEX)` (not `BOX_INDEX`), regenerate points/keys, rebuild graph, recompute metrics, then correlate against slab-sampled λ at those same points.

### C) Use a single canonical T-Web product and record provenance
Use:
`/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid_v2/backend_optimized_ngrid_2048_rsmooth_4`

and ensure any FITS annotation records:
- `tweb_dir`, `ngrid`, `Rsmooth`, `threshold`, `boxsize`, and the mapping convention used.

---

## Scripts added during this chat
- `workflows/abacus_tweb/correlate_unique_halo_metrics_vs_tweb_eigs.py`
  - Computes Pearson between cuGraph metrics and slab-sampled eigenvalues at node positions.

- `workflows/abacus_tweb/validate_unique_halo_eigs_fits_vs_slabs.py`
  - Label-vs-label validation: FITS `LAMBDA*` vs slab-sampled `eig_vals` for unique halos.

---

## One-line takeaway
The biggest issue was **ID/convention mismatch**: `BOX_INDEX` is a **galaxy index** (CubicBox mock) and should not be used to index `halo_info` halos; correct halo linkage uses `(FILE_NUM, HALO_INDEX)`. Keeping IDs + wrapping + slab product consistent is essential before interpreting “feature ↔ eigenvalue” correlations.

