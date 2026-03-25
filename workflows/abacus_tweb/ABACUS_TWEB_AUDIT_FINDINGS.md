# Abacus CutSky T-Web Audit Findings

## Coordinate Clarification (important)

RA/DEC/Z_obs are **sky coordinates**, not Cartesian coordinates.

The correct pipeline concept is:

1. Convert `(RA, DEC, Z_obs)` to 3D Cartesian comoving coordinates (using a chosen cosmology).
2. Build graph features in that Cartesian space (this preserves RSD through `Z_obs`).
3. Assign T-Web labels in a way that is geometrically consistent with how the CutSky mock was generated.

So the issue is **not** "treat RA/DEC/Z as Cartesian directly". The issue is that a naive periodic modulo map after conversion can be inconsistent with CutSky light-cone/remap geometry.

---

## Key Paths (from config)

- **CutSky mock FITS**  
  `/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits`

- **Base Abacus simulation root**  
  `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000`

Relevant config source:
- `TNG/Illustris/shared/config_paths.py`

---

## What We Observed

- T-Web slab metadata checks pass (coverage, shapes, sorting).
- Eigenvalue ordering checks pass (`lambda1 <= lambda2 <= lambda3`).
- But physical signal checks are poor on Abacus CutSky:
  - continuity ratio neighbor/random ~ 1 (noise-like local coherence),
  - Pearson and MI between graph metrics and eigenvalues are near zero.
- In contrast, the same signal checks on IllustrisTNG show strong feature-target coupling.

Interpretation: this strongly points to a **label-domain mismatch / label quality issue** for current Abacus CutSky annotation, not only model/training failure.

---

## Why Labels Can Look Random Even if CutSky Comes from CubicBox

Even though CutSky mocks originate from Abacus cubic simulations ([arXiv:2312.08792](https://arxiv.org/abs/2312.08792)), labels can decorrelate if mapping assumptions are wrong:

1. **Many-to-one periodic folding**  
   Mapping transformed sky coordinates into one cube with `% boxsize` can collapse different physical light-cone locations/replicas onto the same voxel.

2. **Light-cone/remap is not trivially invertible**  
   CutSky creation generally involves geometry/remap/selection operations. If these are not exactly inverted, voxel lookup is effectively scrambled.

3. **RSD-vs-real-space inconsistency**  
   Graph uses `Z_obs` (redshift-space), while T-Web labels are often computed from real-space density. This naturally weakens local feature-label relation along LOS.

4. **Grid/noise regime likely too aggressive**  
   `ngrid=3414` with a 3% particle sample is likely too sparse per cell; with NGP and low smoothing this can produce noisy Hessian eigenvalues.

---

## Code-Level Audit Notes

- `annotate_cutsky_with_tweb.py` currently uses:
  - Planck18 comoving distance conversion,
  - fixed observer origin `[-990, -990, -990]`,
  - periodic modulo map into one box.

- `abacus_process_particles2.py` default MPI slab workflow currently uses:
  - `ngrid=3414`,
  - NGP deposition in the default run path,
  - T-Web smoothing configured separately (`RSMOOTH=2.0` in `abacus_cactus_tweb.py`).

These are plausible contributors to noisy/weak labels in CutSky training context.

---

## Recommended Next Steps (minimal and high value)

1. **Recompute T-Web with less noisy field settings**
   - test `ngrid=512` and `768`,
   - use `CIC` (or `TSC`) deposition,
   - test `Rsmooth=4` and `8`.

2. **Keep using `Z_obs` for graph construction** (required for DESI/BGS use-case).

3. **Make label mapping explicit and configurable**
   - configurable redshift column (`Z` vs `Z_COSMO`),
   - configurable cosmology/origin,
   - replace hardcoded modulo mapping with CutSky-consistent remap once exact construction details are confirmed.

4. **Run the same QC after each label variant**
   - continuity ratio,
   - Pearson/MI feature-label signal,
   - quick tabular baseline sanity.

---

## Bottom Line

You are correct to require CutSky + RSD for deployment realism.  
The current evidence suggests the main failure mode is likely in **how labels are being constructed/assigned in CutSky geometry**, plus possibly an overly noisy T-Web grid configuration.

---

## What We Saw in SBI and Regression-GNN Training

These model-level results are consistent with the data/label diagnosis above:

- **Partitioned FlowJAX-SBI (GraphNet + flow) on Abacus**
  - Training ran stably after engineering fixes (partitioning, distributed/pmap, mixed precision, checkpoint/resume).
  - Loss dropped early and then plateaued quickly.
  - Posterior diagnostics showed compressed predictions near the mean (small-magnitude eigenvalue outputs), weak calibration/coverage behavior, and poor predictive structure.

- **Regression-GNN micro-overfit tests**
  - Even on tiny subsets/single partitions, the GNN regression objective did not strongly overfit as expected.
  - This reduced confidence that the bottleneck is purely optimization or distributed training mechanics.

- **Tabular baselines on the same feature set**
  - Mean/linear/MLP baselines on the Abacus graph metrics also showed weak explanatory power.
  - Full-scale tabular MLP on Y1/Y5-scale data achieved very low explanatory signal (R2 near zero overall), reinforcing that current feature-target coupling is weak.

- **A/B checks against IllustrisTNG**
  - Equivalent checks on Illustris showed substantially stronger feature-target signal and better learnability.
  - This points away from a generic bug in the modeling stack and toward an Abacus CutSky label/domain mismatch (plus noise regime).

Practical interpretation: the poor SBI and regression-GNN outcomes look like **symptoms** of low-quality or mismatched supervision rather than the primary root cause.

---

## Mock Catalog Linkage Findings (new)

### What exists in the CutSky FITS schema

The CutSky file includes:

- `FILE_NUM`
- `HALO_INDEX`
- `BOX_INDEX`
- `Z` and `Z_COSMO`

File audited:
- `/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits`

### Direct recovery path for host-halo box-frame position

A practical non-inversion path exists:

1. Use `(FILE_NUM, HALO_INDEX)` from CutSky row.
2. Open corresponding base halo file:
   - `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.200/halo_info/halo_info_{FILE_NUM:03d}.asdf`
3. Read halo center coordinates (`x_com` or `x_L2com`) at row `HALO_INDEX`.

This recovers host-halo box-frame position **without** inverting light-cone sky coordinates.

### Notes on coordinate units

- `halo_info` coordinates were found in normalized box units near `[-0.5, 0.5)`.
- Convert to physical box coordinates consistently using `BoxSize` from halo header metadata.

### `BOX_INDEX` status

- `BOX_INDEX` is present and appears to be an internal linkage key.
- It did **not** behave as a simple direct row index into `CubicBox/BGS_box_ph000.fits` under naive lookup.
- Recommended: prioritize `(FILE_NUM, HALO_INDEX)` for robust host-halo mapping.

---

## Consistency Check vs Current `mocks_with_eigs` Implementation

Current implementation in `annotate_cutsky_with_tweb.py`:

- computes Cartesian coordinates from `RA, DEC, Z_COSMO`,
- applies fixed observer-origin shift,
- applies `% boxsize`,
- does voxel lookup in T-Web slabs.

So current `mocks_with_eigs` is using **geometric inversion/modulo mapping**, not the halo-index linkage method.

### Important inconsistency identified

- Graph construction path (`build_abacus_graph.py`) is configured to use observed `Z` by default (RSD-preserving).
- Label assignment path (`annotate_cutsky_with_tweb.py`) currently uses `Z_COSMO`.

This creates a feature/label domain mismatch (RSD-in-features vs non-RSD-style label mapping), which is a plausible source of weak learnability.

---

## Role of abacusutils / abacusnbody

`abacusutils` (via `abacusnbody`) is directly useful here:

- robust reading of Abacus halo ASDF products,
- schema-aware access to halo properties and metadata,
- avoids brittle manual decoding.

Reference:
- [abacusutils documentation](https://abacusutils.readthedocs.io/en/latest/)

---

## Updated Bottom Line

The strongest current hypothesis is:

1. The main failure mode is **label construction/assignment mismatch** for CutSky geometry and RSD usage.
2. Current modulo-based sky inversion likely injects effective label noise.
3. A better path is to use direct host-halo linkage via `(FILE_NUM, HALO_INDEX)` where possible, while also reducing T-Web field noise (`ngrid`, assignment kernel, smoothing).

---

## Practical File/Field Map (what is needed for what)

Tie-in to CutSky mock fields:

- Sky fields: `RA`, `DEC`, `Z`, `Z_COSMO`
- Linkage fields: `FILE_NUM`, `HALO_INDEX`, `BOX_INDEX`
- Selection fields: `IN_Y1`, `IN_Y5`

Which base products are used:

- `halo_info/halo_info_{FILE_NUM:03d}.asdf`
  - Use for host-halo linkage via `HALO_INDEX` to recover halo position (`x_com` or `x_L2com`) and optional halo properties.
- `field_rv_A/` + `halo_rv_A/`
  - Use for density-field construction for T-Web.
- `field_pid_A/` + `halo_pid_A/`
  - Only needed for particle identity tracing (not required for current annotation/labeling workflow).
- `header`
  - Snapshot metadata (box size, cosmology, particle mass, output settings) for consistency checks and conversions.

---

## Current Status and Near-Term Plan

As seen in `annotate_cutsky_with_tweb.py`, T-Web has already been computed and stored slabwise, and annotation currently maps each galaxy to slab-local voxel indices using sky-coordinate conversion + periodic modulo.

Planned sequence:

1. **Change annotation method first** (without rebuilding density):
   - swap to host-halo linkage path using `(FILE_NUM, HALO_INDEX)` to recover box-frame halo position,
   - assign existing slab eigenvalues at that linked position,
   - regenerate annotated mock and re-run correlation/continuity diagnostics.
2. **Only if signal remains weak**, rebuild density/T-Web with less noisy settings:
   - smaller `ngrid`,
   - CIC/TSC assignment,
   - larger smoothing (`Rsmooth`).

This isolates mapping/label-assignment effects before expensive density recomputation.

---

## Clear Pathway: Get Gridpoint Eigenvalues for a Given Mock Galaxy

There are two pathways.

### Path A (current implementation in `annotate_cutsky_with_tweb.py`)

1. Read galaxy row fields: `RA`, `DEC`, `Z_COSMO`.
2. Convert to Cartesian comoving coordinates.
3. Apply observer-origin shift and periodic modulo into `[0, boxsize)`.
4. Convert `(x,y,z)` to voxel indices `(ix,iy,iz)`.
5. Use slab map `ix -> slab_id` and local x-index `lix = ix - slab_xstart[slab_id]`.
6. Read slab file `abacus_cactus_tweb_rank*.npz`.
7. Assign:
   - `CWEB = cweb[lix, iy, iz]`
   - `LAMBDA1/2/3 = eig_vals[0/1/2, lix, iy, iz]`.

### Path B (recommended test path before new density build)

1. Read galaxy row fields: `FILE_NUM`, `HALO_INDEX`.
2. Open base halo file: `halo_info/halo_info_{FILE_NUM:03d}.asdf`.
3. Fetch host halo position from row `HALO_INDEX`:
   - `x_com` or `x_L2com` (and convert units consistently to box frame if needed).
4. Convert halo `(x,y,z)` to voxel indices `(ix,iy,iz)`.
5. Use the same slab/local indexing and lookup:
   - `CWEB`, `LAMBDA1`, `LAMBDA2`, `LAMBDA3`.

Decision point:
- If Path B materially improves feature-label correlation and local continuity, mapping was the primary issue.
- If not, proceed to density/T-Web rebuild with lower-noise field settings.

