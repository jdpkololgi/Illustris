# Cache training outline — new stage-3 mock wedges

Products live under `/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/`.
Graph + SBI caches go under `graph_constructions/` and `sbi_caches/` (same layout as the post-collision wedge).

**Status (2026-05-19):** NPZ + `wedge_targets.fits` exist for both wedges. **Per-wedge Delaunay graphs and `gnn_arrays.npz` are not built yet.**

Canonical reference (existing end-to-end stack):  
`$HOME/TNG/Illustris/workflows/abacus_tweb/BUILD_JRAPH_CACHES.md` — `staged_mock_wedge_stage3_postcollision_rs7`.

---

## Wedge catalogues

| Tag | NPZ | `wedge_targets.fits` | n_gal | RA | Dec | z |
|-----|-----|----------------------|-------|----|-----|---|
| `ra120_140_dec16p5_26p7_z0p2_0p3` | `staged_mock_wedge_stage3_ra120_140_dec16p5_26p7_z0p2_0p3_rs7.npz` | `staged_mock_wedge_stage3_ra120_140_dec16p5_26p7_z0p2_0p3_wedge_targets.fits` | 40,339 | 120–140 | 16.5–26.7 | 0.2–0.3 |
| `wedge2_ra128_138_dec18_25_z0_0p5` | `staged_mock_wedge_stage3_wedge2_ra128_138_dec18_25_z0_0p5_rs7.npz` | `staged_mock_wedge_stage3_wedge2_ra128_138_dec18_25_z0_0p5_wedge_targets.fits` | 47,593 | 128–138 | 18–25 | 0.0–0.5 |

Manifests: `*_rs7.manifest.json` (class fractions, join stats).

### Wedge 2 sky-box flag

**Recommended** wedge-2 box (from `build_staged_mock_wedge_variants.py` / optional JSON): **RA 127–133, Dec 20–24**, z 0–0.5.

**Currently built NPZ** uses defaults **RA 128–138, Dec 18–25**, z 0–0.5. There is **no** `staged_mock_wedge_stage3_wedge2_ra127_133_dec20_24_z0_0p5_rs7.npz` on disk.

To rebuild wedge 2 with the recommended box:

```bash
cd $HOME/TNG/Illustris/workflows/abacus_tweb
cat > /pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/wedge2_ra_dec_recommendation.json <<'EOF'
{
  "ra_min": 127.0,
  "ra_max": 133.0,
  "dec_min": 20.0,
  "dec_max": 24.0,
  "z_min": 0.0,
  "z_max": 0.5,
  "tag": "wedge2_ra127_133_dec20_24_z0_0p5"
}
EOF
env -u PYTHONPATH -u PYTHONHOME \
  /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u build_staged_mock_wedge_variants.py \
  --wedge wedge2 --no-write-xyz
```

---

## Shared environment

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module purge || true
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd $HOME/TNG/Illustris
WEDGE=/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge
GRAPH=/pscratch/sd/d/dkololgi/abacus/graph_constructions
CACHE=/pscratch/sd/d/dkololgi/abacus/sbi_caches
```

Use **CPU** nodes for graph build + cache; **GPU** for `abacus_graph_features_cugraph.py` and `jraph_pipeline.py`.

---

## Per-wedge pipeline

Replace `<TAG>` and `<PREFIX>`:

| Wedge | `<TAG>` | `<PREFIX>` |
|-------|---------|------------|
| 1 | `ra120_140_dec16p5_26p7_z0p2_0p3` | `staged_mock_wedge_stage3_ra120_140_dec16p5_26p7_z0p2_0p3_rs7` |
| 2 | `wedge2_ra128_138_dec18_25_z0_0p5` | `staged_mock_wedge_stage3_wedge2_ra128_138_dec18_25_z0_0p5_rs7` |

Paths:

- Truth NPZ: `$WEDGE/staged_mock_wedge_stage3_<TAG>_rs7.npz`
- Targets FITS: `$WEDGE/staged_mock_wedge_stage3_<TAG>_wedge_targets.fits` (rows sorted by FILE_NUM, HALO_INDEX, BOX_INDEX)

### Step 1 — Export comoving points + Delaunay graph

NPZs have `ra`, `dec`, `z` only. Either extend `export_staged_mock_wedge_points.py` to convert sky → Mpc with Planck18, or write `$GRAPH/<PREFIX>_points_xyz.npy` in the same row order as the NPZ.

```bash
# Example: after points_xyz exists (N,3) comoving Mpc, node order = NPZ order)
srun -n1 -c32 $PY workflows/abacus_tweb/build_abacus_graph.py \
  --points-path $GRAPH/<PREFIX>_points_xyz.npy \
  --out-dir $GRAPH \
  --out-prefix <PREFIX> \
  --mode delaunay
```

Produces: `<PREFIX>_edges_combined_idx.npy`, `<PREFIX>_metadata.json`, tetrahedra arrays, etc.  
(`cosmic_env`; alpha_sq = ∞ for Delaunay mode.)

**Reference (post-collision wedge, 17,957 nodes):**  
`$GRAPH/staged_mock_wedge_stage3_postcollision_rs7_edges_combined_idx.npy`

### Step 2 — cuGraph features → GNN arrays

```bash
srun --gpus=1 $PY workflows/abacus_tweb/abacus_graph_features_cugraph.py \
  --artifacts-dir $GRAPH \
  --prefix <PREFIX> \
  --metadata-path $GRAPH/<PREFIX>_metadata.json
```

Produces:

- `$GRAPH/<PREFIX>_cugraph_gnn_arrays.npz`
- `$GRAPH/<PREFIX>_cugraph_gnn_metadata.json`

**Reference:** `staged_mock_wedge_stage3_postcollision_rs7_cugraph_gnn_metadata.json`

### Step 3 — Verify `wedge_targets.fits` row order

```bash
$PY - <<'PY'
import numpy as np, fitsio
from pathlib import Path
tag = "<TAG>"  # set per wedge
w = Path("/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge")
npz = np.load(w / f"staged_mock_wedge_stage3_{tag}_rs7.npz")
tab = fitsio.read(w / f"staged_mock_wedge_stage3_{tag}_wedge_targets.fits")
assert len(tab) == len(npz["ra"])
assert len(tab) == int(npz["n_gal"])
print("OK", len(tab), "rows aligned")
PY
```

Graph node `i` must match FITS/NPZ row `i` (same stable sort by halo triple).

### Step 4 — (Optional) T-Web annotate for 15-d regression

For **15-d** targets (derivatives), run `annotate_cutsky_with_tweb_eigs.py` on a science export FITS in graph order, then copy/write `$GRAPH/<PREFIX>_wedge_targets.fits` with 15-d columns (mirror post-collision flow in `BUILD_JRAPH_CACHES.md`).

For **3-d** transformed-increment cache, NPZ λ columns are enough (see `build_staged_mock_wedge_sbi_cache.py`).

### Step 5 — `build_abacus_sbi_cache.py`

**3-d (NPZ truth, no annotate):**

```bash
srun -n1 -c32 $PY workflows/abacus_tweb/build_abacus_sbi_cache.py \
  --gnn-metadata-path $GRAPH/<PREFIX>_cugraph_gnn_metadata.json \
  --targets-npz-path $WEDGE/staged_mock_wedge_stage3_<TAG>_rs7.npz \
  --no-apply-y1y5-filter --no-exclude-invalid-box-index \
  --three-targets-only \
  --output-cache-path $CACHE/<PREFIX>_sbi_cache.pkl
```

**15-d (annotated wedge_targets in graph_constructions):**

```bash
srun -n1 -c32 $PY workflows/abacus_tweb/build_abacus_sbi_cache.py \
  --gnn-metadata-path $GRAPH/<PREFIX>_cugraph_gnn_metadata.json \
  --targets-catalog-path $GRAPH/<PREFIX>_wedge_targets.fits \
  --no-apply-y1y5-filter --no-exclude-invalid-box-index \
  --output-cache-path $CACHE/<PREFIX>_sbi_cache_15d.pkl
```

**Reference Abacus training wedge (annotated CutSky, different z slice 0.25–0.30):**

- GNN: `abacus_delaunay_wedge_ra120_140_dec16p5_26p7_z0p25_0p30_rs7_15d_cugraph_gnn_metadata.json`
- Cache: `abacus_delaunay_wedge_ra120_140_dec16p5_26p7_z0p25_0p30_rs7_15d_sbi_cache.pkl`

**Reference staged mock (post-collision):**

```bash
# see ph000/wedge/run_stage3_sbi_cache_15d.sh
--gnn-metadata-path .../staged_mock_wedge_stage3_postcollision_rs7_cugraph_gnn_metadata.json
--targets-catalog-path .../staged_mock_wedge_stage3_postcollision_rs7_wedge_targets.fits
--output-cache-path .../staged_mock_wedge_stage3_postcollision_rs7_sbi_cache_15d.pkl
```

### Step 6 — `jraph_pipeline.py` (regression)

**15-d cache:**

```bash
srun --gpus=1 $PY workflows/jraph/jraph_pipeline.py \
  --prediction_mode regression \
  --cache_path $CACHE/<PREFIX>_sbi_cache_15d.pkl \
  --no_transformed_eig \
  --output_dir /pscratch/sd/d/dkololgi/abacus/jraph_runs/<PREFIX>_15d
```

**3-d cache:** omit `--no_transformed_eig` unless cache was built with `--no-transformed-eig`.

---

## Related Abacus wedge (not stage-3 mock)

Subset of full CutSky graph (Y1|Y5 + R_MAG), different z window — useful calibration reference only:

| Artifact | Path |
|----------|------|
| Subset script | `workflows/abacus_tweb/subset_abacus_graph_wedge_for_sbi.py` |
| GNN metadata | `$GRAPH/abacus_delaunay_wedge_ra120_140_dec16p5_26p7_z0p25_0p30_rs7_15d_cugraph_gnn_metadata.json` |

Do **not** mix that graph with stage-3 mock NPZ row order; build a **dedicated** graph per table above.

---

## Visualization

Notebook: `$HOME/TNG/Illustris/workflows/abacus_tweb/visualize_staged_mock_stage3_new_wedges_3d.ipynb`  
HTML: same directory, `visualize_staged_mock_stage3_new_wedges_3d.html`
