# Staged mock wedge stage 3 — jraph / SBI caches (post-collision, rs7)

## Products (2026-05-19)

| Role | Path |
|------|------|
| Stage-3 datcomb | `/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/datcomb_brightwdup.fits` |
| Wedge truth NPZ (coords + legacy λ from join; **not** used for 15-d cache) | `/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/staged_mock_wedge_stage3_postcollision_rs7.npz` |
| Annotate input (17,957 rows, **graph/NPZ row order**) | `/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/staged_mock_stage3_science_for_annotate.fits` |
| T-Web annotated stage-3 mock | `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/staged_mock_stage3_postcollision_rs7_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits` |
| cuGraph GNN metadata (**keep**) | `/pscratch/sd/d/dkololgi/abacus/graph_constructions/staged_mock_wedge_stage3_postcollision_rs7_cugraph_gnn_metadata.json` |
| Wedge targets FITS (15-d columns, **same order as graph nodes**) | `/pscratch/sd/d/dkololgi/abacus/graph_constructions/staged_mock_wedge_stage3_postcollision_rs7_wedge_targets.fits` |
| **15-d SBI cache** | `/pscratch/sd/d/dkololgi/abacus/sbi_caches/staged_mock_wedge_stage3_postcollision_rs7_sbi_cache_15d.pkl` |

Wedge: RA 120–140, Dec 16.5–26.7, z 0.25–0.3; `COLLISION==0`; dedupe `(FILE_NUM, HALO_INDEX, BOX_INDEX)`; n_gal = **17,957**.

Reference annotated CutSky (full):  
`/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits`

## Removed (incorrect 3-d / NPZ-shortcut artifacts)

- `/pscratch/sd/d/dkololgi/abacus/sbi_caches/staged_mock_wedge_stage3_postcollision_rs7_sbi_cache.pkl`
- `/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/staged_mock_wedge_stage3_postcollision_rs7_annotated.fits` (+ manifest)

## Rebuild annotate (compute node)

```bash
unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
module purge
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd $HOME/TNG/Illustris
$PY workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py \
  --cutsky /pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/staged_mock_stage3_science_for_annotate.fits \
  --output-dir /pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/ \
  --output-name staged_mock_stage3_postcollision_rs7_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits \
  --overwrite
```

## Rebuild 15-d cache (graph unchanged)

```bash
unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
module purge
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd $HOME/TNG/Illustris
$PY workflows/abacus_tweb/build_abacus_sbi_cache.py \
  --gnn-metadata-path /pscratch/sd/d/dkololgi/abacus/graph_constructions/staged_mock_wedge_stage3_postcollision_rs7_cugraph_gnn_metadata.json \
  --targets-catalog-path /pscratch/sd/d/dkololgi/abacus/graph_constructions/staged_mock_wedge_stage3_postcollision_rs7_wedge_targets.fits \
  --no-apply-y1y5-filter --no-exclude-invalid-box-index \
  --output-cache-path /pscratch/sd/d/dkololgi/abacus/sbi_caches/staged_mock_wedge_stage3_postcollision_rs7_sbi_cache_15d.pkl
```

Verified: `regression_targets.shape == (17957, 15)`, `stats is None`.

## jraph regression training (15-d cache)

Run on a **GPU** compute node; use the 15-d cache and `--no_transformed_eig` (cache already holds assembled/scaled 15-d targets).

```bash
unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
module purge
# module load ...  # GPU / jax cuda as for your usual jraph jobs
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
cd $HOME/TNG/Illustris
$PY workflows/jraph/jraph_pipeline.py \
  --cache_path /pscratch/sd/d/dkololgi/abacus/sbi_caches/staged_mock_wedge_stage3_postcollision_rs7_sbi_cache_15d.pkl \
  --prediction_mode regression \
  --no_transformed_eig \
  --output_dir /pscratch/sd/d/dkololgi/abacus/jraph_runs/staged_mock_wedge_stage3_postcollision_rs7_15d
```

Optional: `--heteroscedastic_15d` for mean+logvar head (30 outputs).
