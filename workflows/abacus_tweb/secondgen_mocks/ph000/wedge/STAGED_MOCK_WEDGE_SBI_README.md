# Staged mock wedge products (stage 3, rs7)

Built with `TNG/Illustris/workflows/abacus_tweb/build_staged_mock_wedge_variants.py`.

## Wedges (2026-05-19)

| Tag | RA | Dec | z | n_gal (deduped) |
|-----|----|-----|---|-----------------|
| `ra120_140_dec16p5_26p7_z0p2_0p3` | 120–140 | 16.5–26.7 | 0.2–0.3 | 40,339 |
| `wedge2_ra128_138_dec18_25_z0_0p5` | 128–138 | 18–25 | 0.0–0.5 | 47,593 |

Per wedge: `staged_mock_wedge_stage3_<tag>_rs7.npz`, `*_wedge_targets.fits` (rows sorted by FILE_NUM, HALO_INDEX, BOX_INDEX; TARGETID = packed triple), `*_manifest.json`.

Filters: `datcomb_brightwdup.fits`, sky wedge, `COLLISION==0`, unique (FILE_NUM, HALO_INDEX, BOX_INDEX); eigenvalues from annotated CutSky join.

## Next step for `build_abacus_sbi_cache.py`

The cache builder needs **graph node arrays** aligned to targets:

1. Run mock graph construction / `subset_abacus_graph_wedge_for_sbi.py` (or equivalent) on the **stage-3 exported point set** for each wedge.
2. Produce `<prefix>_cugraph_gnn_metadata.json` (+ NPZ node/edge files) with the same row count and order as the wedge FITS/NPZ.
3. Then either:
   - `build_abacus_sbi_cache.py --gnn-metadata-path ... --targets-catalog-path ..._wedge_targets.fits`, or
   - `build_staged_mock_wedge_sbi_cache.py` with `--targets-npz-path ..._rs7.npz` once metadata exists.

Wedge2 sky box: drop `wedge2_ra_dec_recommendation.json` in this directory to override RA/Dec defaults (z stays 0–0.5 unless set in JSON).

## Rebuild

```bash
cd $HOME/TNG/Illustris/workflows/abacus_tweb
env -u PYTHONPATH -u PYTHONHOME \
  /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python -u build_staged_mock_wedge_variants.py \
  --wedge both --no-write-xyz
```
