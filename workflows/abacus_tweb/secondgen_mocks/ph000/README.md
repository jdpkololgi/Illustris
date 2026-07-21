# SecondGen Abacus mocks — `ph000` staging

This directory supports linking **SecondGen mock galaxies** to **T-Web eigenvalues** derived from the **base simulation particle density field**, ahead of **GNN / cosmic web** analysis. It is **not** a DisPerSE-focused project.

**Upstream DESI LSS scripts** under `scripts/` (`upstream_*`) are **copies for reference and stage isolation** of survey effects relevant to mock realism: **footprint**, **fibre assignment / collisions**, and **RSD**. They are **not** maintained here for DisPerSE workflows.

Downstream tooling typically expects paths under NERSC CFS/SCRATCH and the `desi_environment` stack.

## Glossary (paths and names)

| Term | Meaning |
|------|--------|
| **cosmosim CutSky** | Stage-0 truth input with halo linkage, e.g. `/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/.../cutsky_BGS_..._ph000.fits`. This repo’s science defaults to **that** file (this phase) unless you deliberately switch phase. |
| **DA2 `AbacusSummitBGS_v2`** | Survey-style mock **products** under a **different** path tree (`survey/catalogs/DA2/...`), built from second-gen Abacus mocks. **Do not confuse** with the string passed as LSS **`mockver`**: LSS uses flags such as `ab_secondgen_cosmosim` vs `ab2ndgen` for `pota` / tooling, which are **not** the same token as a filesystem directory name like `AbacusSummitBGS_v2`. |
| **`ph000` ↔ `mock0` / `forFA0`** | Keep the Abacus phase index and mock realization aligned: `..._ph000.fits` ↔ realization `0` and `forFA0.fits`-style outputs so halo links and fibreassign products stay consistent. |

## Mock release / layout variants on CFS

Names like **`AbacusSummit_v4_1`** (and similar) denote other mock **release or layout** variants that may appear under paths such as `survey/catalogs/DA2/mocks/SecondGenMocks/`. This staging tree targets **cosmosim CutSky** for **`ph000`**; switch CutSky phase and downstream paths together if you change realization.

## Directory layout

| Path | Role |
|------|------|
| `stage_1/` | CutSky subset outputs (`stage1_cutsky_subset.py` default `--out`). |
| `stage_2/` | Intermediate targets / `forFA*.fits`-style products (see upstream `prepare_mocks_Y3_bright.py`, `getpotaDA2_mock.py`). |
| `stage_3/` | Mock catalogs / alt MTL–related outputs from `mkCat_SecondGen_amtl.py`. |
| `stage_4/` | **Not used** in this pipeline spec (no fourth processing stage defined here); empty placeholder only. |
| `scripts/` | Local wrappers + **upstream** copies of DESI LSS `mock_tools` scripts (prefixed `upstream_`). |

## `ph000` vs `mock0` naming

- **`ph000`**: Abacus CutSky files use `..._ph000.fits` for this simulation phase.
- **`mock0`**: LSS / fiberassign examples often pass `--realization 0`, producing paths like `mock0` or `forFA0.fits`.
- **Rule of thumb**: use the **same integer** for Abacus phase and mock realization (`ph000` ↔ realization `0` ↔ `forFA0`) unless you deliberately mix phases.

## NGC / SGC (reference geometry) vs DR2 tiles

- **`sim_NGC` / `sim_SGC` in `Y3-mocks-DisPerSE-runs-1.ipynb`**: simple **RA/Dec rectangles** on `Tb_sim` to split a *simulation* catalog. They are **not** the DR2 tile footprint; they are reproduced here only as a **convenient sky/magnitude mask** for early subsets.
- **DR2 / survey geometry**: `prepare_mocks_Y3_bright.py` can apply DESI targeting masks (`--apply_mask`), imaging-related bits, and fiberassign tile logic — that is the **observational** footprint, distinct from the notebook cap masks.

Exact notebook expressions on `Tb_sim` (verbatim):

```python
sim_SGC = ((Tb_sim['RA'] < 40.) | (Tb_sim['RA'] > 330.)) & (Tb_sim['DEC'] > -15) & (Tb_sim['DEC'] < 30) & (Tb_sim['R_MAG_APP'] < 19.5)
sim_NGC = (Tb_sim['RA'] < 270.) & (Tb_sim['RA'] > 120.) & (Tb_sim['DEC'] > -5) & (Tb_sim['DEC'] < 75) & (Tb_sim['R_MAG_APP'] < 19.5)
```

`scripts/stage1_cutsky_subset.py` applies the same geometry for `--cap NGC|SGC`, with the magnitude edge tied to `--rbandcut` (default `19.5`, matching the notebook).

## Pipeline stages (1–3)

1. **Stage 1 — CutSky subset**  
   - Run `scripts/stage1_cutsky_subset.py` on `cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits` (or your symlinked copy).  
   - Produces a compact FITS with native **`Z`** (observed / RSD) and **`Z_COSMO`** (true), plus the **full** SecondGen BGS CutSky column set (magnitudes/colors, **`FILE_NUM`**, **`HALO_INDEX`**, **`BOX_INDEX`**, footprint flags, etc.) after the magnitude and optional cap mask.

2. **Stage 2 — Targets for fiberassign + potential assignments**  
   - Use upstream `upstream_prepare_mocks_Y3_bright.py` to build `forFA{real}.fits` under your `--base_output` SecondGen tree (requires `LSS`, `desitarget`, correct `NERSC_HOST`, and typically `source /global/common/software/desi/desi_environment.sh main`). The patched script keeps **`Z`** and **`Z_COSMO`** as the primary science columns and adds **`TRUEZ`** / **`RSDZ`** as **float32 duplicates** of **`Z_COSMO`** / **`Z`** for LSS / `mkfulldat` (`mockz='RSDZ'`) and fiberassign-style expectations, without any **`Z_OBS`** column.  
   - Then `upstream_getpotaDA2_mock.py` for potential assignments / tile-related products (`--mock ab2ndgen`, `--realization`, `--prog BRIGHT`, etc., matching your output layout).

3. **Stage 3 — Mock data vectors / MTL pipeline**  
   - Run `upstream_mkCat_SecondGen_amtl.py` with `--mockver`, `--base_output`, `--pota`, `--targDir`, and switches appropriate to DA2 / your `simName` path (see script `--help` and LSS docs).

## Upstream scripts (GitHub `main`)

Fetched with `curl -fSL --connect-timeout 15 --max-time 180` into `scripts/`:

- `upstream_prepare_mocks_Y3_bright.py` — [`prepare_mocks_Y3_bright.py`](https://raw.githubusercontent.com/desihub/LSS/main/scripts/mock_tools/prepare_mocks_Y3_bright.py)
- `upstream_getpotaDA2_mock.py` — [`getpotaDA2_mock.py`](https://raw.githubusercontent.com/desihub/LSS/main/scripts/mock_tools/getpotaDA2_mock.py)
- `upstream_mkCat_SecondGen_amtl.py` — [`mkCat_SecondGen_amtl.py`](https://raw.githubusercontent.com/desihub/LSS/main/scripts/mock_tools/mkCat_SecondGen_amtl.py)

If a future refetch fails twice, see `FETCH_FAILED.md` (stderr log) and fix network or URLs before proceeding.

## Example commands

```bash
# Stage 1 — full footprint, magnitude cut only
python3 scripts/stage1_cutsky_subset.py \
  --cutsky /global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits \
  --cap ALL --rbandcut 19.5

# Stage 1 — notebook-style NGC / SGC caps
python3 scripts/stage1_cutsky_subset.py --cap NGC
python3 scripts/stage1_cutsky_subset.py --cap SGC
```

Stage 2/3 depend on your chosen `--base_output`, SecondGen version suffix (`AbacusSummitBGS_v2` vs `AbacusSummit` + `mock_version`), and DA2 paths. Thin commented templates: `run_stage2_example.sh`, `run_stage3_example.sh`.

## Redshift columns: `Z`, `Z_COSMO`, and `TRUEZ` / `RSDZ`

- **`Z_COSMO`**: true / cosmological redshift (CutSky name). Used consistently in stage 1 subset FITS and in `forFA*.fits` from the patched prepare script.  
- **`Z`**: observed redshift from CutSky (includes RSD along the line of sight in the mock). Same native name in stage 1 and `forFA*.fits`.  
- **`TRUEZ` / `RSDZ`**: optional **duplicates** written on `forFA*.fits` so DESI LSS paths that expect legacy names (e.g. `mkfulldat` with `mockz='RSDZ'`) keep working; values match **`Z_COSMO`** and **`Z`** respectively. There is **no** `Z_OBS` column in this ph000 workflow.

After fiberassign and spec pipeline steps, columns may still be joined or renamed again; always inspect the FITS you feed to each step.

## Column propagation (CutSky → forFA → pota → LSS)

**Verified CutSky schema (extension 1, `fitsio`, 2026-05-13):** file  
`/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits`  
has 24 columns with dtypes: `R_MAG_APP`, `R_MAG_ABS`, `G_R_REST`, `G_R_OBS`, `HALO_MASS`, `Z_COSMO`, `Z` as **float32** (`>f4`); `RA`, `DEC` as **float64** (`>f8`); `CEN`, `RES`, `FILE_NUM`, `HALO_INDEX`, `BOX_INDEX`, and all `IN_Y1` / `NGC_Y1` / … / `S_Y5` fields as **int32** (`>i4`). If a checklist assumed **int16** for bitmask or linkage integers, the on-disk types are **int32** (not a name mismatch).

| Output | Columns |
|--------|---------|
| **Stage 1** (`stage1_cutsky_subset.py`) | Full CutSky set above; **`Z`** and **`Z_COSMO`** kept with CutSky names. |
| **`forFA{real}.fits`** (`upstream_prepare_mocks_Y3_bright.py`, `ab_secondgen_cosmosim` / generic cutsky read) | All CutSky fields read in the prepare script, plus LSS targets columns (`TARGETID`, `DESI_TARGET`, `BGS_TARGET`, …). **`Z`**, **`Z_COSMO`**, duplicate **`TRUEZ`**, **`RSDZ`**, and **`ABSMAG_R`** / **`REST_GMR_0P1`** aliases for bright downstream. |
| **`pota-*.fits`** (`upstream_getpotaDA2_mock.py`, `ab2ndgen`) | All columns present on `forFA` are carried on per-tile target FITS and into the `pota` table via `tarcols` / joins (no separate allowlist beyond FITS content). |
| **COMBD / `datcomb_*`** (`upstream_mkCat_SecondGen_amtl.py`) | **`FORFA_BRIGHT_TAGALONG`** extends FASSIGN/FAVAIL joins and **`pota_cols`** when `usepota=y` so linkage, redshift, and survey fields from `forFA` merge onto tile-level products. |
| **Clustering data** (`mkclusdat`) | For **`BGS*`** tracers the patched script passes **`extracols=mkclus_extracols`** (derived from **`FORFA_BRIGHT_TAGALONG`**) into `LSS.main.cattools.mkclusdat` **if** your installed **`LSS`** package honors that keyword; which columns survive in `clustering.dat.h5` is defined there, not only in this copied script. |
| **Clustering randoms** (`mkclusran`) | **`rcols`** is built for BGS (base list plus linkage / photometry / redshift names) and passed to **`ct.mkclusran`**; propagation and subsetting are implemented inside **`LSS`**. |

## Clarifying questions (for the user / pipeline owner)

1. Should Stage 1 outputs feed **directly** into a custom `prepare_mocks` path, or only serve T-Web / GNN side analyses while `forFA*.fits` stays from the standard `prepare_mocks_Y3_bright.py` CutSky read?  
2. Which **`--mockver` / `AbacusSummit` directory suffix** (`AbacusSummitBGS_v2`, `AbacusSummit_v4_1`, etc.) matches your DA2 mock production layout?  
3. Do you require **`IN_Y5` / imaging bitmask** columns reintroduced for BGS (some `prepare_mocks` branches were temporarily altered upstream)?  
4. For clustering catalogs, canonical names here are **`Z`** / **`Z_COSMO`**; **`TRUEZ`** / **`RSDZ`** exist only as duplicates for LSS compatibility when needed.
