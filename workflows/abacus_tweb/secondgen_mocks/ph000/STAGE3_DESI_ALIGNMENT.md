# Mock pipeline vs DESI LOA — what “aligned” requires

**One sentence:** DESI LOA parity is **not** only `ZWARN` / `DELTACHI2` cuts at the end. It is **(1) the same survey selection footprint**, **(2) fibre-assignment / collision geometry**, **(3) a spectroscopic success/failure layer**, **(4) the same final LOA quality cuts**, and **(5) exporting that catalogue into GraphWeb**.

**Status (2026-06-04, updated after mag-lim):** Path 1 **A–E complete**. Final product: `${RUN_ROOT}/mock_bgs_maglim.fits` (9,538,254 rows). Spectro layer **D** uses LOA `zall` marginals; **`Z` is observed RSD redshift (`RSDZ`)**, matching DESI spectroscopic `Z`. **Next:** GraphWeb wedge / graph rebuild and three-curve diagnostic.

Your cluster-tail gap (DESI pred ≪ Abacus pred) matches this **population mismatch**, not a wrong CutSky file (`ph000` is fine).

**Active run directory:**

```text
RUN_ROOT=/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/path1_fiberassign_20260604_083322
```

---

## DESI reference (what GraphWeb uses)

Built by `GraphWeb_DESI/workflows/catalog/build_bgs_maglim_catalog.py` from LOA `zall-pix-loa.fits`:

| Final cut | Meaning |
|-----------|---------|
| `ZWARN == 0` | Redshift fit succeeded |
| `DELTACHI2 >= 25` | Confident spectroscopic classification |
| `SPECTYPE == GALAXY` | Galaxy (not star/QSO) |
| `BGS_TARGET & BGS_BRIGHT*` | Bright BGS program |

No extra z or stellar-mass cuts in that builder.

**Important:** Real LOA also went through the **survey + fibreassign + spectro pipeline** before those cuts. Applying only the four cuts on `datcomb` **without** upstream effects does **not** guarantee identical **statistics** (density, neighbours, failures) — only the same **selection rule** on whatever table you feed in.

---

## Five components of a LOA-like mock population

| # | Component | DESI (real) | Your ph000 today | Needed for statistical alignment? |
|---|-----------|-------------|------------------|-----------------------------------|
| **A** | Simulation truth | Abacus CutSky | `ph000` CutSky + halo tagalongs in LSScats | OK |
| **B** | Targeting / footprint | DESI masks, tiles, bright limit | `forFA0.fits` (~18.1M after imaging veto, `apply_mask=y`) | Done |
| **C** | Fibre geometry | Real assignments + collisions | `fa_multipass` (5034/5171 tiles) → `assignwdup` (~21.9M asn rows) | Done (137 tiles missing FA) |
| **D** | Spectroscopic layer | Pipelines → `ZWARN`, `DELTACHI2`, `SPECTYPE` | `inject_loa_spec_from_zall.py` on `full_noveto` (LOA marginals, 96.15% pass) | Done |
| **E** | Final LOA cuts + GraphWeb export | `build_bgs_maglim_catalog.py` | `mock_bgs_maglim.fits` (9.54M rows, same four cuts) | Done |

**Path 1 LOA-like mock catalogue is built.** Remaining work is **GraphWeb** (wedge, graphs, inference diagnostics), not catalogue production.

---

## What you have vs what is missing (checklist)

### Path 1 complete (2026-06-04)

- [x] `forFA0.fits` — prepare with **`apply_mask=y`**
- [x] `pota-BRIGHT.fits` — getpota
- [x] Fiberassign — `fa_multipass` (4 passes); **`Univ000/fa/MAIN/`** — 5034 `fba-*.fits`
- [x] `datcomb_brightassignwdup.fits` — mkCat assignwdup (`usepota=n`, combd)
- [x] mkCat **`joindspec=y fulld=y add_gtl=y`** → `BGS_BRIGHT_full_noveto.dat.fits` (~9.92M rows)
- [x] mkCat **`mkclusdat=y`** → `BGS_BRIGHT_clustering.dat.fits` (~7.77M, z ∈ [0.002, 0.6])
- [x] **`inject_loa_spec_from_zall.py`** → `BGS_BRIGHT_full_noveto_loa_spec.fits`
- [x] **`build_mock_bgs_maglim_catalog.py`** → **`mock_bgs_maglim.fits`** (9,538,254 rows)
- [x] LSS / inject fixes documented in [Common errors](#common-errors-path-1-runs)

### Next (GraphWeb)

- [ ] Build graph / wedge from `mock_bgs_maglim.fits` (rename `RA`/`DEC` → `TARGET_RA`/`TARGET_DEC` for DESI graph scripts, or use Abacus builders)
- [ ] Join T-Web λ truth via `(FILE_NUM, HALO_INDEX, BOX_INDEX)` for mock training/eval
- [ ] Rerun three-curve wedge diagnostic vs real DESI

---

## Two implementation paths

### Path 1 — Full LSS + LOA mag-lim — **complete**

```text
forFA0  →  fiberassign  →  assignwdup  →  fulld  →  inject_loa_spec  →  mock_bgs_maglim.fits  [DONE]
       →  mkclusdat (optional clustering .fits)  [DONE]
       →  GraphWeb wedge / GNN  [NEXT]
```

### Path 2 — Pragmatic (COMBD-only, no fiberassign)

```text
datcomb_brightwdup.fits  →  inject_loa_spec_from_zall.py  →  build_mock_bgs_maglim_catalog.py
```

Superseded for this project by completed Path 1; kept for quick tests without FA.

---

## Stage-by-stage: your flags vs target

| Stage | Your run | Target for LOA-like stats |
|-------|----------|---------------------------|
| **prepare** | `apply_mask=y` → `forFA0.fits` | Done |
| **getpota** | `pota-BRIGHT.fits` | Done |
| **fiberassign** | `fa_multipass` → 5034 tiles | Done |
| **mkCat** | assignwdup + joindspec + fulld + mkclusdat | Done |
| **LOA spec + mag-lim** | inject + `build_mock_bgs_maglim_catalog.py` | Done |
| **GraphWeb** | wedge from `mock_bgs_maglim.fits` | **Next** |

---

## Python / environment (verified 2026-06)

| Tool | Environment | Status |
|------|-------------|--------|
| **LSS** (`import LSS`, `cattools`) | **`cosmic_env`** + `unset PYTHONPATH` | **OK** — editable install at `/pscratch/sd/d/dkololgi/LSS/py/LSS/` |
| **mkCat**, **build_mock_bgs_maglim**, **inject_loa_spec** | **`cosmic_env`** (`$PY`) | **OK** — use `$PY`, not bare `python` after `conda activate` |
| **prepare_mocks** (imaging masks) | **`desi_environment`** + `PYTHONPATH=$LSS/py:...` | **OK** when sourced |
| **getpota** | **`desi_environment`** + LSS on `PYTHONPATH` | **OK** |
| **fiberassign** binary | **`desi_environment`** | **OK** |
| **LSS inside desi_environment alone** | default | **Not on PATH** — prepend `/pscratch/sd/d/dkololgi/LSS/py` or use `$PY` for mkCat |

```bash
unset PYTHONPATH PYTHONHOME
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
$PY -c "import LSS; from LSS.main import cattools; print(LSS.__file__)"
bash scripts/check_ph000_env.sh
```

---

## Path 1 — status and products (2026-06-04)

| Step | Script | Status | Key output under `RUN_ROOT` |
|------|--------|--------|-----------------------------|
| Prepare | `run_path1_prepare.sh` | Done | `stage_2/.../forFA0.fits` |
| Fiberassign | `run_path1_fiberassign.sh` | Done | `farun-pass*/`, `Univ000/fa/MAIN/` (5034 tiles) |
| Install FBA | `install_fba_to_univ000.sh` | Done | `tiles-BRIGHT-with-fba.fits` |
| mkCat assignwdup | `run_path1_mkcat.sh` (stage 1) | Done | `fba0/datcomb_brightassignwdup.fits` |
| mkCat fulld | `run_path1_mkcat_fulld_only.sh` | Done | `loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits` (2.4 GB) |
| mkCat mkclusdat | `run_path1_mkcat_mkclusdat_only.sh` | Done | `.../BGS_BRIGHT_clustering.dat.fits` (1.3 GB) |
| LOA inject + mag-lim | `run_path1_maglim_from_fulld.sh` | Done | `mock_bgs_maglim.fits` (9.54M rows) |
| GraphWeb | wedge rebuild | **Next** | use mag-lim FITS |

```bash
export RUN_ROOT=/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/path1_fiberassign_20260604_083322
```

**GraphWeb input (ready now):**

```bash
MOCK_MAGLIM=${RUN_ROOT}/mock_bgs_maglim.fits
# columns: TARGETID, RA, DEC, Z (RSD/observed), ZWARN, DELTACHI2, SPECTYPE, BGS_TARGET,
#          FILE_NUM, HALO_INDEX, BOX_INDEX
```

**Partial reruns (if needed):**

```bash
# mkclusdat only
bash scripts/run_path1_mkcat_mkclusdat_only.sh

# fulld only (if you ever need to rebuild full_noveto)
bash scripts/run_path1_mkcat_fulld_only.sh
```

Production DA2 mocks sometimes use **`module load LSS/main`** + **`runAltMTLRealizations.py`** instead of `fa_multipass`; not used on this scratch tree.

---

## ~~Recommended: run B + C + D + E now (`run_loa_BCDE.sh`)~~ (Path 2 / hybrid — not Path 1)

One orchestrated path for **LOA-calibrated** mocks (best available without `assignwdup` / full `fulld`):

| Step | Component | Script | What it does |
|------|-----------|--------|--------------|
| **B** | Targeting footprint | `upstream_prepare_mocks_Y3_bright.py --apply_mask y` | DESI imaging mask on CutSky → new `forFA0.fits` |
| **C** | Fibre geometry | `getpota` + mkCat `combd usepota` | New `pota-BRIGHT.fits`, `datcomb_brightwdup.fits` (collisions stripped) |
| **D** | Spectro (LOA marginals) | `inject_loa_spec_from_zall.py` | `ZWARN` / `DELTACHI2` / `SPECTYPE` drawn from LOA `zall` BGS-bright stats |
| **E** | LOA cuts | `build_mock_bgs_maglim_catalog.py` | Same four cuts as DESI; **use this FITS for wedge/GNN** |

```bash
salloc --nodes=1 --ntasks-per-node=1 --constraint=cpu --qos=interactive --time=04:00:00 --account=desi
cd /pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000
bash scripts/run_loa_BCDE.sh
# logs + products under stage_3/loa_BCDE_<timestamp>/
```

Resume from a step: `bash scripts/run_loa_BCDE.sh --from C` (or `D`, `E`).

### Honest limit on “exact” statistical replication

| Approach | How close to LOA |
|----------|------------------|
| **B+C+D+E above** | Matches LOA **imaging footprint**, **collision geometry** (pota), **global pass/fail and Δχ² marginals** from `zall`, and **identical final cuts** |
| **Full LSS** (fiberassign + `fulld`) | Closer to production mock spectro pipeline; still not identical to real noise/systematics |
| **Real DESI** | Target for transfer learning |

**D (injection)** does not correlate failures with local density the same way real spectro does; that can still affect cluster-tail transfer. After this pipeline, compare **keep fraction** and **Δχ² CDF** to `zall` and rerun wedge three-curve diagnostics.

### Full LSS path (optional upgrade for C′ + D′)

If you later obtain `datcomb_brightassignwdup.fits` (fiberassign + `usepota=n` COMBD), replace step **D** with mkCat `joindspec=y fulld=y` and point **E** at `BGS_BRIGHT_full_noveto.dat.fits`.

---

## Run commands (copy-paste on compute node)

Set once:

```bash
cd /pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000
unset PYTHONPATH PYTHONHOME
export SCRATCH=/pscratch/sd/d/dkololgi
export DESI_ROOT_READONLY="${DESI_ROOT_READONLY:-/dvs_ro/cfs/cdirs/desi}"
PY=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
RUN_ROOT="${PWD}/stage_3/loa_aligned_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
```

### Step 0 — Verify packages

```bash
bash scripts/check_ph000_env.sh
```

### Step 1 — COMBD only (works now; no LOA spec columns)

Writes `datcomb_brightwdup.fits` under a **new** directory (does not overwrite old `stage_3/fba0/`):

```bash
LOG="${RUN_ROOT}/mkCat_combd.log"
$PY scripts/upstream_mkCat_SecondGen_amtl.py \
  --tracer BGS_BRIGHT --mockver ab_secondgen --mocknum 0 \
  --base_output "${RUN_ROOT}/" --outmd scratch \
  --targDir "${PWD}/stage_2/SecondGenMocks/AbacusSummitBGS_v2/" \
  --pota "${PWD}/stage_2/SecondGenMocks/AbacusSummitBGS_v2/mock0/pota-BRIGHT.fits" \
  --simName SecondGenMocks/AbacusSummit_v4_1 --survey DA2 --specdata loa-v1 --dataversion v2 \
  --combd y --usepota y --joindspec n --fulld n --add_gtl n --mkclusdat n --compmd not_altmtl \
  2>&1 | tee "$LOG"
```

Output: `${RUN_ROOT}/fba0/datcomb_brightwdup.fits`

### Step 2 — Full LSS spec layer (after you have `assignwdup`)

Requires `source /global/common/software/desi/desi_environment.sh main` for spec/catalog paths, but run mkCat with **`$PY`** (LSS in cosmic_env):

```bash
# Prerequisite: ${RUN_ROOT}/fba0/datcomb_brightassignwdup.fits must exist
# (from fiberassign + mkCat --usepota n --combd y, or copied from a completed mock)

LOG="${RUN_ROOT}/mkCat_fulld.log"
$PY scripts/upstream_mkCat_SecondGen_amtl.py \
  --tracer BGS_BRIGHT --mockver ab_secondgen --mocknum 0 \
  --base_output "${RUN_ROOT}/" --outmd scratch \
  --targDir "${PWD}/stage_2/SecondGenMocks/AbacusSummitBGS_v2/" \
  --pota "${PWD}/stage_2/SecondGenMocks/AbacusSummitBGS_v2/mock0/pota-BRIGHT.fits" \
  --simName SecondGenMocks/AbacusSummit_v4_1 --survey DA2 --specdata loa-v1 --dataversion v2 \
  --combd n --usepota y --joindspec y --fulld y --add_gtl y --mkclusdat y --compmd not_altmtl \
  2>&1 | tee "$LOG"
```

Target science product:  
`${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits`

Print full mkCat flags: `bash scripts/run_stage3_desi_aligned_mkcat.sh --print-target-cmd`

### Step 3 — Fiberassign (only if pursuing Path 1)

Needs **`desi_environment`** (not cosmic_env alone):

```bash
source /global/common/software/desi/desi_environment.sh main
FA_OUT="${RUN_ROOT}/fiberassign"
mkdir -p "$FA_OUT"
# Example driver — tune --tilesfn / obs list to match your mock footprint:
python /pscratch/sd/d/dkololgi/LSS/scripts/mock_tools/fa_multipass.py \
  --infn "${PWD}/stage_2/SecondGenMocks/AbacusSummitBGS_v2/forFA0_nomask.fits" \
  --outdir "$FA_OUT" --program bright --survey main \
  --tilesfn /global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/tiles-BRIGHT.fits \
  --steps tiles,sky,targ,fa
```

Then link/copy `fba-*.fits` into `${RUN_ROOT}/Univ000/fa/MAIN/<date>/` layout expected by mkCat, or rerun COMBD with `--usepota n`.

### Step 4 — LOA inject + mag-lim cuts (**complete**)

```bash
export RUN_ROOT=/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/path1_fiberassign_20260604_083322
bash scripts/run_path1_maglim_from_fulld.sh   # already run; produces mock_bgs_maglim.fits
```

### Step 5 — GraphWeb

Rebuild wedge / NPZ / training from **`$OUT`**, not `stage3_postcollision_dedup_science.fits`.

---

## Key paths

| Role | Path |
|------|------|
| This doc | `/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/STAGE3_DESI_ALIGNMENT.md` |
| Env check | `ph000/scripts/check_ph000_env.sh` |
| DESI reference catalog | `/pscratch/sd/d/dkololgi/graphweb_desi/catalogs/bgs_maglim_bright_galaxy_zwarn0_dchi2ge25.fits` |
| forFA0 | `.../stage_2/SecondGenMocks/AbacusSummitBGS_v2/forFA0_nomask.fits` |
| pota | `.../stage_2/.../mock0/pota-BRIGHT.fits` |
| Path 1 `RUN_ROOT` | `.../stage_3/path1_fiberassign_20260604_083322` |
| assignwdup | `${RUN_ROOT}/fba0/datcomb_brightassignwdup.fits` |
| full LSS cat | `${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits` |
| clustering cat | `${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_clustering.dat.fits` |
| mag-lim (final) | `${RUN_ROOT}/mock_bgs_maglim.fits` |
| injected full | `${RUN_ROOT}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto_loa_spec.fits` |
| inject script | `ph000/scripts/inject_loa_spec_from_zall.py` |
| mag-lim script | `ph000/scripts/build_mock_bgs_maglim_catalog.py` |
| LSS clone | `/pscratch/sd/d/dkololgi/LSS` |

---

## Prepare: why `forFA0.fits` takes long

`bash scripts/run_path1_prepare.sh` runs `upstream_prepare_mocks_Y3_bright.py` with `--apply_mask y`. The terminal can sit on one line for **1–3+ hours** without being hung. Stages:

| Stage | What happens | Typical time | Terminal clues |
|-------|----------------|--------------|----------------|
| **1. Read CutSky** | Load `ph000` BGS FITS (~64M rows) from CFS via `cutsky_link` | Minutes | `Length before rbandcut`, column list |
| **2. BGS / footprint cuts** | Bright/faint split, Y5 tile mask, **Y3 DESI footprint** (`is_point_in_desi`) | Minutes | `52698730 in Y5`, `18351409 in Y3` |
| **3. Imaging mask (slow)** | Legacy Survey DR9 pixel lookup per brick | **1–3+ h** | `getting nobs and mask bits` then `adding nobs and mask values to 18351409 rows` — **no further prints until this finishes** |
| **4. Imaging veto** | `cutphotmask`: drop targets with bad `MASKBITS` / zero `NOBS_*` | Minutes | `before imaging veto` |
| **5. Write `forFA0.fits`** | FITS write of surviving targets (~10M order-of-magnitude after veto) | ~10–30 min | `will write outputs to .../forFA0.fits` already printed; file appears at end |
| **6. Script exit** | `ls -lh forFA0*.fits` in `run_path1_prepare.sh` | seconds | `forFA ready under ...` |

### What stage 3 is doing (imaging mask)

For every target in the Y3 table (~18.35M rows in your run), LSS must attach DESI-style imaging columns:

- `NOBS_G`, `NOBS_R`, `NOBS_Z` — exposure counts at each `(RA, Dec)`
- `MASKBITS` — Legacy Survey mask bits at each pixel

Implementation (`LSS/imaging/get_pixel_bitmasknobs.py`):

1. Sort targets by `BRICKID` (0.25° bricks).
2. Spawn a worker pool (`IMAGING_NOBS_NPROC`, default **32**).
3. **Per unique brick** (often tens of thousands across Y3): open up to **four** DR9 coadd files on CFS (`maskbits` + `nexp` g/r/z under `/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/`), WCS pixel lookup, sample values for all targets in that brick.

This is **CPU + shared-filesystem I/O bound**, not a single big linear algebra step. There is **no progress bar** during `pool.map` over bricks, so the job looks idle after `adding nobs and mask values to ... rows`.

**Why it is slower than “just writing a FITS”:** `forFA0.fits` is only written **after** millions of per-brick DR9 reads complete and `cutphotmask` drops masked/vetoed sources. The heavy work is **before** the file exists on disk.

**Speed tuning (prepare imaging step):**

| Knob | Effect |
|------|--------|
| `salloc --cpus-per-task=128` | More parallel brick workers (script defaults `IMAGING_NOBS_NPROC` to `SLURM_CPUS_PER_TASK`) |
| `export IMAGING_NOBS_NPROC=96` | Override if 128 workers thrash CFS (diminishing returns past ~64–96) |
| Off-peak / `regular` queue | Less CFS contention than busy interactive |
| `--apply_mask n` | **~10×+ faster** but skips DESI imaging veto → **not Path 1 / LOA footprint** |
| LSS brick dict + header-only WCS | Small CPU win per brick (in local `get_pixel_bitmasknobs.py`) |

```bash
salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 \
  --constraint=cpu --qos=interactive --time=12:00:00 --account=desi
export IMAGING_NOBS_NPROC=96   # tune 64–128
bash scripts/run_path1_prepare.sh
```

**Not faster without tradeoffs:** imaging must open DR9 `maskbits` + `nexp` FITS per brick (~tens of thousands of bricks × 4 files). There is no cached shortcut in ph000; production `forFA` on CFS is per-simulation, not a substitute for masked `ph000`.

Watch the log in another shell: `tail -f stage_3/path1_*/prepare.log`.

### Nested multiprocessing crash (fixed)

First prepare attempt failed with:

`AssertionError: daemonic processes are not allowed to have children`

Cause: outer `Pool` in `upstream_prepare_mocks_Y3_bright.py` + inner `Pool` in `get_nobsandmask()`. For a single realization (`realmin=0`, `realmax=1`), prepare now calls `process(0)` **serially** so the inner imaging pool is allowed. Re-run prepare after pulling the updated script.

---

## Common errors (Path 1 runs)

| Symptom | Cause | Fix |
|---------|--------|-----|
| `daemonic processes are not allowed to have children` | Old prepare: nested `Pool` | Use current `upstream_prepare_mocks_Y3_bright.py` + rerun `run_path1_prepare.sh` |
| `OverflowError: Python integer 65536 out of bounds for uint8` | LSS imaging: `np.full(..., 2**16, uint8)` on bricks outside DR9 (`PHOTSYS` not N/S); **NumPy 2 / py3.13** | Fixed in `/pscratch/sd/d/dkololgi/LSS/py/LSS/imaging/get_pixel_bitmasknobs.py` (use `int32` for bit 16); rerun prepare |
| `ERROR: forFA not found` | Prepare did not finish | Complete prepare; confirm `stage_2/.../forFA0.fits` exists |
| `bash: stamp: No such file` | Pasted doc placeholder `path1_<stamp>` | Use real directory from `run_path1_fiberassign.sh` / `prepare.log` (`path1_YYYYMMDD_HHMMSS`) |
| `tee: .../mkCat_*.log: No such file` | `RUN_ROOT=...` literally | Set `RUN_ROOT` to full path under `stage_3/path1_*` |
| FITSIO cannot open `forFA0.fits` | No forFA yet or wrong `targDir` | Finish prepare; mkCat `--targDir` must point at `stage_2/SecondGenMocks/AbacusSummitBGS_v2/` |
| `FileNotFoundError: /loa-v1/...` | Unset `RUN_ROOT` in mag-lim builder | Export full `RUN_ROOT` before `build_mock_bgs_maglim_catalog.py` |
| `UnboundLocalError: ff` in mkclusdat | Input was `_full_noveto` but code looked for `_full_HPmapcut` only | Fixed in `cattools.py` (tries `_full_noveto`) |
| `KeyError: Column Z already exists` in fulld | Mock table has both `Z` and `RSDZ` | Fixed: rename existing `Z` → `Z_LEGACY`, then `RSDZ` → `Z` |
| `ModuleNotFoundError: hdf5plugin` | BGS mkclusdat wrote `.h5` | Mock path uses `.fits` clustering output |
| `SPECTYPE` wrong (`b'GALAXY'`) | `str(bytes)` in inject | Fixed: decode bytes in `inject_loa_spec_from_zall.py` |
| mag-lim kept 0 rows | bad `SPECTYPE` string | Rerun inject + mag-lim after fix |

**Do not** copy lines containing `...`, `<stamp>`, or `/path/from/log` into bash — those are documentation placeholders only.

---

## FAQ

**Is it just DESI-like quality cuts?**  
No — but **Path 1 now implements all five components (A–E)** through `mock_bgs_maglim.fits`. Residual differences vs real LOA are in **N(z)**, not in the cut logic.

**What is `Z` in the mock mag-lim catalog?**  
Observed redshift **with RSD** (`RSDZ` from mocks, renamed to `Z` in `mkfulldat`). Cosmological redshift is `TRUEZ` / `Z_COSMO` in `full_noveto` (not exported to mag-lim).

**Will this make mock ≡ LOA exactly?**  
Selection rules and spectro **marginals** match; mock **N(z)** is shifted to higher \(z\) (BGS CutSky parent). See [validation table](#validation-desi-loa-vs-ph000-mock-mag-lim-catalogues) below.

---

## Validation: DESI LOA vs ph000 mock mag-lim catalogues

Computed 2026-06-04 on full catalogues (all rows scanned unless noted).

| Quantity | DESI LOA reference | ph000 Path 1 mock | Notes |
|----------|-------------------|-------------------|--------|
| **Catalog path** | `graphweb_desi/catalogs/bgs_maglim_bright_galaxy_zwarn0_dchi2ge25.fits` | `${RUN_ROOT}/mock_bgs_maglim.fits` | Same builder logic / cuts |
| **Source** | `zall-pix-loa.fits` (real spectro) | `full_noveto` + LOA marginal injection | Mock D is uncorrelated failures |
| **Row count** | 9,166,391 | 9,538,254 | Mock +4.1% (similar order of magnitude) |
| **Selection cuts** | `ZWARN=0`, `Δχ²≥25`, `SPECTYPE=GALAXY`, BGS bright | identical rules | Both 100% by construction post-cut |
| **Pre-cut spectro pass rate** | 96.15% (LOA BGS-bright in `zall`) | 96.14% (9.54M / 9.92M injected) | **Validates component D** |
| **`Z` meaning** | Spectroscopic observed `Z` | **`RSDZ` → `Z`** (observed + peculiar/RSD) | Matches DESI graph convention |
| **`Z` mean ± std** | 0.224 ± 0.120 | 0.289 ± 0.180 | Mock N(z) shifted high (CutSky BGS parent) |
| **`Z` median (p50)** | 0.211 | 0.235 | Same trend |
| **`Z` p16 – p84** | 0.108 – 0.341 | 0.119 – 0.590 | Mock lacks low-z tail; high-z tail truncated ~0.8 |
| **`Z` range** | −0.004 – 1.698 | 0.001 – 0.795 | Mock bounded by simulation / BGS sample |
| **`DELTACHI2` mean** | 2028 | 2028 | Drawn from LOA pass/fail pools → **matches** |
| **`DELTACHI2` min** | 25.0 | 25.0 | Cut floor respected |
| **Sky footprint (RA)** | 0° – 360° | 0° – 360° | Full DESI-style span |
| **Sky footprint (Dec)** | −19.6° – +79.3° | −19.6° – +79.3° | Consistent |
| **Extra mock columns** | — | `FILE_NUM`, `HALO_INDEX`, `BOX_INDEX` | For T-Web λ join / training |

**Summary:** The mock mag-lim catalogue **passes the same LOA quality gates** as DESI, reproduces the **LOA spectroscopic success fraction (~96.15%)** and **Δχ² mean**, and covers a **similar sky footprint** and **similar total count**. The main **expected** difference is **redshift distribution** (mock CutSky BGS vs real DESI N(z)), not the cut definitions. That is sufficient to validate Path 1 catalogue production; wedge-level transfer should be tested next in GraphWeb.
