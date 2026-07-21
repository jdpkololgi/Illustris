# P0S — Preservation & Migration Manifest

Generated 2026-07-21. Covers `/pscratch/sd/d/dkololgi` ahead of the NERSC downtime /
scratch-purge freeze. **No pscratch files were modified**; this step only *reads* scratch
and *adds* copies to Git/home and HPSS.

## Storage contract (from plan P0S)

| Tier | Holds | Durability |
|---|---|---|
| Git/home | source code, scripts, schemas, configs, manifests, hashes, env specs, decisions, compact evidence | backed up |
| CFS | irreplaceable reusable catalogues, scalers, selected checkpoints/predictions, canaries, release bundles | durable GPFS |
| HPSS | large expensive-to-rebuild density, T-Web, graph, staged-mock, archival bundles | tape, survives purge |
| pscratch | active or reproducible intermediates, temporary caches, current outputs | **purge-eligible** |

## 1. Environment reproducibility  — DONE (`env/`)

Both production environments captured (from-history YAML, full YAML, explicit spec,
`conda list`, `pip freeze`, import smoke test); CUDA/module metadata in
`env/system_cuda_modules.txt`.

| Env | scratch path | Python | Key versions | Smoke |
|---|---|---|---|---|
| `cosmic_env` | `/pscratch/sd/d/dkololgi/conda/envs/cosmic_env` | 3.11.15 | jax/jaxlib 0.7.2, flax 0.10.4, optax 0.2.6, distrax 0.1.7, jraph 0.0.6.dev0, torch 2.9.1, numpy 2.3.5, gudhi 3.11.0, astropy 7.2.0 | PASS |
| `rapids-gnn` | `/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn` | 3.11.14 | cudf 26.02.01, cugraph 26.02.00, cupy 14.0.1, rmm 26.02.00, numpy 2.2.6 (cuGraph ops need a GPU node) | versions PASS (login) |

> Rebuild guard (NERSC gotcha observed during capture): `unset PYTHONPATH PYTHONHOME;
> export PYTHONNOUSERSITE=1` before using either env, or DESI `desiconda` site-packages
> shadow numpy.

## 2. Scratch-only source under `SecondGen_Mocks/ph000/` — COPIED to Git

42 source/metadata files (0.21 MB), inventory + sha256 in `ph000_source_inventory.raw.tsv`.
**Overlap check:** none of these existed in `TNG/Illustris` or `GraphWeb_DESI` before today —
they were genuinely scratch-only. All copied with **0 checksum mismatches**.

| Class | Count | Copied to | Priority | Notes |
|---|---|---|---|---|
| Local pipeline source (`.py`/`.sh`/`.slurm`) — stage1 subset, maglim catalog, loa spec inject, audits, path1 runners, stage3 runners, wedge annotate/cache | 21 | `workflows/abacus_tweb/secondgen_mocks/ph000/{,scripts/,wedge/}` | **P1** irreplaceable | maintained here |
| Upstream DESI LSS copies (`upstream_prepare_mocks_Y3_bright.py`, `upstream_getpotaDA2_mock.py`, `upstream_mkCat_SecondGen_amtl.py`) | 3 | same tree, `scripts/` | P3 | recoverable from `desihub/LSS` main; pinned copy kept for reproducibility |
| Documentation (`README.md`, `STAGE3_DESI_ALIGNMENT.md`, `CACHE_TRAINING_NEW_WEDGES.md`, `STAGED_MOCK_WEDGE_SBI_README.md`) | 4 | same tree | P1 | pipeline spec + column-propagation contract |
| Generated manifests/summaries (`.json`) | 14 (15 files, some dup content) | `docs/evidence/p0s/ph000_manifests/` | P2 | small provenance evidence |

**Producing chain (from README):** Stage 1 `stage1_cutsky_subset.py` (reads CFS cosmosim
CutSky `cutsky_BGS_z0.200_..._ph000.fits`) → Stage 2 `upstream_prepare_mocks_Y3_bright.py`
+ `upstream_getpotaDA2_mock.py` (`forFA*`, `pota-*`) → Stage 3 `upstream_mkCat_SecondGen_amtl.py`
(COMBD/clustering) → wedge annotate + SBI cache. Requires `desi_environment.sh main`, `LSS`,
`desitarget`. Downstream consumers: `workflows/abacus_tweb/` graph/T-Web + SBI cache builders.

## 3. HPSS bulk archive — DONE + VERIFIED

99 `htar` archives at `/home/d/dkololgi/pscratch_backup_20260721/`, each CRC-verified
(`htar -Kv`), **0 failures** after the verify sweep. Listing: `~/hpss_backup/HPSS_CONTENTS.txt`;
per-archive status: `~/hpss_backup/logs_20260721/manifest.tsv`.

## 4. Dry-run move table (byte totals by tier)

| Destination | What | Approx size | Status |
|---|---|---|---|
| Git/home | ph000 source + env specs + this manifest | ~0.5 MB | **copied** |
| HPSS | all derived density/T-Web/graph/staged-mock/mock/SBI-cache/output dirs | **~2.7 TB** | **archived + verified** |
| (skipped) public | `AbacusSummit_densities/AbacusSummit_base_c000_ph000` | ~385 GB | re-downloadable from CFS `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit` |
| (skipped) regenerable | `conda`, `.cache`, `.local`, `.cursor-server`, `jax_cache` | ~50 GB | rebuildable from env specs (§1) |
| **pscratch total** | | **3.2 TB** | |

Nothing in the ~2.7 TB payload is required to remain on scratch: it is either archived to
HPSS (recoverable via `htar -xvf`) or regenerable from tracked code + retained CFS inputs.

## 5. CFS-tier candidates — PENDING USER REVIEW (not yet copied)

CFS holds the *convenient-restore* subset (vs HPSS deep archive). Candidates: canonical
catalogues/index (`abacus/p1b_full_footprint/`), fitted scalers, selected best checkpoints,
golden canaries, release bundles. **Blocker:** the `desi` CFS project is at **93 % (5.39/5.73 PB)**,
so a target dir + space check under `/global/cfs/cdirs/desi/users/dkololgi/` is needed before
copying. Deferred pending your go-ahead (HPSS already provides durable coverage).

## Plan P0S checklist status

- [x] Inventory scratch-only scripts/config under `SecondGen_Mocks/ph000/`
- [x] Seed with known source candidates (all present & captured)
- [x] Classify each item (source / upstream-copy / docs / evidence)
- [x] Record size, checksum, producing command, dependencies, consumers, destination, priority
- [x] Identify overlap with `workflows/abacus_tweb/` (none pre-existing)
- [x] Define versioned repo destination + copy scratch-only source (verified checksums)
- [x] Export `cosmic_env` + `rapids-gnn` reproducibility records (YAML/spec/lists/pip/smoke/CUDA)
- [x] RAPIDS/cuGraph versions + graph-metric env note
- [x] Dry-run move table with byte totals (Git/home, CFS, HPSS)
- [~] Copies performed + destination checksums verified — **Git + HPSS done; CFS pending review (§5)**
- [ ] CFS copies of irreplaceable reusable catalogues/checkpoints (awaiting target + space decision)
