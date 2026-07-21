#!/usr/bin/env bash
# Path 1 — Copy fa_multipass fba-*.fits into mkCat layout: Univ000/fa/MAIN/<YYYYMMDD>/fba-*.fits
# fadate per tile matches DESI real tile fiberassign headers (mkCat usepota=n).
set -euo pipefail

PH000="/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000"
FA_OUT="${FA_OUT:?Set FA_OUT to fa_multipass --outdir}"
RUN_ROOT="${RUN_ROOT:-${PH000}/stage_3/path1_$(date +%Y%m%d_%H%M%S)}"
FBADIR="${RUN_ROOT}/Univ000/fa/MAIN"
mkdir -p "$FBADIR"

source /global/common/software/desi/desi_environment.sh main
export FA_OUT RUN_ROOT FBADIR
export PYTHONPATH="/pscratch/sd/d/dkololgi/LSS/py:${PYTHONPATH:-}"

python <<PY
import os, glob, re, shutil
import LSS.common_tools as common

fa_out = os.environ["FA_OUT"]
fbadir = os.environ["FBADIR"]
# Multipass assigns different tile subsets per pass — merge all passes; keep highest pass per tile.
by_tile = {}
for passid in range(4):
    pat = os.path.join(fa_out, "faruns", f"farun-pass{passid}", "fba-*.fits")
    for src in glob.glob(pat):
        m = re.search(r"fba-(\d+)\.fits$", os.path.basename(src))
        if not m:
            continue
        tile = int(m.group(1))
        prev = by_tile.get(tile)
        if prev is None or passid > prev[0]:
            by_tile[tile] = (passid, src)
if not by_tile:
    raise SystemExit("No fba-*.fits under " + fa_out)

# Index tiles already installed (resume); fadate needs a CFS header read per *new* tile.
existing = {}
for path in glob.glob(os.path.join(fbadir, "*", "fba-*.fits")):
    m = re.search(r"fba-(\d+)\.fits$", os.path.basename(path))
    if m:
        existing[int(m.group(1))] = path
print("Already in Univ000:", len(existing), "of", len(by_tile), "tiles")

fadate_cache = {}
n = 0
skipped = 0
ntiles = len(by_tile)
for i, (tile, (passid, src)) in enumerate(sorted(by_tile.items()), 1):
    basename = os.path.basename(src)
    if tile in existing and os.path.isfile(existing[tile]):
        skipped += 1
        if i % 500 == 0 or i == ntiles:
            print(f"progress {i}/{ntiles} skipped={skipped} copied={n}", flush=True)
        continue
    if tile not in fadate_cache:
        fadate_cache[tile] = common.return_altmtl_fba_fadate(tile)
    fadate = fadate_cache[tile]
    destdir = os.path.join(fbadir, fadate)
    os.makedirs(destdir, exist_ok=True)
    dest = os.path.join(destdir, basename)
    if not os.path.isfile(dest):
        shutil.copy2(src, dest)
        existing[tile] = dest
        n += 1
    if i % 100 == 0 or i == ntiles:
        print(f"progress {i}/{ntiles} skipped={skipped} copied={n}", flush=True)
print("Merged passes: unique tiles =", len(by_tile), "skipped =", skipped, "new copies =", n, "into", fbadir)

# mkCat loops all tiles-BRIGHT; fa_multipass may omit ~100+ edge tiles — use subset with fba on disk.
import fitsio
import numpy as np
tiles_fn = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/tiles-BRIGHT.fits"
tiles_all = fitsio.read(tiles_fn)
have = np.array(sorted(by_tile.keys()))
sel = np.isin(tiles_all["TILEID"], have)
tiles_out = os.path.join(os.environ["RUN_ROOT"], "tiles-BRIGHT-with-fba.fits")
fitsio.write(tiles_out, tiles_all[sel], clobber=True)
missing = np.setdiff1d(tiles_all["TILEID"], have)
print("Wrote", tiles_out, "n=", sel.sum(), "of", len(tiles_all), "catalog tiles")
if len(missing):
    print("WARNING:", len(missing), "tiles in tiles-BRIGHT have no fba (excluded from mkCat)")
PY

echo "mkCat fbadir: $FBADIR"
echo "export LSS_TILES_FITS=${RUN_ROOT}/tiles-BRIGHT-with-fba.fits"
echo "Set RUN_ROOT=$RUN_ROOT for mkCat --base_output ${RUN_ROOT}/"
