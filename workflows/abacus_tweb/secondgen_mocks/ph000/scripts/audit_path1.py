#!/usr/bin/env python3
"""Standalone audit of the Path-1 mock generation data products.

Reads only the columns it needs, in a memory-bounded way, and prints
distribution diagnostics for each stage so we can find flaws.
"""
import os
import numpy as np
import fitsio

RUN = "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/path1_fiberassign_20260604_083322"
FULL = f"{RUN}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits"
INJ = f"{RUN}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto_loa_spec.fits"
MAGLIM = f"{RUN}/mock_bgs_maglim.fits"
DESI = "/pscratch/sd/d/dkololgi/graphweb_desi/catalogs/bgs_maglim_bright_galaxy_zwarn0_dchi2ge25.fits"
CUTSKY = "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits"


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def colnames(path):
    with fitsio.FITS(path) as f:
        return f[1].get_colnames(), int(f[1].get_nrows())


def zstats(z, label):
    z = np.asarray(z, dtype=np.float64)
    fin = np.isfinite(z)
    z = z[fin]
    pct = np.percentile(z, [0, 1, 16, 50, 84, 99, 100])
    print(f"{label}: n={z.size:,}  mean={z.mean():.4f}  std={z.std():.4f}")
    print(f"   min={pct[0]:.4f} p1={pct[1]:.4f} p16={pct[2]:.4f} "
          f"p50={pct[3]:.4f} p84={pct[4]:.4f} p99={pct[5]:.4f} max={pct[6]:.4f}")
    # coarse histogram
    edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 2.0])
    h, _ = np.histogram(z, bins=edges)
    for i in range(len(h)):
        print(f"     z[{edges[i]:.2f},{edges[i+1]:.2f}): {h[i]:>10,} ({100*h[i]/z.size:5.2f}%)")


# ---- 1. full_noveto (pre-injection) ----
hdr("1. full_noveto (pre-injection) schema + ZWARN/Z")
cn, nr = colnames(FULL)
print(f"rows={nr:,}  ncols={len(cn)}")
print("cols:", cn)
with fitsio.FITS(FULL) as f:
    h = f[1]
    # ZWARN present pre-injection?
    rd = h.read(columns=[c for c in ["ZWARN", "Z", "ZWARN_MTL", "LOCATION", "TILELOCID",
                                     "FRACZ_TILELOCID", "COMP_TILE", "BOX_INDEX",
                                     "TARGETID", "BGS_TARGET"] if c in cn])
for c in rd.dtype.names:
    a = rd[c]
    if c in ("Z",):
        zstats(a, "  full_noveto Z")
    elif a.dtype.kind in "iuf":
        u, ct = np.unique(a, return_counts=True)
        if u.size <= 12:
            print(f"  {c}: " + ", ".join(f"{v}:{n:,}" for v, n in zip(u, ct)))
        else:
            print(f"  {c}: min={a.min()} max={a.max()} mean={float(a.mean()):.4g} nuniq={u.size:,}")

# TARGETID uniqueness in full_noveto
if "TARGETID" in rd.dtype.names:
    tid = rd["TARGETID"]
    print(f"  TARGETID: n={tid.size:,} nuniq={np.unique(tid).size:,} "
          f"(dup rows={tid.size-np.unique(tid).size:,})")

# ---- 2. injected file ----
hdr("2. injected full_noveto_loa_spec: ZWARN/DELTACHI2/SPECTYPE")
cn2, nr2 = colnames(INJ)
with fitsio.FITS(INJ) as f:
    d = f[1].read(columns=["ZWARN", "DELTACHI2", "SPECTYPE"])
zw = d["ZWARN"]; dc = d["DELTACHI2"].astype(np.float64); sp = d["SPECTYPE"]
print(f"rows={nr2:,}")
u, ct = np.unique(zw, return_counts=True)
print("  ZWARN:", ", ".join(f"{v}:{n:,} ({100*n/zw.size:.2f}%)" for v, n in zip(u, ct)))
print(f"  DELTACHI2: mean={dc.mean():.2f} min={dc.min():.3f} max={dc.max():.1f} "
      f"median={np.median(dc):.2f}")
pas = zw == 0
print(f"  DELTACHI2|pass(ZWARN=0): mean={dc[pas].mean():.2f} min={dc[pas].min():.3f} "
      f"frac<25={100*np.mean(dc[pas]<25):.3f}%")
print(f"  DELTACHI2|fail: max={dc[~pas].max():.3f} frac>=25={100*np.mean(dc[~pas]>=25):.3f}%")
spu, spct = np.unique(sp, return_counts=True)
print("  SPECTYPE:", ", ".join(f"{repr(v)}:{n:,}" for v, n in zip(spu, spct)))
pass_frac = float(np.mean(pas))
print(f"  overall pass frac (ZWARN==0) = {pass_frac:.4f}")

# ---- 3. final mag-lim ----
hdr("3. mock_bgs_maglim.fits final product")
cn3, nr3 = colnames(MAGLIM)
print(f"rows={nr3:,}  cols={cn3}")
with fitsio.FITS(MAGLIM) as f:
    m = f[1].read(columns=["RA", "DEC", "Z", "TARGETID", "BOX_INDEX", "HALO_INDEX",
                           "FILE_NUM", "ZWARN", "DELTACHI2", "SPECTYPE"])
zstats(m["Z"], "  maglim Z")
print(f"  RA span: {m['RA'].min():.3f} - {m['RA'].max():.3f}")
print(f"  DEC span: {m['DEC'].min():.3f} - {m['DEC'].max():.3f}")
tid = m["TARGETID"]
print(f"  TARGETID nuniq={np.unique(tid).size:,} of {tid.size:,} "
      f"(dups={tid.size-np.unique(tid).size:,})")
bx = m["BOX_INDEX"]
print(f"  BOX_INDEX==-1: {int(np.sum(bx==-1)):,} ({100*np.mean(bx==-1):.3f}%)  "
      f"min={bx.min()} max={bx.max()}")
print(f"  Z<=0: {int(np.sum(m['Z']<=0)):,}   Z<0.001: {int(np.sum(m['Z']<0.001)):,}")
print(f"  ZWARN!=0 leaked: {int(np.sum(m['ZWARN']!=0)):,}")
sp3u, sp3ct = np.unique(m["SPECTYPE"], return_counts=True)
print("  SPECTYPE:", ", ".join(f"{repr(v)}:{n:,}" for v, n in zip(sp3u, sp3ct)))

# ---- 4. DESI reference ----
hdr("4. DESI LOA reference catalog Z")
try:
    cnD, nrD = colnames(DESI)
    with fitsio.FITS(DESI) as f:
        zcol = "Z" if "Z" in cnD else [c for c in cnD if c.upper().startswith("Z")][0]
        zD = f[1].read(columns=[zcol])[zcol]
    print(f"rows={nrD:,}  zcol={zcol}")
    zstats(zD, "  DESI Z")
except Exception as e:
    print("DESI ref read failed:", e)

# ---- 5. parent CutSky N(z) to explain shift ----
hdr("5. Parent CutSky N(z) (full file + r<19.5 + within wedge)")
try:
    cnC, nrC = colnames(CUTSKY)
    print(f"CutSky rows={nrC:,} cols={cnC}")
    with fitsio.FITS(CUTSKY) as f:
        c = f[1].read(columns=["Z", "Z_COSMO", "R_MAG_APP", "RA", "DEC"])
    zstats(c["Z"], "  CutSky Z (all)")
    mr = c["R_MAG_APP"] < 19.5
    zstats(c["Z"][mr], "  CutSky Z (r<19.5)")
    # within production wedge
    w = mr & (c["RA"] >= 120) & (c["RA"] <= 160) & (c["DEC"] >= 14.5) & (c["DEC"] <= 30.6)
    zstats(c["Z"][w], "  CutSky Z (r<19.5, RA120-160 Dec14.5-30.6)")
except Exception as e:
    print("CutSky read failed:", e)

print("\nDONE")
