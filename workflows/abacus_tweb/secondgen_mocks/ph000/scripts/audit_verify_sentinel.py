#!/usr/bin/env python3
"""Verify the z~0.59 spike == fibre-unobserved (ZWARN=999999) targets
resurrected by the spec injection, and check wedge contamination."""
import numpy as np
import fitsio

RUN = "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/path1_fiberassign_20260604_083322"
FULL = f"{RUN}/loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits"
MAGLIM = f"{RUN}/mock_bgs_maglim.fits"

print("== full_noveto: split by observation status ==")
d = fitsio.read(FULL, columns=["ZWARN", "Z", "Z_LEGACY", "Z_COSMO", "TRUEZ",
                               "LOCATION_ASSIGNED", "GOODHARDLOC", "TARGETID"])
zw = d["ZWARN"]
obs = zw == 0
unobs = zw == 999999
print(f"observed (ZWARN=0):    {obs.sum():,}")
print(f"unobserved (999999):   {unobs.sum():,}")
for lab, m in [("OBSERVED", obs), ("UNOBSERVED", unobs)]:
    z = d["Z"][m]; zc = d["Z_COSMO"][m]; tz = d["TRUEZ"][m]
    print(f"\n[{lab}] n={m.sum():,}")
    print(f"  Z       : mean={z.mean():.4f} std={z.std():.4f} "
          f"min={z.min():.4f} med={np.median(z):.4f} max={z.max():.4f} "
          f"nuniq={np.unique(np.round(z,4)).size}")
    print(f"  Z_COSMO : mean={zc.mean():.4f} std={zc.std():.4f} "
          f"min={zc.min():.4f} med={np.median(zc):.4f} max={zc.max():.4f}")
    print(f"  TRUEZ   : mean={tz.mean():.4f} std={tz.std():.4f} "
          f"min={tz.min():.4f} med={np.median(tz):.4f} max={tz.max():.4f}")
    # most common Z value
    vals, cts = np.unique(np.round(z, 4), return_counts=True)
    top = np.argsort(cts)[::-1][:3]
    print(f"  top Z values: " + ", ".join(f"{vals[i]}:{cts[i]:,}" for i in top))
    # how many of these unobserved have a TRUTH redshift in the wedge?
    inw = (tz >= 0.2) & (tz <= 0.3)
    print(f"  TRUEZ in [0.2,0.3]: {inw.sum():,} ({100*inw.mean():.1f}%)")

print("\n== where does the observed-target N(z) sit (this is the REAL mock N(z)) ==")
zobs = d["Z"][obs]
edges = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.7])
h,_ = np.histogram(zobs, bins=edges)
for i in range(len(h)):
    print(f"  z[{edges[i]:.2f},{edges[i+1]:.2f}): {h[i]:>9,} ({100*h[i]/zobs.size:5.2f}%)")
print(f"  observed Z mean={zobs.mean():.4f} median={np.median(zobs):.4f}")

print("\n== maglim: how many rows have Z within wedge bounds, and are they real? ==")
m = fitsio.read(MAGLIM, columns=["Z", "RA", "DEC"])
inwed = (m["Z"]>=0.2)&(m["Z"]<=0.3)&(m["RA"]>=120)&(m["RA"]<=160)&(m["DEC"]>=14.5)&(m["DEC"]<=30.6)
print(f"  maglim rows in wedge box (RA120-160,Dec14.5-30.6,z0.2-0.3): {inwed.sum():,}")
spike = (m["Z"]>0.55)&(m["Z"]<0.62)
print(f"  maglim rows in z(0.55,0.62) sentinel band: {spike.sum():,} ({100*spike.mean():.2f}%)")
print("DONE")
