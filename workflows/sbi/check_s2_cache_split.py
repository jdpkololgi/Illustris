"""Is the s2 shell cache's own train/test split SPATIAL or RANDOM?

Decisive for reading S1(b): gate_t2_cnn_counts.py trains on cache["masks"]. If those masks are a
random node split, S1(b)'s CNN numbers (0.902/0.847/...) are inflated by spatial leakage -- a test
galaxy's neighbours sit in train, and a field model can simply interpolate. The spatial holdout
(train RA<145 / val 145-150 / test RA>=150, halo-disjoint, gutter) exists precisely to stop that.
"""
import numpy as np, pickle, fitsio
from pathlib import Path

R = Path("/pscratch/sd/d/dkololgi/abacus")
CACHE = "sbi_caches/s2_shell_{tag}_si_union/processed_jraph_data_mc1e+09_v2_scaled_3_linear_eig.pkl"
TARGETS = "s2_shells/shell_{tag}_final_wedge_targets.fits"

for tag in ["0p15_0p25", "0p45_0p55"]:
    c = pickle.load(open(R / CACHE.format(tag=tag), "rb"))
    tr, va, te = (np.asarray(m).astype(bool) for m in c["masks"])
    t = fitsio.read(R / TARGETS.format(tag=tag), columns=["RA", "DEC", "HALO_INDEX", "FILE_NUM", "BOX_INDEX"])
    ra = t["RA"]
    print(f"\n=== {tag} ===")
    print(f"  n={len(ra):,}  train/val/test = {tr.sum():,}/{va.sum():,}/{te.sum():,} "
          f"({100*tr.mean():.0f}/{100*va.mean():.0f}/{100*te.mean():.0f}%)")
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        if m.sum():
            print(f"  {nm:5s} RA range [{ra[m].min():7.2f},{ra[m].max():7.2f}]  "
                  f"mean {ra[m].mean():7.2f}  median {np.median(ra[m]):7.2f}")
    # A spatial split => train/test RA ranges barely overlap. A random split => identical ranges.
    if tr.sum() and te.sum():
        overlap = (min(ra[tr].max(), ra[te].max()) - max(ra[tr].min(), ra[te].min()))
        span = ra.max() - ra.min()
        print(f"  RA overlap of train & test = {overlap:.2f} deg of {span:.2f} deg total "
              f"({100*overlap/span:.0f}%)  -> ~100% means RANDOM split (leaky), ~0% means SPATIAL")
        # halo sharing is the other tell
        H = np.stack([t["FILE_NUM"], t["BOX_INDEX"], t["HALO_INDEX"]], 1).astype(np.int64)
        hv = np.ascontiguousarray(H).view([('', H.dtype)] * 3).ravel()
        shared = np.intersect1d(np.unique(hv[tr]), np.unique(hv[te]))
        print(f"  halos shared between train & test: {len(shared):,}  -> >0 means NOT halo-disjoint")
