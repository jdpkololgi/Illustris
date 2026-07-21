"""P9-style residual complementarity audit: is a CIC+learned hybrid worth building?

Honest protocol: blend weights are FIT on half the validation super-blocks and EVALUATED on the
other half (spatially disjoint), both directions, averaged. Fitting and scoring on the same rows
would manufacture a gain.
"""
import importlib.util, json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score

REPO = Path("/global/u2/d/dkololgi/TNG/Illustris")
_s = importlib.util.spec_from_file_location("p8ev", REPO / "workflows/abacus_tweb/plot_p8_smoke_eval.py")
p8ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(p8ev)
REC = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/recovery_v1")

ids, truth, short, meta = p8ev.load_rotation(0)
preds = {"CIC": short["CIC (train-affine)"]}
for m, nm in (("graph", "G-PATCH"), ("unet", "U-PATCH")):
    d = REC / m / "rotation_0/seed_42"
    rid = np.load(d / "best_validation_parent_node_id.npy")
    rp = np.load(d / "best_validation_eigenvalues.npy").astype(np.float64)
    o = np.argsort(rid); pos = np.searchsorted(rid[o], ids)
    preds[nm] = rp[o[pos]]

y = truth[:, 0]
shell = meta["shell"]; sb = meta["superblock_id"]
SH = ["0.15-0.25", "0.25-0.35", "0.35-0.45", "0.45-0.55"]

print("=== 1. residual correlation (lambda1), per shell ===")
print(f"{'shell':12s} {'U vs CIC':>9s} {'G vs CIC':>9s} {'U vs G':>9s}   {'n':>8s}")
for s in range(4):
    m = shell == s
    ru, rg, rc = preds['U-PATCH'][m,0]-y[m], preds['G-PATCH'][m,0]-y[m], preds['CIC'][m,0]-y[m]
    print(f"{SH[s]:12s} {np.corrcoef(ru,rc)[0,1]:9.3f} {np.corrcoef(rg,rc)[0,1]:9.3f} "
          f"{np.corrcoef(ru,rg)[0,1]:9.3f} {m.sum():9,}")
ru, rg, rc = preds['U-PATCH'][:,0]-y, preds['G-PATCH'][:,0]-y, preds['CIC'][:,0]-y
print(f"{'ALL':12s} {np.corrcoef(ru,rc)[0,1]:9.3f} {np.corrcoef(rg,rc)[0,1]:9.3f} {np.corrcoef(ru,rg)[0,1]:9.3f}")

# spatially disjoint blend: split validation super-blocks in two
usb = np.unique(sb); rng = np.random.default_rng(0); rng.shuffle(usb)
half = set(usb[:len(usb)//2].tolist())
A = np.array([v in half for v in sb]); B = ~A

def blend_eval(cols):
    """Fit least-squares blend on A score on B and vice versa; return macro + per-shell R2."""
    X = np.column_stack([preds[c][:,0] for c in cols] + [np.ones(len(y))])
    out = np.zeros(len(y))
    for fit, ev in ((A,B),(B,A)):
        w, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
        out[ev] = X[ev] @ w
    per = [r2_score(y[shell==s], out[shell==s]) for s in range(4)]
    return float(np.mean(per)), per

print("\n=== 2. out-of-sample blends (weights fit on disjoint half of validation super-blocks) ===")
print(f"{'model / blend':28s} {'macro':>7s}  per-shell")
singles = {}
for c in ("CIC", "G-PATCH", "U-PATCH"):
    per = [r2_score(y[shell==s], preds[c][shell==s,0]) for s in range(4)]
    singles[c] = np.mean(per)
    print(f"{c:28s} {np.mean(per):7.4f}  " + " ".join(f"{v:6.3f}" for v in per))
for cols in (["U-PATCH","CIC"], ["G-PATCH","CIC"], ["U-PATCH","G-PATCH"], ["U-PATCH","G-PATCH","CIC"]):
    mac, per = blend_eval(cols)
    gain = mac - max(singles[c] for c in cols)
    print(f"{'+'.join(cols):28s} {mac:7.4f}  " + " ".join(f"{v:6.3f}" for v in per) +
          f"   gain vs best member {gain:+.4f}")

print("\n=== 3. where would a CIC-residual actually help? per-shell CIC vs U ===")
for s in range(4):
    m = shell == s
    print(f"  {SH[s]}: CIC {r2_score(y[m], preds['CIC'][m,0]):+.3f}   U-PATCH {r2_score(y[m], preds['U-PATCH'][m,0]):+.3f}")
