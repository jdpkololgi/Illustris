"""Draw-zero inspection: every registered ph006 pair, central z-index, no cherry-picking."""
import json
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v3")
    data = root/"coupled_lowz_20261029_v1/cartesian_v2"
    out = Path(__file__).resolve().parents[2]/"docs/figures/lowz_ph006_draw_zero"
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((data/"geometry/ph006/GEOMETRY_COMPLETE.json").read_text())["pairs"]
    for cap, job in [("NGC",61610899),("SGC",61582948)]:
        directory = root/f"lowz_cap_draw_{job}"/cap
        fig, axes = plt.subplots(8, 4, figsize=(10,15), layout="constrained")
        with h5py.File(directory/"density/FIELD.h5") as f, h5py.File(directory/"tides/COMPLETED_FIELD.h5") as t:
            for ax,r in zip(axes,[r for r in rows if r["cap"]==cap],strict=True):
                raw = np.array(r["center"])-16
                offset = (np.array(r["grid"]["origin_mpc_h"])+3.383*raw-f.attrs["origin_mpc_h"])/6.766
                start = np.rint(offset).astype(int)
                np.testing.assert_allclose(start,offset,rtol=0,atol=1e-8)
                a,b,c = start
                sl = (slice(a,a+32),slice(b,b+16),c+8)
                mask = f["support"][sl]>0
                xyz = [f.attrs["origin_mpc_h"][i]+6.766*(k+.5) for i,k in enumerate((np.arange(a,a+32)[:,None],np.arange(b,b+16)[None,:],c+8))]
                from workflows.sbi.e2e_coupled_coordinates import radius_mpc_h
                rad = np.sqrt(sum(x*x for x in xyz))
                lo,hi=radius_mpc_h(np.array([.01,.55]))
                mask &= (rad>=lo)&(rad<hi)
                receipt = json.loads((data/"targets/ph006"/(r["pair_id"]+".json")).read_text())
                with h5py.File(receipt["outputs"][0]["path"]) as src:
                    truth = src["rho_joint"][16:48,16:32,24]
                    v = src["fullbox_tensor_cores"][:].reshape(32,16,16,6)[:,:,8,:]
                    eig = np.linalg.eigvalsh(v[...,[0,1,2,1,3,4,2,4,5]].reshape(32,16,3,3))
                fields=[truth,f["rho"][(0,*sl)],(eig>.2).sum(-1),(t["eigenvalues"][(0,*sl)]>.2).sum(-1)]
                for i,(p,x) in enumerate(zip(ax,fields)):
                    im=p.imshow(np.ma.array(x.T,mask=~mask.T),origin="lower",interpolation="nearest",
                                cmap="viridis" if i<2 else plt.get_cmap("plasma",4),
                                vmin=0 if i<2 else -.5,vmax=2 if i<2 else 3.5)
                    p.set_xticks([]);p.set_yticks([])
                ax[0].set_ylabel(r["pair_id"].split(cap+"_")[1].replace("_00",""),fontsize=8)
        for p,title in zip(axes[0],["True density","Draw density","True T-web","Draw T-web"]): p.set_title(title)
        fig.suptitle(f"ph006 {cap} | one draw | central slices of all eight registered pairs\n"
                     "Density scale 0–2 (clipped for display); web: dark to light = void, sheet, filament, knot; white = masked")
        fig.savefig(out/f"{cap}.png",dpi=110);plt.close(fig)

if __name__=="__main__": main()
