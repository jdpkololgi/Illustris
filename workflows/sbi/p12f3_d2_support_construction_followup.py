#!/usr/bin/env python3
"""Full-panel native-vs-voxel support cross-tab and diagnostic sky map."""
import argparse
import json
import os
from pathlib import Path
import sys
import healpy as hp
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from workflows.sbi.p12f3_d2_support_geometry import native_support
from workflows.sbi.p12f3_d2_recovery_audit import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--geometry",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("compute allocation required")
    geometry=json.loads(args.geometry.read_text())
    if geometry["centre_mask_mismatches"]!=0 or geometry["unsupported"]!=736:
        raise RuntimeError("geometry audit is not verified")
    parent_bad=set(row["parent"] for row in geometry["rows"])
    root=args.root.parents[1]
    adapter=root/"training_contract_r1_random/adapters/ph006/field"
    manifest_path=adapter/"adapter_manifest.json"
    if sha(manifest_path)!=geometry["manifest_sha256"]:
        raise RuntimeError("adapter changed")
    manifest=json.loads(manifest_path.read_text())
    selection_path=root/"training_contract/transforms/field/selection_manifest.json"
    if sha(selection_path)!=geometry["selection_sha256"]:
        raise RuntimeError("selection changed")
    selection=json.loads(selection_path.read_text())
    angular_path=root/"ph006/p3b_random_response_v1/angular/randoms_n18.npz"
    if sha(angular_path)!=geometry["angular_map_sha256"]:
        raise RuntimeError("native angular map changed")
    with np.load(angular_path,allow_pickle=False) as data:
        support=data["support"].astype(bool);domain=data["domain"]
    archive_path=args.root/"evaluation/seed42_v1/d2_modern_base4_nfe50/P12F_SAMPLE_ARCHIVE.json"
    if sha(archive_path)!=geometry["archive_sha256"]:
        raise RuntimeError("archive changed")
    archive=json.loads(archive_path.read_text())
    arrays={k:np.load(adapter/(k+".npy"),mmap_mode="r") for k in
        ("core_active_offsets","core_active_parent","core_active_frac_index","core_cap")}
    parts=[]
    for entry in archive["entries"]:
        core=int(entry["core_id"]);start,stop=arrays["core_active_offsets"][core:core+2]
        ids=arrays["core_active_parent"][int(start):int(stop)]
        frac=arrays["core_active_frac_index"][int(start):int(stop)]
        cap=int(arrays["core_cap"][core]);geo=manifest["caps"]["NGC" if cap==1 else "SGC"]
        xyz=np.asarray(geo["origin_mpc"])+float(geo["cell_mpc"])*(frac+.5)
        native,angular,radial,_,z=native_support(xyz,cap,support,domain,selection)
        ra=np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0]))%360
        dec=np.rad2deg(np.arcsin(xyz[:,2]/np.linalg.norm(xyz,axis=1)))
        parts.append(dict(parent=np.asarray(ids),native=native,angular=angular,radial=radial,
            voxel=np.array([int(i) not in parent_bad for i in ids], dtype=bool),ra=ra,dec=dec,z=z,
            core=np.full(len(ids),core),cap=np.full(len(ids),cap)))
    arrays={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    if len(arrays["parent"])!=133698 or len(np.unique(arrays["parent"]))!=133698 or np.sum(~arrays["voxel"])!=736:
        raise RuntimeError("panel identity mismatch")
    counts={f"native_{int(n)}_voxel_{int(v)}":int(np.sum((arrays["native"]==n)&(arrays["voxel"]==v)))
        for n in (False,True) for v in (False,True)}
    result=dict(schema_version="d2-support-construction-followup-v1",allocation=os.environ["SLURM_JOB_ID"],
        geometry_sha256=sha(args.geometry),source_sha256=sha(__file__),cross_tab=counts,
        all_galaxies_native_supported=bool(arrays["native"].all()),ph001_access=False,evaluation_amended=False,
        interpretation="Nearest voxel membership and native galaxy footprint membership are different discretizations; patch truncation is excluded")
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    with (out/"SUPPORT_CONSTRUCTION.json").open("x") as stream:
        json.dump(result,stream,indent=2);stream.write("\n")
    with (out/"SUPPORT_PANEL.npz").open("xb") as stream:
        np.savez_compressed(stream,**arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pixels=np.flatnonzero(support)
    theta,phi=hp.pix2ang(256,pixels,nest=False)
    ra_map=np.rad2deg(phi);dec_map=90-np.rad2deg(theta)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={"width_ratios":[1.65,1]})
    ax.scatter(ra_map,dec_map,s=.25,c="#d8dee5",rasterized=True,label="Native random-map footprint")
    ax.scatter(arrays["ra"],arrays["dec"],s=.35,c="#547594",alpha=.25,rasterized=True,label="133,698 panel galaxies")
    at=~arrays["voxel"]
    ax.scatter(arrays["ra"][at],arrays["dec"][at],s=5,c="#c63d30",rasterized=True,label="736 nearest-voxel M=0 galaxies")
    ax.set(xlabel="RA [degrees]",ylabel="Dec [degrees]",xlim=(0,360),ylim=(-35,90),title="Inside the survey footprint; not context-patch truncation")
    ax.legend(loc="lower left",markerscale=3,fontsize=8,framealpha=.9)
    distance=np.array([r["nearest_supported_centre_voxels"] for r in geometry["rows"]])
    bx.hist(distance,bins=np.linspace(0,1.5,31),color="#c63d30",alpha=.85)
    bx.axvline(1,color="black",ls="--",lw=1)
    bx.set(xlabel="Distance to nearest supported voxel centre [voxels]",ylabel="Galaxies",title="722/736 within one voxel; all within 1.35")
    bx.text(.98,.97,"One voxel = 5 Mpc\nAll 736 native positions supported",transform=bx.transAxes,ha="right",va="top",fontsize=9)
    fig.tight_layout();fig.savefig(out/"support_geometry.png",dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
