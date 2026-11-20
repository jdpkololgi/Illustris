"""Phase-local BRIGHT point neighborhoods; no target or simulator-only features."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p12a_lowz_20261029_v1/training_contract_v1")
REF = Path("/pscratch/sd/d/dkololgi/abacus/p12a_lowz_full_seed42_61548250/full/unet/seed_42")
DEV = ("ph006", "ph012", "ph013")
BIN = 16  # lattice cells, index only; no quantisation of model coordinates


def read(p):
    return json.loads(p.read_text())


def array(p):
    return np.load(p, mmap_mode="r", allow_pickle=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def roles(root):
    m = read(root/"TRAINING_LOADER_READY.json")
    r = m["roles"]
    tr, va, sealed = map(tuple, (r["training"], r["validation_and_selection"], r["sealed_blind_test"]))
    require(m["pass"] and va == DEV and len(tr)==20 and set(sealed)=={"ph018","ph019"}
            and len(set(tr+va+sealed))==25, "invalid phase split")
    return tr+va


def pack(xyz, parent, origin, cell):
    bins = np.floor(((xyz-origin)/cell-.5)/BIN).astype(np.int64)
    lo = bins.min(0)
    shape = bins.max(0)-lo+1
    k = np.ravel_multi_index((bins-lo).T,shape)
    order = np.argsort(k,kind="stable")
    offsets = np.r_[0,np.cumsum(np.bincount(k,minlength=int(shape.prod())))]
    return parent[order], offsets, lo, shape


def query(xyz, index, origin, cell, left, right):
    parent, offsets, lo, shape = index
    start = np.maximum(np.floor(left/BIN).astype(int)-lo,0)
    stop = np.minimum(np.ceil(right/BIN).astype(int)-lo,shape)
    if np.any(start>=stop):
        return np.empty(0,dtype=np.int64)
    bins = (np.indices(stop-start).reshape(3,-1).T+start)
    keys = np.ravel_multi_index(bins.T,shape)
    rows = np.concatenate([parent[offsets[k]:offsets[k+1]] for k in keys])
    frac = (xyz[rows,:3]-origin)/cell-.5
    return np.sort(rows[np.all((frac>=left)&(frac<right),axis=1)])


class PointPhase:
    def __init__(self, root, directory, phase):
        if phase not in roles(root):
            raise PermissionError(phase)
        self.a = root/"adapters"/phase/"field"
        self.m = read(self.a/"adapter_manifest.json")
        self.xyz = array(Path(self.m["points"]))
        self.arr = {k:array(self.a/("core_"+k+".npy")) for k in
                    ("voxel_start","voxel_stop","cap","active_offsets","active_parent")}
        self.bin = {c:tuple(array(directory/phase/f"{c}_{k}.npy") for k in
                        ("parent","offset","lo","shape")) for c in ("NGC","SGC")}

    def patch(self,i,halo=24):
        a = self.arr
        require(0<=i<len(a["cap"]) and halo>=0,"bad patch")
        cap = ("SGC","NGC")[int(a["cap"][i])]
        g = self.m["caps"][cap]
        left = (a["voxel_start"][i].astype(int)-halo)//8*8
        right = ((a["voxel_stop"][i].astype(int)+halo+7)//8)*8
        left,right=np.maximum(left,0),np.minimum(right,g["shape"])
        rows=query(self.xyz,self.bin[cap],np.array(g["origin_mpc"]),g["cell_mpc"],left,right)
        x,y=a["active_offsets"][i:i+2]
        core=a["active_parent"][x:y]
        ix=np.searchsorted(rows,core)
        require(np.all(ix<len(rows)) and np.array_equal(rows[ix],core),"missing authoritative galaxy")
        return rows, self.xyz[rows,:3], ix


def build(root,directory,phase,ref):
    require(phase in roles(root),"phase is sealed or unregistered")
    p=root/"phases"/phase
    a=root/"adapters"/phase/"field"
    contract=read(p/"phase_contract.json")
    meta=read(Path(contract["inputs"]["p1_manifest"]))
    require(meta["phase"]==phase and meta["parent_rows_are_canonical_rows"],"wrong row basis")
    points=array(Path(meta["points"]))
    targetid=array(p/"parent_targetid.npy")
    z=array(p/"parent_redshift.npy")
    with np.load(meta["index"]) as f:
        np.testing.assert_array_equal(f["parent_node_id"],np.arange(len(points)))
        np.testing.assert_array_equal(f["targetid"],targetid)
        context=f["context"].astype(bool)
        np.testing.assert_array_equal(f["cap"],points[:,3])
    gids=read(a/"adapter_manifest.json")["caps"]
    dest=directory/phase
    dest.mkdir()
    for c,g in gids.items():
        parent=np.flatnonzero(context & (points[:,3]==g["cap_id"]))
        require(np.isfinite(points[parent,:3]).all(),"nonfinite observed position")
        for k,v in zip(("parent","offset","lo","shape"),pack(points[parent,:3],parent,
                np.array(g["origin_mpc"]),g["cell_mpc"])):
            np.save(dest/f"{c}_{k}.npy",v)
    parents=array(a/"core_active_parent.npy")
    with np.load(contract["inputs"]["assignment"]) as s:
        ok=s["supervised_eligible"].astype(bool)
        wanted=s["parent_node_id"][ok]
        cores=s["core_id"][ok]
        order=np.lexsort((wanted,cores))
        np.testing.assert_array_equal(parents,wanted[order])
        np.testing.assert_array_equal(array(a/"core_active_offsets.npy"),np.r_[0,np.cumsum(
            np.bincount(cores,minlength=len(array(a/"core_cap.npy"))))])
        np.testing.assert_array_equal(s["cap"][ok],points[wanted,3])
        shell=s["shell"][ok]
        hist=np.bincount(shell,minlength=4)
        np.testing.assert_array_equal(hist,[contract["shell_counts"][str(i)] for i in range(4)])
    require(len(np.unique(wanted))==len(wanted)==contract["authoritative_rows"] and
            len(np.unique(targetid[wanted]))==len(wanted) and context[wanted].all(),"row identity")
    require(np.isfinite(z[wanted]).all() and z[wanted].min()>=.01 and z[wanted].max()<=.55
            and (z[wanted]<.15).any(),"low-z or domain failure")
    if phase in DEV:
        np.testing.assert_array_equal(np.sort(wanted),np.sort(array(ref/f"best_validation_{phase}_parent_node_id.npy")))
    loader=PointPhase(root,directory,phase)
    ids=array(p/("validation_core_id.npy" if phase in DEV else "training_core_id.npy"))
    for i in ids[np.linspace(0,len(ids)-1,12,dtype=int)]:
        rows,_,ix=loader.patch(i)
        require(context[rows].all() and len(ix)>0 and len(np.unique(rows))==len(rows),"patch mismatch")
    receipt=dict(phase=phase,authoritative=len(wanted),shell_counts=hist.tolist(),
                 below_z015=int((z[wanted]<.15).sum()),context=int(context.sum()),sampled_patches=12)
    (dest/"INDEX_READY.json").write_text(json.dumps(receipt))
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    for phase in (*DEV,*(p for p in roles(ROOT) if p not in DEV)):
        build(ROOT,args.directory,phase,REF)
    (args.directory/"INDEX_READY.json").write_text(json.dumps({"phases":roles(ROOT),"pass":True}))


if __name__=="__main__":
    main()
