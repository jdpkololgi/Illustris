"""Information-loss probe on fixed TNG truth under nested random thinning.

No neural training or inference. Periodic neighbour counts are compared with
the same native saved R~2 tidal trace at fixed query galaxies for every level.
"""
import hashlib
import json
import struct
import zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

OUT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010/legacy_selection_tests_20261014')
GRAPH=Path('/pscratch/sd/d/dkololgi/tng_illustris/cache/from_repo_root/processed_gcn_data_alpha_mc1e+09.pt')
EIG=Path('/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/snap_099_eig_vals_0.npz')
def mmap_member(path,name):
    with zipfile.ZipFile(path) as z:info=z.getinfo(name+'.npy');assert info.compress_type==zipfile.ZIP_STORED
    with path.open('rb') as f:
        f.seek(info.header_offset);h=f.read(30);fn,ex=struct.unpack_from('<HH',h,26);f.seek(fn+ex,1)
        version=np.lib.format.read_magic(f);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,order,dtype=reader(f);assert not order;offset=f.tell()
    return np.memmap(path,mode='r',offset=offset,dtype=dtype,shape=shape)

def main():
    torch.set_num_threads(1);graph,_,_=torch.load(GRAPH,map_location='cpu',weights_only=False)
    pos=graph.pos.numpy()*.6774;box=205.;assert np.all((pos>=0)&(pos<box))
    rng=np.random.default_rng(20261014);query=np.sort(rng.choice(len(pos),10000,replace=False));pix=(pos[query]/box*512).astype(int)
    eig=mmap_member(EIG,'eig_vals');trace=np.array(eig[:,pix[:,0],pix[:,1],pix[:,2]]).sum(0)
    records=[]
    for seed in [20261014,20261015,20261016]:
        uniform=np.random.default_rng(seed).random(len(pos))
        for frac in [1.,.3,.1,.03,.01]:
            selected=uniform<frac;tree=cKDTree(pos[selected],boxsize=box)
            for radius in [2.,7.]:
                count=tree.query_ball_point(pos[query],radius,workers=1,return_length=True)-selected[query]
                assert (count>=0).all()
                # Small independent periodic brute-force check.
                if seed==20261014:
                    for j in [0,4999,9999]:
                        delta=abs(pos[selected]-pos[query[j]]);delta=np.minimum(delta,box-delta)
                        n=(np.sum(delta*delta,axis=1)<=radius**2).sum()-selected[query[j]]
                        assert n==count[j]
                records.append(dict(seed=seed,fraction=frac,tracers=int(selected.sum()),n_h3_mpc3=float(selected.sum()/box**3),radius_hmpc=radius,median_count=float(np.median(count)),zero_fraction=float(np.mean(count==0)),spearman_with_fixed_R2_trace=float(spearmanr(count,trace).statistic)))
            print(seed,frac,int(selected.sum()),flush=True)
    current=np.load('/global/homes/d/dkololgi/fmsu_tokyo_20261010/raw_environment_checks_20261014/raw_environment_catalogue.npz')
    nc=current['NTILDE_MPC3']/.6766**3
    result=dict(rows=len(pos),box_hmpc=box,full_density_h3_mpc3=len(pos)/box**3,queries=len(query),query_index_sha256=hashlib.sha256(query.tobytes()).hexdigest(),current_ntilde_h3_mpc3_quantiles=np.quantile(nc,[.1,.5,.9]).tolist(),records=records,
                method='Three seeds, nested independent Bernoulli thinning of identical periodic TNG tracers. Same 10000 query galaxies and native saved R~2 tidal trace at every level. Query self excluded whether present or absent in thinned sample. No trained predictor or property outcomes; this isolates count information under random thinning only. No claim of exact DESI selection or galaxy bias parity. R7 here is a count aperture, not a new R7 truth field.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'tracer_thinning.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
