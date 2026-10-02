"""Independent stored-field and catalogue-coverage checks, not calibration."""
import argparse,json,os
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c,e2e_coupled_views as views

def blocks(a):
    b,x,y,z=a.shape
    return a.reshape(b,x//4,4,y//4,4,z//4,4)

def verify(root,atlas):
    c.require_compute();plan=json.loads((atlas/'PLAN.json').read_text())
    assert c.sha256(atlas/'vac_properties.npz')==plan['properties_sha256']
    with np.load(atlas/'vac_properties.npz') as f:xyz=f['xyz'];caps=f['CAP']
    results=[]
    for cap,capid in [('NGC',1),('SGC',0)]:
        directory=root/cap;receipt=json.loads((directory/'COMPLETE.json').read_text())
        assert receipt['passed'] and receipt['draws']==4 and receipt['windows']==plan['counts'][cap]['tiles']
        binding=json.loads((directory/'BINDING.json').read_text())
        assert binding['nfe']==128 and binding['plan']==c.sha256(atlas/'PLAN.json')
        assert c.sha256(directory/'BINDING.json')==receipt['binding']
        path=Path(receipt['field']['path']);assert c.sha256(path)==receipt['field']['sha256']
        chart=views.load_chart(binding['normalizer']);maximum=0.;count=0
        with h5py.File(path,'r') as f:
            coverage=f['coverage'][:];shape=np.array(coverage.shape);origin=np.asarray(f.attrs['origin_mpc_h'])
            offset=np.rint((origin-np.asarray(plan['grids'][cap]['origin_mpc_h']))/27.064-f.attrs['coarse_origin_raw_cells']/8).astype(int)
            assert np.array_equal(offset,[16,18,18]),offset
            for lo in range(0,shape[0],16):
                hi=min(lo+16,shape[0]);rho=f['rho'][:,lo:hi].astype('f8');valid=coverage[lo:hi]>0
                assert np.array_equal(np.isfinite(rho).all(0),valid)
                assert np.all(rho[:,valid]>0)
                block_valid=blocks(valid[None].astype('u1')).min((2,4,6))[0].astype(bool)
                assert np.array_equal(block_valid,blocks(valid[None].astype('u1')).max((2,4,6))[0].astype(bool))
                coarse=f['coarse_logrho_standardized'][:,offset[0]+lo//4:offset[0]+hi//4,
                    offset[1]:offset[1]+shape[1]//4,offset[2]:offset[2]+shape[2]//4].astype('f8')
                coarse=np.exp(coarse*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
                means=blocks(rho).mean((2,4,6))
                if np.any(block_valid):
                    maximum=max(maximum,float(np.max(abs(means[:,block_valid]-coarse[:,block_valid])/coarse[:,block_valid])))
                count+=int(valid.sum())
            assert maximum<2e-6,maximum
            rows=np.flatnonzero(caps==capid);checked=0
            for start in range(0,len(rows),100000):
                points=(xyz[rows[start:start+100000]]-origin)/6.766-.5
                lower=np.floor(points).astype(int)
                assert np.all(lower>=0) and np.all(lower+1<shape)
                for dx in (0,1):
                    for dy in (0,1):
                        for dz in (0,1):assert np.all(coverage[lower[:,0]+dx,lower[:,1]+dy,lower[:,2]+dz]>0)
                checked+=len(points)
            assert checked==plan['counts'][cap]['vac_rows']
        result={'cap':cap,'catalogue_rows_with_all_eight_interpolation_corners_covered':checked,
                'covered_native_cells':count,'stored_float32_mass_max_relative_error':maximum,'field_sha256':c.sha256(path)}
        results.append(result);print(json.dumps(result),flush=True)
    c.atomic_json(root/'VERIFIED.json',{'passed':True,'job':os.environ['SLURM_JOB_ID'],'caps':results,
        'posterior_qualified':False,'scope':'Storage arithmetic and geometric catalogue coverage only; no mock truth or statistical calibration.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--atlas',type=Path,required=True)
    a=p.parse_args();verify(a.root,a.atlas)
