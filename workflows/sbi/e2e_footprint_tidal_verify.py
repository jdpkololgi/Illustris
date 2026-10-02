"""Independent stored-product validation of the isolated footprint control."""
import argparse,json,os
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_footprint_tidal_control import load_cap,pool2,source_at_targets


def verify(root,out):
    c.require_compute();receipt=json.loads((out/'COMPLETE.json').read_text())
    binding=json.loads((out/'BINDING.json').read_text())
    assert receipt['passed'] and receipt['control_only'] and not receipt['posterior_qualified']
    assert c.sha256(out/'BINDING.json')==receipt['binding']
    assert binding['draws']==4 and not binding['exterior_marginalized']
    caps=[load_cap(root,cap) for cap in ('NGC','SGC')];results=[]
    for target in caps:
        record=next(r for r in receipt['results'] if r['cap']==target['cap']);path=Path(record['field']['path'])
        assert c.sha256(path)==record['field']['sha256']
        threshold=binding['threshold'];mask=target['support']>=.5;maximum_trace=0.;maximum_eigen=0.
        class_counts=np.zeros(4,np.int64)
        cross_occupied=0
        for src in caps:
            if src['cap']!=target['cap']:
                cross_occupied+=int(np.count_nonzero(source_at_targets(src['support'],target['shape'],
                    (target['origin']-src['origin'])/6.766)[mask]>0))
        assert cross_occupied==0,'Discrete cap sources overlap at reported target centres'
        with h5py.File(path) as f:
            assert np.array_equal(mask,f['mask'][:])
            assert np.array_equal(target['support'],f['support'][:])
            dm=(pool2(target['support'])>=.5)&(pool2((target['coverage']>0).astype(float))==1)
            assert np.array_equal(dm,f['display_mask'][:])
            for draw in range(4):
                expected=sum(source_at_targets(src['source'][draw],target['shape'],
                    (target['origin']-src['origin'])/6.766) for src in caps)
                for lo in range(0,target['shape'][0],8):
                    hi=min(lo+8,target['shape'][0]);m=mask[lo:hi]
                    t=f['tensor'][draw,lo:hi].astype(float);e=f['eigenvalues'][draw,lo:hi];labels=f['class'][draw,lo:hi]
                    assert np.array_equal(np.isfinite(t).all(-1),m)
                    assert np.array_equal(np.isfinite(e).all(-1),m)
                    assert np.all(np.isnan(t[~m])) and np.all(np.isnan(e[~m])) and np.all(labels[~m]==255)
                    assert np.all(np.diff(e[m],axis=-1)>=0)
                    assert np.array_equal(labels[m],(e[m]>threshold).sum(-1))
                    class_counts+=np.bincount(labels[m],minlength=4)
                    if np.any(m):
                        maximum_trace=max(maximum_trace,float(np.max(abs(t[...,[0,3,5]].sum(-1)[m]-expected[lo:hi][m]))))
                        # Independent matrix assembly on every 97th valid cell.
                        v=t[m][::97];a=np.empty((len(v),3,3))
                        a[:,0,:]=v[:,[0,1,2]];a[:,1,:]=v[:,[1,3,4]];a[:,2,:]=v[:,[2,4,5]]
                        maximum_eigen=max(maximum_eigen,float(np.max(abs(np.linalg.eigvalsh(a)-e[m][::97]))))
            for lo in range(0,target['shape'][0],8):
                hi=min(lo+8,target['shape'][0]);m=mask[lo:hi]
                labels=f['class'][:,lo:hi];p=f['class_probability'][lo:hi]
                actual=np.stack([(labels==k).mean(0) for k in range(4)],axis=-1)
                assert np.array_equal(p[m],actual[m]) and np.all(p[m].sum(-1)==1)
                assert np.all(np.isnan(p[~m]))
            de=f['display_eigenvalues'][:];dp=f['display_class_probability'][:]
            assert np.array_equal(np.isfinite(de).all((0,4)),dm)
            labels=(de>threshold).sum(-1)
            p=np.stack([(labels==k).mean(0) for k in range(4)],axis=-1)
            assert np.array_equal(p[dm],dp[dm]) and np.all(np.isnan(dp[~dm]))
        assert maximum_trace<3e-7 and maximum_eigen<3e-7
        result=dict(cap=target['cap'],supported_native_cells=int(mask.sum()),display_cells=int(dm.sum()),
            stored_tensor_trace_max_abs=maximum_trace,stored_tensor_eigen_max_abs=maximum_eigen,
            pooled_drawwise_class_fractions=(class_counts/class_counts.sum()).tolist(),masks_unchanged=True,
            classification_exact=True,overlapping_other_cap_source_cells=cross_occupied,sha256=record['field']['sha256'])
        print(json.dumps(result),flush=True);results.append(result)
    c.atomic_json(out/'VERIFIED.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],results=results,
        numerical_only=True,posterior_calibrated=False,exterior_error_measured=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify(a.root,a.out)
