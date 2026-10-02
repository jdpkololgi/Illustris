"""Observed-footprint, free-space tidal pushforward of saved coherent draws.

No mock truth, density resampling, exterior fit, or production VAC mutation.
NGC and SGC source contributions are added in physical coordinates; draw IDs
are paired under the existing independent-cap approximation, not a calibrated
cross-cap posterior. Masks and densities are immutable inputs.
"""
import argparse,json,os,time,gc
from pathlib import Path
import h5py
import numpy as np
from scipy import fft,ndimage
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_footprint_tidal_operator as op


def pool2(a):
    x,y,z=a.shape[:3];tail=a.shape[3:]
    return a.reshape(x//2,2,y//2,2,z//2,2,*tail).mean((1,3,5))


def load_cap(root,cap):
    receipt=json.loads((root/cap/'COMPLETE.json').read_text());path=root/cap/'FIELD.h5'
    if not receipt['passed'] or c.sha256(path)!=receipt['field']['sha256']:raise ValueError('Density binding failed')
    with h5py.File(path) as f:
        support=f['support'][:];coverage=f['coverage'][:]
        if np.any((support>0)&(coverage==0)):raise ValueError('Unsupported observed source')
        occupied=np.where(support>0)
        low=np.array([x.min()//2*2 for x in occupied]);high=np.array([(x.max()+2)//2*2 for x in occupied])
        crop=tuple(slice(int(a),int(b)) for a,b in zip(low,high))
        support=support[crop];coverage=coverage[crop]
        source=np.stack([op.source_delta(f['rho'][i][crop],support,coverage) for i in range(4)])
        origin=np.asarray(f.attrs['origin_mpc_h'])+low*6.766
    return dict(cap=cap,source=source,support=support,coverage=coverage,shape=support.shape,
                origin=origin,original_crop=[low.tolist(),high.tolist()],density_sha256=receipt['field']['sha256'])


def source_at_targets(source,target_shape,offset):
    result=np.zeros(target_shape,np.float64);ti=[];si=[]
    for n,ns,o in zip(target_shape,source.shape,offset):
        index=np.floor(np.arange(n)+o+.5).astype(int);valid=(index>=0)&(index<ns)
        ti.append(np.flatnonzero(valid));si.append(index[valid])
    if all(len(v)>0 for v in ti):result[np.ix_(*ti)]=source[np.ix_(*si)]
    return result


def run(root,out,workers=32,threshold=.2,benchmark=False):
    c.require_compute()
    if not np.isfinite(threshold):raise ValueError('Finite threshold required')
    if not json.loads((root/'VERIFIED.json').read_text())['passed']:raise ValueError('Density verification missing')
    out.mkdir(parents=True,exist_ok=False)
    caps=[load_cap(root,cap) for cap in ('NGC','SGC')]
    binding=dict(schema='observed-footprint-tidal-control-v1',job=os.environ['SLURM_JOB_ID'],
        draws=4,threshold=threshold,cell_mpc_h=6.766,components=['xx','xy','xz','yy','yz','zz'],
        boundary='isolated/free-space; delta=0 outside observed source; no periodic images',
        source_mask='unchanged fractional occupancy from binary DESI support; not completeness/exposure weighting',
        reported_native_mask='support>=0.5',reported_display_mask='pool2(support)>=0.5 and complete density coverage',
        discretization='piecewise-constant cubic source cells, exact cuboid integral at target cell centres',
        smoothing='inherits saved smoothed density; no additional Gaussian smoothing; mask applied after density smoothing',
        exterior_marginalized=False,posterior_qualified=False,
        cap_pairing='same draw index under product of independent cap samplers; cross-cap covariance unqualified',
        source={Path(__file__).name:c.sha256(__file__),Path(op.__file__).name:c.sha256(op.__file__)},
        inputs=[dict(cap=v['cap'],density_sha256=v['density_sha256'],crop=v['original_crop'],origin=v['origin'].tolist()) for v in caps])
    c.atomic_json(out/'BINDING.json',binding);results=[];begin=time.monotonic()
    for target in caps:
        shape=target['shape'];total=np.zeros((4,*shape,6),np.float64);expected=np.zeros((4,*shape),np.float64)
        terms=[]
        for src in caps:
            offset=(target['origin']-src['origin'])/6.766;start=time.monotonic()
            kernels=op.kernel(src['shape'],shape,offset);pad=kernels.shape[1:]
            kernel_seconds=time.monotonic()-start
            print(json.dumps(dict(target=target['cap'],source=src['cap'],kernel_seconds=kernel_seconds,padded_shape=pad)),flush=True)
            spectra=[fft.rfftn(d,s=pad,workers=workers) for d in src['source']]
            cut=tuple(slice(0,n) for n in shape);cross_squared=0.;max_component=0.
            for component in range(6):
                k=fft.rfftn(kernels[component],workers=workers)
                for draw,d in enumerate(spectra):
                    value=fft.irfftn(d*k,s=pad,workers=workers)[cut]
                    total[draw,...,component]+=value
                    cross_squared+=float(np.sum(value*value));max_component=max(max_component,float(np.max(abs(value))))
                del k
                print(json.dumps(dict(target=target['cap'],source=src['cap'],component=component,seconds=time.monotonic()-start)),flush=True)
                if benchmark:
                    record=dict(passed=bool(np.isfinite(total).all()),kernel_seconds=kernel_seconds,
                        one_component_four_draw_seconds=time.monotonic()-start-kernel_seconds,padded_shape=pad,
                        target_shape=shape,source_shape=src['shape'],posterior_output=False)
                    c.atomic_json(out/'BENCHMARK.json',record);print(json.dumps(record),flush=True);return
            for draw in range(4):expected[draw]+=source_at_targets(src['source'][draw],shape,offset)
            terms.append(dict(source=src['cap'],seconds=time.monotonic()-start,
                component_rms=float(np.sqrt(cross_squared/(4*np.prod(shape)*6))),max_abs=max_component))
            del kernels,spectra,value;gc.collect()
        trace_error=float(np.max(abs(total[...,[0,3,5]].sum(-1)-expected)))
        if not np.isfinite(total).all() or trace_error>2e-7:raise ValueError(f'Poisson trace failed: {trace_error}')
        valid=target['support']>=.5
        display_mask=(pool2(target['support'])>=.5)&(pool2((target['coverage']>0).astype(float))==1)
        path=out/(target['cap']+'_TIDAL.h5');display_eigen=[]
        with h5py.File(path,'x') as f:
            f.attrs.update(cap=target['cap'],cell_mpc_h=6.766,threshold=threshold,origin_mpc_h=target['origin'],
                           boundary=binding['boundary'],posterior_qualified=False,exterior_marginalized=False)
            for name,a in [('support',target['support']),('mask',valid),('display_mask',display_mask)]:f.create_dataset(name,data=a,compression='lzf')
            distance=ndimage.distance_transform_edt(np.pad(target['support']==1,1))[1:-1,1:-1,1:-1]*6.766
            f.create_dataset('distance_to_nonfull_support_mpc_h',data=distance.astype('f4'),compression='lzf')
            tensor=f.create_dataset('tensor',shape=total.shape,dtype='f4',chunks=(1,8,32,32,6),compression='lzf')
            eigen=f.create_dataset('eigenvalues',shape=(4,*shape,3),dtype='f8',chunks=(1,8,32,32,3),compression='lzf')
            labels=f.create_dataset('class',shape=(4,*shape),dtype='u1',chunks=(1,8,32,32),compression='lzf',fillvalue=255)
            counts=np.zeros((*shape,4),dtype='u1')
            for draw in range(4):
                # Average tensors before diagonalising at display resolution,
                # matching the existing viewer convention; never average labels.
                de=op.eigensystem(pool2(total[draw]));de[~display_mask]=np.nan;display_eigen.append(de)
                for lo in range(0,shape[0],8):
                    hi=min(lo+8,shape[0]);v=total[draw,lo:hi];e=op.eigensystem(v);m=valid[lo:hi]
                    label=(e>threshold).sum(-1).astype('u1')
                    for cls in range(4):counts[lo:hi,...,cls]+=(label==cls)&m
                    e[~m]=np.nan;t=v.astype('f4');t[~m]=np.nan;label[~m]=255
                    tensor[draw,lo:hi]=t;eigen[draw,lo:hi]=e;labels[draw,lo:hi]=label
            prob=counts.astype('f4')/4;prob[~valid]=np.nan
            f.create_dataset('class_probability',data=prob,compression='lzf')
            de=np.stack(display_eigen);f.create_dataset('display_eigenvalues',data=de,compression='lzf')
            dp=np.stack([((de>threshold).sum(-1)==cls).mean(0) for cls in range(4)],axis=-1).astype('f4');dp[~display_mask]=np.nan
            f.create_dataset('display_class_probability',data=dp,compression='lzf')
            f.attrs['display_cell_mpc_h']=13.532;f.attrs['display_origin_mpc_h']=target['origin']+6.766
        result=dict(cap=target['cap'],field=c.file_record(path,content_hash=True),trace_max_abs=trace_error,
            supported_native_cells=int(valid.sum()),supported_display_cells=int(display_mask.sum()),terms=terms)
        results.append(result);print(json.dumps(result),flush=True)
        del total,expected,counts,prob,de,dp,display_eigen;gc.collect()
    c.atomic_json(out/'COMPLETE.json',dict(passed=True,binding=c.sha256(out/'BINDING.json'),results=results,
        seconds=time.monotonic()-begin,posterior_qualified=False,control_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--threshold',type=float,default=.2);p.add_argument('--benchmark',action='store_true')
    a=p.parse_args();run(a.root,a.out,a.workers,a.threshold,a.benchmark)
