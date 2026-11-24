"""Build a DESI cap map only from 32 complete Loa prediction pairs; no truth or refitting."""
import sys,os,socket
from contextlib import ExitStack
from pathlib import Path
import lowz_confirmation_inputs as b


def run(manifest,expected,cap):
    assert __debug__ and socket.gethostname().startswith('nid') and os.environ['SLURM_JOB_ID']
    assert Path.cwd()==b.DEPLOY and os.environ['GRAPHWEB_CFM_DATA_CONFIG']==b.CONFIG
    assert manifest.parent.parent==b.BASE and manifest.name=='RESPONSE_PREPARED.json' and b.sha(manifest)==expected
    sys.path.insert(0,str(b.DEPLOY))
    import h5py,numpy as np
    from workflows.sbi import e2e_coupled_coordinates as coord
    from workflows.sbi import e2e_coupled_contract as c
    assert cap in ('NGC','SGC')
    d=manifest.parent/cap; r=b.read(manifest)
    assert r['passed'] and b.sha(d/'RESPONSE_COMPLETE.json')==r['caps'][cap]
    output=d/('map_'+os.environ['SLURM_JOB_ID']);output.mkdir(exist_ok=False)
    edges=np.array([.01,.15,.25,.35,.45,.55]);counts=np.zeros(5,dtype='i8')
    path=output/'CAP_MAP.h5'; receipts=[]
    with ExitStack() as stack:
        inputs=[]
        for i in range(32):
            q=d/'predictions'/f'draw_{i:03d}'
            pred=b.read(q/'INPUTS.json');den=b.read(q/'density/COMPLETE.json');tide=b.read(q/'tides/COMPLETE.json')
            assert pred['response_manifest']==expected and pred['method']==r['method']
            assert all(v['phase']=='Loa' and v['cap']==cap for v in (pred,den,tide))
            assert den['first']==tide['first']==pred['draw']==i and den['passed'] and tide['passed']
            assert tide['binding']==b.sha(q/'tides/BINDING.json')
            assert b.read(q/'tides/BINDING.json')['input_receipt']['sha256']==b.sha(q/'density/COMPLETE.json')
            a=stack.enter_context(h5py.File(b.verify_record(den['field'],q/'density'),'r'))
            t=stack.enter_context(h5py.File(b.verify_record(tide['output'],q/'tides'),'r'))
            inputs.append((a,t));receipts.append([b.sha(q/'density/COMPLETE.json'),b.sha(q/'tides/COMPLETE.json')])
        first=inputs[0][0];shape=first['rho'].shape[1:];o=first.attrs['origin_mpc_h'];dx=6.766
        for a,t in inputs:
            assert a['rho'].shape==(1,*shape) and t['eigenvalues'].shape==(1,*shape,3)
            assert a.attrs['cell_mpc_h']==dx and np.array_equal(a.attrs['origin_mpc_h'],o) and np.array_equal(t.attrs['origin_mpc_h'],o)
        f=stack.enter_context(h5py.File(path,'x'));f.attrs.update(cap=cap,origin_mpc_h=o,cell_mpc_h=dx,draws=32,
            density='rho / mean rho',threshold=.2,smoothing_mpc_h=7.,class_order='void,sheet,filament,knot',
            uncertainty='pointwise empirical 5th,50th,95th percentiles',real_data_coverage_qualified=False)
        for name,n,dt in [('shell',0,'i1'),('support',0,'f4'),('rho_draw0',0,'f4'),('rho_quantiles',3,'f4'),
                          ('lambda_draw0',3,'f4'),('web_probability',4,'f4')]:
            f.create_dataset(name,shape=(*shape,*((n,) if n else ())),dtype=dt,chunks=True,compression='lzf',
                             fillvalue=-1 if dt=='i1' else np.nan)
        for lo in range(shape[0]):
            s=first['support'][lo];cov=first['coverage'][lo]
            x=o[0]+(lo+.5)*dx;y=o[1]+(np.arange(shape[1])+.5)*dx;z=o[2]+(np.arange(shape[2])+.5)*dx
            shell=np.searchsorted(edges,coord.redshift(np.sqrt(x*x+y[:,None]**2+z[None,:]**2)),side='right')-1
            m=(cov>0)&(s>0)&(shell>=0)&(shell<5);f['support'][lo]=s;f['shell'][lo]=np.where(m,shell,-1)
            if not np.any(m):continue
            counts+=np.bincount(shell[m],minlength=5)
            density=[];classes=np.zeros((m.sum(),4),dtype='i4')
            def write(name,value):
                arr=np.full((*m.shape,*value.shape[1:]),np.nan,dtype='f4');arr[m]=value;f[name][lo]=arr
            for i,(a,t) in enumerate(inputs):
                assert np.array_equal(a['support'][lo],s) and np.array_equal(a['coverage'][lo],cov)
                rho=a['rho'][0,lo][m];lam=t['eigenvalues'][0,lo][m]
                assert np.isfinite(rho).all() and (rho>0).all() and np.isfinite(lam).all() and (np.diff(lam,axis=-1)>=0).all()
                if i==0:write('rho_draw0',rho);write('lambda_draw0',lam)
                density.append(rho);classes+=np.eye(4,dtype='i4')[(lam>.2).sum(-1)]
            assert (classes.sum(1)==32).all()
            write('rho_quantiles',np.quantile(density,[.05,.5,.95],axis=0).T);write('web_probability',classes/32)
    c.atomic_json(output/'MAP_COMPLETE.json',dict(passed=True,cap=cap,source=b.sha(__file__),manifest=expected,
        receipts=receipts,output=c.file_record(path,content_hash=True),shells=edges.tolist(),cells=counts.tolist(),
        qualification='conditional on mock-trained model; not calibrated against real matter truth'))


if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
