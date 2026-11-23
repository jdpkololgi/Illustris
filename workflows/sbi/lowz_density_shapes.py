"""Paired truth/draw PDFs and tapered pseudo-spectra in registered dev core pairs."""
import os
from collections import defaultdict
import h5py
import numpy as np
import lowz_ensemble_scores as e
from matplotlib import pyplot as plt


def run():
    p=e.pilot; e.c.require_compute()
    ref=e.root/'lowz_calibration_61647276/CALIBRATION.json'
    archive=e.read(ref); prior=e.root/'lowz_finish_dev_61811977/FINISH_COMPLETE.json'
    replay=e.read(prior); assert replay['passed'] and replay['source_sha256']==e.c.sha256(ref)
    previous={r['pair']:r for r in replay['records']}
    out=e.root/('lowz_dev_density_shapes_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    edges=np.linspace(0,4,65);hist=defaultdict(lambda:np.zeros((33,len(edges)),dtype='i8'));powers=defaultdict(list)
    medratio=defaultdict(list); rows=[]; checked=set()
    for ph in e.phases:
        p.phase=ph
        for cap in ('NGC','SGC'):
            for row in p.rows(cap):
                pair=row['pair_id']; vals=[];meta=previous[pair]
                truth=p.read_target(p.checked(e.read(p.coord.ROOT/'targets'/ph/(pair+'.json'))),ph,pair)
                y=p.owned_core_arrays(truth['rho_joint']).reshape(32,16,16)
                for i in range(32):
                    path=e.root.__class__(archive['paths'][str((ph,cap,i))])/'density/COMPLETE.json'
                    rec=e.read(path);assert rec['passed'] and (rec['phase'],rec['cap'],rec['first'])==(ph,cap,i)
                    item=rec['field']
                    if path not in checked:
                        assert e.c.sha256(item['path'])==item['sha256']
                        checked.add(path)
                    with h5py.File(item['path']) as f:
                        sl=p.cuts(row,f.attrs['origin_mpc_h'])
                        if i==0:
                            xyz=np.meshgrid(*[(np.arange(s.start,s.stop)+.5)*6.766+o for s,o in zip(sl,f.attrs['origin_mpc_h'])],indexing='ij')
                            z=p.coord.redshift(np.sqrt(sum(a*a for a in xyz)));m=(f['support'][sl]>0)&(z>=.01)&(z<.55)
                        vals.append(f['rho'][(0,*sl)][m])
                # Same truth and per-draw pixels as the original contrast replay.
                a=np.stack([y[m],*vals]);assert np.isfinite(a).all() and (a>0).all()
                np.testing.assert_allclose(a.std(1),meta['contrast']['sd'],rtol=2e-6)
                counts=np.stack([np.bincount(np.minimum(np.searchsorted(edges,v,side='right')-1,len(edges)-1),minlength=len(edges)) for v in a])
                assert (counts.sum(1)==int(m.sum())).all()
                hist[ph]+=counts
                power=np.asarray(meta['contrast']['power'])
                powers[ph].append(power)
                medratio[ph].append((np.median(power[:,1:],axis=1)/power[:,0]).tolist())
                rows.append(dict(phase=ph,cap=cap,pair=pair,cells=int(m.sum()),counts=counts.tolist()))
    assert len(checked)==192 and len(rows)==48
    k=np.array(meta['contrast']['k_edges'])
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for col,ph in enumerate(e.phases):
        h=hist[ph]/hist[ph].sum(1,keepdims=True)
        for ax,a,v in [(axes[0,col],h[:,:-1],(edges[:-1]+edges[1:])/2),
                       (axes[1,col],np.mean(powers[ph],axis=0).T,(k[:-1]+k[1:])/2)]:
            ax.fill_between(v,*np.quantile(a[1:],[.05,.95],axis=0),color='steelblue',alpha=.25,label='5–95% draws')
            ax.plot(v,np.median(a[1:],axis=0),color='steelblue',label='Median draw')
            ax.plot(v,a[0],color='black',lw=1,label='Truth')
            ax.set_yscale('log')
        axes[0,col].set_title(ph);axes[0,col].set_xlabel('Density / cosmic mean');axes[1,col].set_xlabel('k [h/Mpc]')
    axes[0,0].set_ylabel('Cell fraction / bin');axes[1,0].set_ylabel('Windowed local power');axes[0,-1].legend(fontsize=8)
    fig.savefig(out/'DENSITY_SHAPES.pdf');plt.close(fig)
    e.c.atomic_json(out/'SHAPES.json',dict(rows=rows,edges=edges.tolist(),last_bin='all rho>=4',draw_axis='truth then 0..31',
        phase_median_pair_power_ratio={ph:np.median(v,axis=0).tolist() for ph,v in medratio.items()},
        k_edges=k.tolist(),input=e.c.sha256(prior),passed=True,scope='Supported development core pairs; masked demeaned tapered spectra, not deconvolved survey P(k). Cap draw numbers paired by index for display only.'))


if __name__=='__main__':run()
