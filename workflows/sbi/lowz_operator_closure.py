"""Scoring-only oracle-density control. Never an input to posterior generation.

Use the production campaign config and import path as for lowz_finish_replay.
CPU prepare once per development phase; GPU tides separately approved.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import fft
import lowz_ph006_cap_draw as pilot
from workflows.sbi import e2e_coupled_operators as op


DEV=('ph006','ph012','ph013')
read=lambda p: json.loads(p.read_text())
c=pilot.c


def prepare(phase, root):
    c.require_compute()
    if phase not in DEV or tuple(c.DEVELOPMENT)!=DEV: raise ValueError('Not development')
    pilot.phase=phase
    ref=c.ROOT.parent/'lowz_calibration_61647276/CALIBRATION.json'
    if c.sha256(ref)!='72a63078c8728b35f5564c95c3af7871c33171333038cd5d13cbb10c5987bbae':
        raise ValueError('Unexpected score and path registry')
    paths=read(ref)['paths']
    source=c.ROOT/'matter'/phase/'DENSITY_COMPLETE.json'
    native=c.verify_receipt(source)
    assert native['phase']==phase and native['ngrid']==2048
    freeze=pilot.freeze()
    spectrum,_=op.smoothed_spectrum(native['outputs'][0]['path'],32)
    field=fft.irfftn(spectrum,s=(2048,)*3,workers=32,overwrite_x=True)
    del spectrum
    for cap in ('NGC','SGC'):
        dst=root/phase/cap/'oracle_input'; dst.mkdir(parents=True,exist_ok=False)
        rows=pilot.rows(cap); grid=rows[0]['grid']
        rec=Path(paths[str((phase,cap,0))])/'density/COMPLETE.json'
        fixed=read(rec)
        assert fixed['passed'] and (fixed['phase'],fixed['cap'],fixed['first'],fixed['draws'])==(phase,cap,0,1)
        item=fixed['field']; assert c.sha256(item['path'])==item['sha256']
        dstfile=dst/'FIELD.h5'; errors={}
        with h5py.File(item['path']) as src,h5py.File(dstfile,'x') as dest:
            origin=np.array(src.attrs['origin_mpc_h'])
            shape=src['rho'].shape[1:]; dx=float(grid['cell_mpc_h'])
            assert dx==3.383 and float(src.attrs['cell_mpc_h'])==2*dx
            start=(origin-np.array(grid['origin_mpc_h']))/dx
            np.testing.assert_allclose(start,np.rint(start),rtol=0,atol=1e-8)
            starts=[int(round(a))+2*np.arange(n) for a,n in zip(start,shape)]
            rho=1+op.sample_averaged_local(field,starts,2,grid)
            finite=np.isfinite(src['rho'][0])
            np.testing.assert_array_equal(finite,src['coverage'][:]>0)
            if not (np.isfinite(rho[finite]).all() and (rho[finite]>0).all()):
                raise ValueError('Invalid R7 density on modelled cells')
            rho[~finite]=np.nan
            for row in rows:
                pair=row['pair_id']; r=read(pilot.coord.ROOT/'targets'/phase/(pair+'.json'))
                assert r['binding']['native_receipt_sha256']==c.sha256(source)
                tr=pilot.read_target(pilot.checked(r),phase,pair)
                sl=pilot.cuts(row,origin); supported=src['support'][sl]>0
                expected=pilot.owned_core_arrays(tr['rho_joint']).reshape(32,16,16)
                np.testing.assert_allclose(rho[sl][supported],expected[supported],rtol=0,atol=2e-6)
                errors[pair]=float(abs(rho[sl][supported]-expected[supported]).max())
            src.copy('support',dest);src.copy('coverage',dest)
            dest.create_dataset('rho',data=rho[None].astype('f4'),compression='lzf')
            for k in ('origin_mpc_h','phase','cap','cell_mpc_h','first_draw'):dest.attrs[k]=src.attrs[k]
        c.atomic_json(dst/'COMPLETE.json',dict(phase=phase,cap=cap,first=0,draws=1,passed=True,
            field=c.file_record(dstfile,content_hash=True),native_sha256=c.sha256(source),
            production_source=c.file_record(rec,content_hash=True),prior_sha256=freeze['prior_sha256'],
            script_sha256=c.sha256(__file__),matched_core_errors=errors,
            scope='Oracle density only where production supplied finite cells; no true exterior. Diagnostic only.'))


def score(phase,cap,root):
    """Decompose *signed* per-cell error. Single paired exterior seed, not closure of its distribution."""
    c.require_compute()
    if phase not in DEV or cap not in ('NGC','SGC'): raise ValueError('Not development')
    pilot.phase=phase
    ref=c.ROOT.parent/'lowz_calibration_61647276/CALIBRATION.json'
    assert c.sha256(ref)=='72a63078c8728b35f5564c95c3af7871c33171333038cd5d13cbb10c5987bbae'
    paths=read(ref)['paths']
    r=root/phase/cap; matched=read(r/'matched_tides/COMPLETE.json')
    assert (matched['passed'],matched['phase'],matched['cap'],matched['first'],matched['draws'])==(True,phase,cap,0,1)
    binding_path=r/'matched_tides/BINDING.json'
    assert matched['binding']==c.sha256(binding_path)
    binding=read(binding_path)
    assert (matched['method'],binding['method'],binding['padding'])==('learned','learned',32)
    assert (binding['phase'],binding['cap'])==(phase,cap)
    assert binding['prior']['sha256']==pilot.freeze()['prior_sha256']
    assert binding['input_receipt']['sha256']==c.sha256(r/'oracle_input/COMPLETE.json')
    item=matched['output'];assert c.sha256(item['path'])==item['sha256']
    oracle=read(r/'oracle_input/COMPLETE.json')['field']
    assert c.sha256(oracle['path'])==oracle['sha256']
    rows=[]
    with h5py.File(item['path']) as t,h5py.File(oracle['path']) as f:
        origin=f.attrs['origin_mpc_h']
        for row in pilot.rows(cap):
            pair=row['pair_id'];sl=pilot.cuts(row,origin)
            tr=pilot.read_target(pilot.checked(read(pilot.coord.ROOT/'targets'/phase/(pair+'.json'))),phase,pair)
            full=pilot.tidal.eigensystem(tr['fullbox_tensor_cores']).reshape(32,16,16,3)
            v=t['eigenvalues'][(0,*sl)]
            xyz=np.meshgrid(*[(np.arange(s.start,s.stop)+.5)*6.766+o for s,o in zip(sl,origin)],indexing='ij')
            z=pilot.coord.redshift(np.sqrt(sum(x*x for x in xyz))); sup=f['support'][sl]>0
            sampled=[]
            for i in range(32):
                d=Path(paths[str((phase,cap,i))])/'tides';rec=read(d/'COMPLETE.json')
                assert (rec['passed'],rec['phase'],rec['cap'],rec['first'])==(True,phase,cap,i)
                with h5py.File(rec['output']['path']) as g:sampled.append(g['eigenvalues'][(0,*sl)])
            x=np.array(sampled,dtype='f8');mu=x.mean(0);sd=x.std(0,ddof=1)
            for lo,hi in zip([.01,.15,.25,.35,.45],[.15,.25,.35,.45,.55]):
                m=sup&(z>=lo)&(z<hi)
                if not m.any():continue
                if not (np.isfinite(mu[m]).all() and np.isfinite(sd[m]).all() and (sd[m]>0).all()
                        and np.isfinite(v[m]).all() and np.isfinite(full[m]).all()):raise ValueError('Invalid closure')
                gap=v[m]-full[m]; remaining=mu[m]-v[m]; total=mu[m]-full[m]
                np.testing.assert_allclose(total,remaining+gap,atol=1e-10,rtol=0)
                parts={k:dict(mean=e.mean(0).tolist(),mean_sd_units=(e/sd[m]).mean(0).tolist())
                       for k,e in [('original',total),('operator_plus_exterior',gap),('remaining',remaining)]}
                rows.append(dict(pair=pair,z=[lo,hi],cells=int(m.sum()),stratum=row['support_stratum'],parts=parts))
    c.atomic_json(r/'CLOSURE_SCORE.json',dict(phase=phase,cap=cap,rows=rows,matched_output=item,
        scope='Conditional diagnostic: one oracle exterior draw, same seed as production draw zero; not pure tensor discretization error or proof of a correctable bias.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=DEV); p.add_argument('stage',choices=('prepare','tides','score'))
    p.add_argument('--cap',choices=('NGC','SGC'))
    a=p.parse_args();root=c.ROOT.parent/'lowz_dev_oracle_density_v1'
    if a.stage=='prepare':
        if a.cap:p.error('CPU preparation shares one native FFT for both caps')
        prepare(a.phase,root)
    elif a.stage=='score':
        if not a.cap:p.error('Select one cap for scoring')
        score(a.phase,a.cap,root)
    else:
        if not a.cap:p.error('Select one cap for tides')
        r=root/a.phase/a.cap; src=r/'oracle_input/COMPLETE.json'
        meta=read(src)
        assert (meta['phase'],meta['cap'],meta['prior_sha256'])==(a.phase,a.cap,pilot.freeze()['prior_sha256'])
        pilot.run_tides(src,pilot.prior/'PRIOR.json',r/'matched_tides','learned',32)


if __name__=='__main__':main()
