"""Aggregate six matched oracle-density controls; diagnose shear without tuning."""
import os
import h5py
import numpy as np
import lowz_operator_closure as o


def run():
    o.c.require_compute()
    root=o.c.ROOT.parent/'lowz_dev_oracle_density_v1'
    card=o.c.ROOT.parent/'lowz_finish_scorecard_61848915/SCORECARD.json'
    sc=o.read(card)
    assert sc['passed'];reg={r['pair']:r for r in sc['regions']}
    cells=[];cores=[]; receipts={}
    for ph in o.DEV:
        o.pilot.phase=ph
        for cap in ('NGC','SGC'):
            d=root/ph/cap
            rec=d/'CLOSURE_SCORE.json';rr=o.read(rec)
            assert (rr['phase'],rr['cap'])==(ph,cap)
            receipts[ph+'_'+cap]=o.c.sha256(rec)
            cells.extend(dict(phase=ph,cap=cap,**r) for r in rr['rows'])
            oracle=o.read(d/'oracle_input/COMPLETE.json')
            with h5py.File(rr['matched_output']['path']) as t,h5py.File(oracle['field']['path']) as f:
                for row in o.pilot.rows(cap):
                    pair=row['pair_id'];sl=o.pilot.cuts(row,f.attrs['origin_mpc_h'])
                    v=t['eigenvalues'][(0,*sl)].astype('f8').reshape(2,4096,3).mean(1)
                    matched=(v[:,2]-v[:,0])/np.sqrt(2)
                    r=reg[pair];y=np.array(r['truth'])[[1,4]]
                    x=np.array(r['draws'])[:,[1,4]];sd=x.std(0,ddof=1)
                    assert np.isfinite(matched).all() and (sd>0).all()
                    cores.append(dict(phase=ph,cap=cap,pair=pair,original=((x.mean(0)-y)/sd).tolist(),
                        operator=((matched-y)/sd).tolist(),remaining=((x.mean(0)-matched)/sd).tolist()))
    assert len(cores)==48
    sums=[]
    for ph in [*o.DEV,'all']:
        cs=[r for r in cells if ph=='all' or r['phase']==ph];rs=[r for r in cores if ph=='all' or r['phase']==ph]
        sums.append(dict(phase=ph,cells=sum(r['cells'] for r in cs),tidal_bias_sd={
            k:np.average([r['parts'][k]['mean_sd_units'] for r in cs],axis=0,weights=[r['cells'] for r in cs]).tolist()
            for k in ['original','operator_plus_exterior','remaining']},
            regional_shear_bias_sd={k:float(np.mean([r[k] for r in rs])) for k in ['original','operator','remaining']}))
    assert sums[-1]['cells']==344302
    o.c.atomic_json(root/'CLOSURE_READOUT.json',dict(summary=sums,cores=cores,inputs=receipts,card=o.c.sha256(card),
        passed=True,scope='One oracle exterior seed per cap, production mask. Signed additive diagnostic, not bias correction or a variance decomposition.'))


if __name__=='__main__':run()
