"""Small deterministic source-row probe; no reannotation or mock inference."""
import json
from pathlib import Path
import fitsio
import numpy as np

OUT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010/legacy_selection_tests_20261014')
BASE=Path('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit')
def main():
    path=BASE/'CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits'
    with fitsio.FITS(path) as f:
        starts=np.linspace(0,f[1].get_nrows()-4000,128).astype(int)
        d=np.concatenate([f[1].read(rows=np.arange(s,s+4000),columns=['Z','Z_COSMO','RES','CEN','HALO_MASS','FILE_NUM','HALO_INDEX','BOX_INDEX']) for s in starts])
    records=[]
    for lo,hi in [(0,.06),(.06,.10),(.10,.149),(.149,.151),(.151,.2),(.2,.3),(.3,.6)]:
        for res in np.unique(d['RES']):
            k=(d['Z_COSMO']>=lo)&(d['Z_COSMO']<hi)&(d['RES']==res)
            if not k.any():continue
            records.append(dict(zlo=lo,zhi=hi,RES=int(res),rows=int(k.sum()),box_minus_one=int((d['BOX_INDEX'][k]==-1).sum()),file_negative=int((d['FILE_NUM'][k]<0).sum()),halo_index_min=int(d['HALO_INDEX'][k].min()),halo_mass_min=float(d['HALO_MASS'][k].min()),halo_mass_median=float(np.median(d['HALO_MASS'][k])),halo_mass_max=float(d['HALO_MASS'][k].max())))
    result=dict(source=str(path),sample_rows=len(d),sampling='128 evenly spaced contiguous blocks of 4000 rows; descriptive probe, not full census',groups=records,
                cubic_box_versions={v:sorted(p.name for p in (BASE/'CubicBox/BGS'/v).iterdir()) for v in ['v0','v0.1','v1']},
                public_ph000_halo_snapshots=sorted(p.name for p in Path('/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos').iterdir()))
    # A bounded halo-mass fingerprint tests whether HALO_INDEX means the same
    # parent-halo row at competing epochs; high-z rows are the positive control.
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    vals,counts=np.unique(d['FILE_NUM'][(d['Z']<.15)&(d['RES']==1)],return_counts=True)
    fn=int(vals[counts.argmax()]); sample=d[(d['FILE_NUM']==fn)&(d['RES']==1)]
    result['halo_index_mass_probe']=[]
    for epoch in ['0.100','0.200']:
        hp=Path('/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos')/('z'+epoch)/'halo_info'/f'halo_info_{fn:03d}.asdf'
        cat=CompaSOHaloCatalog(str(hp),fields=['N'],subsamples=False,cleaned=False,verbose=False)
        pm=float(cat.header['ParticleMassHMsun']); index=sample['HALO_INDEX'];ok=(index>=0)&(index<len(cat.halos))
        masses=np.full(len(sample),np.nan);masses[ok]=np.asarray(cat.halos['N'])[index[ok]]*pm/1e10
        for lo,hi in [(0,.15),(.2,.3)]:
            k=ok&(sample['Z']>=lo)&(sample['Z']<hi)
            ratio=masses[k]/sample['HALO_MASS'][k]
            result['halo_index_mass_probe'].append(dict(epoch=epoch,file_num=fn,zlo=lo,zhi=hi,rows=int(k.sum()),fraction_within_one_percent=float(np.mean(abs(ratio-1)<.01)),median_mass_ratio=float(np.median(ratio))))
    (OUT/'lowz_product_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
