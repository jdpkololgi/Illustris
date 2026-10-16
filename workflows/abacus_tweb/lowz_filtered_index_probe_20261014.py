"""Test the public generator's filtered-index hypothesis on one slab only."""
import json
from pathlib import Path
import fitsio
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

OUT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010/legacy_selection_tests_20261014')
SRC='/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits'
HP='/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.200/halo_info/halo_info_001.asdf'

def main():
    with fitsio.FITS(SRC) as f:
        starts=np.linspace(0,f[1].get_nrows()-4000,128).astype(int)
        d=np.concatenate([f[1].read(rows=np.arange(s,s+4000),columns=['Z','RES','CEN','HALO_MASS','FILE_NUM','HALO_INDEX']) for s in starts])
    d=d[(d['FILE_NUM']==1)&(d['RES']==1)&(d['Z']<.15)]
    records=[]
    for clean in [False,True]:
        try:
            c=CompaSOHaloCatalog(HP,fields=['N','x_L2com'],cleaned=clean,subsamples=False,verbose=False)
        except Exception as e:
            records.append(dict(cleaned=clean,error=str(e)));continue
        mass=np.asarray(c.halos['N'],dtype=float)*float(c.header['ParticleMassHMsun'])
        pos=np.array(c.halos['x_L2com'],dtype=float)+1000
        pos[pos>1000]-=2000
        # The retrieved template is c004, whereas this source is c000.
        # Use this actual source header cosmology; no parameter fitting to matches.
        om=float(c.header['Omega_M']); h=float(c.header['H0'])/100
        radius=FlatLambdaCDM(H0=h*100,Om0=om).comoving_distance(.16).value*h
        keep=(mass>=1e11)&(np.linalg.norm(pos,axis=1)<radius)
        m=mass[keep]/1e10;idx=d['HALO_INDEX'];ok=(idx>=0)&(idx<len(m))
        ratio=m[idx[ok]]/d['HALO_MASS'][ok]
        records.append(dict(cleaned=clean,halo_rows=len(mass),filtered_rows=len(m),sample_rows=len(d),inbounds=int(ok.sum()),radius_hmpc=radius,Omega_M=om,h=h,fraction_mass_within_1pct=float(np.mean(abs(ratio-1)<.01)),median_ratio=float(np.median(ratio))))
    r=dict(source=SRC,halo_source=HP,records=records,scope='One slab and sampled resolved low-z rows. Mass/radius cuts reconstructed from public template, not certified production mapping. No catalogue modifications.')
    (OUT/'lowz_filtered_index_probe.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
