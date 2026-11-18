import time,json,sys
sys.path.insert(0,'/global/homes/d/dkololgi/TNG/Illustris/workflows/abacus_tweb')
import fitsio,numpy as np
from annotate_cutsky_with_tweb_eigs import verify_source_rows,TRUTH_COLUMNS
root='/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1/ph001'
source='/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph001.fits'
with fitsio.FITS(root+'/catalogues/ph001_bright_parent_metadata_v2.fits') as f,fitsio.FITS(source) as s:
 c=f[1][2000000:2010000];ids=c['SOURCE_ROW'];t=time.monotonic();verify_source_rows(c,s[1],ids);print('rows_seconds',time.monotonic()-t,flush=True)
 t=time.monotonic();lo,hi=int(ids.min()),int(ids.max())+1;ref=s[1][lo:hi][ids-lo];print('span_seconds',time.monotonic()-t,'span',hi-lo,'exact',all(np.array_equal(c[k],ref[k]) for k in TRUTH_COLUMNS),flush=True)
from annotate_cutsky_with_tweb_eigs import annotate_chunk
from tweb_grid_sampling import TWebGrid
t=time.monotonic();g=TWebGrid(root+'/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7');print('grid_init',time.monotonic()-t,flush=True)
t=time.monotonic();out=annotate_chunk(c,ids,g);print('annotation_seconds',time.monotonic()-t,'rows',len(c),flush=True)
t=time.monotonic()
for lo,hi,p in g.slabs:
 eig,cls=g._arrays(lo,hi,p);g.cache[p]=(np.array(eig),np.array(cls))
print('resident_load_seconds',time.monotonic()-t,flush=True)
t=time.monotonic();other=annotate_chunk(c,ids,g);print('resident_annotation_seconds',time.monotonic()-t,'exact',np.array_equal(other,out),flush=True)
