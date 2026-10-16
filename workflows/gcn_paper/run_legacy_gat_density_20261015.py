"""Sequential one-thread execution, explicitly authorized login fallback."""
import os
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parent
out=Path('/global/u2/d/dkololgi/fmsu_tokyo_20261010/legacy_gat_density_20261015')
env=os.environ.copy()
for key in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD']:env.pop(key,None)
env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
base=Path('/pscratch/sd/d/dkololgi/conda/envs')
stages=[('topology','thin'),('metrics','thin'),('train','thin','--radius','2'),
        ('topology','dense'),('metrics','dense'),('topology','loa'),('metrics','loa'),
        ('train','dense','--radius','2'),('train','dense','--radius','7'),('train','thin','--radius','7'),
        ('infer','dense','--radius','2'),('infer','dense','--radius','7'),
        ('infer','thin','--radius','2'),('infer','thin','--radius','7'),('verify','dense')]
out.mkdir(exist_ok=True)
for args in stages:
    py=base/('rapids-gnn' if args[0]=='metrics' else 'cosmic_env')/'bin/python'
    log=out/('_'.join(args).replace('--','')+'.log')
    print('START',args,flush=True)
    with log.open('w') as f:
        result=subprocess.run([str(py),str(root/'legacy_gat_density_20261015.py'),*args],env=env,stdout=f,stderr=subprocess.STDOUT)
    print('FINISH',args,result.returncode,flush=True)
    if result.returncode:
        print(log.read_text()[-5000:],flush=True);sys.exit(result.returncode)
