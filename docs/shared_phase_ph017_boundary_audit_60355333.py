"""Diagnose the failed ph017 native-group voxel gate; do not issue qualification."""
import json,sys,os
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'workflows/abacus_tweb'))
from galaxy_position_mapping import field_indices,periodic_delta
from tweb_grid_sampling import TWebGrid
if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Compute step required')
root=Path('/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1/ph017')
with np.load(root/'qualification_audit/ph017_sample.npz') as f:
 data={k:f[k] for k in f.files}
mask=(data['reference_kind']=='unresolved_central')&(data['FILE_NUM']==33)
a=data['reconstructed_native'][mask];b=data['independent_native'][mask]
ia,ib=field_indices(a),field_indices(b);bad=np.any(ia!=ib,axis=1)
grid=TWebGrid(root/'targets/tweb/backend_optimized_ngrid_2048_rsmooth_7')
x,_=grid._sample(ia[bad]);y,_=grid._sample(ib[bad]);h=2000/2048
records=[]
for j,i in enumerate(np.flatnonzero(bad)):
 d=periodic_delta(a[i],b[i]);axis=np.flatnonzero(ia[i]!=ib[i])
 rec=dict(source_row=int(data['original_row'][mask][i]),native_source_row=int(data['native_source_row'][mask][i]),reconstructed=a[i].tolist(),independent=b[i].tolist(),residual_mpc_h=float(np.linalg.norm(d)),coordinate_delta_mpc_h=d.tolist(),reconstructed_cell=ia[i].tolist(),independent_cell=ib[i].tolist(),changed_axes=axis.tolist(),periodic_cell_difference=((ia[i]-ib[i]+1024)%2048-1024).tolist(),distance_to_nearest_boundary_mpc_h=(np.minimum(np.mod(a[i],h),h-np.mod(a[i],h))).tolist(),eigenvalue_difference=(x[j]-y[j]).tolist())
 records.append(rec)
r=dict(phase='ph017',group_rows=int(mask.sum()),different_voxel_rows=int(bad.sum()),records=records,original_gate_unchanged=True,qualification_issued=False)
p=root/'qualification_audit/boundary_diagnosis.json';p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
