import importlib.util,json,sys
from pathlib import Path
import numpy as np
repo=Path('/global/homes/d/dkololgi/TNG/Illustris')
script=repo/'outputs/p10_unattended_20261021/source/workflows/abacus_tweb/p4_build_spatial_manifest.py'
spec=importlib.util.spec_from_file_location('p4_diagnostic',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
root=Path('/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph012')
args={'schema':root/'contracts/p4_schema_v1.json','probe':root/'p4_patches/core_size_probe.json','points':root/'p1_canonical/points.npy','index':root/'p1_canonical/canonical_index.npz','catalogue':json.loads((root/'p1_canonical/manifest.json').read_text())['parent'],'p1-manifest':root/'p1_canonical/manifest.json','p2-manifest':root/'p2_union/p2b_union_manifest.json','p3-manifest':root/'p3_fields/field_manifest.json','out-dir':root/'p4_recovery_diagnostic_20261026'}
sys.argv=[str(script)]+[s for k,v in args.items() for s in ('--'+k,str(v))]
try:mod.main()
except RuntimeError as exc:
 if 'no deterministic fold candidate passed' not in str(exc):raise
 tb=exc.__traceback__
 while tb.tb_next:tb=tb.tb_next
 loc=tb.tb_frame.f_locals
 keys=['component_counts','component_context','component_cores','super_component']
 dest=root/'p4_recovery_diagnostic_20261026/fold_inputs.npz'
 np.savez_compressed(dest,**{k:loc[k] for k in keys})
 print('DIAGNOSTIC_SAVED',dest,{k:loc[k].shape for k in keys},flush=True)
 print(loc['fold_candidate_audit'],flush=True)
