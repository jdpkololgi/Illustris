"""Compare row-level response construction against saved registered 3-D grids."""
import json
from pathlib import Path
import numpy as np
from workflows.abacus_tweb import p3br_build_random_response as b
from workflows.sbi.p12a_parent_response import sample_centres
from workflows.sbi.p12_prepare_base_response_dataset import sample_random_support_distance
root=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')
manifest=json.loads((root/'ph000/p3b_random_response_v1/manifest.json').read_text())
selection=json.loads(Path(manifest['selection_manifest']).read_text())
angular=b.load_map(Path(manifest['angular_map']))
points=np.load(manifest['points'],mmap_mode='r')
records=[]
for cap,name in b.CAPS:
    component=manifest['components'][name];spec=b.grid_from_component(component)
    ids=np.flatnonzero(points[:,3]==cap)
    ix=np.floor((np.asarray(points[ids,0],dtype=np.float64)-spec.origin[0])/spec.cell_mpc).astype(np.int64)
    valid=(ix>=0)&(ix<spec.shape[0]);ids=ids[valid];ix=ix[valid]
    plane=ix[len(ix)//2];chosen=ids[ix==plane]
    chosen=chosen[np.linspace(0,len(chosen)-1,min(32,len(chosen)),dtype=np.int64)]
    support=np.asarray(angular['support'],dtype=bool)&((angular['domain']//2)==cap)
    angle=b.angular_boundary_distance(support)
    actual,actual_support=sample_centres(points[chosen],component,support,angle,selection)
    expected,expected_support=sample_random_support_distance(manifest,points,chosen)
    np.testing.assert_array_equal(actual_support,expected_support)
    np.testing.assert_array_equal(actual,expected)
    records.append({'cap':name,'rows':len(chosen),'grid_x_index':int(plane),'exact_value_and_support_parity':True})
report={'pass':True,'reference_manifest':manifest['p3a_manifest'],'panels':records,
        'scope':'exact 64-row existing-grid parity plus synthetic whole-grid regression; no calibration claim'}
path=Path('outputs/field_restart_repair_20261026/PARENT_RESPONSE_PARITY.json');path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
