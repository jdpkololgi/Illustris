from pathlib import Path
import json,hashlib,datetime
repo=Path('/global/homes/d/dkololgi/TNG/Illustris')
p12=Path('/pscratch/sd/d/dkololgi/abacus/p12a_r7_20phase_20261027_v1')
cfm=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20phase_20261027_v1')
def record(p):
 if not p.is_file():return {'exists':False,'path':str(p)}
 data=p.read_bytes();r={'exists':True,'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 if p.suffix=='.json':
  v=json.loads(data)
  for k in ['phase','role','pass','gates','rows','output','join','qa','field_gate','completed_components','verification_scope']:
   if k in v:r[k]=v[k]
 return r
r={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'60355333','status':'metadata_snapshot','scope':'Metadata receipt inventory; not a fresh full-payload revalidation','p12':{},'cfm':{},'old_p12_staging':record(p12/'EXISTING_INPUTS_STAGED.json')}
for ph in ['ph001','ph014','ph015','ph016','ph017']:
 root=p12/ph
 paths={'tweb':root/'targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json','qualification':root/'qualification.json','qualification_audit':root/'qualification_audit/report.json','parent':root/f'catalogues/{ph}_bright_parent_metadata_v2.fits.complete.json','annotation':root/f'catalogues/{ph}_bright_parent_galaxy_realspace_v2_tweb.fits.manifest.json','observed':root/f'catalogues/{ph}_observed_galaxy_realspace_v2_tweb.fits.complete.json','p1':root/'p1_canonical/CATALOGUE_COMPLETE.json','p2':root/'p2_graph/P2_COMPLETE.json','p2_reuse':root/'P2_GEOMETRY_REUSED.json','p3':root/'p3_fields/FIELD_COMPLETE','p4':root/'p4_patches/PATCH_MANIFEST_COMPLETE','phase_complete':root/'PHASE_COMPLETE.json'}
 r['p12'][ph]={k:record(v) for k,v in paths.items()}
r['ph017_boundary_diagnosis']=record(p12/'ph017/qualification_audit/boundary_diagnosis.json')
for ph in ['ph001','ph004','ph005','ph006']:
 paths={'native':cfm/f'matter/{ph}/DENSITY_COMPLETE.json','parent':cfm/f'observations/{ph}/PARENT_COMPLETE.json','observed':cfm/f'observations/{ph}/OBSERVED_COMPLETE.json','angular':cfm/f'observations/{ph}/angular/ANGULAR_COMPLETE.json','NGC':cfm/f'cartesian_v2/observations/{ph}/NGC_COMPLETE.json','SGC':cfm/f'cartesian_v2/observations/{ph}/SGC_COMPLETE.json','geometry':cfm/f'cartesian_v2/geometry/{ph}/GEOMETRY_COMPLETE.json','density_targets':cfm/f'cartesian_v2/targets/{ph}/DENSITY_STAGE.json','targets_complete':cfm/f'cartesian_v2/targets/{ph}/TARGETS_COMPLETE.json'}
 paths.update({f'tensor_{i}':cfm/f'cartesian_v2/targets/{ph}/TENSOR_{i}_STAGE.json' for i in range(6)})
 r['cfm'][ph]={k:record(v) for k,v in paths.items()}
r['tests']={'shared_phase_contract_source_tests':10,'annotation_tests':15,'passed':True,'job_step':'60355333.36'}
r['restart_approval']={'steps':['60355333.4','60355333.6'],'requested':True,'approved':True,'cancelled':False,'reason':'User explicitly approved these restarts at 06:34 UTC; insufficient allocation time remained, so no restart or cancellation was issued.'}
(repo/'docs/shared_phase_preparation_60355333.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'p12_complete':[p for p,v in r['p12'].items() if v['phase_complete']['exists']],'p12_p1':[p for p,v in r['p12'].items() if v['p1']['exists']],'cfm_geometry':[p for p,v in r['cfm'].items() if v['geometry']['exists']],'cfm_targets_complete':[p for p,v in r['cfm'].items() if v['targets_complete']['exists']]}))
