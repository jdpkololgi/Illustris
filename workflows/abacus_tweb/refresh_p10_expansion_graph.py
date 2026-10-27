"""Publish a local AST-only refresh after checking for concurrent graph edits."""
from pathlib import Path
import hashlib,json,os,shutil,tempfile
root=Path('/global/u2/d/dkololgi/TNG/Illustris');original=root/'graphify-out';stage=Path(tempfile.mkdtemp(prefix='bgs-plots-graph-'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
initial=sha(original/'graph.json')
for name in ['graph.json','.graphify_labels.json']:
 if (original/name).exists():shutil.copy2(original/name,stage/name)
os.environ['GRAPHIFY_OUT']=str(stage)
from graphify.watch import _rebuild_code
names=['p10_phase_assets.py','p10_run_tweb.py','p10_expansion_preflight.py','validate_galaxy_position_mapping.py','annotate_cutsky_with_tweb_eigs.py','refresh_p10_expansion_graph.py','galaxy_label_equivalence.py','certify_galaxy_position_scope.py','galaxy_position_validation_status.py','p10_build_observed_truth.py','smoke_galaxy_position_annotation.py','run_galaxy_position_tests.py','diagnose_ph007_particle_ambiguity.py']
names += ['p4_build_spatial_manifest.py']
names += ['p10_materialize_phase_schemas.py','p3a_catalogue_field_closure.py','p10_prepare_training_contract.py','p10_training_contract.py','p10_validate_phase_products.py','adopt_corrected_p1_20261021.py','reuse_corrected_p2_20261021.py','adopt_p10_existing_inputs_20261021.py']
if not _rebuild_code(root,changed_paths=[root/'workflows/abacus_tweb'/n for n in names]+[root/'tests/test_p10_phase_expansion.py',root/'tests/phase4/test_p4_spatial_utils.py']):raise RuntimeError('Staged AST refresh failed')
if sha(original/'graph.json')!=initial:raise RuntimeError('Concurrent index modification; refusing publication')
for name in ['graph.json','GRAPH_REPORT.md','.graphify_labels.json','graph.html']:
 if (stage/name).exists():
  tmp=original/(name+'.bgs-plots-new');shutil.copy2(stage/name,tmp);tmp.replace(original/name)
(root/'outputs/p10_expansion_pilot_20261020/graph_refresh.json').write_text(json.dumps({'success':True,'mode':'GRAPHIFY_OUT local staging with normal flock; original graph hash checked before publication','before_sha256':initial,'after_sha256':sha(original/'graph.json'),'changed_files':names},indent=2)+'\n')
print('GRAPH_REFRESH_COMPLETE',flush=True)
