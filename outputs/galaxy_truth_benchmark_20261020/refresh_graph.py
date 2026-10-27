from pathlib import Path
import hashlib,json,os,shutil,tempfile
root=Path('/global/u2/d/dkololgi/TNG/Illustris');original=root/'graphify-out';stage=Path(tempfile.mkdtemp(prefix='bgs-plots-graph-'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
initial=sha(original/'graph.json')
for name in ['graph.json','.graphify_labels.json']:
 if (original/name).exists():shutil.copy2(original/name,stage/name)
os.environ['GRAPHIFY_OUT']=str(stage)
from graphify.watch import _rebuild_code
names=['validate_galaxy_position_mapping.py','galaxy_truth_contract.py','p10_build_phase_index.py','p10_prepare_training_contract.py','build_abacus_sbi_cache.py','audit_galaxy_truth_replacement.py','run_galaxy_position_tests.py','run_galaxy_truth_benchmark.py']
if not _rebuild_code(root,changed_paths=[root/'workflows/abacus_tweb'/n for n in names]+[root/'tests/test_galaxy_truth_consumers.py']):raise RuntimeError('Staged AST refresh failed')
if sha(original/'graph.json')!=initial:raise RuntimeError('Concurrent index modification; refusing publication')
for name in ['graph.json','GRAPH_REPORT.md','.graphify_labels.json','graph.html']:
 if (stage/name).exists():
  tmp=original/(name+'.bgs-plots-new');shutil.copy2(stage/name,tmp);tmp.replace(original/name)
(root/'outputs/galaxy_truth_benchmark_20261020/graph_refresh.json').write_text(json.dumps({'success':True,'mode':'GRAPHIFY_OUT local staging with normal flock; original graph hash checked before publication','before_sha256':initial,'after_sha256':sha(original/'graph.json'),'changed_files':names},indent=2)+'\n')
print('GRAPH_REFRESH_COMPLETE',flush=True)
