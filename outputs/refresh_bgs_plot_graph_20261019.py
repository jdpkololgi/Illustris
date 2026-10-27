from pathlib import Path
import hashlib,json,os,shutil,tempfile
root=Path('/global/u2/d/dkololgi/TNG/Illustris');original=root/'graphify-out';stage=Path(tempfile.mkdtemp(prefix='bgs-plots-graph-'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
initial=sha(original/'graph.json')
for name in ['graph.json','.graphify_labels.json']:
 if (original/name).exists():shutil.copy2(original/name,stage/name)
os.environ['GRAPHIFY_OUT']=str(stage)
from graphify.watch import _rebuild_code
names=['plot_bgs_field_alignment.py','plot_bgs_galaxy_density_slices.py','inspect_bgs_smoothing_scales.py','plot_bgs_scale_support_redshift.py']
if not _rebuild_code(root,changed_paths=[root/'workflows/abacus_tweb'/n for n in names]):raise RuntimeError('Staged AST refresh failed')
if sha(original/'graph.json')!=initial:raise RuntimeError('Concurrent index modification; refusing publication')
for name in ['graph.json','GRAPH_REPORT.md','.graphify_labels.json','graph.html']:
 if (stage/name).exists():
  tmp=original/(name+'.bgs-plots-new');shutil.copy2(stage/name,tmp);tmp.replace(original/name)
(root/'outputs/bgs_scale_support_redshift_20261019/graph_refresh.json').write_text(json.dumps({'success':True,'mode':'GRAPHIFY_OUT local staging with normal flock; original graph hash checked before publication','before_sha256':initial,'after_sha256':sha(original/'graph.json'),'changed_files':names},indent=2)+'\n')
print('GRAPH_REFRESH_COMPLETE',flush=True)
