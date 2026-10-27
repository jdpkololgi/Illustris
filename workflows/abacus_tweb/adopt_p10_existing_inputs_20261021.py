"""Adopt unchanged ph001 blind inputs and the qualified ph006 target directory."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from workflows.abacus_tweb.p10_phase_assets import sha256_file
base=Path('/pscratch/sd/d/dkololgi/abacus')
old=base/'p10_multiphase';new=base/'p10_galaxy_v2_20261020'
source=old/'ph001';dest=new/'ph001'
marker=source/'BLIND_INPUT_COMPLETE.json';receipt=json.loads(marker.read_text())
p1=json.loads((source/'p1_canonical/manifest.json').read_text())
if not receipt['pass'] or receipt['phase']!='ph001' or p1['target_truth_present']:
    raise ValueError('Existing blind input contract is invalid')
for name,path in receipt['artifacts'].items():
    if sha256_file(Path(path))!=receipt['artifact_sha256'][name]:raise ValueError('Stale blind artifact: '+name)
def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.resolve()!=src.resolve():raise FileExistsError(dst)
    else:dst.symlink_to(src,target_is_directory=True)
for name in ('catalogues','p1_canonical','p2_graph','p2_union','p3_fields','p4_patches','contracts'):
    link(source/name,dest/name)
out=dest/'BLIND_INPUT_COMPLETE.json'
if out.exists():
    if out.read_bytes()!=marker.read_bytes():raise FileExistsError(out)
else:out.write_bytes(marker.read_bytes())
link(old/'ph006/targets',new/'ph006/targets')
print('ph001 blind inputs and ph006 targets adopted; no reserved truth accessed',flush=True)
