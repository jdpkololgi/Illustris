"""Verify and link completed corrected catalogues into the expanded campaign."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from workflows.abacus_tweb.p10_phase_assets import sha256_file
from workflows.abacus_tweb.galaxy_truth_contract import verify_observed_truth

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);args=ap.parse_args()
    phase=args.phase
    if phase not in ('ph000','ph002','ph003','ph004','ph005','ph006'):
        raise ValueError('Only existing corrected development catalogues are eligible')
    base=Path('/pscratch/sd/d/dkololgi/abacus')
    source=(base/'galaxy_truth_benchmark_20261020/job60038639/replacement' if phase=='ph002' else base/'galaxy_truth_remaining_20261020')/phase
    p1=source/'p1_canonical';m=json.loads((p1/'manifest.json').read_text());c=json.loads((p1/'CATALOGUE_COMPLETE.json').read_text())
    if m['phase']!=phase or not m['target_truth_present']:raise ValueError('P1 identity mismatch')
    for name,key in [('canonical_index.npz','canonical_index_sha256'),('points.npy','points_sha256')]:
        if sha256_file(p1/name)!=c['artifacts'][key]:raise ValueError('P1 payload mismatch')
    lineage=verify_observed_truth(Path(m['parent']),phase)
    if lineage!=m['truth_lineage']:raise ValueError('P1 truth lineage mismatch')
    dest=base/'p10_galaxy_v2_20261020'/phase;dest.mkdir(parents=True,exist_ok=True)
    for name in ('p1_canonical','catalogues'):
        link=dest/name
        if link.exists() or link.is_symlink():
            if link.resolve()!=(source/name).resolve():raise FileExistsError(link)
        else:link.symlink_to(source/name,target_is_directory=True)
    receipt={'phase':phase,'source':str(source),'p1_manifest_sha256':sha256_file(p1/'manifest.json'),'observed_sha256':m['parent_sha256'],'pass':True}
    (dest/'CORRECTED_P1_ADOPTED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
