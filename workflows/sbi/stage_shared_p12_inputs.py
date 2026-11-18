"""Stage completed P12 inputs with private metadata and immutable payload links.

Verifies historical completion metadata; preserves the original payload receipts.
This is not a fresh full-payload audit. New transforms live outside these inputs.
"""
import json
from pathlib import Path
import shutil
from shared.phase_policy import REPO, bound_policy
from workflows.sbi.e2e_coupled_contract import require_compute, atomic_json
from workflows.sbi.shared_phase_migration import sha256

CONFIG=REPO/'configs/phase_migration_20261027.json'
DIRECTORIES=('p1_canonical','p2_graph','p2_union','p3_fields','p4_patches','contracts')


def stage():
    require_compute(); cfg=json.loads(CONFIG.read_text()); policy=bound_policy(cfg)
    old=Path(cfg['old_p12a_root']); new=Path(cfg['new_p12a_root'])
    results={}
    for phase in policy['training']+policy['development_and_calibration']:
        source=old/phase; proof_path=source/'PHASE_COMPLETE.json'
        if phase in cfg['p12a_new_label_phases']:continue
        proof=json.loads(proof_path.read_text())
        if proof.get('pass') is not True or proof.get('phase')!=phase:
            raise ValueError('Existing phase completion failed')
        for key,path in proof['artifacts'].items():
            if sha256(path)!=proof['artifact_sha256'][key]:
                raise ValueError('Historical phase metadata changed: '+str(path))
        p1=json.loads((source/'p1_canonical/manifest.json').read_text())
        if (p1.get('phase')!=phase or p1.get('target_truth_present') is not True
                or p1.get('truth_lineage',{}).get('label_policy_version')!='galaxy-realspace-all-populations-v2'):
            raise ValueError('Corrected P1 lineage required')
        destination=new/phase
        if destination.exists():raise FileExistsError(destination)
        members={name:list((source/name).iterdir()) for name in DIRECTORIES}
        for name, entries in members.items():
            flat=[]
            for item in entries:
                if not item.is_dir():flat.append(item);continue
                if name!='p2_graph' or item.name!='caps':
                    raise ValueError('Unexpected nested input directory; inspect before staging')
                caps=list(item.iterdir())
                if {cap.name for cap in caps}!={'ngc','sgc'}:
                    raise ValueError('Unexpected graph cap directories')
                for cap in caps:
                    leaves=list(cap.iterdir())
                    if any(leaf.is_dir() for leaf in leaves):
                        raise ValueError('Unexpected nested graph payload directory')
                    flat.extend(leaves)
            members[name]=flat
        destination.mkdir(parents=True)
        for name,entries in members.items():
            folder=destination/name;folder.mkdir()
            for item in entries:
                target=folder/item.relative_to(source/name)
                target.parent.mkdir(parents=True,exist_ok=True)
                if item.suffix in ('.json','.txt') or 'COMPLETE' in item.name:
                    shutil.copy2(item,target)
                else:
                    target.symlink_to(item.resolve())
        shutil.copy2(proof_path,destination/'PHASE_COMPLETE.json')
        record=dict(phase=phase,phase_policy=cfg['phase_policy'],source=str(source),
                    source_completion_sha256=sha256(proof_path),
                    role='training' if phase in policy['training'] else 'development',
                    scope='Historical corrected phase proof and bound metadata verified; payload checks inherited',
                    private_metadata=True,immutable_payload_references=True,**{'pass':True})
        atomic_json(destination/'INPUT_REUSE_COMPLETE.json',record)
        results[phase]=record
        print('Staged corrected inputs',phase,flush=True)
    atomic_json(new/'EXISTING_INPUTS_STAGED.json',dict(phases=results,phase_policy=cfg['phase_policy'],
                training_ready=False,scope='18 existing corrected phases only',**{'pass':len(results)==18}))

if __name__=='__main__':stage()
