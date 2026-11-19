"""One-time, hash-pinned validation schema migration. No updates or model conversion."""
import json
import os
from pathlib import Path
from workflows.abacus_tweb import p10_train_arm_a as arm

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/p12a_lowz_full_seed42_61548250')
ORIGINAL = '6ceb15b449e0751757dfecff082af98becb7093f1c96e90d0aaac56e98bd5231'
CHANGES = {
    'workflows/abacus_tweb/p10_train_arm_a.py': (
        '4db2b52c061544fe76e9eb42f56343ce578c6b2058e2d71364ef824218e86d1e',
        '4c84ffb96819b08a145eb2337fb74fc6889af2a46f3dade67dc5d1f83f7925fc'),
    'workflows/abacus_tweb/p8_deterministic_common.py': (
        'c7636ce698b69041bcf5d090a709780eb68ad267ea8ad57948f076d94c589cd2',
        '38c0efd10c2d9fa8be7d13d122a4e569707eed78513faa21cc4fa96f33fb264a')}


def migrate():
    dest = ROOT/'full/unet/seed_42'
    with arm.acquire_run_lock(dest/'.run.lock', purpose='epoch 1 validation-schema repair'):
        checkpoint = dest/'arm_a_checkpoint.pt'
        if arm.sha256(checkpoint) != ORIGINAL:
            raise RuntimeError('Unexpected starting checkpoint')
        state = arm.torch_load(checkpoint, 'cpu')
        old = state['source_contract']
        new = arm.source_contract()
        if old.keys() != new.keys() or {k:(old[k],new[k]) for k in old if old[k]!=new[k]} != CHANGES:
            raise RuntimeError('Source change outside reviewed validation repair')
        manifest = json.loads((dest/'run_manifest.json').read_text())
        if (state['global_step'] != 346364 or state['resume']['epoch'] != 1
            or state['resume']['cursor'] != manifest['training_cores_per_epoch']
            or state['resume']['cursor'] != 346364 or state['history']
            or manifest['validation_and_selection_phases'] != ['ph006','ph012','ph013']
            or state['frozen_arguments'] != manifest['frozen_arguments']
            or (dest/'epoch_history.jsonl').stat().st_size != 0):
            raise RuntimeError('Not the unvalidated end of epoch 1')
        paths = {k: Path(manifest[k]) for k in ('target_scaler','training_ready_marker')}
        paths['field_transform'] = paths['training_ready_marker'].parent/'transforms/field/field_transform.json'
        for key,path in paths.items():
            if arm.sha256(path) != manifest[key+'_sha256']:
                raise RuntimeError('Changed input: '+key)
        backup = dest/'pre_validation_repair_61548250'
        backup.mkdir(exist_ok=False)
        os.link(checkpoint, backup/checkpoint.name)
        os.link(dest/'run_manifest.json', backup/'run_manifest.json')
        # The trainer uses atomic replacement, so the links retain the original bytes.
        state['source_contract'] = new
        arm.atomic_torch_save(state, checkpoint)
        arm.atomic_json(backup/'migration.json', dict(original=ORIGINAL, migrated=arm.sha256(checkpoint),
            changed_fields=['source_contract'], changed_sources=CHANGES, step=346364, epoch=1, cursor=346364))
        print('Verified recovery at epoch 1 end: no epoch 1 updates remain.',flush=True)


if __name__ == '__main__':
    migrate()
