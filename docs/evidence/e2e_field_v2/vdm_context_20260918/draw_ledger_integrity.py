"""Audit exact registered draw identities and receipt bindings, without scoring."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.time()
    assert socket.gethostname().startswith('nid'), 'compute node required'
    assert os.environ['SLURM_JOB_ID'] == read(ROOT/'resources/10_START.json')['job']
    manifest = digest(ROOT/'MANIFEST.json')
    assert manifest == '67e1f9d0edc73b58c761e35f1f2dc13d4fe102e27f260104ecdb87b725a9bf10'
    ledger = read(ROOT/'DRAW_LEDGER.json')
    tasks = ledger['tasks']
    assert len(tasks) == 688 and sum(t['count'] for t in tasks) == 33280
    assert len({t['task_id'] for t in tasks}) == 688
    frozen = read(ROOT/'MODELS_FROZEN.json')
    checkpoints = {}
    for name, sha in frozen['checkpoints'].items():
        parts = Path(name).parts
        branch, update = parts[1], int(parts[2].split('_')[1])
        checkpoints[branch, update] = sha
        pointer = read(ROOT/'models'/branch/f'CHECKPOINT_{update:06d}.json')
        assert pointer['sha256'] == sha
        assert (ROOT/'models'/branch/pointer['path']).resolve() == (ROOT/name).resolve()
        assert read((ROOT/name).parent/'COMMITTED.json') == pointer

    expected_parents, fine_chunks, seeds = {}, {}, {}
    payload_bytes = 0
    by_branch = {}
    for task in tasks:
        key = task['task_id']
        folder = ROOT/'draws'/key
        complete = read(folder/'COMPLETE.json')
        assert complete['task'] == task and complete['manifest_sha256'] == manifest
        offsets = list(range(task['start'], task['start']+task['count'], 8))
        assert set(complete['chunks']) == {str(i) for i in offsets}
        assert {p.stem for p in folder.glob('*.json') if p.stem.isdigit()} == {f'{i:06d}' for i in offsets}
        ids = []
        for offset in offsets:
            path = folder/f'{offset:06d}.json'
            receipt = read(path)
            sha = digest(path)
            assert sha == complete['chunks'][str(offset)]
            fine_sha = checkpoints[f"{task['arm']}_fine_seed{task['replica']}", task['checkpoint']]
            coarse_sha = checkpoints[f"D_coarse_seed{task['replica']}", task['checkpoint']] if task['arm'] == 'D' else None
            binding = dict(manifest_sha256=manifest, task=task, ids=list(range(offset, offset+8)),
                           checkpoint_sha256=fine_sha, coarse_checkpoint_sha256=coarse_sha)
            assert receipt['binding'] == binding, path
            payload = (folder/receipt['file']).resolve()
            assert payload.parent == folder.resolve() and payload.is_file()
            payload_bytes += payload.stat().st_size
            fine_chunks[str(path.relative_to(ROOT))] = sha
            ids.extend(binding['ids'])
            assert len(receipt['seeds']) == 8
            # Fine random numbers are paired across arms/checkpoints/solver steps,
            # but owned by each distinct core, never by a shared parent domain.
            purpose = 'joint' if task['purpose'].startswith('joint') else task['purpose']
            for draw, seed in zip(binding['ids'], receipt['seeds']):
                address = (task['replica'], task['anchor'], purpose, draw)
                assert seeds.setdefault(address, seed) == seed
        assert ids == list(range(task['start'], task['start']+task['count']))
        branch = f"{task['arm']}_seed{task['replica']}"
        by_branch[branch] = by_branch.get(branch, 0)+len(ids)
        if task['arm'] == 'D' and task['coarse_mode'] != 'oracle_diagnostic':
            count = 32 if task['coarse_mode'] == 'fixed_mean' else task['count']
            purpose = 'joint' if task['purpose'].startswith('joint') else task['purpose']
            for offset in range(0, count, 8):
                name = (f"parents/D_seed{task['replica']}/update_{task['checkpoint']:06d}/"
                        f"{purpose}/{task['domain']}/steps{task['steps']}/{offset:06d}.json")
                binding = dict(manifest_sha256=manifest, checkpoint_sha256=coarse_sha,
                    domain=task['domain'], replica=task['replica'], checkpoint=task['checkpoint'],
                    steps=task['steps'], purpose=purpose, ids=list(range(offset, offset+8)))
                assert expected_parents.setdefault(name, binding) == binding
    actual = {str(p.relative_to(ROOT)) for p in (ROOT/'parents').glob('*/*/*/*/*/*.json') if p.stem.isdigit()}
    assert actual == set(expected_parents)
    assert len(actual)*8 == 7872
    parents = {}
    for name, binding in sorted(expected_parents.items()):
        path = ROOT/name
        receipt = read(path)
        assert receipt['binding'] == binding, name
        payload = (path.parent/receipt['file']).resolve()
        assert payload.parent == path.parent.resolve() and payload.is_file()
        payload_bytes += payload.stat().st_size
        parents[name] = digest(path)
    actual_tasks = {p.name for p in (ROOT/'draws').iterdir() if p.is_dir()}
    assert actual_tasks == {t['task_id'] for t in tasks}
    for branch in by_branch:
        assert read(ROOT/'sampling'/f'{branch}_ALL_COMPLETE.json')['manifest_sha256'] == manifest
    result = dict(passed=True, job=os.environ['SLURM_JOB_ID'], step=os.environ.get('SLURM_STEP_ID'),
        node=socket.gethostname(), started_unix=started, ended_unix=time.time(),
        auditor_sha256=digest(__file__), manifest_sha256=manifest,
        ledger_sha256=digest(ROOT/'DRAW_LEDGER.json'), tasks=len(tasks), fine_draws=33280,
        unique_coarse_draws=7872, fine_chunk_receipts=fine_chunks, parent_chunk_receipts=parents,
        by_branch=by_branch, payload_bytes_present=payload_bytes,
        paired_fine_random_addresses_verified=len(seeds),
        scope='Exact ledger, IDs, checkpoint/manifest bindings, paired RNG receipts and payload presence; full report verifies actual array hashes')
    with (ROOT/'analysis/DRAW_LEDGER_INTEGRITY.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('DRAW_LEDGER_INTEGRITY_PASS', {k: result[k] for k in ('tasks', 'fine_draws', 'unique_coarse_draws', 'payload_bytes_present', 'by_branch')}, flush=True)


if __name__ == '__main__':
    main()
