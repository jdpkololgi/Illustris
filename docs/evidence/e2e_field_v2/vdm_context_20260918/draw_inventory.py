"""Independent final inventory audit; reads payloads on the existing report node."""
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
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    started = time.time()
    allocation = read(ROOT/'resources/10_START.json')
    check(allocation['stage'] == 'report' and allocation['gpus'] == 0, 'allocation stage')
    check(os.environ.get('SLURM_JOB_ID') == allocation['job'], 'allocation identity')
    check(socket.gethostname().startswith('nid'), 'compute node required')
    manifest_sha = digest(ROOT/'MANIFEST.json')
    check(manifest_sha == '67e1f9d0edc73b58c761e35f1f2dc13d4fe102e27f260104ecdb87b725a9bf10', 'manifest drift')
    frozen_sha = digest(ROOT/'MODELS_FROZEN.json')
    check(frozen_sha == '04abc713e9a1baa87288640ac10672b318ee7fd218ebcb2229eb78323bd3ea57', 'model freeze drift')
    manifest, frozen = read(ROOT/'MANIFEST.json'), read(ROOT/'MODELS_FROZEN.json')
    check(digest(ROOT/'DRAW_LEDGER.json') == manifest['data_receipts']['DRAW_LEDGER.json'], 'ledger drift')
    ledger = read(ROOT/'DRAW_LEDGER.json')
    tasks = ledger['tasks']
    expected_ids = {t['task_id'] for t in tasks}
    check(len(tasks) == len(expected_ids) == 688, 'case identity/count')
    check({p.name for p in (ROOT/'draws').iterdir() if p.is_dir()} == expected_ids, 'unexpected or missing draw directories')
    payloads, fine_receipts, coarse_bindings, checkpoint_cache = {}, {}, {}, {}

    def checkpoint(task, factor):
        key = (task['arm'], factor, task['replica'], task['checkpoint'])
        if key not in checkpoint_cache:
            branch = ROOT/'models'/f"{task['arm']}_{factor}_seed{task['replica']}"
            pointer = read(branch/f"CHECKPOINT_{task['checkpoint']:06d}.json")
            path = (branch/pointer['path']).resolve()
            check(path.is_relative_to(branch.resolve()), 'checkpoint path escape')
            check(read(path.parent/'COMMITTED.json') == pointer, 'checkpoint pointer drift')
            check(frozen['checkpoints'][str(path.relative_to(ROOT.resolve()))] == pointer['sha256'], 'unfrozen checkpoint')
            checkpoint_cache[key] = pointer['sha256']
        return checkpoint_cache[key]

    def payload(receipt_path, binding):
        row = read(receipt_path)
        check(row['binding'] == binding, f'binding drift: {receipt_path}')
        path = (receipt_path.parent/row['file']).resolve()
        check(path.parent == receipt_path.parent.resolve(), 'payload path escape')
        key = str(path.relative_to(ROOT.resolve()))
        check(key not in payloads, 'unexpected payload alias')
        actual = digest(path)
        check(actual == row['sha256'], f'payload hash mismatch: {path}')
        payloads[key] = dict(sha256=actual, bytes=path.stat().st_size)
        return digest(receipt_path)

    for index, task in enumerate(tasks):
        folder = ROOT/'draws'/task['task_id']
        complete = read(folder/'COMPLETE.json')
        check(complete['task'] == task and complete['manifest_sha256'] == manifest_sha, 'completion binding')
        starts = list(range(task['start'], task['start']+task['count'], 8))
        check(task['count'] % 8 == 0 and set(complete['chunks']) == {str(i) for i in starts}, 'chunk coverage')
        check({p.name for p in folder.glob('[0-9]*.json')} == {f'{i:06d}.json' for i in starts}, 'unexpected fine chunk')
        for first in starts:
            path = folder/f'{first:06d}.json'
            binding = dict(manifest_sha256=manifest_sha, task=task, ids=list(range(first, first+8)),
                checkpoint_sha256=checkpoint(task, 'fine'),
                coarse_checkpoint_sha256=checkpoint(task, 'coarse') if task['arm'] == 'D' else None)
            receipt_sha = payload(path, binding)
            check(receipt_sha == complete['chunks'][str(first)], 'completion chunk hash')
            fine_receipts[str(path.relative_to(ROOT))] = receipt_sha
        if task['arm'] == 'D' and task['coarse_mode'] != 'oracle_diagnostic':
            count = 32 if task['coarse_mode'] == 'fixed_mean' else task['count']
            purpose = 'joint' if task['purpose'].startswith('joint') else task['purpose']
            parent = ROOT/'parents'/f"D_seed{task['replica']}"/f"update_{task['checkpoint']:06d}"/purpose/task['domain']/f"steps{task['steps']}"
            for first in range(0, count, 8):
                path = parent/f'{first:06d}.json'
                binding = dict(manifest_sha256=manifest_sha, checkpoint_sha256=checkpoint(task, 'coarse'),
                    domain=task['domain'], replica=task['replica'], checkpoint=task['checkpoint'],
                    steps=task['steps'], purpose=purpose, ids=list(range(first, first+8)))
                check(path not in coarse_bindings or coarse_bindings[path] == binding, 'shared parent mismatch')
                coarse_bindings[path] = binding
        if index % 64 == 0:
            print('FINE_CASES_VERIFIED', index+1, flush=True)
    check(set((ROOT/'parents').glob('*/*/*/*/*/[0-9]*.json')) == set(coarse_bindings), 'coarse inventory mismatch')
    coarse_receipts = {str(p.relative_to(ROOT)):payload(p, b) for p, b in sorted(coarse_bindings.items())}
    check(len(fine_receipts)*8 == ledger['central_draws'] == 33280, 'fine draw count')
    check(len(coarse_receipts)*8 == ledger['coarse_draws'] == 7872, 'coarse draw count')
    result = dict(passed=True, job=allocation['job'], step=os.environ.get('SLURM_STEP_ID'),
        node=socket.gethostname(), started_unix=started, ended_unix=time.time(),
        auditor_sha256=digest(__file__), manifest_sha256=manifest_sha, models_frozen_sha256=frozen_sha,
        ledger_sha256=digest(ROOT/'DRAW_LEDGER.json'), cases=len(tasks), fine_draws=33280, coarse_draws=7872,
        fine_receipts=fine_receipts, coarse_receipts=coarse_receipts, payloads=payloads,
        payload_bytes=sum(p['bytes'] for p in payloads.values()),
        scope='Exact registered case/chunk/draw identities, shared-parent bindings, and all actual draw payload hashes; no new metrics or fitting')
    with (ROOT/'analysis/DRAW_INTEGRITY.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('DRAW_INTEGRITY_PASS', len(payloads), result['payload_bytes'], flush=True)


if __name__ == '__main__':
    main()
