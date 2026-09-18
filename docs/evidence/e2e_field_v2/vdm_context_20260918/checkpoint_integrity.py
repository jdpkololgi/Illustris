"""Read-only checkpoint/source audit within the registered CPU report allocation."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.time()
    allocation = read(ROOT/'resources/10_START.json')
    assert allocation['stage'] == 'report' and allocation['gpus'] == 0
    assert os.environ.get('SLURM_JOB_ID') == allocation['job']
    assert socket.gethostname().startswith('nid'), 'compute node required'
    assert digest(ROOT/'MANIFEST.json') == '67e1f9d0edc73b58c761e35f1f2dc13d4fe102e27f260104ecdb87b725a9bf10'
    assert digest(ROOT/'MODELS_FROZEN.json') == '04abc713e9a1baa87288640ac10672b318ee7fd218ebcb2229eb78323bd3ea57'
    manifest, frozen = read(ROOT/'MANIFEST.json'), read(ROOT/'MODELS_FROZEN.json')
    assert len(frozen['checkpoints']) == 30 and len(frozen['branch_receipts']) == 10
    for name, expected in manifest['source_sha256'].items():
        assert digest(Path(manifest['source'])/name) == expected, name
    for name, expected in manifest['data_receipts'].items():
        assert digest(ROOT/name) == expected, name
    for name, expected in frozen['branch_receipts'].items():
        assert digest(ROOT/name/'COMPLETE.json') == expected, name
    checkpoints = {}
    for name, expected in frozen['checkpoints'].items():
        path = (ROOT/name).resolve()
        assert path.is_relative_to((ROOT/'models').resolve())
        actual = digest(path)
        assert actual == expected, name
        checkpoints[name] = dict(sha256=actual, bytes=path.stat().st_size)
        print('CHECKPOINT_VERIFIED', name, flush=True)
    assert digest(ROOT/'MODELS_FROZEN.json') == '04abc713e9a1baa87288640ac10672b318ee7fd218ebcb2229eb78323bd3ea57'
    result = dict(passed=True, job=allocation['job'], step=os.environ.get('SLURM_STEP_ID'),
        node=socket.gethostname(), started_unix=started, ended_unix=time.time(),
        auditor_sha256=digest(__file__), manifest_sha256=digest(ROOT/'MANIFEST.json'),
        models_frozen_sha256=digest(ROOT/'MODELS_FROZEN.json'), checkpoints=checkpoints,
        source_files_verified=len(manifest['source_sha256']),
        data_receipts_verified=len(manifest['data_receipts']),
        model_payload_bytes=sum(x['bytes'] for x in checkpoints.values()),
        scope='Actual 30 checkpoint payloads and frozen source/metadata; draw arrays are verified by full report')
    with (ROOT/'analysis/CHECKPOINT_INTEGRITY.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print('CHECKPOINT_INTEGRITY_PASS', result['model_payload_bytes'], flush=True)


if __name__ == '__main__':
    main()
