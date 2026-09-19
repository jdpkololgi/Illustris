"""Closed, separate authority for coupled-field data preparation (not training)."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/e2e_coupled_data_v1.json'
ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1')
TRAIN = ('ph000', 'ph002', 'ph003', 'ph007', 'ph008', 'ph009', 'ph010',
         'ph011', 'ph020', 'ph021', 'ph022', 'ph023', 'ph024')
DEVELOPMENT = ('ph012', 'ph013')
CONFIRMATION = tuple(f'ph{i:03d}' for i in range(14, 20))
ROLES = {**dict.fromkeys(TRAIN, 'train'), **dict.fromkeys(DEVELOPMENT, 'development'),
         **dict.fromkeys(CONFIRMATION, 'confirmation')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            result.update(block)
    return result.hexdigest()


def config(path=CONFIG):
    value = json.loads(Path(path).read_text())
    if (value['schema'] != 'e2e-coupled-data-v1' or value['phase_roles'] != ROLES
            or value['root'] != str(ROOT) or value['scientific_training_authorized']
            or value['production_ready'] or value['forbidden_phases'] != ['ph001', 'ph006']):
        raise ValueError('unregistered coupled data authority')
    return value


def phase_guard(phase):
    if phase not in ROLES:
        raise PermissionError(f'Phase not authorized for coupled preparation: {phase}')
    return phase


def guarded(path, phase=None, output=False):
    path = Path(path)
    if phase is not None:
        phase_guard(phase)
    for name in (str(path), str(path.resolve())):
        phases = set(re.findall(r'ph\d{3}', name))
        if not phases <= ROLES.keys() or (phase and phases and phases != {phase}):
            raise PermissionError('forbidden/mismatched phase in path')
    resolved = path.resolve()
    if output and ROOT not in resolved.parents:
        raise PermissionError('output must be a child of the registered coupled root')
    return resolved


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('nid'):
        raise RuntimeError('payload processing requires a Slurm compute node')


def atomic_json(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.json-', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(name, path)
    else:
        os.link(name, path)
        os.unlink(name)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def single_writer(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'WRITER.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def file_record(path, *, content_hash=False):
    path = Path(path)
    stat = path.stat()
    result = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if content_hash:
        result['sha256'] = sha256(path)
    return result


def provenance():
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
                config_sha256=sha256(CONFIG), host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                contract_source_sha256=sha256(__file__))


def paths(phase):
    phase_guard(phase)
    old = json.loads((REPO / config()['source_registry']).read_text())
    result = {k: v.format(phase=phase, mock=int(phase[2:]))
              for k, v in old['path_templates'].items() if k != 'phase_output'}
    if phase == 'ph000':
        result.update(old['phases'][phase].get('observation_path_overrides', {}))
    result = {k: guarded(v, phase) for k, v in result.items()}
    result['full'] = result['lss'] / config()['observation']['full_catalogue']
    result['randoms'] = [result['lss'] / f'BGS_BRIGHT_{i}_full_HPmapcut.ran.fits'
                         for i in config()['observation']['random_ids']]
    return result


def verify_receipt(path, *, payload=True):
    receipt = json.loads(Path(path).read_text())
    if receipt['config_sha256'] != sha256(CONFIG) or not receipt['pass']:
        raise ValueError('receipt config/failure mismatch')
    if payload:
        for record in receipt.get('outputs', []):
            file = guarded(record['path'], receipt.get('phase'))
            if file.stat().st_size != record['bytes'] or sha256(file) != record['sha256']:
                raise ValueError(f'payload changed: {file}')
    return receipt


def bind_run():
    """Fail closed when an existing preparation root has a different authority."""
    c = config()
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'DATA_AUTHORITY.json'
    binding = dict(schema='e2e-coupled-data-authority-v1', config=c, config_sha256=sha256(CONFIG))
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('existing preparation root has a different data authority')
    else:
        atomic_json(path, binding)
    return binding
