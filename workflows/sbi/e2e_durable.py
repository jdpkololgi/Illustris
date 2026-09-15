"""Transactional checkpoint publication for trusted, local research jobs."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
from workflows.sbi import e2e_wide_pipeline as p


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_json(path, value, replace=False):
    """Flush before publication. Immutable receipts never silently overwrite."""
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(prefix='.json-', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(name, path)
    else:
        os.link(name, path)
        os.unlink(name)
    sync_dir(path.parent)


@contextlib.contextmanager
def single_writer(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'WRITER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def save(root, **kwargs):
    """Only LATEST commits a checkpoint; incomplete generations are preserved."""
    root = Path(root)
    generation = Path(tempfile.mkdtemp(prefix=f"checkpoint_{kwargs['step']:06d}_", dir=root))
    path = generation / 'state.pt'
    p.save_checkpoint(path, **kwargs)
    for file in (path, path.with_suffix('.json')):
        with file.open('rb') as stream:
            os.fsync(stream.fileno())
    receipt = dict(path=str(path.relative_to(root)), step=kwargs['step'],
                   sha256=p.sha256(path), binding_sha256=p.digest(kwargs['binding']))
    publish_json(generation / 'COMMITTED.json', receipt)
    sync_dir(root)
    publish_json(root / 'LATEST.json', receipt, replace=True)
    return receipt


def load(root, binding):
    root = Path(root)
    receipt = json.loads((root / 'LATEST.json').read_text())
    path = (root / receipt['path']).resolve()
    if root.resolve() not in path.parents or path.name != 'state.pt':
        raise ValueError('checkpoint path escapes branch')
    committed = json.loads((path.parent / 'COMMITTED.json').read_text())
    if receipt != committed or receipt['sha256'] != p.sha256(path):
        raise ValueError('incomplete or corrupt checkpoint; no silent fallback')
    state = p.load_checkpoint(path, binding, 'fine', 'diffusion')
    if state['step'] != receipt['step']:
        raise ValueError('checkpoint step mismatch')
    return state, receipt
