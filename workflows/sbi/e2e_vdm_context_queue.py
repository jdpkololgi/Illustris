"""Fixed-work queue locks and receipt checks, without scientific adaptation."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from workflows.sbi.e2e_vdm_context_data import read_json
from workflows.sbi.e2e_field_build_products import sha256


@contextmanager
def task_lock(folder,blocking=False):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'QUEUE.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def task_done(root,task,limit=None):
    """Metadata readiness only; case()/report() still verify array payload hashes."""
    count=task['count'] if limit is None else min(task['count'],limit)
    folder=root/'draws'/task['task_id']
    for first in range(task['start'],task['start']+count,8):
        path=folder/f'{first:06d}.json'
        if not path.exists():
            return False
        receipt=read_json(path)
        if (receipt['binding']['task']!=task or receipt['binding']['ids']!=list(range(first,first+8))
                or receipt['binding']['manifest_sha256']!=sha256(root/'MANIFEST.json')):
            raise ValueError('sampling queue receipt binding drift')
    if count==task['count']:
        complete=folder/'COMPLETE.json'
        if not complete.exists():
            return False
        receipt=read_json(complete)
        expected={str(i):sha256(folder/f'{i:06d}.json') for i in range(task['start'],task['start']+count,8)}
        if receipt['task']!=task or receipt['chunks']!=expected or receipt['manifest_sha256']!=sha256(root/'MANIFEST.json'):
            raise ValueError('completed sampling queue receipt drift')
    return True


def disk_bytes(root):
    """Bounded experiment tree only; count hard-linked files once."""
    import os
    seen=set()
    total=0
    for directory,_,files in os.walk(root,followlinks=False):
        for name in files:
            path=Path(directory)/name
            if path.is_symlink():
                continue
            try:
                state=path.stat()
            except FileNotFoundError:
                continue  # Another worker atomically published its temporary file.
            key=(state.st_dev,state.st_ino)
            if key not in seen:
                seen.add(key)
                total+=state.st_size
    return total
