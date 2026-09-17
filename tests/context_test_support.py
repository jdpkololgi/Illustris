"""Temporary test paths must not accidentally resemble sealed phase names."""
from contextlib import contextmanager
import tempfile

from workflows.sbi.e2e_vdm_context_data import guarded


@contextmanager
def safe_temporary_directory():
    # Keep the production guard conservative. Random tempfile suffixes can
    # contain e.g. ph017, which the reader correctly refuses to open.
    for _ in range(8):
        with tempfile.TemporaryDirectory() as path:
            try:
                guarded(path)
            except PermissionError:
                continue
            yield path
            return
    raise RuntimeError('could not create a phase-neutral test directory')
