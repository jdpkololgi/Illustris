#!/usr/bin/env python3
"""Small fail-closed publication helpers for the P12-A blind state machine.

Every immutable artifact is first written and fsynced under a unique temporary
name.  A same-filesystem hard link then publishes the complete file with
``O_EXCL`` semantics: an existing destination is never replaced, and a crash
cannot leave a partially written canonical marker.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping

import numpy as np


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_file_exclusive(temporary: Path, destination: Path) -> None:
    """Atomically publish a complete same-filesystem file without overwrite."""

    temporary = temporary.resolve()
    destination = destination.resolve()
    if temporary.parent != destination.parent:
        raise ValueError("exclusive publication requires a same-directory temporary")
    try:
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite immutable blind artifact: {destination}"
        ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably and atomically publish a JSON object exactly once."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("xb") as stream:
            data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        publish_file_exclusive(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_npz_exclusive(path: Path, *, compressed: bool = True, **arrays: Any) -> None:
    """Durably and atomically publish an NPZ archive exactly once."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("xb") as stream:
            writer = np.savez_compressed if compressed else np.savez
            writer(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        publish_file_exclusive(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_npz_exact(path: Path, arrays: Mapping[str, Any]) -> None:
    """Require an NPZ to contain exactly the supplied arrays and values."""

    path = path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise RuntimeError(f"immutable NPZ schema differs: {path}")
        for name, expected in arrays.items():
            if not np.array_equal(np.asarray(archive[name]), np.asarray(expected)):
                raise RuntimeError(f"immutable NPZ array differs: {path}:{name}")


def write_or_validate_npz_exclusive(
    path: Path, *, compressed: bool = True, **arrays: Any
) -> None:
    """Publish once, or adopt only an exact complete crash-orphaned archive."""

    path = path.resolve()
    if path.exists():
        validate_npz_exact(path, arrays)
        return
    try:
        write_npz_exclusive(path, compressed=compressed, **arrays)
    except FileExistsError:
        # A concurrent writer may have won the O_EXCL publication race.  Its
        # result is usable only if it is byte-content equivalent at array level.
        validate_npz_exact(path, arrays)


def publish_existing_exclusive(source: Path, destination: Path) -> None:
    """Publish a completed attempt artifact while retaining its attempt copy."""

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        _fsync_directory(destination.parent)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite immutable blind artifact: {destination}"
        ) from error
