"""Bounded CPU parallelism for targetless full-cap caches, not agent autonomy."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import json
import multiprocessing
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c


def task(phase, cap, root):
    from workflows.sbi.e2e_exterior_fullcap import prepare
    out = root/phase/cap
    marker = out/'PREPARED.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        for item in record['outputs']:
            if c.sha256(item['path']) != item['sha256']:
                raise ValueError('Existing prepared output changed')
        return dict(phase=phase, cap=cap, existing=True, seconds=record['seconds'])
    logs = root/'logs'; logs.mkdir(parents=True, exist_ok=True)
    with (logs/f'{phase}_{cap}.log').open('x', buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            prepare(phase, cap, out)
    record = json.loads(marker.read_text())
    return dict(phase=phase, cap=cap, existing=False, seconds=record['seconds'])


def run(root, phases, workers):
    c.require_compute()
    if not 1 <= workers <= 8 or len(phases) != len(set(phases)):
        raise ValueError('Invalid bounded cache preparation')
    from workflows.sbi.e2e_exterior_fullcap import input_guard
    for phase in phases:
        input_guard(phase)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(task, phase, cap, root) for phase in phases for cap in ('NGC', 'SGC')]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phases', nargs='+', required=True); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(); run(a.root, a.phases, a.workers)
