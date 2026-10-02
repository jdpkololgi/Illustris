"""Resolve one immutable array task, then execute only its registered unit."""
import argparse
import json
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_exterior_batch import run


def main():
    p = argparse.ArgumentParser(); p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True); p.add_argument('--tasks-sha', required=True)
    p.add_argument('--index', type=int, required=True); a = p.parse_args()
    if c.sha256(a.tasks) != a.tasks_sha:
        raise ValueError('Task manifest changed')
    tasks = json.loads(a.tasks.read_text())
    if not 0 <= a.index < len(tasks):
        raise ValueError('Unknown array task')
    task = tasks[a.index]
    run(task['phase'], task['cap'], task['first'], a.freeze)


if __name__ == '__main__':
    main()
