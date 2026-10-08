"""Checkpoint completed products without treating an incomplete panel as a result."""
import os
import time
import e2e_true_coarse_fraction_saved as io


def run():
    io.compute_guard()
    completed, missing, partial = [], [], []
    for phase in io.PHASES:
        for cap in ('NGC', 'SGC'):
            for first in range(0, 32, 4):
                address = dict(phase=phase, cap=cap, first=first, draws=4)
                core = io.OUT/'cores'/phase/cap/f'd{first:03d}'
                field = io.OUT/'fields'/phase/cap/f'd{first:03d}'
                if (core/'COMPLETE.json').exists():
                    rec = io.read(core/'COMPLETE.json')
                    if not rec['passed'] or any(rec[k] != v for k, v in address.items()):
                        raise ValueError('Invalid completed core address')
                    io.verify(rec['output'])
                    generation = io.read(field/'COMPLETE.json')
                    if not generation['passed'] or any(generation[k] != v for k, v in address.items()):
                        raise ValueError('Invalid completed field address')
                    io.verify(generation['field'])
                    completed.append(dict(**address, core=io.record(core/'COMPLETE.json'),
                                          field=io.record(field/'COMPLETE.json')))
                elif core.exists() or field.exists():
                    partial.append(address)
                else:
                    missing.append(address)
    io.write(io.OUT/'PARTIAL_STATUS_20261008.json', dict(
        diagnostic_complete=False, inference_branch=None,
        completed_batches=completed, missing_batches=missing, partial_batches=partial,
        completed_count=len(completed), expected_count=32,
        sources=[io.record(__file__), io.record(io.__file__)], inputs=list(io.CACHE.values()),
        scope='one parent seed, two reused phases, diagnostic only',
        job=os.environ['SLURM_JOB_ID'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        status='Checkpoint only; no full A/B conclusion until all registered batches complete.'))
    print(io.json.dumps(dict(completed=len(completed), missing=len(missing), partial=len(partial))), flush=True)


if __name__ == '__main__':
    run()
