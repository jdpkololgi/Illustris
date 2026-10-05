"""Bounded hardware check using the immutable production implementation."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sampler', 'tides'], required=True)
    p.add_argument('--cap', choices=['NGC', 'SGC'], required=True)
    a = p.parse_args()
    root = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
    freeze_path = root/'SELECTION_FREEZE.json'
    source = root/'source_78156ac3417555ef269f'
    sys.path.insert(0, str(source))
    from workflows.sbi.e2e_exterior_batch import check_freeze
    from workflows.sbi import e2e_coupled_contract as c
    import torch
    c.require_compute()
    freeze = check_freeze(freeze_path)
    props = torch.cuda.get_device_properties(0)
    if not 39 <= props.total_memory/2**30 <= 41:
        raise ValueError('This test requires an actual 40-GB GPU')
    out = root/'verification/hbm40_20261005_v1'/a.cap/a.stage
    torch.cuda.reset_peak_memory_stats()
    if a.stage == 'sampler':
        from workflows.sbi.e2e_exterior_fullcap import run
        run(root/'fullcap_inputs/ph014'/a.cap, out, first=0,
            count=4, nfe=128, benchmark=True, freeze=freeze_path)
    else:
        from workflows.sbi.e2e_exterior_cap_tides import run
        run(root/'fullcap_density/ph014'/a.cap/'d000/COMPLETE.json',
            Path(freeze['prior']['path']), out, method=freeze['method'],
            padding=freeze['fullcap_padding'], freeze=freeze_path)
    torch.cuda.synchronize()
    record = dict(stage=a.stage, cap=a.cap, phase='ph014',
                  job=os.environ['SLURM_JOB_ID'], device=props.name,
                  total_gib=props.total_memory/2**30,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  passed=True, production_output=False,
                  source_snapshot=str(source), freeze_sha256=c.sha256(freeze_path),
                  harness_sha256=c.sha256(__file__),
                  scope='sampler NFE2 memory test or exact four-draw tidal replay; no scientific retuning')
    c.atomic_json(out/'HARDWARE_CHECK.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
