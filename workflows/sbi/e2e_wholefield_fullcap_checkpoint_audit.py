"""Independent completed-output checkpoint audit; no partial-panel scoring."""
import json
import os
from pathlib import Path

import numpy as np

from workflows.sbi import e2e_wholefield_fullcap_run as gen
from workflows.sbi.e2e_wholefield_sensitivity import fine_vector


def run():
    gen.compute()
    root = gen.ROOT
    reuse = gen.read(root/'REUSE_VERIFIED.json')
    gen.verify_record(reuse['manifest'])
    done = gen.read(root/'allocations/s003/EXTRACTION_COMPLETE.json')
    expected = set(reuse['tasks'])
    if not done['passed'] or set(done['completed']+done['skipped_verified']) != expected:
        raise ValueError('Extraction inventory mismatch')
    manifest = gen.record(root/'extraction_v1/MANIFEST.json')
    extraction = gen.read(manifest['path']); gen.verify_record(extraction['source'])
    receipts = []; parity = trace = 0.; exact_replays = 0
    for key in sorted(expected):
        path = root/'extraction_v1/cores'/key/'COMPLETE.json'
        rec = gen.read(path)
        if not rec['passed'] or rec['key'] != key or rec['draws'] != 4:
            raise ValueError('Extraction identity mismatch')
        gen.verify_record(rec['binding']); gen.verify_record(rec['output'])
        bound = gen.read(rec['binding']['path'])
        if bound['extraction'] != manifest:
            raise ValueError('Extraction source binding mismatch')
        for item in (bound['density'], bound['tides'], bound['geometry'], *bound['targets']):
            gen.verify_record(item)
        with np.load(rec['output']['path'], allow_pickle=False) as f:
            if f['eigen'].shape != (2, 8, 4, 2, 4096, 3) or f['density'].shape != (8, 4, 2, 4096):
                raise ValueError('Wrong core shape')
            if np.any(f['density'] <= 0) or not np.isfinite(f['eigen']).all():
                raise ValueError('Invalid physical cores')
            if np.min(np.diff(f['eigen'], axis=-1)) < 0:
                raise ValueError('Unordered eigenvalues')
            for eigen in f['eigen']:
                np.testing.assert_allclose(eigen.sum(-1), f['density']-1, atol=1e-9, rtol=0)
            np.testing.assert_array_equal(fine_vector(f['density'], f['eigen'][1])[0], f['draws'])
            np.testing.assert_array_equal(fine_vector(f['truth_density'], f['truth_eigen'])[0], f['truth'])
            if len(set(f['pair_id'])) != 8:
                raise ValueError('Duplicate pair identities')
        parity = max(parity, rec['original_parity_maxabs'])
        trace = max(trace, rec['correction_trace_maxabs'])
        if rec['phase'] in ('ph012', 'ph013') and rec['seed'] == 17:
            if rec['corrected_parity_maxabs'] != 0:
                raise ValueError('Previously exact corrected replay changed')
            exact_replays += 1
        receipts.append(gen.record(path))
    allocations = [gen.read(root/'allocations'/segment/'ACCOUNTED.json') for segment in ('s001', 's002', 's003')]
    tasks = gen.read(root/'MANIFEST.json')['tasks']
    new_completed = [t['key'] for t in tasks if (root/'tasks'/t['key']/'COMPLETE.json').exists()]
    result = dict(passed=True, extracted_tasks=len(receipts), exact_corrected_replays=exact_replays,
                  original_parity_maxabs=parity, correction_trace_maxabs=trace,
                  new_generation_complete=new_completed, outstanding_generation_tasks=192-len(expected)-len(new_completed),
                  allocated_gpu_hours=sum(a['allocated_gpu_hours'] for a in allocations),
                  source=gen.record(__file__), extraction_manifest=manifest, receipts=receipts,
                  job=os.environ['SLURM_JOB_ID'], gate2_ready=False, posterior_qualified=False)
    gen.write(root/'CHECKPOINT_AUDIT_20261010.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'receipts'}), flush=True)


if __name__ == '__main__':
    run()
