"""Read-only small real-cap equivalence benchmark for validation caching."""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi.cfm_geometry_validation_cache import cached_grid_validation


def run(output):
    c.require_compute();phase='ph001';c.phase_guard(phase)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    config=copy.deepcopy(c.config());config['geometry']['candidates_per_cap']=256
    record=coord.verify_receipt(coord.ROOT/'observations'/phase/'NGC_COMPLETE.json')
    timings=[];results=[]
    with patch.object(c,'config',return_value=config):
        started=time.monotonic();results.append(geometry.candidate_buckets(phase,'NGC',record))
        timings.append(time.monotonic()-started)
        started=time.monotonic()
        with cached_grid_validation() as counts:
            results.append(geometry.candidate_buckets(phase,'NGC',record))
        timings.append(time.monotonic()-started)
    if results[0]!=results[1]:raise ValueError('Cached path changes candidate results')
    report=dict(phase=phase,cap='NGC',draws=256,exact_candidate_and_rejection_equality=True,
        reference_seconds=timings[0],cached_seconds=timings[1],
        validation_calls=counts['calls'],unique_validated_grids=counts['unique_grids'],
        source_code_sha256=c.sha256(__file__),geometry_sha256=c.sha256(geometry.__file__),
        coordinate_sha256=c.sha256(coord.CONFIG),counts_or_targets_read=False)
    c.atomic_json(output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
