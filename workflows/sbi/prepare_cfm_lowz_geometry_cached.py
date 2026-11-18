"""Same low-z geometry algorithm with transaction-bounded validation caching."""
import argparse
from pathlib import Path

from workflows.sbi import prepare_cfm_lowz_campaign as campaign
from workflows.sbi import cfm_geometry_validation_cache as cache


def build(phase):
    c=campaign.c
    c.require_compute();c.phase_guard(phase)
    sources=[Path(__file__),Path(cache.__file__),Path(campaign.__file__),
             Path(campaign.geometry.__file__),Path(campaign.candidates.__file__)]
    bindings={str(path):c.sha256(path) for path in sources}
    original=c.atomic_json
    with cache.cached_grid_validation() as status:
        def commit(path,value):
            # Fail before any completion receipt is written, not after.
            status['check_authority']()
            if any(c.sha256(path)!=digest for path,digest in bindings.items()):
                raise ValueError('Geometry implementation changed during execution')
            value=dict(value,execution_acceleration=dict(
                method='identical-grid-validation-cache-v1',source_bindings=bindings,
                validation_calls=status['calls'],unique_grids=status['unique_grids'],
                coordinate_authority_rechecked_before_commit=True,
                numerical_algorithm_changed=False))
            original(path,value)
        c.atomic_json=commit
        try:
            return campaign.build_geometry(phase)
        finally:
            c.atomic_json=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True)
    build(parser.parse_args().phase)
