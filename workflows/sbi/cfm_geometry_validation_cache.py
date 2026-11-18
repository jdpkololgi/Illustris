"""Process-local cache of identical grid checks during label-free geometry.

Only file-validation overhead is cached: candidate draws, coordinate arithmetic,
support measurements, sorting and non-overlap selection remain unchanged.
The coordinate-authority bytes are checked before and after the context. Never
use this context across threads or allow a scientific receipt to commit before
check_authority() succeeds.
"""
from contextlib import contextmanager
import json

from workflows.sbi import e2e_coupled_coordinates as coord


@contextmanager
def cached_grid_validation():
    original = coord.validate_grid
    before = coord.c.sha256(coord.CONFIG)
    checked = set()
    counts = dict(calls=0,unique_grids=0)

    def check_authority():
        if coord.c.sha256(coord.CONFIG) != before:
            raise ValueError('Coordinate authority changed during cached validation')

    def validate(grid):
        counts['calls'] += 1
        key = json.dumps(grid,sort_keys=True,allow_nan=False,separators=(',',':'))
        if key not in checked:
            original(grid)
            checked.add(key)
            counts['unique_grids'] += 1
        return grid

    counts['check_authority'] = check_authority
    coord.validate_grid = validate
    try:
        yield counts
        check_authority()
    finally:
        coord.validate_grid = original
