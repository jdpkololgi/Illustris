import copy
import unittest
from unittest.mock import patch
import numpy as np

from workflows.sbi import cfm_geometry_validation_cache as cache
from workflows.sbi import e2e_coupled_geometry as geometry


class GeometryCacheTests(unittest.TestCase):
    def grid(self):
        return dict(origin_mpc_h=[-100.,-200.,-300.],shape=[128,128,128],
            cell_mpc_h=3.383,distance_unit='Mpc/h',
            coordinate_sha256=cache.coord.c.sha256(cache.coord.CONFIG))

    def test_identical_coordinate_results_and_once_per_grid_validation(self):
        grid=self.grid();centers=np.random.default_rng(4).integers(-64,200,(20,3))
        reference=[geometry.domain_positions(x,grid) for x in centers]
        original=cache.coord.validate_grid
        with patch.object(cache.coord,'validate_grid',wraps=original) as calls:
            with cache.cached_grid_validation() as status:
                actual=[geometry.domain_positions(x,grid) for x in centers]
                self.assertEqual(status['calls'],60)
                self.assertEqual(status['unique_grids'],1)
            self.assertEqual(calls.call_count,1)
        for a,b in zip(reference,actual):
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertIs(cache.coord.validate_grid,original)

    def test_changed_grid_is_revalidated(self):
        grid=self.grid()
        with cache.cached_grid_validation():
            cache.coord.validate_grid(grid)
            bad=copy.deepcopy(grid);bad['cell_mpc_h']=1.
            with self.assertRaises(ValueError):cache.coord.validate_grid(bad)

    def test_changed_authority_fails_and_restores_function(self):
        original=cache.coord.validate_grid
        with patch.object(cache.coord.c,'sha256',side_effect=['before','after']):
            with self.assertRaisesRegex(ValueError,'authority changed'):
                with cache.cached_grid_validation():pass
        self.assertIs(cache.coord.validate_grid,original)

    def test_wrapper_checks_authority_before_committing(self):
        from workflows.sbi import prepare_cfm_lowz_geometry_cached as wrapper
        c=wrapper.campaign.c
        original=c.atomic_json
        def fake_build(phase):
            c.atomic_json('unused',{'phase':phase})
        with patch.object(c,'require_compute'),patch.object(c,'phase_guard'), \
             patch.object(wrapper.campaign,'build_geometry',side_effect=fake_build), \
             patch.object(c,'atomic_json') as write, \
             patch.object(cache.coord.c,'sha256',side_effect=['source']*5+['before','after']):
            with self.assertRaisesRegex(ValueError,'authority changed'):
                wrapper.build('ph001')
            write.assert_not_called()
        self.assertIs(c.atomic_json,original)


if __name__=='__main__':unittest.main()
