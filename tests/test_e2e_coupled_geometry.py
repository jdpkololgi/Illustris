import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as contract
from workflows.sbi import e2e_coupled_geometry as geometry
from workflows.sbi import e2e_coupled_geometry_kernel as kernel
from workflows.sbi.e2e_coupled_geometry import admissible,domain_positions,periodic_delta
from workflows.sbi.e2e_coupled_coordinates import grid_record
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec


class CoupledGeometryTests(unittest.TestCase):
    def test_pair_centers_and_periodic_ownership(self):
        grid=grid_record(GridSpec((0,0,0),(200,200,200),3.383,27.064))
        middle,cores=domain_positions([64,64,64],grid)
        np.testing.assert_allclose(periodic_delta(cores[1],cores[0]),[108.256,0,0],atol=1e-10)
        np.testing.assert_allclose(periodic_delta(middle,cores[0]),[54.128,0,0],atol=1e-10)
        self.assertTrue(admissible(middle,cores,[],[]))
        self.assertFalse(admissible(middle+2000,cores+2000,[middle],cores))
        # Centers far enough apart can still overlap at their nearer end cores.
        self.assertFalse(admissible(middle+[180,0,0],cores+[180,0,0],[middle],cores))
        self.assertTrue(admissible(middle+[240,0,0],cores+[240,0,0],[middle],cores))

    def test_integer_support_blocks_and_candidates_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'ph007_support.h5'
            shape=(147,145,149)
            support=np.ones(shape,dtype=np.uint8)
            support[:25]=0; support[:,75:100]=0; support[:,:,100:]=0
            with h5py.File(path,'x') as f: f['support_random']=support
            grid=grid_record(GridSpec((500,-200,100),shape,3.383,27.064))
            record=dict(grid=grid,outputs=[dict(path=str(path))])
            cfg=contract.config(); cfg['geometry']['candidates_per_cap']=256
            with patch.object(contract,'config',return_value=cfg):
                expected,expected_diagnostics=geometry.candidate_buckets('ph007','NGC',record)
                actual,diagnostics=kernel.candidate_buckets('ph007','NGC',record)
            self.assertEqual(diagnostics,expected_diagnostics)
            self.assertEqual(actual,expected)
            counts=kernel.support_block_counts(support)
            self.assertEqual(int(counts.sum()),int(support.sum()))
            for center in ([16,16,16],[48,64,80],[96,104,112]):
                low=(np.array(center)-16)//8
                pooled=counts[tuple(slice(v,v+4) for v in low)].sum()/32768.
                direct=support[tuple(slice(v-16,v+16) for v in center)].mean()
                self.assertEqual(pooled,direct)


if __name__=='__main__':
    unittest.main()
