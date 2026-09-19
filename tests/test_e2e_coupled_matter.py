import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from workflows.sbi.e2e_coupled_matter import load_checkpoint,save_checkpoint,crc_manifest


class CoupledMatterTests(unittest.TestCase):
    def test_two_slot_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            grid,state=load_checkpoint(root,'binding',(4,4,4))
            grid[0,0,0]=3
            state.update(next_file=1,particles=3)
            state=save_checkpoint(root,grid,state,'binding')
            first=state['slot']
            recovered,old=load_checkpoint(root,'binding',(4,4,4))
            np.testing.assert_array_equal(grid,recovered)
            # Simulate a crash after writing the inactive slot but before commit.
            np.save(root/f'counts_slot{1-first}.npy',np.zeros_like(grid))
            recovered,old=load_checkpoint(root,'binding',(4,4,4))
            np.testing.assert_array_equal(grid,recovered)
            grid[1,1,1]=7
            state.update(next_file=2,particles=10)
            state=save_checkpoint(root,grid,state,'binding')
            self.assertNotEqual(first,state['slot'])
            recovered,_=load_checkpoint(root,'binding',(4,4,4))
            np.testing.assert_array_equal(grid,recovered)
            with self.assertRaises(ValueError):
                load_checkpoint(root,'changed',(4,4,4))
            np.save(root/f"counts_slot{state['slot']}.npy",np.zeros_like(grid))
            with self.assertRaises(ValueError):
                load_checkpoint(root,'binding',(4,4,4))


if __name__=='__main__':
    unittest.main()
