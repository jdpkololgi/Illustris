from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from workflows.sbi import e2e_coupled_native_pool as pool


class NativePoolTests(unittest.TestCase):
    def test_serial_reader_initialization_retains_phase_check(self):
        saved=MagicMock(); saved.tree={'header':{'SimName':'AbacusSummit_base_c000_ph024'}}
        with patch('asdf.open') as opened, patch.object(pool.c,'paths',return_value={'snapshot_root':Path('/tmp/ph024')}):
            opened.return_value.__enter__.return_value=saved
            pool.warm_particle_reader('ph024')
            self.assertEqual(opened.call_args.args[0],Path('/tmp/ph024/field_rv_A/field_rv_A_000.asdf'))
            saved.tree['header']['SimName']='AbacusSummit_base_c000_ph023'
            with self.assertRaises(ValueError): pool.warm_particle_reader('ph024')


if __name__=='__main__': unittest.main()
