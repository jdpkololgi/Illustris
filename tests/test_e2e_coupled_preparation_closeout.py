from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workflows.sbi import e2e_coupled_preparation_closeout as close


class PreparationCloseoutTests(unittest.TestCase):
    def test_missing_data_release_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(close.coord,'ROOT',Path(tmp)):
            with self.assertRaises(FileNotFoundError): close.collect()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_active_or_over_budget_cannot_close(self):
        approval=dict(cpu_node_hours=64,gpu_hours=4,scratch_bytes=100)
        usage=dict(cpu_node_hours=20,gpu_hours=.2,allocations=[dict(state='RUNNING')])
        with self.assertRaises(RuntimeError): close.require_terminal_budget(usage,50,approval)
        usage['allocations'][0]['state']='COMPLETED'
        close.require_terminal_budget(usage,50,approval)
        with self.assertRaises(RuntimeError): close.require_terminal_budget(usage,101,approval)
        usage['gpu_hours']=4.1
        with self.assertRaises(RuntimeError): close.require_terminal_budget(usage,50,approval)

    def test_no_scientific_authority_or_partial_gpu_panel(self):
        data=dict(data_products_qualified=True,scientific_training_authorized=False,posterior_calibration_claim=False)
        technical=dict(synthetic_only=True,scientific_fit=False,cases=list(close.gpu.CASES))
        close.require_no_scientific_authority(data,technical)
        with self.assertRaises(ValueError):
            close.require_no_scientific_authority({**data,'scientific_training_authorized':True},technical)
        with self.assertRaises(ValueError):
            close.require_no_scientific_authority(data,{**technical,'cases':technical['cases'][:-1]})

    def test_archive_refuses_overwrite_and_unregistered_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): close.archive({},[],Path(tmp))
            with self.assertRaises(PermissionError): close.archive({},[Path('/tmp/unregistered.json')],Path(tmp)/'new')
            self.assertFalse((Path(tmp)/'new').exists())


if __name__ == '__main__':
    unittest.main()
