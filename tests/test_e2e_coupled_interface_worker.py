import unittest
from unittest.mock import patch
from workflows.sbi import e2e_coupled_interface_worker as worker
from workflows.sbi import e2e_coupled_data_release as release


class InterfacePublicationTests(unittest.TestCase):
    def test_default_does_not_publish_data_release(self):
        complete=dict(pairs=1792,offset_cases=11776,**{'pass':True})
        with patch.object(worker.qa,'run',return_value=complete), patch.object(release,'run') as publish:
            self.assertEqual(worker.finalize(),0)
            publish.assert_not_called()

    def test_explicit_release_follows_full_interface_success(self):
        complete=dict(pairs=1792,offset_cases=11776,**{'pass':True})
        with patch.object(worker.qa,'run',return_value=complete), patch.object(release,'run',
                return_value={'data_products_qualified':True}) as publish:
            self.assertEqual(worker.finalize(True),0)
            publish.assert_called_once_with()

    def test_failed_interface_cannot_publish(self):
        with patch.object(worker.qa,'run',return_value={'pass':False}), patch.object(release,'run') as publish:
            with self.assertRaises(ValueError): worker.finalize(True)
            publish.assert_not_called()
        with patch.object(worker.qa,'run',side_effect=FileNotFoundError), patch.object(release,'run') as publish:
            with self.assertRaises(FileNotFoundError): worker.finalize(True)
            publish.assert_not_called()


if __name__=='__main__': unittest.main()
