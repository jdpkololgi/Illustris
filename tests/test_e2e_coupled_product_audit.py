import json
from pathlib import Path
import tempfile
import unittest
import h5py
from workflows.sbi import e2e_coupled_product_audit as audit
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord


class ProductAuditTests(unittest.TestCase):
    def test_content_corruption_and_source_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ph007.txt'; p.write_text('a')
            record=c.file_record(p,content_hash=True); check=audit.Audit('ph007')
            check.file(record); p.write_text('b')
            with self.assertRaises(ValueError): check.file(record)
            record=c.file_record(p); record['sha256']='0'*64
            with self.assertRaises(ValueError): audit.Audit('ph007').file(record)

    def test_incomplete_duplicate_or_cross_directory_pairs_rejected(self):
        root=Path('/tmp/conditions/ph007'); item=dict(path=str(root/'p.json'),sha256='x')
        self.assertEqual(set(audit.pair_index({'pair_receipts':[item]},['p'],root)),{'p'})
        for rows in ([],[item,item]):
            with self.assertRaises(ValueError): audit.pair_index({'pair_receipts':rows},['p'],root)
        with self.assertRaises(PermissionError):
            audit.pair_index({'pair_receipts':[dict(item,path='/tmp/targets/ph007/p.json')]},['p'],root)

    def test_external_target_link_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.h5'
            with h5py.File(path,'w') as f:
                f.attrs.update(schema=audit.targets.SCHEMA,phase='ph007',pair_id='p',role='train',
                    distance_unit='Mpc/h',coordinate_sha256=c.sha256(coord.CONFIG),
                    smoothing_mpc_h=7.,smoothing_count=1,contains_observations=False)
                f['rho_joint']=h5py.ExternalLink('/nonexistent/target.h5','/rho')
                f.create_dataset('coarse_rho_extended',data=[1.]); f.create_dataset('fullbox_tensor_cores',data=[1.])
            with self.assertRaises(PermissionError): audit.read_target(path,'ph007','p')

    def test_sealed_and_mismatched_phase_rejected(self):
        with self.assertRaises(PermissionError): audit.Audit('ph001')
        with self.assertRaises(PermissionError):
            audit.Audit('ph007').file(dict(path='/tmp/ph008/data',bytes=0,sha256='x'))


if __name__=='__main__': unittest.main()
