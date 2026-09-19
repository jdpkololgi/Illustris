import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_pairing import successful, validate_successful_join
from workflows.sbi.e2e_coupled_stage_b import parse_listing, MEMBERS, archive


class CoupledContractTests(unittest.TestCase):
    def test_approved_panel(self):
        v = c.config()
        self.assertEqual(len(c.TRAIN), 13)
        self.assertEqual(len(c.DEVELOPMENT), 2)
        self.assertEqual(len(c.CONFIRMATION), 6)
        self.assertFalse(v['scientific_training_authorized'])
        self.assertEqual(v['approval']['scratch_bytes'], int(4.5*2**40))

    def test_phase_guards(self):
        for phase in ('ph001', 'ph004', 'ph005', 'ph006', 'ph025'):
            with self.assertRaises(PermissionError):
                c.phase_guard(phase)
        with self.assertRaises(PermissionError):
            c.guarded('/tmp/ph020/from_ph001')
        with self.assertRaises(PermissionError):
            c.guarded('/tmp/ph020', 'ph021')
        with self.assertRaises(PermissionError):
            c.guarded(c.ROOT/'../escape', output=True)

    def test_symlink_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'innocent'
            path.symlink_to('/tmp/ph001')
            with self.assertRaises(PermissionError):
                c.guarded(path)

    def test_no_login_payloads(self):
        with patch.dict('os.environ', {'SLURM_JOB_ID':'1'}), patch(
                'socket.gethostname', return_value='login21'):
            with self.assertRaises(RuntimeError):
                c.require_compute()

    def test_immutable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'receipt.json'
            c.atomic_json(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                c.atomic_json(path, {'value': 2})
            self.assertEqual(json.loads(path.read_text()), {'value':1})

    def test_exact_hpss_inventory(self):
        names = [f'{d}/{kind}_rv_B_{i:03d}.asdf' for d,kind in zip(MEMBERS,('field','halo'))
                 for i in range(34)] + [f'{d}/checksums.crc32' for d in MEMBERS]
        listing = '\n'.join(f'HTAR: -rw-r--r-- owner/group 123 2020-01-01 00:00 {name}' for name in names)
        self.assertEqual(len(parse_listing(listing)),70)
        with self.assertRaises(ValueError):
            parse_listing('\n'.join(listing.splitlines()[:-1]))
        with self.assertRaises(PermissionError):
            archive('ph001')
        with self.assertRaises(PermissionError):
            archive('ph007')

    def test_successful_and_exact_join(self):
        a = np.zeros(4, dtype=[('TARGETID','i8'),('RA','f8'),('DEC','f8'),
                              ('Z_not4clus','f8'),('ZWARN','i8'),('BGS_TARGET','i8')])
        a['TARGETID'] = np.arange(1,5)
        a['Z_not4clus'] = [.2,.3,np.nan,.4]
        a['ZWARN'][3] = 1
        a['BGS_TARGET'] = 2
        self.assertEqual(successful(a).tolist(), [True,True,False,False])
        b = np.zeros(2,dtype=[('TARGETID','i8'),('RA','f8'),('DEC','f8'),('Z','f4')])
        b['TARGETID'] = [1,2]; b['Z'] = [.2,.3]
        validate_successful_join(a[:2],b)
        b['RA'][0] = 1e-3
        with self.assertRaises(ValueError):
            validate_successful_join(a[:2],b)


if __name__ == '__main__':
    unittest.main()
