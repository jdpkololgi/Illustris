"""Non-destructive archive and consumer guard invariants."""
import copy
import sys
from pathlib import Path
import unittest
import tempfile
import json
import fitsio
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflows/abacus_tweb'))
from product_lifecycle import archive_in_place,lookup,require_current,verify_catalogue_identity,reject_historical_input
from audit_product_lifecycle import refs

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.registry={'units':[{'path':'/products/old','status':'quarantine_legacy_target'},
          {'path':'/products/native','status':'retain_native_or_direct_field'},
          {'path':'/products/new','status':'corrected_replacement'}]}
    def test_archiving_changes_registry_not_paths_or_original_snapshot(self):
        original=copy.deepcopy(self.registry)
        r=archive_in_place(self.registry,['/products/old'])
        self.assertEqual(self.registry,original)
        self.assertEqual(r['units'][0]['path'],'/products/old')
        self.assertEqual(r['units'][0]['status'],'archived_in_place')
        self.assertFalse(r['physical_actions_performed'])
    def test_native_fields_and_unreviewed_subpaths_cannot_be_archived(self):
        for p in ['/products/native','/products/old/subfile']:
            with self.assertRaises(ValueError):archive_in_place(self.registry,[p])
    def test_old_and_unknown_products_are_blocked_for_new_science(self):
        for p in ['/products/old/file','/products/native/file','/products/absent']:
            with self.assertRaises(ValueError):require_current(self.registry,p)
        self.assertEqual(require_current(self.registry,'/products/new/file')['status'],'corrected_replacement')
    def test_default_build_guard_blocks_registered_history_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'registry.json';p.write_text(json.dumps(self.registry))
            with self.assertRaises(ValueError):reject_historical_input('/products/old/catalogue.fits',p)
            reject_historical_input('/products/new/catalogue.fits',p)
            reject_historical_input('/products/unknown/catalogue.fits',p)

    def test_replacement_rejects_reordered_galaxies(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=np.zeros(2,dtype=[('RA','f8'),('DEC','f8'),('Z','f8')]);a['RA']=[10,20];a['Z']=.2
            old=Path(tmp)/'old.fits';new=Path(tmp)/'new.fits'
            fitsio.write(old,a);fitsio.write(new,a)
            self.assertTrue(verify_catalogue_identity(old,new)['exact'])
            fitsio.write(new,a[::-1],clobber=True)
            with self.assertRaises(ValueError):verify_catalogue_identity(old,new)

    def test_output_and_code_paths_are_not_target_dependencies(self):
        x={'input':{'path':'/pscratch/input.fits'},'outputs':[{'path':'/pscratch/output.npz'}],
           'sources':{'path':'/global/project/build.py'}}
        self.assertEqual([r['direction'] for r in refs(x)],['input','output','code'])
if __name__=='__main__':unittest.main()
