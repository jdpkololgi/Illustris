import unittest
from unittest.mock import patch
from pathlib import Path
from workflows.sbi.e2e_wholefield_pilot_train import validate_source_revision, run

class FieldRestartRevisionTests(unittest.TestCase):
    def test_revision_binds_both_parent_and_current_sources(self):
        revision={'sources':{'train.py':{'parent_sha256':'old','sha256':'new'}}}
        validate_source_revision({'train.py':'old','model.py':'same'}, {'train.py':'new','model.py':'same'}, revision)
        with self.assertRaises(ValueError):
            validate_source_revision({'train.py':'wrong'}, {'train.py':'new'}, revision)
        with self.assertRaises(ValueError):
            validate_source_revision({'train.py':'old','model.py':'same'}, {'train.py':'new','model.py':'changed'}, revision)
    def test_revision_cannot_reuse_historical_output_root(self):
        import json
        from workflows.sbi import e2e_wholefield_pilot_train as t
        with patch.object(t,'setup'), patch.object(t,'sources',return_value={'train':'new'}), \
             patch.object(Path,'read_text',side_effect=[json.dumps({'passed':True,'sources':{'train':'old'}}),json.dumps({'sources':{'train':{'parent_sha256':'old','sha256':'new'}}})]):
            with self.assertRaisesRegex(ValueError,'separate output root'):
                run('unet','x',17,1,source_revision=Path('revision.json'))

if __name__ == '__main__': unittest.main()
