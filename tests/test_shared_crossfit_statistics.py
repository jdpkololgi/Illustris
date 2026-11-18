import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from workflows.sbi import p12_prepare_crossfit_contracts as cross


class CachedStatisticsTests(unittest.TestCase):
    def test_exact_reuse_and_input_drift_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files={k:root/k for k in ('p1_manifest','p3_manifest','cores')}
            for path in files.values():path.write_bytes(b'fixed input')
            files['p4_root']=root/'p4';files['p4_root'].mkdir()
            context=files['p4_root']/'context_assignment.npz';context.write_bytes(b'fixed context')
            bins=len(np.arange(cross.Z_MIN,cross.Z_MAX+.5*cross.BIN_WIDTH,cross.BIN_WIDTH))-1
            result={'counts':np.arange(2*bins,dtype=np.int64).reshape(2,bins),
                    'volume':np.full((2,bins),1.2345678901234567), 'audit':{'label_free':True}}
            with patch.object(cross,'phase_files',return_value=files), patch.object(cross,'phase_selection_statistics',return_value=result) as fit:
                first=cross.cached_phase_selection_statistics(root,'ph000',root/'cache')
                second=cross.cached_phase_selection_statistics(root,'ph000',root/'cache')
                self.assertEqual(fit.call_count,1)
                for key in ('counts','volume'):self.assertTrue(np.array_equal(first[key],second[key]))
                context.write_bytes(b'changed context')
                with self.assertRaisesRegex(ValueError,'source drift'):
                    cross.cached_phase_selection_statistics(root,'ph000',root/'cache')

if __name__=='__main__':unittest.main()
