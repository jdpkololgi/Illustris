import unittest

from workflows.sbi.e2e_vdm_context_operator_audit import audit


class OperatorAuditTests(unittest.TestCase):
    def test_trace_exactness_does_not_establish_tidal_exactness(self):
        result=audit()
        for row in result['records']:
            self.assertLess(row['density_roundtrip_max_abs'],1e-14)
            self.assertLess(row['trace_error_max_abs'],1e-14)
            self.assertLess(row['commutator_identity_max_abs'],1e-14)
            self.assertLess(row['commutator_trace_max_abs'],1e-14)
        self.assertLess(result['records'][0]['rms_tensor_error_over_reference'],1e-14)
        self.assertLess(result['records'][1]['rms_tensor_error_over_reference'],1e-14)
        self.assertGreater(result['records'][2]['rms_tensor_error_over_reference'],.1)
        self.assertGreater(result['records'][3]['rms_tensor_error_over_reference'],.1)


if __name__=='__main__':
    unittest.main()
