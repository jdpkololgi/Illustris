import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np

from workflows.sbi.e2e_vdm_context_operator_audit import audit
from workflows.sbi.e2e_vdm_context_physics import tensor_controls,composite_tensor,consistent_tensor,repeat_spatial
from workflows.sbi.e2e_field_build_products import tensor_from_delta
from workflows.sbi import e2e_vdm_context_physics as physics
from workflows.sbi.e2e_durable import publish_json


class OperatorAuditTests(unittest.TestCase):
    def test_release_binds_failed_gate_new_gate_source_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'data').mkdir()
            publish_json(root/'data/REPRESENTATION_GATE.json',dict(training_launch_allowed=False))
            failed=physics.sha256(root/'data/REPRESENTATION_GATE.json')
            publish_json(root/'data/GEOMETRY.json',{})
            publish_json(root/'data/NORMALIZATION.json',{})
            publish_json(root/'PHYSICS_V2_SOURCE.json',dict(source_sha256={
                'workflows/sbi/e2e_vdm_context_physics.py':physics.sha256(physics.__file__)}))
            gate=dict(operator=physics.OPERATOR,scientific_representation_pass=True,
                training_launch_allowed=True,evaluator_sha256=physics.sha256(physics.__file__),
                config_sha256=physics.sha256(physics.CONFIG),
                geometry_sha256=physics.sha256(root/'data/GEOMETRY.json'),
                source_receipt_sha256=physics.sha256(root/'PHYSICS_V2_SOURCE.json'),
                failed_v1_sha256=failed,records=[{}]*32,heldout_used=False,
                median_per_anchor_relative_rmse_reduction=[.4,.5,.6])
            publish_json(root/'data/REPRESENTATION_GATE_V2.json',gate)
            paths=['data/REPRESENTATION_GATE.json','data/REPRESENTATION_GATE_V2.json',
                   'data/NORMALIZATION.json','PHYSICS_V2_SOURCE.json']
            release=dict(operator=physics.OPERATOR,representation_pass=True,
                         receipts={p:physics.sha256(root/p) for p in paths})
            publish_json(root/'data/REPRESENTATION_RELEASE.json',release)
            with patch.object(physics,'FAILED_V1_SHA256',failed):
                self.assertEqual(physics.verify_representation_release(root),release)
                with self.assertRaises(FileExistsError):
                    publish_json(root/'data/REPRESENTATION_RELEASE.json',release)
                publish_json(root/'data/NORMALIZATION.json',dict(drift=True),replace=True)
                with self.assertRaises(ValueError):
                    physics.verify_representation_release(root)

    def test_consistent_density_recovers_all_matched_domain_modes(self):
        for row in tensor_controls():
            self.assertLess(row['rms_tensor_error_over_reference'],1e-12)

    def test_shared_density_crop_alignment(self):
        coarse=np.random.default_rng(8).normal(size=(4,)*3)
        wide=tensor_from_delta(repeat_spatial(coarse),6.766)
        for start in (0,1,2):
            crop=(slice(start,start+2),slice(1,3),slice(0,2))
            fine=repeat_spatial(coarse[crop])
            expected=wide[start*4:(start+2)*4,4:12,0:8]
            np.testing.assert_allclose(consistent_tensor(fine,coarse,crop,6.766),expected,atol=1e-12)
        with patch('workflows.sbi.e2e_vdm_context_physics.consistent_tensor') as helper:
            composite_tensor(np.zeros((48,)*3),np.zeros((48,)*3),(32,-32,0))
            self.assertEqual(helper.call_args.args[2],(slice(22,34),slice(14,26),slice(18,30)))

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
