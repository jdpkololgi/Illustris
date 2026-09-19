import unittest
from contextlib import nullcontext
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from workflows.sbi import e2e_coupled_normalization as norm
from workflows.sbi import e2e_coupled_contract as c


class CoupledNormalizationTests(unittest.TestCase):
    def reports(self):
        reports={}
        for index,phase in enumerate(c.TRAIN):
            # Deliberately different presentation counts must NOT reweight phases.
            reports[phase]={}
            for name,n in (('joint',12),('wide',12),('coarse_logrho',1),('fine_residual',1)):
                mean=0 if name=='fine_residual' else index
                reports[phase][name]=dict(mean=[mean]*n,second_moment=[mean**2+4]*n,
                                          voxels_per_channel=100*(index+1))
        return reports

    def test_equal_phase_global_not_pooled_sample_weight(self):
        result=norm.equal_phase_statistics(self.reports())
        self.assertEqual(result['joint']['mean'][0],6)
        self.assertAlmostEqual(result['joint']['std'][0],np.sqrt(18))
        self.assertEqual(result['joint']['mean'][1],0)
        self.assertEqual(result['joint']['std'][1],1)
        self.assertEqual(result['fine_residual'],dict(mean=[0.],std=[2.]))

    def test_exact_training_panel_required(self):
        reports=self.reports(); del reports['ph024']
        with self.assertRaises(PermissionError): norm.equal_phase_statistics(reports)
        reports['ph014']=reports['ph007']
        with self.assertRaises(PermissionError): norm.equal_phase_statistics(reports)

    def test_nonzero_residual_mean_rejected(self):
        reports=self.reports(); reports['ph007']['fine_residual']['mean']=[.1]
        with self.assertRaises(ValueError): norm.equal_phase_statistics(reports)

    def test_heldout_statistics_rejected_before_io(self):
        with patch.object(norm.coord,'verify_receipt',side_effect=AssertionError('unexpected IO')):
            for phase in ('ph001','ph006','ph012','ph014','ph004'):
                with self.assertRaises(PermissionError): norm.phase_statistics(phase,Path('/unused'),{})

    def test_incremental_collection_cannot_publish_partial_normalizer(self):
        from workflows.sbi import e2e_coupled_audit_worker as audits
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(norm.coord,'ROOT',Path(tmp)), \
                patch.object(c,'require_compute'), patch.object(norm.coord,'bind'), \
                patch.object(norm.coord,'require_host_checks'), \
                patch.object(c,'single_writer',return_value=nullcontext()), \
                patch.object(norm,'source_binding',return_value={}), \
                patch.object(audits,'qualified',side_effect=lambda p:p=='ph007'), \
                patch.object(norm,'phase_statistics',return_value=(self.reports()['ph007'],[])) as collect, \
                patch.object(c,'atomic_json',side_effect=AssertionError('partial publication')):
            result=norm.fit(collect_ready=True)
            self.assertFalse(result['complete'])
            self.assertEqual(result['collected_phases'],['ph007'])
            self.assertEqual(len(result['pending_phases']),12)
            self.assertEqual(collect.call_args.args[0],'ph007')


if __name__=='__main__': unittest.main()
