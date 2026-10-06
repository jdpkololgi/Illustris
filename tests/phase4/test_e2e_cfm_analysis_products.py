import unittest
import numpy as np
from workflows.sbi.e2e_cfm_analysis_products import guard,select_arrays,fraction_draws


class AnalysisProductTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.zeros((2,2,32,2,4096),dtype='u1');self.labels[1]=2
        self.counts=np.ones((2,4096),dtype='i4')
        self.summary=dict(original=np.tile([-.2,.1,.4],(32,2,1)),
                          corrected=np.tile([-.3,.1,.5],(32,2,1)),physical_truth=np.zeros((2,3)))
        for mi,method in enumerate(('original','corrected')):
            for ti,th in enumerate((.2,0.)):
                self.summary[f'{method}_fractions_{th}']=fraction_draws(self.labels[mi,ti],self.counts)

    def test_corrected_default_and_consistent_classes(self):
        q=select_arrays(self.summary,self.labels,self.counts)
        np.testing.assert_array_equal(q['eigenvalue_means'],self.summary['corrected'])
        np.testing.assert_array_equal(q['galaxy_fraction_draws'][...,2],1)
        np.testing.assert_array_equal(q['cell_class_probabilities'][...,2],1)

    def test_original_remains_explicit(self):
        q=select_arrays(self.summary,self.labels,self.counts,'original')
        np.testing.assert_array_equal(q['eigenvalue_means'],self.summary['original'])
        np.testing.assert_array_equal(q['galaxy_fraction_draws'][...,0],1)

    def test_physical_trace_guard(self):
        self.summary['corrected']+=1
        with self.assertRaises(AssertionError):select_arrays(self.summary,self.labels,self.counts)

    def test_phase_operator_threshold_guards(self):
        for phase in ('ph001','ph014','ph018','Loa'):
            with self.assertRaises(PermissionError):guard(phase,'corrected',.2)
        for method,th in [('other',.2),('corrected',.1)]:
            with self.assertRaises(ValueError):guard('ph012',method,th)

    def test_empty_weights_fail(self):
        with self.assertRaises(ValueError):fraction_draws(self.labels[0,0],np.zeros_like(self.counts))


if __name__=='__main__':unittest.main()
