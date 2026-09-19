import unittest
from unittest.mock import patch
import numpy as np
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_condition_products as products


def chart():
    result={key:dict(mean=[0.]*12,std=[2.]*12,channels=list(channels))
            for key,channels in (('joint',products.LOCAL_CHANNELS),('wide',products.WIDE_CHANNELS))}
    result.update(coarse_logrho=dict(mean=[.3],std=[.7]),fine_residual=dict(mean=[0.],std=[.2]))
    for key in ('joint','wide'):
        for index,name in enumerate(result[key]['channels']):
            if name in views.IDENTITY: result[key]['std'][index]=1.
    return result


class CoupledViewTests(unittest.TestCase):
    def test_float32_normalized_roundtrip_all_offsets(self):
        rng=np.random.default_rng(2); coarse=np.ones((16,12,12))
        rho=op.decode(coarse,rng.normal(0,.2,(64,48,48))); wide=np.ones((56,56,56))
        for offset in op.layout()['context_offsets_raw']:
            encoded=views.target_view(rho,wide,'ph007',offset,chart())
            actual=views.decode_view(encoded['coarse_logrho'],encoded['fine_residual'],'ph007',offset,chart())
            np.testing.assert_allclose(actual,rho,rtol=2e-6,atol=0)

    def test_assembly_of_truth_is_exact_and_not_overlap_average(self):
        x=np.arange(64*48*48).reshape(64,48,48)
        left,right=views.independent_parents(x)
        np.testing.assert_array_equal(views.assemble_independent(left,right),x)
        left=np.ones((1,48,48,48)); right=np.full_like(left,3.)
        result=views.assemble_independent(left,right)
        np.testing.assert_array_equal(result[:,:32],1.)
        np.testing.assert_array_equal(result[:,32:],3.)

    def test_generated_assembly_preserves_shared_coarse_block_mass(self):
        rng=np.random.default_rng(3); coarse=np.exp(rng.normal(0,.2,(16,12,12)))
        left=op.decode(coarse[:12],rng.normal(size=(48,48,48)))
        right=op.decode(coarse[4:],rng.normal(size=(48,48,48)))
        assembled=views.assemble_independent(left,right)
        np.testing.assert_allclose(op.mean_pool(assembled),coarse,atol=2e-15,rtol=2e-15)

    def test_shared_coarse_fixes_each_draws_owned_core_and_pair_masses(self):
        # Coupled residuals may change non-block-aligned spatial statistics,
        # but must not receive credit for changing this coarse-only observable.
        rng=np.random.default_rng(732)
        for _ in range(4):
            coarse=np.exp(rng.normal(0,.5,(16,12,12)))
            independent=views.assemble_independent(
                op.decode(coarse[:12],rng.normal(0,2,(48,48,48))),
                op.decode(coarse[4:],rng.normal(0,2,(48,48,48))))
            joint=op.decode(coarse,rng.normal(0,3,(64,48,48)))
            for region in [np.s_[16:32,16:32,16:32],np.s_[32:48,16:32,16:32],
                           np.s_[16:48,16:32,16:32],np.s_[:,:,:]]:
                np.testing.assert_allclose(independent[region].mean(),joint[region].mean(),
                                           rtol=2e-15,atol=2e-15)

    def test_fit_and_augmentation_role_firewalls_precede_io(self):
        with patch.object(views,'load_chart',side_effect=AssertionError('must not open chart')):
            for phase in ('ph012','ph014','ph001'):
                with self.assertRaises(PermissionError): views.load_training_pair(phase,'id','sha')
        with self.assertRaises(PermissionError): views.crop_slices('ph014',(32,0,0))
        with self.assertRaises(ValueError): views.crop_slices('ph007',(8,0,0))

    def test_global_chart_shape_and_zero_mode_are_enforced(self):
        n=chart(); n['fine_residual']['mean']=[.1]
        with self.assertRaises(ValueError): views.validate_chart(n)
        n=chart(); n['wide']['std'][1]=0
        with self.assertRaises(ValueError): views.validate_chart(n)
        n=chart(); n['joint']['std'][1]=2
        with self.assertRaises(ValueError): views.validate_chart(n)


if __name__=='__main__': unittest.main()
