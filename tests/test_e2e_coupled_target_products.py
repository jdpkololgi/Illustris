import unittest
import numpy as np
from workflows.sbi import e2e_coupled_target_products as products
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec


class CoupledTargetTests(unittest.TestCase):
    def test_block_alignment_all_context_offsets_and_reference_trace(self):
        rng=np.random.default_rng(17)
        rho=np.exp(rng.normal(0,.2,(64,48,48)))
        wide=np.ones((56,56,56))
        wide[20:36,22:34,22:34]=op.mean_pool(rho)
        qa=products.density_qa(rho,wide)
        self.assertLess(qa['coarse_fine_max_relative'],1e-14)
        tensor=products.owned_core_arrays(op.tensor_from_delta(rho-1,6.766))
        # owned_core_arrays retains any trailing tensor component axis.
        self.assertLess(products.fullbox_qa(rho,tensor)['fullbox_trace_max_abs'],1e-14)
        wide[20,22,22]*=1.01
        with self.assertRaises(ValueError): products.density_qa(rho,wide)
        with self.assertRaises(ValueError): products.fullbox_qa(rho,tensor+.1)

    def test_native_cubic_averages_commute_with_block_mean(self):
        shape=(32,32,32)
        axes=np.meshgrid(*(np.arange(32)*2*np.pi/32 for _ in range(3)),indexing='ij',sparse=True)
        field=.04*(np.cos(axes[0])+np.sin(axes[1])+np.cos(axes[2]))
        grid=coord.grid_record(GridSpec((-233.,343.,126.),(200,200,200),3.383,27.064))
        row=dict(center=[64,80,96],grid=grid)
        starts,average=products.starts_for(row,'joint')
        rho=1+op.sample_averaged_local(field,starts,average,grid)
        starts,average=products.starts_for(row,'wide')
        wide=1+op.sample_averaged_local(field,starts,average,grid)
        qa=products.density_qa(rho,wide)
        self.assertLess(qa['coarse_fine_max_relative'],1e-13)
        cores=products.owned_core_arrays(rho)
        for side,offset in enumerate(([0,0,0],[32,0,0])):
            center=np.array(row['center'])+offset
            direct=1+op.sample_averaged_local(field,[v-16+2*np.arange(16) for v in center],2,grid)
            np.testing.assert_allclose(cores[side],direct,atol=1e-14,rtol=1e-14)

    def test_positive_density_is_not_silently_clipped(self):
        rho=np.ones((64,48,48)); wide=np.ones((56,56,56))
        rho[0,0,0]=0
        with self.assertRaises(ValueError): products.density_qa(rho,wide)


if __name__=='__main__': unittest.main()
