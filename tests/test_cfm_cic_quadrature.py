import unittest
import numpy as np
from workflows.sbi.cfm_cic_quadrature import contract_lattice, project
from workflows.sbi.diagnose_cfm_lowz_cic_response import weighted_cic
from workflows.sbi.prepare_cfm_lowz_local_condition_canary import project_crop
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec


class QuadratureTests(unittest.TestCase):
    def test_exact_scatter_equivalence(self):
        for factor in (2,4):
            shape=np.array([3,4,2]); cell=2.; origin=np.array([3.,5.,7.])
            fine_shape=(shape+2)*factor
            intensity=np.random.default_rng(42).uniform(size=tuple(fine_shape))
            axes=[origin[i]-cell+(np.arange(fine_shape[i])+.5)*cell/factor for i in range(3)]
            xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
            spec=GridSpec(tuple(origin-cell),tuple(shape+2),cell,0.)
            reference=np.zeros(spec.shape)
            weighted_cic(xyz,intensity.ravel()*(cell/factor)**3,spec,reference)
            np.testing.assert_allclose(contract_lattice(intensity,factor,cell),reference[1:-1,1:-1,1:-1],rtol=2e-14,atol=1e-13)

    def test_point_selection_projection_equivalence(self):
        size=12*256**2
        angular=dict(support=np.ones(size,bool),domain=np.zeros(size,np.uint8),angular_response=np.ones(size))
        plan=dict(context_redshift=[0.,.6],excluded_sentinel_redshift=[.585,.595])
        selection=dict(schema='lowz-point-selection-candidate-v1',density_unit='(Mpc/h)^-3',
                       bin_edges=[0.,.01,.03,.6],density=[[2.,4.,1.],[2.,4.,1.]],valid=[[True]*3]*2)
        for factor in (2,4):
            args=([20.,20.,20.],(3,3,3),3.383,angular,'SGC',plan,selection,factor)
            np.testing.assert_allclose(project(*args),project_crop(*args),rtol=2e-14,atol=1e-12)


if __name__=='__main__': unittest.main()
