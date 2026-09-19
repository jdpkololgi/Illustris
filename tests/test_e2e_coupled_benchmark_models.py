from dataclasses import replace
import unittest
import torch
from workflows.sbi import e2e_coupled_benchmark_models as models
from workflows.sbi.e2e_direct_vdm import LinearSchedule
from workflows.sbi.e2e_vdm_context_models import projected_vlb


def condition(device='meta',region='joint',coarse=True):
    make=lambda shape:torch.zeros(shape,device=device)
    return models.FieldCondition(make((1,12,64,48,48)),make((1,12,48,48,48)),
        make((1,3)),region,make((1,1,64,48,48)) if coarse else None,
        make((1,1,48,48,48)) if coarse else None,'training_truth' if coarse else None)


class CoupledBenchmarkModelTests(unittest.TestCase):
    def test_physical_rectangular_feature_positions_match_parent_crops(self):
        offset=torch.tensor([[108.256,0.,0.]],dtype=torch.float64)
        joint=models.physical_positions((8,6,6),models.FINE_CELL,8,offset).reshape(1,8,6,6,3)
        for region,start,shift in [('left',0,-54.128),('right',2,54.128)]:
            parent=models.physical_positions((6,6,6),models.FINE_CELL,8,offset+offset.new_tensor([shift,0,0]))
            torch.testing.assert_close(parent.reshape(1,6,6,6,3),joint[:,start:start+6],rtol=0,atol=1e-12)
        self.assertAlmostEqual(float(joint[0,0,0,0,0]),108.256-433.024/2+3.383)
        wide=models.physical_positions((6,6,6),models.WIDE_CELL,8,torch.zeros_like(offset))
        self.assertAlmostEqual(float(wide[0,0,0]),-1299.072/2+13.532)

    def test_full_size_shapes_and_matched_parent_joint_parameters_without_compute(self):
        counts=[]
        # Meta tensors exercise every real spatial shape without intensive login
        # node convolution. Actual forward/backward timings still require GPU.
        for stage,domain,region in [('coarse','wide','wide'),('fine','parent','left'),('fine','parent','right'),('fine','joint','joint')]:
            with torch.device('meta'):
                model=models.CoupledBackbone(stage,domain)
                cond=condition(region=region,coarse=stage=='fine')
                z=torch.zeros((1,1,*models.SHAPES[region])); result=model(z,torch.zeros(1),cond)
            self.assertEqual(result.shape,z.shape)
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(counts)),1)

    def test_truth_provenance_and_domain_guards(self):
        fine=condition(); z=torch.empty((1,1,64,48,48),device='meta')
        fine.validate(z,'fine','joint',True)
        with self.assertRaises(PermissionError): fine.validate(z,'fine','joint',False)
        replace(fine,coarse_source='sampled').validate(z,'fine','joint',False)
        with self.assertRaises(PermissionError): replace(fine,coarse_source='sampled').validate(z,'fine','joint',True)
        with self.assertRaises(ValueError): fine.validate(z,'fine','parent',True)
        coarse=replace(fine,region='wide'); wide=torch.empty((1,1,48,48,48),device='meta')
        with self.assertRaises(PermissionError): coarse.validate(wide,'coarse','wide',False)

    def test_vdm_objective_matches_previous_projected_vlb(self):
        class Toy(torch.nn.Module):
            stage='fine'; arm='D'
            def __init__(self):
                super().__init__(); self.schedule=LinearSchedule(learned=False)
            def forward(self,z,clock,condition): return .2*z
        model=Toy(); x=models.project(torch.randn(2,1,8,8,8,generator=torch.Generator().manual_seed(2)))
        first,_=models.objective(model,x,None,torch.Generator().manual_seed(3),'vdm')
        second,_=projected_vlb(model,x,None,torch.Generator().manual_seed(3))
        torch.testing.assert_close(first,second,atol=2e-6,rtol=2e-6)

    def test_cfm_zero_target_exact_velocity_has_zero_error(self):
        class Oracle(torch.nn.Module):
            stage='fine'
            def __init__(self):
                super().__init__(); self.schedule=LinearSchedule(learned=False)
            def forward(self,z,clock,condition):
                t=(clock-self.schedule.low)/self.schedule.slope
                return -z/(1-t[:,None,None,None,None])
        model=Oracle(); x=torch.zeros(2,1,8,8,8)
        value,terms=models.objective(model,x,None,torch.Generator().manual_seed(17),'cfm')
        self.assertLess(float(value),1e-10)
        self.assertEqual(terms['independent_dimensions'],504)

    def test_heun_constant_velocity_exact_endpoint_and_seed_replay(self):
        class Oracle(torch.nn.Module):
            stage='coarse'; domain='wide'
            def __init__(self):
                super().__init__(); self.schedule=LinearSchedule(learned=False)
            def forward(self,z,clock,condition): return torch.ones_like(z)
        model=Oracle(); cond=condition(device='cpu',region='wide',coarse=False)
        actual=models.sample(model,cond,'cfm',4,[23])
        expected=torch.randn(actual.shape,generator=torch.Generator().manual_seed(23))+1
        torch.testing.assert_close(actual,expected,atol=5e-7,rtol=5e-7)
        torch.testing.assert_close(actual,models.sample(model,cond,'cfm',4,[23]),atol=0,rtol=0)
        self.assertTrue(model.training)


if __name__=='__main__': unittest.main()
