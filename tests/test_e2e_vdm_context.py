"""Focused invariants; full-size physical and GPU gates are separate."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from workflows.sbi.e2e_vdm_context_data import guarded, spec, admissible, periodic_delta, source_center
from workflows.sbi.e2e_vdm_context_metrics import calibration, fair_energy, central_order_interval
from workflows.sbi.e2e_vdm_context_models import (
    ContextVDM, FineCondition, block_mean, replicate, project, encode_density,
    decode_density, projected_vlb, coupled_sample)
from workflows.sbi.e2e_vdm_context_products import sample_averaged, mean_pool
from workflows.sbi.e2e_field_regenerate_spectral import interpolate
from workflows.sbi.e2e_vdm_context_dataset import context_crop,coarse_local_crop
from workflows.sbi.e2e_vdm_context_physics import composite_tensor,plane_controls
from workflows.sbi.e2e_vdm_context_train import new_model,update_model,checkpoint,restore
from workflows.sbi.e2e_wide_continue import equal_state
from workflows.sbi.e2e_vdm_context_tasks import draw_tasks,draw_seed,coarse_cache_key,task_seed


class GeometryTests(unittest.TestCase):
    def test_full_draw_ledger_and_shared_addresses(self):
        rows=[]
        for phase in ('ph000','ph002','ph003','ph004','ph005'):
            for cap in ('NGC','SGC'):
                for shell in range(4):
                    for support in ('interior','boundary'):
                        rows.append(dict(anchor_id=f'{phase}_{cap}_s{shell}_{support}_00',
                            phase=phase,cap=cap,shell=shell,support_stratum=support,
                            small_train=phase in ('ph000','ph002'),center=[128,128,128]))
        ledger=draw_tasks(rows)
        self.assertEqual((ledger['central_draws'],ledger['coarse_draws']),(33280,7872))
        pairs=[t for t in ledger['tasks'] if t['arm']=='D' and t['purpose']=='joint' and t['replica']==0]
        first,second=pairs[:2]
        self.assertNotEqual(first['anchor'],second['anchor'])
        self.assertEqual(first['domain'],second['domain'])
        self.assertEqual(draw_seed(0,first['domain'],0,'coarse','joint'),
                         draw_seed(0,second['domain'],0,'coarse','joint'))
        self.assertEqual(task_seed(first,0,'coarse'),task_seed(second,0,'coarse'))
        self.assertNotEqual(task_seed(first,0,'fine'),task_seed(second,0,'fine'))
        self.assertEqual(task_seed(first,0,'fine'),task_seed(dict(first,purpose='joint_fixed_mean'),0,'fine'))
        self.assertNotEqual(coarse_cache_key('D',0,20480,first['domain'],0,250,'joint'),
                            coarse_cache_key('D',0,20480,first['domain'],0,500,'joint'))

    def test_roles_and_symlink(self):
        for path in ('/tmp/ph001/density', '/tmp/ph006/observations', '/tmp/ph007'):
            with self.assertRaises(PermissionError):
                guarded(path)
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp)/'safe'
            link.symlink_to('/tmp/ph001')
            with self.assertRaises(PermissionError):
                guarded(link/'data')

    def test_periodic_alias_and_nonoverlap(self):
        c = spec()
        self.assertFalse(admissible([1990, 0, 0], [[10, 0, 0]], c))
        # Euclidean separation can pass while aligned central cubes still overlap.
        self.assertFalse(admissible([100, 100, 100], [[0, 0, 0]], c))
        self.assertTrue(admissible([163, 0, 0], [[0, 0, 0]], c))
        np.testing.assert_allclose(periodic_delta([1990, 0, 0], [10, 0, 0], 2000), [-20, 0, 0])

    def test_center_is_symmetric_boundary(self):
        c = spec()
        out = source_center([8, 8, 8], {'origin_mpc': [0, 0, 0]}, c)
        np.testing.assert_allclose(out, (40*.6766-1000) % 2000)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.z = torch.randn(2, 1, 8, 8, 8)
        self.condition = FineCondition(torch.randn(2, 24, 8, 8, 8),
                                      torch.randn(2, 12, 8, 8, 8), torch.zeros(2, 3))

    def test_mass_roundtrip_and_gauge(self):
        rho = torch.exp(self.z.double())
        coarse, u = encode_density(rho)
        torch.testing.assert_close(decode_density(coarse, u), rho, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(block_mean(u), torch.zeros_like(coarse), atol=1e-15, rtol=0)
        changed = u+replicate(torch.randn_like(coarse))
        torch.testing.assert_close(decode_density(coarse, changed), rho, atol=1e-12, rtol=1e-12)
        extreme = self.z.double()*100
        decoded = decode_density(coarse, extreme)
        self.assertTrue(torch.isfinite(decoded).all())
        torch.testing.assert_close(block_mean(decoded), coarse, atol=1e-12, rtol=1e-12)

    def test_projector(self):
        x = self.z.double()
        p = project(x)
        torch.testing.assert_close(project(p), p, atol=1e-15, rtol=1e-15)
        torch.testing.assert_close((p*(x-p)).sum(), x.new_zeros(()), atol=1e-12, rtol=0)
        draws = project(torch.randn(4096, 1, 4, 4, 4))
        self.assertAlmostEqual(float(draws.square().sum((1,2,3,4)).mean()), 63, delta=.6)

    def test_parameter_matching_and_information_firewall(self):
        counts = []
        for arm in 'ABCD':
            m = ContextVDM(arm, base=8, levels=1)
            counts.append(sum(p.numel() for p in m.parameters()))
        self.assertEqual(len(set(counts)), 1)
        m = ContextVDM('A', base=8, levels=1).eval()
        torch.nn.init.normal_(m.output.weight, std=.01)
        c = self.condition
        spatially_permuted = replace(c, wide=c.wide.flip(-1))
        torch.testing.assert_close(m(self.z, torch.zeros(2), c),
                                   m(self.z, torch.zeros(2), spatially_permuted), atol=1e-6, rtol=1e-6)
        with self.assertRaises(PermissionError):
            m(self.z, torch.zeros(2), replace(c, coarse_local=self.z))

    def test_coarse_inference_provenance_and_gradient(self):
        m = ContextVDM('D', base=8, levels=1)
        c = replace(self.condition, coarse_local=self.z, coarse_wide=self.z,
                    coarse_source='training_truth')
        g = torch.Generator().manual_seed(6)
        value, parts = projected_vlb(m, project(self.z), c, g)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(parts['independent_dimensions'], 504)
        m.eval()
        with self.assertRaises(PermissionError):
            m(self.z, torch.zeros(2), c)
        c = replace(c, coarse_source='sampled')
        y = coupled_sample(m, c, 2, [11, 12], noise_grid=4)
        torch.testing.assert_close(block_mean(y), torch.zeros(2,1,2,2,2), atol=2e-6, rtol=0)
        torch.testing.assert_close(y, coupled_sample(m, c, 2, [11, 12], noise_grid=4), atol=0, rtol=0)

    def test_exact_optimizer_rng_resume_and_corruption(self):
        c=dict(spec(),unet_base=8,unet_levels=1)
        model,opt,gen=new_model(c,0,'C','fine','cpu')
        history=[update_model(model,opt,gen,self.z,self.condition,c,0)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            binding=dict(test='exact-resume')
            saved=checkpoint(root,model,opt,gen,binding,'fine',1,history)
            expected=update_model(model,opt,gen,self.z,self.condition,c,1)
            replay,optimizer,generator=new_model(c,0,'C','fine','cpu')
            step,restored,last=restore(root,binding,replay,optimizer,generator,'fine')
            self.assertEqual((step,restored,last),(1,history,saved))
            actual=update_model(replay,optimizer,generator,self.z,self.condition,c,1)
            self.assertEqual(expected,actual)
            self.assertTrue(equal_state(model.state_dict(),replay.state_dict()))
            self.assertTrue(equal_state(opt.state_dict(),optimizer.state_dict()))
            with (root/saved['path']).open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError):
                restore(root,binding,replay,optimizer,generator,'fine')


class MetricsTests(unittest.TestCase):
    def test_fair_crps_and_scalar_energy(self):
        x = np.array([[1., 2.], [3., 4.], [6., 5.]])
        y = np.array([2., 4.])
        brute = np.abs(x-y).mean(0)-sum(np.abs(x[i]-x[j]) for i in range(3) for j in range(i+1,3))/6
        np.testing.assert_allclose(calibration(x,y)['crps'], brute)
        for j in range(2):
            self.assertAlmostEqual(float(fair_energy(x[:,j,None], y[j,None])), brute[j])

    def test_attainable_gaussian_and_loggaussian(self):
        rng = np.random.default_rng(8)
        samples = rng.normal(size=(65, 8192))
        for x in (samples, np.exp(samples)):
            result = calibration(x[:-1], x[-1])
            self.assertAlmostEqual(float(result['rank'].mean()), .5, delta=.015)
            for level in (.5, .68, .9, .95):
                self.assertAlmostEqual(float(result[f'covered_{level}'].mean()),
                                       result['attainable'][str(level)], delta=.015)
        self.assertEqual(central_order_interval(64,.9), (2,61,59/65))
        self.assertEqual(central_order_interval(128,.95), (2,125,123/129))


class ProductTests(unittest.TestCase):
    def test_context_offsets_and_block_geometry(self):
        self.assertEqual(context_crop((32,0,0)),(slice(8,56),slice(4,52),slice(4,52)))
        self.assertEqual(coarse_local_crop((32,0,0),(0,0,0)),(slice(22,34),slice(18,30),slice(18,30)))
        with self.assertRaises(ValueError):
            context_crop((1,0,0))

    def test_composite_trace_and_dc_transfer(self):
        control=plane_controls()
        self.assertEqual(len(control),5)
        tensor=composite_tensor(np.full((48,)*3,.4),np.full((48,)*3,.4))
        np.testing.assert_allclose(tensor[...,[0,3,5]],.4/3,atol=1e-12)
        rng=np.random.default_rng(33)
        fine,coarse=rng.normal(size=(48,)*3),rng.normal(size=(48,)*3)
        tensor=composite_tensor(fine,coarse)
        np.testing.assert_allclose(tensor[...,[0,3,5]].sum(-1),fine,atol=1e-12)

    def test_averaged_operator_exact_original_interpolation_and_mass(self):
        c = dict(raw_cell_mpc=1.,coordinate_h=1.,box_offset_mpc_h=0.,box_mpc_h=32.)
        grid = dict(origin_mpc=[-.37,.19,1.23])
        field = np.random.default_rng(11).normal(size=(32,32,32))
        starts = [np.arange(-8,8,2)]*3
        fine = sample_averaged(field,starts,2,grid,c)
        coords = [np.mod(np.arange(-8,8)+.5+origin,32) for origin in grid['origin_mpc']]
        direct = mean_pool(interpolate(field,coords,degree=3),2)
        np.testing.assert_allclose(fine,direct,atol=3e-15,rtol=3e-15)
        coarse = sample_averaged(field,[np.array([-8,0])]*3,8,grid,c)
        np.testing.assert_allclose(mean_pool(fine,4),coarse,atol=3e-15,rtol=3e-15)
        np.testing.assert_allclose(sample_averaged(np.ones_like(field),starts,2,grid,c),1,atol=1e-15)


if __name__ == '__main__':
    unittest.main()
