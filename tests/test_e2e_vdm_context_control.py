import unittest

from workflows.sbi.e2e_vdm_context_control import factors
from workflows.sbi.e2e_vdm_context_smoke import forecast
from workflows.sbi.e2e_vdm_context_tasks import draw_tasks
from workflows.sbi.e2e_vdm_context_data import spec


class ControlTests(unittest.TestCase):
    def test_ten_factors_and_exact_resource_accounting(self):
        self.assertEqual(len(set(factors())),10)
        self.assertEqual(sum(f=='coarse' for _,_,f in factors()),2)
        rows=[]
        for phase in ('ph000','ph002','ph003','ph004','ph005'):
            for cap in ('NGC','SGC'):
                for shell in range(4):
                    for support in ('interior','boundary'):
                        rows.append(dict(anchor_id=f'{phase}_{cap}_s{shell}_{support}_00',phase=phase,
                            cap=cap,shell=shell,support_stratum=support,center=[128]*3,
                            small_train=phase in ('ph000','ph002')))
        timings={key:dict(update_seconds=.18,draw_seconds=5.5) for key in
                 ('A_fine','B_fine','C_fine','D_fine','D_coarse')}
        result=forecast(timings,draw_tasks(rows),spec())
        self.assertAlmostEqual(result['training_gpu_hours'],10.24)
        self.assertAlmostEqual(result['inference_gpu_hours'],41792*5.5/3600)
        self.assertTrue(result['within_ceiling'])
        for value in timings.values():
            value['draw_seconds']=11.
        self.assertFalse(forecast(timings,draw_tasks(rows),spec())['within_ceiling'])


if __name__=='__main__':
    unittest.main()
