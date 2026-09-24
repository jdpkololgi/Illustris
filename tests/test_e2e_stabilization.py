import unittest
import numpy as np
import torch
from workflows.sbi.e2e_stabilization import learning_rate,update_ema,offset_scatter


class StabilizationTests(unittest.TestCase):
    def test_schedule(self):
        self.assertAlmostEqual(learning_rate('decay',0),3e-4)
        self.assertAlmostEqual(learning_rate('decay',32768),3e-6)
        self.assertEqual(learning_rate('constant',32768),3e-4)

    def test_ema_does_not_change_student(self):
        a=torch.nn.Linear(1,1,bias=False);b=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():a.weight.fill_(2);b.weight.fill_(0)
        update_ema(b,a,.75)
        self.assertEqual(a.weight.item(),2);self.assertEqual(b.weight.item(),.5)

    def test_offset_scatter_identity(self):
        rows=[dict(case=i,signed_error_posterior_sd=v,mc_standard_error_posterior_sd=.1) for i,v in enumerate([1,2,3,4])]
        for r in offset_scatter(rows):
            self.assertAlmostEqual(r['offset']**2+r['scatter_sd']**2,r['raw_mse'])
            self.assertAlmostEqual(r['mc_corrected_offset_squared']+r['mc_corrected_scatter_squared'],r['raw_mse']-.01)


if __name__=='__main__':unittest.main()
