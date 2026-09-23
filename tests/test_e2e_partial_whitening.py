import unittest
import numpy as np
import torch
from workflows.sbi.e2e_spectral_preconditioning import Spectral
from workflows.sbi.e2e_partial_whitening import items,choose,selection_score


class PartialTests(unittest.TestCase):
    def test_design_balance(self):
        panel=items();self.assertEqual(len(panel),40)
        self.assertEqual(sum(i['parent_name'] is not None for i in panel),8)
        self.assertEqual(len({i['name'] for i in panel}),40)
        for rank in range(4):
            owned=[i for k,i in enumerate(panel) if (k//4+k%4)%4==rank]
            self.assertEqual(len(owned),10)
            self.assertGreaterEqual(sum(i['fixed'] is None for i in owned),4)
            self.assertLessEqual(sum(i['fixed'] is None for i in owned),6)

    def test_exponents_and_unchanged_parent(self):
        power=np.arange(1,65,dtype=float).reshape(4,4,4)
        for alpha in [.2,.25,.3,.35,.5]:
            t=Spectral(power,alpha)
            np.testing.assert_allclose(t.scale.numpy(),power**(-alpha),rtol=2e-7)
        torch.testing.assert_close(Spectral(power,.5).scale,torch.tensor(power,dtype=torch.float32).rsqrt(),rtol=0,atol=0)

    def test_selection_ignores_privileged_and_fixed(self):
        rows=[]
        for a in [.2,.25,.3,.35]:
            for seed in [17,29]:
                for case in range(4):
                    rows.append(dict(alpha=a,seed=seed,case=case,nfe=256,exact=False,fixed=None,
                        mean_rms=.08+abs(a-.25),covariance_relative=.1,variance_ratio=1.,octant_coverage=.9,power_ratio=[1.]*7))
        self.assertEqual(choose(rows)['alpha'],.25)
        altered=[r|dict(exact=True,mean_rms=0.) for r in rows]+[r|dict(fixed=0,mean_rms=0.) for r in rows]
        self.assertEqual(choose(rows+altered)['alpha'],.25)
        with self.assertRaises(ValueError):choose(rows[:-1])


if __name__=='__main__':unittest.main()
