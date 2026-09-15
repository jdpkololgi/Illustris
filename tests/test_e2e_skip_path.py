import unittest
import torch
from workflows.sbi.e2e_wide_models import ConditionalFieldNet
from workflows.sbi.e2e_multinoise_models import build, coefficients
from workflows.sbi.e2e_skip_path_test import NearIdentityResidual


class SkipTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(2)

    def test_only_fixed_skip_changes(self):
        base = build('unet_film', ConditionalFieldNet(2, 4, 3, 2))
        model = NearIdentityResidual(base.net, base.tau)
        torch.nn.init.normal_(base.net.base.output.weight, std=.1)
        x = torch.randn(3, 1, 8, 8, 8); c = torch.randn(3, 2, 8, 8, 8)
        t = torch.tensor([0., .03, 1.]); a, b, _ = coefficients(t)
        old = base(x, t, c, c); new = model(x, t, c, c)
        torch.testing.assert_close(new-old, -a*b*x, atol=2e-6, rtol=2e-5)
        self.assertTrue(torch.isfinite(new).all())
        self.assertEqual(set(base.state_dict()), set(model.state_dict()))

    def test_clean_skip_fourth_order_and_v_identity(self):
        t = torch.linspace(0, 1, 101); a, b, d = coefficients(t)
        target = torch.randn(101, 1, 2, 2, 2); noise = torch.randn_like(target); raw = torch.randn_like(target)
        x = a*target+b*noise
        v = -a*b*x-raw/d; clean = a*(1+b*b)*x+(b/d)*raw
        torch.testing.assert_close(clean, a*x-b*v)
        torch.testing.assert_close(a*(1+b*b)*(a*target), (1-b**4)*target, atol=1e-6, rtol=1e-5)
        self.assertLessEqual(float((a*b).max()), .500001)
        torch.testing.assert_close(clean-target, -b*(v-(a*noise-b*target)), atol=2e-6, rtol=2e-5)

    def test_report_rejects_partial(self):
        from workflows.sbi.e2e_skip_path_report import summarize
        with self.assertRaises(ValueError):
            summarize(dict(registration={'config': {}}, complete=False, results=[], baseline=[], fields=[], checkpoints={}), {})


if __name__ == '__main__':
    unittest.main()
