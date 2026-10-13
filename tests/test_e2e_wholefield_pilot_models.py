import unittest
import torch
from workflows.sbi.e2e_wholefield_pilot_models import patchify, unpatchify, Condition
from workflows.sbi.e2e_wholefield_pilot_train import selected, bridge, arrays, STEPS, EPOCH_STEPS
from workflows.sbi.e2e_vdm_context_models import block_mean, project


class WholefieldTests(unittest.TestCase):
    def test_rectangular_tokens(self):
        x = torch.arange(2*3*8*12*16).reshape(2, 3, 8, 12, 16)
        self.assertTrue(torch.equal(unpatchify(patchify(x), (8, 12, 16)), x))

    def test_training_exposure(self):
        ids = list(range(624))
        for epoch in (0, 31):
            order = sum((selected(ids, 17, s) for s in range(epoch*312, (epoch+1)*312)), [])
            self.assertEqual(sorted(order), ids)
        self.assertEqual(STEPS, 32*EPOCH_STEPS)
        self.assertEqual(624*96*64*64, 1664*64*48*48)

    def test_addressed_noise(self):
        x = project(torch.randn(2, 1, 8, 8, 8))
        a = bridge(x, 17, 3); torch.rand(100); b = bridge(x, 17, 3)
        for left, right in zip(a, b):
            self.assertTrue(torch.equal(left, right))
        self.assertLess(float(block_mean(a[0]).abs().max()), 1e-6)

    def test_target_guards_before_io(self):
        for phase, training in [('ph018', True), ('ph012', True), ('ph014', False), ('ph000', False)]:
            with self.assertRaises(PermissionError):
                arrays((phase, 'nonexistent'), {}, training)

    def test_condition_guard(self):
        # Expanded views validate dimensions without allocating the full arrays.
        a = torch.zeros(1)
        cond = Condition(a.expand(1, 12, 96, 64, 64), a.expand(1, 12, 48, 48, 48),
                         a.expand(1, 1, 48, 48, 48), 'training_truth')
        cond.validate(True)
        with self.assertRaises(PermissionError):
            cond.validate(False)

    def test_owned_core_coordinates(self):
        from workflows.sbi.e2e_wholefield_pilot_evaluate import CORE
        from workflows.sbi.e2e_wholefield_pilot_data import OLD_CROP
        from workflows.sbi.e2e_coupled_operators import layout
        for side, core in enumerate(layout()['owned_core_crops_in_joint']):
            for axis, (left, right) in enumerate(core):
                shift = 16*side if axis == 0 else 0
                self.assertEqual(OLD_CROP[axis].start+left, CORE[axis].start+shift)
                self.assertEqual(right-left, 16)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
