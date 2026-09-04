import unittest

import torch

from workflows.sbi.p12f3_train_conditional_gaussian import restore_rng_state, split_selected, shuffle_seed


class ConditionalGaussianTrainingTests(unittest.TestCase):
    def test_cpu_rng_round_trip(self):
        original = torch.get_rng_state()
        torch.manual_seed(17)
        frozen = torch.get_rng_state().clone()
        _ = torch.rand(5)
        restore_rng_state({"torch_rng": frozen, "cuda_rng": []})
        first = torch.rand(5)
        restore_rng_state({"torch_rng": frozen, "cuda_rng": []})
        second = torch.rand(5)
        torch.set_rng_state(original)
        torch.testing.assert_close(first, second)

    def test_split_is_complete_disjoint_and_deterministic(self):
        selected = {"ph000": list(range(100)), "ph002": list(range(100, 200))}
        first = split_selected(selected, ("ph000", "ph002"), 0.1, 42)
        second = split_selected(selected, ("ph000", "ph002"), 0.1, 42)
        self.assertEqual(first, second)
        for phase in selected:
            train, validation = set(first[0][phase]), set(first[1][phase])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(selected[phase]))
            self.assertEqual(len(validation), 10)

    def test_shuffle_seed_is_stable_and_phase_specific(self):
        self.assertEqual(shuffle_seed(42, "ph002", 7), shuffle_seed(42, "ph002", 7))
        self.assertNotEqual(shuffle_seed(42, "ph002", 7), shuffle_seed(42, "ph003", 7))


if __name__ == "__main__":
    unittest.main()
