import unittest

from workflows.abacus_tweb.p3br_run_r1_throughput_canary import validate_canary


class TestP3brR1Canary(unittest.TestCase):
    def test_exact_update_contract(self):
        validate_canary({"pass": True, "global_step": 1000, "cursor": 1000}, expected_updates=1000)

    def test_wrong_update_count_fails(self):
        with self.assertRaisesRegex(RuntimeError, "rather than 1000"):
            validate_canary({"pass": True, "global_step": 999, "cursor": 999}, expected_updates=1000)


if __name__ == "__main__":
    unittest.main()
