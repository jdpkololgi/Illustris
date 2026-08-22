import unittest

from workflows.abacus_tweb.run_p3br_pipeline import CANARY, PHASES


class P3brPipelineTest(unittest.TestCase):
    def test_blind_phase_is_absent(self):
        self.assertNotIn("ph001", PHASES)
        self.assertEqual(CANARY, ("ph000", "ph006"))


if __name__ == "__main__":
    unittest.main()
