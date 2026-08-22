import unittest

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader


class RandomResponseLoaderTest(unittest.TestCase):
    def test_is_narrow_subclass(self):
        self.assertTrue(issubclass(P10RandomResponseLoader, P10PhaseBalancedLoader))
        self.assertIsNot(
            P10RandomResponseLoader.field_adapter,
            P10PhaseBalancedLoader.field_adapter,
        )


if __name__ == "__main__":
    unittest.main()
