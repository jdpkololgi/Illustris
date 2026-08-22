import unittest

from workflows.abacus_tweb.p10_train_random_response import requested_model


class RandomResponseTrainerTest(unittest.TestCase):
    def test_model_parser(self):
        self.assertEqual(requested_model(["--model", "unet"]), "unet")
        self.assertEqual(requested_model(["--model=graph"]), "graph")
        self.assertIsNone(requested_model([]))


if __name__ == "__main__":
    unittest.main()
