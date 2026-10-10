import unittest
from workflows.sbi.e2e_wholefield_fullcap_parallel import validate_topology


class ParallelTests(unittest.TestCase):
    def test_valid_ranks(self):
        for tasks in (2, 4):
            for rank in range(tasks):
                validate_topology(tasks, tasks, rank, 1)

    def test_reject_wrong_resources(self):
        for args in ((4, 2, 0, 1), (4, 4, 4, 1), (4, 4, -1, 1),
                     (4, 4, 0, 4), (4, 4, 0, 0), (1, 1, 0, 1)):
            with self.assertRaises(ValueError):
                validate_topology(*args)


if __name__ == '__main__':
    unittest.main()
