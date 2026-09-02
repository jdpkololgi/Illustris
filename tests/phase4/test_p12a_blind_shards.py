from __future__ import annotations

import unittest

import numpy as np

from workflows.sbi.p12a_blind_shards import core_safe_shards


class P12ABlindShardTest(unittest.TestCase):
    def test_four_shards_are_complete_and_do_not_split_cores(self):
        core = np.repeat(np.arange(11), np.arange(1, 12))
        shards = core_safe_shards(core, 4)
        self.assertEqual(len(shards), 4)
        self.assertEqual(shards[0]["start"], 0)
        self.assertEqual(shards[-1]["stop"], len(core))
        self.assertEqual(sum(item["rows"] for item in shards), len(core))
        for left, right in zip(shards[:-1], shards[1:]):
            self.assertEqual(left["stop"], right["start"])
            self.assertNotEqual(core[left["stop"] - 1], core[right["start"]])

    def test_invalid_core_order_or_excess_shards_fails(self):
        with self.assertRaises(ValueError):
            core_safe_shards(np.asarray([0, 2, 1]), 2)
        with self.assertRaises(ValueError):
            core_safe_shards(np.asarray([0, 0, 1, 1]), 3)


if __name__ == "__main__":
    unittest.main()
