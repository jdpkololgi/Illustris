import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows/abacus_tweb/p10_freeze_multitracer_epoch15.py"
)
SPEC = importlib.util.spec_from_file_location("p10_freeze_multitracer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FreezeMultitracerEpoch15Test(unittest.TestCase):
    def history(self, count=15):
        return [
            {
                "epoch": epoch,
                "global_step": epoch * MODULE.EXPECTED_CORES,
                "complete_epoch_coverage": True,
                "unique_cores_seen": MODULE.EXPECTED_CORES,
                "repeat_cores": 0,
                "primary_macro_r2_lambda1": 0.5 + epoch / 1000,
                "diagnostic_first_three_shell_macro_r2_lambda1": 0.6,
                "per_shell_lambda1_r2": {"shell": 0.5},
            }
            for epoch in range(1, count + 1)
        ]

    def test_load_history_requires_exact_epoch_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in self.history()) + "\n")
            self.assertEqual(len(MODULE.load_history(path)), 15)

    def test_load_history_refuses_overshoot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in self.history(16)) + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "exactly epochs"):
                MODULE.load_history(path)

    def test_load_history_refuses_incomplete_coverage(self):
        rows = self.history()
        rows[-1]["complete_epoch_coverage"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(RuntimeError, "lacks complete coverage"):
                MODULE.load_history(path)


if __name__ == "__main__":
    unittest.main()
