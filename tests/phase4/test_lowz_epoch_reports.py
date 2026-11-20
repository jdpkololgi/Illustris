import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from workflows.abacus_tweb.lowz_epoch_reports import install, retain_report


def report(epoch):
    phases = ["ph006","ph012","ph013"]
    return dict(validation_phases=phases, complete_phase_coverage=True, per_phase={p:dict(
        complete_phase_coverage=True,validation_phase=p,runtime=dict(
            epoch=epoch,global_step=346364*epoch,patch_failures=0)) for p in phases})


class Reports(unittest.TestCase):
    def test_record_before_selection_without_altering_result(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root/"arm"/"unet"/"seed_42"
            dest.mkdir(parents=True)
            trainer = SimpleNamespace(aggregate_validation_reports=lambda x:x)
            install(trainer, SimpleNamespace(output_root=root,run_name="arm",model="unet",seed=42))
            for epoch in (1,2):
                result = report(epoch), {"loss":2.}
                self.assertIs(trainer.aggregate_validation_reports(result), result)
            for epoch in (1,2):
                self.assertEqual(json.loads((dest/f"validation_reports/epoch_{epoch:04d}.json").read_bytes()),report(epoch))

    def test_mismatched_phase_cannot_replace_valid_epoch(self):
        with TemporaryDirectory() as tmp:
            retain_report(tmp,report(1))
            bad = report(1)
            bad["per_phase"]["ph013"]["runtime"]["epoch"] = 2
            with self.assertRaises(ValueError):
                retain_report(tmp,bad)
            self.assertEqual(json.loads((Path(tmp)/"validation_reports/epoch_0001.json").read_bytes()),report(1))
