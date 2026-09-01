import json
from pathlib import Path
import tempfile
import unittest

from workflows.abacus_tweb.p3br_prepare_r1_contract import build_ready_marker
from workflows.abacus_tweb.p10_training_contract import sha256


class P3brReadyMarkerTest(unittest.TestCase):
    def test_recovery_output_owns_inventory_and_transform_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            output = root / "mirror_v2"
            base.mkdir()
            (base / "TRAINING_LOADER_READY.json").write_text(
                json.dumps({"schema_version": "base", "pass": True})
            )
            inventory = output / "adapter_inventory.json"
            transform = output / "transforms" / "field" / "field_transform.json"
            transform.parent.mkdir(parents=True)
            inventory.write_text(json.dumps({"schema_version": "p3br-r1-adapter-inventory-v1"}))
            transform.write_text(json.dumps({"schema_version": "p3br-r1-field-transform-v1"}))
            adapters = {
                "ph002": {
                    "path": str(output / "adapters/ph002/field/adapter_manifest.json"),
                    "sha256": "a" * 64,
                    "pass": True,
                }
            }
            ready = build_ready_marker(
                {
                    "adapter_inventory": str(base / "adapter_inventory.json"),
                    "adapter_inventory_sha256": "0" * 64,
                    "pass": True,
                },
                output=output,
                base_contract=base,
                adapters=adapters,
                inventory_path=inventory,
                field={"pass": True},
            )
            self.assertEqual(Path(ready["adapter_inventory"]), inventory)
            self.assertEqual(ready["adapter_inventory_sha256"], sha256(inventory))
            self.assertEqual(Path(ready["field_transform"]), transform)
            self.assertEqual(ready["field_transform_sha256"], sha256(transform))
            self.assertEqual(ready["base_contract"], str(base))
            self.assertFalse(ready["ph001_product_built"])
            self.assertFalse(ready["ph001_opened"])
            self.assertTrue(ready["pass"])


if __name__ == "__main__":
    unittest.main()
