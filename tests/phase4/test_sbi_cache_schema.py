import unittest

from shared.sbi_cache_schema import get_scaler_key, validate_sbi_cache_payload


class TestSbiCacheSchema(unittest.TestCase):
    def test_accepts_target_scaler_schema(self) -> None:
        payload = {
            "graph": object(),
            "regression_targets": object(),
            "masks": (object(), object(), object()),
            "target_scaler": object(),
        }
        validate_sbi_cache_payload(payload)
        self.assertEqual(get_scaler_key(payload), "target_scaler")

    def test_accepts_legacy_eigenvalue_scaler_schema(self) -> None:
        payload = {
            "graph": object(),
            "regression_targets": object(),
            "masks": (object(), object(), object()),
            "eigenvalue_scaler": object(),
        }
        validate_sbi_cache_payload(payload)
        self.assertEqual(get_scaler_key(payload), "eigenvalue_scaler")

    def test_rejects_missing_required_keys(self) -> None:
        payload = {"graph": object(), "target_scaler": object()}
        with self.assertRaises(ValueError):
            validate_sbi_cache_payload(payload)

    def test_rejects_invalid_masks_shape(self) -> None:
        payload = {
            "graph": object(),
            "regression_targets": object(),
            "masks": (object(), object()),
            "target_scaler": object(),
        }
        with self.assertRaises(ValueError):
            validate_sbi_cache_payload(payload)


if __name__ == "__main__":
    unittest.main()
