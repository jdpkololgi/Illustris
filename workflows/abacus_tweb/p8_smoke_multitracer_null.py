#!/usr/bin/env python3
"""Verify the neural FAINT null changes only FAINT count-derived channels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p8_train_multitracer_unet_patch import (
    MT,
    MultitracerFieldAdapter,
    model_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--control-product", default="faint_position_null_cic")
    parser.add_argument(
        "--control-manifest",
        type=Path,
        default=MT / "classical/control_fields/bf_proxy_response_v1/manifest.json",
    )
    args = parser.parse_args()
    core = int(
        np.load(
            "/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/"
            f"rotation_{args.rotation}/training_core_id.npy"
        )[0]
    )
    with MultitracerFieldAdapter(
        product="bf_proxy_response_v1", rotation=args.rotation, root=MT
    ) as proxy, MultitracerFieldAdapter(
        product="bf_proxy_response_v1",
        rotation=args.rotation,
        root=MT,
        faint_control_manifest=args.control_manifest,
        faint_control_product=args.control_product,
    ) as null:
        proxy_extracted = proxy.extract(core)
        null_extracted = null.extract(core)
        proxy_bright, proxy_counts, proxy_exposure, proxy_density = proxy_extracted
        null_bright, null_counts, null_exposure, null_density = null_extracted
        assert np.array_equal(proxy_bright.values, null_bright.values)
        assert np.array_equal(
            proxy_bright.authoritative_parent_id,
            null_bright.authoritative_parent_id,
        )
        assert np.array_equal(proxy_exposure, null_exposure)
        assert proxy_counts.shape == null_counts.shape
        assert proxy_density.shape == null_density.shape
        assert np.all(np.isfinite(null_counts)) and np.all(np.isfinite(null_density))
        count_l1 = float(np.sum(np.abs(proxy_counts - null_counts), dtype=np.float64))
        density_l1 = float(np.sum(np.abs(proxy_density - null_density), dtype=np.float64))
        assert count_l1 > 0.0 and density_l1 > 0.0
        _, proxy_values, proxy_points = model_inputs(proxy, proxy_extracted, "cpu")
        _, null_values, null_points = model_inputs(null, null_extracted, "cpu")
        proxy_values = proxy_values.numpy()
        null_values = null_values.numpy()
        assert np.array_equal(proxy_values[:, :3], null_values[:, :3])
        assert np.array_equal(proxy_values[:, 5], null_values[:, 5])
        assert np.array_equal(proxy_points.numpy(), null_points.numpy())
        assert not np.array_equal(proxy_values[:, 3], null_values[:, 3])
        assert not np.array_equal(proxy_values[:, 4], null_values[:, 4])
        print(json.dumps({
            "pass": True,
            "rotation": args.rotation,
            "core_id": core,
            "shape": list(proxy_counts.shape),
            "faint_count_l1": count_l1,
            "faint_density_l1": density_l1,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
