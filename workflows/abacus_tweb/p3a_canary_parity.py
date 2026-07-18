#!/usr/bin/env python3
"""P1a wedge parity canary for canonical P3a field primitives."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitsio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from workflows.abacus_tweb.p3a_build_canonical_fields import (
    cic_deposit,
    coordinate_block,
    fractional_index,
    grid_from_xyz,
    log_count_ratio,
)
from workflows.sbi.gate_c_unet_fullrange import (
    DEC_MAX,
    DEC_MIN,
    RA_MAX,
    RA_MIN,
    WedgeGrid,
    make_grid_coords,
    radial_nbar,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalogue", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p1_canonical/"
                     "ph000_path1_wedge/canonical_catalogue.fits"),
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/"
                     "p1a_canary_parity.json"),
    )
    ap.add_argument("--cell-mpc", type=float, default=5.0)
    ap.add_argument("--padding-mpc", type=float, default=40.0)
    ap.add_argument("--sample", type=int, default=8192)
    args = ap.parse_args()

    table = fitsio.read(
        str(args.catalogue),
        columns=["X", "Y", "Z_CART", "RA", "DEC", "Z"],
    )
    xyz = np.column_stack([table["X"], table["Y"], table["Z_CART"]]).astype(np.float64)
    reference = WedgeGrid(xyz, cell=args.cell_mpc, pad=args.padding_mpc)
    spec = grid_from_xyz(xyz, args.cell_mpc, args.padding_mpc)

    counts_reference = reference.cic_deposit(xyz)
    counts_new, deposit = cic_deposit(xyz, spec)
    counts_max_abs = float(np.max(np.abs(counts_reference - counts_new)))
    counts_sum_abs = float(abs(np.sum(counts_new, dtype=np.float64) - len(xyz)))

    frac_reference = reference.frac_index(xyz)
    frac_new = fractional_index(xyz, spec)
    frac_max_abs = float(np.max(np.abs(frac_reference - frac_new)))

    gx, gy, gz = coordinate_block(
        spec, (slice(0, spec.shape[0]), slice(0, spec.shape[1]), slice(0, spec.shape[2]))
    )
    shape = spec.shape
    xx = np.broadcast_to(gx, shape)
    yy = np.broadcast_to(gy, shape)
    zz = np.broadcast_to(gz, shape)
    radius = np.maximum(np.sqrt(xx * xx + yy * yy + zz * zz), 1e-12)
    los_new = np.stack([xx / radius, yy / radius, zz / radius], axis=0).astype(np.float32)
    los_reference = reference.los_hat().astype(np.float32)
    los_max_abs = float(np.max(np.abs(los_reference - los_new)))

    r_gal = np.linalg.norm(xyz, axis=1)
    omega = np.radians(RA_MAX - RA_MIN) * (
        np.sin(np.radians(DEC_MAX)) - np.sin(np.radians(DEC_MIN))
    )
    r_centers, nbar = radial_nbar(r_gal, omega)
    mask = reference.survey_mask(r_gal)
    mu_reference = reference.expected_counts(r_centers, nbar, mask)
    mu_new = (
        np.interp(reference.radius(), r_centers, nbar, left=0.0, right=0.0)
        * mask * args.cell_mpc ** 3
    ).astype(np.float32)
    mu_max_abs = float(np.max(np.abs(mu_reference - mu_new)))

    contrast_new = log_count_ratio(counts_new, mu_new, mask, 1e-3, 1e-4)
    selected = mask > 1e-4
    contrast_reference = np.zeros_like(mu_new, dtype=np.float32)
    contrast_reference[selected] = np.log(
        (counts_reference[selected].astype(np.float64) + 1e-3)
        / (mu_reference[selected].astype(np.float64) + 1e-3)
    ).astype(np.float32)
    contrast_max_abs = float(np.max(np.abs(contrast_reference - contrast_new)))

    rng = np.random.default_rng(42)
    sample = rng.choice(len(xyz), size=min(args.sample, len(xyz)), replace=False)
    frac_sample = frac_new[sample]
    grid_pts = make_grid_coords(frac_sample, spec.shape)
    with torch.no_grad():
        volume = torch.from_numpy(counts_new[None, None])
        torch_sample = F.grid_sample(
            volume, grid_pts, mode="bilinear", align_corners=True, padding_mode="border"
        )[0, 0, 0, 0].numpy()
    scipy_sample = map_coordinates(counts_new, frac_sample.T, order=1, mode="nearest")
    interpolation_corr = float(np.corrcoef(torch_sample, scipy_sample)[0, 1])

    gates = {
        "cic_exact": counts_max_abs == 0.0,
        "cic_conserved": counts_sum_abs < 1e-3 and deposit["lost_weight"] == 0.0,
        "fractional_index_exact": frac_max_abs == 0.0,
        "los_exact": los_max_abs < 1e-6,
        "expected_count_exact": mu_max_abs < 1e-6,
        "contrast_exact": contrast_max_abs < 1e-6,
        "interpolation_axis_corr": interpolation_corr > 0.99999,
    }
    payload = {
        "catalogue": str(args.catalogue),
        "rows": int(len(xyz)),
        "grid": spec.as_dict(),
        "metrics": {
            "counts_max_abs": counts_max_abs,
            "counts_sum_abs": counts_sum_abs,
            "fractional_index_max_abs": frac_max_abs,
            "los_max_abs": los_max_abs,
            "expected_counts_max_abs": mu_max_abs,
            "contrast_max_abs": contrast_max_abs,
            "interpolation_corr": interpolation_corr,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise RuntimeError(f"P3a P1a parity failed: {gates}")


if __name__ == "__main__":
    main()
