#!/usr/bin/env python3
"""Label-free trained-model deployment convergence for P6 and P7."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[2]
P6_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter")
SELECTION = P6_ROOT / "fullcap_selection_v1/selection_manifest.json"
P6_OUT = P6_ROOT / "trained_convergence_v1"
P7_OUT = Path("/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter/trained_convergence_v1")
CHECKPOINT = Path("/pscratch/sd/d/dkololgi/abacus/field_level_tests/T2_transfer/t2_model_seed42.pt")
CELL_MPC = 5.0
RSMOOTH_MPC = 10.345846881466155
ALIGNMENT = 8
SHELL_EDGES = np.asarray([0.15, 0.25, 0.35, 0.45, 0.55])
P6_HALOS = (8, 16, 24, 32, 48, 64, 80)
P6_GATES = {
    "prediction_nrmse": 0.02, "prediction_p95": 0.08,
    "latent_nrmse": 0.02, "worst_core_nrmse": 0.04,
    "boundary_abs_spearman": 0.20, "boundary_trivial_nrmse": 0.002,
    "subdivision_nrmse": 0.02, "subdivision_p95": 0.08,
    "support_distance_mpc": 2.0 * RSMOOTH_MPC,
}
P7_CONFIGS = (
    {"name": "h32_p8_a8", "halo": 32, "padding": 8, "apodization": 8, "eligible": True},
    {"name": "h48_p12_a12", "halo": 48, "padding": 12, "apodization": 12, "eligible": True},
    {"name": "h64_p16_a16", "halo": 64, "padding": 16, "apodization": 16, "eligible": True},
    {"name": "h72_p20_a20", "halo": 72, "padding": 20, "apodization": 20, "eligible": True},
    {"name": "h80_p16_a20_padding_control", "halo": 80, "padding": 16, "apodization": 20, "eligible": False},
    {"name": "h80_p24_a16_apod_control", "halo": 80, "padding": 24, "apodization": 16, "eligible": False},
    {"name": "h80_p24_a20_reference", "halo": 80, "padding": 24, "apodization": 20, "eligible": False},
)
P7_GATES = {
    "density_nrmse": 0.02, "tensor_nrmse": 0.03,
    "eigenvalue_nrmse": 0.03, "eigenvalue_p95": 0.08,
    "large_gap_median_angle_deg": 5.0,
    "large_gap_p95_angle_deg": 15.0,
    "trace_max_abs_error": 2.0e-10,
    "boundary_abs_spearman": 0.20,
    "boundary_trivial_eigenvalue_nrmse": 0.02,
    "support_distance_mpc": 2.0 * RSMOOTH_MPC,
}


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p6 = import_file("p6_conv", REPO / "workflows/abacus_tweb/p6_field_patch_utils.py")
p7 = import_file("p7_conv", REPO / "workflows/abacus_tweb/p7_ftier_patch_utils.py")
t2 = import_file("t2_conv", REPO / "workflows/sbi/gate_t2_cnn_counts.py")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def rank_corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    rx, ry = np.empty(len(x)), np.empty(len(y))
    rx[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    ry[np.argsort(y, kind="mergesort")] = np.arange(len(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def compare(got, ref):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    diff = got - ref
    scale = max(float(np.std(ref)), 1e-6)
    absolute = np.abs(diff).ravel()
    return {
        "n": int(diff.size), "reference_std": scale,
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "nrmse": float(np.sqrt(np.mean(diff ** 2)) / scale),
        "p95_abs_over_std": float(np.quantile(absolute, 0.95) / scale),
        "max_abs_over_std": float(np.max(absolute) / scale),
    }


class ChannelLayerNorm3d(torch.nn.Module):
    """Per-voxel channel normalization with no spatial dependency."""

    def __init__(self, source):
        super().__init__()
        self.eps = float(source.eps)
        self.weight = torch.nn.Parameter(source.weight.detach().clone())
        self.bias = torch.nn.Parameter(source.bias.detach().clone())

    def forward(self, values):
        mean = values.mean(dim=1, keepdim=True)
        variance = values.var(dim=1, keepdim=True, unbiased=False)
        normalized = (values - mean) * torch.rsqrt(variance + self.eps)
        shape = (1, -1) + (1,) * (values.ndim - 2)
        return normalized * self.weight.view(shape) + self.bias.view(shape)


def replace_spatial_groupnorm(module):
    """Replace patch-global GroupNorm while preserving learned affine terms."""
    replaced = []
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.GroupNorm):
            setattr(module, name, ChannelLayerNorm3d(child))
            replaced.append(name)
        else:
            replaced.extend(
                f"{name}.{item}" for item in replace_spatial_groupnorm(child)
            )
    return replaced

def load_model(path, device):
    try:
        saved = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location=device)
    model = t2.CNNCountsModel(in_ch=3, lat_ch=32, base=24).to(device)
    model.load_state_dict(saved["state_dict"])
    replaced = replace_spatial_groupnorm(model)
    if len(replaced) != 14:
        raise RuntimeError(f"expected 14 GroupNorm layers, replaced {len(replaced)}")
    model.eval()
    return model, np.asarray(saved["tmu"]), np.asarray(saved["tsd"]), saved, replaced



def predict(model, patch, normalization, tmu, tsd, device):
    normalized = p6.apply_frozen_normalization(patch, normalization)
    at = {name: i for i, name in enumerate(patch.channel_names)}
    inputs = np.stack([
        normalized[at["counts"]],
        np.clip(np.expm1(np.clip(patch.values[at["log_count_ratio"]], -20, 4)), -1, 20),
        patch.values[at["exposure_apodized"]],
    ]).astype(np.float32)
    values = torch.from_numpy(inputs[None]).to(device)
    points = t2.make_grid_coords(
        patch.authoritative_frac_index_local, values.shape[2:]
    ).to(device)
    with torch.no_grad():
        latent = model.unet(values)
        sampled = F.grid_sample(
            latent, points, mode="bilinear", align_corners=True,
            padding_mode="border",
        )[0, :, 0, 0].T
        output = model.head(sampled).cpu().numpy() * tsd + tmu
    return output.astype(np.float32), latent[0, 0].float().cpu().numpy()


def support_distance(patch):
    mask = patch.values[patch.channel_names.index("exposure_binary")] > 0.05
    distance = distance_transform_edt(
        np.pad(mask, 1, constant_values=False)
    )[1:-1, 1:-1, 1:-1] * CELL_MPC
    return p6.trilinear_sample(distance, patch.authoritative_frac_index_local)


def select_cores(adapter, selection, required_halo):
    radius = np.asarray(selection["cosmology"]["radius_grid_mpc"])
    redshift = np.asarray(selection["cosmology"]["redshift_grid"])
    shapes = {cap: np.asarray(adapter._handle(cap)["counts"].shape) for cap in (0, 1)}
    best = {}
    for core in range(len(adapter.core_fold)):
        cap = int(adapter.core_cap[core])
        start, stop = np.asarray(adapter.core_start[core]), np.asarray(adapter.core_stop[core])
        margin = int(np.min(np.r_[start, shapes[cap] - stop]))
        if margin < required_halo + ALIGNMENT:
            continue
        n = int(adapter.core_offsets[core + 1] - adapter.core_offsets[core])
        if n < 5:
            continue
        grid = adapter.manifest["caps"][p6.CAP_NAME[cap]]
        frac = 0.5 * (start + stop - 1)
        xyz = np.asarray(grid["origin_mpc"]) + (frac + 0.5) * grid["cell_mpc"]
        z = float(np.interp(np.linalg.norm(xyz), radius, redshift))
        shell = int(np.searchsorted(SHELL_EDGES, z, side="right") - 1)
        if 0 <= shell < 4 and ((cap, shell) not in best or n > best[(cap, shell)][1]):
            best[(cap, shell)] = (core, n, z, margin)
    missing = [(cap, shell) for cap in (0, 1) for shell in range(4)
               if (cap, shell) not in best]
    if missing:
        raise RuntimeError(f"no 80-voxel reference core for strata {missing}")
    return [
        {"core_id": v[0], "cap": k[0], "shell": k[1],
         "n_authoritative": v[1], "centre_redshift": v[2],
         "lattice_margin_voxels": v[3]}
        for k, v in sorted(best.items())
    ]


def point_eigensystem(components, patch):
    sample = {k: p6.trilinear_sample(v, patch.authoritative_frac_index_local)
              for k, v in components.items()}
    tensor = np.empty((len(patch.authoritative_parent_id), 3, 3))
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = sample["xx"], sample["yy"], sample["zz"]
    tensor[:, 0, 1] = tensor[:, 1, 0] = sample["xy"]
    tensor[:, 0, 2] = tensor[:, 2, 0] = sample["xz"]
    tensor[:, 1, 2] = tensor[:, 2, 1] = sample["yz"]
    eig, vec = np.linalg.eigh(tensor)
    return tensor, eig, vec


def run_p6(adapter, cores, model, normalization, tmu, tsd, device):
    reference_halo = P6_HALOS[-1]
    aggregate = {
        h: {"got": [], "ref": [], "field": [], "field_ref": [],
            "distance": [], "error": [], "per_core": []}
        for h in P6_HALOS[:-1]
    }
    reference_cache, per_core = {}, {}
    for meta in cores:
        core = meta["core_id"]
        cache = {}
        for halo in P6_HALOS:
            patch = adapter.extract(core, halo, alignment_voxels=ALIGNMENT)
            cache[halo] = (patch,) + predict(
                model, patch, normalization, tmu, tsd, device
            )
        ref_patch, ref_pred, ref_field = cache[reference_halo]
        reference_cache[core] = cache[reference_halo]
        ref_core = ref_field[ref_patch.core_slice]
        distance = support_distance(ref_patch)
        retained = distance >= P6_GATES["support_distance_mpc"]
        per_core[str(core)] = {"meta": meta, "halos": {}}
        for halo in P6_HALOS[:-1]:
            patch, pred, field = cache[halo]
            field = field[patch.core_slice]
            pred_metric, field_metric = compare(pred, ref_pred), compare(field, ref_core)
            error = np.sqrt(np.mean((pred - ref_pred) ** 2, axis=1))
            rho = rank_corr(distance[retained], error[retained]) if retained.sum() >= 50 else None
            per_core[str(core)]["halos"][str(halo)] = {
                "prediction": pred_metric, "latent_core": field_metric,
                "retained_boundary_n": int(retained.sum()),
                "retained_boundary_spearman": rho,
                "actual_halo_low": patch.available_halo_low,
                "actual_halo_high": patch.available_halo_high,
            }
            store = aggregate[halo]
            store["got"].append(pred)
            store["ref"].append(ref_pred)
            store["field"].append(field.ravel())
            store["field_ref"].append(ref_core.ravel())
            store["distance"].append(distance[retained])
            store["error"].append(error[retained])
            store["per_core"].append(pred_metric)

    summary, selected = {}, None
    for halo, store in aggregate.items():
        pred = compare(np.concatenate(store["got"]), np.concatenate(store["ref"]))
        field = compare(np.concatenate(store["field"]), np.concatenate(store["field_ref"]))
        distance, error = np.concatenate(store["distance"]), np.concatenate(store["error"])
        rho = rank_corr(distance, error) if len(distance) >= 50 else None
        worst = max(x["nrmse"] for x in store["per_core"])
        boundary_pass = (
            pred["nrmse"] <= P6_GATES["boundary_trivial_nrmse"]
            or (rho is not None and abs(rho) <= P6_GATES["boundary_abs_spearman"])
        )
        passed = (
            pred["nrmse"] <= P6_GATES["prediction_nrmse"]
            and pred["p95_abs_over_std"] <= P6_GATES["prediction_p95"]
            and field["nrmse"] <= P6_GATES["latent_nrmse"]
            and worst <= P6_GATES["worst_core_nrmse"]
            and boundary_pass
        )
        summary[str(halo)] = {
            "prediction": pred, "latent_core": field,
            "worst_core_prediction_nrmse": worst,
            "retained_boundary_n": int(len(distance)),
            "retained_boundary_spearman": rho,
            "boundary_pass": boundary_pass, "passes": passed,
        }

    # Freeze the smallest context on a stable convergence tail. A single
    # passing point followed by a larger failing context is not convergence.
    ordered_halos = list(aggregate)
    for index, halo in enumerate(ordered_halos):
        if all(summary[str(larger)]["passes"] for larger in ordered_halos[index:]):
            selected = halo
            break

    child_got, child_ref = [], []
    if selected is not None:
        for meta in cores:
            core = meta["core_id"]
            parent, parent_pred, _ = reference_cache[core]
            start, stop = parent.core_start.copy(), parent.core_stop.copy()
            axis = int(np.argmax(stop - start))
            middle = int((start[axis] + stop[axis]) // 2)
            lookup = {int(i): p for i, p in zip(parent.authoritative_parent_id, parent_pred)}
            for high in (False, True):
                lo, hi = start.copy(), stop.copy()
                if high:
                    lo[axis] = middle
                    use = parent.authoritative_frac_index_global[:, axis] >= middle
                else:
                    hi[axis] = middle
                    use = parent.authoritative_frac_index_global[:, axis] < middle
                child = adapter.extract_bounds(
                    cap=parent.cap, core_start=lo, core_stop=hi,
                    context_halo_voxels=selected, alignment_voxels=ALIGNMENT,
                    core_id=core, fold=parent.fold,
                    authoritative_parent_id=parent.authoritative_parent_id[use],
                    authoritative_frac_index_global=parent.authoritative_frac_index_global[use],
                )
                got, _ = predict(model, child, normalization, tmu, tsd, device)
                child_got.append(got)
                child_ref.append(np.asarray([lookup[int(i)] for i in child.authoritative_parent_id]))
    subdivision = (
        compare(np.concatenate(child_got), np.concatenate(child_ref))
        if child_got else None
    )
    subdivision_pass = (
        subdivision is not None
        and subdivision["nrmse"] <= P6_GATES["subdivision_nrmse"]
        and subdivision["p95_abs_over_std"] <= P6_GATES["subdivision_p95"]
    )
    return {
        "gates": P6_GATES, "halos_voxels": P6_HALOS,
        "reference_halo_voxels": reference_halo,
        "selected_halo_voxels": selected, "summary": summary,
        "per_core": per_core, "subdivision": subdivision,
        "subdivision_pass": subdivision_pass,
        "passes": selected is not None and subdivision_pass,
        "canary_role": (
            "frozen trained T2 U-Net; label-free structural test only; "
            "the full-cap channel mapping is fixed and never refit per patch"
        ),
    }


def orientation_summary(got_vec, ref_vec, ref_eig):
    angle = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(got_vec * ref_vec, axis=1)), 0, 1
    )))
    g12, g23 = ref_eig[:, 1] - ref_eig[:, 0], ref_eig[:, 2] - ref_eig[:, 1]
    gaps = np.stack([g12, np.minimum(g12, g23), g23], axis=1)
    output = {}
    for axis in range(3):
        q = np.quantile(gaps[:, axis], [0, 0.25, 0.5, 0.75, 1])
        bins = []
        for index in range(4):
            mask = (gaps[:, axis] >= q[index]) & (
                gaps[:, axis] <= q[index + 1] if index == 3
                else gaps[:, axis] < q[index + 1]
            )
            values = angle[mask, axis]
            bins.append({
                "gap_low": float(q[index]), "gap_high": float(q[index + 1]),
                "n": int(mask.sum()), "median_angle_deg": float(np.median(values)),
                "p95_angle_deg": float(np.quantile(values, 0.95)),
            })
        large = gaps[:, axis] >= q[2]
        output[f"axis_{axis}"] = {
            "bins": bins, "large_gap_n": int(large.sum()),
            "large_gap_median_angle_deg": float(np.median(angle[large, axis])),
            "large_gap_p95_angle_deg": float(np.quantile(angle[large, axis], 0.95)),
        }
    return output


def run_p7(adapter, cores, model, normalization, tmu, tsd, device):
    reference = P7_CONFIGS[-1]
    data = {
        cfg["name"]: {"density": [], "density_ref": [], "tensor": [],
                      "tensor_ref": [], "eig": [], "eig_ref": [],
                      "vec": [], "vec_ref": [], "trace": [],
                      "distance": [], "eigenvalue_error": []}
        for cfg in P7_CONFIGS[:-1]
    }
    for meta in cores:
        results = {}
        for cfg in P7_CONFIGS:
            patch = adapter.extract(
                meta["core_id"], cfg["halo"], alignment_voxels=ALIGNMENT
            )
            _, learned = predict(model, patch, normalization, tmu, tsd, device)
            components, smoothed = p7.fft_tidal_components(
                learned, cell_mpc=CELL_MPC, rsmooth_mpc=RSMOOTH_MPC,
                apodization_width_voxels=cfg["apodization"],
                padding_voxels=cfg["padding"],
            )
            tensor, eig, vec = point_eigensystem(components, patch)
            results[cfg["name"]] = {
                "density": p6.trilinear_sample(
                    learned, patch.authoritative_frac_index_local
                ),
                "tensor": tensor, "eig": eig, "vec": vec,
                "trace": p7.trace_max_abs_error(components, smoothed),
                "support_distance_mpc": support_distance(patch),
            }
        ref = results[reference["name"]]
        for cfg in P7_CONFIGS[:-1]:
            got, store = results[cfg["name"]], data[cfg["name"]]
            for name in ("density", "tensor", "eig", "vec"):
                store[name].append(got[name])
                store[name + "_ref"].append(ref[name])
            store["trace"].append(got["trace"])
            store["distance"].append(ref["support_distance_mpc"])
            store["eigenvalue_error"].append(np.sqrt(np.mean(
                (got["eig"] - ref["eig"]) ** 2, axis=1
            )))

    summary, selected = {}, None
    for cfg in P7_CONFIGS[:-1]:
        store = data[cfg["name"]]
        density = compare(np.concatenate(store["density"]), np.concatenate(store["density_ref"]))
        tensor = compare(np.concatenate(store["tensor"]), np.concatenate(store["tensor_ref"]))
        eig = compare(np.concatenate(store["eig"]), np.concatenate(store["eig_ref"]))
        orient = orientation_summary(
            np.concatenate(store["vec"]), np.concatenate(store["vec_ref"]),
            np.concatenate(store["eig_ref"]),
        )
        median = max(v["large_gap_median_angle_deg"] for v in orient.values())
        p95 = max(v["large_gap_p95_angle_deg"] for v in orient.values())
        trace = max(store["trace"])
        distance = np.concatenate(store["distance"])
        point_error = np.concatenate(store["eigenvalue_error"])
        retained = distance >= P7_GATES["support_distance_mpc"]
        rho = (
            rank_corr(distance[retained], point_error[retained])
            if retained.sum() >= 50 else None
        )
        boundary_pass = (
            eig["nrmse"] <= P7_GATES["boundary_trivial_eigenvalue_nrmse"]
            or (rho is not None and abs(rho) <= P7_GATES["boundary_abs_spearman"])
        )
        scale = eig["reference_std"]
        near = ~retained
        near_mean = float(np.mean(point_error[near]) / scale) if near.any() else None
        retained_mean = (
            float(np.mean(point_error[retained]) / scale) if retained.any() else None
        )
        passed = (
            density["nrmse"] <= P7_GATES["density_nrmse"]
            and tensor["nrmse"] <= P7_GATES["tensor_nrmse"]
            and eig["nrmse"] <= P7_GATES["eigenvalue_nrmse"]
            and eig["p95_abs_over_std"] <= P7_GATES["eigenvalue_p95"]
            and median <= P7_GATES["large_gap_median_angle_deg"]
            and p95 <= P7_GATES["large_gap_p95_angle_deg"]
            and trace <= P7_GATES["trace_max_abs_error"]
            and boundary_pass
        )
        summary[cfg["name"]] = {
            "config": cfg, "density": density, "tensor": tensor,
            "eigenvalues": eig, "orientation": orient,
            "worst_large_gap_median_angle_deg": median,
            "worst_large_gap_p95_angle_deg": p95,
            "trace_max_abs_error": trace, "passes": passed,
            "retained_boundary_n": int(retained.sum()),
            "near_boundary_n": int(near.sum()),
            "retained_boundary_spearman": rho,
            "near_boundary_mean_error_over_std": near_mean,
            "retained_mean_error_over_std": retained_mean,
            "boundary_pass": boundary_pass,
        }
        if selected is None and passed and cfg.get("eligible", False):
            selected = cfg
    return {
        "gates": P7_GATES, "configs": P7_CONFIGS,
        "reference_config": reference, "selected_config": selected,
        "summary": summary, "passes": selected is not None,
        "canary_role": (
            "trained U-Net latent stresses learned-field spectra plus nonlocal FFT; "
            "P5 separately proves graph context for arbitrary weights; "
            "the final F-PATCH checkpoint must repeat this gate"
        ),
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(report), indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p6-root", type=Path, default=P6_ROOT)
    parser.add_argument("--selection-manifest", type=Path, default=SELECTION)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    selection = json.loads(args.selection_manifest.read_text())
    normalization = selection["rotations"][str(args.rotation)]["normalization"]
    model, tmu, tsd, saved, replaced_norms = load_model(args.checkpoint, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with p6.CanonicalFieldPatchAdapter(
        args.p6_root, selection_manifest=args.selection_manifest,
        rotation=args.rotation,
    ) as adapter:
        cores = select_cores(adapter, selection, max(P6_HALOS))
        report6 = run_p6(adapter, cores, model, normalization, tmu, tsd, device)
        fft_cores = [x for x in cores if x["shell"] in (0, 3)]
        report7 = run_p7(adapter, fft_cores, model, normalization, tmu, tsd, device)
    common = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha_at_execution": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_home_test_r2": saved.get("home_test_r2"),
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "rotation": args.rotation, "device": str(device),
        "representative_cores": cores,
        "target_labels_loaded": False,
        "spatial_normalization_policy": "per_voxel_channel_layer_norm",
        "replaced_groupnorm_modules": replaced_norms,
        "final_u_patch_must_train_with_patch_safe_normalization": True,
        "wall_seconds_total": time.time() - started,
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated())
        if device.type == "cuda" else None,
    }
    report6, report7 = {**common, **report6}, {
        **common, "representative_cores": fft_cores, **report7
    }
    write_report(P6_OUT / "trained_convergence_report.json", report6)
    write_report(P7_OUT / "trained_convergence_report.json", report7)
    if report6["passes"]:
        (P6_OUT / "UNET_PATCH_READY").write_text(
            "P6 trained structural convergence passed; see report.\n"
        )
    if report7["passes"]:
        (P7_OUT / "FTIER_PATCH_READY").write_text(
            "P7 learned-field FFT convergence passed; final F-PATCH repeats gate.\n"
        )
    result = {
        "p6_passes": report6["passes"],
        "p6_selected_halo_voxels": report6["selected_halo_voxels"],
        "p7_passes": report7["passes"],
        "p7_selected_config": report7["selected_config"],
    }
    print(json.dumps(serial(result), indent=2))
    if not report6["passes"] or not report7["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

