#!/usr/bin/env python3
"""Bounded P12 training-only coordinate/host-label audit, not a V0 release gate.

Inventory mode reads small receipts only. Numerical mode requires a compute
node, reads deterministic FITS rows, at most six halo slabs, and mapped NPZ cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = ("ph002", "ph003", "ph004", "ph005")
SHELLS = (.15, .25, .35, .45, .55)
ROTATION_Z = np.array([-.8676661490190047, -.1980763734312015, .4559837761750669])
LIMIT = 32 * 1024**2


def record(path, small=False):
    path = Path(path)
    info = path.stat()
    row = dict(path=str(path), bytes=info.st_size, mtime_ns=info.st_mtime_ns,
               content_hash_verified=False)
    if small:
        if info.st_size > LIMIT:
            raise ValueError(f"small-artifact limit: {path}")
        with path.open("rb") as f:
            data = f.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise ValueError("artifact grew beyond limit")
        row.update(sha256=hashlib.sha256(data).hexdigest(), content_hash_verified=True)
    return row


def read_json(path):
    rec = record(path, small=True)
    with Path(path).open("rb") as f:
        data = f.read(LIMIT + 1)
    if hashlib.sha256(data).hexdigest() != rec["sha256"]:
        raise ValueError("receipt changed during read")
    return json.loads(data), rec


def phase_guard(phase):
    if phase not in ALLOWED:
        raise ValueError("only ph002-005 are licensed here; ph000 needs a separate imported-lineage audit")


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not socket.gethostname().startswith("nid"):
        raise RuntimeError("numerical audit requires a Slurm compute node")


def mapped_member(path, key):
    """Map an uncompressed numeric NPY member without loading a 6 GiB slab.

    This is partial sampling, not ZIP CRC/full-file integrity verification.
    Compressed members are rejected rather than silently inflated on a login node.
    """
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError("requires uncompressed, unencrypted NPZ members")
        with archive.open(info) as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError("unsupported NPY version")
            header_bytes = f.tell()
        if dtype.hasobject or header_bytes + int(np.prod(shape)) * dtype.itemsize != info.file_size:
            raise ValueError("invalid numeric member size/dtype")
        with open(path, "rb") as f:
            f.seek(info.header_offset)
            header = f.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("invalid local ZIP header")
        name_bytes, extra_bytes = struct.unpack_from("<HH", header, 26)
        offset = info.header_offset + 30 + name_bytes + extra_bytes + header_bytes
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=offset,
                     order="F" if fortran else "C")


def grid_indices(position):
    # Annotation explicitly casts native positions to float32 before binning.
    pos = np.asarray(position, dtype=np.float32)
    return np.clip(np.floor(np.mod(pos, 2000.) / (2000. / 2048)).astype(np.int64), 0, 2047)


def assert_parent_join(observed, parent):
    for key in ("TARGETID", "FILE_NUM", "HALO_INDEX", "BOX_INDEX",
                "RA", "DEC", "LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB"):
        if not np.array_equal(observed[key], parent[key]):
            raise ValueError(f"canonical/parent mismatch: {key}")


def inventory(phase):
    phase_guard(phase)
    registry, registry_rec = read_json(ROOT / "configs/p10_phase_registry_v1.json")
    if phase not in registry["model_phase_contract"]["training"]:
        raise ValueError("phase no longer in training contract")
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    p1, p1_rec = read_json(root / "p1_canonical/CATALOGUE_COMPLETE.json")
    if p1.get("phase") != phase or p1.get("pass") is not True:
        raise ValueError("invalid canonical marker")
    if p1["geometry"]["units"] != "Mpc" or p1["geometry"]["cosmology"] != "Astropy Planck18":
        raise ValueError("unexpected recorded observer convention")
    observed = Path(p1["canonical_parent"]["path"])
    receipt, receipt_rec = read_json(str(observed) + ".complete.json")
    if receipt["phase"] != phase or receipt["output"]["sha256"] != p1["canonical_parent"]["sha256"]:
        raise ValueError("observed receipt binding mismatch")
    parent = Path(receipt["annotated_parent"]["path"])
    points = Path(p1["artifacts"]["points"])
    for path in (observed, parent, points):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("unexpected cross-phase input path")
    if observed.stat().st_size != p1["canonical_parent"]["bytes"]:
        raise ValueError("canonical FITS byte-size mismatch")
    return dict(phase=phase, registry=registry_rec, p1=p1_rec, observed_receipt=receipt_rec,
                observed=record(observed), parent=record(parent), points=record(points),
                recorded_observed_sha256=p1["canonical_parent"]["sha256"],
                recorded_points_sha256=p1["artifacts"]["points_sha256"],
                expected_rows=p1["counts"]["total"],
                snapshot_root=registry["path_templates"]["snapshot_root"].format(phase=phase),
                tweb_dir=str(root / "targets/tweb/backend_optimized_ngrid_2048_rsmooth_7"))


def native_labels(inv, observed, selected):
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    xyz = np.empty((len(selected), 3), dtype=np.float32)
    sources = []
    for slab in sorted(set(observed["FILE_NUM"][selected].tolist())):
        path = Path(inv["snapshot_root"]) / "halo_info" / f"halo_info_{slab:03d}.asdf"
        cat = CompaSOHaloCatalog(path, fields=["x_com"], subsamples=False,
                                cleaned=False, convert_units=True, verbose=False)
        if cat.header["SimName"] != f"AbacusSummit_base_c000_{inv['phase']}":
            raise ValueError("native halo phase mismatch")
        mask = observed["FILE_NUM"][selected] == slab
        ids = observed["HALO_INDEX"][selected][mask]
        if np.any(ids < 0) or np.any(ids >= len(cat.halos)):
            raise ValueError("native host index out of range")
        xyz[mask] = cat.halos["x_com"][ids]
        sources.append(record(path))
        del cat
    ijk = grid_indices(xyz)
    labels = np.full((len(selected), 3), np.nan, dtype=np.float32)
    classes = np.full(len(selected), -1, dtype=np.int16)
    covered = np.zeros(len(selected), dtype=np.int8)
    slab_records = []
    spans = []
    for path in sorted(Path(inv["tweb_dir"]).glob("abacus_cactus_tweb_rank*.npz")):
        with np.load(path, allow_pickle=False) as f:
            lo, hi = int(f["x_start"]), int(f["x_end"])
            if (int(f["ngrid"]), float(f["boxsize"]), float(f["Rsmooth"])) != (2048, 2000., 7.):
                raise ValueError("native T-Web grid/box/smoothing mismatch")
            if not np.isclose(float(f["threshold"]), .2, rtol=0., atol=1e-7):
                raise ValueError("native T-Web threshold mismatch")
        spans.append((lo, hi))
        mask = (ijk[:, 0] >= lo) & (ijk[:, 0] < hi)
        if not mask.any():
            continue
        eig = mapped_member(path, "eig_vals")
        cweb = mapped_member(path, "cweb")
        if eig.shape != (3, hi-lo, 2048, 2048) or cweb.shape != (hi-lo, 2048, 2048):
            raise ValueError("native member shape mismatch")
        x, y, z = ijk[mask].T
        labels[mask] = eig[:, x-lo, y, z].T
        classes[mask] = cweb[x-lo, y, z]
        covered[mask] += 1
        slab_records.append(record(path))
        del eig, cweb
    spans.sort()
    if not spans or spans[0][0] != 0 or spans[-1][1] != 2048 or any(a[1] != b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError("incomplete native slab coverage")
    if not np.all(covered == 1) or not np.all(np.isfinite(labels)):
        raise ValueError("native sample has missing or multiply assigned cells")
    stored = np.column_stack([observed[k][selected] for k in ("LAMBDA1", "LAMBDA2", "LAMBDA3")])
    return dict(rows=len(selected), host_position_field_tested="x_com",
                labels_equal_at_float32=bool(np.array_equal(labels, stored)),
                max_abs_label_difference=float(np.max(np.abs(labels-stored))),
                cweb_equal=bool(np.array_equal(classes, observed["CWEB"][selected])),
                observed_sample_indices=selected.tolist(), native_voxels=ijk.tolist(),
                host_sources=sources, tweb_sources=slab_records,
                sampled_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest())


def choose_native_rows(data, caps, max_slabs=6):
    """Cover cap/shell strata using geometry only, then sample <=16 per stratum.

    Global slab populations can select just NGC. Greedily cover missing strata
    first, with capped per-stratum counts and slab ID as deterministic tie breaks.
    """
    valid = ((data["Z"] >= .15) & (data["Z"] < .55) & (data["BOX_INDEX"] >= 0)
             & (data["FILE_NUM"] >= 0) & (data["HALO_INDEX"] >= 0))
    strata_masks = [valid & (caps == cap) & (data["Z"] >= lo) & (data["Z"] < hi)
                   for cap in (0, 1) for lo, hi in zip(SHELLS, SHELLS[1:])]
    candidates = sorted(set(data["FILE_NUM"][valid].tolist()))
    counts = {slab: np.array([np.sum(mask & (data["FILE_NUM"] == slab))
                             for mask in strata_masks]) for slab in candidates}
    selected_slabs = []
    covered = np.zeros(8, dtype=int)
    while candidates and len(selected_slabs) < max_slabs:
        slab = max(candidates, key=lambda s: (int(np.sum((covered == 0) & (counts[s] > 0))),
                    int(np.minimum(covered + counts[s], 16).sum()), -s))
        if np.minimum(covered + counts[slab], 16).sum() == np.minimum(covered, 16).sum():
            break
        selected_slabs.append(slab)
        covered += counts[slab]
        candidates.remove(slab)
    selected, strata = [], {}
    for index, mask in enumerate(strata_masks):
        rows = np.flatnonzero(mask & np.isin(data["FILE_NUM"], selected_slabs))[:16]
        selected.extend(rows.tolist())
        strata[f"cap{index//4}_shell{index%4}"] = len(rows)
    return np.asarray(selected, dtype=int), strata, selected_slabs


def numerical(inv, samples):
    import fitsio
    from astropy.cosmology import Planck18
    require_compute()
    if not 128 <= samples <= 65536:
        raise ValueError("sample budget must be 128..65536")
    names = ["TARGETID", "RA", "DEC", "Z", "FILE_NUM", "HALO_INDEX", "BOX_INDEX",
             "LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB"]
    with fitsio.FITS(inv["observed"]["path"]) as f:
        n = f[1].get_nrows()
        if n != inv["expected_rows"]:
            raise ValueError("canonical row count mismatch")
        rows = np.unique(np.linspace(0, n-1, samples, dtype=np.int64))
        data = f[1].read(rows=rows, columns=names)
    with fitsio.FITS(inv["parent"]["path"]) as f:
        ids = data["TARGETID"].astype(np.int64)
        if np.any(ids <= 0) or np.any(ids > f[1].get_nrows()):
            raise ValueError("parent TARGETID out of range")
        parent = f[1].read(rows=ids-1, columns=names)
    assert_parent_join(data, parent)
    ra, dec = np.deg2rad(data["RA"]), np.deg2rad(data["DEC"])
    unit = np.column_stack((np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)))
    zgrid = np.linspace(0., .85, 17001)
    radius = np.interp(data["Z"], zgrid, Planck18.comoving_distance(zgrid).value)
    expected = unit * radius[:, None]
    points = np.load(inv["points"]["path"], mmap_mode="r", allow_pickle=False)
    if points.shape != (n, 4) or points.dtype != np.dtype("float64"):
        raise ValueError("unexpected canonical point shape or precision")
    stored = np.asarray(points[rows])
    replay_error = np.linalg.norm(stored[:, :3] - expected, axis=1)
    science = (data["Z"] >= .15) & (data["Z"] < .55)
    if not science.any():
        raise ValueError("sample has no science-range rows")
    exact = Planck18.comoving_distance(data["Z"][science]).value[:, None] * unit[science]
    integration_error = np.linalg.norm(stored[science, :3]-exact, axis=1)
    caps = (unit @ ROTATION_Z > 0).astype(int)
    selected, strata, slabs = choose_native_rows(data, caps)
    if not len(selected):
        raise ValueError("no licensed native-host samples")
    native = native_labels(inv, data, np.asarray(selected, dtype=int))
    checks = dict(parent_targetid_sky_host_labels_equal=True,
                  coordinate_replay=float(replay_error.max()) <= 1e-9,
                  independent_distance_integration=float(integration_error.max()) <= 1e-5,
                  cap_identity=bool(np.array_equal(caps, stored[:, 3])),
                  native_x_com_labels=native["labels_equal_at_float32"],
                  native_x_com_classes=native["cweb_equal"],
                  all_cap_shells_sampled=all(value > 0 for value in strata.values()))
    return dict(checks=checks, partial_checks_pass=all(checks.values()),
                sampled_observed_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                sampled_parent_sha256=hashlib.sha256(parent.tobytes()).hexdigest(),
                sampled_points_sha256=hashlib.sha256(stored.tobytes()).hexdigest(),
                sampled_canonical_rows=rows.tolist(), sampled_targetids=ids.tolist(),
                cap_shell_native_rows=strata, selected_halo_slabs=slabs,
                max_coordinate_replay_error_mpc=float(replay_error.max()),
                max_exact_distance_error_mpc=float(integration_error.max()), native=native)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=ALLOWED, default="ph002")
    p.add_argument("--samples", type=int, default=16384)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--input-inventory", type=Path,
                   help="required frozen inventory for numerical execution")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.inventory_only:
        require_compute()
    inv = inventory(args.phase)
    source = record(__file__, small=True)
    frozen_inventory = None
    if not args.inventory_only:
        if args.input_inventory is None:
            raise ValueError("numerical execution requires --input-inventory")
        frozen, frozen_inventory = read_json(args.input_inventory)
        if (frozen.get("schema") != "p12a-coordinate-sample-audit-v1"
                or frozen.get("inventory_only") is not True
                or frozen.get("inputs") != inv
                or frozen.get("source", {}).get("sha256") != source["sha256"]):
            raise ValueError("source or inputs differ from frozen inventory")
    report = dict(schema="p12a-coordinate-sample-audit-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  source=source, inputs=inv, frozen_inventory=frozen_inventory,
                  inventory_only=args.inventory_only,
                  coordinate_audit_status="unresolved", ready_for_desi_canary=False,
                  numerical_tolerances=dict(replay_mpc=1e-9, independent_distance_mpc=1e-5,
                                            native_labels="exact after float32 storage"),
                  outstanding=["full_encoder_and_oof_lineage", "ph000_imported_lineage",
                               "annotation_run_position_field_provenance", "central_satellite_unresolved_strata",
                               "pinned_native_distance_comparison", "response_volume_and_support_consistency",
                               "loa_adapter_replay", "cross_phase_replication"],
                  execution=dict(host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID")))
    if not args.inventory_only:
        report["numerical"] = numerical(inv, args.samples)
        if inventory(args.phase) != inv or record(__file__, small=True)["sha256"] != source["sha256"]:
            raise ValueError("source or input identity changed during audit")
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(output=str(args.output), coordinate_audit_status="unresolved",
                          partial_checks_pass=report.get("numerical", {}).get("partial_checks_pass"))))
    return 2 if args.inventory_only or report["numerical"]["partial_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
