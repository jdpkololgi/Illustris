"""Train-only wide384_f4 research data. No scientific release is implied.

Targets are kept separate from observations. The last fine condition channel
is the shared parent realization (teacher forcing in a training example); an
inference caller MUST replace it with its sampled parent, never local truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from workflows.sbi.e2e_field_build_products import require_compute, sha256
from workflows.sbi.e2e_field_dataset import Moments, transformed
from workflows.sbi.e2e_field_regenerate_spectral import interpolate
from workflows.sbi.e2e_field_wide_coarse import local_coords, padded_extract

PHASES = ("ph000", "ph002", "ph003")
VARIANT = {"name": "wide384_f4", "span_fine_cells": 384, "factor": 4}
COARSE_CHANNELS = ("counts", "expected_counts_random", "support_random",
                   "angular_response", "exposure_apodized_random", "ntilde_mpc3",
                   "geometry_valid_fraction", "log_count_ratio_random",
                   "los_x", "los_y", "los_z", "observer_radius_mpc")
LOCAL_CHANNELS = ("counts", "support_random", "angular_response",
                  "exposure_apodized_random", "expected_counts_random",
                  "log_count_ratio_random", "distance_to_support_boundary",
                  "ntilde_mpc3", "los_x", "los_y", "los_z", "observer_redshift")
FINE_CHANNELS = LOCAL_CHANNELS + ("shared_coarse_on_fine",)
IDENTITY = {"support_random", "angular_response", "exposure_apodized_random",
            "geometry_valid_fraction", "los_x", "los_y", "los_z"}


def guarded_path(path):
    p = Path(path)
    for value in (str(p), str(p.resolve())):
        if any(phase not in PHASES for phase in re.findall(r"ph\d{3}", value)):
            raise PermissionError("non-training phase path prohibited")
    return p.resolve()


def _json(path):
    with guarded_path(path).open() as f:
        return json.load(f)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def normalization_source_hashes():
    """Bind fitted moments to preprocessing, interpolation and extraction code."""
    root = Path(__file__).resolve().parent
    names = ("e2e_wide_data.py", "e2e_field_dataset.py",
             "e2e_field_regenerate_spectral.py", "e2e_field_wide_coarse.py",
             "e2e_field_build_products.py")
    return {name: sha256(root/name) for name in names}


def coarse_to_fine(coarse, fine_side=96, factor=4):
    """Exact builder quintic stencil; accepts a physical 3-D scalar field."""
    x = np.asarray(coarse)
    if x.ndim == 4 and x.shape[0] == 1:
        return coarse_to_fine(x[0], fine_side, factor)[None]
    if x.ndim != 3 or len(set(x.shape)) != 1 or not np.isfinite(x).all():
        raise ValueError("coarse field must be a finite cube")
    if x.size > 32**3:
        require_compute()
    variant = {"span_fine_cells": x.shape[0]*factor, "factor": factor}
    return interpolate(x, local_coords(variant, fine_side), degree=5).astype(np.float32)


def inverse_target(value, normalization, level):
    if level not in ("coarse", "fine"):
        raise ValueError("unknown target level")
    stats = normalization["targets"][level]
    return np.asarray(value)*stats["std"] + stats["mean"]


def _checked(x, shape, name):
    x = np.asarray(x, dtype=np.float32)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"invalid shape/values: {name}")
    return x


class WideResearchDataset:
    """Metadata-only construction; payload reads and verification need compute."""

    def __init__(self, products_root, normalization_path=None, verify_payloads=False):
        self.root = guarded_path(products_root)
        manifest_path = self.root/"WIDE_COARSE_PRODUCTS_COMPLETE.json"
        self.manifest = _json(manifest_path)
        started = _json(self.root/"WIDE_COARSE_STARTED.json")
        config = started["config"]
        if (self.manifest.get("training_ready") is not False or
                config["phases"] != list(PHASES) or VARIANT not in config["variants"] or
                config["fine_side"] != 96 or config["interpolation_degree"] != 5 or
                self.manifest["registration"] != started["registration"]):
            raise ValueError("unsupported research product contract")
        self.rows = self.manifest["parents"]
        if (len(self.rows) != 96 or len({r["anchor_id"] for r in self.rows}) != 96 or
                any(sum(r["phase"] == p for r in self.rows) != 32 for p in PHASES)):
            raise ValueError("requires original 96 training anchors")
        for row in self.rows:
            if (row["role"] != "train" or row["phase"] not in PHASES or
                    not row["anchor_id"].startswith(row["phase"]+"_") or
                    row["group"] != row["anchor_id"] or row["cap"] not in ("NGC", "SGC") or
                    any(c % 4 for c in row["center"])):
                raise PermissionError("invalid training row")
            guarded_path(row["shard"])
        index_path = guarded_path(config["dataset_root"])/"DATASET_INDEX.json"
        index = _json(index_path)
        if sha256(index_path) != self.manifest["registration"]["dataset_index_sha256"]:
            raise ValueError("spectral index drift")
        source_rows = {r["anchor_id"]: r for r in index["parents"] if r["phase"] in PHASES}
        if any(source_rows.get(r["anchor_id"]) != r for r in self.rows):
            raise ValueError("wide/source row mismatch")
        shards = {str(guarded_path(s["path"])): s["sha256"] for s in index["shards"]
                  if any(str(s["path"]) == r["shard"] for r in self.rows)}
        self.payloads = dict(shards)
        if (len(self.manifest["phases"]) != 3 or
                {p["phase"] for p in self.manifest["phases"]} != set(PHASES)):
            raise ValueError("missing phase receipt")
        for phase in self.manifest["phases"]:
            for record in [phase["fields"], *phase["responses"]]:
                path = guarded_path(record["path"])
                if path.parent != self.root:
                    raise ValueError("wide payload outside registered root")
                self.payloads[str(path)] = record["sha256"]
        for row in self.rows:
            for path in (row["shard"], self.root/f"{row['phase']}_fields.h5",
                         self.root/f"{row['phase']}_{row['cap']}_response.h5"):
                if str(guarded_path(path)) not in self.payloads:
                    raise ValueError("missing payload provenance")
        self.binding = {"manifest_sha256": sha256(manifest_path),
                        "index_sha256": sha256(index_path), "variant": VARIANT,
                        "started_sha256": sha256(self.root/"WIDE_COARSE_STARTED.json"),
                        "reader_sha256": sha256(__file__),
                        "normalization_source_sha256": normalization_source_hashes(),
                        "payloads": self.payloads,
                        "coarse_channels": list(COARSE_CHANNELS), "fine_channels": list(FINE_CHANNELS)}
        self.verified = False
        self.normalization = None
        if verify_payloads:
            self.verify_payloads()
        if normalization_path is not None:
            self.normalization = _json(normalization_path)
            n = self.normalization
            if (n.get("schema") != "e2e-wide-normalization-v1" or n.get("binding") != self.binding or
                    n.get("fit_phases") != list(PHASES) or n.get("fit_roles") != ["train"] or
                    n.get("sha256") != _digest({k:v for k,v in n.items() if k != "sha256"})):
                raise ValueError("normalization binding mismatch")
            if (set(n["targets"]) != {"coarse", "fine"} or
                    set(n["coarse"]) != set(COARSE_CHANNELS)-IDENTITY or
                    set(n["fine"]) != set(LOCAL_CHANNELS)-IDENTITY or n.get("training_ready") is not False):
                raise ValueError("normalization channel/target schema mismatch")
            for stats in [*n["targets"].values(), *n["coarse"].values(), *n["fine"].values()]:
                if not np.isfinite([stats["mean"], stats["std"]]).all() or stats["std"] <= 0:
                    raise ValueError("invalid normalization scale")

    def __len__(self):
        return len(self.rows)

    def verify_payloads(self):
        require_compute()
        for path, expected in self.payloads.items():
            if sha256(guarded_path(path)) != expected:
                raise ValueError("payload hash mismatch")
        self.verified = True

    def raw(self, index):
        return self._read(index, include_targets=True)

    def _read(self, index, *, include_targets):
        require_compute()
        row = self.rows[index]
        if row["phase"] not in PHASES or row["role"] != "train":
            raise PermissionError("training phases only")
        with h5py.File(guarded_path(self.root/f"{row['phase']}_fields.h5"), "r") as f:
            if f.attrs.get("role") != "train":
                raise PermissionError("field role mismatch")
            g = f[row["anchor_id"]][VARIANT["name"]]
            if include_targets:
                coarse = _checked(g["coarse_delta"][:], (96,)*3, "coarse")
                fine = _checked(g["fine_residual"][:], (96,)*3, "fine")
            response_path = guarded_path(g.attrs["response_file"])
            expected_path = self.root/f"{row['phase']}_{row['cap']}_response.h5"
            start = (np.asarray(row["center"])-192)//4
            if (response_path != expected_path or g.attrs["response_group"] != "factor4" or
                    int(g.attrs["response_side"]) != 96 or not np.array_equal(start, g.attrs["response_start"])):
                raise ValueError("response alignment/provenance mismatch")
        with h5py.File(response_path, "r") as f:
            channels = {name: padded_extract(f["factor4"][name], start, 96)
                        for name in COARSE_CHANNELS[:8]}
        positions = [row["grid"]["origin_mpc"][a] +
                     (start[a]+np.arange(96)+.5)*4*row["grid"]["cell_mpc"] for a in range(3)]
        xyz = np.meshgrid(*positions, indexing="ij", sparse=True)
        radius = np.sqrt(sum(x*x for x in xyz))
        for name, x in zip(("los_x", "los_y", "los_z"), xyz):
            channels[name] = np.broadcast_to(x/np.maximum(radius, 1e-30), radius.shape)
        channels["observer_radius_mpc"] = radius
        crop = (slice(16,112),)*3
        with h5py.File(guarded_path(row["shard"]), "r") as f:
            g = f[row["group"]]
            local = {}
            for name in LOCAL_CHANNELS:
                ds = g["condition_raw"][name]
                if ds.shape != (128,)*3:
                    raise ValueError("local observation alignment mismatch")
                local[name] = ds[crop]
            masks = {name: _checked(g["masks"][name][:], (96,)*3, name)
                     for name in ("latent_domain", "truth_available", "loss_domain",
                                  "science_core_geometry", "observed_parent", "science_supported")} if include_targets else {}
        shared = coarse_to_fine(coarse) if include_targets else np.zeros((96,)*3, dtype=np.float32)
        return {**({"coarse_target": coarse[None], "fine_target": fine[None]} if include_targets else {}),
                "coarse_condition": np.stack([_checked(channels[n], (96,)*3, n) for n in COARSE_CHANNELS]),
                "fine_condition": np.stack([_checked(local[n], (96,)*3, n) for n in LOCAL_CHANNELS]+[shared]),
                "anchor_id": row["anchor_id"], "role": "train", "phase": row["phase"],
                "geometry": {"coarse_side": 96, "fine_side": 96, "factor": 4,
                             "fine_cell_mpc_h": 3.383, "coarse_cell_mpc_h": 13.532},
                "masks": masks, "training_ready": False}

    def __getitem__(self, index):
        return self._normalize(self.raw(index))

    def inference_conditions(self, index):
        """No density/residual/oracle/mask payload reads. Last channel is unset."""
        item = self._normalize(self._read(index, include_targets=False))
        item["fine_condition"][-1] = 0
        return item

    def inverse_target(self, value, name="coarse"):
        return inverse_target(value, self.normalization, name)

    def normalize_targets(self, value, name="coarse"):
        stats = self.normalization["targets"][name]
        return (np.asarray(value)-stats["mean"])/stats["std"]

    def _normalize(self, item):
        if self.normalization is None:
            raise ValueError("fit/load train-only normalization before normalized access")
        n = self.normalization
        for level, names in (("coarse", COARSE_CHANNELS), ("fine", LOCAL_CHANNELS)):
            stats = n["targets"][level]
            if level+"_target" in item:
                item[level+"_target"] = (item[level+"_target"]-stats["mean"])/stats["std"]
            for i, name in enumerate(names):
                x = transformed(name, item[level+"_condition"][i])
                if name not in IDENTITY:
                    stats = n[level][name]
                    x = (x-stats["mean"])/stats["std"]
                item[level+"_condition"][i] = x
        stats = n["targets"]["coarse"]
        item["fine_condition"][-1] = (item["fine_condition"][-1]-stats["mean"])/stats["std"]
        return item


def fit_normalization(dataset, output_path):
    """Streaming equal-anchor voxel moments, with full payload hash verification."""
    require_compute()
    output_path = guarded_path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    dataset.verify_payloads()
    before = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
              for p in dataset.payloads}
    targets = {level: Moments() for level in ("coarse", "fine")}
    channels = {level: {name: Moments() for name in names if name not in IDENTITY}
                for level, names in (("coarse", COARSE_CHANNELS), ("fine", LOCAL_CHANNELS))}
    for i in range(len(dataset)):
        item = dataset.raw(i)
        if item["role"] != "train" or item["phase"] not in PHASES:
            raise PermissionError("non-training normalization input")
        for level, names in (("coarse", COARSE_CHANNELS), ("fine", LOCAL_CHANNELS)):
            targets[level].add(item[level+"_target"])
            for j, name in enumerate(names):
                if name in channels[level]:
                    channels[level][name].add(transformed(name, item[level+"_condition"][j]))
    if any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != state
           for p, state in before.items()):
        raise ValueError("payload changed during normalization")
    result = {"schema": "e2e-wide-normalization-v1", "binding": dataset.binding,
              "fit_phases": list(PHASES), "fit_roles": ["train"],
              "targets": {k:v.report() for k,v in targets.items()},
              **{level: {k:v.report() for k,v in values.items()} for level, values in channels.items()},
              "training_ready": False}
    result["sha256"] = _digest(result)
    with output_path.open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    return result
