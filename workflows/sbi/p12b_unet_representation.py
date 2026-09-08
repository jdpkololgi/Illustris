#!/usr/bin/env python3
"""Prepare, verify and run the registered P12-B U-Net representation pilot.

Per-galaxy FMPE only: the encoder is a feature field, not a sampled tidal field.
All heavy stages require an existing Slurm GPU allocation. This script does not
submit, cancel, extend, or chain allocations. Exit 75 means checkpointed partial.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workflows.sbi.p12b_unet_representation_common import (
    safe_path, sha256, read_json, atomic_json, atomic_npz, atomic_torch, finite,
    theta_from_eigen, eigen_from_theta, physical_log_jacobian, context_boxes,
    disjoint_candidates, balanced_cores, build_flow, heun_sample, heun_log_prob,
    row_scores, clustered_difference,
)
from workflows.abacus_tweb.p8_train_unet_patch import UPatch, model_inputs, CHANNELS
from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter
from workflows.abacus_tweb.p8_deterministic_common import increments_to_eigenvalues, unscale_increments
from workflows.sbi.p12_export_unet_summaries import ntilde_at_rows, parent_to_assignment_index


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(event, **fields):
    print(json.dumps({"utc": utc_now(), "event": event, **fields}, allow_nan=False), flush=True)


class Pilot:
    def __init__(self, config_path, max_seconds):
        self.config_path = safe_path(config_path)
        self.c = read_json(config_path)
        c = self.c
        if c["posterior_training_phase"] != "ph005" or c["diagnostic_phase"] != "ph006":
            raise ValueError("only registered ph005/ph006 pilot roles are supported")
        if c["ph001_access"] or c["posthoc_calibration"] or c["production_promotion_allowed"]:
            raise PermissionError("pilot cannot open ph001, recalibrate or promote")
        if c["arms"] != ["point", "frozen", "joint"] or c["context_dimensions"] != 39:
            raise ValueError("registered nested three-arm interface changed")
        self.root = safe_path(c["output_root"])
        self.phase_root = safe_path(c["phase_root"])
        self.contract = safe_path(self.phase_root / c["encoder_contract"])
        self.encoder_run = safe_path(self.phase_root / c["encoder_run"])
        self.device = "cuda"
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("real-data stages require a Slurm GPU allocation")
        self.deadline = time.monotonic() + max_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".pilot.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.sources = self.source_contract()
        self.config_sha = sha256(config_path)
        self.cache = OrderedDict()
        torch.set_num_threads(8)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def source_contract(self):
        paths = [Path(__file__), REPO / "workflows/sbi/p12b_unet_representation_common.py",
                 REPO / "workflows/abacus_tweb/p8_train_unet_patch.py",
                 REPO / "workflows/abacus_tweb/p6_field_patch_utils.py",
                 REPO / "workflows/abacus_tweb/p8_deterministic_common.py",
                 REPO / "workflows/sbi/p12_export_unet_summaries.py",
                 REPO / "workflows/abacus_tweb/p10_training_contract.py",
                 REPO / "workflows/abacus_tweb/p8_train_patch_recovery.py"]
        import sbi.neural_nets.estimators.flowmatching_estimator as installed
        paths.append(Path(installed.__file__))
        return {str(path): sha256(path) for path in paths}

    def check_sources(self):
        if self.source_contract() != self.sources or sha256(self.config_path) != self.config_sha:
            raise RuntimeError("source or configuration changed during pilot")

    def near_deadline(self, reserve=120):
        return time.monotonic() + reserve >= self.deadline

    def encoder(self):
        path = self.encoder_run / "arm_a_checkpoint.pt"
        state = torch.load(safe_path(path), map_location="cpu", weights_only=False)
        marker = read_json(self.encoder_run / "ARM_A_TRAINING_COMPLETE.json")
        if (state.get("schema_version") != "p10-arm-a-checkpoint-v1" or state["model"] != "unet"
                or state["global_step"] != marker["global_steps"] or marker["epochs_completed"] != 20
                or marker["training_phases"] != self.c["encoder_training_phases"]
                or not state["frozen_arguments"]["disable_early_stopping"]):
            raise RuntimeError("encoder is not the registered terminal epoch-20 omitted-phase model")
        if safe_path(state["frozen_arguments"]["contract_root"]).resolve() != self.contract.resolve():
            raise RuntimeError("encoder contract path mismatch")
        model = UPatch(base=self.c["unet_base"], latent_channels=self.c["latent_dimensions"])
        model.load_state_dict(state["model_state"])
        return model.to(self.device).eval()

    def bind(self, registry, path, expected=None):
        value = sha256(path)
        if expected is not None and value != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
        registry[str(safe_path(path))] = value
        return value

    def verify_files(self, registry):
        for path, expected in registry.items():
            if sha256(path) != expected:
                raise RuntimeError(f"frozen artifact changed: {path}")

    def phase_metadata(self, phase, sources):
        base = self.phase_root / "training_contract"
        record_path = base / "phases" / phase / "phase_contract.json"
        self.bind(sources, record_path)
        record = read_json(record_path)
        if not record["pass"] or not record["truth_present"]:
            raise RuntimeError("visible phase contract does not pass")
        for name in ("assignment", "cores", "p1_manifest"):
            self.bind(sources, record["inputs"][name], record["inputs"][name + "_sha256"])
        adapter_root = base / "adapters" / phase / "field"
        self.bind(sources, adapter_root / "adapter_manifest.json")
        for name in ("core_voxel_start", "core_voxel_stop", "core_fold", "core_cap",
                     "core_active_offsets", "core_active_parent", "core_active_frac_index"):
            self.bind(sources, adapter_root / f"{name}.npy")
        adapter = CanonicalFieldPatchAdapter(adapter_root,
            selection_manifest=self.contract / "transforms/field/selection_manifest.json", rotation=0)
        # Existing response arrays are read-only and keyed through their own OOF parent order.
        oof_path = self.phase_root / "p12_oof_summaries" / phase / "OOF_SUMMARY_COMPLETE.json"
        self.bind(sources, oof_path)
        oof = read_json(oof_path)
        parent_path = oof["arrays"]["parent_node_id"]
        parent_sha = self.bind(sources, parent_path, oof["array_sha256"]["parent_node_id"])
        parent = np.load(safe_path(parent_path), mmap_mode="r")
        cache_path = self.phase_root / "p12a_random_support_parent_cache_v2" / phase / "P12A_RANDOM_SUPPORT_CACHE_READY.json"
        self.bind(sources, cache_path)
        response = read_json(cache_path)
        if not response["pass"] or response["phase"] != phase or response["parent_sha256"] != parent_sha:
            raise RuntimeError("response cache parent identity mismatch")
        self.bind(sources, response["audit"]["field_manifest"], response["field_manifest_sha256"])
        for name in ("distance", "support"):
            self.bind(sources, response[name + "_path"], response[name + "_sha256"])
        distance = np.load(safe_path(response["distance_path"]), mmap_mode="r")
        support = np.load(safe_path(response["support_path"]), mmap_mode="r")
        if len(np.unique(parent)) != len(parent) or len(distance) != len(parent) or len(support) != len(parent):
            raise RuntimeError("response cache rows are not unique and aligned")
        lookup = np.full(record["parent_rows"], -1, dtype=np.int64)
        lookup[parent] = np.arange(len(parent))
        cores = dict(np.load(safe_path(record["inputs"]["cores"]), allow_pickle=False))
        if not np.array_equal(cores["core_id"], np.arange(len(adapter.core_cap))):
            raise RuntimeError("core IDs are not dense adapter indices")
        if not np.array_equal(cores["fold"], adapter.core_fold) or not np.array_equal(cores["cap"], adapter.core_cap):
            raise RuntimeError("P4 and adapter core identity mismatch")
        eligible = []
        for core in np.flatnonzero(np.diff(adapter.core_offsets) > 0):
            ids, _ = adapter.authoritative(int(core))
            rows = lookup[ids]
            if np.any(rows < 0):
                raise RuntimeError("authoritative parent absent from response cache")
            if np.any(support[rows]):
                eligible.append(int(core))
        return dict(record=record, adapter=adapter, cores=cores, lookup=lookup,
                    distance=distance, support=support, eligible=np.array(eligible))

    def prepare(self):
        ready_path = self.root / "P12B_DATA_READY.json"
        if ready_path.exists():
            self.load_ready()
            log("reuse_verified_data", manifest=str(ready_path))
            return True
        started = time.monotonic()
        sources = {}
        for path in (self.encoder_run / "arm_a_checkpoint.pt", self.encoder_run / "ARM_A_TRAINING_COMPLETE.json",
                     self.contract / "transforms/field/selection_manifest.json",
                     self.contract / "transforms/field/field_transform.json", self.contract / "transforms/target_scaler.json"):
            self.bind(sources, path)
        normalization = read_json(self.contract / "transforms/field/field_transform.json")["normalization"]
        scaler = read_json(self.contract / "transforms/target_scaler.json")
        selection = read_json(self.contract / "transforms/field/selection_manifest.json")
        metadata = {phase: self.phase_metadata(phase, sources) for phase in ("ph005", "ph006")}
        rng = np.random.default_rng(self.c["panel_seed"])
        panel = []
        for phase, meta in metadata.items():
            cores, adapter, eligible = meta["cores"], meta["adapter"], meta["eligible"]
            shell = np.argmax(cores["active_count_by_shell"], axis=1)
            low, high = context_boxes(adapter.core_start, adapter.core_stop,
                                     self.c["halo_voxels"], self.c["alignment_voxels"])
            if phase == "ph005":
                internal = balanced_cores(eligible[np.isin(adapter.core_fold[eligible], self.c["internal_folds"])],
                    adapter.core_cap, shell, self.c["cores"]["internal"], rng)
                train = eligible[np.isin(adapter.core_fold[eligible], self.c["train_folds"])]
                train = disjoint_candidates(train, adapter.core_cap, low, high, internal)
                train = balanced_cores(train, adapter.core_cap, shell, self.c["cores"]["train"], rng)
                roles = {"train": train, "internal": internal}
            else:
                roles = {"diagnostic": balanced_cores(eligible, adapter.core_cap, shell,
                                                       self.c["cores"]["diagnostic"], rng)}
            for role, ids in roles.items():
                for core in ids:
                    parents, _ = adapter.authoritative(int(core))
                    supported = np.flatnonzero(meta["support"][meta["lookup"][parents]])
                    chosen = rng.choice(supported, min(len(supported), self.c["rows_per_core"]), replace=False)
                    chosen = np.sort(chosen)
                    panel.append(dict(phase=phase, role=role, core=int(core), cap=int(adapter.core_cap[core]),
                        fold=int(adapter.core_fold[core]), superblock=int(cores["superblock_id"][core]),
                        dominant_shell=int(shell[core]), context_low=low[core].tolist(), context_high=high[core].tolist(),
                        local_row=chosen.tolist(), parent=parents[chosen].tolist()))
        # Freeze identities and geometric split before reading any target values.
        panel_path = self.root / "P12B_PANEL_FROZEN.json"
        frozen = dict(config_sha256=self.config_sha, sources=self.sources, inputs=sources, panel=panel)
        if panel_path.exists() and read_json(panel_path) != frozen:
            raise RuntimeError("existing frozen panel differs; never overwrite it")
        atomic_json(panel_path, frozen)
        for phase in metadata:
            selected = [p for row in panel if row["phase"] == phase for p in row["parent"]]
            if len(set(selected)) != len(selected):
                raise RuntimeError("duplicate selected authoritative parents")
        model = self.encoder()
        artifacts = []
        transform_x, transform_y = [], []
        for phase, meta in metadata.items():
            record = meta["record"]
            self.bind(sources, record["target"]["path"], record["target"]["sha256"])
            truth = np.load(safe_path(record["target"]["path"]), mmap_mode="r")
            redshift_path = self.phase_root / "training_contract/phases" / phase / "parent_redshift.npy"
            self.bind(sources, redshift_path)
            redshift = np.load(redshift_path, mmap_mode="r")
            with np.load(safe_path(record["inputs"]["assignment"]), allow_pickle=False) as assignment:
                assignment_index = parent_to_assignment_index(assignment, len(truth))
                assignment_shell = assignment["shell"]
                assignment_cap = assignment["cap"]
                for row in (row for row in panel if row["phase"] == phase):
                    if self.near_deadline():
                        log("partial_preparation", completed=len(artifacts))
                        return False
                    path = self.root / "patches" / f'{phase}_{row["role"]}_{row["core"]:06d}.npz'
                    patch = meta["adapter"].extract(row["core"], self.c["halo_voxels"], CHANNELS,
                                                  alignment_voxels=self.c["alignment_voxels"])
                    chosen = np.asarray(row["local_row"])
                    parent = patch.authoritative_parent_id[chosen]
                    if not np.array_equal(parent, row["parent"]):
                        raise RuntimeError("panel parent order changed")
                    values, points = model_inputs(patch, normalization, self.device)
                    points = points[..., chosen, :]
                    with torch.no_grad():
                        latent = model.sample_latent(values, points)
                        base = increments_to_eigenvalues(unscale_increments(model.head(latent).cpu().numpy(), scaler)).astype(np.float32)
                    z = np.asarray(redshift[parent], dtype=np.float32)
                    ai = assignment_index[parent]
                    if np.any(ai < 0) or np.any(assignment_cap[ai] != row["cap"]):
                        raise RuntimeError("P4 assignment join failed")
                    cap = np.full(len(parent), row["cap"], dtype=np.int8)
                    ntilde = ntilde_at_rows(selection, cap, z)
                    dist = np.asarray(meta["distance"][meta["lookup"][parent]])
                    if np.any(ntilde <= 0) or np.any(dist < 0):
                        raise RuntimeError("invalid physical response")
                    response = np.column_stack((z, np.log(ntilde), cap, np.log1p(dist))).astype(np.float32)
                    eigen = np.asarray(truth[parent], dtype=np.float64)
                    theta = theta_from_eigen(eigen).astype(np.float32)
                    if not np.allclose(eigen_from_theta(theta), eigen, atol=1e-6, rtol=1e-6):
                        raise RuntimeError("ordered-softplus roundtrip failed")
                    arrays = dict(values=values.cpu().numpy(), points=points.cpu().numpy(), parent=parent,
                        latent=latent.cpu().numpy(), base=base, response=response, truth=eigen, theta=theta,
                        shell=assignment_shell[ai], cap=cap, superblock=np.full(len(parent), row["superblock"], dtype=np.int64),
                        fold=np.full(len(parent), row["fold"], dtype=np.int8),
                        context_start=patch.context_start, context_stop=patch.context_stop)
                    finite(*arrays.values())
                    atomic_npz(path, **arrays)
                    artifacts.append(dict(path=str(path), sha256=sha256(path), phase=phase, role=row["role"],
                                          core=row["core"], rows=len(parent), superblock=row["superblock"]))
                    if row["role"] == "train":
                        transform_x.append(np.column_stack((base, response, arrays["latent"])))
                        transform_y.append(theta)
                    if len(artifacts) % 16 == 0:
                        log("prepare_progress", cores=len(artifacts), total=len(panel), seconds=time.monotonic()-started)
            meta["adapter"].close()
        x = np.concatenate(transform_x).astype(np.float64)
        y = np.concatenate(transform_y).astype(np.float64)
        transforms = dict(x_mean=x.mean(0).tolist(), x_std=np.maximum(x.std(0), 1e-6).tolist(),
                          theta_mean=y.mean(0).tolist(), theta_std=y.std(0).tolist(),
                          fit_role="ph005 selected training rows only", rows=len(x))
        finite(np.array(transforms["x_std"]), np.array(transforms["theta_std"]))
        if np.min(transforms["theta_std"]) <= 0:
            raise RuntimeError("degenerate target scaling")
        atomic_json(self.root / "transforms.json", transforms)
        self.verify_files(sources)
        self.check_sources()
        manifest = dict(schema_version="p12b-data-v1", pass_=True, created_utc=utc_now(),
            config_sha256=self.config_sha, source_hashes=self.sources, input_hashes=sources,
            panel_sha256=sha256(panel_path), transforms_sha256=sha256(self.root / "transforms.json"),
            artifacts=artifacts, rows_by_role={role: sum(a["rows"] for a in artifacts if a["role"] == role)
                for role in ("train", "internal", "diagnostic")},
            validation_scope="Selected small arrays rehashed; source HDF5 payloads bound by existing manifests, not rehashed",
            ph001_access=False, coherent_field_posterior=False, production_promotion_allowed=False,
            seconds=time.monotonic()-started)
        manifest["pass"] = manifest.pop("pass_")
        atomic_json(ready_path, manifest)
        log("data_ready", rows=manifest["rows_by_role"], cores=len(artifacts))
        return True

    def load_ready(self):
        ready = read_json(self.root / "P12B_DATA_READY.json")
        if not ready["pass"] or ready["source_hashes"] != self.sources or ready["config_sha256"] != self.config_sha:
            raise RuntimeError("data contract/source mismatch")
        self.verify_files(ready["input_hashes"])
        self.verify_files({a["path"]: a["sha256"] for a in ready["artifacts"]})
        self.verify_files({str(self.root / "transforms.json"): ready["transforms_sha256"],
                           str(self.root / "P12B_PANEL_FROZEN.json"): ready["panel_sha256"]})
        self.ready = ready
        self.ready_sha = sha256(self.root / "P12B_DATA_READY.json")
        self.transforms = read_json(self.root / "transforms.json")
        self.x_mean = torch.tensor(self.transforms["x_mean"], device=self.device, dtype=torch.float32)
        self.x_std = torch.tensor(self.transforms["x_std"], device=self.device, dtype=torch.float32)
        self.y_mean = torch.tensor(self.transforms["theta_mean"], device=self.device, dtype=torch.float32)
        self.y_std = torch.tensor(self.transforms["theta_std"], device=self.device, dtype=torch.float32)
        return ready

    def patch(self, artifact):
        path = artifact["path"]
        if path not in self.cache:
            with np.load(safe_path(path), allow_pickle=False) as archive:
                self.cache[path] = {k: archive[k] for k in archive.files}
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path]

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def condition(self, arm, encoder, patch):
        common = self.tensor(np.column_stack((patch["base"], patch["response"])))
        common = (common - self.x_mean[:7]) / self.x_std[:7]
        if arm == "point":
            latent = torch.zeros((len(common), 32), device=self.device)
        else:
            raw = (encoder.sample_latent(self.tensor(patch["values"]), self.tensor(patch["points"]))
                   if arm == "joint" else self.tensor(patch["latent"]))
            latent = (raw - self.x_mean[7:]) / self.x_std[7:]
        return torch.cat((common, latent), dim=-1)

    def models(self, arm):
        encoder = self.encoder()
        encoder.requires_grad_(arm == "joint")
        encoder.head.requires_grad_(False)
        flow = build_flow(self.c, self.device)
        groups = [{"params": flow.parameters(), "lr": self.c["head_learning_rate"]}]
        if arm == "joint":
            groups.append({"params": encoder.unet.parameters(), "lr": self.c["encoder_learning_rate"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.c["weight_decay"])
        return encoder, flow, optimizer

    def step(self, arm, encoder, flow, optimizer, patch, update):
        optimizer.zero_grad(set_to_none=True)
        context = self.condition(arm, encoder, patch)
        theta = (self.tensor(patch["theta"]) - self.y_mean) / self.y_std
        # Step-index seeds make noise/time identical across arms and resumes.
        torch.manual_seed(self.c["seed"] + 100000 + update)
        loss = flow.loss(theta, context).mean()
        finite(loss)
        loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.c["gradient_clip"], error_if_nonfinite=True)
        encoder_gradient = max((float(p.grad.detach().abs().max()) for p in encoder.unet.parameters() if p.grad is not None), default=0.0)
        optimizer.step()
        finite(*[p.detach() for p in parameters])
        return float(loss.detach()), float(norm), encoder_gradient

    def smoke(self):
        self.load_ready()
        artifact = next(a for a in self.ready["artifacts"] if a["role"] == "train")
        patch = self.patch(artifact)
        reports = {}
        for arm in self.c["arms"]:
            started = time.monotonic()
            encoder, flow, optimizer = self.models(arm)
            with torch.no_grad():
                raw = encoder.sample_latent(self.tensor(patch["values"]), self.tensor(patch["points"]))
            parity = float((raw - self.tensor(patch["latent"])).abs().max())
            if parity > 1e-6:
                raise RuntimeError("cached/online latent mismatch")
            # SBI's zero-initialized output gives zero context gradient at update 0.
            # This discarded smoke warm-up is identical for every arm.
            self.step(arm, encoder, flow, optimizer, patch, 0)
            before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            loss, norm, gradient = self.step(arm, encoder, flow, optimizer, patch, 1)
            changed = any(not torch.equal(v, before[k]) for k, v in encoder.state_dict().items())
            if (arm == "joint") != (gradient > 0 and changed):
                raise RuntimeError("frozen/joint encoder gradient contract failed")
            if any(not torch.equal(v, before[k]) for k, v in encoder.state_dict().items() if k.startswith("head.")):
                raise RuntimeError("old deterministic head changed")
            replay_path = self.root / "smoke" / f"{arm}_replay.pt"
            atomic_torch(replay_path, dict(encoder=encoder.state_dict(), flow=flow.state_dict(), optimizer=optimizer.state_dict()))
            self.step(arm, encoder, flow, optimizer, patch, 2)
            expected = {"encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                        "flow": {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}}
            state = torch.load(replay_path, map_location=self.device, weights_only=False)
            encoder.load_state_dict(state["encoder"])
            flow.load_state_dict(state["flow"])
            optimizer.load_state_dict(state["optimizer"])
            self.step(arm, encoder, flow, optimizer, patch, 2)
            replay_error = max(float((v.cpu()-expected[name][k]).abs().max())
                for name, model in (("encoder", encoder), ("flow", flow)) for k, v in model.state_dict().items())
            if replay_error > 1e-6:
                raise RuntimeError("checkpoint replay differs beyond CUDA numerical tolerance")
            with torch.no_grad():
                context = self.condition(arm, encoder, patch)[:4]
            noise = torch.randn(4, 3, device=self.device)
            sample = heun_sample(flow, context, noise, 8)
            density = heun_log_prob(flow, context, sample, 8)
            finite(sample, density)
            reports[arm] = dict(loss=loss, gradient_norm=norm, encoder_max_gradient=gradient,
                encoder_changed=changed, cached_online_max_abs=parity, checkpoint_replay_max_abs=replay_error,
                replay_tolerance=1e-6, seconds=time.monotonic()-started,
                cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                trainable_parameters=sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
            log("smoke_arm_pass", arm=arm, **reports[arm])
            del encoder, flow, optimizer, before, expected, state, raw
            torch.cuda.empty_cache()
        self.check_sources()
        atomic_json(self.root / "P12B_SMOKE_PASS.json", dict(passed=True, data_sha256=self.ready_sha,
            source_hashes=self.sources, config_sha256=self.config_sha, reports=reports,
            replay_note="Step-index RNG exact; CUDA grid_sample backward replay checked to 1e-6, not claimed bitwise deterministic"))
        return True

    def save_checkpoint(self, arm, encoder, flow, optimizer, update, rows, elapsed):
        self.check_sources()
        atomic_torch(self.root / arm / "checkpoint.pt", dict(schema_version="p12b-checkpoint-v1", arm=arm,
            data_sha256=self.ready_sha, source_hashes=self.sources, config_sha256=self.config_sha,
            encoder=encoder.state_dict() if arm == "joint" else None, flow=flow.state_dict(),
            optimizer=optimizer.state_dict(), update=update, row_presentations=rows, elapsed_seconds=elapsed))

    def train(self, arm):
        encoder, flow, optimizer = self.models(arm)
        output = self.root / arm
        output.mkdir(exist_ok=True)
        update, rows, elapsed = 0, 0, 0.0
        path = output / "checkpoint.pt"
        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=False)
            if (state["data_sha256"] != self.ready_sha or state["source_hashes"] != self.sources
                    or state["config_sha256"] != self.config_sha or state["arm"] != arm):
                raise RuntimeError("checkpoint resume contract mismatch")
            flow.load_state_dict(state["flow"])
            if arm == "joint":
                encoder.load_state_dict(state["encoder"])
            optimizer.load_state_dict(state["optimizer"])
            update, rows, elapsed = state["update"], state["row_presentations"], state["elapsed_seconds"]
            if update == self.c["training_updates"] and (output / "TRAINING_COMPLETE.json").exists():
                marker = read_json(output / "TRAINING_COMPLETE.json")
                if marker["checkpoint_sha256"] != sha256(path):
                    raise RuntimeError("terminal training checkpoint changed")
                encoder.eval()
                flow.eval()
                return encoder, flow
        train = [a for a in self.ready["artifacts"] if a["role"] == "train"]
        schedule = []
        rng = np.random.default_rng(self.c["seed"])
        while len(schedule) < self.c["training_updates"]:
            schedule.extend(rng.permutation(len(train)).tolist())
        started = time.monotonic()
        trace = (output / "loss_trace.jsonl").open("a")
        encoder.train(arm == "joint")
        flow.train()
        while update < self.c["training_updates"]:
            if self.near_deadline():
                self.save_checkpoint(arm, encoder, flow, optimizer, update, rows, elapsed + time.monotonic()-started)
                log("checkpointed_partial", arm=arm, update=update)
                trace.close()
                return None
            artifact = train[schedule[update]]
            patch = self.patch(artifact)
            loss, norm, gradient = self.step(arm, encoder, flow, optimizer, patch, update)
            if arm == "joint" and update > 0 and gradient <= 0:
                raise RuntimeError("joint encoder gradient disappeared")
            update += 1
            rows += artifact["rows"]
            if update % self.c["log_every"] == 0 or update == 1:
                row = dict(utc=utc_now(), arm=arm, update=update, loss=loss, gradient_norm=norm,
                    encoder_max_gradient=gradient, row_presentations=rows,
                    elapsed_seconds=elapsed+time.monotonic()-started, cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                log("training", **row)
            if update % self.c["checkpoint_every"] == 0:
                self.save_checkpoint(arm, encoder, flow, optimizer, update, rows, elapsed+time.monotonic()-started)
        self.save_checkpoint(arm, encoder, flow, optimizer, update, rows, elapsed+time.monotonic()-started)
        trace.close()
        atomic_json(output / "TRAINING_COMPLETE.json", dict(arm=arm, updates=update, row_presentations=rows,
            unique_train_cores=len(train), unique_train_rows=sum(a["rows"] for a in train),
            trainable_parameters=sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
            elapsed_seconds=elapsed+time.monotonic()-started, checkpoint_sha256=sha256(path),
            data_sha256=self.ready_sha, source_hashes=self.sources, config_sha256=self.config_sha))
        encoder.eval()
        flow.eval()
        return encoder, flow

    def evaluate(self, arm, encoder, flow):
        for role in ("internal", "diagnostic"):
            artifacts = [a for a in self.ready["artifacts"] if a["role"] == role]
            for index, artifact in enumerate(artifacts):
                if self.near_deadline():
                    return False
                path = self.root / arm / "evaluation" / f'{role}_{artifact["core"]:06d}.npz'
                marker = path.with_suffix(".json")
                if marker.exists():
                    done = read_json(marker)
                    if done["data_sha256"] != self.ready_sha or done["checkpoint_sha256"] != sha256(self.root / arm / "checkpoint.pt"):
                        raise RuntimeError("evaluation resume mismatch")
                    self.verify_files({str(path): done["sha256"]})
                    continue
                patch = self.patch(artifact)
                n = min(self.c["evaluation_rows_per_core"], len(patch["parent"]))
                # Common, predeclared evenly spaced subset; no truth-based row selection.
                chosen = np.linspace(0, len(patch["parent"])-1, n, dtype=int)
                with torch.no_grad():
                    context = self.condition(arm, encoder, patch)[chosen]
                draws = self.c["posterior_draws"]
                condition = context.repeat_interleave(draws, dim=0)
                torch.manual_seed(self.c["seed"] + 200000 + (role == "diagnostic")*100000 + artifact["core"])
                noise = torch.randn(n*draws, 3, device=self.device)
                sample = heun_sample(flow, condition, noise, self.c["sample_steps"]).view(n, draws, 3)
                theta = (sample*self.y_std+self.y_mean).cpu().numpy()
                eigen = eigen_from_theta(theta)
                truth = patch["truth"][chosen]
                scores = row_scores(eigen, truth)
                density_rows = min(self.c["log_score_rows_per_core"], n)
                target = (self.tensor(patch["theta"][chosen[:density_rows]])-self.y_mean)/self.y_std
                logp = heun_log_prob(flow, context[:density_rows], target, self.c["sample_steps"]).cpu().numpy()
                logp = logp + physical_log_jacobian(truth[:density_rows], self.transforms["theta_std"])
                # Every 16th fixed core, common noise; no adaptive solver selection.
                check = {}
                if index % 16 == 0:
                    refined = heun_sample(flow, condition, noise, self.c["sample_check_steps"]).view(n, draws, 3)
                    ref_eigen = eigen_from_theta((refined*self.y_std+self.y_mean).cpu().numpy())
                    ref_scores = row_scores(ref_eigen, truth)
                    ref_logp = heun_log_prob(flow, context[:density_rows], target, self.c["sample_check_steps"]).cpu().numpy()
                    ref_logp += physical_log_jacobian(truth[:density_rows], self.transforms["theta_std"])
                    check = dict(scaled_mean_abs=float((sample-refined).abs().mean()), rows=n,
                        coverage_difference=np.concatenate([scores[f"coverage{level}"].mean(0)-ref_scores[f"coverage{level}"].mean(0) for level in (68,90)]).tolist(),
                        log_score_mean_abs=float(np.abs(logp-ref_logp).mean()))
                arrays = dict(samples=eigen, truth=truth, parent=patch["parent"][chosen], shell=patch["shell"][chosen],
                    cap=patch["cap"][chosen], superblock=patch["superblock"][chosen], log_prob=logp,
                    log_prob_parent=patch["parent"][chosen[:density_rows]], **scores)
                atomic_npz(path, **arrays)
                atomic_json(marker, dict(data_sha256=self.ready_sha, checkpoint_sha256=sha256(self.root/arm/"checkpoint.pt"),
                    sha256=sha256(path), sampler_check=check))
                if (index+1) % 16 == 0:
                    log("evaluation", arm=arm, role=role, cores=index+1, total=len(artifacts))
        return True

    def summarize(self):
        import tarp
        result = dict(schema_version="p12b-comparison-v1", technical_completion=True,
            production_promotion_allowed=False, conditional_calibration_pass=False,
            estimand="Equal selected-galaxy panel; cap/dominant-shell balanced cores, not full footprint",
            data_sha256=self.ready_sha, source_hashes=self.sources, config_sha256=self.config_sha,
            sampler={}, arms={}, paired_differences={})
        data = {}
        for arm in self.c["arms"]:
            result["arms"][arm] = {}
            result["sampler"][arm] = {}
            for role in ("internal", "diagnostic"):
                selected = [a for a in self.ready["artifacts"] if a["role"] == role]
                parts, checks = [], []
                for artifact in selected:
                    path = self.root / arm / "evaluation" / f'{role}_{artifact["core"]:06d}.npz'
                    marker = read_json(path.with_suffix(".json"))
                    self.verify_files({str(path): marker["sha256"]})
                    with np.load(path) as archive:
                        parts.append({k: archive[k] for k in archive.files})
                    if marker["sampler_check"]:
                        checks.append(marker["sampler_check"])
                combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                data[(arm, role)] = combined
                report = {key: np.mean(combined[key], axis=0).tolist()
                    for key in ("energy", "crps", "coverage68", "coverage90", "width68", "width90", "log_prob")}
                report["rows"] = len(combined["parent"])
                report["log_score_rows"] = len(combined["log_prob"])
                residual = np.sum((combined["truth"]-combined["mean"])**2, axis=0)
                total = np.sum((combined["truth"]-combined["truth"].mean(0))**2, axis=0)
                report["r2"] = (1-residual/total).tolist()
                report["by_shell"] = {}
                for shell in range(4):
                    mask = combined["shell"] == shell
                    report["by_shell"][str(shell)] = {"rows": int(mask.sum())}
                    if mask.any():
                        report["by_shell"][str(shell)].update({key: combined[key][mask].mean(0).tolist()
                            for key in ("energy", "crps", "coverage68", "coverage90", "width68", "width90")})
                report["tarp"] = {}
                for coordinates in ("physical_eigenvalues", "anchor_physical_gaps"):
                    samples, truth = combined["samples"], combined["truth"]
                    if coordinates == "anchor_physical_gaps":
                        samples = np.concatenate((samples[..., :1], np.diff(samples, axis=-1)), axis=-1)
                        truth = np.concatenate((truth[..., :1], np.diff(truth, axis=-1)), axis=-1)
                    np.random.seed(self.c["seed"])
                    ecp, alpha = tarp.get_tarp_coverage(samples.transpose(1,0,2), truth, references="random",
                        metric="euclidean", norm=True, bootstrap=False, seed=self.c["seed"])
                    report["tarp"][coordinates] = dict(ecp=np.asarray(ecp).tolist(), alpha=np.asarray(alpha).tolist(),
                        max_abs=float(np.max(np.abs(ecp-alpha))), uncertainty="Descriptive curve only; correlated rows are not IID")
                result["arms"][arm][role] = report
                weights = np.array([check["rows"] for check in checks])
                maxima = {key: float(np.average([check[key] for check in checks], weights=weights))
                          for key in ("scaled_mean_abs", "log_score_mean_abs")}
                maxima["coverage_max_abs"] = float(np.max(np.abs(np.average(
                    [check["coverage_difference"] for check in checks], axis=0, weights=weights))))
                maxima["checked_rows"] = int(weights.sum())
                maxima["pass"] = bool(maxima["scaled_mean_abs"] <= self.c["sampler_max_mean_abs_scaled_difference"]
                    and maxima["coverage_max_abs"] <= self.c["sampler_max_coverage_difference"]
                    and maxima["log_score_mean_abs"] <= self.c["sampler_max_log_score_difference"])
                result["sampler"][arm][role] = maxima
        for role in ("internal", "diagnostic"):
            result["paired_differences"][role] = {}
            for first, second in (("frozen", "point"), ("joint", "frozen"), ("joint", "point")):
                a, b = data[(first, role)], data[(second, role)]
                if not np.array_equal(a["parent"], b["parent"]):
                    raise RuntimeError("arm evaluation identities mismatch")
                clusters = np.column_stack((a["cap"], a["superblock"]))
                summary = {"energy": clustered_difference(a["energy"], b["energy"], clusters,
                    self.c["bootstrap_repetitions"], self.c["seed"])}
                for dim in range(3):
                    summary[f"crps_lambda{dim+1}"] = clustered_difference(a["crps"][:,dim], b["crps"][:,dim], clusters,
                        self.c["bootstrap_repetitions"], self.c["seed"])
                    for level in (68,90):
                        summary[f"coverage{level}_lambda{dim+1}"] = clustered_difference(
                            a[f"coverage{level}"][:,dim], b[f"coverage{level}"][:,dim], clusters,
                            self.c["bootstrap_repetitions"], self.c["seed"])
                result["paired_differences"][role][f"{first}_minus_{second}"] = summary
        result["sampler_gate_pass"] = all(v["pass"] for arm in result["sampler"].values() for v in arm.values())
        result["scientific_interpretation"] = (
            "Development representation pilot only. Require proper-score and shell-coverage review, phase replication, and independent confirmation."
            if result["sampler_gate_pass"] else "Sampler refinement gate failed; do not interpret arm ranking until separately authorized numerical diagnosis.")
        self.check_sources()
        atomic_json(self.root / "P12B_COMPARISON.json", result)
        log("technical_complete", sampler_gate_pass=result["sampler_gate_pass"])

    def run(self):
        self.load_ready()
        smoke = read_json(self.root / "P12B_SMOKE_PASS.json")
        if not smoke["passed"] or smoke["data_sha256"] != self.ready_sha or smoke["source_hashes"] != self.sources:
            raise RuntimeError("matching real-patch smoke is required")
        # Complete matched training before evaluating transfer diagnostics.
        for arm in self.c["arms"]:
            models = self.train(arm)
            if models is None:
                return False
            del models
            torch.cuda.empty_cache()
        for arm in self.c["arms"]:
            models = self.train(arm)  # idempotent load of terminal registered checkpoint
            if models is None or not self.evaluate(arm, *models):
                return False
            del models
            torch.cuda.empty_cache()
        if self.near_deadline(reserve=180):
            return False
        self.summarize()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "run", "all"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/p12b_unet_representation_v1.json")
    parser.add_argument("--max-runtime-seconds", type=int, default=6900)
    args = parser.parse_args()
    pilot = Pilot(args.config, args.max_runtime_seconds)
    import sbi
    launch = dict(utc=utc_now(), stage=args.stage, allocation=os.environ["SLURM_JOB_ID"], node=socket.gethostname(),
        command=sys.argv, python=sys.executable, torch=torch.__version__, sbi=sbi.__version__,
        gpu=torch.cuda.get_device_name(), source_hashes=pilot.sources, config_sha256=pilot.config_sha,
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    atomic_json(pilot.root / f'LAUNCH_{os.environ["SLURM_JOB_ID"]}_{args.stage}.json', launch)
    log("launch", **launch)
    stages = ("prepare", "smoke", "run") if args.stage == "all" else (args.stage,)
    try:
        for stage in stages:
            if not getattr(pilot, stage)():
                atomic_json(pilot.root / "P12B_PARTIAL.json", dict(stage=stage, utc=utc_now(), allocation=os.environ["SLURM_JOB_ID"],
                    reason="Interactive soft deadline; no automatic extension or batch fallback", config_sha256=pilot.config_sha))
                return 75
    except Exception as exc:
        atomic_json(pilot.root / f'P12B_FAILURE_{os.environ["SLURM_JOB_ID"]}.json', dict(utc=utc_now(),
            stage=stage, error=type(exc).__name__, message=str(exc), config_sha256=pilot.config_sha))
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
