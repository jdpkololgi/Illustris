"""Bounded wide384_f4 CFM/DIFF research pipeline.

Commands prepare/train/sample/diagnose require a user-authorized Slurm compute
session. No scheduler operations occur here. Only the three training phases are
accessible. A canary is an engineering run, never a calibration/release claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import torch

from workflows.sbi.e2e_field_build_products import require_compute, sha256, tensor_from_delta, eigs
from workflows.sbi.e2e_field_error_budget import science, tensor_metrics
from workflows.sbi.e2e_field_regenerate_spectral import interpolate
from workflows.sbi.e2e_field_wide_coarse import local_coords
from workflows.sbi.e2e_wide_data import (
    PHASES, COARSE_CHANNELS, FINE_CHANNELS, WideResearchDataset,
    coarse_to_fine, guarded_path, fit_normalization,
)
from workflows.sbi.e2e_wide_models import ConditionalFieldNet, flow_matching_loss, diffusion_loss, sample_field

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/e2e_wide_pipeline_v1.json"
SCRATCH_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v2")
SOURCE_FILES = (
    "workflows/sbi/e2e_wide_pipeline.py", "workflows/sbi/e2e_wide_data.py",
    "workflows/sbi/e2e_wide_models.py", "workflows/sbi/e2e_field_wide_coarse.py",
    "workflows/sbi/e2e_field_regenerate_spectral.py",
    "workflows/sbi/e2e_field_build_products.py", "workflows/sbi/e2e_field_dataset.py",
    "workflows/sbi/e2e_field_error_budget.py", "configs/e2e_field_domain_gate_v2.json",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_config(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text())
    frozen = {"schema": "e2e-wide-pipeline-v1", "variant": "wide384_f4",
              "phases": list(PHASES), "coarse_side": 96, "fine_side": 96, "core_side": 32,
              "fine_cell_mpc_h": 3.383, "coarse_factor": 4, "interpolation_degree": 5,
              "diagnostic_config": "configs/e2e_field_domain_gate_v2.json",
              "model": {"base_channels": 8, "levels": 3},
              "training_ready": False, "r0_physics_pass": False, "science_training_authorized": False}
    if any(c.get(k) != v for k, v in frozen.items()):
        raise ValueError("not the bounded wide384_f4 research contract")
    t, s = c["training"], c["sampling"]
    if any(type(t[k]) is not int for k in ("seed", "batch_size", "maximum_updates", "checkpoint_every")):
        raise ValueError("training budgets and seed must be integers")
    if any(type(s[k]) is not int for k in ("seed", "cfm_steps", "diffusion_steps")):
        raise ValueError("sampling budgets and seed must be integers")
    if (t["batch_size"] != 1 or not 1 <= t["maximum_updates"] <= 192 or
            not 1 <= t["checkpoint_every"] <= t["maximum_updates"] or
            not 0 < t["learning_rate"] <= .001 or not 0 <= t["weight_decay"] <= .1 or
            not 0 < t["clip_gradient_norm"] <= 10):
        raise ValueError("invalid or expanded canary budget")
    if (s["cfm_solver"] != "heun" or not 1 <= s["cfm_steps"] <= 16 or
            s["diffusion_steps"] != 2*s["cfm_steps"]):
        raise ValueError("samplers must have equal per-stage network-evaluation budgets")
    out = guarded_path(c["output_root"])
    if out == SCRATCH_ROOT.resolve() or not out.is_relative_to(SCRATCH_ROOT.resolve()):
        raise ValueError("outputs must be a dedicated E2E Scratch child")
    if out == guarded_path(c["products_root"]) or out.is_relative_to(guarded_path(c["products_root"])):
        raise ValueError("preserve frozen products; use a separate output root")
    return c


def output_path(c, path):
    root, p = guarded_path(c["output_root"]), guarded_path(path)
    if p == root or not p.is_relative_to(root):
        raise ValueError("output must be inside the configured pipeline root")
    return p


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def provenance(c, dataset):
    return {"config": c, "config_sha256": digest(c), "data_binding": dataset.binding,
            "normalization_sha256": dataset.normalization["sha256"],
            "source_sha256": {p: sha256(REPO/p) for p in SOURCE_FILES}}


def dataset_for(c, normalization, verify=True):
    manifest = guarded_path(c["products_root"])/"WIDE_COARSE_PRODUCTS_COMPLETE.json"
    if sha256(manifest) != c["products_manifest_sha256"]:
        raise ValueError("frozen product manifest changed")
    return WideResearchDataset(c["products_root"], normalization, verify_payloads=verify)


def seed_for(seed, anchor, sample_id, stage):
    """Stable address: stage streams separate; anchor parents are NOT a full-cap draw."""
    return int(digest([int(seed), str(anchor), str(sample_id), str(stage)])[:15], 16)


def example_index(step, count, seed):
    if count < 1 or step < 0:
        raise ValueError("invalid training position")
    epoch, offset = divmod(step, count)
    return int(np.random.default_rng(seed_for(seed, epoch, "order", "shared")).permutation(count)[offset])


def build_model(c, stage, device):
    if stage not in ("coarse", "fine"):
        raise ValueError("unknown stage")
    return ConditionalFieldNet(
        condition_channels=len(COARSE_CHANNELS if stage == "coarse" else FINE_CHANNELS),
        wide_condition_channels=0 if stage == "coarse" else len(COARSE_CHANNELS),
        **c["model"],
    ).to(device)


def runtime():
    require_compute()
    if not torch.cuda.is_available():
        raise RuntimeError("train/sample need a GPU in the approved allocation")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise RuntimeError("use CUBLAS_WORKSPACE_CONFIG=:4096:8 for repeatable canaries")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0")


def tensor(value, device):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device).unsqueeze(0)


def train_update(model, optimizer, target, condition, method, generator, clip, wide_condition=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_fn = flow_matching_loss if method == "cfm" else diffusion_loss
    if method not in ("cfm", "diffusion"):
        raise ValueError("unknown objective")
    loss = loss_fn(model, target, condition, generator, wide_condition=wide_condition)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm_before_clip": float(norm)}


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "objective": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    generator.set_state(state["objective"].cpu())


def save_checkpoint(path, *, model, optimizer, generator, binding, stage, method, step, history):
    """Unique immutable checkpoints; a failed temporary write is preserved."""
    path = Path(path)
    tmp = path.with_suffix(".partial")
    with tmp.open("xb") as f:
        torch.save({"schema": "e2e-wide-checkpoint-v1", "binding": binding,
                    "stage": stage, "method": method, "step": step,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": rng_state(generator), "history": history,
                    "torch_version": str(torch.__version__), "training_ready": False}, f)
    # Hard-link publication is atomic and never overwrites an existing checkpoint.
    os.link(tmp, path)
    tmp.unlink()
    write_json(path.with_suffix(".json"), {"checkpoint_sha256": sha256(path),
               "step": step, "stage": stage, "method": method,
               "binding_sha256": digest(binding), "training_ready": False})


def load_checkpoint(path, binding, stage=None, method=None):
    """Only load this pipeline's local checkpoints (torch pickle is not untrusted input)."""
    path = guarded_path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != receipt["checkpoint_sha256"] or digest(binding) != receipt["binding_sha256"]:
        raise ValueError("checkpoint checksum/provenance mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("schema") != "e2e-wide-checkpoint-v1" or state["binding"] != binding or
            (stage is not None and state["stage"] != stage) or
            (method is not None and state["method"] != method) or
            state["torch_version"] != str(torch.__version__)):
        raise ValueError("checkpoint contract/stage/method/runtime mismatch")
    return state


def train(c, normalization, stage, method, output, stop_after=None, resume=None):
    device = runtime()
    ds = dataset_for(c, normalization)
    binding = provenance(c, ds)
    out = output_path(c, output)
    cfg = c["training"]
    stop = cfg["maximum_updates"] if stop_after is None else stop_after
    if not 1 <= stop <= cfg["maximum_updates"]:
        raise ValueError("stop must be within the registered canary budget")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = build_model(c, stage, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed_for(cfg["seed"], "fit", "noise", stage))
    step, history = 0, []
    out.mkdir(parents=True, exist_ok=False)
    if resume:
        state = load_checkpoint(output_path(c, resume), binding, stage, method)
        step, history = state["step"], state["history"]
        if not step < stop:
            raise ValueError("resume must advance, within the original total budget")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], generator)
    write_json(out/"RUN.json", {"binding": binding, "stage": stage, "method": method,
                               "resume": str(resume) if resume else None, "start_step": step,
                               "stop_step": stop, "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    start = time.monotonic()
    while step < stop:
        item = ds[example_index(step, len(ds), cfg["seed"])]
        record = train_update(model, optimizer, tensor(item[stage+"_target"], device),
                              tensor(item[stage+"_condition"], device), method, generator,
                              cfg["clip_gradient_norm"],
                              tensor(item["coarse_condition"], device) if stage == "fine" else None)
        step += 1
        record.update(step=step, anchor_id=item["anchor_id"])
        history.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        if step % cfg["checkpoint_every"] == 0 or step == stop:
            save_checkpoint(out/f"step_{step:06d}.pt", model=model, optimizer=optimizer,
                            generator=generator, binding=binding, stage=stage, method=method,
                            step=step, history=history)
    write_json(out/"CANARY_COMPLETE.json", {"updates": step, "elapsed_seconds": time.monotonic()-start,
               "stage": stage, "method": method, "history": history, "training_ready": False,
               "claim": "training-phase engineering only; no held-out calibration or model selection"})


def generate_pair(c, ds, item, coarse_model, fine_model, method, sample_id, device):
    """Observation-only ancestral sampling; no dataset target read in this function."""
    s = c["sampling"]
    kwargs = {"method": method, "steps": s["cfm_steps" if method == "cfm" else "diffusion_steps"],
              "solver": s["cfm_solver"]}
    seeds = {stage: seed_for(s["seed"], item["anchor_id"], sample_id, stage) for stage in ("coarse", "fine")}
    wide = tensor(item["coarse_condition"], device)
    coarse_z = sample_field(coarse_model, wide, generator=torch.Generator(device=device).manual_seed(seeds["coarse"]), **kwargs)
    coarse = ds.inverse_target(coarse_z[0, 0].cpu().numpy(), "coarse")
    local_coarse = coarse_to_fine(coarse, fine_side=c["fine_side"], factor=c["coarse_factor"])
    condition = item["fine_condition"].copy()
    condition[-1] = ds.normalize_targets(local_coarse, "coarse")
    fine_z = sample_field(fine_model, tensor(condition, device), wide_condition=wide,
                         generator=torch.Generator(device=device).manual_seed(seeds["fine"]), **kwargs)
    fine = ds.inverse_target(fine_z[0, 0].cpu().numpy(), "fine")
    return coarse, fine, seeds


def reconstruct(coarse, fine, cell=3.383, factor=4, core_side=32):
    """Exactly the audited wide-tensor + local-residual-tensor map, once R7 only."""
    coarse, fine = np.asarray(coarse), np.asarray(fine)
    if coarse.size > 32**3 or fine.size > 32**3:
        require_compute()
    if (fine.ndim != 3 or coarse.ndim != 3 or len(set(fine.shape)) != 1 or
            len(set(coarse.shape)) != 1 or not 0 < core_side <= fine.shape[0] or
            not np.isfinite(coarse).all() or not np.isfinite(fine).all()):
        raise ValueError("invalid generated parent")
    start = (fine.shape[0]-core_side)//2
    core = (slice(start, start+core_side),)*3
    variant = {"span_fine_cells": coarse.shape[0]*factor, "factor": factor}
    up = coarse_to_fine(coarse, fine_side=fine.shape[0], factor=factor)
    wide_t = tensor_from_delta(coarse, cell*factor)
    wide_core = np.stack([interpolate(wide_t[..., j], local_coords(variant, core_side), degree=5)
                          for j in range(6)], axis=-1)
    combined = wide_core + tensor_from_delta(fine, cell)[core]
    delta = up + fine
    closure = float(np.max(np.abs(combined[..., [0, 3, 5]].sum(axis=-1)-delta[core])))
    tolerance = 2e-6 * max(1., float(np.max(np.abs(delta[core]))))
    if not np.isfinite(combined).all() or closure > tolerance:
        raise ValueError("generated tensor trace closure failed")
    return {"delta_local96": delta, "tensor_core": combined, "eigen_core": eigs(combined),
            "trace_max_abs": closure}


def sample(c, normalization, coarse_checkpoint, fine_checkpoint, anchor, sample_id, output):
    device = runtime()
    ds = dataset_for(c, normalization)
    binding = provenance(c, ds)
    checkpoints = {"coarse": output_path(c, coarse_checkpoint), "fine": output_path(c, fine_checkpoint)}
    states = {stage: load_checkpoint(path, binding, stage=stage) for stage, path in checkpoints.items()}
    method = states["coarse"]["method"]
    if states["fine"]["method"] != method or states["coarse"]["step"] != states["fine"]["step"]:
        raise ValueError("both stages must have the same method and training budget")
    matches = [i for i, r in enumerate(ds.rows) if r["anchor_id"] == anchor]
    if len(matches) != 1:
        raise PermissionError("anchor must be a registered training parent")
    out = output_path(c, output)
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    models = {stage: build_model(c, stage, device) for stage in states}
    for stage, model in models.items():
        model.load_state_dict(states[stage]["model"])
    item = ds.inference_conditions(matches[0])
    coarse, fine, seeds = generate_pair(c, ds, item, models["coarse"], models["fine"], method, sample_id, device)
    result = reconstruct(coarse, fine)
    identity = {"anchor_id": anchor, "sample_id": str(sample_id), "seeds": seeds, "method": method,
                "checkpoint_sha256": {k: sha256(v) for k,v in checkpoints.items()},
                "binding_sha256": digest(binding), "training_ready": False,
                "coherence": "shared wide/local parent and all its child crops; not cross-anchor full-cap coherence"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "x") as f:
        f.attrs["schema"] = "e2e-wide-sample-v1"
        f.attrs["identity_json"] = json.dumps(identity, sort_keys=True)
        for name, value in {"coarse_delta": coarse, "fine_residual": fine, **result}.items():
            f.create_dataset(name, data=value)
    write_json(out.with_suffix(".json"), {**identity, "sample_sha256": sha256(out),
               "trace_max_abs": result["trace_max_abs"], "complete": True})


def diagnose(c, normalization, samples, output):
    """Per-draw training diagnostics, deliberately NOT coverage/SBC/TARP or selection."""
    require_compute()
    ds = dataset_for(c, normalization)
    binding = provenance(c, ds)
    cfg = json.loads((REPO/c["diagnostic_config"]).read_text())
    rows = {r["anchor_id"]: r for r in ds.rows}
    records = []
    core = (slice(32,64),)*3
    seen = set()
    for path in samples:
        p = output_path(c, path)
        receipt = json.loads(p.with_suffix(".json").read_text())
        if sha256(p) != receipt["sample_sha256"] or receipt["binding_sha256"] != digest(binding):
            raise ValueError("sample checksum/provenance mismatch")
        with h5py.File(p, "r") as f:
            identity = json.loads(f.attrs["identity_json"])
            if any(receipt.get(k) != v for k,v in identity.items()):
                raise ValueError("sample/receipt identity mismatch")
            key = (identity["anchor_id"], identity["sample_id"], identity["method"])
            if key in seen or identity["anchor_id"] not in rows:
                raise ValueError("duplicate or non-training sample")
            seen.add(key)
            predicted = f["tensor_core"][:]
        row = rows[identity["anchor_id"]]
        with h5py.File(guarded_path(row["shard"]), "r") as f:
            g = f[row["group"]]
            truth = g["tensor_spectral"][core]
            mask = g["masks/observed_parent"][core].astype(bool)
        masks = {}
        for name, support in (("complete_core", np.ones((32,)*3, bool)), ("observed_core", mask)):
            if not support.any():
                raise ValueError("empty diagnostic support")
            masks[name] = {"metrics": tensor_metrics(predicted, truth, support),
                           "draw": science(eigs(predicted), support, c["fine_cell_mpc_h"], cfg),
                           "truth": science(eigs(truth), support, c["fine_cell_mpc_h"], cfg)}
        records.append({"identity": identity, "phase": row["phase"], "cap": row["cap"],
                        "shell": row["shell"], "support_stratum": row["support_stratum"], "masks": masks})
    write_json(output_path(c, output), {"schema": "e2e-wide-training-diagnostics-v1", "records": records,
               "training_ready": False, "calibration_pass": None,
               "claim": "per-draw training checks, not posterior calibration; three phase units, correlated anchors/voxels"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="fit verified training-only normalization on compute")
    prepare.add_argument("--output", required=True, type=Path)
    for name in ("train", "sample", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--normalization", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        if name == "train":
            p.add_argument("--stage", choices=("coarse", "fine"), required=True)
            p.add_argument("--method", choices=("cfm", "diffusion"), required=True)
            p.add_argument("--stop-after", type=int)
            p.add_argument("--resume", type=Path)
        elif name == "sample":
            p.add_argument("--coarse-checkpoint", required=True, type=Path)
            p.add_argument("--fine-checkpoint", required=True, type=Path)
            p.add_argument("--anchor", required=True)
            p.add_argument("--sample-id", required=True)
        else:
            p.add_argument("--samples", nargs="+", type=Path, required=True)
    args = vars(parser.parse_args())
    c = read_config(args.pop("config"))
    command = args.pop("command")
    if command == "prepare":
        require_compute()
        out = output_path(c, args["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        fit_normalization(dataset_for(c, None, verify=False), out)
    else:
        args["normalization"] = output_path(c, args["normalization"])
        {"train": train, "sample": sample, "diagnose": diagnose}[command](c, **args)


if __name__ == "__main__":
    main()
