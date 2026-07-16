#!/usr/bin/env python3
"""Export frozen tiled GraphNet posterior summaries on canonical P0 rows."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_BAD = ("/global/homes/d/dkololgi/.local/lib/python3.10/site-packages",
        "/global/homes/d/dkololgi/.local/lib/python3.11/site-packages",
        "/global/u2/d/dkololgi/.local/lib/python3.10/site-packages",
        "/global/u2/d/dkololgi/.local/lib/python3.11/site-packages")
for _p in _BAD:
    while _p in sys.path:
        sys.path.remove(_p)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np

from workflows.sbi.plot_flowjax_posteriors import (
    batched_sample_posterior,
    create_gnn_and_flow,
    load_flowjax_model,
)
from shared.eigenvalue_transformations import samples_to_raw_eigenvalues


def parse_model(text: str) -> tuple[str, Path]:
    name, path = text.split("=", 1)
    return name, Path(path)


def canonical_indices(cache: dict, payload: dict, split_index: int) -> np.ndarray:
    """Map an active tile core to canonical rows with a hard truth-alignment gate."""
    shell = str(payload["tile_shell"])
    lo, hi = map(float, payload["tile_ra_core"])
    mask = np.asarray(cache["masks"][split_index], bool)
    use = mask & (np.asarray(cache["shell"]).astype(str) == shell)
    ra = np.asarray(cache["ra"], float)
    use &= (ra >= lo) & (ra < hi)
    ci = np.where(use)[0]
    li = np.where(np.asarray(payload["masks"][split_index], bool))[0]
    if len(ci) != len(li):
        raise RuntimeError(f"{shell} split {split_index}: canonical/tile count {len(ci)} != {len(li)}")
    tile_truth = np.asarray(payload["eigenvalues_raw"], float)[li]
    canonical_truth = np.asarray(cache["eigenvalues_raw"], float)[ci]
    if not np.allclose(tile_truth, canonical_truth, atol=1e-7, rtol=0):
        raise RuntimeError(f"{shell} split {split_index}: canonical/tile truth order mismatch")
    return ci


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PKL")
    ap.add_argument("--tiles-dir", type=Path, required=True)
    ap.add_argument("--canonical-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    with args.canonical_cache.open("rb") as f:
        canonical = pickle.load(f)
    truth = np.asarray(canonical["eigenvalues_raw"], float)
    n = len(truth)
    expected_score = np.asarray(canonical["masks"][1], bool) | np.asarray(canonical["masks"][2], bool)
    manifest = json.loads((args.tiles_dir / "manifest.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_number, model_text in enumerate(args.model):
        name, model_path = parse_model(model_text)
        gnn_params, config, target_scaler, flow_filename, inc = load_flowjax_model(str(model_path))
        pred_mean = np.full((n, 3), np.nan, np.float32)
        class_prob = np.full((n, 4), np.nan, np.float32)
        score_mask = np.zeros(n, bool)
        print(f"[{name}] {model_path.name}; increment={inc}", flush=True)
        for tile_number, tile_meta in enumerate(manifest["tiles"]):
            if not tile_meta["val"] and not tile_meta["test"]:
                continue
            with (args.tiles_dir / tile_meta["file"]).open("rb") as f:
                payload = pickle.load(f)
            graph = payload["graph"]
            gnn, flow = create_gnn_and_flow(config, flow_filename, graph,
                                            jax.random.key(args.seed + model_number))
            emb = np.asarray(gnn.apply(gnn_params, jax.random.key(0), graph, is_training=False))
            for split_index in (1, 2):
                local = np.where(np.asarray(payload["masks"][split_index], bool))[0]
                if not len(local):
                    continue
                canonical_row = canonical_indices(canonical, payload, split_index)
                key = jax.random.fold_in(jax.random.key(args.seed + 100 * model_number),
                                         10 * tile_number + split_index)
                samples = batched_sample_posterior(flow, emb[local], args.n_samples, key)
                raw = np.stack([samples_to_raw_eigenvalues(samples[i], target_scaler, inc)
                                for i in range(len(local))], axis=0)
                pred_mean[canonical_row] = raw.mean(axis=1).astype(np.float32)
                cls = np.sum(raw > 0.2, axis=2)
                class_prob[canonical_row] = np.stack(
                    [np.mean(cls == k, axis=1) for k in range(4)], axis=1).astype(np.float32)
                if np.any(score_mask[canonical_row]):
                    raise RuntimeError(f"{name}: canonical rows scored more than once")
                score_mask[canonical_row] = True
                print(f"  {tile_meta['file']} split={split_index} n={len(local):,}", flush=True)
            del payload, graph, emb

        if not np.array_equal(score_mask, expected_score):
            missing = int(np.sum(expected_score & ~score_mask))
            extra = int(np.sum(score_mask & ~expected_score))
            raise RuntimeError(f"{name}: canonical score mask mismatch; missing={missing} extra={extra}")
        if not np.allclose(class_prob[score_mask].sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"{name}: class probabilities do not sum to one")
        provenance = {"model": str(model_path), "tiles_dir": str(args.tiles_dir),
                      "canonical_cache": str(args.canonical_cache),
                      "n_samples": args.n_samples, "seed": args.seed,
                      "threshold": 0.2, "splits": ["validation", "test"]}
        out = args.out_dir / f"{name}_canonical_predictions.npz"
        np.savez_compressed(out, pred_mean=pred_mean, class_prob=class_prob,
                            score_mask=score_mask, truth=truth.astype(np.float32),
                            targetid=np.asarray(canonical["tid"], np.int64),
                            provenance_json=np.array(json.dumps(provenance)))
        print(f"[{name}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
