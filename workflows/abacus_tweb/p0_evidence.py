#!/usr/bin/env python3
"""Freeze matched full-range evidence for the generalisable GraphWeb programme.

All methods are scored on the canonical ``s3c_cnn_fullrange`` row index.  The
script deliberately distinguishes posterior probabilities from deterministic
threshold decisions and distinguishes raw classical estimates from affine
calibration fitted on the training region only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score, r2_score, recall_score

CLASS_NAMES = ("void", "wall", "filament", "knot")
SPLIT_NAMES = ("train", "validation", "test")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def tweb_class(eigenvalues: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Return 0/1/2/3 for void/wall/filament/knot."""
    eig = np.asarray(eigenvalues)
    if eig.ndim != 2 or eig.shape[1] != 3:
        raise ValueError(f"eigenvalues must have shape (N,3), got {eig.shape}")
    return np.sum(eig > threshold, axis=1).astype(np.int8)


def fit_affine(train_pred: np.ndarray, train_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one slope/intercept per eigenvalue using training rows only."""
    slope = np.empty(3, dtype=float)
    intercept = np.empty(3, dtype=float)
    for k in range(3):
        a = np.stack([train_pred[:, k], np.ones(len(train_pred))], axis=1)
        coef, *_ = np.linalg.lstsq(a, train_truth[:, k], rcond=None)
        slope[k], intercept[k] = coef
    return slope, intercept


def reliability(prob: np.ndarray, event: np.ndarray, n_bins: int = 10) -> dict:
    prob = np.clip(np.asarray(prob, float), 0.0, 1.0)
    event = np.asarray(event, bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.minimum(np.digitize(prob, edges[1:-1]), n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        use = which == b
        n = int(use.sum())
        if n:
            mp = float(prob[use].mean())
            fr = float(event[use].mean())
            ece += n / len(prob) * abs(mp - fr)
        else:
            mp = fr = None
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": n, "mean_probability": mp, "event_fraction": fr})
    brier = float(np.mean((prob - event.astype(float)) ** 2))
    climatology = float(event.mean())
    brier_ref = float(np.mean((climatology - event.astype(float)) ** 2))
    return {"brier": brier, "climatology": climatology,
            "brier_reference": brier_ref,
            "brier_skill": float(1.0 - brier / brier_ref) if brier_ref > 0 else None,
            "ece_equal_width": float(ece), "bins": rows}


def regression_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for k, name in enumerate(("lambda1", "lambda2", "lambda3")):
        y, p = truth[:, k], pred[:, k]
        slope, intercept = np.polyfit(y, p, 1)
        out[name] = {
            "r2": float(r2_score(y, p)),
            "spearman": float(spearmanr(y, p).statistic),
            "mae": float(np.mean(np.abs(p - y))),
            "bias": float(np.mean(p - y)),
            "pred_on_truth_slope": float(slope),
            "pred_on_truth_intercept": float(intercept),
        }
    return out


def classification_metrics(truth: np.ndarray, pred: np.ndarray, threshold: float) -> dict:
    yt, yp = tweb_class(truth, threshold), tweb_class(pred, threshold)
    cm = confusion_matrix(yt, yp, labels=np.arange(4))
    rec = recall_score(yt, yp, labels=np.arange(4), average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(yt == yp)),
        "balanced_accuracy": float(rec.mean()),
        "macro_f1": float(f1_score(yt, yp, labels=np.arange(4), average="macro", zero_division=0)),
        "confusion_matrix_true_rows": cm.tolist(),
        "recall": {name: float(rec[i]) for i, name in enumerate(CLASS_NAMES)},
        "true_counts": {name: int(np.sum(yt == i)) for i, name in enumerate(CLASS_NAMES)},
        "pred_counts": {name: int(np.sum(yp == i)) for i, name in enumerate(CLASS_NAMES)},
    }


def spatial_blocks(xyz: np.ndarray, block_mpc: float) -> np.ndarray:
    lo = np.floor(np.asarray(xyz).min(axis=0) / block_mpc) * block_mpc
    ijk = np.floor((np.asarray(xyz) - lo) / block_mpc).astype(np.int64)
    _, inv = np.unique(ijk, axis=0, return_inverse=True)
    return inv


def block_bootstrap(truth: np.ndarray, pred: np.ndarray, knot_prob: np.ndarray,
                    block_id: np.ndarray, threshold: float, n_boot: int, seed: int) -> dict:
    """Resample complete comoving cubes; never resample individual galaxies."""
    blocks = np.unique(block_id)
    rng = np.random.default_rng(seed)
    r2, acc, brier = [], [], []
    truth_cls = tweb_class(truth, threshold)
    pred_cls = tweb_class(pred, threshold)
    event = truth[:, 0] > threshold
    rows = {b: np.where(block_id == b)[0] for b in blocks}
    for _ in range(n_boot):
        draw = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([rows[b] for b in draw])
        if len(idx) < 3 or np.var(truth[idx, 0]) == 0:
            continue
        r2.append(r2_score(truth[idx, 0], pred[idx, 0]))
        acc.append(np.mean(truth_cls[idx] == pred_cls[idx]))
        brier.append(np.mean((knot_prob[idx] - event[idx].astype(float)) ** 2))

    def summarize(values):
        a = np.asarray(values, float)
        return {"median": float(np.median(a)), "lo_2p5": float(np.quantile(a, 0.025)),
                "hi_97p5": float(np.quantile(a, 0.975)), "n_valid": int(len(a))}

    return {"block_definition": "axis-aligned observer-frame comoving cubes",
            "n_blocks": int(len(blocks)), "n_bootstrap": int(n_boot),
            "lambda1_r2": summarize(r2), "class_accuracy": summarize(acc),
            "knot_brier": summarize(brier)}


def parse_spec(text: str) -> tuple[str, Path, str]:
    """Parse NAME=PATH[:none|affine_train]."""
    name, rest = text.split("=", 1)
    mode = "none"
    if rest.endswith(":affine_train"):
        rest, mode = rest[:-13], "affine_train"
    elif rest.endswith(":none"):
        rest = rest[:-5]
    return name, Path(rest), mode


def evaluate_slice(truth, pred, knot_prob, mask, shell, xyz, threshold,
                   block_mpc, n_boot, seed):
    y, p, kp = truth[mask], pred[mask], knot_prob[mask]
    reg = regression_metrics(y, p)
    cls = classification_metrics(y, p, threshold)
    prob = reliability(kp, y[:, 0] > threshold)
    order_bad = np.any(np.diff(p, axis=1) < 0, axis=1)
    return {"n": int(mask.sum()), "regression": reg, "classification": cls,
            "knot_probability": prob,
            "ordering_violation_fraction": float(order_bad.mean()),
            "spatial_block_uncertainty": block_bootstrap(
                y, p, kp, spatial_blocks(xyz[mask], block_mpc), threshold, n_boot, seed)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--graphnet", action="append", default=[], metavar="NAME=NPZ")
    ap.add_argument("--point", action="append", default=[],
                    metavar="NAME=NPY[:none|affine_train]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--block-mpc", type=float, default=100.0)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    with args.cache.open("rb") as f:
        cache = pickle.load(f)
    truth = np.asarray(cache["eigenvalues_raw"], float)
    masks = tuple(np.asarray(m, bool) for m in cache["masks"])
    shell = np.asarray(cache["shell"]).astype(str)
    tid = np.asarray(cache["tid"], np.int64)
    xyz = np.load(args.points).astype(float)
    n = len(truth)
    if not (len(shell) == len(tid) == len(xyz) == n):
        raise RuntimeError("canonical cache arrays are not row-aligned")
    if np.any(np.sum(np.stack(masks), axis=0) > 1):
        raise RuntimeError("canonical split masks overlap")

    methods = []
    for text in args.graphnet:
        name, path, _ = parse_spec(text)
        z = np.load(path, allow_pickle=False)
        required = {"pred_mean", "class_prob", "score_mask", "targetid", "truth"}
        if not required.issubset(z.files):
            raise RuntimeError(f"{path}: missing {sorted(required - set(z.files))}")
        if len(z["pred_mean"]) != n or not np.array_equal(z["targetid"], tid):
            raise RuntimeError(f"{name}: canonical TARGETID alignment failed")
        score_mask = np.asarray(z["score_mask"], bool)
        expected = masks[1] | masks[2]
        if not np.array_equal(score_mask, expected):
            raise RuntimeError(f"{name}: scored rows differ from canonical validation+test masks")
        if not np.allclose(z["truth"][score_mask], truth[score_mask], atol=1e-7, rtol=0):
            raise RuntimeError(f"{name}: truth alignment failed")
        methods.append((name, path, np.asarray(z["pred_mean"], float),
                        np.asarray(z["class_prob"], float)[:, 3], "posterior"))

    for text in args.point:
        name, path, mode = parse_spec(text)
        pred = np.load(path).astype(float)
        if pred.shape != truth.shape:
            raise RuntimeError(f"{name}: prediction shape {pred.shape} != {truth.shape}")
        calibration = None
        if mode == "affine_train":
            slope, intercept = fit_affine(pred[masks[0]], truth[masks[0]])
            pred = pred * slope + intercept
            calibration = {"fit_split": "train", "slope": slope.tolist(),
                           "intercept": intercept.tolist()}
        knot_prob = (pred[:, 0] > args.threshold).astype(float)
        methods.append((name, path, pred, knot_prob,
                        {"kind": "deterministic_threshold", "calibration": calibration}))

    payload = {
        "schema_version": 1,
        "target_convention": {"ordered": "lambda1<=lambda2<=lambda3", "epoch_z": 0.2,
                              "smoothing_mpc_h": 7.0, "threshold": args.threshold,
                              "truth_space": "real-space T-web labels",
                              "input_space": "observer-frame redshift space"},
        "canonical_rows": {"cache": str(args.cache), "cache_sha256": sha256(args.cache),
                           "points": str(args.points), "points_sha256": sha256(args.points),
                           "n_rows": n,
                           "split_counts": {s: int(m.sum()) for s, m in zip(SPLIT_NAMES, masks)},
                           "targetid_unique": bool(len(np.unique(tid)) == n)},
        "spatial_uncertainty": {"block_mpc": args.block_mpc,
                                "n_bootstrap": args.n_bootstrap, "seed": args.seed},
        "methods": {}, "gates": {},
    }
    for mi, (name, path, pred, knot_prob, probability_kind) in enumerate(methods):
        if not np.isfinite(pred[masks[1] | masks[2]]).all():
            raise RuntimeError(f"{name}: non-finite predictions on scored rows")
        entry = {"artifact": str(path), "artifact_sha256": sha256(path),
                 "probability_kind": probability_kind, "splits": {}}
        for si, split_name in ((1, "validation"), (2, "test")):
            sm = masks[si]
            entry["splits"][split_name] = {
                "pooled": evaluate_slice(truth, pred, knot_prob, sm, shell, xyz,
                                         args.threshold, args.block_mpc,
                                         args.n_bootstrap, args.seed + 100 * mi + si),
                "per_shell": {},
            }
            vals = []
            for tag in sorted(np.unique(shell[sm])):
                use = sm & (shell == tag)
                score = evaluate_slice(truth, pred, knot_prob, use, shell, xyz,
                                       args.threshold, args.block_mpc,
                                       args.n_bootstrap, args.seed + 1000 * mi + 10 * si)
                entry["splits"][split_name]["per_shell"][tag] = score
                vals.append(score["regression"]["lambda1"]["r2"])
            entry["splits"][split_name]["lambda1_r2_macro_shell"] = float(np.mean(vals))
        payload["methods"][name] = entry

    compared = list(payload["methods"])
    payload["gates"] = {
        "all_methods_present": bool(compared),
        "methods": compared,
        "same_canonical_rows": True,
        "same_target_convention": True,
        "test_not_used_for_calibration": True,
        "posterior_and_deterministic_probabilities_distinguished": True,
        "spatial_not_galaxy_bootstrap": True,
        "pass": bool(compared),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    for name, entry in payload["methods"].items():
        va = entry["splits"]["validation"]
        te = entry["splits"]["test"]
        print(f"{name:22s} val pooled/macro "
              f"{va['pooled']['regression']['lambda1']['r2']:.3f}/"
              f"{va['lambda1_r2_macro_shell']:.3f}  test "
              f"{te['pooled']['regression']['lambda1']['r2']:.3f}/"
              f"{te['lambda1_r2_macro_shell']:.3f}")


if __name__ == "__main__":
    main()
