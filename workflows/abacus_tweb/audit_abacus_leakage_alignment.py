#!/usr/bin/env python3
"""Leakage and alignment audit for Abacus graph-feature targets.

This script is designed to diagnose large train/test performance gaps and
potential leakage/mismatch issues by running a small battery of checks:

1) Basic data integrity:
   - row counts between feature parquet and target catalog after Y1/Y5 filtering
   - finite-value checks
2) Regression sanity:
   - Random split R2
   - Grouped split R2 (by FILE_NUM if available; fallback to RA quantile blocks)
   - Permuted-target control (should collapse near 0 R2)
3) Optional CWEB classification sanity:
   - Random split balanced accuracy
   - Grouped split balanced accuracy
   - Permuted-target control

Outputs a compact JSON summary for reproducible comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitsio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


DEFAULT_GNN_META = "/pscratch/sd/d/dkololgi/abacus/graph_constructions/abacus_alpha_cugraph_gnn_metadata.json"
DEFAULT_OUTPUT = "/pscratch/sd/d/dkololgi/abacus/alignment_diagnostics/leakage_alignment_audit.json"


def _resolve_col(dtype_names, candidates):
    names = {n.upper(): n for n in dtype_names}
    for c in candidates:
        k = c.upper()
        if k in names:
            return names[k]
    raise KeyError(f"None of candidate columns {list(candidates)} found in {dtype_names[:20]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gnn-metadata-path", default=DEFAULT_GNN_META)
    p.add_argument("--catalog-path", default="", help="Optional override for source/annotated FITS catalog.")
    p.add_argument(
        "--feature-columns",
        default="Degree,Clustering,Density,Neigh Density,I_eig1,I_eig2,I_eig3",
        help="Comma-separated node feature columns from node_features parquet.",
    )
    p.add_argument("--max-rows", type=int, default=300_000, help="Random subsample cap for audit speed.")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=DEFAULT_OUTPUT)
    return p.parse_args()


def _apply_optional_y1y5_filter(table: np.ndarray) -> np.ndarray:
    names = {n.upper(): n for n in table.dtype.names}
    in_y1 = names.get("IN_Y1") or names.get("Y1")
    in_y5 = names.get("IN_Y5") or names.get("Y5")
    if in_y1 is None and in_y5 is None:
        return np.ones(len(table), dtype=bool)
    mask = np.zeros(len(table), dtype=bool)
    if in_y1 is not None:
        mask |= np.asarray(table[in_y1]) == 1
    if in_y5 is not None:
        mask |= np.asarray(table[in_y5]) == 1
    return mask


def _load_catalog_targets(tab: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    l1 = _resolve_col(tab.dtype.names, ("LAMBDA1", "L1", "EIG1", "LAM1", "LAMBDA_1"))
    l2 = _resolve_col(tab.dtype.names, ("LAMBDA2", "L2", "EIG2", "LAM2", "LAMBDA_2"))
    l3 = _resolve_col(tab.dtype.names, ("LAMBDA3", "L3", "EIG3", "LAM3", "LAMBDA_3"))
    y = np.stack(
        [
            np.asarray(tab[l1], dtype=np.float64),
            np.asarray(tab[l2], dtype=np.float64),
            np.asarray(tab[l3], dtype=np.float64),
        ],
        axis=1,
    )
    cweb = None
    names = {n.upper(): n for n in tab.dtype.names}
    if "CWEB" in names:
        cweb = np.asarray(tab[names["CWEB"]], dtype=np.int32)
    return y, cweb


def _group_ids_from_table(tab: np.ndarray) -> np.ndarray | None:
    names = {n.upper(): n for n in tab.dtype.names}
    if "FILE_NUM" in names:
        return np.asarray(tab[names["FILE_NUM"]], dtype=np.int32)
    if "RA" in names:
        # Fallback proxy grouping to reduce local leakage under random split.
        ra = np.asarray(tab[names["RA"]], dtype=np.float64)
        edges = np.quantile(ra, np.linspace(0, 1, 33))
        # Ensure strict monotonic bins.
        edges = np.maximum.accumulate(edges + np.linspace(0, 1e-9, edges.size))
        gid = np.digitize(ra, edges[1:-1], right=False).astype(np.int32)
        return gid
    return None


def _align_by_optional_filter(
    x: np.ndarray,
    y_full: np.ndarray,
    mask: np.ndarray,
    groups_full: np.ndarray | None,
    cweb_full: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    """Align feature/target rows under both common construction modes.

    Mode A: features and catalog are both full-length -> apply mask to both.
    Mode B: features are already Y1/Y5-filtered -> apply mask to catalog arrays only.
    """
    n_x = int(x.shape[0])
    n_y = int(y_full.shape[0])
    n_mask = int(mask.sum())

    meta = {
        "n_x_before": n_x,
        "n_y_before": n_y,
        "n_mask_true": n_mask,
        "alignment_mode": "unknown",
    }

    if n_x == n_y:
        # Full-length feature parquet; filter both sides.
        y = y_full[mask]
        groups = groups_full[mask] if groups_full is not None else None
        cweb = cweb_full[mask] if cweb_full is not None else None
        x_aligned = x[mask]
        meta["alignment_mode"] = "full_x_and_full_catalog_apply_mask_to_both"
    elif n_x == n_mask:
        # Feature parquet already filtered; only filter catalog-side arrays.
        x_aligned = x
        y = y_full[mask]
        groups = groups_full[mask] if groups_full is not None else None
        cweb = cweb_full[mask] if cweb_full is not None else None
        meta["alignment_mode"] = "x_prefiltered_apply_mask_to_catalog_only"
    else:
        raise ValueError(
            "Could not align features and catalog rows. "
            f"x rows={n_x:,}, y rows={n_y:,}, mask true={n_mask:,}. "
            "Expected either x==y (full/full) or x==mask.sum() (prefiltered x)."
        )

    if x_aligned.shape[0] != y.shape[0]:
        raise ValueError(f"Aligned mismatch: x={x_aligned.shape[0]:,}, y={y.shape[0]:,}")

    meta["n_after_alignment"] = int(x_aligned.shape[0])
    return x_aligned, y, groups, cweb, meta


def _regression_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, seed: int, test_frac: float) -> dict:
    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=24,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=seed,
    )

    # Random split
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_frac, random_state=seed)
    reg.fit(x_tr, y_tr)
    y_hat = reg.predict(x_te)
    r2_random = r2_score(y_te, y_hat, multioutput="raw_values")

    # Grouped split
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        tr_idx, te_idx = next(gss.split(x, y, groups=groups))
        reg.fit(x[tr_idx], y[tr_idx])
        y_hat_g = reg.predict(x[te_idx])
        r2_group = r2_score(y[te_idx], y_hat_g, multioutput="raw_values")
    else:
        r2_group = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    # Permuted target control
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y_tr.shape[0])
    reg.fit(x_tr, y_tr[perm])
    y_hat_perm = reg.predict(x_te)
    r2_perm = r2_score(y_te, y_hat_perm, multioutput="raw_values")

    return {
        "r2_random": [float(v) for v in r2_random],
        "r2_grouped": [float(v) for v in r2_group],
        "r2_permuted_control": [float(v) for v in r2_perm],
    }


def _classification_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, seed: int, test_frac: float) -> dict:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=24,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=seed,
    )

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_frac, random_state=seed, stratify=y)
    clf.fit(x_tr, y_tr)
    y_hat = clf.predict(x_te)
    bal_random = balanced_accuracy_score(y_te, y_hat)

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        tr_idx, te_idx = next(gss.split(x, y, groups=groups))
        clf.fit(x[tr_idx], y[tr_idx])
        y_hat_g = clf.predict(x[te_idx])
        bal_group = balanced_accuracy_score(y[te_idx], y_hat_g)
    else:
        bal_group = float("nan")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(y_tr.shape[0])
    clf.fit(x_tr, y_tr[perm])
    y_hat_perm = clf.predict(x_te)
    bal_perm = balanced_accuracy_score(y_te, y_hat_perm)

    return {
        "balanced_accuracy_random": float(bal_random),
        "balanced_accuracy_grouped": float(bal_group),
        "balanced_accuracy_permuted_control": float(bal_perm),
    }


def main() -> None:
    args = parse_args()
    gnn_meta_path = Path(args.gnn_metadata_path).expanduser().resolve()
    if not gnn_meta_path.exists():
        raise FileNotFoundError(f"GNN metadata not found: {gnn_meta_path}")
    with gnn_meta_path.open("r", encoding="utf-8") as f:
        gnn_meta = json.load(f)

    node_parquet = Path(gnn_meta["outputs"]["node_features"]).expanduser().resolve()
    if args.catalog_path:
        catalog_path = Path(args.catalog_path).expanduser().resolve()
    else:
        # compatible with current metadata schema
        if "source_path" in gnn_meta:
            catalog_path = Path(gnn_meta["source_path"]).expanduser().resolve()
        elif "input_metadata_path" in gnn_meta:
            with Path(gnn_meta["input_metadata_path"]).expanduser().resolve().open("r", encoding="utf-8") as f:
                inp_meta = json.load(f)
            catalog_path = Path(inp_meta["source_path"]).expanduser().resolve()
        else:
            raise KeyError("Could not resolve catalog path from metadata. Pass --catalog-path.")

    feat_cols = [c.strip() for c in args.feature_columns.split(",") if c.strip()]
    x = pd.read_parquet(node_parquet, columns=feat_cols).to_numpy(dtype=np.float64)
    tab = fitsio.read(str(catalog_path))
    mask = _apply_optional_y1y5_filter(tab)
    y, cweb = _load_catalog_targets(tab)

    groups = _group_ids_from_table(tab)
    x0 = int(x.shape[0])
    y0 = int(y.shape[0])
    x, y, groups, cweb, align_meta = _align_by_optional_filter(
        x=x,
        y_full=y,
        mask=mask,
        groups_full=groups,
        cweb_full=cweb,
    )

    rng = np.random.default_rng(args.seed)
    n = x.shape[0]
    if args.max_rows is not None and n > int(args.max_rows):
        idx = rng.choice(n, size=int(args.max_rows), replace=False)
        x = x[idx]
        y = y[idx]
        groups = groups[idx] if groups is not None else None
        cweb = cweb[idx] if cweb is not None else None

    finite_x = bool(np.isfinite(x).all())
    finite_y = bool(np.isfinite(y).all())

    reg_scores = _regression_scores(x, y, groups=groups, seed=args.seed, test_frac=args.test_frac)
    cls_scores = _classification_scores(x, cweb, groups=groups, seed=args.seed, test_frac=args.test_frac) if cweb is not None else None

    out = {
        "gnn_metadata_path": str(gnn_meta_path),
        "node_parquet": str(node_parquet),
        "catalog_path": str(catalog_path),
        "feature_columns": feat_cols,
        "rows_before_filter": {"features": x0, "targets": y0},
        "rows_after_filter": int(mask.sum()),
        "alignment": align_meta,
        "rows_used_for_audit": int(x.shape[0]),
        "grouping": "FILE_NUM" if (groups is not None and "FILE_NUM" in {n.upper() for n in tab.dtype.names}) else ("RA_quantile_bins" if groups is not None else "none"),
        "finite_checks": {"x_all_finite": finite_x, "y_all_finite": finite_y},
        "regression": reg_scores,
        "classification": cls_scores,
    }

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("=" * 72)
    print("Leakage / Alignment Audit Summary")
    print("=" * 72)
    print(f"Catalog: {catalog_path}")
    print(f"Rows used: {x.shape[0]:,}")
    print(f"Grouping: {out['grouping']}")
    print(f"Regression R2 random   : {reg_scores['r2_random']}")
    print(f"Regression R2 grouped  : {reg_scores['r2_grouped']}")
    print(f"Regression R2 permuted : {reg_scores['r2_permuted_control']}")
    if cls_scores is not None:
        print(f"CWEB bal.acc random    : {cls_scores['balanced_accuracy_random']:.4f}")
        print(f"CWEB bal.acc grouped   : {cls_scores['balanced_accuracy_grouped']:.4f}")
        print(f"CWEB bal.acc permuted  : {cls_scores['balanced_accuracy_permuted_control']:.4f}")
    print(f"Saved report: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

