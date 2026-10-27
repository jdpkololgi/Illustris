#!/usr/bin/env python3
"""Check one full corrected P10 catalogue and its eligibility for geometry reuse.

Catalogue validation and downstream reuse are separate results. A valid new
catalogue with different membership/masks is retained, but cannot borrow old
patches or training contracts. Historical catalogues are read only for comparison.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import sys

import fitsio
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workflows.abacus_tweb.galaxy_truth_contract import label_validity, verify_observed_truth
from workflows.abacus_tweb.p10_target_contract import stored_class_consistency
from workflows.abacus_tweb.product_lifecycle import digest
from workflows.abacus_tweb.tweb_grid_sampling import LABEL_COLUMNS

IDENTITY = ("TARGETID", "RA", "DEC", "Z")
SELECTION = ("ZWARN", "BGS_TARGET", "R_MAG_APP", "R_MAG_ABS", "G_R_REST",
             "G_R_OBS", "NTILE", "COMP_TILE", "FRACZ_TILELOCID", "FRAC_TLOBS_TILES",
             "PHOTSYS", "MASKBITS", "NOBS_G", "NOBS_R", "NOBS_Z", "GOODPRI",
             "GOODHARDLOC", "LOCATION_ASSIGNED", "FILE_NUM", "BOX_INDEX", "HALO_INDEX")


def exact(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)


def host_fold_audit(table, assignment):
    """BOX_INDEX identifies cubic galaxy rows; it is not a host-group key."""
    eligible = assignment["supervised_eligible"].astype(bool)
    ids = assignment["parent_node_id"][eligible]
    fold = assignment["fold"][eligible]
    hosts = table[ids]
    # Low-z filtered halo indices and high-z original halo indices cannot be
    # merged without an independently recovered namespace map.
    low = hosts["BOX_INDEX"] < 0
    order = np.lexsort((hosts["HALO_INDEX"], hosts["FILE_NUM"], low))
    same = ((low[order][1:] == low[order][:-1])
            & (hosts["FILE_NUM"][order][1:] == hosts["FILE_NUM"][order][:-1])
            & (hosts["HALO_INDEX"][order][1:] == hosts["HALO_INDEX"][order][:-1]))
    cross = same & (fold[order][1:] != fold[order][:-1])
    return {
        "eligible_rows": int(len(ids)), "low_branch_eligible_rows": int(low.sum()),
        "same_host_adjacent_pairs": int(same.sum()),
        "cross_fold_host_pairs": int(cross.sum()),
        "key": "branch, FILE_NUM, HALO_INDEX (not BOX_INDEX)",
        "pass": not low.any() and not cross.any(),
        "low_high_namespace_closure_required": bool(low.any()),
    }


def compare_p1(old_dir, new_dir):
    with np.load(old_dir / "canonical_index.npz") as a, np.load(new_dir / "canonical_index.npz") as b:
        checks = {key: exact(a[key], b[key]) for key in
                  ("parent_node_id", "targetid", "cap", "shell", "active", "context")}
        # valid_target is intentionally corrected. It is not an observed selection.
        checks["valid_target_changed_rows"] = (
            int(np.count_nonzero(a["valid_target"] != b["valid_target"]))
            if a["valid_target"].shape == b["valid_target"].shape else None)
    a = np.load(old_dir / "points.npy", mmap_mode="r")
    b = np.load(new_dir / "points.npy", mmap_mode="r")
    checks["points"] = a.shape == b.shape and all(
        exact(a[start:start+100000], b[start:start+100000])
        for start in range(0, len(a), 100000))
    return checks


def summarize_difference(groups, old, new, cap):
    delta = np.column_stack([new[f"LAMBDA{k}"].astype("f8") - old[f"LAMBDA{k}"] for k in (1, 2, 3)])
    changed = old["CWEB"] != new["CWEB"]
    for zname in ("Z", "Z_COSMO"):
        zbin = np.digitize(new[zname], [0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.6])
        mass = np.digitize(new["HALO_MASS"], [10, 100, 1000, 10000])
        keys = np.column_stack((zbin, new["RES"], new["CEN"], cap, mass))
        unique, inv = np.unique(keys, axis=0, return_inverse=True)
        count = np.bincount(inv)
        sums = np.column_stack([np.bincount(inv, weights=delta[:, k]) for k in range(3)])
        squares = np.column_stack([np.bincount(inv, weights=delta[:, k]**2) for k in range(3)])
        classes = np.bincount(inv, weights=changed)
        for i, key in enumerate(unique):
            row = groups[(zname, *map(int, key))]
            row["n"] += int(count[i])
            row["sum"] += sums[i]
            row["square"] += squares[i]
            row["class_changed"] += int(classes[i])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True)
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--replacement-root", type=Path, required=True)
    p.add_argument("--observed", type=Path, required=True)
    args = p.parse_args()
    if not (os.environ.get("SLURM_STEP_ID") and socket.gethostname().startswith("nid")):
        p.error("Full catalogue audit requires an approved compute step")
    if args.phase not in ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006"):
        p.error("Development/selection phases only")
    out = args.replacement_root / args.phase
    output = out / "TARGET_REPLACEMENT_COMPLETE.json"
    if output.exists():
        raise FileExistsError(output)
    old_dir = args.legacy_root / args.phase / "p1_canonical"
    new_dir = out / "p1_canonical"
    old_manifest = json.loads((old_dir / "manifest.json").read_text())
    new_manifest = json.loads((new_dir / "manifest.json").read_text())
    lineage = verify_observed_truth(args.observed, args.phase)
    if new_manifest["truth_lineage"] != lineage:
        raise ValueError("New P1 and observed catalogue lineage differ")
    old_path = Path(old_manifest["parent"])
    checks = compare_p1(old_dir, new_dir)
    groups = defaultdict(lambda: dict(n=0, sum=np.zeros(3), square=np.zeros(3), class_changed=0))
    with fitsio.FITS(old_path) as fa, fitsio.FITS(args.observed) as fb:
        a, b = fa[1], fb[1]
        n = b.get_nrows()
        same_rows = a.get_nrows() == n
        shared_selection = [k for k in SELECTION if k in a.get_colnames() and k in b.get_colnames()]
        missing_selection = sorted(set(SELECTION) - set(shared_selection))
        identical = {k: same_rows for k in (*IDENTITY, *shared_selection)}
        matched = 0
        boundary = 0
        old_columns = list(IDENTITY) + shared_selection + ["LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB"]
        old_all = a.read(columns=old_columns)
        old_order = np.argsort(old_all["TARGETID"])
        sorted_ids = old_all["TARGETID"][old_order]
        if np.any(np.diff(sorted_ids) <= 0):
            raise ValueError("Legacy TARGETIDs are not unique; cannot establish paired identity")
        with np.load(new_dir / "canonical_index.npz") as index:
            cap = index["cap"]
        for start in range(0, n, 100000):
            stop = min(n, start+100000)
            new = b[start:stop]
            if not label_validity(new).all():
                raise ValueError("Invalid corrected target row")
            if not all(np.isfinite(new[k]).all() for k in LABEL_COLUMNS):
                raise ValueError("Nonfinite eigenvalue/derivative target")
            eigen = np.column_stack([new[f"LAMBDA{k}"] for k in (1, 2, 3)])
            classes = stored_class_consistency(eigen, new["CWEB"])
            if classes["nonboundary_mismatch"].any():
                raise ValueError("Corrected class/eigenvalue mismatch")
            boundary += int(classes["boundary_ambiguous"].sum())
            if same_rows:
                old = old_all[start:stop]
                for key in identical:
                    identical[key] &= exact(old[key], new[key])
            # ph000 can have different membership. Pair only exact identities;
            # an equal TARGETID by itself is insufficient across catalogues.
            ix = np.searchsorted(sorted_ids, new["TARGETID"])
            present = ix < len(sorted_ids)
            ix = np.minimum(ix, len(sorted_ids)-1)
            old = old_all[old_order[ix]]
            paired = present
            for key in IDENTITY:
                paired &= old[key] == new[key]
            if paired.any():
                summarize_difference(groups, old[paired], new[paired], cap[start:stop][paired])
                matched += int(paired.sum())
    selection_exact = all(identical.values()) and not missing_selection
    geometry_exact = all(checks[k] for k in ("parent_node_id", "targetid", "cap", "shell", "active", "context", "points"))
    fold = {"pass": False, "reason": "Membership/selection/masks differ; P4 requires rebuild"}
    bindings = {}
    if selection_exact and geometry_exact:
        p4 = args.legacy_root / args.phase / "p4_patches"
        assignment = np.load(p4 / "active_assignment.npz")
        table = fitsio.read(args.observed, columns=["TARGETID", "FILE_NUM", "BOX_INDEX", "HALO_INDEX"])
        ids = assignment["parent_node_id"]
        if not exact(table["TARGETID"][ids], assignment["targetid"]):
            raise ValueError("P4 row identity differs from corrected catalogue")
        fold = host_fold_audit(table, assignment)
        # Bind all reused P4 payloads, including support masks and rotations.
        for path in sorted(p4.glob("*")):
            if path.is_file() and path.suffix in (".json", ".npy", ".npz"):
                bindings[str(path.resolve())] = digest(path)
    result = {
        "schema_version": "p10-galaxy-truth-replacement-v2", "phase": args.phase,
        "job_id": os.environ["SLURM_JOB_ID"],
        "catalogue_validated": True, "rows": n,
        "truth_lineage": lineage, "legacy_parent": str(old_path),
        "p1_manifest_sha256": digest(new_dir / "manifest.json"),
        "legacy_p1_manifest_sha256": digest(old_dir / "manifest.json"),
        "catalogue_identity_and_selection": identical,
        "missing_selection_columns": missing_selection,
        "p1_comparison": checks, "fold_audit": fold,
        "reused_payload_sha256": bindings,
        "paired_label_rows": matched, "class_boundary_ambiguity_rows": boundary,
        "difference_bin_edges": {
            "redshift": [0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.6],
            "halo_mass_1e10_msun_h": [10, 100, 1000, 10000],
        },
        "label_differences": [
            {"redshift_coordinate": k[0], "z_bin": k[1], "RES": k[2], "CEN": k[3],
             "cap": k[4], "mass_bin": k[5], "rows": v["n"],
             "mean_delta": (v["sum"]/v["n"]).tolist(),
             "rms_delta": np.sqrt(v["square"]/v["n"]).tolist(),
             "changed_classes": v["class_changed"]}
            for k, v in sorted(groups.items())
        ],
        # This audit alone never certifies topology/field payload reuse.
        "pass": False,
        "training_ready": False,
        "geometry_reuse_candidate": bool(selection_exact and geometry_exact),
        "next_action": "Rebind verified graph/field payloads and resolve fold audit before training.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("phase", "rows", "catalogue_validated", "geometry_reuse_candidate", "fold_audit")}), flush=True)


if __name__ == "__main__":
    main()
