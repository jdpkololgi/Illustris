"""Prepare the corrected P12-A development rows independently of OOF training."""
import json
import os
from pathlib import Path

import numpy as np
from workflows.sbi.p12_prepare_base_response_dataset import (
    FEATURE_NAMES, concatenate, sha256, softplus_coordinates, stratified_indices,
    parent_to_assignment_index,
)

base = Path("/pscratch/sd/d/dkololgi/abacus")
campaign = base / "p12a_lowz_20261029_v1"
full = campaign / "training_contract_v1"
phases = ["ph006", "ph012", "ph013"]


def read(path):
    return json.loads(path.read_text())


def sample(phase, offset, *, training=False):
    binding = campaign / "p12_crossfit_contracts" / ("omit_" + phase) if training else full
    root = binding / "phases" / phase
    marker = base / "p12a_lowz_development_summaries_61930825" / phase / "OOF_SUMMARY_COMPLETE.json"
    if training:
        marker = base / "oof_campaign_20261125/summaries" / phase / "OOF_SUMMARY_COMPLETE.json"
    summary = read(marker)
    contract = read(root / "phase_contract.json")["inputs"]
    geometry = campaign / phase / "p12a_lowz_geometry_response/GEOMETRY_RESPONSE_COMPLETE.json"
    response = read(geometry)
    roles = read(full / "TRAINING_LOADER_READY.json")["roles"]
    assert roles["validation_and_selection"] == phases
    if training:
        assert phase == roles["training"][offset] and summary["out_of_fold"]
        assert not summary["sealed_phase_opened"] and Path(summary["contract_root"]) == binding
    else:
        assert phase not in roles["training"]
    assert summary["pass"] and summary["phase"] == response["phase"] == phase
    assert summary["training_phases"] == [p for p in roles["training"] if p != phase] and summary["terminal_epochs"] == 4
    assert summary["contract_marker_sha256"] == sha256(binding / "TRAINING_LOADER_READY.json")
    assert response["canonical_index_sha256"] == read(Path(contract["p1_manifest"]))["index_sha256"]
    assert response["geometry_features_ready"] and summary["checkpoint_selection"] == "terminal_fixed_budget"
    for record, names in [(summary, ["truth","base_prediction","response","parent_node_id"]),
                          (response, ["distance","support"])]:
        for name in names:
            assert sha256(Path(record["arrays"][name])) == record["array_sha256"][name]
    arr = summary["arrays"]
    parent = np.load(arr["parent_node_id"], mmap_mode="r")
    assert len(parent) == summary["rows"] and parent.min() >= 0 and parent.max() < response["rows"]
    support = np.load(response["arrays"]["support"], mmap_mode="r")
    eligible = np.flatnonzero(support[parent])
    cov = np.load(arr["response"])
    count = 100000 if training else 200000
    chosen, weights, strata = stratified_indices(cov["shell"][eligible], count, 12142 + offset)
    idx = eligible[chosen]
    ids = parent[idx]
    assert len(np.unique(ids)) == count
    assignment = np.load(contract["assignment"])
    assert sha256(Path(contract["assignment"])) == contract["assignment_sha256"]
    row = parent_to_assignment_index(assignment, response["rows"])[ids]
    assert (row >= 0).all() and assignment["supervised_eligible"][row].all()
    cap, shell = cov["cap"][idx], cov["shell"][idx]
    assert np.array_equal(cap, assignment["cap"][row]) and np.array_equal(shell, assignment["shell"][row])
    truth = np.load(arr["truth"], mmap_mode="r")[idx]
    assert np.array_equal(truth, np.load(root/"parent_eigenvalues.npy", mmap_mode="r")[ids])
    pred = np.load(arr["base_prediction"], mmap_mode="r")[idx]
    z = cov["redshift"][idx]
    assert np.array_equal(z, np.load(root/"parent_redshift.npy", mmap_mode="r")[ids].astype(z.dtype))
    distance = np.load(response["arrays"]["distance"], mmap_mode="r")[ids]
    x = np.column_stack((pred,z,np.log(np.maximum(cov["ntilde_mpc3"][idx],1e-12)),cap,np.log1p(distance))).astype("f4")
    assert np.isfinite(x).all() and (distance >= 0).all() and ((z>=.01)&(z<.55)).all()
    assert (np.diff(truth,axis=1)>=0).all()
    rows = dict(context=x, theta_softplus=softplus_coordinates(truth), truth_eigenvalues=truth,
                base_prediction_eigenvalues=pred, parent_node_id=ids, phase_index=np.full(len(ids),offset if training else 20+offset,"u1"),
                cap=cap,shell=shell,natural_weight=weights,random_support=support[ids])
    audit = dict(summary_sha256=sha256(marker), geometry_sha256=sha256(geometry), strata=strata,
                 low_z_rows=int((z<.15).sum()), rows=len(ids), index=offset if training else 20+offset)
    print(phase, audit, flush=True)
    return rows, audit


if __name__ == "__main__":
    out = base / ("p12a_development_table_" + os.environ["SLURM_JOB_ID"])
    out.mkdir()
    results = [sample(p,i) for i,p in enumerate(phases)]
    path = out / "selection_sample.npz"
    np.savez(path, **concatenate([r for r,_ in results]))
    with np.load(path) as stored:
        assert stored["context"].shape == (600000,7)
        assert np.array_equal(np.unique(stored["phase_index"],return_counts=True)[1],[200000]*3)
    report = dict(features=FEATURE_NAMES, distance_unit="Mpc", sha256=sha256(path),
                  spatial_split_qualified=False, split_key="phase_index",
                  phases=dict(zip(phases,[a for _,a in results])), rows=600000, passed=True)
    (out/"DEVELOPMENT_TABLE_COMPLETE.json").write_text(json.dumps(report,indent=2))
