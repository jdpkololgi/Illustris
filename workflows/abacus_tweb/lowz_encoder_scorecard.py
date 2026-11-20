"""Read small, atomic trainer reports, matched at one epoch, without opening sealed data.

Example: python -m workflows.abacus_tweb.lowz_encoder_scorecard --epoch 2 RUN_DIR ...
Each report comes from evaluate_complete_phase, which rejects duplicate/missing
galaxy IDs. This collects that scorer's output; it does not rerun inference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p12a_lowz_20261029_v1/training_contract_v1")
DEV = ("ph006", "ph012", "ph013")
SHELLS = ("0p01_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
SCORER = "workflows/abacus_tweb/p8_deterministic_common.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def arm(m):
    name = m["frozen_arguments"]["run_name"]
    src = m["source_contract"]
    if m["model"] == "unet":
        for arch in ("large_kernel", "window_attention"):
            if name.startswith(f"sweep_{arch}_") and "workflows/abacus_tweb/p12a_encoder_sweep.py" in src:
                return arch
        if name.startswith("litept_s_") and "workflows/abacus_tweb/lowz_litept_train.py" in src:
            return "litept_s"
        if name == "full" and not any("encoder_sweep" in k or "litept" in k for k in src):
            return "p12a"
    if (m["model"] == "graph" and name.startswith("lowz_gpatch_") and m["graph_transform_sha256"]
            and "workflows/abacus_tweb/lowz_gpatch_train.py" in src):
        return "inductive_gpatch"
    raise ValueError(f"unregistered encoder: {name}")


def reduce_report(report, counts, epoch, step):
    require(report["complete_phase_coverage"] and tuple(report["validation_phases"]) == DEV
            and set(report["per_phase"]) == set(DEV), "wrong development phases")
    phases = {}
    for phase in DEV:
        p = report["per_phase"][phase]
        rt = p["runtime"]
        require(p["complete_phase_coverage"] and p["validation_phase"] == phase
                and rt["patch_failures"] == 0 and (rt["epoch"], rt["global_step"]) == (epoch, step),
                f"{phase}: incomplete or unmatched checkpoint")
        require(set(p["per_shell"]) == set(SHELLS), f"{phase}: wrong shells")
        require(p["n_authoritative"] == sum(counts[phase]), f"{phase}: wrong total")
        rows = {}
        for shell, n in zip(SHELLS, counts[phase], strict=True):
            s = p["per_shell"][shell]
            scores = [s[f"lambda{i}"]["r2"] for i in (1, 2, 3)]
            require(s["n"] == n and n > 2 and all(math.isfinite(x) and x <= 1 for x in scores),
                    f"{phase} {shell}: invalid rows or scores")
            rows[shell] = {"n": n, "r2": scores}
        phases[phase] = rows
    macro = [fmean(phases[p][s]["r2"][i] for p in DEV for s in SHELLS) for i in range(3)]
    require(report["n_authoritative"] == sum(sum(ns) for ns in counts.values())
            and math.isclose(macro[0], report["primary_macro_r2_lambda1"], abs_tol=1e-12, rel_tol=0),
            "score aggregation mismatch")
    return phases, macro


def scorecard(directories, epoch):
    ready_file = ROOT / "TRAINING_LOADER_READY.json"
    raw = ready_file.read_bytes()
    ready = json.loads(raw)
    roles = ready["roles"]
    require(ready["pass"] and tuple(roles["validation_and_selection"]) == DEV and
            set(roles["sealed_blind_test"]) == {"ph018", "ph019"}, "unexpected split")
    counts = {p: [int(v[str(i)]) for i in range(4)] for p in DEV
              for v in [read(ROOT / "phases" / p / "phase_contract.json")["shell_counts"]]}
    models, bindings = {}, []
    for d in directories:
        m = read(d / "run_manifest.json")
        arch = arm(m)
        require(arch not in models and not m["phase_is_model_input"] and not m["blind_truth_accessed"]
                and m["training_phases"] == roles["training"]
                and m["validation_and_selection_phases"] == list(DEV)
                and m["training_ready_marker_sha256"] == hashlib.sha256(raw).hexdigest(),
                f"{arch}: duplicate or unmatched contract")
        step = epoch * m["training_cores_per_epoch"]
        h = [json.loads(s) for s in (d / "epoch_history.jsonl").read_text().splitlines()]
        rows = [row for row in h if row["epoch"] == epoch]
        require(len(rows) == 1, f"{arch}: no unique completed epoch {epoch}")
        h = rows[0]
        require(h["complete_epoch_coverage"] and h["global_step"] == step and h["repeat_cores"] == 0
                and h["unique_cores_seen"] == m["training_cores_per_epoch"], "incomplete epoch")
        raw_report = (d / "best_validation_report.json").read_bytes()
        p, macro = reduce_report(json.loads(raw_report), counts, epoch, step)
        require(math.isclose(macro[0], h["primary_macro_r2_lambda1"], abs_tol=1e-12, rel_tol=0),
                "epoch history/report mismatch")
        bindings.append((step, m["seed"], h["epoch_sha256"], m["source_contract"][SCORER],
                         m["objective"], m["target_scaler_sha256"], m["scheduler_total_updates"]))
        models[arch] = dict(macro_r2=macro, phases=p, source_run=str(d.resolve()),
                            report_sha256=hashlib.sha256(raw_report).hexdigest())
    require(len(set(bindings)) == 1, "models have mismatched training/evaluation bindings")
    step, seed, order, scorer, objective, scaler, schedule = bindings[0]
    return dict(epoch=epoch, step=step, seed=seed, epoch_order_sha256=order, scorer_sha256=scorer,
                target_scaler_sha256=scaler, objective=objective, schedule_updates=schedule,
                training_ready_sha256=hashlib.sha256(raw).hexdigest(), eigenvalues=["lambda1","lambda2","lambda3"],
                models=models, ranking_lambda1=(sorted(models, key=lambda name: -models[name]["macro_r2"][0])
                                                 if len(models) > 1 else None))


def merge(cards):
    """Combine trusted saved outputs from this reader after best-checkpoint turnover."""
    first = cards[0]
    keys = ("epoch", "step", "seed", "epoch_order_sha256", "scorer_sha256", "target_scaler_sha256",
            "objective", "schedule_updates", "training_ready_sha256", "eigenvalues")
    models = {}
    expected_counts = None
    for c in cards:
        require(all(c[k] == first[k] for k in keys), "saved scorecards have unmatched bindings")
        for name, m in c["models"].items():
            require(name not in models, f"duplicate scorecard for {name}")
            phases = m["phases"]
            require(set(phases) == set(DEV) and all(set(p) == set(SHELLS) for p in phases.values()),
                    "incomplete saved phase/shell coverage")
            counts = [phases[p][s]["n"] for p in DEV for s in SHELLS]
            require(all(n > 2 for n in counts) and (expected_counts is None or expected_counts == counts),
                    "mismatched saved shell counts")
            expected_counts = counts
            for i in range(3):
                vals = [phases[p][s]["r2"][i] for p in DEV for s in SHELLS]
                require(all(math.isfinite(v) and v <= 1 for v in vals) and math.isclose(
                    fmean(vals), m["macro_r2"][i], rel_tol=0, abs_tol=1e-12), "invalid saved macro")
            models[name] = m
    return {**{k: first[k] for k in keys}, "models": models, "ranking_lambda1":
            sorted(models, key=lambda n: -models[n]["macro_r2"][0]) if len(models) > 1 else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--saved", type=Path, action="append", default=[],
                        help="Previously frozen, trusted output of this reader (repeatable)")
    parser.add_argument("directories", nargs="*", type=Path)
    a = parser.parse_args()
    cards = [read(p) for p in a.saved]
    if a.directories:
        cards.append(scorecard(a.directories, a.epoch))
    require(cards and all(c["epoch"] == a.epoch for c in cards), "need matching epoch scorecards")
    print(json.dumps(merge(cards), indent=2, allow_nan=False))
