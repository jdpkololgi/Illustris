"""Corrected phase-local G-PATCH using the original Arm-A objective and complete-phase scorer."""
import argparse
import json
import os
from pathlib import Path
import sys

from workflows.abacus_tweb.lowz_graph_patch_index import CorrectedPatchAdapter
from workflows.abacus_tweb.lowz_graph_union_pilot import digest
from workflows.abacus_tweb.lowz_point_loader import DEV, ROOT, read, require, roles
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader


def bind(index_path, transform_dir, output):
    """Create a separately qualified graph view; never edit the running P12-A contract."""
    visible = roles(ROOT)
    index = read(index_path)
    transform = read(transform_dir / "graph_transform.json")
    require(set(index)==set(visible)==set(transform["applied"]) and transform["pass"]
            and transform["schema"]=="corrected-lowz-gpatch-features-v1"
            and transform["geom_index_sha256"]==digest(index_path)
            and transform["fit_phases"]==[p for p in visible if p not in DEV]
            and transform["selection_sha256"]==digest(ROOT / "transforms/field/selection_manifest.json"),
            "unbound graph products")
    markers = {}
    for p in visible:
        d = Path(index[p])
        marker = d / "PATCH_GEOMETRY_READY.json"
        r = read(marker)
        t = transform["applied"][p]
        require(r["passed"] and r["phase"]==p and r["union_sha256"]==t["union_sha256"]
                and digest(t["path"])==t["sha256"], f"{p}: graph feature product changed")
        markers[p] = digest(marker)
    output.mkdir(exist_ok=False)
    for name in ("phases","TRAINING_LOADER_READY.json"):
        (output / name).symlink_to(ROOT / name)
    (output / "transforms").mkdir()
    for name in ("field","target_scaler.json"):
        (output / "transforms" / name).symlink_to(ROOT / "transforms" / name)
    (output / "transforms/graph").symlink_to(transform_dir.resolve())
    (output / "GRAPH_VIEW_READY.json").write_text(json.dumps(dict(passed=True,
        geometry_index=str(index_path.resolve()),geometry_sha256=digest(index_path),
        graph_markers=markers,transform_sha256=digest(transform_dir / "graph_transform.json"),
        base_manifest_sha256=digest(ROOT / "TRAINING_LOADER_READY.json"))))


class Loader(P10PhaseBalancedLoader):
    def __init__(self, root, **kw):
        ready = read(root / "GRAPH_VIEW_READY.json")
        path = Path(ready["geometry_index"])
        require(not kw.get("include_blind") and ready["passed"]
                and digest(path)==ready["geometry_sha256"]
                and digest(root / "TRAINING_LOADER_READY.json")==ready["base_manifest_sha256"]
                and digest(root / "transforms/graph/graph_transform.json")==ready["transform_sha256"],
                "unqualified graph view")
        for name in ("phases","TRAINING_LOADER_READY.json","transforms/field","transforms/target_scaler.json"):
            require((root/name).resolve()==(ROOT/name).resolve(),"view is not bound to corrected P12-A inputs")
        super().__init__(root,**kw)
        self.geometry = read(path)
        self.markers = ready["graph_markers"]

    def graph_adapter(self, phase):
        if phase not in self._graph:
            require(phase in self.training_phases+self.validation_phases, "no sealed graph access")
            d = Path(self.geometry[phase])
            require(digest(d / "PATCH_GEOMETRY_READY.json")==self.markers[phase], "geometry marker changed")
            self._graph[phase] = CorrectedPatchAdapter(phase,d,self.root/"transforms/graph")
        return self._graph[phase]


def main():
    require("SLURM_STEP_ID" in os.environ, "compute step required")
    if sys.argv[1:2]==["bind"]:
        ap = argparse.ArgumentParser()
        for name in ("index","transform_dir","output"):
            ap.add_argument(name,type=Path)
        a = ap.parse_args(sys.argv[2:])
        bind(a.index,a.transform_dir,a.output)
        return
    from workflows.abacus_tweb import p10_train_arm_a as trainer
    from workflows.abacus_tweb.lowz_graph_streaming import StreamingGraphPatchNet, owned_patch
    trainer.graph_impl.GraphPatchNet = StreamingGraphPatchNet
    trainer.graph_impl.transformed_patch = owned_patch
    args = trainer.parse_args()
    require(args.model=="graph" and args.run_name.startswith("lowz_gpatch_")
            and not args.p11_dense_view and args.multitracer_view is None, "wrong arm")
    from workflows.abacus_tweb.lowz_epoch_reports import install
    install(trainer, args)
    source = trainer.source_contract
    lib = Path(__file__).resolve().parent
    def hashes():
        keys = ["lowz_gpatch_train.py","lowz_graph_patch_index.py","lowz_graph_normalize.py",
                "lowz_point_loader.py","p5_graph_patch_utils.py","lowz_epoch_reports.py","lowz_graph_attention.py",
                "lowz_graph_streaming.py"]
        return {**source(),**{f"workflows/abacus_tweb/{f}":digest(lib/f) for f in keys},
                str(args.contract_root/"GRAPH_VIEW_READY.json"):digest(args.contract_root/"GRAPH_VIEW_READY.json")}
    trainer.source_contract = hashes
    trainer.P10PhaseBalancedLoader = Loader
    trainer.main()


if __name__=="__main__":
    main()
