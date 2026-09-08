#!/usr/bin/env python3
"""Explicit additive D2 native-galaxy support amendment; frozen fields unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import healpy as hp
import numpy as np

SCHEMA="p12f3-d2-native-galaxy-support-amendment-v1"
RULE="native-random-angular-cap-and-radial-support-at-authoritative-galaxy-v1"
SOURCE=Path("/global/u2/d/dkololgi/TNG/Illustris_d2_467f442")
REVISION="467f442c5c54864658fdfaf948335d6e11a647fe"


def sha(path):
    path=Path(path)
    if "ph001" in str(path) or "ph001" in str(path.resolve()):
        raise PermissionError("native amendment never opens ph001")
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def native_membership(xyz,cap_id,angular_support,domain,selection):
    xyz=np.asarray(xyz,dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError("invalid native galaxy coordinates")
    if not len(xyz):return np.zeros(0,dtype=bool)
    radius=np.linalg.norm(xyz,axis=1)
    pixel=hp.vec2pix(256,*xyz.T,nest=False)
    z=np.interp(radius,selection["cosmology"]["radius_grid_mpc"],selection["cosmology"]["redshift_grid"])
    radial=(z>=.10)&(z<.60)&~((z>=.585)&(z<.595))
    return angular_support[pixel]&(domain[pixel]//2==cap_id)&radial


def require_native_record(record):
    coordinates=np.asarray(record["galaxy_frac_index_local"])
    value=record.get("galaxy_native_support")
    if value is None:
        raise RuntimeError("missing verified native galaxy membership")
    support=np.asarray(value)
    if support.dtype!=np.bool_ or support.shape!=(len(coordinates),) or not support.all():
        raise RuntimeError("derived calibration includes a native-footprint-unsupported galaxy")
    if record.get("galaxy_support_rule")!=RULE:
        raise RuntimeError("native galaxy support rule is not verified")


class NativeSupport:
    def __init__(self,path,evaluator=None):
        self.path=Path(path);self.sha256=sha(path);self.marker=read(path)
        m=self.marker
        if (m.get("schema_version")!=SCHEMA or m.get("rule")!=RULE or
            m.get("user_authorized") is not True or m.get("retain_all_galaxies") is not True or
            m.get("voxel_masks_unchanged") is not True or m.get("ph001_access") is not False):
            raise RuntimeError("unsafe native-support amendment")
        for key,row in m["files"].items():
            if sha(row["path"])!=row["sha256"]:raise RuntimeError(f"amendment source changed: {key}")
        if evaluator is not None and Path(evaluator).resolve()!=Path(m["files"]["evaluator"]["path"]).resolve():
            raise RuntimeError("unregistered amended evaluator")
        if Path(__file__).resolve()!=Path(m["files"]["helper"]["path"]).resolve():
            raise RuntimeError("unregistered support verifier")
        base=read(m["files"]["base_contract"]["path"])
        if base["frozen_digest"]!=m["base_contract_digest"] or base["git_revision_at_freeze"]!=REVISION:
            raise RuntimeError("base D2 contract changed")
        construction=read(m["files"]["construction_audit"]["path"])
        if construction["cross_tab"]!={"native_0_voxel_0":0,"native_0_voxel_1":0,"native_1_voxel_0":736,"native_1_voxel_1":132962}:
            raise RuntimeError("full-panel construction audit changed")
        self.manifest=read(m["files"]["adapter_manifest"]["path"])
        self.selection=read(m["files"]["selection"]["path"])
        with np.load(m["files"]["angular_map"]["path"],allow_pickle=False) as data:
            self.angular=data["support"].astype(bool);self.domain=data["domain"]
        self.arrays={key:np.load(m["files"][key]["path"],mmap_mode="r") for key in
            ("core_voxel_start","core_active_offsets","core_active_parent","core_active_frac_index","core_cap")}
        self.counts={}

    def verify_record(self,record,core,method):
        core=int(core);a=self.arrays
        lo,hi=a["core_active_offsets"][core:core+2]
        parents=a["core_active_parent"][int(lo):int(hi)]
        frac=a["core_active_frac_index"][int(lo):int(hi)]
        bounds=np.asarray(record["core_bounds"])
        context_start=a["core_voxel_start"][core]-bounds[0]
        coordinates=np.asarray(record["galaxy_frac_index_local"],dtype=np.float64)
        if coordinates.shape!=frac.shape or not np.array_equal(coordinates.astype(np.float32),(frac-context_start).astype(np.float32)):
            raise RuntimeError(f"native amendment galaxy identity/order mismatch: {method}/{core}")
        cap=int(a["core_cap"][core]);geo=self.manifest["caps"]["NGC" if cap==1 else "SGC"]
        xyz=np.asarray(geo["origin_mpc"])+float(geo["cell_mpc"])*(frac+.5)
        native=native_membership(xyz,cap,self.angular,self.domain,self.selection)
        record["galaxy_native_support"]=native
        record["galaxy_support_rule"]=RULE
        require_native_record(record)
        mask=np.asarray(record["support"],dtype=bool)
        if len(coordinates):
            if not np.isfinite(coordinates).all() or np.any(coordinates<0) or np.any(coordinates>np.array(mask.shape)-1):
                raise RuntimeError("native amendment cannot excuse patch-coordinate truncation")
            nearest=np.rint(coordinates).astype(int)
            unsupported=int(np.sum(~mask[tuple(nearest.T)]))
        else:unsupported=0
        if core in self.counts.setdefault(method,{}):raise RuntimeError("duplicate native support core")
        self.counts[method][core]=dict(rows=len(parents),nearest_voxel_m0=unsupported)

    def summary(self,require_complete=False):
        result={method:dict(cores=len(rows),galaxies=sum(r["rows"] for r in rows.values()),
            nearest_voxel_m0=sum(r["nearest_voxel_m0"] for r in rows.values()),native_unsupported=0)
            for method,rows in self.counts.items()}
        if require_complete and any(row!={"cores":256,"galaxies":133698,"nearest_voxel_m0":736,"native_unsupported":0} for row in result.values()):
            raise RuntimeError("amended all-galaxy panel counts changed")
        return dict(rule=RULE,amendment=str(self.path.resolve()),amendment_sha256=self.sha256,
            methods=result,retained_all_galaxies=True,voxel_masks_unchanged=True)


def freeze(args):
    root=args.root;phase_root=root.parents[1]
    adapter=phase_root/"training_contract_r1_random/adapters/ph006/field"
    recovery=root/"recovery_20260906"
    source=Path(__file__).resolve().parent
    paths=dict(base_contract=root/"D2_CONTRACT_FROZEN.json",
        geometry_audit=recovery/"SUPPORT_GEOMETRY_57986464.json",
        construction_audit=recovery/"construction_57986464/SUPPORT_CONSTRUCTION.json",
        support_panel=recovery/"construction_57986464/SUPPORT_PANEL.npz",
        helper=Path(__file__).resolve(),evaluator=source/"p12f3_d2_evaluate_native_support.py",
        launcher=source/"submit_p12f3_d2_native_support.slurm",
        original_evaluator=SOURCE/"workflows/sbi/p12f3_d2_evaluate.py",
        adapter_manifest=adapter/"adapter_manifest.json",
        selection=phase_root/"training_contract/transforms/field/selection_manifest.json",
        angular_map=phase_root/"ph006/p3b_random_response_v1/angular/randoms_n18.npz")
    paths.update({key:adapter/(key+".npy") for key in
        ("core_voxel_start","core_active_offsets","core_active_parent","core_active_frac_index","core_cap")})
    marker=dict(schema_version=SCHEMA,rule=RULE,user_authorized=True,
        authorization="2026-09-06 user approved retaining all galaxies with native-footprint check after full-panel audit",
        retain_all_galaxies=True,voxel_masks_unchanged=True,field_targets_unchanged=True,
        thresholds_unchanged=True,ph001_access=False,base_contract_digest=read(paths["base_contract"])["frozen_digest"],
        files={key:dict(path=str(path.resolve()),sha256=sha(path)) for key,path in paths.items()})
    target=root/"D2_NATIVE_SUPPORT_AMENDMENT.json"
    if target.exists():
        if read(target)!=marker:raise RuntimeError("existing amendment differs")
    else:
        with target.open("x") as stream:json.dump(marker,stream,indent=2);stream.write("\n")
    NativeSupport(target)
    print(json.dumps(dict(amendment=str(target),sha256=sha(target)),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--root",type=Path)
    parser.add_argument("--amendment",type=Path)
    parser.add_argument("--verify-evaluations",nargs=2,type=Path)
    parser.add_argument("--require-smoke",type=Path)
    args=parser.parse_args()
    if args.freeze:freeze(args);return
    native=NativeSupport(args.amendment)
    if args.require_smoke:
        smoke=read(args.require_smoke)
        if smoke.get("pass") is not True or smoke.get("native_support",{}).get("amendment_sha256")!=native.sha256:
            raise RuntimeError("native evaluation requires the matching real-patch smoke")
    if args.verify_evaluations:
        for path,nfe in zip(args.verify_evaluations,(50,100)):
            marker=read(path)
            if marker.get("native_support_amendment_sha256")!=native.sha256 or marker["frozen"].get("native_support_amendment_sha256")!=native.sha256 or marker["network_evaluations"]!=nfe:
                raise RuntimeError("decision input lacks matching native-support amendment")
    print(json.dumps(dict(verified=True,amendment_sha256=native.sha256)))


if __name__=="__main__":main()
