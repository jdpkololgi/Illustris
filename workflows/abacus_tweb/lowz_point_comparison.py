"""Score frozen same-epoch P12-A, LitePT and G-PATCH physical eigenvalue exports."""
import argparse
import json
import numpy as np
from pathlib import Path
from workflows.abacus_tweb.lowz_point_loader import DEV, ROOT, array, read
from workflows.abacus_tweb.p8_deterministic_common import evaluate_complete_phase
from workflows.abacus_tweb.p10_redshift_contract import shell_names


def compare(directories):
    result={}
    for phase in DEV:
        contract=read(ROOT/"phases"/phase/"phase_contract.json")
        shells=shell_names(read(Path(contract["inputs"]["p1_manifest"]))["redshift_contract"])
        truth=array(Path(contract["target"]["path"]))
        with np.load(contract["inputs"]["assignment"]) as assignment:
            for name,d in directories.items():
                p=d/f"best_validation_{phase}"
                report=evaluate_complete_phase(phase=phase,parent_node_id=array(Path(str(p)+"_parent_node_id.npy")),
                    predicted_eigenvalues=array(Path(str(p)+"_eigenvalues.npy")),truth_by_parent=truth,
                    assignment=assignment,whole_phase_only=True,shell_names=shells)
                result.setdefault(name,{})[phase]={"rows":report["n_authoritative"],
                    "macro_lambda1":report["primary_macro_r2_lambda1"],"shell_r2":{
                    s:[report["per_shell"][s][f"lambda{i}"]["r2"] for i in (1,2,3)] for s in shells}}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arm in ("p12a","litept","gpatch"):
        parser.add_argument("--"+arm,type=Path,required=True)
    print(json.dumps(compare(vars(parser.parse_args())),indent=2,allow_nan=False))
