"""Build one auditable shard per omitted phase, then join once."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from workflows.sbi.lowz_development_table import base, full, read, sample, concatenate, sha256, FEATURE_NAMES

out = Path(os.environ["FMPE_RUN"])
phases = read(full / "TRAINING_LOADER_READY.json")["roles"]["training"]
assert len(set(phases)) == 20 and not set(phases) & {"ph006","ph012","ph013","ph018","ph019"}

def publish(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

if __name__ == "__main__":
    if sys.argv[1] == "shard":
        i = int(os.environ["SLURM_PROCID"]) + 10*int(sys.argv[2])
        p = phases[i]
        rows, audit = sample(p,i,training=True)
        path = out / (p+".npz")
        np.savez(path, **rows)
        publish(out/(p+".json"), dict(audit,sha256=sha256(path)))
    else:
        shards = []
        receipts = {}
        for i,p in enumerate(phases):
            path = out/(p+".npz")
            receipt = read(out/(p+".json"))
            assert receipt["sha256"] == sha256(path) and receipt["low_z_rows"] > 0
            with np.load(path) as f:
                rows = dict(f)
            assert rows["context"].shape == (100000,7) and (rows["phase_index"] == i).all()
            assert np.isfinite(rows["theta_softplus"]).all()
            shards.append(rows)
            receipts[p] = receipt
        path = out/"training_sample.npz"
        np.savez(path, **concatenate(shards))
        dev = base/"p12a_development_table_62094904/selection_sample.npz"
        assert sha256(dev) == "9f792dd000cadffdd72e5270ffa3d6bf172e6f3704aa31006b965acb0b86fef9"
        publish(out/"P12A_DATASET_READY.json",dict(training=dict(path=str(path),sha256=sha256(path),rows=2000000),
            validation=dict(path=str(dev)),features=FEATURE_NAMES,phases=receipts,sealed_phase_opened=False,**{"pass":True}))
