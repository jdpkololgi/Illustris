"""Small shared checks for consumers of the corrected galaxy-position truth."""
from pathlib import Path
import json

import fitsio
import numpy as np

from workflows.abacus_tweb.galaxy_position_mapping import MAPPING_VERSION
from workflows.abacus_tweb.galaxy_position_validation_status import LABEL_POLICY_VERSION
from workflows.abacus_tweb.product_lifecycle import digest, reject_historical_input


def label_validity(table):
    """Availability is independent of the old halo lookup and INDEP_POS."""
    required = {"HAS_LABEL", "TWEB_STATUS", "LAMBDA1", "LAMBDA2", "LAMBDA3"}
    if not required.issubset(table.dtype.names or ()):
        raise ValueError("Corrected targets lack label-status columns")
    eigen = np.column_stack([table[k] for k in ("LAMBDA1", "LAMBDA2", "LAMBDA3")])
    return ((table["HAS_LABEL"] == 1) & (table["TWEB_STATUS"] == 1)
            & np.isfinite(eigen).all(axis=1)
            & (eigen[:, 0] <= eigen[:, 1]) & (eigen[:, 1] <= eigen[:, 2]))


def verify_observed_truth(path, phase):
    """Bind a downstream build to a complete corrected join and annotation."""
    path = Path(path).resolve()
    reject_historical_input(path)
    receipt_path = Path(str(path) + ".complete.json")
    rec = json.loads(receipt_path.read_text())
    if (rec.get("schema_version") != "p10-observed-truth-complete-v3"
            or rec.get("phase") != phase
            or rec.get("join", {}).get("label_complete") is not True
            or Path(rec["output"]["path"]).resolve() != path
            or rec["output"]["sha256"] != digest(path)):
        raise ValueError("Observed truth is not a complete, hash-bound v2 replacement")
    annotation_path = Path(rec["annotated_parent"]["path"] + ".manifest.json")
    if digest(annotation_path) != rec["annotation_receipt_sha256"]:
        raise ValueError("Observed truth annotation receipt changed")
    annotation = json.loads(annotation_path.read_text())
    if (annotation.get("completed") is not True or annotation.get("phase") != phase
            or annotation.get("label_policy_version") != LABEL_POLICY_VERSION
            or annotation.get("input_rows_fully_labelled") is not True
            or annotation.get("source_rows_verified") is not True):
        raise ValueError("Observed truth lacks the all-population annotation contract")
    with fitsio.FITS(path) as f:
        header = f[1].read_header()
        if (header.get("TWPOLICY") != LABEL_POLICY_VERSION
                or header.get("TWMAP") != MAPPING_VERSION
                or header.get("SIMPHASE") != phase
                or f[1].get_nrows() != rec["output"]["rows"]):
            raise ValueError("Observed truth header/row-count mismatch")
    return {
        "label_policy_version": LABEL_POLICY_VERSION,
        "mapping_version": MAPPING_VERSION,
        "observed_receipt": str(receipt_path),
        "observed_receipt_sha256": digest(receipt_path),
        "observed_sha256": rec["output"]["sha256"],
        "annotation_receipt_sha256": rec["annotation_receipt_sha256"],
        "validity": "HAS_LABEL=1 and TWEB_STATUS=1 and finite ordered eigenvalues",
        "independent_status_masks_truth": False,
    }
