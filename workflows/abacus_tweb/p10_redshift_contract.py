"""Versioned observed-redshift domain shared by P12-A preparation stages.

Historical products retain their original defaults. A low-z campaign must bind
this complete contract, not just change a catalogue cut. Distances remain the
P12 Planck18-Mpc observer coordinates; this is not the CFM coordinate contract.
"""
from __future__ import annotations

import copy
import math


LEGACY = {
    "schema": "p12a-redshift-domain-v1",
    "science": [0.15, 0.55],
    "context": [0.10, 0.60],
    "positive_context_only": True,
    "sentinel": [0.585, 0.595],
    "shells": [[0.15, 0.25], [0.25, 0.35], [0.35, 0.45], [0.45, 0.55]],
    "diagnostic_edges": [0.15, 0.25, 0.35, 0.45, 0.55],
}


def validate_domain(domain: dict) -> dict:
    value = copy.deepcopy(domain)
    if value.get("schema") != "p12a-redshift-domain-v1":
        raise ValueError("Unrecognized P12-A redshift domain")
    for key in ("science", "context", "sentinel"):
        pair = value[key]
        if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or not 0 <= pair[0] < pair[1]:
            raise ValueError("Invalid redshift bounds: " + key)
    science, context, sentinel = (value[k] for k in ("science", "context", "sentinel"))
    if not context[0] <= science[0] < science[1] <= context[1]:
        raise ValueError("Science must lie inside context")
    if science[0] <= 0 or value.get("positive_context_only") is not True:
        raise ValueError("Positive observed redshifts required")
    if sentinel[0] < science[1] and sentinel[1] > science[0]:
        raise ValueError("Sentinel overlaps supervision")
    shells = value["shells"]
    if len(shells) != 4 or any(len(s) != 2 or not all(math.isfinite(x) for x in s) or s[0] >= s[1] for s in shells):
        raise ValueError("Invalid supervised shells")
    if shells[0][0] != science[0] or shells[-1][1] != science[1]:
        raise ValueError("The existing P12 encoder requires four contiguous science shells")
    if any(a[1] != b[0] for a, b in zip(shells, shells[1:])):
        raise ValueError("Supervised shells must be contiguous")
    edges = value["diagnostic_edges"]
    if (len(edges) < 2 or edges[0] != science[0] or edges[-1] != science[1]
            or not all(math.isfinite(x) for x in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError("Invalid diagnostic edges")
    return value


def domain_from(record: dict) -> dict:
    return validate_domain(record.get("redshift_contract", LEGACY))


def shell_names(domain: dict) -> tuple[str, ...]:
    return tuple(f"{lo:.2f}_{hi:.2f}".replace(".", "p")
                 for lo, hi in validate_domain(domain)["shells"])


def require_matching_domains(*records: dict) -> dict:
    domains = [domain_from(record) for record in records]
    if not domains or any(domain != domains[0] for domain in domains[1:]):
        raise ValueError("P12-A redshift domain mismatch; rebuild dependent products")
    return domains[0]
