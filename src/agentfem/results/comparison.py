# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only comparison of explicitly selected result quantities/histories.

No field mapping, unit conversion, resampling or scientific trust promotion.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numpy as np


def compare_records(reference: Mapping, candidate: Mapping, *,
                    quantities: Sequence[str] = (), histories: Sequence[str] = (),
                    atol: float = 0.0, rtol: float = 0.0) -> dict:
    """Compare selected observables with explicit componentwise tolerances.

    Admission requires completed result manifests, identical declared units and
    shapes, and identical history coordinates. Unknown units fail closed: use
    an explicit dimensionless unit such as ``1`` where appropriate. Metadata
    identity is reported, not equated with physical equivalence. Relative error
    at zero reference norm is unavailable; absolute error remains reported.
    """
    for name, tolerance in (("atol", atol), ("rtol", rtol)):
        if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"AFM-COMPARE-TOLERANCE: {name} must be finite and nonnegative")
    if isinstance(quantities, str) or isinstance(histories, str):
        raise ValueError("AFM-COMPARE-SELECTION: pass a sequence of names, not a string")
    quantities, histories = tuple(quantities), tuple(histories)
    if not quantities and not histories:
        raise ValueError("AFM-COMPARE-SELECTION: select at least one quantity or history")
    for selection in (quantities, histories):
        if any(not isinstance(n, str) or not n.strip() for n in selection) or len(set(selection)) != len(selection):
            raise ValueError("AFM-COMPARE-SELECTION: names must be nonempty and unique")
    for record in (reference, candidate):
        if (not isinstance(record, Mapping) or record.get("schema") != "agentfem.simulation-result"
                or record.get("schema_version") != "0.1.0"
                or record.get("status") != "completed"):
            raise ValueError("AFM-COMPARE-RECORD: require completed supported SimulationResult manifests")

    checks = []
    for kind, names, key in (("quantity", quantities, "quantity_records"),
                             ("history", histories, "history_records")):
        def index(record):
            items = record.get(key, [])
            if not isinstance(items, (list, tuple)) or any(not isinstance(x, Mapping) for x in items):
                raise ValueError("AFM-COMPARE-RECORD: malformed observable records")
            result = {}
            for item in items:
                name = item.get("name")
                if not isinstance(name, str) or name in result:
                    raise ValueError("AFM-COMPARE-RECORD: invalid or duplicate observable name")
                result[name] = item
            return result
        left, right = index(reference), index(candidate)
        for name in names:
            item = {"name": name, "kind": kind, "status": "incompatible"}
            checks.append(item)
            if name not in left or name not in right:
                item.update(code="AFM-COMPARE-MISSING", message="Observable missing from one result")
                continue
            a, b = left[name], right[name]
            fields = ("unit", "kind") if kind == "quantity" else ("unit", "abscissa_name", "abscissa_unit")
            if any(not isinstance(a.get(k), str) or not a[k].strip() or a.get(k) != b.get(k) for k in fields):
                item.update(code="AFM-COMPARE-SEMANTICS", message="Units or observable conventions missing/different")
                continue
            try:
                value_key = "value" if kind == "quantity" else "values"
                x, y = (np.asarray(r[value_key], dtype=float) for r in (a, b))
                if x.shape != y.shape or x.size == 0 or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise ValueError("Values must have matching nonempty shapes and be finite")
                if kind == "quantity" and any(tuple(r.get("shape", ())) != x.shape for r in (a, b)):
                    raise ValueError("Declared quantity shape differs from values")
                if kind == "history":
                    u, v = (np.asarray(r["abscissa"], dtype=float) for r in (a, b))
                    if (u.ndim != 1 or x.ndim == 0 or len(u) != len(x)
                            or not np.isfinite(u).all() or not np.array_equal(u, v)
                            or np.any(np.diff(u) <= 0)):
                        raise ValueError("History coordinates must match exactly and strictly increase; no automatic interpolation")
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    delta = np.abs(y - x)
                    allowance = atol + rtol * np.abs(x)
                    maximum = float(delta.max())
                    # Scaled L2 avoids overflow from squaring large finite data.
                    scale = float(np.max(np.abs(x)))
                    relative = None if scale == 0 else float(np.linalg.norm((y-x)/scale) / np.linalg.norm(x/scale))
                if relative is not None and not math.isfinite(relative):
                    raise ValueError("Relative error is not representable")
            except (KeyError, TypeError, ValueError, FloatingPointError) as error:
                item.update(code="AFM-COMPARE-DATA", message=str(error))
                continue
            item.update(status="within_tolerance" if bool(np.all(delta <= allowance)) else "outside_tolerance",
                        unit=a["unit"], maximum_absolute_error=maximum,
                        relative_l2_error=relative, shape=list(x.shape), atol=atol, rtol=rtol)
    status = ("incompatible" if any(c["status"] == "incompatible" for c in checks)
              else "outside_tolerance" if any(c["status"] == "outside_tolerance" for c in checks)
              else "within_tolerance")
    return {"schema": "agentfem.result-comparison", "schema_version": "0.1.0",
            "status": status, "checks": checks,
            "reference_trust": reference.get("trust_level"),
            "candidate_trust": candidate.get("trust_level"),
            "scientific_inputs_equal": (reference["scientific_inputs"] == candidate["scientific_inputs"]
                if reference.get("scientific_inputs") and candidate.get("scientific_inputs") else None),
            "scope": "selected_observables_only_not_scientific_validation"}
