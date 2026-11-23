"""Instantaneous frozen-state rate accounting, not measured mechanism fractions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools import p92_recoverable_spectrum as spectrum
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material


def rate_budget(state, material, *, stress_control, strain_rate=0.):
    if np.iscomplexobj(state):
        raise ValueError("Require real spectrum state")
    y = np.asarray(state, dtype=float)
    if y.shape != (12,) or not np.isfinite(y).all() or np.any(y[[3, 6, 9]] < 0):
        raise ValueError("Require finite nonnegative-history spectrum state")
    if type(stress_control) is not bool or not np.isfinite(strain_rate):
        raise ValueError("Invalid control")
    r = spectrum.rates(y, strain_rate, material, stress_control=stress_control)
    backbone, _ = spectrum.evolving_backbone(y, material)
    branches, flows = [], []
    for offset, branch in zip((1, 4), backbone.branches, strict=True):
        effective = y[0]-sum(y[offset:offset+2])
        flow = r[offset+2]*np.sign(effective)
        flows.append(flow)
        branches.append(dict(effective_stress_MPa=float(effective),
            threshold_MPa=branch.strength, overstress_MPa=float(max(abs(effective)-branch.strength, 0.)),
            signed_flow_per_s=float(flow), accumulated_rate_per_s=float(r[offset+2])))
    total = float(sum(flows)+sum(r[10:]))
    closure = float(r[7]-r[0]/backbone.modulus-total)
    return dict(branches=branches, recoverable_rates_per_s=r[10:].tolist(),
        total_nonelastic_rate_per_s=total, stress_rate_MPa_per_s=float(r[0]),
        strain_rate_per_s=float(r[7]), rate_closure_per_s=closure,
        scope="Instantaneous model decomposition at the segment endpoint, not an observed secant or physical identification")


def report(frozen):
    module, mat = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not spectrum:
        raise ValueError("Rate accounting requires explicit spectrum identity")
    protocols = {}
    for name, item in frozen["protocols"].items():
        if not item.get("numerical", {}).get("passed", False):
            raise ValueError("Require numerically verified frozen trajectory")
        prediction = item["prediction"]
        if (prediction["source_sha256"] != frozen["source_sha256"][name]
                or prediction["protocol"] != name or prediction["model_id"] != spectrum.MODEL_ID
                or tuple(prediction["state_layout"]) != spectrum.STATE_LAYOUT):
            raise ValueError("Source identity changed")
        if any(s["segment"] != i for i, s in enumerate(prediction["segments"])):
            raise ValueError("Unaligned segment identities")
        for h in prediction["hold_diagnostics"]:
            index = h["segment"]
            if (type(index) is not int or not 0 <= index < len(prediction["segments"])
                    or prediction["segments"][index]["cycle"] != h["cycle"]
                    or prediction["segments"][index]["stress_control"] != h["stress_control"]):
                raise ValueError("Unaligned hold identity")
        protocols[name] = [dict(cycle=h["cycle"], segment=h["segment"],
            stress_control=h["stress_control"], duration_s=h["duration_s"],
            endpoint=rate_budget(prediction["segments"][h["segment"]]["end_state"], mat,
                                 stress_control=h["stress_control"]))
            for h in prediction["hold_diagnostics"]]
    return dict(schema="agentfem.research.spectrum-endpoint-rate-budget.v1",
        model_id=spectrum.MODEL_ID, source_sha256=frozen["source_sha256"],
        protocols=protocols, fitted=False, engineering_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    result = report(json.loads(raw))
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)
