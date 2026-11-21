"""Audit signed branch flow on a frozen continuous SLC trajectory.

This decomposes a model prediction, not experimental damage or dissipation.
No measured state is substituted, and no constitutive parameter is changed.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from tools.p92_measured_fit import write_new
from tools.p92_published_recovery_reference import Parameters
from tools.p92_recovery_discriminator import terms
from tools.p92_two_mechanism import Mechanism, integrate


def audit_hold(old, duration, mechanisms):
    old = np.asarray(old, dtype=float)
    solution = integrate(old, duration, 0., mechanisms, stress_control=True, rtol=1e-10)
    times = np.unique(np.r_[0., np.array([1., 3., 10., 30., 100., 300., 900.])[
        np.array([1., 3., 10., 30., 100., 300., 900.]) < duration], duration])

    def branch(t, offset, mechanism):
        state = solution.sol(t)
        return terms(np.r_[state[0], state[offset:offset+4]], mechanism)

    records, integrals, error_bounds = [], [], []
    for offset, mechanism in zip((1, 5), mechanisms, strict=True):
        # Split at accepted ODE intervals; the signed integral is not equal
        # to accumulated |flow| if the flow reverses during the hold.
        total = error = 0.
        for a, b in zip(solution.t[:-1], solution.t[1:], strict=True):
            value, estimate = quad(lambda t, o=offset, m=mechanism: branch(t, o, m)["signed_flow_per_s"],
                a, b, epsabs=1e-13, epsrel=1e-7)
            total += value
            error += estimate
        integrals.append(total)
        error_bounds.append(error)
    for t in times:
        records.append(dict(time_s=float(t), branches=[branch(t, o, m)
            for o, m in zip((1, 5), mechanisms, strict=True)]))
    end = solution.y[:, -1]
    inferred = end[9]-old[9]-(end[0]-old[0])/mechanisms[0].parameters.E
    closure = float(sum(integrals)-inferred)
    return dict(end_state=end.tolist(), samples=records,
        signed_inelastic_strain_by_branch=integrals,
        quadrature_error_estimates=error_bounds, inelastic_increment_from_kinematics=float(inferred),
        strain_closure_error=closure, numerical_closure_passed=abs(closure) < 1e-8,
        scope="Model branch-flow decomposition; not stored energy, damage or measured mechanism shares")


def audit(report):
    cards = report["parameters"]
    if sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest() != report["parameter_sha256"]:
        raise ValueError("Frozen parameter identity changed")
    candidates = [r for r in report["records"] if r["protocol"] == "SLC"]
    if len(candidates) != 1:
        raise ValueError("Require exactly one SLC prediction")
    mechanisms = tuple(Mechanism(Parameters(**c["parameters"]), c["exponent"], c["enabled"])
                       for c in cards)
    prediction = candidates[0]["waveform"]
    records = []
    for hold in prediction["hold_diagnostics"]:
        if not hold["stress_control"]:
            continue
        i = hold["segment"]
        old = prediction["segment_endpoints"][i-1]["state"]
        result = audit_hold(old, hold["duration_s"], mechanisms)
        delta = np.abs(np.asarray(result["end_state"])-prediction["segment_endpoints"][i]["state"])
        if (delta[[0, 1, 2, 3, 5, 6, 7]].max() >= 1e-4 or delta[[4, 8, 9]].max() >= 1e-8
                or not result["numerical_closure_passed"]):
            raise RuntimeError("Replayed hold does not match the frozen trajectory")
        records.append(dict(cycle=hold["cycle"], **result))
    return dict(schema="agentfem.research.p92-hold-flow-audit.v1", records=records,
        parameter_sha256=report["parameter_sha256"], source_sha256=report["source_sha256"],
        parameter_refit=False, engineering_qualified=False, life_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    result = audit(json.loads(args.frozen.read_text()))
    result["frozen_report_sha256"] = sha256(args.frozen.read_bytes()).hexdigest()
    write_new(args.output, result)
