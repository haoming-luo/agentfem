"""Conditional early-cycle kinematic-timescale experiment, not an admitted law.

Freeze the branch-time card; vary only fast/slow C1 (first-backstress evolution
per inelastic strain), not saturation, flow law, recovery or initial history.
Artificial entry-state probes are instantaneous diagnostics, never fed into a
trajectory. They cannot establish experimentally observed causal mechanisms.
"""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path, replay
from tools.p92_measured_fit import frozen_predictions, write_new
from tools.p92_published_recovery_reference import Parameters
from tools.p92_recovery_discriminator import terms
from tools.p92_two_mechanism import Mechanism


def decode(z, mechanisms):
    z = np.asarray(z, dtype=float)
    if z.shape != (2,) or not np.isfinite(z).all() or len(mechanisms) != 2:
        raise ValueError("Require two finite log10 C1 scales and two branches")
    return tuple(replace(m, parameters=replace(m.parameters, C1=m.parameters.C1*10.**v))
                 for m, v in zip(mechanisms, z, strict=True))


def entry_probes(state, mechanisms):
    """Conditional instantaneous rates with stress held fixed; no state mutation."""
    state = np.asarray(state, dtype=float)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("Require finite continuous ten-component entry state")
    records = []
    for offset, m in zip((1, 5), mechanisms, strict=True):
        local = np.r_[state[0], state[offset:offset+4]]
        no_x = local.copy()
        no_x[1:3] = 0.
        no_R = local.copy()
        no_R[3] = 0.
        records.append(dict(actual=terms(local, m),
            zero_backstress_probe=terms(no_x, m), zero_isotropic_probe=terms(no_R, m)))
    return records


def diagnose(path, mechanisms):
    _, result = replay(path, mechanisms)
    holds = []
    for hold in result["hold_diagnostics"]:
        i = hold["segment"]
        if i:
            entry = result["segment_endpoints"][i-1]["state"]
        else:
            entry = np.zeros(10)
            entry[0], entry[9] = path["initial_stress"], path["initial_strain"]
        holds.append(dict(**hold, continuous_entry_state=np.asarray(entry).tolist(),
                          conditional_entry_probes=entry_probes(entry, mechanisms)))
    return dict(protocol=path["protocol"], holds=holds,
                reversed_post_hold_ramps=result["reversed_post_hold_ramps"])


def identify(seed, sources, *, max_nfev=40):
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("Require positive evaluation budget")
    cards = seed["parameters"]
    if (seed["training_cycles"] != [1, 2, 3] or not seed["fit"]["success"] or
            sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest() != seed["parameter_sha256"]):
        raise ValueError("Require intact early-cycle frozen seed")
    if len(sources) != 3 or {d["protocol"] for d in sources} != {"RF", "SLR", "SLC"}:
        raise ValueError("Require three distinct protocols")
    if any(d["sha256"] != seed["source_sha256"][d["protocol"]] for d in sources):
        raise ValueError("Changed source identity")
    base = tuple(Mechanism(Parameters(**m["parameters"]), m["exponent"], m["enabled"]) for m in cards)
    paths = [measured_path(d, cycles=3) for d in sources]
    calls = 0

    def objective(z, rtol=1e-8):
        nonlocal calls
        calls += 1
        value = np.concatenate([replay(p, decode(z, base), rtol=rtol,
            allow_incompatible=True)[0] for p in paths])
        if calls % 20 == 1:
            print(json.dumps(dict(evaluation=calls, cost=float(.5*value@value))), flush=True)
        return value

    def jacobian(z):
        h = 1e-3
        return np.column_stack([(objective(z+e*h)-objective(z-e*h))/(2*h)
                                for e in np.eye(2)])

    initial_residual = objective(np.zeros(2))
    result = least_squares(objective, np.zeros(2), bounds=([-1., -1.], [1., 1.]),
        jac=jacobian, x_scale="jac", max_nfev=max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    mechanisms = decode(result.x, base)
    identity = [asdict(m) for m in mechanisms]
    report = dict(schema="agentfem.research.p92-transient-history.v1", training_cycles=[1, 2, 3],
        later_cycles_role="previously inspected development, not blind validation",
        coordinate_names=["log10_fast_C1_scale", "log10_slow_C1_scale"],
        scaled_parameters=result.x.tolist(), bounds=[[-1., -1.], [1., 1.]],
        seed_parameter_sha256=seed["parameter_sha256"], source_sha256=seed["source_sha256"],
        parameters=identity, parameter_sha256=sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        fit=dict(success=bool(result.success), message=result.message, cost=float(result.cost),
            initial_cost=float(.5*initial_residual@initial_residual), nfev=result.nfev,
            residual_evaluations=calls, active_bounds=result.active_mask.tolist(),
            singular_values=np.linalg.svd(result.jac, compute_uv=False).tolist()),
        baseline_early_holds=[diagnose(p, base) for p in paths],
        engineering_qualified=False, life_qualified=False, independent_validation=False,
        scope="Conditional two-coordinate fit, all other seed parameters fixed; probes never reset trajectory")
    if result.success:
        report["training_refinement_max_scaled_residual_change"] = float(np.max(
            abs(objective(result.x)-objective(result.x, rtol=1e-10))))
        report["fitted_early_holds"] = [diagnose(p, mechanisms) for p in paths]
        report["frozen_predictions"] = frozen_predictions(report, sources)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    report = identify(json.loads(args.seed.read_text()),
        [read_source(args.sources/f"Data_{p}.lis", protocol=p) for p in ("RF", "SLR", "SLC")])
    report["driver_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(args.output, report)
    print(json.dumps(report["fit"]), flush=True)
    raise SystemExit(0 if report["fit"]["success"] else 1)
