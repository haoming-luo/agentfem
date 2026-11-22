"""Read-only attenuation bounds on an already computed spectrum trajectory.

Bounds concern the homogeneous contribution of an initial backstress, not
the final backstress (which also has a flow-driven source). No fit or solve.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_frozen import decode_material
from tools.p92_measured_fit import write_new


def maximum_factor_contrast(first_history, second_history, residual_fraction):
    """Maximum absolute factor contrast over b>=0, not a stress-error bound."""
    lo, hi = sorted((first_history, second_history))
    if (not np.isfinite([lo, hi, residual_fraction]).all() or lo < 0
            or not 0 < residual_fraction <= 1):
        raise ValueError("Require nonnegative histories and a positive residual fraction")
    if lo == hi or residual_fraction == 1.:
        return dict(maximum_absolute_contrast=0., rate_at_maximum=0., attained=True)
    if lo == 0.:
        return dict(maximum_absolute_contrast=1-residual_fraction,
                    rate_at_maximum=None, attained=False)
    b = np.log1p((hi-lo)/lo)/(hi-lo)
    # expm1 keeps nearly equal histories resolved without subtracting exponentials.
    contrast = (1-residual_fraction)*np.exp(-b*lo)*(-np.expm1(-b*(hi-lo)))
    return dict(maximum_absolute_contrast=float(contrast), rate_at_maximum=float(b), attained=True)


def backstress_budget(solution, material, *, stress_control=False):
    """Resolve signed generation/recovery along an accepted dense trajectory."""
    start, end = solution.y[:, 0], solution.y[:, -1]
    duration = solution.t[-1]-solution.t[0]
    attenuation_bounds(material, start, end, duration)
    def terms(t):
        state = solution.sol(t)
        rates = spectrum.rates(state, 0., material, stress_control=stress_control)
        rows = []
        for j, (offset, branch) in enumerate(zip((1, 4), material.backbone.branches, strict=True)):
            x = state[offset:offset+2]
            effective = state[0]-sum(x)
            v = np.sign(effective)*rates[offset+2]
            h, a = np.asarray(branch.hardening), np.asarray(branch.saturation)
            active = np.ones(2) if not material.directional_recovery else (x*effective > 0.)
            dynamic = -h/a*(abs(x)/a)**branch.ratcheting*rates[offset+2]*active*x
            p = sum(state[[3, 6]]) if material.static_history == "shared" else state[offset+2]
            factor = material.static_ratios[j]+(1-material.static_ratios[j])*np.exp(-material.static_history_rates[j]*p)
            static = -np.asarray(branch.static_recovery)*factor*x
            rows.extend(np.column_stack((h*v, dynamic, static)))
        return np.asarray(rows)
    values, error = quad_vec(terms, solution.t[0], solution.t[-1], epsabs=1e-8, epsrel=1e-9,
                             points=solution.t[1:-1])
    actual = end[[1, 2, 4, 5]]-start[[1, 2, 4, 5]]
    closure = values.sum(axis=1)-actual
    if error > 1e-5 or np.max(abs(closure)) > 1e-5:
        raise ValueError("Backstress contribution quadrature did not close")
    return dict(names=["A_fast", "A_slow", "B_fast", "B_slow"],
        terms=["flow_generation", "dynamic_recovery", "static_recovery"],
        signed_increments_MPa=values.tolist(), net_change_MPa=actual.tolist(),
        closure_MPa=closure.tolist(), quadrature_error_estimate_MPa=float(error),
        scope="Conditional model attribution; not experimentally measured backstress")


def attenuation_bounds(material, start, end, duration):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if (start.shape != (12,) or end.shape != (12,) or not np.isfinite([*start, *end, duration]).all()
            or duration <= 0 or np.any(start[[3, 6]] < 0)
            or np.any(end[[3, 6]] < start[[3, 6]])
            or material.static_powers != (0., 0.) or material.hardening_ratios != (1., 1.)
            or any(b.saturation_ratio != 1. for b in material.backbone.branches)):
        raise ValueError("Require monotone history and constant-modulus linear static recovery")
    rows = []
    for j, (offset, branch) in enumerate(zip((1, 4), material.backbone.branches, strict=True)):
        indices = [3, 6] if material.static_history == "shared" else [offset+2]
        rho, b = material.static_ratios[j], material.static_history_rates[j]
        factor = rho+(1-rho)*np.exp(-b*np.array([sum(start[indices]), sum(end[indices])]))
        exposure_min = duration*np.asarray(branch.static_recovery)*factor[1]
        exposure_max = duration*np.asarray(branch.static_recovery)*factor[0]
        upper = np.exp(-exposure_min)
        rows.append(dict(branch="AB"[j], accumulated_increment=float(end[offset+2]-start[offset+2]),
            static_factor_start=float(factor[0]), static_factor_end=float(factor[1]),
            static_only_multiplier_lower=np.exp(-exposure_max).tolist(),
            homogeneous_memory_upper=upper.tolist(),
            initial_backstress_MPa=start[offset:offset+2].tolist(),
            initial_contribution_abs_upper_MPa=(abs(start[offset:offset+2])*upper).tolist(),
            final_backstress_MPa=end[offset:offset+2].tolist()))
    return rows


def audit(frozen, *, contributions=False):
    module, material = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not spectrum:
        raise ValueError("This audit requires the original spectrum state identity")
    protocols = {}
    for name in ("SLR", "SLC"):
        item = frozen["protocols"][name]
        if not item["numerical"]["passed"]:
            raise ValueError("Require a numerically verified frozen trajectory")
        if "prediction" not in item:
            raise ValueError("Memory audit needs full frozen trajectories, not a compact response summary")
        prediction = item["prediction"]
        if tuple(prediction["state_layout"]) != spectrum.STATE_LAYOUT:
            raise ValueError("State layout differs from spectrum identity")
        segments = {r["segment"]: r for r in prediction["segments"] if r["cycle"] == 1}
        hold = next(r for r in prediction["hold_diagnostics"] if r["cycle"] == 1 and r["segment"] == 3)
        if hold["stress_control"] != (name == "SLC") or hold["duration_s"] != 1800.:
            raise ValueError("Expected the declared first long-hold controls")
        start, end, entry = (segments[i]["end_state"] for i in (2, 3, 4))
        p = np.array(entry)[[3, 6]]
        entry_factors = [rho+(1-rho)*np.exp(-b*(sum(p) if material.static_history == "shared" else p[j]))
            for j, (rho, b) in enumerate(zip(material.static_ratios, material.static_history_rates, strict=True))]
        protocols[name] = dict(long_hold=attenuation_bounds(material, start, end, hold["duration_s"]),
            audit_input=dict(start_state=start, end_state=end, duration_s=hold["duration_s"]),
            positive_hold_entry=dict(accumulated_A=float(p[0]), accumulated_B=float(p[1]),
                                     total_accumulated=float(sum(p)), static_factors=entry_factors))
        if contributions:
            budgets = {}
            for index in (3, 5):
                diagnostic = next(r for r in prediction["hold_diagnostics"]
                                  if r["cycle"] == 1 and r["segment"] == index)
                old = np.asarray(segments[index-1]["end_state"])
                solution = spectrum.integrate(old, diagnostic["duration_s"], 0., material,
                                               stress_control=diagnostic["stress_control"])
                if not np.allclose(solution.y[:, -1], segments[index]["end_state"], rtol=1e-9, atol=1e-9):
                    raise ValueError("Contribution replay differs from frozen state")
                budgets[str(index)] = backstress_budget(solution, material,
                                                       stress_control=diagnostic["stress_control"])
            protocols[name]["contribution_budgets"] = budgets
    return dict(schema="agentfem.research.history-memory-bound.v1", model_id=frozen["model_id"],
        parameters=frozen["parameters"],
        source_sha256=frozen["source_sha256"], protocols=protocols, fitted=False,
        engineering_qualified=False,
        scope="Conditional homogeneous-memory bounds, not a bound on total regenerated backstress or stress response")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frozen", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--contributions", action="store_true")
    args = p.parse_args()
    raw = args.frozen.read_bytes()
    result = audit(json.loads(raw), contributions=args.contributions)
    result["frozen_sha256"] = sha256(raw).hexdigest()
    result["driver_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(args.output, result)
