"""Conditional full recorded-stress diagnostic, never a response admission."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools import p92_recoverable_spectrum as kernel
from tools.p92_stress_ripple_audit import stress_path, refine_stress_controls
from tools.p92_hold_flow_budget import budget


def scalar_or_vector(evaluate):
    """Adapt the audited vector-only dense evaluator for scalar quadrature."""
    def wrapped(query):
        values = np.asarray(query)
        result = evaluate(np.atleast_1d(values))
        return result[:, 0] if values.ndim == 0 else result
    return wrapped


def mechanical_balances(solutions, initial, material):
    """Check each independent replay, not only the displayed reference arm."""
    initial_energy = kernel.stored_energy(initial, material)
    balances = [float(s[8, -1]-s[9, -1]-kernel.stored_energy(s[:, -1], material)
                      + initial_energy) for s in solutions]
    limit = 1e-8  # MPa: work per reference volume, not a stress-fit tolerance.
    return dict(values_MPa=balances, absolute_limit_MPa=limit,
                passed=bool(np.isfinite(balances).all() and np.all(np.abs(balances) <= limit)))


def evaluate(frozen, directory, *, cycles=1):
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("Require a positive integer cycle count")
    module, material = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not kernel:
        raise ValueError("Require the unchanged spectrum reference")
    records = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(directory/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = measured_path(source, cycles=cycles)
        ids = np.flatnonzero((source["cycle"] >= 1) & (source["cycle"] <= cycles))
        if (not len(ids) or ids[0] != 0 or np.any(np.diff(ids) != 1)
                or not np.array_equal(np.unique(source["cycle"][ids]), np.arange(1, cycles+1))):
            raise ValueError("Require contiguous recorded cycles beginning at one")
        t = source["time_s"][ids]
        if np.any(np.diff(t) <= 0):
            raise ValueError("Recorded timestamps are not uniquely ordered")
        t = t-t[0]
        stress, strain = source["stress_Pa"][ids]/1e6, source["strain"][ids]
        initial = np.zeros(12)
        initial[0], initial[7] = stress[0], strain[0]
        waves = (t, stress-stress[0])
        solutions, dense = [], []
        for name, controls, method, rtol in (
                ("recorded", waves, "Radau", 1e-9),
                ("independent", waves, "BDF", 1e-12),
                ("subdivided", refine_stress_controls([waves])[0], "Radau", 1e-9)):
            sol = stress_path(initial, *controls, material, kernel, method=method, rtol=rtol,
                              atol_scale=.01 if method == "BDF" else 1.)
            solutions.append(sol.sol(t))
            dense.append(SimpleNamespace(sol=scalar_or_vector(sol.sol), t=t))
            print(json.dumps(dict(protocol=protocol, arm=name, samples=len(t))), flush=True)
        limits = np.array([1e-4, 1e-4, 1e-4, 1e-8, 1e-4, 1e-4, 1e-8, 1e-8, 1e-6, 1e-6, 1e-8, 1e-8])
        differences = [np.max(abs(other-solutions[0]), axis=1) for other in solutions[1:]]
        rows = []
        for i, segment in enumerate(path["segments"]):
            if not (segment["stress_control"] or segment["strain_hold"]):
                continue
            early = np.asarray(segment["source_indices"])[
                np.asarray(segment["sample_times"]) <= min(10., .1*segment["duration"])]
            for window, indices in (("early", early), ("late", segment["tail"]["source_indices"])):
                if len(indices) < 2:
                    rows.append(dict(cycle=segment["cycle"], segment=i, window=window, status="unresolved"))
                    continue
                a, b = int(indices[0]), int(indices[-1])
                dt = float(t[b]-t[a])
                observed = float(strain[b]-strain[a])
                changes = [float(s[7, b]-s[7, a]) for s in solutions]
                flow_budget = budget(dense[0], material, float(t[a]), float(t[b]))
                alternate = budget(dense[1], material, float(t[a]), float(t[b]))
                flow_budget["independent_signed_branch_changes"] = alternate["signed_branch_strain_changes"]
                flow_budget["independent_accounting_passed"] = alternate["accounting_passed"]
                flow_budget["independent_signed_flow_passed"] = bool(np.allclose(
                    flow_budget["signed_branch_strain_changes"],
                    alternate["signed_branch_strain_changes"], rtol=0., atol=1e-8))
                rows.append(dict(cycle=segment["cycle"], segment=i, window=window, original_stress_control=segment["stress_control"],
                    source_indices=[a, b], duration_s=dt, observed_strain_change=observed,
                    predicted_strain_change=changes[0], strain_change_residual=changes[0]-observed,
                    strain_rate_residual_per_s=(changes[0]-observed)/dt,
                    independent_strain_change=changes[1], subdivided_strain_change=changes[2],
                    signed_flow_budget=flow_budget))
        last = solutions[0][:, -1]
        balances = mechanical_balances(solutions, initial, material)
        records[protocol] = dict(initial_state=initial.tolist(), records=rows,
            maximum_state_differences=[x.tolist() for x in differences],
            numerical_passed=bool(balances["passed"] and all(np.all(x <= limits) for x in differences)
                and all(r.get("status") == "unresolved" or (
                    r["signed_flow_budget"]["accounting_passed"]
                    and r["signed_flow_budget"]["independent_accounting_passed"]
                    and r["signed_flow_budget"]["independent_signed_flow_passed"])
                    for r in rows)),
            numerical_limits=limits.tolist(), mechanical_balance_MPa=balances["values_MPa"][0],
            mechanical_balance_checks=balances,
            final_strain_residual=float(last[7]-strain[-1]),
            maximum_strain_residual=float(max(abs(solutions[0][7]-strain))))
    return dict(schema="agentfem.research.observed-stress-replay.v1", records=records,
        cycles=cycles, independent_integration=dict(method="BDF", rtol=1e-12, atol_scale=.01),
        model_id=frozen["model_id"], parameters=frozen["parameters"], source_sha256=frozen["source_sha256"],
        initial_memory_assumption="Zero internal memory, observed initial stress/strain; not independently verified prehistory",
        fitted=False, engineering_qualified=False, life_qualified=False, forward_validation=False,
        scope="Conditional recorded-stress path; strain is predicted, prescribed stress is not validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "source_dir", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_observed_stress_replay.py", "p92_stress_ripple_audit.py",
             "p92_hold_flow_budget.py", "p92_recoverable_spectrum.py",
             "p92_two_rate_reference.py", "p92_recoverable_frozen.py",
             "p92_measured_controls.py", "p92_bam_source.py", "p92_hold_observations.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    result = evaluate(json.loads(raw), args.source_dir, cycles=args.cycles)
    if args.frozen.read_bytes() != raw or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Frozen replay inputs changed")
    result["frozen_sha256"] = sha256(raw).hexdigest()
    result["driver_sha256"] = before
    write_new(args.output, result)
