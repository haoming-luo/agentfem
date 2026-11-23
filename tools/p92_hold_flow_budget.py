"""Signed strain budget of a frozen hold, without fitting or state reconstruction."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

from tools import p92_recoverable_spectrum as kernel
from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def budget(solution, material, start, stop):
    if not np.isfinite([start, stop]).all() or not solution.t[0] <= start < stop <= solution.t[-1]:
        raise ValueError("Require an interval inside the integrated hold")
    endpoints = solution.sol(np.array([start, stop]))
    delta = endpoints[:, 1]-endpoints[:, 0]

    def signed_flow(t):
        state = solution.sol(t)
        rates = kernel.rates(state, 0., material)
        return np.array([rates[i+2]*np.sign(state[0]-sum(state[i:i+2])) for i in (1, 4)])

    flow, error = quad_vec(signed_flow, start, stop, epsabs=1e-12, epsrel=1e-8)
    elastic = delta[0]/material.backbone.modulus
    recoverable = delta[10:12]
    closure = delta[7]-elastic-recoverable.sum()-flow.sum()
    duration = stop-start
    return dict(interval_s=[float(start), float(stop)], duration_s=float(duration),
        total_strain_change=float(delta[7]), elastic_strain_change=float(elastic),
        recoverable_strain_changes=recoverable.tolist(),
        signed_branch_strain_changes=flow.tolist(),
        accumulated_branch_changes=delta[[3, 6]].tolist(),
        quadrature_error_estimate=float(error), strain_closure_error=float(closure),
        predicted_stress_secant_MPa_per_s=float(delta[0]/duration),
        total_strain_stress_secant_MPa_per_s=float(material.backbone.modulus*delta[7]/duration),
        signed_branch_stress_secants_MPa_per_s=(-material.backbone.modulus*flow/duration).tolist(),
        recoverable_stress_secants_MPa_per_s=(-material.backbone.modulus*recoverable/duration).tolist(),
        fixed_flow_without_recovery_secant_MPa_per_s=float(-material.backbone.modulus*flow.sum()/duration),
        accounting_passed=bool(abs(closure) < 1e-9 and error < 1e-10),
        scope="Signed frozen-path accounting; removing a component here is not a new causal prediction")


def evaluate(frozen, directory):
    module, material = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not kernel:
        raise ValueError("Require the original explicit spectrum state layout")
    records = {}
    for name in ("RF", "SLR", "SLC"):
        source = read_source(directory/f"Data_{name}.lis", protocol=name)
        if source["sha256"] != frozen["source_sha256"][name]:
            raise ValueError("Changed source identity")
        path = attach_waveforms(measured_path(source, cycles=1), source)
        index = next(i for i, s in enumerate(path["segments"])
                     if s["strain_hold"] and s["target"] > 0)
        segment = path["segments"][index]
        prediction = frozen["protocols"][name]["prediction"]
        if prediction["state_layout"] != list(kernel.STATE_LAYOUT):
            raise ValueError("Changed state identity")
        # Rebuild the short causal prefix from the public source and frozen
        # card, so compact evidence need not ship every sampled segment.
        _, causal = replay(path, material, kernel=kernel)
        final_difference = abs(np.asarray(causal["cycles"][0]["end_state"])
                               - np.asarray(prediction["cycles"][0]["end_state"]))
        if (final_difference[[0, 1, 2, 4, 5]].max() > 1e-4
                or final_difference[[3, 6, 7, 10, 11]].max() > 1e-8
                or final_difference[[8, 9]].max() > 1e-6):
            raise ValueError("Causal replay does not reproduce the frozen first cycle")
        incoming = np.asarray(causal["segments"][index-1]["end_state"])
        solution = kernel.integrate(incoming, segment["duration"], 0., material)
        independent = kernel.integrate(incoming, segment["duration"], 0., material,
                                       method="BDF", rtol=1e-10)
        end_difference = abs(solution.y[:, -1]-np.asarray(causal["segments"][index]["end_state"]))
        if (end_difference[[0, 1, 2, 4, 5]].max() > 1e-4
                or end_difference[[3, 6, 7, 10, 11]].max() > 1e-8
                or end_difference[[8, 9]].max() > 1e-6):
            raise ValueError("Integrated hold does not reproduce the frozen continuation")
        t = np.asarray(segment["sample_times"])
        early = t <= min(10., .1*segment["duration"])
        windows = dict(early=(t[early], np.asarray(segment["measured_stress"])[early],
                              np.asarray(segment["measured_strain"])[early]),
                       late=(np.asarray(segment["tail"]["times"]), np.asarray(segment["tail"]["stress"]),
                             np.asarray(segment["tail"]["strain"])))
        rows = {}
        for window, (times, stress, strain) in windows.items():
            if len(times) < 2:
                raise ValueError("Require two observations for a resolved window")
            a, b = float(times[0]), float(times[-1])
            row, alternate = budget(solution, material, a, b), budget(independent, material, a, b)
            row["observed_stress_secant_MPa_per_s"] = float((stress[-1]-stress[0])/(b-a))
            row["recorded_strain_change"] = float(strain[-1]-strain[0])
            row["recorded_strain_elastic_secant_MPa_per_s"] = float(
                material.backbone.modulus*(strain[-1]-strain[0])/(b-a))
            row["recorded_strain_scope"] = "Measurement diagnostic, not an imposed controller command or uncertainty distribution"
            row["independent"] = alternate
            differences = [abs(row[k]-alternate[k]) for k in (
                "predicted_stress_secant_MPa_per_s", "fixed_flow_without_recovery_secant_MPa_per_s")]
            row["numerical_passed"] = bool(row["accounting_passed"] and alternate["accounting_passed"]
                                           and max(differences) < 1e-5)
            rows[window] = row
        records[name] = dict(segment=index, incoming_state=incoming.tolist(), windows=rows,
                             frozen_end_state_absolute_difference=end_difference.tolist(),
                             frozen_cycle_state_absolute_difference=final_difference.tolist())
    return dict(schema="agentfem.research.signed-hold-flow-budget.v1", records=records,
        model_id=frozen["model_id"], parameters=frozen["parameters"], source_sha256=frozen["source_sha256"],
        fitted=False, state_reconstruction=False, engineering_qualified=False, life_qualified=False,
        scope="Frozen accepted-path decomposition; signed flow is not accumulated plastic strain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "source_dir", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_hold_flow_budget.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_recoverable_frozen.py", "p92_measured_controls.py", "p92_bam_source.py",
             "p92_two_rate_controls.py", "p92_waveform_controls.py", "p92_hold_observations.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    result = evaluate(json.loads(raw), args.source_dir)
    if args.frozen.read_bytes() != raw or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Frozen budget inputs changed")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
