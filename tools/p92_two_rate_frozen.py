"""Frozen conditional two-rate card replay with the unchanged response screen."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms
from tools.p92_waveform_experiment import refine_knots


def numerical_difference(a, b, *, extra_strain_indices=(),
                         stress_state_indices=(0, 1, 2, 4, 5), dimensionless_indices=()):
    for key in ("model_id", "state_layout", "protocol", "source_sha256"):
        left, right = a.get(key), b.get(key)
        if key == "state_layout" and left is not None and right is not None:
            left, right = tuple(left), tuple(right)
        if left != right:
            raise ValueError(f"Numerical comparison identity differs: {key}")
    pairs = list(zip(a["segments"], b["segments"], strict=True))
    if not pairs:
        raise ValueError("Numerical comparison requires completed segments")
    for x, y in pairs:
        for key in ("segment", "cycle", "stress_control", "source_indices"):
            if x.get(key) != y.get(key):
                raise ValueError(f"Numerical comparison observations differ: {key}")
        for key in ("sampled_stress_MPa", "sampled_strain", "end_state"):
            left, right = np.asarray(x[key]), np.asarray(y[key])
            if (left.ndim != 1 or left.size == 0 or right.shape != left.shape
                    or not np.isfinite(left).all() or not np.isfinite(right).all()):
                raise ValueError(f"Numerical comparison requires aligned finite arrays: {key}")
    stress = max(float(np.max(abs(np.asarray(x["sampled_stress_MPa"])-y["sampled_stress_MPa"])))
                 for x, y in pairs)
    strain = max(float(np.max(abs(np.asarray(x["sampled_strain"])-y["sampled_strain"])))
                 for x, y in pairs)
    extrema = 0.
    for x, y in zip(a.get("cycles", ()), b.get("cycles", ()), strict=True):
        if x["cycle"] != y["cycle"]:
            raise ValueError("Numerical comparison cycle identities differ")
        for key in ("sampled_peak_MPa", "sampled_valley_MPa"):
            if not np.isfinite([x[key], y[key]]).all():
                raise ValueError("Numerical comparison requires finite cycle extrema")
            extrema = max(extrema, abs(x[key]-y[key]))
    stress = max(stress, extrema)
    delta = np.asarray([np.asarray(x["end_state"])-y["end_state"] for x, y in pairs])
    groups = (*stress_state_indices, 3, 6, 7, *extra_strain_indices, 8, 9, *dimensionless_indices)
    if len(groups) != delta.shape[1] or set(groups) != set(range(delta.shape[1])):
        raise ValueError("Numerical comparison must cover each state exactly once")
    stress = max(stress, float(np.max(abs(delta[:, list(stress_state_indices)]))))
    strain = max(strain, float(np.max(abs(delta[:, [3, 6, 7, *extra_strain_indices]]))))
    energy = float(np.max(abs(delta[:, [8, 9]])))
    dimensionless = float(np.max(abs(delta[:, list(dimensionless_indices)]))) if dimensionless_indices else 0.
    return dict(stress_state_difference_MPa=stress, strain_state_difference=strain,
                work_dissipation_difference_MPa=energy,
                dimensionless_state_difference=dimensionless,
                cycle_extrema_difference_MPa=extrema,
                passed=bool(stress < 1e-4 and strain < 1e-8 and energy < 1e-6 and dimensionless < 1e-8))


def evaluate(fit, source_dir):
    if fit["training_cycles"] != [1, 2, 3] or not fit["conditional_fit"]["success"]:
        raise ValueError("Require a completed early-only conditional fit")
    card = fit["conditional_fit"]["parameters"]
    material = Material(card["modulus"], tuple(Branch(**b) for b in card["branches"]))
    results = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != fit["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(source, cycles=10), source)
        _, prediction = replay(path, material)
        _, independent = replay(path, material, method="BDF", rtol=1e-10)
        _, refined = replay(refine_knots(path), material, rtol=1e-10)
        independent_check = numerical_difference(prediction, independent)
        refinement_check = numerical_difference(prediction, refined)
        comparison = compare_candidate(source, prediction)
        # All three early protocols were used in the conditional fit.
        for row in comparison["records"]:
            row["role"] = "early_calibration" if row["cycle"] <= 3 else "inspected_development"
        comparison.pop("groups")  # original RF-only grouping does not apply here
        results[protocol] = dict(prediction=prediction, comparison=comparison,
            response_screen=screen(prediction, comparison, requested_cycles=10),
            numerical=dict(independent=independent_check, refinement=refinement_check,
                           passed=independent_check["passed"] and refinement_check["passed"]))
    return dict(schema="agentfem.research.two-rate-frozen.v1", parameters=card,
        parameter_sha256=sha256(json.dumps(card, sort_keys=True).encode()).hexdigest(),
        source_sha256=fit["source_sha256"], protocols=results,
        scope="Frozen early fit; cycles 4-10 previously inspected, not independent holdout",
        independent_validation=False, engineering_qualified=False, life_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = ("p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_hold_observations.py",
             "p92_two_rate_frozen.py", "p92_measured_controls.py", "p92_waveform_controls.py",
             "p92_response_screen.py", "p92_bounded_softening.py", "p92_bam_source.py",
             "p92_waveform_experiment.py")
    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    before = identities()
    payload = args.fit.read_bytes()
    result = evaluate(json.loads(payload), args.source_dir)
    if identities() != before or args.fit.read_bytes() != payload:
        raise RuntimeError("Driver or frozen fit changed during verification")
    result["driver_sha256"] = before
    result["fit_file_sha256"] = sha256(payload).hexdigest()
    write_new(args.output, result)
    print(json.dumps({p: dict(response=r["response_screen"]["status"], numerical=r["numerical"])
                      for p, r in result["protocols"].items()}))
