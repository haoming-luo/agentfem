"""Frozen nominal-rate mixed controls without measured endpoint timing."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_hold_observations import summarize
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools.p92_response_screen import screen
from tools.p92_two_rate_frozen import numerical_difference
from tools.p92_recoverable_spectrum import MODEL_ID
from tools.validate_p92_loading_path import nominal_path


def run(path, material, kernel, *, method="Radau", rtol=1e-9, subdivisions=1):
    if (type(subdivisions) is not int or subdivisions < 1
            or path["protocol"] not in {"RF", "SLR", "SLC"}
            or kernel.MODEL_ID != MODEL_ID or len(kernel.STATE_LAYOUT) != 12
            or kernel.STATE_LAYOUT[7] != "strain"):
        raise ValueError("Require the explicit recoverable spectrum and supported controls")
    t, e = nominal_path("SLR" if path["protocol"] == "SLC" else path["protocol"])
    definition = list(zip(np.diff(t), e[:-1], e[1:], strict=True))
    if len(path["segments"]) != len(definition)*path["cycles"]:
        raise ValueError("Source segments do not match the nominal protocol")
    state = np.zeros(len(kernel.STATE_LAYOUT))
    state[0], state[7] = path["initial_stress"], path["initial_strain"]
    initial_energy = kernel.stored_energy(state, material)
    records, holds, cycles, switches = [], [], [], []
    absolute_time = 0.
    for cycle in range(1, path["cycles"]+1):
        peak = valley = float(state[0])
        for local, (_, left, target) in enumerate(definition):
            index = len(records)
            observation = path["segments"][index]
            stress_control = path["protocol"] == "SLC" and local == 3
            hold = left == target
            if not hold and local == 4 and path["protocol"] == "SLC" and state[7] >= target:
                raise ValueError(f"Predicted creep exceeds positive target in cycle {cycle}")
            # Isolate ramp control, retaining the measured replay's hold-time
            # convention (including its documented RF departure exception).
            duration = float(observation["duration"] if hold else abs(target-state[7])/.001)
            if duration <= 0:
                raise ValueError("Zero-duration ramp requires protocol review")
            rate = 0. if hold else float(np.sign(target-state[7])*.001)
            incoming = state.copy()
            parts = []
            for _ in range(subdivisions):
                sol = kernel.integrate(state, duration/subdivisions, rate, material,
                    stress_control=stress_control, method=method, rtol=rtol)
                parts.append(sol)
                state = sol.y[:, -1].copy()
                peak = max(peak, float(sol.y[0].max()))
                valley = min(valley, float(sol.y[0].min()))

            def evaluate(query, duration=duration, parts=parts):
                query = np.asarray(query, dtype=float)
                if query.ndim != 1 or not np.isfinite(query).all() or np.any((query < 0) | (query > duration)):
                    raise ValueError("Nominal observation outside segment")
                width = duration/subdivisions
                indices = np.minimum((query/width).astype(int), subdivisions-1)
                values = np.empty((len(kernel.STATE_LAYOUT), len(query)))
                for j in np.unique(indices):
                    selected = indices == j
                    values[:, selected] = parts[j].sol(query[selected]-j*width)
                return values

            # Ramp values use relative phase only for the numerical comparison;
            # no observed ramp response enters this driver or an error objective.
            sample_times = np.asarray(observation["sample_times"])
            if not hold:
                sample_times = sample_times/observation["duration"]*duration
            samples = evaluate(sample_times)
            if hold:
                if observation["stress_control"] != stress_control:
                    raise ValueError("Observed hold control differs")
                holds.append(summarize(dict(observation, duration=duration), evaluate, index=index, strain_index=7))
            records.append(dict(segment=index, cycle=cycle, stress_control=stress_control,
                source_indices=observation["source_indices"], sampled_stress_MPa=samples[0].tolist(),
                sampled_strain=samples[7].tolist(), end_state=state.tolist()))
            switches.append(dict(cycle=cycle, segment=index, start_s=absolute_time,
                end_s=absolute_time+duration, duration_s=duration, strain_rate_per_s=rate,
                stress_control=stress_control, incoming_strain=float(incoming[7]),
                outgoing_strain=float(state[7]), captured_stress_MPa=float(incoming[0]) if stress_control else None))
            absolute_time += duration
        cycles.append(dict(cycle=cycle, sampled_peak_MPa=peak, sampled_valley_MPa=valley,
                           end_state=state.tolist()))
    return dict(model_id=kernel.MODEL_ID, state_layout=kernel.STATE_LAYOUT,
        protocol=path["protocol"], source_sha256=path["source_sha256"], cycles=cycles,
        segments=records, hold_diagnostics=holds, switches=switches,
        reversed_post_hold_ramps=[], incompatible_continuation=False,
        mechanical_balance_MPa=float(state[8]-state[9]-kernel.stored_energy(state, material)+initial_energy),
        engineering_qualified=False, life_qualified=False,
        scope="Nominal controls; hold-relative observations, not absolute recorded-time reproduction")


def evaluate(frozen, directory):
    kernel, material = decode_material(frozen["parameters"], frozen["model_id"])
    protocols = {}
    for name in ("RF", "SLR", "SLC"):
        source = read_source(directory/f"Data_{name}.lis", protocol=name)
        if source["sha256"] != frozen["source_sha256"][name]:
            raise ValueError("Changed source")
        path = measured_path(source, cycles=10)
        try:
            prediction = run(path, material, kernel)
            independent = run(path, material, kernel, method="BDF", rtol=1e-10)
            numerical = numerical_difference(prediction, independent, extra_strain_indices=(10, 11))
            comparison = compare_candidate(source, prediction)
            comparison.pop("groups")
            protocols[name] = dict(prediction=prediction, comparison=comparison, numerical=numerical,
                response_screen=screen(prediction, comparison, requested_cycles=10))
        except (ValueError, RuntimeError) as exc:
            protocols[name] = dict(failure=dict(type=type(exc).__name__, message=str(exc)),
                                   numerical=dict(passed=False))
        print(json.dumps(dict(protocol=name, completed="failure" not in protocols[name])), flush=True)
    return dict(schema="agentfem.research.autonomous-controls.v1", protocols=protocols,
        model_id=frozen["model_id"], parameters=frozen["parameters"], source_sha256=frozen["source_sha256"],
        fitted=False, engineering_qualified=False, life_qualified=False, independent_validation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "source_dir", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_autonomous_controls.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_hold_observations.py", "p92_recoverable_frozen.py", "p92_measured_controls.py",
             "p92_response_screen.py", "p92_two_rate_frozen.py", "p92_bam_source.py",
             "validate_p92_loading_path.py", "p92_bounded_softening.py")
    before = {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    result = evaluate(json.loads(raw), args.source_dir)
    if args.frozen.read_bytes() != raw or any(
            sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() != h for n, h in before.items()):
        raise RuntimeError("Frozen card or control driver changed")
    result["driver_sha256"] = before
    result["frozen_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)
