"""Fixed-card recorded-strain sensitivity, not a measured controller command."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_frozen import numerical_difference
from tools.p92_waveform_controls import attach_waveforms
from tools.p92_waveform_experiment import refine_knots


def recorded_strain_segments(path, source):
    """Keep predicted creep and continuous states; perturb strain controls only.

    After a force dwell the next ramp starts at the predicted strain, not the
    observed creep strain. Consequently this is not an exact absolute measured
    history there. Quantized observations are a sensitivity input, not commands.
    """
    out = attach_waveforms(path, source)
    selected = (source["cycle"] >= 1) & (source["cycle"] <= path["cycles"])
    ids = np.flatnonzero(selected)
    t, e = source["time_s"][ids], source["strain"][ids]
    start = 0.
    for seg, event in zip(out["segments"], out["switches"], strict=True):
        end = event["time_s"]
        if not seg["stress_control"]:
            inside = np.flatnonzero((t > start) & (t < end))
            times = np.r_[0., t[inside]-start, seg["duration"]]
            values = np.r_[np.interp(start, t, e), e[inside], np.interp(end, t, e)]
            chord = values[0]+(values[-1]-values[0])*times/seg["duration"]
            deviation = values-chord
            deviation[[0, -1]] = 0.
            seg["target"] = float(values[-1])
            seg["waveform"] = dict(times=times, deviation=deviation,
                measured_endpoints=values[[0, -1]].tolist(), source_indices=ids[inside].tolist())
        start = end
    out["waveform_scope"] = "Recorded strain-segment sensitivity; continuous predicted creep; not controller commands"
    return out


def audit(frozen, source_dir):
    if set(frozen["protocols"]) != {"RF", "SLR", "SLC"} or not all(
            p.get("numerical", {}).get("passed", False) for p in frozen["protocols"].values()):
        raise ValueError("Require numerically verified three-protocol frozen card")
    kernel, material = decode_material(frozen["parameters"], frozen["model_id"])
    if len(kernel.STATE_LAYOUT) != 12:
        raise ValueError("This paired diagnostic requires the twelve-state recoverable layout")
    protocols = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = measured_path(source, cycles=3)
        baseline = attach_waveforms(path, source)
        perturbed = recorded_strain_segments(path, source)
        arms = {}
        for name, controls in (("nominal_holds", baseline), ("recorded_strain_sensitivity", perturbed)):
            try:
                _, result = replay(controls, material, kernel=kernel)
                item = dict(completed=True, prediction=result,
                    response_screen=screen(result, compare_candidate(source, result), requested_cycles=3))
                if name == "recorded_strain_sensitivity":
                    _, independent = replay(controls, material, kernel=kernel, method="BDF", rtol=1e-10)
                    _, refined = replay(refine_knots(controls), material, kernel=kernel, rtol=1e-10)
                    item["independent"] = numerical_difference(result, independent, extra_strain_indices=(10, 11))
                    item["refinement"] = numerical_difference(result, refined, extra_strain_indices=(10, 11))
                arms[name] = item
            except (ValueError, RuntimeError) as exc:
                arms[name] = dict(completed=False, failure=str(exc))
        protocols[protocol] = arms
        print(json.dumps(dict(protocol=protocol, completed={k: v["completed"] for k, v in arms.items()})), flush=True)
    return dict(schema="agentfem.research.recorded-hold-sensitivity.v1", protocols=protocols,
        parameters=frozen["parameters"], model_id=frozen["model_id"], source_sha256=frozen["source_sha256"],
        fitted=False, acceptance_thresholds_changed=False, engineering_qualified=False,
        scope="Fixed-card input sensitivity; observed strain is not a verified controller command; not independent validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_recorded_hold_audit.py", "p92_bam_source.py", "p92_measured_controls.py",
             "p92_recoverable_frozen.py", "p92_recoverable_spectrum.py", "p92_recoverable_mobility_reference.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py", "p92_two_rate_frozen.py",
             "p92_waveform_controls.py", "p92_waveform_experiment.py", "p92_response_screen.py",
             "p92_hold_observations.py", "p92_bounded_softening.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    before = identities()
    result = audit(json.loads(raw), args.source_dir)
    if before != identities() or args.frozen.read_bytes() != raw:
        raise RuntimeError("Sensitivity inputs changed")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
