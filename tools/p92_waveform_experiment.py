"""Frozen-card ramp-shape comparison; no parameter fitting or qualification."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path, replay
from tools.p92_measured_fit import write_new
from tools.p92_published_recovery_reference import Parameters
from tools.p92_response_screen import screen
from tools.p92_two_mechanism import Mechanism
from tools.p92_waveform_controls import attach_waveforms


def refine_knots(path):
    out = deepcopy(path)
    for seg in out["segments"]:
        if "waveform" in seg:
            w = seg["waveform"]
            t = np.asarray(w["times"])
            new = np.sort(np.r_[t, .5*(t[1:]+t[:-1])])
            w["deviation"] = np.interp(new, t, w["deviation"])
            w["times"] = new
    return out


def difference(a, b):
    if (len(a["cycles"]) != len(b["cycles"])
            or len(a["segment_endpoints"]) != len(b["segment_endpoints"])):
        return dict(passed=False, reason="Different completed path")
    delta = np.abs(np.array([x["state"] for x in a["segment_endpoints"]])-
                   np.array([x["state"] for x in b["segment_endpoints"]]))
    stress = float(delta[:, [0, 1, 2, 3, 5, 6, 7]].max())
    strain = float(delta[:, [4, 8, 9]].max())
    extrema = max(abs(x[key]-y[key]) for x, y in zip(a["cycles"], b["cycles"], strict=True)
                  for key in ("sampled_peak_MPa", "sampled_valley_MPa")) if a["cycles"] else 0.
    return dict(maximum_stress_state_difference_MPa=stress,
                maximum_strain_state_difference=strain,
                maximum_sampled_extrema_difference_MPa=extrema,
                passed=stress < 1e-4 and strain < 1e-8 and extrema < 1e-4)


def experiment(seed, sources, *, cycles=10, protocols=("RF", "SLR", "SLC")):
    driver_files = ("tools/p92_waveform_controls.py", "tools/p92_waveform_experiment.py",
                    "tools/p92_measured_controls.py", "tools/p92_two_mechanism.py",
                    "tools/p92_response_screen.py")
    driver_identity = {p: sha256(Path(p).read_bytes()).hexdigest() for p in driver_files}
    digest = sha256(json.dumps(seed["parameters"], sort_keys=True).encode()).hexdigest()
    if digest != seed["parameter_sha256"] or not seed["fit"]["success"]:
        raise ValueError("Require intact frozen converged candidate")
    if (not protocols or len(set(protocols)) != len(protocols)
            or not set(protocols) <= {"RF", "SLR", "SLC"}
            or len(sources) != len(protocols) or {d["protocol"] for d in sources} != set(protocols)):
        raise ValueError("Require the declared distinct source protocols")
    mechanisms = tuple(Mechanism(Parameters(**m["parameters"]), m["exponent"], m["enabled"])
                       for m in seed["parameters"])
    started, reports = perf_counter(), []
    for source in sources:
        if source["sha256"] != seed["source_sha256"][source["protocol"]]:
            raise ValueError("Frozen source identity changed")
        linear = measured_path(source, cycles=cycles)
        shaped = attach_waveforms(linear, source)
        residual0, baseline = replay(linear, mechanisms)
        residual1, candidate = replay(shaped, mechanisms)
        _, fine = replay(shaped, mechanisms, method="BDF", rtol=1e-10)
        _, refined = replay(refine_knots(shaped), mechanisms, rtol=1e-10)
        record = dict(protocol=source["protocol"], parameter_sha256=digest,
            linear=baseline, waveform=candidate,
            linear_comparison=compare_candidate(source, baseline),
            waveform_comparison=compare_candidate(source, candidate),
            linear_cost=float(.5*residual0@residual0),
            waveform_cost=float(.5*residual1@residual1),
            independent_integrator=difference(candidate, fine),
            same_shape_knot_refinement=difference(candidate, refined))
        for key in ("linear_comparison", "waveform_comparison"):
            for row in record[key]["records"]:
                row["role"] = "calibration" if row["cycle"] <= 3 else "inspected_development"
            record[key].pop("groups")
            record[key]["scope"] = "Same observed cycle extrema; frozen measured-control development comparison"
        record["linear_response_screen"] = screen(baseline, record["linear_comparison"], requested_cycles=cycles)
        record["waveform_response_screen"] = screen(candidate, record["waveform_comparison"], requested_cycles=cycles)
        record["waveform_switch_brackets"] = {}
        for side in ("lower", "upper"):
            bounded = attach_waveforms(measured_path(source, cycles=cycles, switch_bound=side), source)
            _, sensitivity = replay(bounded, mechanisms)
            record["waveform_switch_brackets"][side] = dict(cycles=sensitivity["cycles"],
                hold_diagnostics=sensitivity["hold_diagnostics"],
                reversed_post_hold_ramps=sensitivity["reversed_post_hold_ramps"])
        if source["protocol"] == "SLC":
            # Attribute the first inherited stress without resetting the
            # preceding compression/dwell history. Effects need not add.
            record["first_cycle_single_ramp_shape"] = {}
            for selected in (0, 2):
                one = attach_waveforms(measured_path(source, cycles=1), source)
                for i, seg in enumerate(one["segments"]):
                    if i != selected:
                        seg.pop("waveform", None)
                _, isolated = replay(one, mechanisms)
                record["first_cycle_single_ramp_shape"][str(selected)] = dict(
                    hold_entry_state=isolated["segment_endpoints"][2]["state"],
                    hold=next(h for h in isolated["hold_diagnostics"] if h["stress_control"]))
            prefix = attach_waveforms(measured_path(source, cycles=1), source, mode="recorded_prefix")
            _, recorded = replay(prefix, mechanisms)
            _, recorded_fine = replay(prefix, mechanisms, method="BDF", rtol=1e-10)
            record["first_cycle_recorded_strain_prefix"] = dict(
                prediction=recorded, numerical=difference(recorded, recorded_fine))
        reports.append(record)
        print(json.dumps(dict(protocol=source["protocol"], linear_cost=record["linear_cost"],
            waveform_cost=record["waveform_cost"], numerical=record["independent_integrator"],
            refinement=record["same_shape_knot_refinement"])), flush=True)
    if driver_identity != {p: sha256(Path(p).read_bytes()).hexdigest() for p in driver_files}:
        raise RuntimeError("Research driver changed during execution; rerun frozen code")
    return dict(schema="agentfem.research.p92-waveform.v1", parameters=seed["parameters"],
        parameter_sha256=digest, source_sha256=seed["source_sha256"], cycles=cycles,
        tested_protocols=list(protocols),
        parameter_refit=False, initial_state="Measured initial stress/strain; virgin hidden histories, unchanged",
        scope="Endpoint-preserving measured strain-ramp shape; holds unchanged; not absolute controller replay",
        later_cycles_role="Previously inspected development, not blind validation",
        driver_sha256=driver_identity,
        records=reports, elapsed_seconds=perf_counter()-started,
        engineering_qualified=False, life_qualified=False, independent_validation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--protocol", choices=("RF", "SLR", "SLC"),
                        help="Explicit partial-protocol diagnostic; default checks all three")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    seed = json.loads(args.seed.read_text())
    protocols = (args.protocol,) if args.protocol else ("RF", "SLR", "SLC")
    sources = [read_source(args.source_dir/f"Data_{p}.lis", protocol=p) for p in protocols]
    report = experiment(seed, sources, cycles=args.cycles, protocols=protocols)
    report["seed_file_sha256"] = sha256(args.seed.read_bytes()).hexdigest()
    write_new(args.output, report)
