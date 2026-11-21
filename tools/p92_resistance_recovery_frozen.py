"""Fixed-card negative-hypothesis continuation; no identification or promotion."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_resistance_recovery_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
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


def evaluate(frozen, source_dir, kind):
    module, base = decode_material(frozen["parameters"], frozen["model_id"])
    if module is not spectrum or kind not in {"strength", "mobility"}:
        raise ValueError("Require frozen spectrum and declared mechanism")
    tau = 180. if kind == "strength" else 1800.
    material = kernel.Material(base, kind, .9, 250., 1/tau)
    results = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(source, cycles=10), source)
        stage = "prediction"
        try:
            _, prediction = replay(path, material, kernel=kernel)
            stage = "independent_integration"
            _, independent = replay(path, material, kernel=kernel, method="BDF", rtol=1e-10)
            stage = "waveform_refinement"
            _, refined = replay(refine_knots(path), material, kernel=kernel, rtol=1e-10)
        except (ValueError, RuntimeError) as exc:
            results[protocol] = dict(response_screen=dict(status="failed"),
                numerical=dict(passed=False, completed=False),
                failure=dict(stage=stage, type=type(exc).__name__, message=str(exc)))
            print(json.dumps(dict(protocol=protocol, failure=results[protocol]["failure"])), flush=True)
            continue
        options = dict(extra_strain_indices=(10, 11), dimensionless_indices=(12, 13))
        checks = dict(independent=numerical_difference(prediction, independent, **options),
                      refinement=numerical_difference(prediction, refined, **options))
        checks["passed"] = checks["independent"]["passed"] and checks["refinement"]["passed"]
        comparison = compare_candidate(source, prediction)
        comparison.pop("groups")
        for row in comparison["records"]:
            row["role"] = "early_mechanism_discrimination" if row["cycle"] <= 3 else "inspected_development"
        results[protocol] = dict(comparison=comparison,
            response_screen=screen(prediction, comparison, requested_cycles=10),
            numerical=checks, hold_diagnostics=prediction["hold_diagnostics"],
            mechanical_balance_MPa=prediction["mechanical_balance_MPa"])
        print(json.dumps(dict(protocol=protocol, numerical=checks["passed"],
                              response=results[protocol]["response_screen"]["status"])), flush=True)
    return dict(schema="agentfem.research.resistance-recovery-frozen.v1",
        model_id=kernel.MODEL_ID, parameters=asdict(material),
        source_sha256=frozen["source_sha256"], protocols=results,
        fitted=False, engineering_qualified=False, life_qualified=False,
        independent_validation=False,
        scope="Frozen negative-hypothesis check; later cycles already inspected, not a blind holdout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=("strength", "mobility"), required=True)
    args = parser.parse_args()
    names = ("p92_resistance_recovery_frozen.py", "p92_resistance_recovery_reference.py",
             "p92_recoverable_spectrum.py", "p92_two_rate_reference.py", "p92_two_rate_controls.py",
             "p92_two_rate_frozen.py", "p92_recoverable_frozen.py", "p92_measured_controls.py",
             "p92_waveform_controls.py", "p92_waveform_experiment.py", "p92_response_screen.py",
             "p92_bam_source.py", "p92_hold_observations.py", "p92_bounded_softening.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw, before = args.frozen.read_bytes(), identities()
    result = evaluate(json.loads(raw), args.source_dir, args.kind)
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Input or implementation changed during verification")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
