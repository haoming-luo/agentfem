"""Frozen mobility card using the existing replay and unchanged response gates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools import p92_mobile_fraction_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_recoverable_frozen import decode_material
from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_frozen import numerical_difference
from tools.p92_waveform_controls import attach_waveforms
from tools.p92_waveform_experiment import refine_knots


def evaluate(fit, source_dir):
    if (fit["model_id"] != kernel.MODEL_ID or fit["training_cycles"] != [1, 2, 3]
            or not fit["fit"]["success"] or not fit["history_audit"]["passed"]):
        raise ValueError("Require successful early-only identifiable mobile-fraction fit")
    card = fit["fit"]["parameters"]
    _, base = decode_material(card["spectrum"], spectrum.MODEL_ID)
    material = kernel.Material(base, card["immobilization"], card["remobilization"])
    protocols = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != fit["source_sha256"][protocol]:
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
            protocols[protocol] = dict(response_screen=dict(status="failed"),
                numerical=dict(passed=False, completed=False),
                failure=dict(stage=stage, type=type(exc).__name__, message=str(exc)))
            continue
        comparison = compare_candidate(source, prediction)
        for row in comparison["records"]:
            row["role"] = "early_calibration" if row["cycle"] <= 3 else "inspected_development"
        comparison.pop("groups")
        checks = [numerical_difference(prediction, other, extra_strain_indices=(10, 11),
                  dimensionless_indices=(12,)) for other in (independent, refined)]
        protocols[protocol] = dict(prediction=prediction, comparison=comparison,
            response_screen=screen(prediction, comparison, requested_cycles=10),
            numerical=dict(independent=checks[0], refinement=checks[1],
                           passed=all(c["passed"] for c in checks)))
        print(json.dumps(dict(protocol=protocol,
              response=protocols[protocol]["response_screen"]["status"],
              numerical=protocols[protocol]["numerical"]["passed"])), flush=True)
    return dict(schema="agentfem.research.recoverable-frozen.v1", model_id=kernel.MODEL_ID,
        parameters=card, source_sha256=fit["source_sha256"], protocols=protocols, fitted=True,
        independent_validation=False, engineering_qualified=False, life_qualified=False,
        scope="Frozen continuous development cycles; original gates, no late refit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "source_dir", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    names = ("p92_mobile_fraction_frozen.py", "p92_mobile_fraction_reference.py",
        "p92_recoverable_spectrum.py", "p92_two_rate_reference.py", "p92_recoverable_frozen.py",
        "p92_two_rate_controls.py", "p92_two_rate_frozen.py", "p92_measured_controls.py",
        "p92_bam_source.py", "p92_waveform_controls.py", "p92_waveform_experiment.py",
        "p92_response_screen.py", "p92_hold_observations.py", "p92_bounded_softening.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    raw, before = args.fit.read_bytes(), identities()
    result = evaluate(json.loads(raw), args.source_dir)
    if before != identities() or raw != args.fit.read_bytes():
        raise RuntimeError("Frozen inputs changed")
    result.update(driver_sha256=before, fit_file_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
