"""Frozen recoverable hypothesis: all response gates and every state checked."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import compare_candidate
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools import p92_recoverable_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools import p92_primary_creep_reference as primary
from tools import p92_shared_drag_reference as shared
from tools import p92_reversal_memory_reference as memory
from tools import p92_transient_strength_reference as transient
from tools import p92_recovery_replacement_reference as replacement
from tools import p92_recoverable_mobility_reference as mobility
from tools import p92_critical_recovery_reference as critical
from tools.p92_response_screen import screen
from tools.p92_two_rate_controls import replay
from tools.p92_two_rate_frozen import numerical_difference
from tools.p92_two_rate_reference import Branch, Material
from tools.p92_waveform_controls import attach_waveforms
from tools.p92_waveform_experiment import refine_knots


def decode_material(card, model_id):
    """One explicit identity decoder for research replay and diagnostics."""
    if model_id == critical.MODEL_ID:
        _, base = decode_material(card["spectrum"], spectrum.MODEL_ID)
        return critical, critical.Material(base)
    if model_id == mobility.MODEL_ID:
        _, base = decode_material(card["spectrum"], spectrum.MODEL_ID)
        return mobility, mobility.Material(base, card["feedback"])
    if model_id == replacement.MODEL_ID:
        branch = dict(card["branch"])
        for name in ("hardening", "saturation", "static_recovery", "ratcheting"):
            branch[name] = tuple(branch[name])
        return replacement, replacement.Material(card["modulus"], Branch(**branch),
            tuple(card["moduli"]), tuple(card["times"]))
    if model_id == transient.MODEL_ID:
        _, base = decode_material(card["spectrum"], spectrum.MODEL_ID)
        return transient, transient.Material(base, card["maximum"], card["build"], card["recovery"])
    if model_id == primary.MODEL_ID:
        _, material = decode_material(card, spectrum.MODEL_ID)
        primary.validate_material(material)
        return primary, material
    if model_id == shared.MODEL_ID:
        return shared, shared.Material(**card)
    if model_id == memory.MODEL_ID:
        _, base = decode_material(card["spectrum"], spectrum.MODEL_ID)
        return memory, memory.Material(base, tuple(card["ratios"]), card["rate"], card["coordinate"])
    backbone = card["backbone"]
    base = Material(backbone["modulus"], tuple(Branch(**b) for b in backbone["branches"]))
    if model_id == kernel.MODEL_ID:
        module = kernel
        mat = kernel.Material(base, card["recoverable_modulus"], card["relaxation_time"],
            strength_ratio=card.get("strength_ratio", 1.),
            strength_softening_rate=card.get("strength_softening_rate", 0.))
    elif model_id == spectrum.MODEL_ID:
        module = spectrum
        mat = spectrum.Material(base, tuple(card["moduli"]), tuple(card["times"]),
            tuple(card.get("strength_ratios", (1., 1.))), tuple(card.get("strength_rates", (0., 0.))),
            tuple(card.get("hardening_ratios", (1., 1.))), tuple(card.get("hardening_rates", (0., 0.))),
            tuple(card.get("static_powers", (0., 0.))),
            tuple(card.get("static_ratios", (1., 1.))),
            tuple(card.get("static_history_rates", (0., 0.))), card.get("static_history", "own"),
            card.get("directional_recovery", False), tuple(card.get("recoverable_powers", (1., 1.))))
    else:
        raise ValueError("Unknown recoverable model identity")
    return module, mat


def evaluate(fit, source_dir, *, fixed_card=False):
    if type(fixed_card) is not bool:
        raise ValueError("fixed_card must be explicit boolean")
    if not fixed_card and (fit["training_cycles"] != [1, 2, 3] or not fit["fit"]["success"]):
        raise ValueError("Require converged early-only joint identification")
    card = fit["parameters"] if fixed_card else fit["fit"]["parameters"]
    model_id = fit.get("model_id", kernel.MODEL_ID)
    module, mat = decode_material(card, model_id)
    extra_indices = tuple(range(10, len(module.STATE_LAYOUT)))
    numerical_options = dict(extra_strain_indices=extra_indices)
    if module is shared:
        numerical_options.update(stress_state_indices=(0, 1, 2, 4), dimensionless_indices=(5,))
    if module is transient:
        numerical_options.update(extra_strain_indices=(10, 11), dimensionless_indices=(12,))
    if module is replacement:
        numerical_options.update(stress_state_indices=(0, 1, 2), extra_strain_indices=(4, 5))
    results = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != fit["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = attach_waveforms(measured_path(source, cycles=10), source)
        stage = "prediction"
        try:
            _, prediction = replay(path, mat, kernel=module)
            stage = "independent_integration"
            _, bdf = replay(path, mat, kernel=module, method="BDF", rtol=1e-10)
            stage = "waveform_refinement"
            _, refined = replay(refine_knots(path), mat, kernel=module, rtol=1e-10)
        except (ValueError, RuntimeError) as exc:
            # A rejected physical path is evidence, not a missing output file.
            # Source/identity errors above remain hard errors, never model failures.
            results[protocol] = dict(response_screen=dict(status="failed"),
                numerical=dict(passed=False, completed=False),
                failure=dict(stage=stage, type=type(exc).__name__, message=str(exc)))
            print(json.dumps(dict(protocol=protocol, **results[protocol])), flush=True)
            continue
        independent = numerical_difference(prediction, bdf, **numerical_options)
        refinement = numerical_difference(prediction, refined, **numerical_options)
        comparison = compare_candidate(source, prediction)
        for row in comparison["records"]:
            row["role"] = ("early_mechanism_discrimination" if fixed_card else "early_calibration") if row["cycle"] <= 3 else "inspected_development"
        comparison.pop("groups")
        results[protocol] = dict(prediction=prediction, comparison=comparison,
            response_screen=screen(prediction, comparison, requested_cycles=10),
            numerical=dict(independent=independent, refinement=refinement,
                           passed=independent["passed"] and refinement["passed"]))
        print(json.dumps(dict(protocol=protocol, response=results[protocol]["response_screen"]["status"],
                              numerical=results[protocol]["numerical"])), flush=True)
    return dict(schema="agentfem.research.recoverable-frozen.v1", parameters=card, model_id=model_id,
        source_sha256=fit["source_sha256"], protocols=results,
        fitted=not fixed_card, independent_validation=False, engineering_qualified=False, life_qualified=False,
        scope="Continuous frozen path; later cycles already inspected development data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.fit.read_bytes()
    names = ("p92_recoverable_reference.py", "p92_recoverable_spectrum.py", "p92_two_rate_reference.py",
             "p92_primary_creep_reference.py",
             "p92_shared_drag_reference.py", "p92_reversal_memory_reference.py",
             "p92_transient_strength_reference.py",
             "p92_recoverable_mobility_reference.py",
             "p92_critical_recovery_reference.py",
             "p92_recovery_replacement_reference.py", "p92_hold_observations.py",
             "p92_two_rate_controls.py", "p92_recoverable_frozen.py", "p92_two_rate_frozen.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_response_screen.py")
    def identities():
        return {n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    before = identities()
    result = evaluate(json.loads(raw), args.source_dir)
    if before != identities() or args.fit.read_bytes() != raw:
        raise RuntimeError("Frozen card or driver changed during replay")
    result["driver_sha256"] = before
    result["fit_file_sha256"] = sha256(raw).hexdigest()
    write_new(args.output, result)
