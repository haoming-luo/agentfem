"""Private Result assembly for direct harmonic response."""

from __future__ import annotations

import numpy as np

from .core import from_solution
from .core import SimulationResult
from .execution import add_execution_trace
from .lifecycle import complete_result


def from_harmonic_step(step, *, output=None, strict_output: bool = False):
    """Publish phasor fields and separated harmonic evidence."""

    result = from_solution(
        step.solution_real,
        name=step.name,
        field_name="U_REAL",
        metadata={"step": step.summary()},
        scientific_inputs={
            "harmonic_system": step.system,
            "harmonic_excitation": {
                "frequency": step.frequency,
                "angular_frequency": step.angular_frequency,
                "phasor_convention": step.system.phasor_convention,
                "load_phase": step.load_phase,
            },
        },
    )
    result.add_field(
        "U_IMAG",
        step.solution_imaginary,
        description="Imaginary displacement phasor component.",
        processing={"phasor_convention": step.system.phasor_convention},
    )
    result.add_field(
        "U_AMPLITUDE",
        step.displacement_amplitude,
        description="Component-wise displacement phasor amplitude.",
        processing={"method": "hypot(U_REAL,U_IMAG)"},
    )
    result.add_field(
        "U_PHASE",
        step.displacement_phase,
        unit="rad",
        description="Component-wise displacement phase.",
        processing={"method": "atan2(U_IMAG,U_REAL)"},
    )
    result.add_quantity("frequency", step.frequency, unit="Hz")
    result.add_quantity("angular_frequency", step.angular_frequency, unit="rad/s")
    result.add_quantities(step.energy_evidence(), kind="energy_evidence")
    if step.algebraic_equilibrium is None:
        raise RuntimeError(
            "Harmonic algebraic-equilibrium evidence requires a completed solve."
        )
    result.add_quantities(step.algebraic_equilibrium, kind="solver_evidence")
    return complete_result(
        step,
        result,
        output=output,
        fields=("U_REAL", "U_IMAG", "U_AMPLITUDE", "U_PHASE"),
        strict_output=strict_output,
    )


def from_harmonic_sweep(step, *, strict_output: bool = False):
    """Publish canonical real-valued histories from a completed sweep."""

    del strict_output
    records = step.canonical_records()
    frequencies = step.frequencies
    result = SimulationResult(
        name=step.name,
        metadata={
            "step": step.summary(),
            "last_solved_frequency": step.last_live_field_frequency,
            "field_retention": "last_executed_frequency_only",
            "live_field_state": (
                "available_for_last_executed_frequency"
                if step.last_live_field_frequency is not None
                else "not_restored_scalar_checkpoint_only"
            ),
            "phase_convention": "wrapped atan2 in [-pi,pi]",
            "zero_amplitude_phase": "numerically reported but physically undefined",
        },
        scientific_inputs=step.scientific_inputs(),
    )
    common = {
        "maximum_displacement_vector_amplitude": [
            record["maximum_displacement_vector_amplitude"] for record in records
        ],
    }
    energy_names = tuple(records[0]["energy"])
    equilibrium_names = tuple(records[0]["equilibrium"])
    common.update(
        {
            name: [record["energy"][name] for record in records]
            for name in energy_names
        }
    )
    common.update(
        {
            name: [record["equilibrium"][name] for record in records]
            for name in equilibrium_names
        }
    )
    result.add_histories(
        frequencies,
        common,
        abscissa_name="frequency",
        abscissa_unit="Hz",
    )

    if records[0]["solve"] is not None:
        result.add_histories(
            frequencies,
            {
                "linear_converged": [
                    float(record["solve"]["converged"]) for record in records
                ],
                "linear_converged_reason": [
                    record["solve"]["converged_reason"] for record in records
                ],
                "linear_iterations": [
                    record["solve"]["iterations"] for record in records
                ],
                "linear_reported_residual_norm": [
                    record["solve"]["residual_norm"] for record in records
                ],
            },
            abscissa_name="frequency",
            abscissa_unit="Hz",
        )

    for response in step.responses:
        values = [record["responses"][response.name] for record in records]
        real = [value.real for value in values]
        imaginary = [value.imag for value in values]
        amplitude = [abs(value) for value in values]
        phase = [np.angle(value) for value in values]
        result.add_histories(
            frequencies,
            {
                f"{response.name}_REAL": real,
                f"{response.name}_IMAG": imaginary,
                f"{response.name}_AMPLITUDE": amplitude,
                f"{response.name}_PHASE": phase,
            },
            units={
                f"{response.name}_REAL": response.unit,
                f"{response.name}_IMAG": response.unit,
                f"{response.name}_AMPLITUDE": response.unit,
                f"{response.name}_PHASE": "rad",
            },
            abscissa_name="frequency",
            abscissa_unit="Hz",
        )
        peak_index = int(np.argmax(amplitude))
        result.add_quantity(
            f"peak_{response.name}_frequency",
            frequencies[peak_index],
            unit="Hz",
        )
        result.add_quantity(
            f"peak_{response.name}_amplitude",
            amplitude[peak_index],
            unit=response.unit,
        )

    amplitudes = result.histories["maximum_displacement_vector_amplitude"].values
    peak_index = int(np.argmax(amplitudes))
    result.add_quantity("peak_frequency", frequencies[peak_index], unit="Hz")
    result.add_quantity(
        "peak_maximum_displacement_vector_amplitude",
        amplitudes[peak_index],
    )
    add_execution_trace(result, step.execution_events)
    result.metadata["execution"].update(
        {
            "retention": "bounded_milestone_preferred",
            "max_events": step.execution_event_capacity,
            "dropped_events": step.dropped_execution_events,
        }
    )
    for checkpoint in step.checkpoints:
        result.add_checkpoint(checkpoint)
    return complete_result(step, result)


__all__ = ()
