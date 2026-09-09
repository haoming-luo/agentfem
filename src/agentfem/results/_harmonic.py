"""Private Result assembly for direct harmonic response."""

from __future__ import annotations

from .core import from_solution
from .lifecycle import complete_result


def from_harmonic_step(step, *, output=None, strict_output: bool = False):
    """Publish phasor fields and separated harmonic evidence."""

    result = from_solution(
        step.solution_real,
        name=step.name,
        field_name="U_REAL",
        metadata={"step": step.summary()},
        scientific_inputs={
            "harmonic_system": step.system.summary(),
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


__all__ = ()
