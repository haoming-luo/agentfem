"""Inspect and exercise the currently available nonlinear material tools.

This example deliberately distinguishes a FEM-integrated law from verified
material-point updates and a fatigue postprocessor.  It is a usage example,
not a substitute for the automated benchmarks.
"""

from __future__ import annotations

import numpy as np

from agentfem import constitutive


def main() -> None:
    for item in constitutive.capabilities():
        print(f"{item.name}: {item.maturity} — {item.available_scope}")

    j2 = constitutive.J2LinearIsotropicHardening(
        young=210.0e3,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=2.0e3,
    )
    state = None
    for strain in (0.0, 0.001, 0.003, 0.005, 0.002, -0.002):
        stress, state = constitutive.update_uniaxial(strain, j2, state)
        print(
            "J2",
            f"strain={strain:+.4e}",
            f"stress={stress:+.4e}",
            f"eqp={state.equivalent_plastic_strain:.4e}",
        )

    creep = constitutive.PowerLawCreep(
        coefficient=1.0e-8,
        stress_exponent=3.0,
        reference_stress=100.0,
        reference_time=1.0,
    )
    creep_history = constitutive.integrate_stress_history(
        creep,
        times=(0.0, 1.0, 10.0, 100.0),
        interval_stresses=(120.0, 120.0, 80.0),
    )
    print("creep history:", creep_history.as_dict())

    curve = constitutive.BasquinCurve(
        fatigue_strength_coefficient=1000.0,
        fatigue_strength_exponent=-0.1,
    )
    history = [0.0, 100.0, 0.0, 100.0, 0.0]
    assessment = constitutive.assess_history(history, curve)
    print("fatigue assessment:", assessment.as_dict())


if __name__ == "__main__":
    main()
