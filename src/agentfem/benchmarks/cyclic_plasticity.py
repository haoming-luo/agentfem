# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Public cyclic-plasticity material benchmarks.

The OFHC copper paths and comparison values come from the SIMULIA example
``cyclictests``.  AgentFEM reconstructs the calibrated combined-hardening law
and solves the same mixed material control, rather than storing AgentFEM's own
response as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constitutive


_ABAQUS_CYCLIC_URL = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMACAEBMKRefMap/simabmk-c-cyclictests.htm"
)


@dataclass(frozen=True)
class CyclicPlasticityBenchmark:
    """Compact acceptance evidence against two published cyclic observables."""

    symmetric_final_peeq: float
    symmetric_reference_peeq: float
    tension_torsion_maximum_normal_stress: float
    tension_torsion_reference_stress: float
    fourth_to_final_cycle_change: float
    relative_tolerance: float
    saturation_tolerance: float
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.cyclic-plasticity-benchmark.v1",
            "benchmark": "abaqus_ofhc_copper_cyclic",
            "source": _ABAQUS_CYCLIC_URL,
            "symmetric_final_peeq": self.symmetric_final_peeq,
            "symmetric_reference_peeq": self.symmetric_reference_peeq,
            "symmetric_relative_error": self.symmetric_relative_error,
            "tension_torsion_maximum_normal_stress": (
                self.tension_torsion_maximum_normal_stress
            ),
            "tension_torsion_reference_stress": (
                self.tension_torsion_reference_stress
            ),
            "tension_torsion_relative_error": (
                self.tension_torsion_relative_error
            ),
            "fourth_to_final_cycle_change": self.fourth_to_final_cycle_change,
            "relative_tolerance": self.relative_tolerance,
            "saturation_tolerance": self.saturation_tolerance,
            "accepted": self.accepted,
        }

    @property
    def symmetric_relative_error(self) -> float:
        return abs(self.symmetric_final_peeq - self.symmetric_reference_peeq) / abs(
            self.symmetric_reference_peeq
        )

    @property
    def tension_torsion_relative_error(self) -> float:
        return abs(
            self.tension_torsion_maximum_normal_stress
            - self.tension_torsion_reference_stress
        ) / abs(self.tension_torsion_reference_stress)


def _abaqus_ofhc_copper():
    equivalent_plastic_strain = (
        0.0,
        0.0068,
        0.0340,
        0.0612,
        0.0884,
        0.1156,
        0.1428,
        0.1700,
        0.1972,
        0.2244,
        0.2516,
    )
    yield_stress = (
        13.0,
        13.0,
        29.37,
        50.13,
        63.91,
        73.01,
        80.76,
        86.25,
        90.498,
        93.95,
        96.155,
    )
    isotropic = constitutive.TabulatedIsotropicHardening(
        equivalent_plastic_strain=equivalent_plastic_strain,
        yield_stress=yield_stress,
        extrapolation="constant",
    )
    return constitutive.chaboche(
        young=104_000.0,
        poisson=0.3,
        yield_stress=13.0,
        backstresses=((33_550.0, 701.3),),
        isotropic_hardening=isotropic,
        name="Abaqus OFHC copper combined hardening",
    )


def _uniaxial_stress_free_transverse_path(*, substeps_per_half_cycle: int):
    anchor_coordinate = np.asarray(
        [0.0, 5.0, *np.arange(15.0, 186.0, 10.0)],
        dtype=float,
    )
    anchor_axial_strain = np.asarray(
        [0.0, 0.0075, *([-0.0075, 0.0075] * 9)],
        dtype=float,
    )
    strain = np.zeros((anchor_coordinate.size, 3, 3))
    strain[:, 0, 0] = anchor_axial_strain
    stress = np.zeros_like(strain)
    strain_control = np.zeros((3, 3), dtype=bool)
    strain_control[0, 0] = True
    return constitutive.material_mixed_path(
        anchor_coordinate,
        strain=strain,
        stress=stress,
        strain_control=strain_control,
        name="abaqus_ofhc_symmetric_strain_cycle",
        coordinate_name="load_time",
    ).refine(substeps_per_half_cycle)


def _tension_torsion_path(*, points_per_cycle: int, cycle_count: int = 9):
    ramp = np.linspace(0.0, 1.0, 11)
    phase = np.linspace(
        0.0,
        2.0 * cycle_count,
        points_per_cycle * cycle_count + 1,
    )[1:]
    coordinate = np.concatenate((ramp, 1.0 + phase))
    strain = np.zeros((coordinate.size, 3, 3))
    stress = np.zeros_like(strain)
    strain[: ramp.size, 0, 0] = 0.01 * ramp
    strain[ramp.size :, 0, 0] = 0.01 * np.cos(np.pi * phase)
    engineering_shear = np.sqrt(3.0) * 0.01 * np.sin(np.pi * phase)
    strain[ramp.size :, 0, 1] = 0.5 * engineering_shear
    strain[ramp.size :, 1, 0] = 0.5 * engineering_shear
    strain_control = np.zeros((3, 3), dtype=bool)
    strain_control[0, 0] = True
    strain_control[0, 1] = strain_control[1, 0] = True
    return constitutive.material_mixed_path(
        coordinate,
        strain=strain,
        stress=stress,
        strain_control=strain_control,
        name="abaqus_ofhc_tension_torsion_cycle",
        coordinate_name="load_time",
    )


def abaqus_ofhc_copper_cyclic_benchmark(
    *,
    substeps_per_half_cycle: int = 10,
    points_per_cycle: int = 20,
    relative_tolerance: float = 0.01,
    saturation_tolerance: float = 0.01,
):
    """Run the published symmetric and nonproportional OFHC copper tests.

    Returns the external assessment plus the two common ``SimulationResult``
    objects.  The 1% default gates are AgentFEM acceptance contracts, not
    tolerances claimed by SIMULIA.
    """

    if int(substeps_per_half_cycle) != substeps_per_half_cycle or int(
        substeps_per_half_cycle
    ) < 1:
        raise ValueError("substeps_per_half_cycle must be a positive integer.")
    if int(points_per_cycle) != points_per_cycle or int(points_per_cycle) < 8:
        raise ValueError("points_per_cycle must be an integer of at least 8.")
    if relative_tolerance <= 0.0 or saturation_tolerance <= 0.0:
        raise ValueError("Benchmark tolerances must be positive.")

    material = _abaqus_ofhc_copper()
    symmetric = material.history(
        _uniaxial_stress_free_transverse_path(
            substeps_per_half_cycle=int(substeps_per_half_cycle)
        ),
        name="abaqus_ofhc_symmetric_cyclic",
    ).solve()
    tension_torsion = material.history(
        _tension_torsion_path(points_per_cycle=int(points_per_cycle)),
        name="abaqus_ofhc_tension_torsion",
    ).solve()

    ramp_points = 11
    points_per_cycle = int(points_per_cycle)
    axial = tension_torsion.stress[:, 0, 0]
    cycle_four = float(
        np.max(
            axial[
                ramp_points + 3 * points_per_cycle :
                ramp_points + 4 * points_per_cycle
            ]
        )
    )
    final_cycle = float(np.max(axial[-points_per_cycle:]))
    saturation_change = abs(final_cycle - cycle_four) / abs(final_cycle)
    symmetric_reference = 0.2367
    normal_stress_reference = 143.1
    symmetric_value = float(symmetric.equivalent_plastic_strain[-1])
    assessment = CyclicPlasticityBenchmark(
        symmetric_final_peeq=symmetric_value,
        symmetric_reference_peeq=symmetric_reference,
        tension_torsion_maximum_normal_stress=final_cycle,
        tension_torsion_reference_stress=normal_stress_reference,
        fourth_to_final_cycle_change=saturation_change,
        relative_tolerance=float(relative_tolerance),
        saturation_tolerance=float(saturation_tolerance),
        accepted=bool(
            abs(symmetric_value - symmetric_reference) / symmetric_reference
            <= relative_tolerance
            and abs(final_cycle - normal_stress_reference) / normal_stress_reference
            <= relative_tolerance
            and saturation_change <= saturation_tolerance
        ),
    )
    results = {
        "symmetric": symmetric.to_result(),
        "tension_torsion": tension_torsion.to_result(),
    }
    for result in results.values():
        result.metadata["external_benchmark"] = assessment.as_dict()
        result.metadata["external_benchmark"]["acceptance_tolerances_are"] = (
            "agentfem_defined"
        )
    return assessment, results


__all__ = [
    "CyclicPlasticityBenchmark",
    "abaqus_ofhc_copper_cyclic_benchmark",
]
