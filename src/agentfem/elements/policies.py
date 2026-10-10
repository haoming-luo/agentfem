# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit numerical formulations, separate from backend element identity."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class UniformStrainHex8:
    """Experimental uniform-gradient formulation with explicit kinematics."""

    hourglass_modulus: float
    hourglass_scale: float
    chunk_size: int = 1024
    kinematics: str = "small_strain"

    def __post_init__(self):
        if self.kinematics not in {"small_strain", "finite_strain"}:
            raise ValueError("kinematics must be small_strain or finite_strain.")
        for name in ("hourglass_modulus", "hourglass_scale"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite.")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")

    def summary(self):
        return {
            "formulation": "uniform_strain_hex8",
            "kinematics": self.kinematics,
            "maturity": "experimental",
            "hourglass_modulus": self.hourglass_modulus,
            "hourglass_scale": self.hourglass_scale,
            "chunk_size": self.chunk_size,
            "hourglass_control": "affine_corrected_energy_stiffness",
            **(
                {
                    "stress_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
                    "strain_shear_convention": "engineering",
                }
                if self.kinematics == "small_strain"
                else {
                    "stress_measure": "first_piola",
                    "kinematic_measure": "deformation_gradient",
                    "storage": "tensor_3x3",
                    "reference_measure": "volume",
                }
            ),
        }


def uniform_strain_hex8(
    *, hourglass_modulus, hourglass_scale, chunk_size=1024, kinematics="small_strain"
):
    """Declare stabilization explicitly; not an automatic C3D8R translation."""
    return UniformStrainHex8(hourglass_modulus, hourglass_scale, chunk_size, kinematics)
