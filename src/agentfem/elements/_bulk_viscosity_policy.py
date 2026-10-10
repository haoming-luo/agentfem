# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private explicit damping coefficients with an inspectable geometry rule.

Physical reference density is not a mass-scaling parameter. The length rule
is volume-equivalent, not an assertion of vendor characteristic-length parity.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BulkViscosityPolicy:
    reference_density: float
    wave_speed: float
    reference_length: float
    linear: float = 0.06
    quadratic: float = 1.2

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, (bool, complex))
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value < 0
                or (name not in ("linear", "quadratic") and value == 0)
            ):
                raise ValueError(f"Invalid bulk viscosity {name}.")

    def options(self, jacobian):
        j = np.asarray(jacobian)
        if np.iscomplexobj(j) or not np.isfinite(j).all() or np.any(j <= 0):
            raise ValueError("Bulk viscosity requires positive finite Jacobians.")
        return {
            "density": self.reference_density / j,
            "wave_speed": self.wave_speed,
            "characteristic_length": self.reference_length * np.cbrt(j),
            "linear": self.linear,
            "quadratic": self.quadratic,
        }

    def summary(self):
        return {
            **{name: float(getattr(self, name)) for name in self.__dataclass_fields__},
            "length_rule": "reference_length_times_cuberoot_J",
            "rate_rule": "end_configuration_displacement_secant",
            "work_rule": "accepted_force_dot_displacement_increment",
            "material_dissipation": False,
            "vendor_equivalent": False,
        }
