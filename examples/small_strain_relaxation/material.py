"""Educational standard linear solid: local update only, no global solver."""

import numpy as np
from agentfem import constitutive as c


class StandardLinearSolid:
    name = "educational_standard_linear_solid"
    rate_independent = False
    stress_free_initial_state = True

    def __init__(
        self,
        bulk=1000.0,
        equilibrium_shear=100.0,
        branch_shear=200.0,
        relaxation_time=1.0,
    ):
        self.parameters = dict(
            bulk=bulk,
            equilibrium_shear=equilibrium_shear,
            branch_shear=branch_shear,
            relaxation_time=relaxation_time,
        )
        if any(
            not np.isfinite(value) or value <= 0 for value in self.parameters.values()
        ):
            raise ValueError(
                "All moduli and relaxation time must be finite and positive."
            )
        self.parameter_schema = c.MaterialParameterSchema(
            "example.standard_linear_solid.parameters",
            tuple(
                c.MaterialParameter(
                    name, unit="s" if name == "relaxation_time" else "Pa", lower=0.0
                )
                for name in self.parameters
            ),
            version="1.0",
        )
        self.state_schema = c.MaterialStateSchema(
            "example.standard_linear_solid.state",
            (
                c.MaterialStateVariable(
                    "EV",
                    shape=(3, 3),
                    unit="1",
                    description="Deviatoric viscous strain.",
                ),
            ),
            version="1.0",
        )
        self.tangent_convention = c.small_strain_tangent_convention()

    def summary(self):
        return {
            "kind": "small_strain_user_material",
            "name": self.name,
            "parameters": dict(self.parameters),
            "state_schema": self.state_schema.summary(),
            "integration": "backward_euler",
            "parameter_source": "synthetic educational values",
        }

    def update(self, point):
        p = point.parameters
        ratio = point.time_increment / p["relaxation_time"]
        strain = point.strain_new
        dev = strain - np.trace(strain) * np.eye(3) / 3
        old_viscous = point.state_old.reshape(3, 3)
        viscous = (old_viscous + ratio * dev) / (1 + ratio)
        stress = (
            p["bulk"] * np.trace(strain) * np.eye(3)
            + 2 * p["equilibrium_shear"] * dev
            + 2 * p["branch_shear"] * (dev - viscous)
        )
        shear = p["equilibrium_shear"] + p["branch_shear"] / (1 + ratio)
        tangent = np.zeros((6, 6))
        tangent[:3, :3] = p["bulk"] - 2 * shear / 3
        tangent[np.arange(6), np.arange(6)] += 2 * shear
        return c.SmallStrainMaterialPointOutput(
            cauchy_stress=stress,
            consistent_tangent=tangent,
            state_new=viscous.ravel(),
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
        )
