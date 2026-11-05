"""Research-only J2 extension of the published scalar P92 reference.

No fitted multiaxial material, temperature interpolation, or life prediction.
The structural solver remains AgentFEM's ordinary small-strain consumer.
"""

from dataclasses import asdict, replace

import numpy as np
from agentfem import constitutive as c

from tools.p92_published_recovery_reference import published_parameters
from tools.p92_tensor_normalization import implicit_tensor_update, tensor


class P92TensorResearchMaterial:
    name = "p92_isothermal_tensor_research_reference"
    rate_independent = False
    stress_free_initial_state = True

    def __init__(self, *, temperature_C, poisson):
        self.reference = published_parameters(temperature_C)
        if (
            isinstance(poisson, (bool, np.bool_))
            or np.iscomplexobj(poisson)
            or np.asarray(poisson).ndim != 0
            or not np.isfinite(poisson)
            or not -1 < poisson < 0.5
        ):
            raise ValueError("Supply a finite Poisson ratio in (-1, 0.5).")
        self.parameters = {"temperature_C": temperature_C, "poisson": poisson}
        self.parameter_schema = c.MaterialParameterSchema(
            "research.p92.tensor.parameters",
            (
                c.MaterialParameter("temperature_C", unit="degC"),
                c.MaterialParameter("poisson", unit="1"),
            ),
            version="1.0",
        )
        self.state_schema = c.MaterialStateSchema(
            "research.p92.tensor.state",
            (
                c.MaterialStateVariable("SIGMA_M", shape=(6,), unit="MPa"),
                c.MaterialStateVariable("X1_DEV", shape=(5,), unit="MPa"),
                c.MaterialStateVariable("X2_DEV", shape=(5,), unit="MPa"),
                c.MaterialStateVariable("R", unit="MPa"),
                c.MaterialStateVariable("P", unit="1"),
            ),
            version="1.0",
        )
        self.tangent_convention = replace(
            c.small_strain_tangent_convention(), symmetric=False
        )

    def summary(self):
        return {
            "kind": "small_strain_user_material",
            "name": self.name,
            "parameters": dict(self.parameters),
            "published_parameters": asdict(self.reference),
            "source": "https://doi.org/10.3390/met10030307",
            "state_schema": self.state_schema.summary(),
            "scope": "research-only isotropic J2 extension; supplied Poisson ratio",
            "integration": "coupled_backward_euler",
            "thermal_input": "optional fixed kelvin matching the selected published temperature; zero increment",
        }

    def update(self, point):
        if point.temperature is not None or point.temperature_increment is not None:
            expected = self.parameters["temperature_C"] + 273.15
            if (
                point.temperature is None
                or point.temperature_increment is None
                or not np.isclose(point.temperature, expected, rtol=1e-12, atol=0.0)
                or point.temperature_increment != 0.0
            ):
                raise ValueError(
                    "This research reference requires its fixed kelvin temperature and zero temperature increment."
                )
        if point.field_variables:
            raise ValueError("External fields are not supported by this reference.")
        if dict(point.parameters) != self.parameters:
            raise ValueError("Research material parameters changed after construction.")
        state, tangent = implicit_tensor_update(
            point.state_old,
            point.strain_new - point.strain_old,
            point.time_increment,
            self.reference,
            poisson=self.parameters["poisson"],
        )
        # Mandel order xx yy zz yz xz xy -> tensor Voigt xx yy zz xy yz xz.
        order = [0, 1, 2, 5, 3, 4]
        scale = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        tangent = tangent[np.ix_(order, order)] * scale[None, :] / scale[:, None]
        return c.SmallStrainMaterialPointOutput(
            cauchy_stress=tensor(state[:6]),
            consistent_tangent=tangent,
            state_new=state,
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
        )
