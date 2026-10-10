# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""A non-native material uses public Step without a new core solver branch."""

import numpy as np

from agentfem import amplitudes, constitutive, elements, fields, models, studies
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchOutput
from test_finite_uniform_hex import _material
from test_finite_hex_step import problem


class ReferenceNeoHookean:
    name = "external_reference_neo_hookean"
    density = 2.0
    tangent_convention = (
        constitutive.MaterialTangentConvention.first_piola_deformation_gradient()
    )
    state_schema = constitutive.MaterialStateSchema(
        name="external_peak_energy",
        variables=(
            constitutive.MaterialStateVariable(name="peak_energy", output_name="PEAKW"),
        ),
    )

    def summary(self):
        return {
            "name": self.name,
            "mu": 30.0,
            "lambda": 40.0,
            "density": self.density,
            "state": self.state_schema.summary(),
        }

    def _response(self, gradient, state_old):
        piola, energy, tangent = _material(gradient)
        stress = (piola @ gradient.swapaxes(1, 2)) / np.linalg.det(gradient)[
            :, None, None
        ]
        return MaterialPointArrayBatchOutput(
            cauchy_stress=stress,
            consistent_tangent=tangent.reshape(-1, 9, 9),
            state_new=np.maximum(state_old, energy[:, None]),
            strain_energy_density=energy,
            dissipation_density_increment=np.zeros(len(gradient)),
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
        )

    def initial_array_response(self, count):
        return self._response(np.tile(np.eye(3), (count, 1, 1)), np.zeros((count, 1)))

    def update(self, point):
        raise AssertionError(
            "The advertised array path must not fall back to per-point Python calls"
        )

    def update_array_batch(self, request):
        return self._response(request.deformation_gradient_new, request.state_old)


def test_external_array_material_uses_ordinary_step_state_fields_and_restart(tmp_path):
    def prepare():
        prototype, _, policy = problem()
        domain = prototype.mesh
        model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
        u = model.field(fields.displacement(domain))
        model.material(ReferenceNeoHookean())
        model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
        model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
        model.fix(
            u,
            on=lambda x: np.isclose(x[0], 1),
            components=0,
            value=amplitudes.Amplitude(
                "external_ramp",
                lambda t: 100 * t * t,
                metadata={"quadratic_coefficient": 100},
            ),
        )
        return model.step(
            target=u,
            element_policy=policy,
            omega_squared_bound=1e8,
            dt=1e-4,
            steps=100,
            progress=False,
        )

    reference = prepare()
    reference.solve_result(field_variables=("S", "PEAKW"))
    partial = prepare()
    partial.run(until_step=39)
    saved = partial.save_checkpoint(tmp_path / "external")
    resumed = prepare()
    resumed.load_checkpoint(saved)
    result = resumed.solve_result(field_variables=("S", "PEAKW"))
    np.testing.assert_array_equal(
        resumed.state.u.value.x.array, reference.state.u.value.x.array
    )
    assert np.max(result.fields["PEAKW"].field.x.array) > 0
    assert resumed.history_records[-1]["material_dissipation"] == 0
    assert resumed.history_records[-1]["relative_energy_balance_error"] < 5e-3
