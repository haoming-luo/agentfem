import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem.constitutive._explicit_material_driver import ExplicitQuadratureResponse
from agentfem.constitutive.user_material import (
    MaterialStateSchema,
    MaterialStateVariable,
)
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual


@pytest.mark.parametrize("rate", [-0.2, 0.0, 0.2])
def test_hex_viscosity_force_power_and_velocity_derivative(rate):
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.hexahedron
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u, v = fem.Function(space), fem.Function(space)
    state = MaterialStateSchema("test", (MaterialStateVariable("h"),))
    response = ExplicitQuadratureResponse.create(domain, state)
    operator = FiniteUniformHexResidual(
        u, response, density=2.0, hourglass_modulus=1.0, hourglass_scale=0.1
    )
    u.interpolate(lambda x: 0.1 * x)
    v.interpolate(lambda x: rate * 1.1 * x)
    options = dict(density=2 / 1.1**3, wave_speed=3.0, characteristic_length=1.1)
    vector, result = operator.assemble_bulk_viscosity(v, **options)
    try:
        theta = 3 * rate
        rho = options["density"]
        q = rho * 0.06 * 3 * 1.1 * theta - rho * (1.2 * 1.1 * min(theta, 0)) ** 2
        np.testing.assert_allclose(result.isotropic_stress, q, atol=1e-14)
        np.testing.assert_allclose(
            result.artificial_power, 1.1**3 * q * theta, atol=1e-14
        )
        assert np.min(result.artificial_power) >= 0
        assert vector.array @ v.x.array == pytest.approx(
            result.artificial_power.sum(), abs=1e-14
        )
        np.testing.assert_allclose(
            vector.array.reshape(-1, 3).sum(axis=0), 0, atol=1e-14
        )
        xyz = space.tabulate_dof_coordinates() + u.x.array.reshape(-1, 3)
        np.testing.assert_allclose(
            np.cross(xyz, vector.array.reshape(-1, 3)).sum(axis=0), 0, atol=1e-14
        )
    finally:
        vector.destroy()
    # All 24 velocity columns: independent numerical derivative of assembled force.
    base = v.x.array.copy()
    derivative = np.empty((len(base), len(base)))
    for j in range(len(base)):
        values = []
        for sign in (-1, 1):
            v.x.array[:] = base
            v.x.array[j] += sign * 1e-7
            force, _ = operator.assemble_bulk_viscosity(v, **options)
            values.append(force.array.copy())
            force.destroy()
        derivative[:, j] = (values[1] - values[0]) / 2e-7
    scaled = derivative / np.sqrt(
        operator.mass_diagonal[:, None] * operator.mass_diagonal[None, :]
    )
    assert np.linalg.eigvalsh((scaled + scaled.T) / 2)[-1] == pytest.approx(
        result.mass_scaled_damping_bound[0], rel=2e-6
    )


def test_viscosity_invalid_coefficients_and_rigid_motion():
    from agentfem.elements._bulk_viscosity import bulk_viscosity_response
    from agentfem.elements._finite_uniform_hex import FiniteUniformHexBatch
    import basix

    x = basix.geometry(basix.CellType.hexahedron)[None]
    cells = FiniteUniformHexBatch(
        x, density=1.0, hourglass_modulus=1.0, hourglass_scale=0.1
    )
    options = dict(density=1.0, wave_speed=1.0, characteristic_length=1.0)
    v = np.cross([0.0, 0.0, 2.0], x) + [1, 2, 3]
    response = bulk_viscosity_response(cells, np.zeros_like(x), v, **options)
    np.testing.assert_allclose(response.internal_force, 0, atol=1e-14)
    for name in options:
        with pytest.raises(ValueError):
            bulk_viscosity_response(
                cells, np.zeros_like(x), v, **{**options, name: 0.0}
            )


def test_viscosity_composes_with_material_trial_and_downstream_rollback(monkeypatch):
    from agentfem.constitutive._explicit_material import ExplicitMaterialBatchOutput

    class Material:
        state_schema = MaterialStateSchema("zero", (MaterialStateVariable("h"),))

        def update_explicit_array_batch(self, request):
            n = request.point_count
            return ExplicitMaterialBatchOutput(
                np.tile(np.diag([.2, .3, .4]), (n, 1, 1)),
                request.state_old + 1,
                self.state_schema,
                np.zeros(n),
            )

    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 2, 1, 1, cell_type=mesh.CellType.hexahedron
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u, v = fem.Function(space), fem.Function(space)
    material = Material()
    response = ExplicitQuadratureResponse.create(domain, material.state_schema)
    operator = FiniteUniformHexResidual(
        u, response, density=1.0, hourglass_modulus=1.0, hourglass_scale=0.1
    )
    v.interpolate(lambda x: -0.1 * x)
    options = dict(
        deformation_gradient_old=np.tile(np.eye(3), (2, 1, 1)),
        time=0.0,
        time_increment=0.01,
        velocity=v,
        bulk_viscosity=dict(density=1.0, wave_speed=2.0, characteristic_length=1.0),
    )
    original_scatter = operator._scatter
    calls = []

    def counted_scatter(values, **kwargs):
        calls.append(1)
        return original_scatter(values, **kwargs)

    monkeypatch.setattr(operator, "_scatter", counted_scatter)
    with pytest.raises(ValueError, match="downstream"):
        with operator.explicit_trial_evaluation(material, **options) as (force, trial):
            assert len(calls) == 1
            assert trial.element_response.physical_energy is None
            assert trial.material_response.internal_work_density_increment[0] == 0
            elastic = original_scatter(trial.element_response.internal_force)
            damping = original_scatter(trial.bulk_viscosity_response.internal_force)
            try:
                np.testing.assert_allclose(force.array, elastic.array + damping.array,
                                           atol=1e-14, rtol=1e-14)
                assert damping.array @ v.x.array == pytest.approx(
                    trial.bulk_viscosity_response.artificial_power.sum()
                )
            finally:
                elastic.destroy()
                damping.destroy()
            raise ValueError("downstream")
    np.testing.assert_array_equal(response.state.trial_state_vectors(), 0.0)
    np.testing.assert_array_equal(response.state.committed_state_vectors(), 0.0)
    with pytest.raises(ValueError, match="both velocity"):
        with operator.explicit_trial_evaluation(
            material, **{**options, "velocity": None}
        ):
            pass
