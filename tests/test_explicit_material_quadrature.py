import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem

from agentfem.constitutive._explicit_material import ExplicitMaterialBatchOutput
from agentfem.constitutive._explicit_material_driver import (
    update_explicit_material_points,
)
from agentfem.constitutive.quadrature import MaterialQuadratureState
from agentfem.constitutive.user_material import (
    MaterialStateSchema,
    MaterialStateVariable,
)


class ResponseOnly:
    state_schema = MaterialStateSchema(
        "response_only_test", (MaterialStateVariable("strain", shape=(3, 3)),)
    )

    def __init__(self):
        self.calls = 0
        self.fail_at = None
        self.reduce = False

    def update_explicit_array_batch(self, request):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("injected provider failure")
        change = request.deformation_gradient_new - request.deformation_gradient_old
        strain = (
            request.state_old.reshape(-1, 3, 3)
            + (change + change.transpose(0, 2, 1)) / 2
        )
        stress = 20 * strain
        work = np.einsum("nij,nij->n", stress, change)
        return ExplicitMaterialBatchOutput(
            stress,
            strain.reshape(-1, 9),
            self.state_schema,
            work,
            requested_time_scale=np.full(len(strain), 0.5 if self.reduce else 1.0),
        )


def fixture():
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 2, 1, 1, cell_type=mesh.CellType.hexahedron
    )
    material = ResponseOnly()
    state = MaterialQuadratureState.create(domain, material.state_schema, degree=1)
    old = np.tile(np.eye(3), (len(state.reference_field.values), 1, 1))
    new = old.copy()
    new[:, 0, 0] = 1.02
    return (
        material,
        state,
        dict(
            deformation_gradient_old=old,
            deformation_gradient_new=new,
            time=0.0,
            time_increment=0.1,
        ),
    )


def test_real_quadrature_trial_commit_rollback_and_piola():
    material, state, options = fixture()
    result = update_explicit_material_points(
        material, state, **options, max_batch_points=1
    )
    assert material.calls == len(state.reference_field.values)
    np.testing.assert_array_equal(state.committed_state_vectors(), 0)
    assert np.max(state.trial_state_vectors()) == pytest.approx(0.02)
    np.testing.assert_allclose(result.first_piola_stress[:, 0, 0], 0.4)
    assert result.response.stored_energy_density is None
    state.rollback()
    np.testing.assert_array_equal(state.trial_state_vectors(), 0)
    result = update_explicit_material_points(material, state, **options)
    state.commit()
    np.testing.assert_array_equal(
        state.committed_state_vectors(), result.response.state_new
    )


@pytest.mark.parametrize("failure", ["provider", "cutback", "assignment"])
def test_failure_keeps_all_real_quadrature_state_unchanged(monkeypatch, failure):
    material, state, options = fixture()
    before = state.committed_state_vectors().copy()
    trial = state.trial_state_vectors().copy()
    if failure == "provider":
        material.fail_at = 2
    elif failure == "cutback":
        material.reduce = True
    else:
        original = state.assign_trial_state_vectors

        def fail(values):
            original(values)
            raise ValueError("injected late assignment failure")

        monkeypatch.setattr(state, "assign_trial_state_vectors", fail)
    with pytest.raises(ValueError):
        update_explicit_material_points(material, state, **options, max_batch_points=1)
    np.testing.assert_array_equal(state.committed_state_vectors(), before)
    np.testing.assert_array_equal(state.trial_state_vectors(), trial)


@pytest.mark.parametrize("nx", [1, 2])
@pytest.mark.parametrize("chunk_size", [1, 4])
@pytest.mark.parametrize("material_batch", [None, 1, 4])
def test_real_hex_force_assembly_without_invented_energy_or_tangent(nx, chunk_size, material_batch):
    from agentfem.constitutive._explicit_material_driver import (
        ExplicitQuadratureResponse,
    )
    from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual

    material = ResponseOnly()
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, nx, 1, 1, cell_type=mesh.CellType.hexahedron
    )
    displacement = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    response = ExplicitQuadratureResponse.create(domain, material.state_schema)
    residual = FiniteUniformHexResidual(
        displacement, response, density=1.0, hourglass_modulus=10.0, hourglass_scale=0.1,
        chunk_size=chunk_size,
        material_batch_points=material_batch,
    )
    displacement.interpolate(lambda x: np.vstack((0.02 * x[0], 0 * x[1], 0 * x[2])))
    old = np.tile(np.eye(3), (nx, 1, 1))
    with residual.explicit_trial_evaluation(
        material, deformation_gradient_old=old, time=0.0, time_increment=0.1
    ) as (vector, trial):
        assert trial.element_response.physical_energy is None
        batch = chunk_size if material_batch is None else material_batch
        assert material.calls == (nx + batch - 1) // batch
        assert not hasattr(response, "tangent")
        forces = vector.array.reshape(-1, 3)
        xyz = displacement.function_space.tabulate_dof_coordinates()
        np.testing.assert_allclose(forces.sum(axis=0), 0, atol=1.0e-14)
        assert forces[xyz[:, 0] > 0.99, 0].sum() == pytest.approx(0.4)
        current = xyz + displacement.x.array.reshape(-1, 3)
        np.testing.assert_allclose(
            np.cross(current, forces).sum(axis=0), 0, atol=1.0e-14
        )
        with pytest.raises(RuntimeError):
            residual.tangent_action(np.zeros_like(displacement.x.array))
    vector.destroy()
    response.commit()
    state_before = response.state.committed_state_vectors().copy()
    fields_before = {
        name: value.x.array.copy() for name, value in response.trial_fields().items()
    }
    displacement.interpolate(lambda x: np.vstack((0.04 * x[0], 0 * x[1], 0 * x[2])))
    with pytest.raises(ValueError, match="downstream"):
        with residual.explicit_trial_evaluation(
            material,
            deformation_gradient_old=trial.deformation_gradient,
            time=0.1,
            time_increment=0.1,
        ):
            raise ValueError("downstream acceptance failure")
    np.testing.assert_array_equal(
        response.state.committed_state_vectors(), state_before
    )
    for name, value in response.trial_fields().items():
        np.testing.assert_array_equal(value.x.array, fields_before[name])


@pytest.mark.parametrize("when", ["provider", "consumer"])
def test_response_trial_rejects_changed_nodal_configuration(monkeypatch, when):
    from agentfem.constitutive._explicit_material_driver import (
        ExplicitQuadratureResponse,
    )
    from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual

    material = ResponseOnly()
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=mesh.CellType.hexahedron
    )
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    response = ExplicitQuadratureResponse.create(domain, material.state_schema)
    residual = FiniteUniformHexResidual(
        u, response, density=1, hourglass_modulus=1, hourglass_scale=0.1
    )
    u.interpolate(lambda x: 0.01 * x)
    before = u.x.array.copy()
    update = material.update_explicit_array_batch

    if when == "provider":

        def corrupt(request):
            output = update(request)
            u.x.array[:] += 0.2
            return output

        monkeypatch.setattr(material, "update_explicit_array_batch", corrupt)
    with pytest.raises(RuntimeError, match="configuration"):
        with residual.explicit_trial_evaluation(
            material,
            deformation_gradient_old=np.eye(3)[None],
            time=0,
            time_increment=0.01,
        ):
            if when == "consumer":
                u.x.array[:] += 0.3
    np.testing.assert_array_equal(u.x.array, before)
    np.testing.assert_array_equal(response.state.committed_state_vectors(), 0)
    np.testing.assert_array_equal(response.state.trial_state_vectors(), 0)


def test_joint_response_commit_rollback_and_commit_failure(monkeypatch):
    from agentfem.constitutive._explicit_material_driver import (
        ExplicitQuadratureResponse,
    )

    material, state, options = fixture()
    response = ExplicitQuadratureResponse.create(state.domain, material.state_schema)
    response.update(material, **options)
    response.commit()
    accepted = {
        name: value.x.array.copy() for name, value in response.trial_fields().items()
    }
    options["deformation_gradient_old"] = options["deformation_gradient_new"].copy()
    options["deformation_gradient_new"] = options["deformation_gradient_old"].copy()
    options["deformation_gradient_new"][:, 0, 0] = 1.04
    response.update(material, **options)
    response.rollback()
    for name, value in response.trial_fields().items():
        np.testing.assert_array_equal(value.x.array, accepted[name])
    response.update(material, **options)
    before = response.state.committed_state_vectors().copy()
    saved_response = response.response_transaction.snapshot()
    original = response.response_transaction.commit

    def fail_after_commit():
        original()
        raise ValueError("injected joint commit failure")

    monkeypatch.setattr(response.response_transaction, "commit", fail_after_commit)
    with pytest.raises(ValueError, match="joint commit"):
        response.commit()
    np.testing.assert_array_equal(response.state.committed_state_vectors(), before)
    for name, value in response.response_transaction.snapshot().items():
        np.testing.assert_array_equal(value, saved_response[name])
