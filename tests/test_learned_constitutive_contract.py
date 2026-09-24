# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    learning,
    materials,
    mesh as agentfem_mesh,
    models,
    steps,
    studies,
)


STATE = constitutive.MaterialStateSchema(
    "elastic_probe",
    (constitutive.MaterialStateVariable("history", shape=(2,)),),
    version="1.0.0",
)
TANGENT = constitutive.MaterialTangentConvention.cauchy_small_strain()
PARAMETERS = (
    learning.MaterialParameterSpec("young", "Pa", minimum=1.0),
    learning.MaterialParameterSpec("poisson", "1", minimum=-0.99, maximum=0.49),
)


def _elastic_matrix(young: float, poisson: float) -> np.ndarray:
    shear = young / (2.0 * (1.0 + poisson))
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = lame
    matrix[:3, :3] += 2.0 * shear * np.eye(3)
    matrix[3:, 3:] = 2.0 * shear * np.eye(3)
    return matrix


@dataclass
class NumPyElasticMaterial:
    name: str = "numpy_elastic_probe"
    state_schema: constitutive.MaterialStateSchema = STATE
    tangent_convention: constitutive.MaterialTangentConvention = TANGENT
    batch_calls: int = 0

    def update(self, point):
        tangent = _elastic_matrix(**point.parameters)
        return constitutive.SmallStrainMaterialPointOutput(
            cauchy_stress=tangent @ point.strain_new,
            consistent_tangent=tangent,
            state_new=point.state_old,
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
            stored_energy_density=0.5 * point.strain_new @ tangent @ point.strain_new,
            diagnostics={"elastic": True},
        )

    def update_batch(self, request):
        self.batch_calls += 1
        tangent = _elastic_matrix(**request.parameters)
        return constitutive.SmallStrainMaterialBatchOutput(
            cauchy_stress=request.strain_new @ tangent.T,
            consistent_tangent=np.broadcast_to(
                tangent, (request.point_count, 6, 6)
            ).copy(),
            state_new=request.state_old,
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
            stored_energy_density=0.5
            * np.einsum("ni,ij,nj->n", request.strain_new, tangent, request.strain_new),
            dissipated_energy_density=np.zeros(request.point_count),
            suggested_time_scale=np.ones(request.point_count),
            applicability_status=("in_domain",) * request.point_count,
            diagnostics={"elastic": np.ones(request.point_count, dtype=bool)},
        )


@dataclass(frozen=True)
class NumPyProvider:
    name: str

    def create(self, specification):
        assert specification.architecture_id == "numpy.elastic.v1"
        return NumPyElasticMaterial()

    def evidence(self, specification):
        return {
            "provider": self.name,
            "runtime": "numpy",
            "artifact": specification.artifact,
        }


def _spec(provider="test.numpy-learned-material"):
    return learning.learned_constitutive(
        provider=provider,
        architecture_id="numpy.elastic.v1",
        artifact="models/elastic-probe",
        revision="immutable-test-revision",
        artifact_sha256="1" * 64,
        tangent_convention=TANGENT,
        parameter_schema=PARAMETERS,
        state_schema=STATE,
        parameters={"young": 190.0e9, "poisson": 0.3},
        model_name="elastic_probe",
        model_version="1.0.0",
        capabilities=("stress", "state", "consistent_tangent", "batch"),
        dataset_id="local/elastic-reference",
        dataset_revision="v1",
    )


def _point(strain=None):
    return constitutive.SmallStrainMaterialPointInput(
        strain_old=np.zeros(6),
        strain_new=np.asarray(
            [1.0e-4, -2.0e-5, 0.0, 3.0e-5, 0.0, 0.0] if strain is None else strain
        ),
        time=0.0,
        time_increment=1.0,
        parameters={"young": 190.0e9, "poisson": 0.3},
        state_old=STATE.initial_state(),
        state_schema=STATE,
    )


def test_spec_roundtrip_and_fingerprint():
    spec = _spec()
    restored = learning.LearnedConstitutiveSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()
    assert spec.to_dict()["schema_version"] == "1.0.0"
    assert restored.fingerprint == spec.fingerprint
    assert restored.parameters == {"young": 190.0e9, "poisson": 0.3}


def test_spec_nested_metadata_is_immutable_and_serializable():
    record = _spec().to_dict()
    record["metadata"] = {"training": {"seeds": [2, 3, 5]}}
    specification = learning.LearnedConstitutiveSpec.from_dict(record)
    with pytest.raises(TypeError):
        specification.metadata["training"]["seeds"] = (7,)
    assert specification.to_dict()["metadata"] == {"training": {"seeds": [2, 3, 5]}}


def test_missing_provider_fails_closed():
    with pytest.raises(
        learning.LearnedConstitutiveProviderError,
        match="AFM-LEARNED-PROVIDER-001.*not active",
    ):
        materials.learned(_spec("test.provider-that-does-not-exist"))


def test_provider_loading_batch_update_and_tangent_check():
    provider = NumPyProvider("test.numpy-learned-material")
    learning.register_learned_constitutive_provider(provider, replace=True)
    material = materials.learned(_spec())
    assert isinstance(material, learning.LearnedConstitutiveMaterial)
    point = _point()
    tangent = constitutive.check_small_strain_material_tangent(
        material, point, tolerance=1.0e-8
    )
    assert tangent.accepted

    request = constitutive.SmallStrainMaterialBatchInput(
        strain_old=np.zeros((3, 6)),
        strain_new=np.stack(
            (point.strain_new, 2 * point.strain_new, -point.strain_new)
        ),
        time=0.0,
        time_increment=1.0,
        parameters=point.parameters,
        state_old=np.zeros((3, STATE.size)),
        state_schema=STATE,
    )
    response = constitutive.update_small_strain_material_batch(material, request)
    assert material.implementation.batch_calls == 1
    assert response.cauchy_stress.shape == (3, 6)
    assert np.all(response.state_new == request.state_old)
    assert response.applicability_status == ("in_domain",) * 3
    evidence = learning.learned_constitutive_evidence(_spec())
    assert evidence["specification_fingerprint"] == _spec().fingerprint
    assert evidence["runtime"]["runtime"] == "numpy"


def test_batch_validation_is_atomic_on_nonfinite_output():
    material = NumPyElasticMaterial()
    original_batch = material.update_batch

    def invalid_batch(request):
        valid = original_batch(request)
        stress = valid.cauchy_stress.copy()
        stress[-1, 0] = np.nan
        return constitutive.SmallStrainMaterialBatchOutput(
            cauchy_stress=stress,
            consistent_tangent=valid.consistent_tangent,
            state_new=valid.state_new,
            tangent_convention=valid.tangent_convention,
            state_schema=valid.state_schema,
            stored_energy_density=valid.stored_energy_density,
            dissipated_energy_density=valid.dissipated_energy_density,
            suggested_time_scale=valid.suggested_time_scale,
            applicability_status=valid.applicability_status,
        )

    material.update_batch = invalid_batch
    request = constitutive.SmallStrainMaterialBatchInput(
        strain_old=np.zeros((2, 6)),
        strain_new=np.ones((2, 6)) * 1.0e-4,
        time=0.0,
        time_increment=1.0,
        parameters={"young": 190.0e9, "poisson": 0.3},
        state_old=np.zeros((2, STATE.size)),
        state_schema=STATE,
    )
    with pytest.raises(ValueError, match="Batch Cauchy stress"):
        constitutive.update_small_strain_material_batch(material, request)


def test_quadrature_driver_preserves_trial_commit_rollback():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
    state = constitutive.MaterialQuadratureState.create(domain, STATE, degree=1)

    class HistoryMaterial(NumPyElasticMaterial):
        def update_batch(self, request):
            response = super().update_batch(request)
            return constitutive.SmallStrainMaterialBatchOutput(
                cauchy_stress=response.cauchy_stress,
                consistent_tangent=response.consistent_tangent,
                state_new=response.state_new + 1.0,
                tangent_convention=response.tangent_convention,
                state_schema=response.state_schema,
                stored_energy_density=response.stored_energy_density,
                dissipated_energy_density=response.dissipated_energy_density,
                suggested_time_scale=response.suggested_time_scale,
                applicability_status=response.applicability_status,
                diagnostics=response.diagnostics,
            )

    material = HistoryMaterial()
    count = len(state.reference_field.values)
    original = state.committed_state_vectors()
    result = constitutive.update_small_strain_quadrature_state(
        material,
        state,
        strain_old=np.zeros((count, 6)),
        strain_new=np.ones((count, 6)) * 1.0e-5,
        time=0.0,
        time_increment=1.0,
        parameters={"young": 190.0e9, "poisson": 0.3},
    )
    assert result.committed is False
    np.testing.assert_allclose(state.committed_state_vectors(), original)
    np.testing.assert_allclose(state.trial_state_vectors(), original + 1.0)
    state.rollback()
    np.testing.assert_allclose(state.trial_state_vectors(), original)

    result = constitutive.update_small_strain_quadrature_state(
        material,
        state,
        strain_old=np.zeros((count, 6)),
        strain_new=np.ones((count, 6)) * 1.0e-5,
        time=0.0,
        time_increment=1.0,
        parameters={"young": 190.0e9, "poisson": 0.3},
        commit=True,
    )
    assert result.committed is True
    np.testing.assert_allclose(state.committed_state_vectors(), original + 1.0)


def test_learned_material_runs_through_standard_global_step():
    provider = NumPyProvider("test.numpy-learned-material")
    learning.register_learned_constitutive_provider(provider, replace=True)
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 1, 1)
    study = studies.static_solid(dimension=3, nonlinear=True, name="learned_bar")
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    material = model.material(materials.learned(_spec()))
    left = agentfem_mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = agentfem_mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((1.0e6, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(1),
        progress=False,
    )
    result = step.solve_result()
    assert result.status == "completed"
    assert step.last_solve_info.converged
    assert step.last_solve_info.as_dict()["kind"] == "small_strain_material_load_path"
    assert np.max(np.abs(displacement.value.x.array)) > 0.0
    assert (
        result.metadata["material"]["specification_fingerprint"] == _spec().fingerprint
    )
    assert (
        result.scientific_inputs["learned_constitutive"].fingerprint
        == _spec().fingerprint
    )


def test_learned_material_checkpoint_restart_is_trajectory_equivalent(tmp_path):
    provider = NumPyProvider("test.numpy-learned-checkpoint")
    learning.register_learned_constitutive_provider(provider, replace=True)
    specification = _spec(provider=provider.name)
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 1, 1)

    def build():
        model = models.create(
            study=studies.static_solid(
                dimension=3, nonlinear=True, name="learned_restart"
            ),
            mesh=domain,
        )
        displacement = model.field(fields.displacement(domain))
        material = model.material(materials.learned(specification))
        left = agentfem_mesh.boundary(
            domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
        )
        right = agentfem_mesh.boundary(
            domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
        )
        model.fix(displacement, on=left, value=0.0)
        model.traction((1.0e6, 0.0, 0.0), on=right)
        step = model.step(
            target=displacement,
            material=material,
            incrementation=steps.fixed(2),
            progress=False,
        )
        return displacement, step

    uninterrupted_displacement, uninterrupted = build()
    uninterrupted.solve(until=0.5)
    shared_checkpoint = Path(
        domain.comm.bcast(
            str(tmp_path / "learned") if domain.comm.rank == 0 else None,
            root=0,
        )
    )
    checkpoint = uninterrupted.save_checkpoint(shared_checkpoint)
    uninterrupted.solve()
    expected_u = uninterrupted_displacement.value.x.array.copy()
    expected_state = uninterrupted.state.state.committed_state_vectors().copy()

    restarted_displacement, restarted = build()
    restarted.load_checkpoint(checkpoint)
    restarted.solve()
    np.testing.assert_allclose(
        restarted_displacement.value.x.array, expected_u, rtol=1.0e-11, atol=1.0e-13
    )
    np.testing.assert_allclose(
        restarted.state.state.committed_state_vectors(),
        expected_state,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
