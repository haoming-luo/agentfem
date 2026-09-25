from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from agentfem import constitutive, learning, materials, mesh, results
from agentfem.learning import constitutive as learned_contract


@pytest.fixture(autouse=True)
def _isolated_provider_registry():
    learned_contract._PROVIDERS.clear()
    yield
    learned_contract._PROVIDERS.clear()


def _schemas():
    state = constitutive.MaterialStateSchema(
        "test.learned_state",
        (
            constitutive.MaterialStateVariable(
                "history", unit="1", description="Accumulated test history."
            ),
        ),
        version="1.0",
    )
    parameters = constitutive.MaterialParameterSchema(
        "test.isotropic_parameters",
        (
            constitutive.MaterialParameter("young", unit="Pa", lower=0.0),
            constitutive.MaterialParameter(
                "poisson", unit="1", lower=-0.999, upper=0.499
            ),
        ),
        version="1.0",
    )
    return state, parameters


class _LinearMaterial:
    name = "test_linear_material"

    def __init__(self, spec, *, batch=True):
        self.spec = spec
        self.state_schema = spec.state_schema
        self.parameter_schema = spec.parameter_schema
        self.tangent_convention = spec.tangent_convention
        self.parameters = spec.parameters
        self.batch_calls = 0
        if not batch:
            self.update_batch = None

    def _tangent(self):
        young = self.parameters["young"]
        poisson = self.parameters["poisson"]
        shear = young / (2.0 * (1.0 + poisson))
        lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        tangent = np.zeros((6, 6))
        tangent[:3, :3] = lame
        tangent[np.arange(3), np.arange(3)] += 2.0 * shear
        tangent[np.arange(3, 6), np.arange(3, 6)] = 2.0 * shear
        return tangent

    def _stress(self, strain):
        young = self.parameters["young"]
        poisson = self.parameters["poisson"]
        shear = young / (2.0 * (1.0 + poisson))
        lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        return lame * np.trace(strain) * np.eye(3) + 2.0 * shear * strain

    def update(self, point):
        stress = self._stress(point.strain_new)
        increment = np.linalg.norm(point.strain_new - point.strain_old)
        return constitutive.SmallStrainMaterialPointOutput(
            cauchy_stress=stress,
            consistent_tangent=self._tangent(),
            state_new=point.state_old + increment,
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
            stored_energy_density=0.5 * np.sum(stress * point.strain_new),
            stored_energy_density_components={
                "ELASTIC": 0.5 * np.sum(stress * point.strain_new)
            },
            dissipation_density_increment=0.0,
            diagnostics={"yield_residual": 0.0},
        )

    def update_batch(self, request):
        self.batch_calls += 1
        stress = np.asarray([self._stress(value) for value in request.strain_new])
        increments = np.linalg.norm(
            request.strain_new - request.strain_old, axis=(1, 2)
        )
        energy = 0.5 * np.sum(stress * request.strain_new, axis=(1, 2))
        return constitutive.SmallStrainMaterialPointBatchOutput(
            cauchy_stress=stress,
            consistent_tangent=np.broadcast_to(
                self._tangent(), (request.point_count, 6, 6)
            ),
            state_new=request.state_old + increments[:, None],
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
            stored_energy_density=energy,
            stored_energy_density_components={"ELASTIC": energy},
            dissipation_density_increment=np.zeros(request.point_count),
        )


def _spec(tmp_path, *, batch=True):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"provider-neutral-test-weights")
    state, parameters = _schemas()
    return learning.learned_constitutive(
        provider="test.numpy_constitutive",
        architecture="test.linear.v1",
        artifact="local-cache://test.linear.v1",
        revision="fixed-test-revision",
        artifact_sha256=sha256(artifact.read_bytes()).hexdigest(),
        parameter_schema=parameters,
        parameters={"young": 190.0e9, "poisson": 0.3},
        state_schema=state,
        tangent_convention=constitutive.small_strain_tangent_convention(),
        capabilities=(
            ("stress", "consistent_tangent", "batch", "energy", "diagnostics")
            if batch
            else ("stress", "consistent_tangent", "energy", "diagnostics")
        ),
        batch_update=batch,
        applicability_domain={"maximum_equivalent_strain": 0.05},
        dataset={"id": "test/dataset", "revision": "fixed"},
    ), artifact


def _provider(*, batch=True):
    capabilities = (
        ("stress", "consistent_tangent", "batch", "energy", "diagnostics")
        if batch
        else ("stress", "consistent_tangent", "energy", "diagnostics")
    )
    return learning.LearnedConstitutiveProvider(
        name="test.numpy_constitutive",
        version="1.0",
        architectures=("test.linear.v1",),
        capabilities=capabilities,
        factory=lambda spec: _LinearMaterial(spec, batch=batch),
    )


def _batch_request(material, count=4):
    strain = np.zeros((count, 3, 3))
    strain[:, 0, 0] = np.linspace(0.0, 0.004, count)
    return constitutive.SmallStrainMaterialPointBatchInput(
        strain_old=np.zeros_like(strain),
        strain_new=strain,
        time=1.0,
        time_increment=1.0,
        parameters=material.specification.parameters,
        state_old=np.zeros((count, material.state_schema.size)),
        state_schema=material.state_schema,
        parameter_schema=material.parameter_schema,
    )


def test_spec_is_portable_fingerprinted_and_verifies_prepared_artifact(tmp_path):
    spec, artifact = _spec(tmp_path)

    assert spec.fingerprint.startswith("sha256:")
    assert len(spec.fingerprint.removeprefix("sha256:")) == 64
    assert spec.verify_artifact(artifact) == artifact.resolve()
    assert spec.summary()["kinematics"] == "small_strain"
    assert spec.summary()["dataset"]["revision"] == "fixed"
    assert spec.parameter_schema.fingerprint.startswith("sha256:")

    artifact.write_bytes(b"changed")
    with pytest.raises(learning.LearnedConstitutiveProviderError, match="CHECKSUM"):
        spec.verify_artifact(artifact)


def test_spec_json_round_trip_and_tamper_rejection(tmp_path):
    spec, _ = _spec(tmp_path)
    manifest = spec.write(tmp_path / "model-spec.json")

    restored = learning.LearnedConstitutiveSpec.read(manifest)
    assert restored == spec
    assert restored.fingerprint == spec.fingerprint

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["specification"]["revision"] = "altered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        learning.LearnedConstitutiveProviderError,
        match="SPEC-FINGERPRINT",
    ):
        learning.LearnedConstitutiveSpec.read(manifest)


def test_batch_energy_component_names_cannot_collide_after_normalization():
    state, _ = _schemas()
    with pytest.raises(ValueError, match="unique after normalization"):
        constitutive.SmallStrainMaterialPointBatchOutput(
            cauchy_stress=np.zeros((1, 3, 3)),
            consistent_tangent=np.zeros((1, 6, 6)),
            state_new=np.zeros((1, state.size)),
            state_schema=state,
            stored_energy_density=np.zeros(1),
            stored_energy_density_components={
                "elastic": np.zeros(1),
                "ELASTIC": np.zeros(1),
            },
        )


def test_spec_requires_stress_and_consistent_tangent_capabilities(tmp_path):
    spec, _ = _spec(tmp_path)
    with pytest.raises(ValueError, match="consistent_tangent"):
        replace(spec, capabilities=("stress", "batch"))


@pytest.mark.parametrize("shear_convention", ("tensor", "engineering"))
def test_declared_matrix_tangent_expands_without_shear_ambiguity(shear_convention):
    matrix = np.arange(36, dtype=float).reshape((6, 6)) / 10.0
    strain = np.array(
        (
            (0.010, 0.004, -0.002),
            (0.004, -0.006, 0.003),
            (-0.002, 0.003, 0.008),
        )
    )
    convention = constitutive.small_strain_tangent_convention(
        shear_convention=shear_convention
    )
    tensor = constitutive.small_strain_matrix_to_tensor(matrix, convention)
    actual = np.einsum("ijkl,kl->ij", tensor, strain)
    strain_vector = np.array(
        (
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            strain[0, 1],
            strain[1, 2],
            strain[0, 2],
        )
    )
    if shear_convention == "engineering":
        strain_vector[3:] *= 2.0
    expected_vector = matrix @ strain_vector
    actual_vector = actual[
        (0, 1, 2, 0, 1, 0),
        (0, 1, 2, 1, 2, 2),
    ]

    np.testing.assert_allclose(actual_vector, expected_vector)


def test_missing_provider_fails_with_addressable_error(tmp_path):
    spec, _ = _spec(tmp_path)
    with pytest.raises(
        learning.LearnedConstitutiveProviderError, match="PROVIDER-MISSING"
    ):
        materials.learned(spec)


def test_registered_numpy_provider_uses_one_batch_call(tmp_path):
    spec, _ = _spec(tmp_path)
    learning.register_learned_constitutive_provider(_provider())
    material = materials.learned(spec)
    request = _batch_request(material)

    response = constitutive.validated_small_strain_batch_update(material, request)

    assert material.implementation.batch_calls == 1
    assert response.point_count == request.point_count
    assert response.summary()["applicability_counts"]["in_domain"] == 4
    assert response.stored_energy_density_components.keys() == {"ELASTIC"}


@pytest.mark.parametrize("mismatch", ("state", "tangent"))
def test_provider_contract_drift_is_rejected_before_execution(tmp_path, mismatch):
    spec, _ = _spec(tmp_path)

    def _factory(selected):
        implementation = _LinearMaterial(selected)
        if mismatch == "state":
            implementation.state_schema = constitutive.MaterialStateSchema(
                "test.wrong_state",
                (),
                version="1.0",
            )
        else:
            implementation.tangent_convention = (
                constitutive.small_strain_tangent_convention(
                    shear_convention="engineering"
                )
            )
        return implementation

    learning.register_learned_constitutive_provider(
        learning.LearnedConstitutiveProvider(
            name="test.numpy_constitutive",
            version="1.0",
            architectures=("test.linear.v1",),
            capabilities=spec.capabilities,
            factory=_factory,
        )
    )
    with pytest.raises(
        learning.LearnedConstitutiveProviderError,
        match="STATE-SCHEMA" if mismatch == "state" else "TANGENT",
    ):
        materials.learned(spec)


def test_empty_batch_diagnostics_do_not_allocate_one_mapping_per_point(tmp_path):
    spec, _ = _spec(tmp_path)
    learning.register_learned_constitutive_provider(_provider())
    material = materials.learned(spec)
    response = material.implementation.update_batch(_batch_request(material, 128))

    assert response.diagnostics == ()


def test_scalar_fallback_and_tangent_check_hold_old_state_fixed(tmp_path):
    spec, _ = _spec(tmp_path, batch=False)
    learning.register_learned_constitutive_provider(_provider(batch=False))
    material = materials.learned(spec)
    request = _batch_request(material, count=3)

    response = constitutive.validated_small_strain_batch_update(material, request)
    point = request.point(2)
    check = constitutive.check_small_strain_material_tangent(
        material, point, tolerance=1.0e-8
    )

    assert response.point_count == 3
    assert check.accepted
    assert check.relative_error < 1.0e-9


def test_scalar_fallback_matches_provider_batch_response(tmp_path):
    batch_spec, _ = _spec(tmp_path, batch=True)
    batch_implementation = _LinearMaterial(batch_spec, batch=True)
    strain = np.zeros((5, 3, 3))
    strain[:, 0, 0] = np.linspace(0.0, 0.005, len(strain))
    request = constitutive.SmallStrainMaterialPointBatchInput(
        strain_old=np.zeros_like(strain),
        strain_new=strain,
        time=1.0,
        time_increment=1.0,
        parameters=batch_spec.parameters,
        state_old=np.zeros((len(strain), batch_spec.state_schema.size)),
        state_schema=batch_spec.state_schema,
        parameter_schema=batch_spec.parameter_schema,
    )
    batch = constitutive.validated_small_strain_batch_update(
        batch_implementation,
        request,
    )
    scalar_implementation = _LinearMaterial(batch_spec, batch=False)
    scalar = constitutive.validated_small_strain_batch_update(
        scalar_implementation,
        request,
    )

    np.testing.assert_allclose(scalar.cauchy_stress, batch.cauchy_stress)
    np.testing.assert_allclose(scalar.consistent_tangent, batch.consistent_tangent)
    np.testing.assert_allclose(scalar.state_new, batch.state_new)


def test_out_of_domain_response_fails_closed(tmp_path):
    spec, _ = _spec(tmp_path)

    class _OutOfDomain(_LinearMaterial):
        def update_batch(self, request):
            baseline = super().update_batch(request)
            return constitutive.SmallStrainMaterialPointBatchOutput(
                cauchy_stress=baseline.cauchy_stress,
                consistent_tangent=baseline.consistent_tangent,
                state_new=baseline.state_new,
                state_schema=baseline.state_schema,
                tangent_convention=baseline.tangent_convention,
                stored_energy_density=baseline.stored_energy_density,
                stored_energy_density_components=(
                    baseline.stored_energy_density_components
                ),
                dissipation_density_increment=(baseline.dissipation_density_increment),
                applicability=("in_domain", "out_of_domain", "in_domain", "in_domain"),
                diagnostics=(
                    {},
                    {"message": "strain exceeds declared domain"},
                    {},
                    {},
                ),
            )

    learning.register_learned_constitutive_provider(
        learning.LearnedConstitutiveProvider(
            name="test.numpy_constitutive",
            version="1.0",
            architectures=("test.linear.v1",),
            capabilities=spec.capabilities,
            factory=lambda selected: _OutOfDomain(selected),
        )
    )
    material = materials.learned(spec)
    with pytest.raises(constitutive.MaterialApplicabilityError, match="OUT-OF-DOMAIN"):
        constitutive.validated_small_strain_batch_update(
            material, _batch_request(material)
        )


def test_result_evidence_records_identity_without_executable_object(tmp_path):
    spec, _ = _spec(tmp_path)
    learning.register_learned_constitutive_provider(_provider())
    material = materials.learned(spec)
    result = results.SimulationResult("learned_material_test")

    learning.record_learned_constitutive_evidence(
        result,
        material,
        runtime={
            "framework": "numpy-test",
            "framework_version": np.__version__,
            "dtype": "float64",
            "device": "cpu",
            "tangent_generation": "analytic_consistent",
            "load_seconds": 0.01,
            "inference_seconds": 0.02,
        },
        diagnostics={
            "maximum_yield_residual": 0.0,
            "out_of_domain_points": 0,
            "cutbacks": 0,
        },
    )

    evidence = result.metadata["learned_constitutive"]
    assert evidence["specification"]["artifact_sha256"] == spec.artifact_sha256
    assert evidence["runtime"]["device"] == "cpu"
    assert "implementation" not in json_safe(evidence)


def test_quadrature_response_preserves_trial_commit_and_failure_rollback(
    tmp_path, monkeypatch
):
    spec, _ = _spec(tmp_path)
    learning.register_learned_constitutive_provider(_provider())
    material = materials.learned(spec)
    domain = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (1, 1), cell_type="triangle")
    response = constitutive.SmallStrainMaterialQuadratureResponse.create(
        domain,
        spec.state_schema,
        degree=2,
        stored_energy=True,
        dissipation=True,
        stored_energy_component_names=("ELASTIC",),
    )
    committed = response.state.committed_state_vectors().copy()
    trial = response.update(
        material,
        strain_old=np.zeros((3, 3)),
        strain_new=np.diag((0.002, 0.0, 0.0)),
        time=1.0,
        time_increment=1.0,
        commit=False,
    )

    np.testing.assert_allclose(response.state.committed_state_vectors(), committed)
    np.testing.assert_allclose(response.state.trial_state_vectors(), trial.state_new)
    response.commit()
    accepted = response.state.committed_state_vectors().copy()
    assert np.all(accepted > committed)

    class _Rejecting(_LinearMaterial):
        def update_batch(self, request):
            baseline = super().update_batch(request)
            return constitutive.SmallStrainMaterialPointBatchOutput(
                cauchy_stress=baseline.cauchy_stress,
                consistent_tangent=baseline.consistent_tangent,
                state_new=baseline.state_new,
                state_schema=baseline.state_schema,
                tangent_convention=baseline.tangent_convention,
                stored_energy_density=baseline.stored_energy_density,
                stored_energy_density_components=(
                    baseline.stored_energy_density_components
                ),
                dissipation_density_increment=(baseline.dissipation_density_increment),
                applicability=("out_of_domain",) * request.point_count,
                diagnostics=tuple(
                    {"message": "outside test domain"}
                    for _ in range(request.point_count)
                ),
            )

    with pytest.raises(RuntimeError, match="OUT-OF-DOMAIN"):
        response.update(
            _Rejecting(spec),
            strain_old=np.diag((0.002, 0.0, 0.0)),
            strain_new=np.diag((0.2, 0.0, 0.0)),
            time=2.0,
            time_increment=1.0,
        )
    np.testing.assert_allclose(response.state.committed_state_vectors(), accepted)
    np.testing.assert_allclose(response.state.trial_state_vectors(), accepted)

    stress_before = response.cauchy_stress.function.x.array.copy()
    tangent_before = response.tangent.function.x.array.copy()

    def _fail_tangent_assignment(values):
        raise RuntimeError("injected tangent assignment failure")

    monkeypatch.setattr(response.tangent, "assign", _fail_tangent_assignment)
    with pytest.raises(RuntimeError, match="injected tangent assignment failure"):
        response.update(
            material,
            strain_old=np.diag((0.002, 0.0, 0.0)),
            strain_new=np.diag((0.003, 0.0, 0.0)),
            time=2.0,
            time_increment=1.0,
        )
    np.testing.assert_allclose(response.state.committed_state_vectors(), accepted)
    np.testing.assert_allclose(response.state.trial_state_vectors(), accepted)
    np.testing.assert_allclose(response.cauchy_stress.function.x.array, stress_before)
    np.testing.assert_allclose(response.tangent.function.x.array, tangent_before)


def json_safe(record):
    import json

    return json.dumps(record, sort_keys=True)
