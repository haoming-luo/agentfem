"""Global acceptance for provider-neutral small-strain materials."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    mechanics,
    mesh,
    models,
    solvers,
    steps,
    studies,
)


class LinearBatchMaterial:
    name = "linear_batch_material"
    rate_independent = True

    def __init__(self, young=1000.0, poisson=0.3):
        self.parameters = {"young": float(young), "poisson": float(poisson)}
        self.state_schema = constitutive.MaterialStateSchema(
            "test.linear_batch_state",
            (
                constitutive.MaterialStateVariable(
                    "history",
                    unit="1",
                    description="Accepted accumulated strain-path length.",
                ),
            ),
            version="1.0",
        )
        self.parameter_schema = constitutive.MaterialParameterSchema(
            "test.linear_batch_parameters",
            (
                constitutive.MaterialParameter("young", unit="Pa", lower=0.0),
                constitutive.MaterialParameter(
                    "poisson", unit="1", lower=-0.999, upper=0.499
                ),
            ),
            version="1.0",
        )
        self.tangent_convention = constitutive.small_strain_tangent_convention()
        self.batch_calls = 0

    def summary(self):
        return {
            "kind": "test_small_strain_user_material",
            "name": self.name,
            "parameters": dict(self.parameters),
            "state_schema": self.state_schema.summary(),
        }

    def _constants(self):
        young = self.parameters["young"]
        poisson = self.parameters["poisson"]
        mu = young / (2.0 * (1.0 + poisson))
        lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        tangent = np.zeros((6, 6))
        tangent[:3, :3] = lame
        tangent[np.arange(3), np.arange(3)] += 2.0 * mu
        tangent[np.arange(3, 6), np.arange(3, 6)] = 2.0 * mu
        return lame, mu, tangent

    def update(self, point):
        request = constitutive.SmallStrainMaterialPointBatchInput(
            strain_old=point.strain_old[None, ...],
            strain_new=point.strain_new[None, ...],
            time=point.time,
            time_increment=point.time_increment,
            parameters=point.parameters,
            state_old=point.state_old[None, ...],
            state_schema=point.state_schema,
            parameter_schema=point.parameter_schema,
        )
        result = self.update_batch(request)
        return constitutive.SmallStrainMaterialPointOutput(
            cauchy_stress=result.cauchy_stress[0],
            consistent_tangent=result.consistent_tangent[0],
            state_new=result.state_new[0],
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
            diagnostics={"provider": "numpy_test"},
        )

    def update_batch(self, request):
        self.batch_calls += 1
        lame, mu, tangent = self._constants()
        strain = request.strain_new
        stress = (
            lame * np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3)
            + 2.0 * mu * strain
        )
        path = np.linalg.norm(request.strain_new - request.strain_old, axis=(1, 2))
        state = request.state_old.copy()
        state[:, 0] += path
        return constitutive.SmallStrainMaterialPointBatchOutput(
            cauchy_stress=stress,
            consistent_tangent=np.broadcast_to(tangent, (len(strain), 6, 6)).copy(),
            state_new=state,
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
            diagnostics=tuple({"provider": "numpy_test"} for _ in strain),
        )


def _problem():
    domain = dolfinx_mesh.create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.asarray((1.0, 0.2, 0.2))],
        [2, 1, 1],
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = LinearBatchMaterial()
    model.material(material)
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=1.0),
        component=0,
        value=0.002,
    )
    return model, displacement, material


def test_generic_small_strain_material_step_solves_and_publishes_state(tmp_path):
    model, displacement, material = _problem()
    step = mechanics.small_strain_material_step(
        displacement=displacement,
        material=material,
        external_force=None,
        constraints=model.constraints,
        study=model.study,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(maximum_iterations=8, line_search="basic"),
        progress=False,
    )
    result = step.solve_result()

    assert step.last_solve_info.completed_step
    assert material.batch_calls > 1
    assert result.fields["S"].location == "quadrature_points"
    assert result.fields["history"].location == "quadrature_points"
    assert result.fields["S_CELL"].location == "cells"
    assert np.max(np.abs(displacement.value.x.array)) == pytest.approx(0.002)

    paused_model, paused_u, paused_material = _problem()
    paused = mechanics.small_strain_material_step(
        displacement=paused_u,
        material=paused_material,
        external_force=None,
        constraints=paused_model.constraints,
        study=paused_model.study,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(maximum_iterations=8, line_search="basic"),
        progress=False,
    )
    paused.solve(until=0.5)
    checkpoint = paused.save_checkpoint(tmp_path / "learned")
    restored_model, restored_u, restored_material = _problem()
    restored = mechanics.small_strain_material_step(
        displacement=restored_u,
        material=restored_material,
        external_force=None,
        constraints=restored_model.constraints,
        study=restored_model.study,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(maximum_iterations=8, line_search="basic"),
        progress=False,
    )
    restored.load_checkpoint(checkpoint)
    restored.solve()
    assert restored.last_solve_info.completed_step
    assert np.allclose(restored_u.value.x.array, displacement.value.x.array)


def test_public_model_step_dispatches_small_strain_material():
    model, displacement, _material = _problem()
    step = model.step(
        target=displacement,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(maximum_iterations=8, line_search="basic"),
        progress=False,
    )

    assert isinstance(step, mechanics.SmallStrainMaterialStep)
    step.solve()
    assert step.last_solve_info.completed_step


@pytest.mark.parametrize("kind", ["j2", "chaboche", "provider", "creep", "maxwell", "maxwell_harmonic"])
@pytest.mark.parametrize("boundary_kind", ["rigid_contact", "foundation"])
def test_small_strain_boundary_operator_is_never_silently_ignored(kind, boundary_kind):
    from agentfem import boundary_models
    from agentfem.step_providers import step_capability

    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    if kind == "creep":
        study = studies.creep_solid(dimension=3)
    elif kind == "maxwell":
        study = studies.viscoelastic_solid(dimension=3)
    elif kind == "maxwell_harmonic":
        study = studies.harmonic_solid(dimension=3)
    else:
        study = studies.static_solid(dimension=3, nonlinear=True)
    model = models.create(study=study, mesh=domain)
    u = model.field(fields.displacement(domain))
    if kind == "j2":
        material = constitutive.J2LinearIsotropicHardening(
            young=1000., poisson=.3, yield_stress=2.)
    elif kind == "chaboche":
        material = constitutive.chaboche(
            young=1000., poisson=.3, yield_stress=2., backstresses=((100., 10.),))
    elif kind == "creep":
        material = constitutive.isotropic_power_law(
            young=1000., poisson=.3, density=1., coefficient=1e-5,
            stress_exponent=2., reference_stress=1.)
    elif kind in {"maxwell", "maxwell_harmonic"}:
        material = constitutive.IsotropicGeneralizedMaxwell.from_prony(
            instantaneous_young_modulus=1000., instantaneous_poisson_ratio=.3,
            shear_relaxation_ratios=[.2], bulk_relaxation_ratios=[.1], relaxation_times=[1.])
    else:
        material = LinearBatchMaterial()
    model.material(material)
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.), name="right")
    model.clamp(u, on=left)
    options = ({"frequency": 1.} if kind == "maxwell_harmonic" else
               {"duration": 1.} if kind in {"creep", "maxwell"} else {})
    assert step_capability(model, target=u, options=options)["supported"]
    if boundary_kind == "rigid_contact":
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            right, boundary_models.rigid_body(boundary_models.rigid_plane(
                point=(1.005, 0., 0.), normal=(-1., 0., 0.))), penalty=1000., name="tool"))
    else:
        model.elastic_foundation(on=right, stiffness=1000.)
    before = u.value.x.array.copy()
    capability = step_capability(model, target=u, options=options)
    assert not capability["supported"]
    assert len(capability["rejection_reasons"]) == 1
    assert any(item["message"].startswith("AFM-SMALL-STRAIN-BOUNDARY")
               for item in capability["rejection_reasons"])
    with pytest.raises(NotImplementedError, match="AFM-SMALL-STRAIN-BOUNDARY"):
        model.step(target=u, **options)
    # Older public compatibility factories must not bypass the same boundary.
    from agentfem import _step_builders

    builder = {
        "j2": model.j2_plasticity_step,
        "chaboche": model.j2_plasticity_step,
        "creep": model.creep_step,
    }.get(kind)
    with pytest.raises(NotImplementedError, match="AFM-SMALL-STRAIN-BOUNDARY"):
        if builder is not None:
            builder(target=u, progress=False, **options)
        else:
            lower = {
                "maxwell": _step_builders.viscoelastic,
                "maxwell_harmonic": _step_builders.harmonic_viscoelastic,
                "provider": _step_builders.small_strain_material,
            }[kind]
            lower(model, target=u, **options)
    assert model.steps == []
    np.testing.assert_array_equal(u.value.x.array, before)
