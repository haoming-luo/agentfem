"""Independent energy, objectivity, condensation and standard-workflow checks."""

import numpy as np
import pytest
from mpi4py import MPI
from agentfem import constitutive, constraints, fields, mesh, models, studies, results
from agentfem.constitutive.geometric_elasticity import response_value


@pytest.mark.parametrize("dimension,plane_stress", [(3, False), (2, False), (2, True)])
def test_energy_gradient_tangent_and_rotation(dimension, plane_stress):
    m = constitutive.saint_venant_kirchhoff(
        young=210000, poisson=0.3, plane_stress=plane_stress
    )
    F = np.eye(dimension)
    F[0, 0] = 1.025
    F[0, 1] = 0.04
    response = response_value(F, m)
    h = 1e-6
    grad = np.zeros_like(F)
    tangent = np.zeros((dimension,) * 4)
    for i in range(dimension):
        for j in range(dimension):
            d = np.zeros_like(F)
            d[i, j] = h
            plus, minus = response_value(F + d, m), response_value(F - d, m)
            grad[i, j] = (plus["energy"] - minus["energy"]) / (2 * h)
            tangent[:, :, i, j] = (plus["first_piola"] - minus["first_piola"]) / (2 * h)
    np.testing.assert_allclose(grad, response["first_piola"], rtol=1e-7, atol=1e-4)
    np.testing.assert_allclose(tangent, response["tangent"], rtol=1e-7, atol=1e-4)
    theta = 1.2
    Q = np.eye(dimension)
    Q[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    rotated = response_value(Q @ F, m)
    assert rotated["energy"] == pytest.approx(response["energy"], rel=1e-12)
    Q3 = np.eye(3)
    Q3[:dimension, :dimension] = Q
    np.testing.assert_allclose(
        rotated["cauchy"], Q3 @ response["cauchy"] @ Q3.T, atol=1e-9
    )
    assert abs(response_value(Q, m)["energy"]) < 1e-20
    if plane_stress:
        assert abs(response["cauchy"][2, 2]) < 1e-9
        F3 = np.eye(3)
        F3[:2, :2] = F
        F3[2, 2] = response["thickness_stretch"]
        full = response_value(
            F3, constitutive.saint_venant_kirchhoff(young=210000, poisson=0.3)
        )
        assert full["energy"] == pytest.approx(response["energy"])
        np.testing.assert_allclose(full["first_piola"][:2, :2], response["first_piola"])


def test_invalid_deformation_rejected():
    m = constitutive.saint_venant_kirchhoff(young=1000, poisson=0.3, plane_stress=True)
    with pytest.raises(ValueError, match="J > 0"):
        response_value(np.diag([-1.0, 1.0]), m)
    with pytest.raises(ValueError, match="thickness"):
        response_value(np.diag([3.0, 3.0]), m)


def test_pin_global_match():
    d = mesh.rectangle((0.0, 0.005), (1.0, 1.005), (4, 4))
    u = fields.displacement(d)
    with pytest.raises(ValueError, match="zero global"):
        constraints.pin(u, at=(0.0, 0.0), components=(1,))
    support = constraints.pin(u, at=(0.0, 0.005), components=(1,))
    count = sum(int(bc.dof_indices()[1]) for bc in support.bcs)
    assert d.comm.allreduce(count, op=MPI.SUM) == 1


@pytest.mark.parametrize(
    "dimension,assumption", [(2, "plane_strain"), (2, "plane_stress"), (3, None)]
)
def test_standard_model_step_homogeneous(dimension, assumption):
    d = (
        mesh.rectangle((0.0, 0.0), (1.0, 1.0), (2, 2), comm=MPI.COMM_SELF)
        if dimension == 2
        else mesh.cuboid(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2), comm=MPI.COMM_SELF
        )
    )
    m = models.create(
        mesh=d,
        study=studies.static_solid(
            dimension=dimension, assumption=assumption, nonlinear=True
        ),
    )
    u = m.field(fields.displacement(d, degree=2))
    material = constitutive.saint_venant_kirchhoff(
        young=1000, poisson=0.3, plane_stress=assumption == "plane_stress"
    )
    m.material(material)
    # Every boundary follows a homogeneous stretch, leaving interior unknowns.
    boundary = mesh.boundary(
        d,
        lambda x: np.any(
            np.isclose(x[:dimension], 0.0) | np.isclose(x[:dimension], 1.0), axis=0
        ),
    )
    from dolfinx import fem

    imposed = fem.Function(u.space)
    imposed.interpolate(
        lambda x: np.vstack(
            [0.01 * x[0]] + [np.zeros_like(x[0]) for _ in range(dimension - 1)]
        )
    )
    m.fix(u, on=mesh.boundary(d, lambda x: np.isclose(x[0], 0.0)), component=0)
    m.prescribe(
        u, 0.01, on=mesh.boundary(d, lambda x: np.isclose(x[0], 1.0)), component=0
    )
    for component in range(1, dimension):
        m.fix(u, on=boundary, component=component)
    step = m.step(target=u, progress=False)
    step.solve()
    np.testing.assert_allclose(u.value.x.array, imposed.x.array, atol=1e-9)
    output = results.finite_strain_cell_fields(
        u, material, variables=("S", "J", "SENER")
    )
    expected = response_value(np.diag([1.01] + [1.0] * (dimension - 1)), material)
    np.testing.assert_allclose(
        output[0].x.array.reshape(-1, 3, 3),
        np.broadcast_to(expected["cauchy"], output[0].x.array.reshape(-1, 3, 3).shape),
        atol=1e-8,
    )
    np.testing.assert_allclose(output[1].x.array, expected["jacobian"], atol=1e-9)


def _axial_model(*, load=1.0, plane_stress=True, poisson=0.0):
    d = mesh.rectangle((0.0, 0.0), (1.0, 0.2), (4, 2), comm=MPI.COMM_SELF)
    m = models.create(
        mesh=d,
        study=studies.static_solid(
            dimension=2,
            assumption="plane_stress" if plane_stress else "plane_strain",
            nonlinear=True,
        ),
    )
    u = m.field(fields.displacement(d, degree=2))
    m.material(
        constitutive.saint_venant_kirchhoff(
            young=1000, poisson=poisson, plane_stress=plane_stress
        )
    )
    m.clamp(u, on=mesh.boundary(d, lambda x: np.isclose(x[0], 0.0)))
    m.traction((load, 0.0), on=mesh.boundary(d, lambda x: np.isclose(x[0], 1.0)))
    return m, u


def test_small_load_limit_and_principal_helpers():
    from agentfem.constitutive import hyperelasticity as h

    errors = []
    for load in (10.0, 1.0, 0.1):
        m, u = _axial_model(load=load)
        m.step(target=u, progress=False).solve()
        tip = float(results.probe(u.value, at=(1.0, 0.1))[0])
        # nu=0 gives exact nominal stress P=E*(stretch^3-stretch)/2.
        assert 500 * ((1 + tip) ** 3 - (1 + tip)) == pytest.approx(load, rel=1e-7)
        errors.append(abs(tip / (load / 1000) - 1))
        p = m.materials[0].item
        assert h.principal_nominal_stress([1 + tip, 1.0], p)[0] == pytest.approx(
            load, rel=1e-7
        )
        assert h.principal_energy_density([1 + tip, 1.0], p) > 0
    assert errors[1] < errors[0] / 9 and errors[2] < errors[1] / 9


def test_stvk_arc_length_matches_independent_uniaxial_law():
    from agentfem import procedures, solvers

    m, u = _axial_model()
    step = m.step(
        target=u,
        procedure=procedures.arc_length(),
        increments=4,
        arc_options=solvers.ArcLengthOptions(
            initial=0.1, maximum=0.1, displacement_scale=0.001
        ),
    )
    result = step.solve_result()
    tip = float(results.probe(u.value, at=(1.0, 0.1))[0])
    assert 500 * ((1 + tip) ** 3 - (1 + tip)) == pytest.approx(
        result.quantity("load_factor"), rel=1e-7
    )


def test_invalid_thickness_rolls_back_last_accepted_displacement():
    from agentfem import steps

    d = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (2, 2), comm=MPI.COMM_SELF)
    m = models.create(
        mesh=d,
        study=studies.static_solid(
            dimension=2, assumption="plane_stress", nonlinear=True
        ),
    )
    u = m.field(fields.displacement(d, degree=1))
    m.material(
        constitutive.saint_venant_kirchhoff(young=1000, poisson=0.3, plane_stress=True)
    )
    m.fix(u, on=mesh.boundary(d, lambda x: np.isclose(x[0], 0.0)), component=0)
    m.prescribe(
        u, 2.0, on=mesh.boundary(d, lambda x: np.isclose(x[0], 1.0)), component=0
    )
    m.fix(
        u, on=mesh.boundary(d, lambda x: np.ones(x.shape[1], dtype=bool)), component=1
    )
    step = m.step(
        target=u, incrementation=steps.FixedIncrementation((0.1, 1.0)), progress=False
    )
    step.solve(until=0.1)
    accepted = u.value.x.array.copy()
    with pytest.raises(RuntimeError):
        step.solve()
    np.testing.assert_array_equal(u.value.x.array, accepted)
    assert step.accepted_load_factor == pytest.approx(0.1)
    assert step.attempted_increments[-1].checks["minimum_thickness_squared"] < 0


def test_constant_quadrature_diagnostic():
    from agentfem.results.quantities import quadrature_extrema
    import ufl

    d = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (2, 2), comm=MPI.COMM_SELF)
    assert quadrature_extrema(ufl.as_ufl(1.0), d) == (1.0, 1.0)


def test_restart_matches_uninterrupted_path(tmp_path):
    from agentfem import steps

    reference, u = _axial_model(load=10.0)
    uninterrupted = reference.step(
        target=u, incrementation=steps.fixed(4), progress=False
    )
    uninterrupted.solve()
    m, v = _axial_model(load=10.0)
    partial = m.step(target=v, incrementation=steps.fixed(4), progress=False)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "stvk.json")
    fresh, w = _axial_model(load=10.0)
    restarted = fresh.step(target=w, incrementation=steps.fixed(4), progress=False)
    restarted.load_checkpoint(saved)
    restarted.solve()
    np.testing.assert_allclose(w.value.x.array, u.value.x.array, atol=1e-12)


def test_provider_identity_and_unsupported_options(tmp_path):
    m, u = _axial_model()
    m.step(target=u, progress=False)
    m.write_ir(tmp_path / "model.json")
    content = (tmp_path / "model.json").read_text()
    assert "saint_venant_kirchhoff" in content
    assert "plane_stress" in content
    with pytest.raises(
        (TypeError, ValueError), match="unsupported|Unknown|unknown|option"
    ):
        m.step(target=u, not_a_real_option=True)


@pytest.mark.parametrize("dimension,plane_stress", [(2, False), (2, True), (3, False)])
def test_ufl_tangent_matches_independent_component_derivative(dimension, plane_stress):
    import ufl
    from dolfinx import fem
    from agentfem.constitutive import hyperelasticity as h

    d = (
        mesh.rectangle((0.0, 0.0), (1.0, 1.0), (1, 1), comm=MPI.COMM_SELF)
        if dimension == 2
        else mesh.cuboid(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), comm=MPI.COMM_SELF
        )
    )
    u = fields.displacement(d, degree=1).value
    F = np.eye(dimension)
    F[0, 0] = 1.02
    F[0, 1] = 0.035
    u.interpolate(lambda x: (F - np.eye(dimension)) @ x[:dimension])
    material = constitutive.saint_venant_kirchhoff(
        young=210000, poisson=0.3, plane_stress=plane_stress
    )
    variable = ufl.variable(h.deformation_gradient(u))
    A = ufl.diff(h.first_piola_from_gradient(variable, material), variable)
    measured = fem.Expression(A, np.full((1, dimension), 0.25)).eval(
        d, np.array([0], dtype=np.int32)
    )
    expected = response_value(F, material)["tangent"]
    np.testing.assert_allclose(
        measured.reshape((dimension,) * 4), expected, rtol=1e-12, atol=1e-9
    )


def test_stvk_automatic_dynamics_rejected():
    d = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (2, 2), comm=MPI.COMM_SELF)
    m = models.create(
        mesh=d,
        study=studies.dynamic_solid(
            dimension=2, assumption="plane_stress", method="explicit"
        ),
    )
    u = m.field(fields.displacement(d))
    m.material(
        constitutive.saint_venant_kirchhoff(
            young=1000, poisson=0.3, density=1, plane_stress=True
        )
    )
    with pytest.raises(NotImplementedError, match="static solid"):
        m.step(target=u, dt=0.001, steps=2, progress=False)


def test_stvk_pressure_rejected():
    m, u = _axial_model()
    m.pressure(1.0, on=mesh.boundary(m.mesh, lambda x: np.isclose(x[0], 1.0)))
    with pytest.raises(NotImplementedError, match="traction"):
        m.step(target=u, progress=False)


def test_empty_point_support_and_valid_point_in_solid_origin():
    d = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (2, 2), comm=MPI.COMM_SELF)
    u = fields.displacement(d)
    c = constraints.pin(u, at=(0.0, 0.0))
    assert len(c.dirichlet) == 2
    for item in c.dirichlet:
        np.testing.assert_allclose(
            item.match_evidence["matched_coordinates"], [[0.0, 0.0]], atol=1e-14
        )
        assert item.summary()["point_match"]["owned_global_dofs"] == 1


def test_shared_material_tangent_dispatch_does_not_fall_back_to_neo_hookean():
    from agentfem import fracture

    m = constitutive.saint_venant_kirchhoff(
        young=1000, poisson=0.3, density=1.0, plane_stress=True
    )
    F = np.diag([1.05, 0.99])
    expected = response_value(F, m)
    np.testing.assert_allclose(
        fracture.neo_hookean_material_tangent(F, m), expected["tangent"]
    )
    wave = fracture.incremental_wave_speeds(F, [1.0, 0.0], m)
    assert wave.deformation_jacobian == pytest.approx(expected["jacobian"])
    assert np.all(wave.speeds > 0)
