# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerical acceptance for solid stability and periodic elastic properties."""

import numpy as np
import pytest
from agentfem import constraints, fields, mesh, models, studies, mechanics
from agentfem.constitutive import isotropic_elastic


def column(*, nx=40, length=10.0, reference=1.0, dim=2, fixed_base=0.0, modes=1):
    if dim == 2:
        domain = mesh.rectangle(
            (0.0, -0.5), (length, 0.5), (nx, 4), cell_type="quadrilateral"
        )
    else:
        domain = mesh.cuboid(
            (0.0, -0.5, -0.75), (length, 0.5, 0.75), (nx, 3, 3), cell_type="hexahedron"
        )
    study = studies.buckling_solid(
        dimension=dim, assumption="plane_stress" if dim == 2 else None
    )
    model = models.create(study=study, mesh=domain)
    u = model.field(fields.displacement(domain, degree=2))
    # nu=0: uniform compression is an exact equilibrated reference with a clamped end.
    model.material(isotropic_elastic(young=1000.0, density=1.0, poisson=0.0))
    model.clamp(
        u,
        on=mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), tag=1, name="clamp"),
    )
    from dolfinx import fem

    ref = fem.Function(u.space)

    def compression(x):
        values = np.zeros((dim, x.shape[1]))
        values[0] = -reference * x[0] / 1000.0
        return values

    ref.interpolate(compression)
    kwargs = {}
    if fixed_base:
        base = ref.copy()
        base.x.array[:] *= fixed_base / reference
        kwargs = {"base_displacement": base, "base_name": "fixed compression"}
    return model.step(
        target=u,
        reference_displacement=ref,
        reference_name="unit compressive stress",
        modes=modes,
        **kwargs,
    )


def test_cantilever_euler_and_load_scaling():
    step = column(length=20.0, nx=50)
    result = step.solve_result()
    expected_stress = np.pi**2 * 1000.0 / (48.0 * 20.0**2)
    critical = result.quantity("load_factors")[0]
    assert critical == pytest.approx(expected_stress, rel=0.01)
    double = column(length=20.0, nx=50, reference=2.0).solve_result()
    assert 2 * double.quantity("load_factors")[0] == pytest.approx(critical, rel=1.0e-7)
    base = column(length=20.0, nx=50, fixed_base=critical * 0.2).solve_result()
    assert base.quantity("load_factors")[0] + critical * 0.2 == pytest.approx(
        critical, rel=1.0e-7
    )
    assert max(result.quantity("relative_residuals")) < 1.0e-8


@pytest.mark.parametrize(
    "dim,assumption", [(2, "plane_stress"), (2, "plane_strain"), (3, None)]
)
def test_homogeneous_periodic_cell(dim, assumption):
    domain = (
        mesh.rectangle((0.0, 0.0), (1.0, 1.0), (3, 3))
        if dim == 2
        else mesh.cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2))
    )
    model = models.create(
        study=studies.static_solid(dimension=dim, assumption=assumption), mesh=domain
    )
    w = model.field(fields.displacement(domain, degree=2))
    model.material(isotropic_elastic(young=1000.0, density=1.0, poisson=0.25))
    result = mechanics.periodic_elasticity(model, w, anchor=(0.0,) * dim)
    modulus = 1000.0 if assumption != "plane_strain" else 1000.0 / (1.0 - 0.25**2)
    poisson = 0.25 if assumption != "plane_strain" else 0.25 / (1.0 - 0.25)
    assert result.quantity("young_moduli") == pytest.approx(
        np.full(dim, modulus), rel=1.0e-8
    )
    assert result.quantity("poisson_ratios")[0, 1] == pytest.approx(poisson, abs=1.0e-8)
    assert max(result.quantity("hill_mandel_relative_errors")) < 1.0e-8
    assert result.quantity("solid_fraction") == pytest.approx(1.0)


def test_anisotropic_engineering_convention():
    C = np.array([[10.0, -2.0, 0.0], [-2.0, 5.0, 0.0], [0.0, 0.0, 3.0]])
    props = mechanics.elastic_engineering_properties(C)
    assert props["young_moduli"] == pytest.approx([9.2, 4.6])
    assert props["poisson_ratios"][0, 1] == pytest.approx(-0.4)
    assert props["poisson_ratios"][1, 0] == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="singular"):
        mechanics.elastic_engineering_properties(
            np.zeros((3, 3)) + np.diag([1.0, 1.0, 0.0])
        )


def test_three_dimensional_column_and_length_scaling():
    three = column(dim=3, length=20.0, nx=36).solve_result()
    two = column(length=20.0, nx=36).solve_result()
    short = column(length=10.0, nx=36).solve_result()
    assert three.quantity("load_factors")[0] == pytest.approx(
        two.quantity("load_factors")[0], rel=0.01
    )
    assert short.quantity("load_factors")[0] / two.quantity("load_factors")[
        0
    ] == pytest.approx(4.0, rel=0.01)


def test_reentrant_cell_energy_and_negative_poisson():
    pytest.importorskip("gmsh")
    domain = mesh.reentrant_honeycomb(mesh_size=0.10).domain
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"), mesh=domain
    )
    w = model.field(fields.displacement(domain, degree=2))
    model.material(isotropic_elastic(young=1000.0, poisson=0.3, density=1.0))
    result = mechanics.periodic_elasticity(model, w, anchor=(0.0, 0.0))
    assert result.quantity("poisson_ratios")[0, 1] < -0.5
    assert result.quantity("poisson_ratios")[1, 0] < -0.2
    assert max(result.quantity("hill_mandel_relative_errors")) < 1.0e-8
    assert max(result.quantity("equilibrium_relative_errors")) < 1.0e-8
    other = mechanics.periodic_elasticity(model, w, anchor=(0.0, 0.0), amplitude=0.002)
    assert other.quantity("effective_stiffness") == pytest.approx(
        result.quantity("effective_stiffness"), rel=1.0e-8, abs=1.0e-8
    )


def test_buckling_rejects_nonzero_perturbation_support():
    step = column(nx=6)
    u = step.target
    step.constraints = (
        constraints.fixed(
            u, location=lambda x: np.isclose(x[0], 0.0), value=(0.1, 0.0)
        ),
    )
    with pytest.raises(NotImplementedError, match="homogeneous"):
        step.solve()


def test_layered_cell_series_and_parallel_response():
    domain = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (8, 4))
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"), mesh=domain
    )
    w = model.field(fields.displacement(domain, degree=2))
    regions = mesh.partition_cells(
        domain,
        left=lambda x: x[0] <= 0.5 + 1.0e-12,
        right=lambda x: x[0] >= 0.5 - 1.0e-12,
    )
    for young, region in ((1000.0, regions["left"]), (2000.0, regions["right"])):
        model.material(
            isotropic_elastic(young=young, poisson=0.0, density=1.0), region=region
        )
    result = mechanics.periodic_elasticity(model, w, anchor=(0.0, 0.0))
    # nu=0 separates series (normal to layers) and parallel (along layers).
    assert result.quantity("young_moduli") == pytest.approx(
        [4000.0 / 3.0, 1500.0], rel=1.0e-9
    )


def test_pin_roller_column_retains_end_rotation():
    step = column(length=20.0, nx=50)
    u = step.target
    step.constraints = (
        constraints.fixed_component(u, 1, location=lambda x: np.isclose(x[0], 0.0)),
        constraints.fixed_component(u, 1, location=lambda x: np.isclose(x[0], 20.0)),
        constraints.pin(u, at=(0.0, 0.0), components=(0,)),
    )
    result = step.solve_result()
    assert result.quantity("load_factors")[0] == pytest.approx(
        np.pi**2 * 1000.0 / (12.0 * 20.0**2), rel=0.01
    )


def test_apparent_poisson_affine_gauge_and_zero_strain():
    domain = mesh.rectangle((0.0, 0.0), (2.0, 1.0), (4, 2))
    u = fields.displacement(domain, degree=2)
    u.value.interpolate(lambda x: np.vstack((0.001 * x[0], 0.0005 * x[1])))
    faces = [
        mesh.boundary(
            domain, lambda x, a=a, v=v: np.isclose(x[a], v), name=n, tag=i + 1
        )
        for i, (a, v, n) in enumerate(
            ((0, 0.0, "left"), (0, 2.0, "right"), (1, 0.0, "bottom"), (1, 1.0, "top"))
        )
    ]
    opts = dict(
        axial_axis=0,
        transverse_axis=1,
        axial_faces=faces[:2],
        transverse_faces=faces[2:],
        axial_length=2.0,
        transverse_length=1.0,
    )
    assert mechanics.apparent_poisson_ratio(u, **opts)[
        "apparent_poisson_ratio"
    ] == pytest.approx(-0.5)
    u.value.x.array[:] = 0.0
    with pytest.raises(ValueError, match="too small"):
        mechanics.apparent_poisson_ratio(u, **opts)
