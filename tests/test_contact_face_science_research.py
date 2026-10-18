"""Independent spring-limit evidence for the bounded whole-face contact route.

The finite-strain elastic limit of the J2 material has a homogeneous
uniaxial solution. Its nominal stress is ``E log(1+u)/(1+u)`` for Poisson
ratio zero, so the contact equilibrium can be solved without AgentFEM's
contact assembly or Newton procedure.
"""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
from scipy.optimize import brentq

from agentfem import (
    boundary_models,
    constitutive,
    fields,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.boundary_models.contact_feature_pair_geometry_research import (
    certify_touching_closed_contact_face,
)
from agentfem.boundary_models.contact_smooth_face_research import (
    integrate_certified_smooth_closed_face,
)

_INITIAL_OVERLAP = 0.05
_SIDE = 0.1


def _wide_closed_tool(*, initial_overlap=_INITIAL_OVERLAP):
    # Its front is one triangle that strictly contains the entire slave face.
    yz = np.asarray(((-3.0, -3.0), (10.0, -3.0), (-3.0, 10.0)))
    front = np.column_stack((np.full(3, 1.0 - initial_overlap), yz))
    back = np.column_stack((np.full(3, 4.0), yz))
    triangles = [(0, 2, 1), (3, 4, 5)]
    for first in range(3):
        second = (first + 1) % 3
        triangles.extend(((first, second, second + 3),
                          (first, second + 3, first + 3)))
    return boundary_models.closed_triangulated_rigid_solid(
        vertices=np.vstack((front, back)), triangles=triangles,
        reviewed_embedded=True,
    )


def _elastic_column(
    *, cells: int, increments: int, penalty: float,
    initial_overlap=_INITIAL_OVERLAP, poisson=0.0,
    surface_mode="closed", tool_motion=False, yield_stress=1e9,
    maximum_iterations=25,
):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    domain.geometry.x[:, 1:] *= _SIDE
    model = models.create(
        study=studies.nonlinear_static(physics="solid_mechanics", dimension=3),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=poisson, yield_stress=yield_stress,
        hardening_modulus=100.0,
    ))
    if surface_mode == "closed":
        surface = _wide_closed_tool(initial_overlap=initial_overlap)
        integration = "certified_closed_two_feature_research"
    elif surface_mode == "plane":
        surface = boundary_models.rigid_plane(
            point=(1.0 - initial_overlap, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
        )
        integration = "gauss"
    else:
        raise ValueError("Unknown benchmark surface mode.")
    motion_schedule = None
    if tool_motion:
        motion_schedule = boundary_models.prescribed_rigid_motion_schedule(
            boundary_models.prescribed_rigid_motion(
                translation=(-0.02, 0.0, 0.0),
                reference_point=(1.0, 0.0, 0.0),
            ),
            start_time=0.0, end_time=1.0,
        )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right, boundary_models.rigid_body(
            surface, motion_schedule=motion_schedule,
        ),
        penalty=penalty, initial_penetration_limit=0.1,
        name="wide_closed_tool",
    ))
    model.traction((30.0, 0.0, 0.0), on=right)
    step = model.step(
        target=displacement, material=material,
        contact_integration=integration,
        incrementation=steps.fixed(increments), progress=False,
        solver_options=solvers.newton(
            relative_tolerance=1e-9, absolute_tolerance=1e-10,
            maximum_iterations=maximum_iterations, line_search="backtracking",
        ),
    )
    return displacement, step


@pytest.mark.parametrize(
    ("cells", "increments", "penalty"),
    ((1, 1, 500.0), (2, 1, 500.0), (1, 3, 500.0),
     (2, 3, 500.0), (1, 1, 250.0), (1, 1, 1000.0)),
)
def test_certified_face_j2_elastic_limit_matches_independent_spring_equilibrium(
    cells, increments, penalty,
):
    displacement, step = _elastic_column(
        cells=cells, increments=increments, penalty=penalty,
    )
    result = step.solve_result()
    assert result.status == "completed"
    u = float(results.probe(displacement, at=(1.0, 0.5 * _SIDE, 0.5 * _SIDE))[0])
    independent = brentq(
        lambda value: 1000.0 * np.log1p(value) / (1.0 + value)
        + penalty * (value + _INITIAL_OVERLAP) - 30.0,
        -0.04, 0.05,
    )
    assert u == pytest.approx(independent, rel=3e-5, abs=1e-8)
    assert result.quantity("wide_closed_tool_contact_energy") == pytest.approx(
        0.5 * penalty * (u + _INITIAL_OVERLAP)**2 * _SIDE**2,
        rel=5e-5, abs=1e-8,
    )
    path = result.metadata["rigid_contact_face_path"]
    assert path["complete"] is True
    assert path["accepted_station_count"] == increments + 1
    assert path["pairs"][0]["all_observed_faces_certified"] is True
    assert np.max(step.response.state.committed_state_vectors()[:, -2]) == pytest.approx(
        0.0, abs=1e-12,
    )


def test_exactly_touching_smooth_face_starts_with_zero_generalized_tangent():
    displacement, step = _elastic_column(
        cells=1, increments=3, penalty=500.0, initial_overlap=0.0,
    )
    initial = step.contact_provider.evaluate().operator
    assert initial.face_activity_kinds == ("touching", "touching")
    assert initial.potential_energy == 0.0
    assert not np.any(initial.nodal_residual)
    assert not len(initial.tangent_values)
    result = step.solve_result()
    assert result.status == "completed"
    u = float(results.probe(displacement, at=(1.0, 0.5 * _SIDE, 0.5 * _SIDE))[0])
    independent = brentq(
        lambda value: 1000.0 * np.log1p(value) / (1.0 + value)
        + 500.0 * value - 30.0,
        0.0, 0.05,
    )
    assert u == pytest.approx(independent, rel=3e-5, abs=1e-8)
    assert result.metadata["rigid_contact_face_path"]["complete"] is True


def test_touching_certificate_does_not_zero_a_resolvable_signed_gap():
    face = np.asarray((
        (1.0, 0.0, 0.0),
        (1.0, _SIDE, 0.0),
        (1.0, _SIDE, _SIDE),
    ))
    tool = _wide_closed_tool(initial_overlap=0.0)
    certificate = certify_touching_closed_contact_face(face, face, tool)
    assert certificate.maximum_absolute_gap <= certificate.numerical_gap_tolerance
    for offset in (-1e-8, 1e-8):
        moved = face + np.asarray((offset, 0.0, 0.0))
        with pytest.raises(NotImplementedError, match="numerically coplanar"):
            certify_touching_closed_contact_face(face, moved, tool)
    shift = np.asarray((1e10, 0.0, 0.0))
    shifted_tool = boundary_models.closed_triangulated_rigid_solid(
        vertices=tool.facets.vertices + shift,
        triangles=tool.facets.triangles,
        reviewed_embedded=True,
    )
    with pytest.raises((ValueError, NotImplementedError)):
        certify_touching_closed_contact_face(face + shift, face + shift, shifted_tool)


def test_partial_smooth_face_energy_gradient_and_tangent_follow_repartition():
    current = np.asarray((
        (0.94, 0.0, 0.0),
        (0.96, 0.08, 0.0),
        (0.97, 0.0, 0.08),
    ))
    tool = _wide_closed_tool()

    def evaluated(points):
        return integrate_certified_smooth_closed_face(
            current, points, tool, penalty=500.0,
        )

    reference = evaluated(current)
    assert 0.0 < reference.active_reference_fraction < 1.0
    direction = np.asarray((
        (0.2, -0.1, 0.3),
        (-0.3, 0.2, 0.1),
        (0.1, 0.1, -0.2),
    ))
    size = 1e-6
    plus = evaluated(current + size * direction)
    minus = evaluated(current - size * direction)
    numerical_energy = (plus.response.potential - minus.response.potential) / (2 * size)
    assert numerical_energy == pytest.approx(
        float(np.sum(reference.response.nodal_residual * direction)),
        rel=2e-6, abs=2e-8,
    )
    numerical_force = (
        plus.response.nodal_residual - minus.response.nodal_residual
    ) / (2 * size)
    np.testing.assert_allclose(
        numerical_force,
        np.einsum("aibj,bj->ai", reference.response.nodal_tangent, direction),
        rtol=2e-6, atol=2e-8,
    )


def test_initial_touching_moving_tool_preserves_work_and_restart(tmp_path):
    settings = dict(
        cells=1, increments=3, penalty=500.0,
        initial_overlap=0.0, tool_motion=True,
    )
    _, full = _elastic_column(**settings)
    result = full.solve_result()
    assert result.status == "completed"
    assert full.energy_history[-1].tool_work > 0.0
    _, partial = _elastic_column(**settings)
    partial.solve(until=1 / 3)
    saved = partial.save_checkpoint(tmp_path / "initial_touching_tool")
    _, restored = _elastic_column(**settings)
    restored.load_checkpoint(saved)
    resumed = restored.solve_result()
    assert resumed.status == "completed"
    np.testing.assert_allclose(
        restored.solution.x.array, full.solution.x.array,
        rtol=1e-9, atol=1e-12,
    )
    assert restored.energy_history[-1].tool_work == pytest.approx(
        full.energy_history[-1].tool_work, rel=1e-9, abs=1e-12,
    )
    assert resumed.metadata["rigid_contact_face_path"] == result.metadata[
        "rigid_contact_face_path"
    ]


@pytest.mark.parametrize(
    ("cells", "increments", "penalty"),
    ((1, 1, 250.0), (1, 1, 500.0), (1, 1, 1000.0),
     (2, 1, 500.0), (2, 3, 250.0), (2, 3, 500.0),
     (2, 3, 1000.0), (3, 1, 500.0), (3, 3, 500.0)),
)
def test_inhomogeneous_clamped_column_matches_independent_planar_backend(
    cells, increments, penalty,
):
    closed_field, closed = _elastic_column(
        cells=cells, increments=increments, penalty=penalty,
        poisson=0.3, surface_mode="closed",
    )
    plane_field, plane = _elastic_column(
        cells=cells, increments=increments, penalty=penalty,
        poisson=0.3, surface_mode="plane",
    )
    closed_result = closed.solve_result()
    plane_result = plane.solve_result()
    assert closed_result.status == plane_result.status == "completed"
    np.testing.assert_allclose(
        closed_field.value.x.array, plane_field.value.x.array,
        rtol=2e-6, atol=2e-8,
    )
    assert closed_result.quantity("wide_closed_tool_contact_energy") == pytest.approx(
        plane_result.quantity("wide_closed_tool_contact_energy"),
        rel=2e-6, abs=1e-8,
    )
    assert closed_result.metadata["rigid_contact_face_path"]["complete"] is True


@pytest.mark.parametrize("cells", (1, 2, 3, 4))
def test_plastic_clamped_column_closed_face_matches_plane_across_meshes(
    cells,
):
    settings = dict(
        cells=cells, increments=6, penalty=500.0,
        poisson=0.3, yield_stress=1.0, maximum_iterations=100,
    )
    closed_field, closed = _elastic_column(**settings, surface_mode="closed")
    plane_field, plane = _elastic_column(**settings, surface_mode="plane")
    closed_result = closed.solve_result()
    plane_result = plane.solve_result()
    assert closed_result.status == plane_result.status == "completed"
    closed_state = closed.response.state.committed_state_vectors()
    plane_state = plane.response.state.committed_state_vectors()
    assert np.max(closed_state[:, -2]) > 1e-4
    np.testing.assert_allclose(
        closed_field.value.x.array, plane_field.value.x.array,
        rtol=2e-6, atol=2e-8,
    )
    np.testing.assert_allclose(closed_state, plane_state, rtol=2e-6, atol=2e-8)
    assert closed_result.quantity("wide_closed_tool_contact_energy") == pytest.approx(
        plane_result.quantity("wide_closed_tool_contact_energy"),
        rel=2e-6, abs=1e-8,
    )
    assert closed_result.metadata["rigid_contact_face_path"]["complete"] is True


def test_near_yield_first_increment_total_tangent_keeps_old_state_fixed():
    settings = dict(
        cells=3, increments=6, penalty=500.0, poisson=0.3,
        yield_stress=1.0, maximum_iterations=100, surface_mode="closed",
    )
    _, solved = _elastic_column(**settings)
    solved.solve(until=1.0 / 6.0)
    reference = solved.solution.x.array.copy()

    _, trial = _elastic_column(**settings)
    trial._apply_loading(1.0 / 6.0)
    direction = np.linspace(-0.2, 0.3, len(reference))
    constrained = np.unique(np.concatenate([
        np.asarray(bc.dof_indices()[0][:bc.dof_indices()[1]], dtype=int)
        for bc in trial.bcs
    ]))
    direction[constrained] = 0.0
    free = np.setdiff1d(np.arange(len(reference)), constrained)

    def correction_rhs(scale):
        trial.solution.x.array[:] = reference + scale * direction
        trial.solution.x.scatter_forward()
        trial._update_response(start_factor=0.0, target_factor=1.0 / 6.0)
        vector, _ = trial._correction_rhs()
        values = vector.array.copy()
        vector.destroy()
        return values

    for base_scale in (-1e-3, 1e-3):
        correction_rhs(base_scale)
        matrix = trial._assemble_tangent()
        input_vector = matrix.createVecRight()
        output_vector = matrix.createVecLeft()
        try:
            input_vector.array[:] = direction
            matrix.mult(input_vector, output_vector)
            expected = -output_vector.array.copy()
        finally:
            output_vector.destroy()
            input_vector.destroy()
        size = 1e-6
        measured = (
            correction_rhs(base_scale + size)
            - correction_rhs(base_scale - size)
        ) / (2.0 * size)
        np.testing.assert_allclose(
            measured[free], expected[free], rtol=2e-4, atol=2e-6,
        )
