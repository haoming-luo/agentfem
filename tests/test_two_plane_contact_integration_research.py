"""Research-only two-feature area plus line derivative oracle."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_two_plane_contact_integration.py"
two_plane_triangle_contact = runpy.run_path(str(_SCRIPT))["two_plane_triangle_contact"]


@pytest.mark.parametrize("width, center", ((0.05, 0.3), (0.2, 0.5), (0.1, 0.8)))
def test_two_plane_integral_and_full_feature_tangent(width, center):
    k = 7.0
    gaps = np.asarray((
        (-center - width, 1 - center - width, -center - width),
        (center - width, center - 1 - width, center - width),
    ))
    normals = np.asarray(((1.0, 0, 0), (-1.0, 0, 0)))
    reference = two_plane_triangle_contact(gaps, normals, penalty=k)
    assert reference["potential"] == pytest.approx(
        k * width**3 * (1 - center) / 3, rel=1e-12, abs=0.0,
    )
    assert np.sum(reference["nodal_residual"][:, 0]) == pytest.approx(
        k * width**3 / 3,
    )
    assert np.sum(reference["nodal_tangent"][0::3, 0::3]) == pytest.approx(
        0.0, abs=1e-12,
    )
    assert np.sum(reference["feature_tangent"][0::3, 0::3]) < 0.0
    direction = np.random.default_rng(71).normal(size=(3, 3))
    h = 1e-7
    changes = np.einsum("ai,fi->fa", direction, normals)
    plus = two_plane_triangle_contact(gaps + h * changes, normals, penalty=k)
    minus = two_plane_triangle_contact(gaps - h * changes, normals, penalty=k)
    energy_derivative = (plus["potential"] - minus["potential"]) / (2 * h)
    force_derivative = (
        plus["nodal_residual"] - minus["nodal_residual"]
    ) / (2 * h)
    assert energy_derivative == pytest.approx(
        np.sum(reference["nodal_residual"] * direction), rel=1e-7,
    )
    np.testing.assert_allclose(
        force_derivative.reshape(-1),
        reference["nodal_tangent"] @ direction.reshape(-1),
        rtol=2e-7, atol=1e-9,
    )
    np.testing.assert_allclose(
        reference["nodal_tangent"], reference["nodal_tangent"].T,
        atol=1e-12,
    )


def test_two_plane_oracle_rejects_coincident_feature_partition():
    with pytest.raises(ValueError, match="unique feature"):
        two_plane_triangle_contact(
            np.zeros((2, 3)), ((1, 0, 0), (-1, 0, 0)),
        )


def test_two_plane_oracle_is_invariant_to_feature_and_vertex_order():
    gaps = np.asarray(((-0.42, 0.61, -0.37), (0.31, -0.28, 0.19)))
    normals = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    reference = two_plane_triangle_contact(gaps, normals, penalty=8.0)
    exchanged = two_plane_triangle_contact(gaps[::-1], normals[::-1], penalty=8.0)
    assert exchanged["potential"] == pytest.approx(reference["potential"], rel=1e-14)
    np.testing.assert_allclose(exchanged["nodal_residual"], reference["nodal_residual"], atol=1e-14)
    np.testing.assert_allclose(exchanged["nodal_tangent"], reference["nodal_tangent"], atol=1e-14)
    permutation = np.asarray((2, 0, 1))
    reordered = two_plane_triangle_contact(gaps[:, permutation], normals, penalty=8.0)
    dofs = np.arange(9).reshape(3, 3)[permutation].reshape(-1)
    assert reordered["potential"] == pytest.approx(reference["potential"], rel=1e-14)
    np.testing.assert_allclose(
        reordered["nodal_residual"], reference["nodal_residual"][permutation], atol=1e-14,
    )
    np.testing.assert_allclose(
        reordered["nodal_tangent"], reference["nodal_tangent"][np.ix_(dofs, dofs)],
        atol=1e-14,
    )


def test_two_plane_oracle_rejects_nonunique_boundary_switch():
    with pytest.raises(ValueError, match="two-sided tangent"):
        two_plane_triangle_contact(
            ((-0.2, 0.8, -0.2), (-0.2, -0.2, -0.2)),
            ((1.0, 0, 0), (-1.0, 0, 0)),
        )


def test_two_plane_oracle_oblique_active_switch_matches_energy_and_force_differences():
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    normals = np.asarray(((0.8, 0.6, 0.0), (-0.8, 0.6, 0.0)))
    gaps = np.column_stack((
        0.8 * (vertices[:, 0] - 0.45) + 0.6 * vertices[:, 1] - 0.2,
        -0.8 * (vertices[:, 0] - 0.45) + 0.6 * vertices[:, 1] - 0.2,
    )).T
    reference = two_plane_triangle_contact(gaps, normals, penalty=13.0)
    assert reference["potential"] > 0.0
    assert np.linalg.norm(reference["feature_tangent"]) > 0.0
    direction = np.random.default_rng(2026).normal(size=(3, 3))
    change = np.einsum("ai,fi->fa", direction, normals)
    for h in (1e-5, 1e-6):
        plus = two_plane_triangle_contact(gaps + h * change, normals, penalty=13.0)
        minus = two_plane_triangle_contact(gaps - h * change, normals, penalty=13.0)
        assert (plus["potential"] - minus["potential"]) / (2 * h) == pytest.approx(
            np.sum(reference["nodal_residual"] * direction), rel=2e-8,
        )
        np.testing.assert_allclose(
            ((plus["nodal_residual"] - minus["nodal_residual"]) / (2 * h)).reshape(-1),
            reference["nodal_tangent"] @ direction.reshape(-1),
            rtol=1e-7, atol=1e-9,
        )


def test_whole_face_three_point_rule_can_miss_active_feature_strip():
    center, width = 0.3, 0.05
    gaps = np.asarray((
        (-center - width, 1 - center - width, -center - width),
        (center - width, center - 1 - width, center - width),
    ))
    shapes = np.asarray(((2 / 3, 1 / 6, 1 / 6),
                         (1 / 6, 2 / 3, 1 / 6),
                         (1 / 6, 1 / 6, 2 / 3)))
    sampled = np.max(gaps @ shapes.T, axis=0)
    assert np.all(sampled > 0.0)
    reference = two_plane_triangle_contact(gaps, ((1, 0, 0), (-1, 0, 0)))
    assert reference["potential"] == pytest.approx(width**3 * (1 - center) / 3)
    assert reference["potential"] > 0.0


def test_two_plane_spring_supported_triangle_global_newton_and_independent_integration():
    """A nine-DOF global fixture checks the contact Hessian after assembly.

    This is a spring-supported surface, not a continuum J2 or Abaqus result.
    """

    center, width, penalty = 0.45, 0.2, 40.0
    gaps = np.asarray((
        (-center - width, 1 - center - width, -center - width),
        (center - width, center - 1 - width, center - width),
    ))
    normals = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    graph = np.asarray(((2, -1, -1), (-1, 2, -1), (-1, -1, 2)))
    stiffness = 30 * np.eye(9) + 2 * np.kron(graph, np.eye(3))
    load = np.asarray((0.3, 0, 0, -0.2, 0, 0, 0.1, 0, 0))

    def evaluate(displacement):
        current_gaps = gaps + np.einsum(
            "ai,fi->fa", displacement.reshape(3, 3), normals,
        )
        contact = two_plane_triangle_contact(
            current_gaps, normals, penalty=penalty,
        )
        residual = stiffness @ displacement - load + contact["nodal_residual"].reshape(-1)
        tangent = stiffness + contact["nodal_tangent"]
        potential = (
            0.5 * displacement @ stiffness @ displacement
            - load @ displacement + contact["potential"]
        )
        return residual, tangent, potential, contact, current_gaps

    displacement = np.zeros(9)
    for _ in range(12):
        residual, tangent, _, _, _ = evaluate(displacement)
        if np.linalg.norm(residual) < 1e-12:
            break
        displacement -= np.linalg.solve(tangent, residual)
    residual, tangent, potential, contact, current_gaps = evaluate(displacement)
    assert np.linalg.norm(residual) < 1e-12
    assert np.linalg.eigvalsh(tangent).min() > 0.0
    assert contact["potential"] > 0.0
    assert np.linalg.norm(contact["feature_tangent"]) > 0.0
    direction = np.random.default_rng(24).normal(size=9)
    h = 1e-7
    plus = evaluate(displacement + h * direction)
    minus = evaluate(displacement - h * direction)
    np.testing.assert_allclose(
        (plus[0] - minus[0]) / (2 * h), tangent @ direction,
        rtol=1e-7, atol=2e-9,
    )
    assert (plus[2] - minus[2]) / (2 * h) == pytest.approx(
        residual @ direction, abs=1e-9,
    )
    # Independent nested one-dimensional integration. At each fixed x, split
    # the y interval at scalar zero and feature-switch roots; use ordinary
    # Gauss points on those intervals, never the oracle's clipped polygons or
    # feature-line tangent. This matters because unsplit Gauss converges much
    # slower for the discontinuous nodal force than for continuous energy.
    coordinates, weights = np.polynomial.legendre.leggauss(32)
    independent_energy = 0.0
    independent_force = np.zeros((3, 3))
    x_breaks = [0.0, 1.0]
    for values in (*current_gaps, current_gaps[0] - current_gaps[1]):
        for intercept, slope in (
            (values[0], values[1] - values[0]),
            (values[2], values[1] - values[2]),
        ):
            if abs(slope) > 1e-14:
                root = -intercept / slope
                if 0 < root < 1:
                    x_breaks.append(root)
    for x_low, x_high in zip(sorted(x_breaks)[:-1], sorted(x_breaks)[1:], strict=True):
        if x_high - x_low < 1e-15:
            continue
        x_nodes = 0.5 * (x_high + x_low) + 0.5 * (x_high - x_low) * coordinates
        x_weights = 0.5 * (x_high - x_low) * weights
        for x, x_weight in zip(x_nodes, x_weights, strict=True):
            upper = 1 - x
            starts = current_gaps[:, 0] + x * (current_gaps[:, 1] - current_gaps[:, 0])
            slopes = current_gaps[:, 2] - current_gaps[:, 0]
            cuts = [0.0, upper]
            for start, slope in (
                (starts[0], slopes[0]), (starts[1], slopes[1]),
                (starts[0] - starts[1], slopes[0] - slopes[1]),
            ):
                if abs(slope) > 1e-14:
                    root = -start / slope
                    if 0 < root < upper:
                        cuts.append(root)
            for low, high in zip(sorted(cuts)[:-1], sorted(cuts)[1:], strict=True):
                if high - low < 1e-15:
                    continue
                for sign in (-1, 1):
                    y = 0.5 * (high + low) + sign * (high - low) / (2 * np.sqrt(3))
                    shape = np.asarray((1 - x - y, x, y))
                    sampled = current_gaps @ shape
                    feature = int(np.argmax(sampled))
                    gap = sampled[feature]
                    if gap >= 0.0:
                        continue
                    weight = x_weight * 0.5 * (high - low)
                    independent_energy += 0.5 * penalty * weight * gap**2
                    independent_force += penalty * weight * gap * shape[:, None] * normals[feature]
    assert contact["potential"] == pytest.approx(independent_energy, rel=1e-12)
    np.testing.assert_allclose(
        contact["nodal_residual"], independent_force, rtol=1e-12, atol=1e-12,
    )
    assert np.isfinite(potential)


def test_two_plane_partition_with_one_continuum_tetrahedron_equilibrates():
    """One linear-elastic P1 tetrahedron supplies a continuum, not spring, K.

    Its base vertex is fixed and transverse displacements are constrained;
    the three top-face x displacements remain free. This is a deliberately
    small manufactured FE problem, not the finite-J2 closed-tool Step.
    """

    young, poisson = 200.0, 0.3
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    shear = young / (2 * (1 + poisson))
    gradients = np.asarray(((-1, -1, 1), (1, 0, 0), (0, 1, 0)))
    volume = 1 / 6
    stiffness = volume * (
        (lame + 2 * shear) * np.outer(gradients[:, 0], gradients[:, 0])
        + shear * np.outer(gradients[:, 1], gradients[:, 1])
        + shear * np.outer(gradients[:, 2], gradients[:, 2])
    )
    assert stiffness.sum() == pytest.approx(volume * shear)
    assert np.linalg.eigvalsh(stiffness).min() > 0.0
    center, width, penalty = 0.45, 0.18, 40.0
    gaps = np.asarray((
        (-center - width, 1 - center - width, -center - width),
        (center - width, center - 1 - width, center - width),
    ))
    normals = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    external = np.asarray((0.03, -0.02, 0.01))
    x_dofs = np.asarray((0, 3, 6))

    def evaluate(u):
        current = gaps + u[None, :] * normals[:, :1]
        contact = two_plane_triangle_contact(current, normals, penalty=penalty)
        residual = stiffness @ u - external + contact["nodal_residual"][:, 0]
        tangent = stiffness + contact["nodal_tangent"][np.ix_(x_dofs, x_dofs)]
        return residual, tangent, contact

    displacement = np.zeros(3)
    for _ in range(10):
        residual, tangent, _ = evaluate(displacement)
        if np.linalg.norm(residual) < 1e-12:
            break
        displacement -= np.linalg.solve(tangent, residual)
    residual, tangent, contact = evaluate(displacement)
    assert np.linalg.norm(residual) < 1e-12
    assert np.linalg.eigvalsh(tangent).min() > 0.0
    assert contact["potential"] > 0
    assert np.linalg.norm(contact["feature_tangent"]) > 0
    direction = np.asarray((0.25, -0.32, 0.17))
    h = 1e-7
    plus, _, _ = evaluate(displacement + h * direction)
    minus, _, _ = evaluate(displacement - h * direction)
    np.testing.assert_allclose(
        (plus - minus) / (2 * h), tangent @ direction,
        rtol=1e-7, atol=1e-9,
    )


def test_manufactured_tetrahedral_stiffness_matches_dolfinx_bulk_operator():
    basix_ufl = pytest.importorskip("basix.ufl")
    ufl = pytest.importorskip("ufl")
    dolfinx = pytest.importorskip("dolfinx")
    fem_petsc = pytest.importorskip("dolfinx.fem.petsc")
    mpi = pytest.importorskip("mpi4py.MPI")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
    domain = dolfinx.mesh.create_mesh(
        mpi.COMM_SELF, np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        ufl.Mesh(basix_ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
        vertices,
    )
    space = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    young, poisson = 200.0, 0.3
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    shear = young / (2 * (1 + poisson))
    form = (
        (lame + 2 * shear) * trial.dx(0) * test.dx(0)
        + shear * trial.dx(1) * test.dx(1)
        + shear * trial.dx(2) * test.dx(2)
    ) * ufl.dx(domain=domain)
    matrix = fem_petsc.assemble_matrix(dolfinx.fem.form(form))
    matrix.assemble()
    dof_positions = space.tabulate_dof_coordinates()
    selected = [
        int(np.flatnonzero(np.all(np.isclose(dof_positions, point), axis=1))[0])
        for point in vertices[:3]
    ]
    assembled = matrix.getValues(selected, selected)
    gradients = np.asarray(((-1, -1, -1), (1, 0, 0), (0, 1, 0)))
    exact = (
        (lame + 2 * shear) * np.outer(gradients[:, 0], gradients[:, 0])
        + shear * np.outer(gradients[:, 1], gradients[:, 1])
        + shear * np.outer(gradients[:, 2], gradients[:, 2])
    ) / 6
    np.testing.assert_allclose(assembled, exact, rtol=1e-12, atol=1e-12)
