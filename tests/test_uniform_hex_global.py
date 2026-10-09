# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent full-integration affine oracle and analytic longitudinal wave."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import state
from agentfem.elements._uniform_hex_dolfinx import UniformHexResidual
from agentfem.time.explicit import central_difference


def _problem(n):
    domain = mesh.create_box(
        MPI.COMM_SELF,
        [[0, 0, 0], [1, 0.2, 0.2]],
        [n, 2, 2],
        cell_type=mesh.CellType.hexahedron,
    )
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u = fem.Function(space)
    c = np.diag([100, 100, 100, 50, 50, 50])  # E=100, nu=0
    op = UniformHexResidual(
        u, c, density=2, hourglass_modulus=50, hourglass_scale=0.1, chunk_size=7
    )
    return domain, space, u, op


def test_global_hex_affine_force_and_mass_match_full_integration():
    from dolfinx.fem import petsc

    domain, space, u, op = _problem(3)
    u.interpolate(
        lambda x: np.vstack((0.01 * x[0] + 0.03 * x[2], 0.02 * x[1], -0.01 * x[2]))
    )
    v = ufl.TestFunction(space)
    reference = petsc.assemble_vector(
        fem.form(ufl.inner(100 * ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx)
    )
    actual = op.assemble_vector()
    try:
        np.testing.assert_allclose(actual.array, reference.array, atol=1e-13)
        force = actual.array.reshape(-1, 3)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-13)
        moment = np.cross(space.tabulate_dof_coordinates(), force).sum(axis=0)
        np.testing.assert_allclose(moment, 0, atol=1e-13)
    finally:
        actual.destroy()
        reference.destroy()
    assert op.mass_diagonal[::3].sum() == pytest.approx(2 * 0.04)
    assert op.energies()["hourglass_energy"] < 1e-26
    expected = fem.assemble_scalar(
        fem.form(50 * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(u))) * ufl.dx)
    )
    assert op.energies()["strain_energy"] == pytest.approx(expected)


def test_global_chunk_gather_matches_full_gather_and_rejects_bad_maps():
    _, _, u, op = _problem(3)
    values = u.x.array.reshape(-1, 3)
    values[:] = np.random.default_rng(123).normal(size=values.shape)
    full = list(op.cells.iter_responses(values[op.cell_nodes]))
    chunks = list(op.cells.iter_responses(values, node_map=op.cell_nodes))
    for (region, expected), (actual_region, actual) in zip(full, chunks, strict=True):
        assert region == actual_region
        assert region.stop - region.start <= op.cells.chunk_size
        for name in (
            "strain",
            "stress",
            "internal_force",
            "physical_energy",
            "hourglass_energy",
        ):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )
    invalid = op.cell_nodes.copy()
    invalid[0, 0] = len(values)
    with pytest.raises(ValueError, match="node map"):
        list(op.cells.iter_responses(values, node_map=invalid))


def test_global_hex_wave_convergence_uses_existing_integrator():
    errors = []
    for n in (8, 16):
        _, space, u, op = _problem(n)
        amplitude = 1e-4
        x = space.tabulate_dof_coordinates()[:, 0]
        shape = np.cos(np.pi * x)
        u.x.array[::3] = amplitude * shape
        history = state.second_order_state(u)
        residual = op.assemble_vector()
        try:
            history.a.value.x.array[:] = -residual.array * op.inv_mass
        finally:
            residual.destroy()
        integrator = central_difference(state=history, mass=op)
        omega = np.pi * np.sqrt(100 / 2)
        stop = np.pi / (4 * omega)
        steps = int(np.ceil(stop / op.stable_dt(safety=0.3)))
        dt = stop / steps
        initial = op.energies()["strain_energy"]
        for i in range(steps):
            integrator.step(dt, time=(i + 1) * dt, residual_operator=op)
        numerical = u.x.array[::3] / amplitude
        errors.append(float(np.max(np.abs(numerical - shape * np.cos(omega * stop)))))
        kinetic = 0.5 * np.sum(op.mass_diagonal * history.v.value.x.array**2)
        energies = op.energies()
        assert energies["hourglass_energy"] < 1e-25
        assert abs(kinetic + energies["strain_energy"] - initial) / initial < 0.002
    assert errors[1] < errors[0] / 3
    assert errors[1] < 0.002


def test_global_hex_rejects_tetrahedra_and_bad_safety():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))))
    with pytest.raises(ValueError, match="Q1 hexahedra"):
        UniformHexResidual(
            u, np.eye(6), density=1, hourglass_modulus=1, hourglass_scale=0.1
        )
    _, _, _, op = _problem(1)
    with pytest.raises(ValueError, match="safety"):
        op.stable_dt(safety=1.1)


@pytest.mark.parametrize("scale", [0.05, 0.1, 0.2])
def test_global_hex_cantilever_bending_refinement(scale):
    """Slender beam reference, not a claim of matching commercial C3D8R."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from agentfem.elements._uniform_hex import UniformHex8

    errors = []
    for level in (1, 2):
        domain = mesh.create_box(
            MPI.COMM_SELF,
            [[0, 0, 0], [10, 1, 1]],
            [20 * level, 2 * level, 4 * level],
            cell_type=mesh.CellType.hexahedron,
        )
        space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        u = fem.Function(space)
        c = np.diag([100, 100, 100, 50, 50, 50])
        op = UniformHexResidual(
            u, c, density=2, hourglass_modulus=50, hourglass_scale=scale
        )
        xyz = space.tabulate_dof_coordinates()
        rows, cols, entries = [], [], []
        for nodes in op.cell_nodes:
            cell = UniformHex8(
                xyz[nodes], c, density=2, hourglass_modulus=50, hourglass_scale=scale
            )
            dofs = (3 * nodes[:, None] + np.arange(3)).ravel()
            rows.extend(np.repeat(dofs, 24))
            cols.extend(np.tile(dofs, 24))
            entries.extend((cell.physical_matrix + cell.hourglass_matrix).ravel())
        ndofs = u.x.array.size
        matrix = coo_matrix((entries, (rows, cols)), shape=(ndofs, ndofs)).tocsr()
        # Integrate constant end traction through the independently assembled UFL load.
        facets = mesh.locate_entities_boundary(
            domain, 2, lambda x: np.isclose(x[0], 10)
        )
        tags = mesh.meshtags(domain, 2, facets, np.ones(len(facets), dtype=np.int32))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        from dolfinx.fem import petsc

        vector = petsc.assemble_vector(
            fem.form(ufl.TestFunction(space)[2] * 1e-4 * ds(1))
        )
        try:
            force = vector.array.copy()
        finally:
            vector.destroy()
        fixed = (
            3 * np.flatnonzero(np.isclose(xyz[:, 0], 0))[:, None] + np.arange(3)
        ).ravel()
        free = np.setdiff1d(np.arange(ndofs), fixed)
        u.x.array[free] = spsolve(matrix[free][:, free], force[free])
        reaction = op.assemble_vector()
        try:
            np.testing.assert_allclose(reaction.array[free], force[free], atol=1e-11)
        finally:
            reaction.destroy()
        tip = float(force @ u.x.array / 1e-4)
        # Euler-Bernoulli plus the rectangular-section Timoshenko shear correction.
        reference = 1e-4 * (10**3 / (3 * 100 * (1 / 12)) + 10 / ((5 / 6) * 50))
        error = abs(tip / reference - 1)
        errors.append(error)
        energy = op.energies()
        fraction = energy["hourglass_energy"] / sum(energy.values())
        print(
            dict(
                level=level,
                scale=scale,
                relative_error=error,
                hourglass_fraction=fraction,
            )
        )
    assert errors[1] < errors[0]
    assert errors[1] < 0.03
