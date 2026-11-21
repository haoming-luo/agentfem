# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent interior DOFs: global inertia, time integration and exact output."""

import numpy as np
import pytest
from mpi4py import MPI
from agentfem import mesh, fields, models, studies, elements, constitutive, results


def build(comm=MPI.COMM_SELF, steps=8, mean=True):
    d = mesh.cuboid((0, 0, 0), (1, 1, 1), (1, 1, 1), comm=comm, cell_type="tetrahedron")
    m = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=d
    )
    u = m.field(
        fields.displacement(
            d,
            element_policy=elements.composite_tet10_policy(
                independent_center=True, mean_dilatation=mean
            ),
        )
    )
    m.material(constitutive.isotropic_elastic(young=100, poisson=0.25, density=2))
    m.clamp(u, on=mesh.boundary_region(d, lambda x: np.isclose(x[0], 0)))
    m.nodal_force((1, 0.2, -0.1), at=(1, 1, 1))
    return m.step(target=u, dt=0.001, steps=steps, save_every=1, progress=False)


@pytest.mark.parametrize("mean", [False, True])
def test_internal_inertia_and_exact_visualization(tmp_path, mean):
    import h5py

    s = build(mean=mean)
    op = s.residual.internal
    assert op.cell_nodes.shape == (6, 11)
    assert op.mass_diagonal.sum() / 3 == pytest.approx(2)
    s.solve_result(output=tmp_path / "internal.xdmf", field_variables=("S", "SENER"))
    u = s.state.u.x.array.reshape(-1, 3)
    assert (
        np.max(np.abs(u[op.cell_nodes[:, 10]] - u[op.cell_nodes[:, 4:10]].mean(1)))
        > 1e-10
    )
    with h5py.File(tmp_path / "internal.h5") as h:
        assert h["Mesh/SourceCellDofs"].shape == (6, 11)
        assert h["Frames/0008/Native/U"].shape == (33, 3)
        np.testing.assert_array_equal(
            h["Frames/0008/Node/U"][:], h["Frames/0008/Native/U"][:]
        )
    xyz = s.state.u.function_space.tabulate_dof_coordinates()
    point = xyz[op.cell_nodes[0, 10]]
    np.testing.assert_allclose(
        results.probe(s.state.u, at=point), u[op.cell_nodes[0, 10]], atol=1e-14
    )


def test_internal_restart_and_global_elastic_reference(tmp_path):
    from scipy.sparse import coo_matrix
    from scipy.linalg import eigh

    s = build(steps=3)
    op = s.residual.internal
    nd = len(op.mass_diagonal)
    dofs = (3 * op.cell_nodes[:, :, None] + np.arange(3)).reshape(-1, 33)
    shape = op.stiffness.shape
    K = coo_matrix(
        (
            op.stiffness.ravel(),
            (
                np.broadcast_to(dofs[:, :, None], shape).ravel(),
                np.broadcast_to(dofs[:, None, :], shape).ravel(),
            ),
        ),
        shape=(nd, nd),
    ).toarray()
    xyz = s.state.u.function_space.tabulate_dof_coordinates()
    fixed = np.repeat(np.isclose(xyz[:, 0], 0), 3)
    free = np.flatnonzero(~fixed)
    eig = eigh(
        K[np.ix_(free, free)], np.diag(op.mass_diagonal[free]), eigvals_only=True
    )
    assert eig.min() > 0
    assert op.stable_dt() <= 2 / np.sqrt(eig.max())
    # Independently advance all boundary AND interior DOFs using the global M,K.
    f = s.residual.external.assemble_vector()
    force = f.array.copy()
    f.destroy()
    u = np.zeros(nd)
    v = np.zeros(nd)
    a = force / op.mass_diagonal
    a[fixed] = 0
    dt = s.dt
    for _ in range(3):
        u += dt * v + 0.5 * dt * dt * a
        u[fixed] = 0
        new = (force - K @ u) / op.mass_diagonal
        new[fixed] = 0
        v += 0.5 * dt * (a + new)
        v[fixed] = 0
        a = new
    s.solve_result()
    np.testing.assert_allclose(s.state.u.x.array, u, atol=1e-16)
    first = build(steps=3)
    first.run(until_step=1)
    first.save_checkpoint(tmp_path / "restart")
    resumed = build(steps=3)
    resumed.load_checkpoint(tmp_path / "restart")
    resumed.solve_result()
    np.testing.assert_array_equal(s.state.u.x.array, resumed.state.u.x.array)


def test_internal_mpi(tmp_path):
    comm = MPI.COMM_WORLD
    s = build(comm)
    serial = build()
    path = comm.bcast(str(tmp_path / "mpi.xdmf"), root=0)
    s.solve_result(output=path, field_variables=("S",))
    serial.solve_result()
    x = s.state.u.function_space.tabulate_dof_coordinates()
    sx = serial.state.u.function_space.tabulate_dof_coordinates()
    n = len(s.residual.internal.mass_diagonal) // 3
    ids = np.argmin(np.linalg.norm(x[:n, None] - sx[None], axis=2), axis=1)
    np.testing.assert_allclose(
        s.state.u.x.array[: n * 3].reshape(-1, 3),
        serial.state.u.x.array.reshape(-1, 3)[ids],
        atol=1e-14,
    )
    for name in ("strain_energy", "hourglass_energy"):
        assert s.residual.internal.energies()[name] == pytest.approx(
            serial.residual.internal.energies()[name], abs=1e-14
        )
