# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent two-block series-compliance oracle for serial P1 interfaces."""

import basix.ufl
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

from agentfem import (
    fracture,
    operators,
    models,
    studies,
    fields,
    constitutive,
    interfaces,
)
from agentfem._elastic_cohesive import ElasticCohesiveLaw
from agentfem._interface_overlap import planar_overlap_pairing
from agentfem._interface_pairing import FixedReferenceCohesiveAssembler
from agentfem._nonmatching_force import NonmatchingCohesiveForce
from agentfem.boundary_models.rigid import TriangulatedRigidSurface


def _blocks(n, m, *, cell_type="tetrahedron", comm=MPI.COMM_SELF):
    parts = [
        mesh.create_box(
            MPI.COMM_SELF,
            [[0, 0, z], [1, 1, z + 1]],
            [size, size, size],
            getattr(mesh.CellType, cell_type),
        )
        for z, size in ((-1, n), (0, m))
    ]
    vertices = np.concatenate([p.geometry.x for p in parts])
    cells = np.concatenate(
        (
            parts[0].geometry.dofmaps[0],
            parts[1].geometry.dofmaps[0] + len(parts[0].geometry.x),
        )
    )
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_type, 1, shape=(3,)))
    return mesh.create_mesh(comm, cells.astype(np.int64) if comm.rank == 0 else np.empty((0, cells.shape[1]), dtype=np.int64),
                            domain, vertices if comm.rank == 0 else np.empty((0, 3)))


def _trace(space, positive):
    domain = space.mesh
    domain.topology.create_connectivity(2, 3)
    facets = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[2], 0))
    xyz = space.tabulate_dof_coordinates()
    triangles = []
    for facet in facets:
        cell = domain.topology.connectivity(2, 3).links(facet)[0]
        cell_dofs = space.dofmap.cell_dofs(cell)
        if bool(xyz[cell_dofs, 2].mean() > 0) != positive:
            continue
        dofs = fem.locate_dofs_topological(
            space, 2, np.asarray([facet], dtype=np.int32)
        )
        points = xyz[dofs]
        normal_z = np.cross(points[1] - points[0], points[2] - points[0])[2]
        if (normal_z > 0) == positive:
            dofs = dofs[::-1]
        triangles.append(dofs)
    triangles = np.asarray(triangles)
    unique, inverse = np.unique(triangles, return_inverse=True)
    surface = TriangulatedRigidSurface(xyz[unique], inverse.reshape(-1, 3))
    return surface, unique


def test_ordinary_model_step_elastic_interface_result(tmp_path, monkeypatch):
    domain = _blocks(1, 2)
    model = models.create(study=studies.static_solid(dimension=3), mesh=domain)
    target = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(young=100.0, poisson=0.0, density=1.0)
    )
    model.fix(target, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(0, 1))
    model.fix(target, on=lambda x: np.isclose(x[2], -1), components=2)
    model.fix(target, on=lambda x: np.isclose(x[2], 1), components=2, value=0.02)
    a, na = _trace(target.value.function_space, False)
    b, nb = _trace(target.value.function_space, True)
    pair = interfaces.pair_nonmatching_triangles(a, b, tolerance=1e-10)
    law = interfaces.elastic_cohesive(normal_stiffness=1000, tangential_stiffness=500)
    force = fracture.nonmatching_cohesive_force(
        pair, target, law, negative_dofs=na, positive_dofs=nb
    )
    step = model.step(target=target, cohesive_force=force)
    result = step.solve_result(output=tmp_path / "interface.xdmf")
    assert abs(result.quantities["energy_balance_residual"].value) < 1e-12
    assert result.quantities["free_residual_norm"].value < 1e-10
    assert result.quantities["interface_stored_energy"].value == pytest.approx(
        (0.02 / 0.021) ** 2 / 2000
    )
    assert (tmp_path / "interface.xdmf").exists()
    with pytest.raises(ValueError, match="overrides"):
        model.step(target=target, cohesive_force=force, K=model.stiffness(target))
    with pytest.raises(ValueError, match="U and RF"):
        step.solve_result(field_variables=("S",))
    minimal = step.solve_result(field_variables=())
    assert set(minimal.fields) == {"U"}
    saved = target.value.x.array.copy()
    saved_state = force.snapshot()
    from agentfem import solvers

    def fail_solve(matrix, rhs, solution, *args, **kwargs):
        solution.set(123.0)
        raise RuntimeError("forced solver failure")

    monkeypatch.setattr(solvers, "solve_matrix_system", fail_solve)
    with pytest.raises(RuntimeError, match="forced"):
        step.solve()
    np.testing.assert_array_equal(target.value.x.array, saved)
    assert force.snapshot() == saved_state


@pytest.mark.parametrize("n,m", [(1, 1), (1, 2), (2, 3)])
def test_two_block_series_compliance(n, m):
    domain = _blocks(n, m)
    space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(space)
    a, na = _trace(space, False)
    b, nb = _trace(space, True)
    pairing = planar_overlap_pairing(a, b, tolerance=1e-10)
    law = ElasticCohesiveLaw(1000, 500)
    assembler = FixedReferenceCohesiveAssembler(pairing, law, tangential="mixed")
    force = NonmatchingCohesiveForce(
        assembler, displacement, negative_dofs=na, positive_dofs=nb
    )
    young = 100.0
    v = ufl.TestFunction(space)
    du = ufl.TrialFunction(space)
    # Poisson zero gives an exact uniaxial solution without lateral restraints.
    expression = (
        young
        * ufl.inner(ufl.sym(ufl.grad(displacement)), ufl.sym(ufl.grad(v)))
        * ufl.dx
    )
    bulk = operators.from_ufl(expression, role="residual", name="bulk_residual")
    tangent = operators.from_ufl(
        ufl.derivative(expression, displacement, du), role="matrix", name="bulk_tangent"
    )
    residual = fracture.FiniteStrainCohesiveResidual(bulk, force)
    matrix = residual.assemble_matrix(tangent)
    rhs = displacement.x.petsc_vec.duplicate()
    rhs.set(0)
    coords = space.tabulate_dof_coordinates()
    bottom = np.flatnonzero(np.isclose(coords[:, 2], -1))
    top = np.flatnonzero(np.isclose(coords[:, 2], 1))
    # Suppress only transverse components plus bottom axial translation.
    fixed = np.unique(
        np.concatenate(
            (
                3 * np.arange(len(coords)),
                3 * np.arange(len(coords)) + 1,
                3 * bottom + 2,
                3 * top + 2,
            )
        )
    ).astype(PETSc.IntType)
    prescribed = displacement.x.petsc_vec.duplicate()
    prescribed.set(0)
    prescribed.array[3 * top + 2] = 0.02
    matrix.zeroRowsColumns(fixed, diag=1, x=prescribed, b=rhs)
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    try:
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(rhs, displacement.x.petsc_vec)
        assert ksp.getConvergedReason() > 0
        reaction = residual.assemble_vector()
        try:
            expected_force = 0.02 / (2 / young + 1 / law.normal_stiffness)
            assert reaction.array[3 * top + 2].sum() == pytest.approx(
                expected_force, rel=1e-10
            )
            free = np.setdiff1d(np.arange(len(reaction.array)), fixed)
            assert np.linalg.norm(reaction.array[free]) < 1e-10
            response = force.begin()
            np.testing.assert_allclose(
                response.jump[:, 2], expected_force / 1000, atol=1e-12
            )
            assert response.stored_energy == pytest.approx(
                expected_force**2 / 2000, rel=1e-10
            )
            force.commit()
        finally:
            reaction.destroy()
    finally:
        ksp.destroy()
        matrix.destroy()
        rhs.destroy()
        prescribed.destroy()
