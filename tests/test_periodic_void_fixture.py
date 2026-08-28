"""Geometry and exact-affine contracts for a true periodic void cell."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    models,
    results,
    solvers,
    steps,
    studies,
)

from periodic_void_fixture import periodic_spherical_void_cell


def test_gmsh_spherical_void_mesh_has_exact_periodic_source_equations():
    pytest.importorskip("gmsh")
    fixture = periodic_spherical_void_cell(MPI.COMM_SELF)
    displacement = fields.displacement(fixture.domain)
    periodicity = fixture.constraint(displacement)
    reduction = periodicity.reduction()

    global_cells = fixture.domain.topology.index_map(3).size_global
    global_vertices = fixture.domain.topology.index_map(0).size_global
    solid_volume = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=fixture.domain))
    )

    assert global_cells > 0
    assert reduction.full_size == 3 * global_vertices
    assert 0 < reduction.reduced_size < reduction.full_size
    assert fixture.periodic_pairing_error < 1.0e-13
    assert len(fixture.equations.equations) > 0
    assert periodicity.reference_cell_volume == pytest.approx(1.0)
    assert solid_volume == pytest.approx(fixture.exact_solid_volume, rel=8.0e-3)
    assert set(np.unique(fixture.cell_tags.values)) == {1}
    assert set(np.unique(fixture.facet_tags.values)) == {10, 20}

    periodicity.apply_affine_increment(0.0, 1.0)
    assert periodicity.mismatch() < 1.0e-12
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(displacement),
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_real_void_finite_strain_j2_uses_public_result_lifecycle():
    pytest.importorskip("gmsh")
    fixture = periodic_spherical_void_cell(
        MPI.COMM_SELF,
        mesh_size=0.25,
        stretch=1.004,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_real_void_smoke",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=2.0e-7,
            absolute_tolerance=1.0e-8,
            maximum_iterations=25,
            line_search="backtracking",
        ),
        quadrature_degree=1,
        output_every=1,
        progress=False,
    )

    result = step.solve_result()
    frames = results.homogenize_periodic_path(
        step.snapshots,
        step.state_transaction.material,
        constraint=periodicity,
    )
    hill_mandel = results.hill_mandel_periodic_path(
        step.snapshots,
        step.state_transaction.material,
        constraint=periodicity,
        frames=frames,
    )
    minimum_j = min(
        float(np.min(np.linalg.det(snapshot.fields["F"].owned_values)))
        for snapshot in step.snapshots
    )
    maximum_peeq = (
        step.response.state.committed["equivalent_plastic_strain"].global_max()
    )

    assert result.status == "completed"
    assert step.last_solve_info.converged
    assert step.accepted_load_factor == pytest.approx(1.0)
    assert periodicity.mismatch() < 1.0e-10
    assert minimum_j > 0.99
    assert maximum_peeq > 1.0e-4
    assert 0.95 < frames[-1].solid_reference_fraction < 0.99
    assert np.isfinite(frames[-1].first_piola_stress).all()
    assert frames[-1].first_piola_stress[0, 0] > 0.0
    assert max(item.relative_error for item in hill_mandel) < 1.0e-8
    assert {
        "Displacement",
        "F",
        "P",
        "S",
        "MISES",
        "SENER",
        "FP",
        "PEEQ",
    } <= set(result.fields)
