"""Independent UFL operator oracle for the fixed-trace AL research candidate."""

from __future__ import annotations

import basix.ufl
from dolfinx import fem, mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import pytest
import ufl

from agentfem import boundary_models
from agentfem import mesh as agent_mesh
from agentfem.boundary_models.dolfinx_contact_trace import (
    dolfinx_boundary_region_contact_trace,
)
from agentfem.boundary_models.normal_augmentation import (
    FixedTraceNormalMultiplierState,
    integrate_fixed_trace_augmented_normal,
)

from test_finite_j2_facet_contact import _patch


@pytest.mark.parametrize("geometry", ["plane", "sphere"])
def test_fixed_multiplier_fem_operator_matches_independent_ufl(geometry):
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1, cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    space = fem.functionspace(
        domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    displacement = fem.Function(space)
    displacement.x.array.reshape(-1, 3)[:] = (-0.08, 0.0, 0.0)
    facets = dolfinx_mesh.locate_entities_boundary(
        domain, 2, lambda x: np.isclose(x[0], 0.0),
    )
    tags = dolfinx_mesh.meshtags(
        domain, 2, facets, np.full(len(facets), 7, dtype=np.int32),
    )
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    adapter = dolfinx_boundary_region_contact_trace(slave, space)
    trace = adapter.trace
    surface = (
        boundary_models.rigid_plane(point=(0, 0, 0), normal=(1, 0, 0))
        if geometry == "plane"
        else boundary_models.rigid_sphere(center=(-0.1, 0.5, 0.5), radius=1.0)
    )
    state = FixedTraceNormalMultiplierState(
        trace,
        pair_identity="test-dolfinx-rigid-pair",
        reference_evaluation=trace.evaluate(adapter.reference_nodal_positions),
        surface_fingerprint=surface.geometry_fingerprint,
        penalty=12.0,
        penetration_tolerance=1e-8,
        complementarity_tolerance=1e-8,
        maximum_augmentations=10,
        length_unit="m",
        pressure_unit="Pa",
        initial_multiplier=np.ones(trace.point_count),
    )
    evaluation = adapter.evaluate(displacement)
    record = boundary_models.ContactProjectionRecord(
        trace.point_ids, surface.project(evaluation.query_points),
    )
    response = state.evaluate(evaluation, record)
    contribution = integrate_fixed_trace_augmented_normal(
        evaluation, record, response,
        surface.gap_differential(record.projection).hessian,
    )
    assert contribution.active_points == trace.point_count

    current = ufl.SpatialCoordinate(domain) + displacement
    gap = (
        current[0] if geometry == "plane" else
        ufl.sqrt(ufl.inner(current - ufl.as_vector((-0.1, 0.5, 0.5)),
                           current - ufl.as_vector((-0.1, 0.5, 0.5)))) - 1.0
    )
    pressure = ufl.max_value(0.0, 1.0 - 12.0 * gap)
    measure = ufl.Measure(
        "ds", domain=domain, subdomain_data=tags,
        metadata={"quadrature_degree": 2},
    )
    functional = (pressure**2 - 1.0) / 24.0 * measure(7)
    test = ufl.TestFunction(space)
    trial = ufl.TrialFunction(space)
    residual_form = fem.form(ufl.derivative(functional, displacement, test))
    tangent_form = fem.form(ufl.derivative(
        ufl.derivative(functional, displacement, test), displacement, trial,
    ))
    reference_functional = fem.assemble_scalar(fem.form(functional))
    reference_residual = fem.petsc.assemble_vector(residual_form)
    reference_tangent = fem.petsc.assemble_matrix(tangent_form)
    reference_tangent.assemble()
    direction = reference_residual.duplicate()
    action = reference_residual.duplicate()
    try:
        np.testing.assert_allclose(
            contribution.algorithmic_functional, reference_functional,
            rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            contribution.nodal_structural_residual.reshape(-1),
            reference_residual.array, rtol=1e-12, atol=1e-12,
        )
        direction.array[:] = np.linspace(-0.2, 0.3, direction.array.size)
        reference_tangent.mult(direction, action)
        np.testing.assert_allclose(
            contribution.tangent_action(direction.array.reshape(-1, 3)).reshape(-1),
            action.array, rtol=1e-11, atol=1e-10,
        )
    finally:
        action.destroy()
        direction.destroy()
        reference_tangent.destroy()
        reference_residual.destroy()


def test_fixed_trace_outer_law_can_balance_a_plastic_global_j2_trial(monkeypatch):
    """Research feasibility probe, not a public Step or atomic acceptance test.

    The ordinary J2 Newton owner solves each frozen-multiplier equilibrium.
    The test alone owns the outer loop; production still needs one Procedure,
    joint rollback/checkpoint, energy semantics, and a spatial-stability gate.
    """

    step = _patch(
        triangulated=True, gap=0.0, penalty=3000.0,
        tool_translation=(-0.005, 0.0, 0.0), applied_traction=0.0,
        increments=1, yield_stress=0.05,
    )[3]
    provider = step.contact_provider
    trace = provider.adapter.trace
    state = FixedTraceNormalMultiplierState(
        trace, pair_identity=provider.name,
        reference_evaluation=trace.evaluate(provider.adapter.reference_nodal_positions),
        surface_fingerprint=provider.surface.geometry_fingerprint,
        penalty=3000.0, penetration_tolerance=1e-6,
        complementarity_tolerance=1e-6, maximum_augmentations=20,
        length_unit="m", pressure_unit="Pa",
    )

    def current():
        evaluation, record, _, _ = provider._current_point_response()
        return evaluation, record

    def evaluate_augmented():
        evaluation, record = current()
        return integrate_fixed_trace_augmented_normal(
            evaluation, record, state.evaluate(evaluation, record),
            np.zeros((trace.point_count, 3, 3)),
            surface_reference_point=provider.pair.rigid_body.reference_point,
        )

    monkeypatch.setattr(provider, "evaluate", evaluate_augmented)
    step._apply_loading(0.0)
    step.state_transaction.initialize()
    step._apply_loading(1.0)
    accepted = False
    for _ in range(state.maximum_augmentations):
        info = step._solve_increment(
            increment=1, attempt=1, start_factor=0.0, target_factor=1.0,
        )
        assert info.converged, info.rejection_reason
        assert step.state_transaction.accepted_factor == 0.0
        evaluation, record = current()
        state.mark_equilibrium_converged(
            evaluation, record,
            total_residual_norm=info.residual_norm,
            total_residual_tolerance=(
                step.solver_options.absolute_tolerance
                + step.solver_options.relative_tolerance * info.initial_residual_norm
            ),
        )
        diagnostics = state.acceptance_diagnostics(evaluation, record)
        if (
            diagnostics["maximum_penetration"] <= state.penetration_tolerance
            and diagnostics["maximum_complementarity"]
            <= state.complementarity_tolerance
        ):
            state.commit(evaluation, record)
            step.state_transaction.commit_increment(
                start_factor=0.0, target_factor=1.0,
            )
            accepted = True
            break
        state.augment_after_converged(evaluation, record)
    assert accepted
    assert state.augmentation_count == 0
    assert step.state_transaction.accepted_factor == 1.0
    assert np.max(step.response.state.committed[
        "equivalent_plastic_strain"
    ].function.x.array) > 0.005
