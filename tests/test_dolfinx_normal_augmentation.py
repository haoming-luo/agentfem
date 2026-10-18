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
    diagnose_fixed_trace_normal_constraint_rank,
    integrate_fixed_trace_augmented_normal,
)

from test_finite_j2_facet_contact import _patch
from test_finite_strain_j2_periodic import _periodic_j2_contact_step


@pytest.mark.parametrize("cells", [1, 2, 3])
def test_gauss_point_multiplier_rank_deficiency_grows_on_planar_p1_mesh(cells):
    """An AL point law is not a stable multiplier *space* by itself.

    Each P1 slave triangle contributes three Gaussian multipliers but shares
    its normal displacement nodes with neighboring triangles. A vertex-lumped
    coalesced trace removes this exact row redundancy; that still is not an
    inf-sup or pressure-convergence proof for a global contact method.
    """

    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, cells, cells, cells,
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    space = fem.functionspace(
        domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    facets = dolfinx_mesh.locate_entities_boundary(
        domain, 2, lambda x: np.isclose(x[0], 0.0),
    )
    tags = dolfinx_mesh.meshtags(
        domain, 2, facets, np.full(len(facets), 7, dtype=np.int32),
    )
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    surface = boundary_models.rigid_plane(point=(0.1, 0, 0), normal=(1, 0, 0))
    pair = boundary_models.rigid_contact_pair(
        slave, boundary_models.rigid_body(surface), penalty=10.0,
    )
    gauss = dolfinx_boundary_region_contact_trace(slave, space)
    nodal = dolfinx_boundary_region_contact_trace(
        slave, space, integration="nodal_lumped",
    )
    nodal = boundary_models.RigidContactTracePolicy(
        "nodal_lumped", coalesce_nodes=True,
    ).apply(nodal, pair)

    def report(adapter):
        trace = adapter.trace
        state = FixedTraceNormalMultiplierState(
            trace,
            pair_identity=pair.name,
            reference_evaluation=trace.evaluate(adapter.reference_nodal_positions),
            surface_fingerprint=surface.geometry_fingerprint,
            penalty=10.0,
            penetration_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            maximum_augmentations=4,
            length_unit="m", pressure_unit="Pa",
        )
        evaluation = trace.evaluate(adapter.reference_nodal_positions)
        projection = surface.project(evaluation.query_points)
        record = boundary_models.ContactProjectionRecord(trace.point_ids, projection)
        return diagnose_fixed_trace_normal_constraint_rank(
            state.evaluate(evaluation, record),
        )

    gauss_rank = report(gauss)
    nodal_rank = report(nodal)
    normal_nodes = (cells + 1) ** 2
    assert gauss_rank["active_point_count"] == 6 * cells**2
    assert gauss_rank["row_rank"] == normal_nodes
    assert gauss_rank["row_nullity"] == 6 * cells**2 - normal_nodes
    assert nodal_rank["active_point_count"] == normal_nodes
    assert nodal_rank["row_rank"] == normal_nodes
    assert nodal_rank["row_nullity"] == 0
    assert not nodal_rank["certifies_spatial_stability"]


def test_actual_periodic_mpc_reduction_makes_even_nodal_multipliers_redundant():
    """A good unreduced trace does not certify the constrained global system.

    This is a controlled all-active algebraic screen, not a claimed physical
    pressure field or an augmented-Lagrangian periodic Step.
    """

    fixture, periodicity, step = _periodic_j2_contact_step()
    provider = step.contact_provider
    transformation = periodicity.reduction(1.0).matrix(fixture.domain.comm)
    dense = transformation.convert("dense")
    try:
        actual_reduction = dense.getDenseArray().copy()
    finally:
        dense.destroy()
        transformation.destroy()

    def rank(adapter):
        provider.adapter = adapter
        trace = adapter.trace
        evaluation, record, _, _ = provider._current_point_response()
        state = FixedTraceNormalMultiplierState(
            trace, pair_identity=provider.name,
            reference_evaluation=trace.evaluate(adapter.reference_nodal_positions),
            surface_fingerprint=provider.surface.geometry_fingerprint,
            penalty=5000.0, penetration_tolerance=1e-6,
            complementarity_tolerance=1e-6, maximum_augmentations=4,
            length_unit="m", pressure_unit="Pa",
            initial_multiplier=np.full(trace.point_count, 1e6),
        )
        response = state.evaluate(evaluation, record)
        return (
            diagnose_fixed_trace_normal_constraint_rank(response),
            diagnose_fixed_trace_normal_constraint_rank(
                response, displacement_reduction=actual_reduction,
            ),
        )

    gauss, reduced_gauss = rank(provider.adapter)
    nodal_adapter = dolfinx_boundary_region_contact_trace(
        provider.pair.slave_boundary,
        step.solution.function_space,
        integration="nodal_lumped",
    )
    nodal_adapter = boundary_models.RigidContactTracePolicy(
        "nodal_lumped", coalesce_nodes=True,
    ).apply(nodal_adapter, provider.pair)
    nodal, reduced_nodal = rank(nodal_adapter)
    assert (gauss["active_point_count"], gauss["row_nullity"]) == (24, 2)
    assert (reduced_gauss["row_rank"], reduced_gauss["row_nullity"]) == (9, 15)
    assert (nodal["active_point_count"], nodal["row_nullity"]) == (9, 0)
    assert (reduced_nodal["row_rank"], reduced_nodal["row_nullity"]) == (5, 4)
    assert not reduced_nodal["certifies_spatial_stability"]


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


@pytest.mark.parametrize("trace_kind", ["gauss", "coalesced_nodal"])
def test_fixed_trace_outer_law_can_balance_a_plastic_global_j2_trial(
    monkeypatch, trace_kind,
):
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
    if trace_kind == "coalesced_nodal":
        adapter = dolfinx_boundary_region_contact_trace(
            provider.pair.slave_boundary,
            step.solution.function_space,
            integration="nodal_lumped",
        )
        provider.adapter = boundary_models.RigidContactTracePolicy(
            "nodal_lumped", coalesce_nodes=True,
        ).apply(adapter, provider.pair)
    trace = provider.adapter.trace
    state = FixedTraceNormalMultiplierState(
        trace, pair_identity=provider.name,
        reference_evaluation=trace.evaluate(provider.adapter.reference_nodal_positions),
        surface_fingerprint=provider.surface.geometry_fingerprint,
        penalty=3000.0, penetration_tolerance=1e-6,
        complementarity_tolerance=1e-6, maximum_augmentations=20,
        length_unit="m", pressure_unit="Pa",
    )
    rank_state = FixedTraceNormalMultiplierState(
        trace, pair_identity=provider.name,
        reference_evaluation=trace.evaluate(provider.adapter.reference_nodal_positions),
        surface_fingerprint=provider.surface.geometry_fingerprint,
        penalty=3000.0, penetration_tolerance=1e-6,
        complementarity_tolerance=1e-6, maximum_augmentations=20,
        length_unit="m", pressure_unit="Pa",
        initial_multiplier=np.ones(trace.point_count),
    )
    reference = trace.evaluate(provider.adapter.reference_nodal_positions)
    reference_record = boundary_models.ContactProjectionRecord(
        trace.point_ids, provider.surface.project(reference.query_points),
    )
    rank = diagnose_fixed_trace_normal_constraint_rank(
        rank_state.evaluate(reference, reference_record),
    )
    assert (rank["row_nullity"] == 0) == (trace_kind == "coalesced_nodal")
    fixed = np.unique(np.concatenate([
        np.asarray(bc.dof_indices()[0], dtype=np.int64) for bc in step.bcs
    ]))
    all_dofs = np.arange(step.solution.x.array.size, dtype=np.int64)
    actual_reduction = np.eye(len(all_dofs))[:, np.setdiff1d(all_dofs, fixed)]
    reduced_rank = diagnose_fixed_trace_normal_constraint_rank(
        rank_state.evaluate(reference, reference_record),
        displacement_reduction=actual_reduction,
    )
    assert reduced_rank["row_nullity"] == rank["row_nullity"]

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
    maximum_peeq = float(np.max(step.response.state.committed[
        "equivalent_plastic_strain"
    ].function.x.array))
    # Nodal and Gaussian traces are distinct scientific discretizations;
    # successful plastic response does not imply identical PEEQ at this mesh.
    assert maximum_peeq > (0.005 if trace_kind == "gauss" else 0.0)


@pytest.mark.parametrize("cells", [1, 2])
def test_coalesced_nodal_al_reaches_independent_hard_plane_j2_limit(
    monkeypatch, cells,
):
    """Research-only whole-increment probe against a scalar hard-contact oracle.

    This is deliberately not an ordinary Step: it tests the proposed
    multiplier space and path on a homogeneous plane, while leaving global
    Procedure ownership, checkpoint and energy admission open.
    """

    indentation = 0.002
    step = _patch(
        triangulated=True, cells=cells, gap=0.0, penalty=3000.0,
        tool_translation=(-indentation, 0.0, 0.0),
        applied_traction=0.0, increments=1, yield_stress=0.1,
        transverse_boundary_fixed=True,
    )[3]
    provider = step.contact_provider
    adapter = dolfinx_boundary_region_contact_trace(
        provider.pair.slave_boundary, step.solution.function_space,
        integration="nodal_lumped",
    )
    provider.adapter = boundary_models.RigidContactTracePolicy(
        "nodal_lumped", coalesce_nodes=True,
    ).apply(adapter, provider.pair)
    trace = provider.adapter.trace
    state = FixedTraceNormalMultiplierState(
        trace, pair_identity=provider.name,
        reference_evaluation=trace.evaluate(provider.adapter.reference_nodal_positions),
        surface_fingerprint=provider.surface.geometry_fingerprint,
        penalty=3000.0, penetration_tolerance=1e-6,
        complementarity_tolerance=1e-6, maximum_augmentations=40,
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
        )

    monkeypatch.setattr(provider, "evaluate", evaluate_augmented)
    step._apply_loading(0.0)
    step.state_transaction.initialize()
    step._apply_loading(1.0)
    equilibria = 0
    for _ in range(state.maximum_augmentations + 1):
        info = step._solve_increment(
            increment=1, attempt=1, start_factor=0.0, target_factor=1.0,
        )
        assert info.converged, info.rejection_reason
        equilibria += 1
        evaluation, record = current()
        state.mark_equilibrium_converged(
            evaluation, record,
            total_residual_norm=info.residual_norm,
            total_residual_tolerance=(
                step.solver_options.absolute_tolerance
                + step.solver_options.relative_tolerance * info.initial_residual_norm
            ),
        )
        diagnostic = state.acceptance_diagnostics(evaluation, record)
        if (diagnostic["maximum_penetration"] <= state.penetration_tolerance
                and diagnostic["maximum_complementarity"]
                <= state.complementarity_tolerance):
            state.commit(evaluation, record)
            step.state_transaction.commit_increment(
                start_factor=0.0, target_factor=1.0,
            )
            break
        state.augment_after_converged(evaluation, record)
    else:
        pytest.fail("Fixed-trace AL failed to meet its declared outer tolerances.")

    stretch = 1.0 - indentation
    young, yield_stress, hardening = 1000.0, 0.1, 100.0
    plastic_increment = max(
        0.0,
        (-young * np.log(stretch) - yield_stress)
        / (1.5 * young + hardening),
    )
    expected_pressure = -young * (np.log(stretch) + plastic_increment) / stretch
    coordinates = step.solution.function_space.tabulate_dof_coordinates()
    right = np.isclose(coordinates[:, 0], 1.0)
    right_displacement = step.solution.x.array.reshape((-1, 3))[right, 0]
    np.testing.assert_allclose(
        right_displacement, -indentation, rtol=0.0, atol=1e-6,
    )
    assert np.max(step.response.state.committed[
        "equivalent_plastic_strain"
    ].function.x.array) == pytest.approx(plastic_increment, abs=2e-6)
    assert -evaluate_augmented().contact_force_on_structure[0] == pytest.approx(
        expected_pressure, abs=2e-3,
    )
    assert 1 < equilibria <= state.maximum_augmentations
