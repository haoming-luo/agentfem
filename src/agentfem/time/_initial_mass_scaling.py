# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reference-operator mass preparation, not a nonlinear stability estimator.

Reuse element/penalty geometry and the common planner. No material is called,
no search projection is selected, and no trial/accepted State is mutated.
"""
import numpy as np

from ._nodal_inertia import plan_nodal_inertia, diagnose_nodal_increment
from ._scaled_inertia import FixedScaledInertia


def _deformable_reference_rows(pair, count):
    """Enclose any frozen interpolation in the initial search neighborhood.

    For point p and nonnegative Q1 weights a with sum(a)=1,
    ||v_p-sum(a_j*v_j)||² <= 2||v_p||² + 2sum(a_j||v_j||²).
    Assign 2*k*w to p and each vertex of every candidate patch. Duplicate
    vertices across patches are counted once per point. Broad-phase false
    positives only increase this reference bound. This is NOT the derivative
    of a moving projection and says nothing about future contact neighborhoods.
    """
    from .._q1_surface_search import Q1SurfaceSearch
    selected = np.flatnonzero(pair.candidate_nodes)
    primary = pair.primary
    search = Q1SurfaceSearch(primary.vertices[primary.quadrilaterals],
                             node_ids=primary.quadrilaterals)
    rows = np.zeros((count, 3))
    points_with_candidates = 0
    max_candidates = 0
    penalty = float(pair.penalty)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Initial contact planning requires positive finite penalty')
    scale = max(1., float(np.max(np.abs(primary.vertices))),
                float(np.max(np.abs(pair.secondary.vertices))))
    radius = pair.search_radius + search.padding + 128*np.finfo(float).eps*scale
    for point_id in selected:
        point = pair.secondary.vertices[point_id]
        facets = np.unique(search.tree.overlapping_facets(point-radius, point+radius) // 2)
        max_candidates = max(max_candidates, len(facets))
        if not len(facets):
            continue
        points_with_candidates += 1
        primary_ids = pair.primary_dofs[np.unique(primary.quadrilaterals[facets])]
        contribution = 2*penalty*pair.secondary.nodal_measures[point_id]
        rows[pair.secondary_dofs[point_id]] += contribution
        rows[primary_ids] += contribution
    if not np.isfinite(rows).all():
        raise ValueError('Deformable reference stiffness overflowed')
    return rows, {'candidate_points': len(selected),
                  'points_with_initial_neighborhood': points_with_candidates,
                  'maximum_candidate_facets': max_candidates,
                  'scope': 'frozen_Q1_interpolation_initial_search_neighborhood',
                  'moving_projection_geometric_tangent_included': False}


def lower_initial_mass_scaling(policy, internal, history, *, dt, cohesive,
                              rigid_adapters, deformable, viscosity):
    from .._penalty_stability import penalty_stencil_row_sums
    from ..elements._bulk_viscosity import bulk_viscosity_response
    if dt > policy.target_increment:
        raise ValueError('Step dt must not exceed the mass-scaling target increment')
    if np.any(history.u.value.x.array) or np.any(history.v.value.x.array):
        raise ValueError('Initial mass planning requires zero displacement and velocity')
    nodes = internal.cell_nodes
    physical = internal.mass_diagonal.reshape(-1, 3)[:, 0]
    count = len(physical)

    def assemble(values):
        if values.shape == nodes.shape:
            return np.repeat(np.bincount(nodes.ravel(), weights=values.ravel(),
                                          minlength=count)[:, None], 3, axis=1)
        return np.column_stack([np.bincount(nodes.ravel(), weights=values[:, :, j].ravel(),
                                           minlength=count) for j in range(3)])

    contributions = {'body_hourglass': assemble(
        internal.cells.tangent_envelope_cell_row_majorant(
            positive_modulus=policy.reference_modulus))}
    details = {}
    unit_mass = np.ones((count, 3))  # request unscaled stiffness, not M^-1/2 K M^-1/2
    if cohesive is not None:
        from .._elastic_cohesive import ElasticCohesiveLaw
        law = cohesive.assembler.law
        if not isinstance(law, ElasticCohesiveLaw) or not (
            law.normal_stiffness == law.tangential_stiffness == law.second_tangential_stiffness
        ):
            raise NotImplementedError('Initial finite bond planning requires isotropic elasticity')
        pairing = cohesive.assembler.pairing
        contributions['cohesive'] = penalty_stencil_row_sums(
            np.concatenate((cohesive.negative_dofs[pairing.negative_nodes],
                            cohesive.positive_dofs[pairing.positive_nodes]), axis=1),
            np.concatenate((-pairing.negative_weights, pairing.positive_weights), axis=1),
            pairing.weights, law.normal_stiffness, unit_mass)
    for pair, adapter in rigid_adapters:
        trace = adapter.trace
        contributions[f'rigid_contact:{pair.name}'] = penalty_stencil_row_sums(
            trace.node_ids, trace.shape_values, trace.weights, pair.law.penalty, unit_mass)
    for pair in deformable:
        name = f'deformable_contact:{pair.name}'
        contributions[name], details[name] = _deformable_reference_rows(pair, count)
    damping = {}
    if viscosity is not None:
        zero = np.zeros((len(nodes), 8, 3))
        response = bulk_viscosity_response(internal.cells, zero, zero,
            include_row_majorant=True, **viscosity.options(np.ones(len(nodes))))
        damping['initial_bulk_viscosity'] = assemble(response.damping_row_majorant)
    plan = plan_nodal_inertia(physical, stiffness_rows=contributions, damping_rows=damping,
        target_increment=policy.target_increment, safety_factor=policy.safety_factor)
    mass = FixedScaledInertia(internal.mass_diagonal, factor=plan.nodal_factors)
    if mass.summary()['added_mass_ratio'] > policy.maximum_added_mass_ratio:
        raise ValueError('Mass scaling exceeds its added-mass budget')
    diagnostics = {}
    coordinates = history.u.value.function_space.tabulate_dof_coordinates()
    for name, nodal_mass in (('physical_mass', physical),
                             ('numerical_mass', plan.numerical_nodal_mass)):
        diagnostic = diagnose_nodal_increment(
            nodal_mass, stiffness_rows=contributions, damping_rows=damping,
            requested_increment=dt, safety_factor=policy.safety_factor)
        for controller in diagnostic['controllers']:
            controller['reference_coordinates'] = coordinates[controller['node_index']].tolist()
        diagnostic['node_identity_scope'] = 'current_serial_displacement_node_order'
        diagnostics[name] = diagnostic
    report = {**policy.summary(), 'plan': plan.summary(), 'operator_details': details,
              'increment_diagnostics': diagnostics,
              'maximum_unscaled_stiffness_rows': {key: float(row.max(initial=0))
                                                 for key, row in contributions.items()},
              'generated_from_current_mesh': True,
              'nonlinear_material_tangent_included': False,
              'curved_contact_negative_curvature_certified': False,
              'quadratic_viscosity_at_zero_velocity': 0.}
    return mass, report
