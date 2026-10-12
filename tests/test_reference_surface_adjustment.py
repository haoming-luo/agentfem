import numpy as np
import pytest

from agentfem._reference_surface_adjustment import propose_reference_surface_adjustment
from agentfem._initial_nodal_bond import initial_nodal_q1_pairing
from agentfem._interface_quadrilateral import BilinearReferenceTrace
from agentfem import interfaces


def traces():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    quads = np.array([[0, 1, 2, 3]])
    return (
        BilinearReferenceTrace(x + [0, 0, 1e-3], quads, tolerance=1e-10),
        BilinearReferenceTrace(x, quads, tolerance=1e-10),
    )


def test_explicit_proposal_preserves_inputs_and_unselected_nodes():
    a, b = traces()
    before = a.vertices.copy()
    mask = np.array([True, True, False, False])
    result = propose_reference_surface_adjustment(
        a, b, selected_nodes=mask, maximum_distance=0.002, tolerance=1e-10
    )
    np.testing.assert_array_equal(a.vertices, before)
    np.testing.assert_array_equal(result.trace.vertices[~mask], before[~mask])
    np.testing.assert_allclose(
        result.trace.vertices[mask], b.vertices[mask], atol=1e-14
    )
    assert not result.adjustment.flags.writeable
    assert not result.selected_nodes.flags.writeable
    assert result.summary()["bulk_mesh_validated"] is False
    pair = initial_nodal_q1_pairing(
        result.trace, b, eligible_nodes=mask, tolerance=1e-10
    )
    np.testing.assert_allclose(pair.reference_mismatch, 0, atol=1e-14)
    other = propose_reference_surface_adjustment(
        a, b, selected_nodes=~mask, maximum_distance=0.002, tolerance=1e-10
    )
    assert result.fingerprint != other.fingerprint


def test_public_reference_preparation_preserves_explicit_eligibility():
    a, b = traces()
    projected = interfaces.project_reference_points(b, a.vertices,
        maximum_distance=.002, require_unique_normal=False)
    np.testing.assert_allclose(projected['distances'], .001)
    proposal = interfaces.propose_reference_adjustment(a, b,
        selected_nodes=np.array([True, True, False, False]), maximum_distance=.002, tolerance=1e-10)
    assert proposal.summary()['selected_node_count'] == 2
    assert proposal.summary()['live_model_modified'] is False


def test_excess_distance_and_invalid_selection_are_atomic():
    a, b = traces()
    saved = a.vertices.copy()
    with pytest.raises(ValueError):
        propose_reference_surface_adjustment(
            a,
            b,
            selected_nodes=np.ones(4, dtype=bool),
            maximum_distance=1e-4,
            tolerance=1e-10,
        )
    for mask in ([1, 1, 1, 1], np.zeros(4, dtype=bool), [True]):
        with pytest.raises(ValueError):
            propose_reference_surface_adjustment(
                a, b, selected_nodes=mask, maximum_distance=0.002, tolerance=1e-10
            )
    np.testing.assert_array_equal(a.vertices, saved)
