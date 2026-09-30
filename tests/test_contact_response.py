from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models


def test_frictionless_penalty_response_matches_plane_potential_semantics():
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    projection = surface.project(((0.0, 1.0), (0.0, 0.0), (0.0, -0.2)))
    law = boundary_models.frictionless_penalty_contact_law(100.0)

    response = law.evaluate(projection)

    np.testing.assert_allclose(response.penetration, (0.0, 0.0, 0.2))
    np.testing.assert_allclose(response.pressures, (0.0, 0.0, 20.0))
    np.testing.assert_allclose(response.potential_densities, (0.0, 0.0, 2.0))
    np.testing.assert_array_equal(response.active, (False, False, True))
    np.testing.assert_allclose(
        response.structural_residual_tractions,
        ((0.0, 0.0), (0.0, 0.0), (0.0, -20.0)),
    )
    np.testing.assert_allclose(
        response.surface_generalized_tractions,
        ((0.0, 0.0), (0.0, 0.0), (0.0, 20.0)),
    )
    np.testing.assert_allclose(
        response.contact_tractions_on_structure,
        -response.structural_residual_tractions,
    )
    np.testing.assert_allclose(
        response.contact_tractions_on_surface,
        -response.surface_generalized_tractions,
    )
    assert response.summary() == {
        "kind": "frictionless_penalty_contact_response",
        "point_count": 3,
        "active_count": 1,
        "invalid_count": 0,
        "invalid_policy": "reject",
        "signed_gap_convention": "positive_admissible_negative_penetration",
        "response_level": "pointwise_unintegrated",
        "potential": "0.5 * penalty * penetration^2",
        "linearization": "not_provided",
        "friction": "none",
    }


def test_penalty_potential_directional_derivative_is_structural_residual():
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    law = boundary_models.frictionless_penalty_contact_law(250.0)
    point = np.asarray((0.3, -0.4), dtype=float)
    epsilon = 1.0e-7

    response = law.evaluate(surface.project((point,)))
    plus = law.evaluate(
        surface.project((point + epsilon * np.asarray(surface.normal),))
    )
    minus = law.evaluate(
        surface.project((point - epsilon * np.asarray(surface.normal),))
    )
    derivative = (plus.potential_densities[0] - minus.potential_densities[0]) / (
        2.0 * epsilon
    )

    np.testing.assert_allclose(
        derivative,
        np.dot(response.structural_residual_tractions[0], surface.normal),
        rtol=1.0e-9,
        atol=1.0e-8,
    )


def test_penalty_response_is_geometry_neutral_and_supports_pointwise_penalty():
    surface = boundary_models.rigid_sphere(
        center=(0.0, 0.0, 0.0),
        radius=1.0,
    )
    projection = surface.project(((0.5, 0.0, 0.0), (0.0, 0.25, 0.0)))
    law = boundary_models.frictionless_penalty_contact_law((10.0, 20.0))

    response = law.evaluate(projection)

    np.testing.assert_allclose(response.penetration, (0.5, 0.75))
    np.testing.assert_allclose(response.pressures, (5.0, 15.0))
    np.testing.assert_allclose(
        response.contact_tractions_on_structure,
        ((5.0, 0.0, 0.0), (0.0, 15.0, 0.0)),
    )
    assert law.summary()["integration"] == "backend_required"
    assert law.summary()["linearization"] == "backend_required"


def test_penalty_response_rejects_invalid_projection_by_default():
    projection = boundary_models.SurfaceProjection(
        surface_name="incomplete_tool",
        surface_kind="triangulated_rigid_surface",
        query_points=((0.0, 0.0, -0.1), (10.0, 10.0, 10.0)),
        closest_points=((0.0, 0.0, 0.0), (np.nan, np.nan, np.nan)),
        normals=((0.0, 0.0, 1.0), (np.nan, np.nan, np.nan)),
        signed_gaps=(-0.1, np.nan),
        valid=(True, False),
        status_codes=("ok", "no_candidate"),
        method="test_search",
        entity_ids=(7, -1),
        geometry_fingerprint="0" * 64,
    )

    with pytest.raises(ValueError, match="refuses invalid projection"):
        boundary_models.frictionless_penalty_contact_law(100.0).evaluate(projection)


def test_penalty_response_can_explicitly_treat_invalid_search_as_inactive():
    projection = boundary_models.SurfaceProjection(
        surface_name="incomplete_tool",
        surface_kind="triangulated_rigid_surface",
        query_points=((0.0, 0.0, -0.1), (10.0, 10.0, 10.0)),
        closest_points=((0.0, 0.0, 0.0), (np.nan, np.nan, np.nan)),
        normals=((0.0, 0.0, 1.0), (np.nan, np.nan, np.nan)),
        signed_gaps=(-0.1, np.nan),
        valid=(True, False),
        status_codes=("ok", "no_candidate"),
        method="test_search",
        entity_ids=(7, -1),
        geometry_fingerprint="0" * 64,
    )
    law = boundary_models.frictionless_penalty_contact_law(
        100.0,
        invalid_policy="inactive",
    )

    response = law.evaluate(projection)

    np.testing.assert_allclose(response.penetration, (0.1, 0.0))
    np.testing.assert_allclose(response.pressures, (10.0, 0.0))
    np.testing.assert_allclose(response.structural_residual_tractions[1], 0.0)
    assert response.summary()["invalid_count"] == 1


def test_penalty_contact_law_validates_inputs_and_supports_empty_batches():
    with pytest.raises(ValueError, match="positive"):
        boundary_models.frictionless_penalty_contact_law(0.0)
    with pytest.raises(ValueError, match="invalid_policy"):
        boundary_models.frictionless_penalty_contact_law(1.0, invalid_policy="guess")

    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    response = boundary_models.frictionless_penalty_contact_law(10.0).evaluate(
        surface.project(np.empty((0, 3)))
    )

    assert response.point_count == 0
    assert response.structural_residual_tractions.shape == (0, 3)
    with pytest.raises(ValueError, match="one value per point"):
        boundary_models.frictionless_penalty_contact_law((1.0, 2.0)).evaluate(
            surface.project(((0.0, 0.0, -1.0),))
        )


def test_penalty_contact_response_arrays_are_immutable():
    surface = boundary_models.rigid_plane(
        point=(0.0, 0.0),
        normal=(0.0, 1.0),
    )
    response = boundary_models.frictionless_penalty_contact_law(10.0).evaluate(
        surface.project(((0.0, -1.0),))
    )

    with pytest.raises(ValueError):
        response.pressures[0] = 0.0
    with pytest.raises(ValueError):
        response.structural_residual_tractions[0, 1] = 0.0
