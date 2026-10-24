from types import SimpleNamespace

import numpy as np
import pytest

from agentfem import boundary_models
from agentfem.boundary_models.initial_gap_bound import (
    certify_initial_convex_triangle_gap,
    certify_initial_triangle_gap,
    initial_gap_bound_method,
)


class SphereDistance:
    """Independent exact signed-distance oracle for a closed ball."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def project(self, points, *, factor):
        assert factor == 0.0
        gaps = np.linalg.norm(np.asarray(points) - self.center, axis=1) - self.radius
        return SimpleNamespace(all_valid=True, signed_gaps=gaps)


TRIANGLE = np.asarray((((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),))


@pytest.mark.parametrize("interpolation", [1, 2])
@pytest.mark.parametrize("geometry", [1, 2])
@pytest.mark.parametrize("topology", ["triangle", "quadrilateral"])
def test_plane_reference_certificate_depends_on_geometry_not_field_order(interpolation, geometry, topology):
    surface = boundary_models.rigid_plane(point=(0., 0., 0.), normal=(1., 0., 0.))
    method = initial_gap_bound_method(surface, interpolation_degree=interpolation,
                                     geometry_degree=geometry, facet_topology=topology)
    if geometry == 2:
        assert method is None  # vertices do not bound a curved reference face
    else:
        assert method == ("exact_plane_nodal" if interpolation == 1 else "exact_plane_geometry_vertices")


def _convex_cube():
    return boundary_models.convex_polyhedral_rigid_surface(
        vertices=(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ),
        triangles=(
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ),
    )


def test_lipschitz_certificate_bounds_whole_face_and_refines():
    projector = SphereDistance((0.2, 0.2, -0.2), 0.1)
    bound = certify_initial_triangle_gap(
        TRIANGLE, projector, penetration_limit=0.05,
    )
    assert bound.maximum_penetration_bound <= 0.05
    assert bound.maximum_refinement_depth > 0
    assert bound.sampled_point_count > 1
    # This sampled check is not the proof; it guards the oracle geometry.
    x = np.linspace(0, 1, 101)
    points = np.asarray([(a, b, 0) for a in x for b in x if a + b <= 1])
    minimum = np.min(projector.project(points, factor=0.0).signed_gaps)
    assert -minimum <= bound.maximum_penetration_bound


def test_interior_overlap_missed_by_three_gauss_points_is_rejected():
    projector = SphereDistance((0.12, 0.12, 0), 0.08)
    gauss = np.asarray(((1 / 6, 1 / 6, 0), (2 / 3, 1 / 6, 0), (1 / 6, 2 / 3, 0)))
    assert np.min(projector.project(gauss, factor=0.0).signed_gaps) > -0.05
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        certify_initial_triangle_gap(TRIANGLE, projector, penetration_limit=0.05)


def test_unresolved_zero_tolerance_fails_closed_without_fabricating_a_bound():
    projector = SphereDistance((0.2, 0.2, -0.1), 0.1)
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-INITIAL-UNCERTIFIED"):
        certify_initial_triangle_gap(
            TRIANGLE, projector, penetration_limit=0.0, maximum_depth=4,
        )


def test_invalid_projection_cannot_certify():
    projector = SphereDistance((0.2, 0.2, -0.2), 0.1)
    projector.project = lambda points, factor: SimpleNamespace(
        all_valid=False, signed_gaps=np.zeros(len(points)),
    )
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-PROJECTION"):
        certify_initial_triangle_gap(TRIANGLE, projector, penetration_limit=0.1)


def test_convex_dominant_support_plane_certifies_near_touch_without_subdivision():
    surface = _convex_cube()
    triangle = np.asarray((((0.0, 0.3, 0.3),
                            (0.0, 0.7, 0.3),
                            (0.0, 0.3, 0.7)),))
    bound = certify_initial_convex_triangle_gap(
        triangle, surface, penetration_limit=1.0e-9,
    )
    assert bound.maximum_refinement_depth == 0
    assert bound.sampled_point_count == 3
    assert bound.maximum_penetration_bound <= 1.0e-9
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-INITIAL-UNCERTIFIED"):
        certify_initial_triangle_gap(
            triangle, surface, penetration_limit=1.0e-9, maximum_depth=4,
        )


def test_convex_support_plane_rejects_nodal_overlap_and_falls_back_at_features():
    surface = _convex_cube()
    penetrating = np.asarray((((0.01, 0.3, 0.3),
                               (0.01, 0.7, 0.3),
                               (0.01, 0.3, 0.7)),))
    with pytest.raises(ValueError, match="AFM-CONTACT-INITIAL-OVERLAP"):
        certify_initial_convex_triangle_gap(
            penetrating, surface, penetration_limit=0.005,
        )
    crossing = np.asarray((((0.1, 0.3, 0.3),
                            (0.3, 0.1, 0.3),
                            (0.3, 0.3, 0.1)),))
    bound = certify_initial_convex_triangle_gap(
        crossing, surface, penetration_limit=0.3,
    )
    assert bound.maximum_refinement_depth > 0
    assert bound.sampled_point_count > 3
    assert bound.maximum_penetration_bound <= 0.3
