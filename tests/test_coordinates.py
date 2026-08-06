from __future__ import annotations

import numpy as np
import pytest

from agentfem import coordinates


def test_cartesian_system_maps_vectors_points_and_tensors_both_ways():
    system = coordinates.cartesian(
        origin=(10.0, 20.0),
        x=(0.0, 1.0),
        y=(-1.0, 0.0),
        name="part",
    )

    np.testing.assert_allclose(system.vector_to_global((2.0, 3.0)), (-3.0, 2.0))
    np.testing.assert_allclose(system.vector_to_local((-3.0, 2.0)), (2.0, 3.0))
    np.testing.assert_allclose(system.point_to_global((2.0, 3.0)), (7.0, 22.0))
    tensor = np.asarray(((1.0, 2.0), (2.0, 5.0)))
    np.testing.assert_allclose(
        system.tensor_to_local(system.tensor_to_global(tensor)),
        tensor,
    )
    assert system.summary()["name"] == "part"


def test_cartesian_system_rejects_scaled_or_left_handed_axes():
    with pytest.raises(ValueError, match="orthonormal"):
        coordinates.cartesian(axes=((2.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="right-handed"):
        coordinates.cartesian(axes=((1.0, 0.0), (0.0, -1.0)))


def test_reference_point_can_be_declared_in_local_coordinates():
    system = coordinates.cartesian(
        origin=(1.0, 2.0, 3.0),
        axes=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    point = coordinates.reference_point((2.0, 3.0, 4.0), system=system, name="RP-1")

    np.testing.assert_allclose(point.coordinates, (-2.0, 4.0, 7.0))
    assert point.summary()["name"] == "RP-1"
