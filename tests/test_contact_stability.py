from __future__ import annotations

import numpy as np
import pytest

from agentfem import boundary_models


def _one_point_trace(*, weight: float = 2.0):
    return boundary_models.ContactTrace(
        point_ids=(11,),
        node_ids=((0,),),
        shape_values=((1.0,),),
        weights=(weight,),
    )


def test_contact_stability_recovers_one_dof_penalty_oscillator_limit():
    rows = boundary_models.contact_penalty_local_row_sums(
        _one_point_trace(),
        ((4.0, 4.0),),
        dimension=2,
        normal_penalty=8.0,
    )

    np.testing.assert_allclose(rows, ((4.0, 4.0),))
    estimate = boundary_models.contact_stability_estimate_from_bound(
        4.0,
        safety_factor=0.8,
        normal_penalty_maximum=8.0,
        tangential_penalty_maximum=None,
        point_count=1,
    )
    assert estimate.unsafed_limit == pytest.approx(1.0)
    assert estimate.selected == pytest.approx(0.8)
    assert estimate.summary()["scope"] == "contact_penalty_contribution_only"


def test_contact_stability_uses_stiffer_tangential_penalty():
    normal = boundary_models.contact_penalty_local_row_sums(
        _one_point_trace(),
        ((4.0, 4.0, 4.0),),
        dimension=3,
        normal_penalty=2.0,
    )
    frictional = boundary_models.contact_penalty_local_row_sums(
        _one_point_trace(),
        ((4.0, 4.0, 4.0),),
        dimension=3,
        normal_penalty=2.0,
        tangential_penalty=8.0,
        friction_coefficient=0.5,
    )

    np.testing.assert_allclose(frictional, 5.5 * normal)


def test_contact_stability_handles_shared_nodes_and_signed_shape_values():
    trace = boundary_models.ContactTrace(
        point_ids=(3,),
        node_ids=((0, 1),),
        shape_values=((1.25, -0.25),),
        weights=(0.5,),
    )
    rows = boundary_models.contact_penalty_local_row_sums(
        trace,
        ((1.0, 4.0), (9.0, 16.0)),
        dimension=2,
        normal_penalty=12.0,
    )

    scaled = np.array((1.25, 0.25 / 3.0))
    expected = np.repeat(
        (6.0 * scaled * np.sum(scaled))[:, None],
        2,
        axis=1,
    )
    np.testing.assert_allclose(rows, expected)


def test_contact_stability_row_bound_dominates_exact_trace_spectrum():
    trace = boundary_models.ContactTrace(
        point_ids=(5, 7),
        node_ids=((0, 1, 2), (0, 1, 2)),
        shape_values=((0.6, 0.3, 0.1), (-0.2, 0.5, 0.7)),
        weights=(0.4, 0.9),
    )
    masses = np.array(((1.0, 1.5), (2.0, 2.5), (3.0, 4.0)))
    penalties = np.array((11.0, 17.0))
    rows = boundary_models.contact_penalty_local_row_sums(
        trace,
        masses,
        dimension=2,
        normal_penalty=penalties,
    )

    normal = np.array((1.0, 2.0), dtype=float)
    normal /= np.linalg.norm(normal)
    stiffness = sum(
        weight * penalty * np.kron(np.outer(shape, shape), np.outer(normal, normal))
        for weight, penalty, shape in zip(
            trace.weights,
            penalties,
            trace.shape_values,
            strict=True,
        )
    )
    inverse_sqrt_mass = np.diag(1.0 / np.sqrt(masses.reshape(-1)))
    scaled = inverse_sqrt_mass @ stiffness @ inverse_sqrt_mass
    exact = float(np.max(np.linalg.eigvalsh(scaled)))
    assert float(np.max(rows)) >= exact


@pytest.mark.parametrize(
    ("masses", "message"),
    [
        (((0.0, 1.0),), "positive mass"),
        (((-1.0, 1.0),), "non-negative"),
        (((1.0, np.nan),), "finite"),
    ],
)
def test_contact_stability_rejects_invalid_contact_mass(masses, message):
    with pytest.raises(ValueError, match=message):
        boundary_models.contact_penalty_local_row_sums(
            _one_point_trace(),
            masses,
            dimension=2,
            normal_penalty=8.0,
        )


def test_contact_stability_estimate_rejects_inconsistent_limit():
    with pytest.raises(ValueError, match="inconsistent"):
        boundary_models.ContactStabilityEstimate(
            selected=0.5,
            unsafed_limit=1.0,
            spectral_radius_upper_bound=4.0,
            safety_factor=0.8,
            normal_penalty_maximum=8.0,
            tangential_penalty_maximum=None,
            friction_coefficient=0.0,
            point_count=1,
        )


def test_noncontact_and_contact_bounds_are_composed_before_selecting_time_step():
    contact = boundary_models.contact_stability_estimate_from_bound(
        4.0,
        safety_factor=0.8,
        normal_penalty_maximum=8.0,
        tangential_penalty_maximum=None,
        point_count=1,
    )

    combined = boundary_models.combine_explicit_stability_bounds(
        contact,
        noncontact_unsafed_limit=1.0,
    )

    assert combined.noncontact_spectral_radius_upper_bound == pytest.approx(4.0)
    assert combined.total_spectral_radius_upper_bound == pytest.approx(8.0)
    assert combined.unsafed_limit == pytest.approx(1.0 / np.sqrt(2.0))
    assert combined.selected == pytest.approx(0.8 / np.sqrt(2.0))
    assert combined.selected < min(0.8, contact.selected)
    assert combined.summary()["composition"] == "additive_spectral_upper_bounds"
