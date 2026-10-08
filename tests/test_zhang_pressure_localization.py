"""Offline localization must retain energy semantics and reject bad data."""

import numpy as np
import pytest

from zhang_2021_pressure_localization import localize


def _inputs():
    J = np.exp(0.1)
    F = np.tile(np.diag([J, 1.0, 1.0]), (2, 2, 1, 1))
    S = np.tile(np.eye(3) * 0.2 / J, (2, 2, 1, 1))
    return dict(
        coordinates=np.zeros((2, 2, 2)),
        weights=np.ones((2, 2)) / 4,
        gradient=F,
        stress=S,
        condensed=np.ones((2, 2)) * 0.3,
        bulk=np.ones((2, 2)) * 10,
        cell_ids=np.array([4, 8]),
        phase=np.array([[False, False], [True, True]]),
        reference_volume=1.0,
        expected={
            "condensed_elastic_energy_density": 0.3,
            "pressure_orthogonality_density": 0.016,
            "pressure_constraint_defect_energy_density": 0.032,
            "primal_elastic_energy_density": 0.348,
        },
    )


def test_kirchhoff_reconstruction_and_weighted_partition():
    report = localize(**_inputs())
    assert report["reconstruction_verified"]
    assert report["inclusion_defect_fraction"] == pytest.approx(0.5)
    assert report["channels"][
        "pressure_constraint_defect_energy_density"
    ] == pytest.approx(0.032)
    assert not report["benchmark_promotion_authorized"]


@pytest.mark.parametrize(
    "problem", ["stress", "weights", "jacobian", "nan", "identity"]
)
def test_reject_mismatched_or_invalid_archive(problem):
    args = _inputs()
    if problem == "stress":
        args["stress"] *= 2
    elif problem == "weights":
        args["weights"][0, 0] = -1
    elif problem == "jacobian":
        args["gradient"][0, 0, 0, 0] = -1
    elif problem == "nan":
        args["gradient"][0, 0, 0, 0] = np.nan
    else:
        args["cell_ids"][:] = 4
    with pytest.raises(ValueError):
        localize(**args)


def test_zero_defect_has_no_concentration_percentage():
    args = _inputs()
    args["gradient"][:] = np.eye(3)
    args["stress"][:] = 0.0
    args["expected"].update(
        pressure_orthogonality_density=0.0,
        pressure_constraint_defect_energy_density=0.0,
        primal_elastic_energy_density=0.3,
    )
    report = localize(**args)
    assert report["inclusion_defect_fraction"] is None
    assert all(item["defect_fraction"] is None for item in report["concentration"])


def test_cell_permutation_preserves_identity_and_concentration():
    args = _inputs()
    original = localize(**args)
    for key in (
        "coordinates",
        "weights",
        "gradient",
        "stress",
        "condensed",
        "bulk",
        "cell_ids",
        "phase",
    ):
        args[key] = args[key][::-1].copy()
    assert localize(**args) == original
