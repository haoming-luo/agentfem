from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from tools.p92_fit_subspace_audit import audit, replacement_basis_from_svd


def report(basis):
    return dict(model_id="synthetic", basis=basis.tolist(), maximum_log_coordinate_norm=.3,
        coordinates=[f"coordinate-{i}" for i in range(basis.shape[0])],
        fit=dict(coordinates=[0.]*basis.shape[1], log_multipliers=[0.]*basis.shape[0]))


def test_support_matches_exhaustive_vertices_not_independent_coordinate_box():
    rng = np.random.default_rng(20261122)
    basis, _ = np.linalg.qr(rng.normal(size=(5, 3)))
    data = report(basis)
    result = audit(data, replacement_indices=(1, 4))
    width = .3/np.sqrt(3)
    vertices = np.array(list(product((-width, width), repeat=3)))@basis.T
    np.testing.assert_allclose([c["reachable_symmetric_half_width"] for c in result["coordinates"]],
                               np.max(abs(vertices), axis=0), atol=1e-14)
    direction = result["replacement"]["weakest_coordinate_combination"]
    expected = np.max(abs(vertices[:, [1, 4]]@direction))
    assert result["replacement"]["weakest_combination_reachable_half_width"] == pytest.approx(expected)
    assert not result["fitted"] and not result["acceptance_thresholds_changed"]


def test_excluded_replacement_direction_is_visible():
    basis = np.eye(4)[:, :2]
    result = audit(report(basis), replacement_indices=(1, 3))
    assert result["replacement"]["projection_gram_eigenvalues"] == [0., 1.]
    assert result["replacement"]["weakest_combination_reachable_half_width"] == 0.
    assert result["coordinates"][3]["reachable_symmetric_half_width"] == 0.


def test_bounds_identity_and_unique_coordinate_checks():
    base = report(np.eye(4)[:, :2])
    for key, value in (("basis", np.ones((4, 2)).tolist()),
                       ("maximum_log_coordinate_norm", np.nan),
                       ("coordinates", ["same"]*4)):
        bad = dict(base, **{key: value})
        with pytest.raises(ValueError):
            audit(bad, replacement_indices=(1, 3))
    for coordinate, multiplier in (([.4, 0.], [.4, 0., 0., 0.]),
                                   ([0., 0.], [.1, 0., 0., 0.])):
        bad = deepcopy(base)
        bad["fit"].update(coordinates=coordinate, log_multipliers=multiplier)
        with pytest.raises(ValueError, match="bounded subspace"):
            audit(bad, replacement_indices=(1, 3))
    for ids in ((1, 1), (1, 5), (True, 3), ()):
        with pytest.raises(ValueError, match="indices"):
            audit(base, replacement_indices=ids)


def test_replacement_basis_keeps_exact_named_axes_and_restricted_spectrum():
    rng = np.random.default_rng(22)
    jac = rng.normal(size=(30, 7))
    _, values, vectors = np.linalg.svd(jac, full_matrices=False)
    basis, report = replacement_basis_from_svd(values, vectors, replacement_indices=(1, 5), retained=4)
    np.testing.assert_allclose(basis.T@basis, np.eye(4), atol=1e-12)
    np.testing.assert_array_equal(basis[:, :2], np.eye(7)[:, [1, 5]])
    np.testing.assert_array_equal(basis[[1, 5], 2:], 0.)
    np.testing.assert_allclose(report["singular_values"], np.linalg.svd(jac@basis, compute_uv=False), atol=1e-12)
    assert report["passed"]


def test_pinning_unobservable_axes_does_not_create_information():
    values = np.array([1., .8, .6, .4, 1e-9, 1e-12])
    _, report = replacement_basis_from_svd(values, np.eye(6), replacement_indices=(4, 5), retained=4)
    assert not report["passed"]
    assert report["condition"] > 1e4
    for bad_values, bad_vectors in ((values[::-1], np.eye(6)), (values, np.ones((6, 6)))):
        with pytest.raises(ValueError, match="right SVD"):
            replacement_basis_from_svd(bad_values, bad_vectors, replacement_indices=(4, 5), retained=4)
