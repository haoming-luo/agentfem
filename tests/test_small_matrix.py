import numpy as np
import pytest
from agentfem._small_matrix import determinants_3x3


def test_well_conditioned_batches_and_empty():
    rng = np.random.default_rng(707)
    values = np.eye(3)+rng.uniform(-.05, .05, (1000, 3, 3))
    np.testing.assert_allclose(determinants_3x3(values), np.linalg.det(values),
                               rtol=2e-15, atol=0)
    assert determinants_3x3(np.empty((0, 3, 3))).shape == (0,)
    assert determinants_3x3(np.eye(3, dtype=int)[None])[0] == 1


def test_cancellation_and_range_use_reference_factorization(monkeypatch):
    base = np.array([[1., 1, 1], [1, 1+1e-14, 1], [1, 1, 1+1e-14]])
    values = np.array([base, np.zeros((3,3)), np.eye(3)*1e-110,
                       np.diag([1e200, 1e200, 1e-200])])
    reference = np.linalg.det(values)
    original = np.linalg.det
    calls = []
    def capture(a):
        calls.append(len(a))
        return original(a)
    monkeypatch.setattr(np.linalg, 'det', capture)
    np.testing.assert_array_equal(determinants_3x3(values), reference)
    assert calls == [4]


def test_intermediate_underflow_cannot_hide_in_normal_final_sum():
    # The first determinant term underflows before multiplication by 1e250;
    # another normal term otherwise makes a naive final-range screen pass.
    values = np.array([[[1e-250, 1e-250, 0.],
                        [0., 1e-250, 1.], [1., 0., 1e250]]])
    np.testing.assert_array_equal(determinants_3x3(values), np.linalg.det(values))
    assert determinants_3x3(values)[0] == pytest.approx(2e-250, rel=1e-12, abs=0)


@pytest.mark.parametrize('value', [np.eye(3), np.ones((2,2,2)),
                                  np.eye(3)[None]*1j, np.full((1,3,3),np.nan)])
def test_bad_input_rejected(value):
    with pytest.raises(ValueError):determinants_3x3(value)
