import numpy as np
import pytest

from tools.p92_common_prefix import aligned_difference


def test_common_support_ignores_nonoverlapping_endpoints():
    result = aligned_difference([0, 1, 2, 3], [100, 2, 4, 100], [1, 1.5, 2], [5, 6, 7])
    assert result["interval_s"] == [1, 2]
    assert result["right_minus_left_time_mean"] == 3
    assert result["difference_time_rms"] == 3
    assert result["left_change"] == result["right_change"] == 2


def test_exact_time_weighting_not_sample_density_weighting():
    result = aligned_difference([0, 1], [0, 0], [0, .01, .02, 1], [0, .01, .02, 1])
    assert result["right_minus_left_time_mean"] == pytest.approx(.5)
    assert result["difference_time_rms"] == pytest.approx(1/np.sqrt(3))


@pytest.mark.parametrize("t,a,u,b", [([0, 1], [0, 1], [2, 3], [0, 1]),
    ([0, 0], [0, 1], [0, 1], [0, 1]), ([0, 1], [0, np.nan], [0, 1], [0, 1]),
    ([0, 1], [0], [0, 1], [0, 1]), ([0, 1], [0, 1j], [0, 1], [0, 1])])
def test_invalid_or_disjoint_records(t, a, u, b):
    with pytest.raises(ValueError):
        aligned_difference(t, a, u, b)
