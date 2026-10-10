# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem.constitutive.material_driver import _cauchy_to_first_piola


@pytest.mark.parametrize("count", [0, 1, 2051])
def test_chunked_piola_conversion_matches_independent_scalar_reference(count):
    rng = np.random.default_rng(37)
    f = np.eye(3)[None] + rng.uniform(-0.1, 0.1, (count, 3, 3))
    sigma = rng.normal(size=(count, 3, 3))
    expected = np.array(
        [np.linalg.det(g) * s @ np.linalg.inv(g).T for g, s in zip(f, sigma)]
    ).reshape(count, 3, 3)
    np.testing.assert_allclose(
        _cauchy_to_first_piola(f, sigma), expected, rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize("invalid", ["shape", "nan", "singular", "inverted"])
def test_piola_conversion_rejects_invalid_kinematics(invalid):
    f, sigma = np.eye(3)[None].copy(), np.eye(3)[None]
    if invalid == "shape":
        f = f.reshape(1, 9)
    elif invalid == "nan":
        f[0, 0, 0] = np.nan
    elif invalid == "singular":
        f[0, 0, 0] = 0
    else:
        f[0, 0, 0] = -1
    with pytest.raises(ValueError):
        _cauchy_to_first_piola(f, sigma)
