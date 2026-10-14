# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from test_finite_uniform_hex import _operator
from agentfem.elements._bulk_viscosity import bulk_viscosity_response


@pytest.mark.parametrize("rate", [-0.2, 0.2])
def test_damping_rows_match_independent_velocity_derivative(rate):
    cells = _operator(True)
    u = 0.03 * cells.coordinates
    v = rate * cells.coordinates
    kwargs = dict(density=2.0, wave_speed=5.0, characteristic_length=0.8)
    value = bulk_viscosity_response(cells, u, v, include_row_majorant=True, **kwargs)
    columns = []
    for direction in np.eye(24):
        d = direction.reshape(1, 8, 3) * 1e-6
        a = bulk_viscosity_response(cells, u, v + d, **kwargs).internal_force
        b = bulk_viscosity_response(cells, u, v - d, **kwargs).internal_force
        columns.append(((a - b) / 2e-6).ravel())
    tangent = np.column_stack(columns)
    np.testing.assert_allclose(
        value.damping_row_majorant.ravel(), np.abs(tangent).sum(axis=1), rtol=1e-9
    )
    assert (
        np.linalg.eigvalsh(np.diag(value.damping_row_majorant.ravel()) - tangent).min()
        > -1e-9
    )
    ordinary = bulk_viscosity_response(cells, u, v, **kwargs)
    np.testing.assert_array_equal(value.internal_force, ordinary.internal_force)
    assert ordinary.damping_row_majorant is None
    assert not value.damping_row_majorant.flags.writeable
