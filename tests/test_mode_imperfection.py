# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Geometry transaction and displacement-controlled stability regressions."""

import runpy
from pathlib import Path
import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
from agentfem import mesh, fields


@pytest.mark.parametrize(
    "cell", ["triangle", "quadrilateral", "tetrahedron", "hexahedron"]
)
def test_mode_amplitude_restore_and_invalid_geometry_rollback(cell):
    dim = 2 if cell in ("triangle", "quadrilateral") else 3
    d = (
        mesh.rectangle((0.0, 0.0), (1.0, 1.0), (4, 4), cell_type=cell)
        if dim == 2
        else mesh.cuboid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2), cell_type=cell)
    )
    u = fem.Function(fields.displacement(d, degree=2).space)
    u.interpolate(
        lambda x: np.vstack([x[0]] + [np.zeros_like(x[0]) for _ in range(dim - 1)])
    )
    original = d.geometry.x.copy()
    receipt = mesh.apply_mode_imperfection(d, [u], amplitudes=[0.1])
    assert d.comm.allreduce(
        float(np.max(d.geometry.x[:, 0] - original[:, 0], initial=0)), op=MPI.MAX
    ) == pytest.approx(0.1)
    receipt.restore()
    np.testing.assert_array_equal(d.geometry.x, original)
    with pytest.raises(ValueError):
        mesh.apply_mode_imperfection(d, [u], amplitudes=[-2.0])
    np.testing.assert_array_equal(d.geometry.x, original)
    u.x.array[:] = 0
    with pytest.raises(ValueError, match="zero mode"):
        mesh.apply_mode_imperfection(d, [u], amplitudes=[0.1])


def test_imperfect_column_paths_and_mesh_increment_refinement():
    solve = runpy.run_path(
        str(Path(__file__).parents[1] / "examples/imperfect_column/case.py")
    )["solve_column"]
    _, small = solve(imperfection=0.02, nx=24, increment=0.1)
    _, large = solve(imperfection=0.1, nx=24, increment=0.1)
    _, fine = solve(imperfection=0.02, nx=40, increment=0.05)
    for curve in (small, large, fine):
        assert curve["load_factor"][-1] == pytest.approx(1.0)
        np.testing.assert_allclose(
            curve["displacement"], -0.05 * curve["load_factor"], atol=1.0e-9
        )
        assert np.all(np.isfinite(curve["reaction"]))
        assert curve["reaction"][0] == pytest.approx(0.0, abs=1.0e-8)
        assert 1.5 < -curve["reaction"][-1] < 2.1
        assert abs(curve["monitor_displacement"][-1]) > 0.5
    assert abs(large["reaction"][-1]) < abs(small["reaction"][-1])
    assert abs(fine["reaction"][-1] / small["reaction"][-1] - 1) < 0.02
    assert (
        abs(fine["monitor_displacement"][-1] / small["monitor_displacement"][-1] - 1)
        < 0.02
    )
