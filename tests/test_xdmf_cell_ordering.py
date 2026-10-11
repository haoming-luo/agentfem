# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent VTK XDMF interpolation checks, not a same-writer round trip."""

from types import SimpleNamespace

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI

from agentfem import results


@pytest.mark.parametrize("cell_name", ["hexahedron", "tetrahedron"])
@pytest.mark.parametrize("degree", [1, 2])
def test_external_xdmf_reader_preserves_volume_cell_interpolation(tmp_path, cell_name, degree):
    vtk = pytest.importorskip("vtk")
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2,
                                  cell_type=getattr(mesh.CellType, cell_name))
    field = fem.Function(fem.functionspace(domain, ("Lagrange", degree, (3,))), name="U")
    field.interpolate(lambda x: .1 * x)
    path = results.write_unified_xdmf_series(
        tmp_path / "volume.xdmf", (SimpleNamespace(solution=field, load_factor=1.),),
        ((),), deformation_scale=0.)
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() == domain.topology.index_map(3).size_local
    expected_nodes = {("hexahedron", 1): 8, ("hexahedron", 2): 27,
                      ("tetrahedron", 1): 4, ("tetrahedron", 2): 10}
    for i in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(i)
        assert cell.GetCellDimension() == 3
        assert cell.GetNumberOfPoints() == expected_nodes[cell_name, degree]
        corners = np.array([cell.GetPoints().GetPoint(j)
                            for j in range(8 if cell_name == "hexahedron" else 4)])
        for local in ([.2, .3, .1], [.1, .1, .7]):
            point = [0.] * 3
            weights = [0.] * cell.GetNumberOfPoints()
            cell.EvaluateLocation(vtk.reference(0), local, point, weights)
            r, s, t = local
            if cell_name == "hexahedron":
                basis = np.array([(1-r)*(1-s)*(1-t), r*(1-s)*(1-t), r*s*(1-t),
                                  (1-r)*s*(1-t), (1-r)*(1-s)*t, r*(1-s)*t,
                                  r*s*t, (1-r)*s*t])
            else:
                basis = np.array([1-r-s-t, r, s, t])
            np.testing.assert_allclose(point, basis @ corners, atol=2e-14)
            values = np.array([grid.GetPointData().GetArray("U").GetTuple(
                cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())])
            np.testing.assert_allclose(np.asarray(weights) @ values, .1*np.asarray(point),
                                       atol=2e-14)
