"""Physical invariants for general mapped meshes, independent of benchmark fixtures."""
import numpy as np
import pytest
from mpi4py import MPI
from agentfem import mesh, results


@pytest.mark.parametrize("dimension,cell_type,degree", [(2,"triangle",1), (2,"quadrilateral",1),
    (2,"quadrilateral",2), (3,"tetrahedron",1), (3,"hexahedron",1), (3,"hexahedron",2)])
def test_mapped_grid_recovers_affine_volume_and_centroid(dimension, cell_type, degree):
    axis = np.linspace(0, 1, 2*degree+1)
    scale = np.arange(2, 2+dimension)
    shift = np.arange(dimension)
    domain = mesh.mapped_grid(axes=(axis,)*dimension,
        mapping=lambda x: scale[:,None]*x+shift[:,None], cell_type=cell_type,
        geometry_degree=degree, comm=MPI.COMM_SELF)
    assert domain.geometry.cmaps[0].degree == degree
    measure = mesh.cell_measure(domain)
    assert results.integral(1.0, measure=measure) == pytest.approx(float(np.prod(scale)))
    # Test the public geometry map independently of its connectivity implementation.
    assert np.min(domain.geometry.x[:,:dimension],axis=0) == pytest.approx(shift)
    assert np.max(domain.geometry.x[:,:dimension],axis=0) == pytest.approx(shift+scale)


@pytest.mark.parametrize("axes,cell_type,degree", [
    (([0,1,1],[0,1]),'quadrilateral',1), (([0,1],[0,1]),'triangle',2),
    (([0,1],[0,1]),'hexahedron',1), (([0,float('nan')],[0,1]),'triangle',1),
])
def test_invalid_grid_is_rejected_before_backend(axes, cell_type, degree):
    with pytest.raises(ValueError):
        mesh.mapped_grid(axes=axes, cell_type=cell_type,geometry_degree=degree,comm=MPI.COMM_SELF)


def test_from_arrays_checks_geometry_width():
    with pytest.raises(ValueError, match="width"):
        mesh.from_arrays(cells=np.array([[0,1,2]]), coordinates=np.array([[0.,0.],[1.,0.],[0.,1.]]),
                         cell_type="quadrilateral",comm=MPI.COMM_SELF)


def test_centroid_selection_preserves_narrow_sampling_band():
    domain=mesh.rectangle((0,0),(1,1),(2,2),comm=MPI.COMM_SELF)
    band=lambda x:(x[1]>.2)&(x[1]<.3)
    assert mesh.locate_cells(domain, band).size == 0
    selected=mesh.locate_cells(domain,band,by="centroid")
    assert selected.size == 2
    with pytest.raises(ValueError,match="vertices.*centroid"):
        mesh.locate_cells(domain,band,by="unknown")
