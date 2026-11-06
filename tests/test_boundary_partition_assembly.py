"""Multiple public boundary loads must assemble in one finite-element form."""
import numpy as np
import pytest
from mpi4py import MPI
from agentfem import fields, loads, mesh, operators


def test_boundary_partition_loads_share_tags_and_preserve_resultants():
    domain = mesh.rectangle((0, 0), (2, 1), (4, 2), comm=MPI.COMM_SELF)
    boundaries = mesh.partition_boundaries(
        domain, left=lambda x: np.isclose(x[0], 0),
        right=lambda x: np.isclose(x[0], 2),
    )
    u = fields.displacement(domain)
    left, right = boundaries["left"], boundaries["right"]
    assert left.facet_tags is right.facet_tags
    force = (operators.boundary_load_vector(loads.traction((0, 2), on=left), target=u)
             + operators.boundary_load_vector(loads.traction((0, 3), on=right), target=u))
    vector = operators.assemble_vector(force)
    # Partition of unity: sum of vertical nodal forces is integrated traction.
    assert np.sum(vector.array.reshape((-1, 2))[:, 1]) == pytest.approx(5)
    vector.destroy()


def test_boundary_partition_rejects_overlapping_facets():
    domain = mesh.rectangle((0, 0), (2, 1), (2, 1), comm=MPI.COMM_SELF)
    with pytest.raises(ValueError, match="overlap"):
        mesh.partition_boundaries(domain, a=lambda x: np.isclose(x[0], 0),
                                  b=lambda x: np.isclose(x[0], 0))
