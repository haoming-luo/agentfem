# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from hashlib import sha256
import numpy as np
import pytest
from mpi4py import MPI
from agentfem import checkpointing, mesh


@pytest.mark.parametrize('dim,width', [(1,2),(2,3),(2,9),(3,4),(3,10),(3,27)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_chunked_keys_preserve_legacy_cell_bytes(dim, width, dtype):
    rng = np.random.default_rng(98123)
    coordinates = rng.normal(size=(250,dim)).astype(dtype)
    coordinates[4] = coordinates[5]  # repeated nodes/keys keep the same ordering
    cells = rng.integers(0,250,size=(27,width))
    lower = coordinates.min(axis=0).astype(np.float64)
    policy = lower, 64*np.finfo(np.float64).eps*8
    reference = []
    for cell in cells:
        points = np.asarray(coordinates[cell], dtype=np.float64)
        keys = np.rint((points-lower)/policy[1]).astype(np.int64)
        keys = keys[np.lexsort(tuple(keys[:,i] for i in reversed(range(dim))))]
        reference.append(sha256(np.ascontiguousarray(keys).tobytes()).hexdigest())
    for size in (1, width*dim*4, 65536):
        assert checkpointing._cell_geometry_signatures(coordinates,cells,policy=policy,chunk_scalars=size) == reference
    assert checkpointing._cell_geometry_signatures(coordinates,cells[:0],policy=policy) == []


@pytest.mark.parametrize('dimension,degree', [(2,1),(2,2),(3,1),(3,2)])
def test_real_curved_mesh_identity_preserves_reference_hash(dimension, degree):
    axes = tuple(np.linspace(0.,1.,2*degree+1) for _ in range(dimension))
    def mapping(x):
        y=x.copy(); y[-1] += .05*np.sin(np.pi*x[0]); return y
    d = mesh.mapped_grid(axes=axes,mapping=mapping,
        cell_type='quadrilateral' if dimension==2 else 'hexahedron',geometry_degree=degree,comm=MPI.COMM_WORLD)
    policy = checkpointing._coordinate_key_policy(d)
    local=[]
    for nodes in d.geometry.dofmaps[0][:d.topology.index_map(d.topology.dim).size_local]:
        keys=checkpointing._quantized_coordinate_keys(d.geometry.x[nodes],d,policy=policy)
        keys=keys[checkpointing._coordinate_order(keys)]
        local.append(sha256(np.ascontiguousarray(keys).tobytes()).hexdigest())
    hashes=sorted(v for part in d.comm.allgather(local) for v in part)
    digest=sha256(); digest.update(str(d.topology.cell_name()).encode())
    for value in hashes:
        digest.update(value.encode('ascii'))
    identity=checkpointing.mesh_portable_identity(d)
    assert identity['geometry_connectivity_hash'] == digest.hexdigest()
    assert identity['global_cells'] == len(hashes)


def test_single_precision_extrema_use_matching_collective_buffers():
    from types import SimpleNamespace
    class Comm:
        def Allreduce(self, send, receive, op):
            assert send.dtype == receive.dtype == np.dtype('float64')
            receive[:] = send
    d = SimpleNamespace(geometry=SimpleNamespace(x=np.array([[1.,2.],[3.,4.]], dtype=np.float32), dim=2), comm=Comm())
    lower,tolerance=checkpointing._coordinate_key_policy(d)
    np.testing.assert_array_equal(lower,[1.,2.])
    assert tolerance == 64*np.finfo(np.float64).eps*4


def test_actual_single_precision_mesh_has_correct_global_bounds():
    from dolfinx import mesh as dm
    d = dm.create_unit_square(MPI.COMM_WORLD,4,4,dtype=np.float32)
    assert d.geometry.x.dtype == np.dtype('float32')
    lower,tolerance=checkpointing._coordinate_key_policy(d)
    np.testing.assert_array_equal(lower,[0.,0.])
    assert tolerance == 64*np.finfo(np.float64).eps
    assert checkpointing.mesh_portable_identity(d)['global_cells'] == 32
