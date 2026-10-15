# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import numpy as np
import pytest
from agentfem import mesh, spaces


def archive(tmp_path, *, disconnected=False, mutate=None):
    if disconnected:
        cube = np.array([[x,y,z] for z in (0.,1.) for y in (0.,1.) for x in (0.,1.)])
        coordinates = np.concatenate((cube, cube + [1.,0.,0.]))
        cells = np.arange(16).reshape(2,8)
    else:
        coordinates = np.array([[x,y,z] for z in (0.,1.) for y in (0.,1.) for x in (0.,1.,2.)])
        cells = np.array([[0,1,3,4,6,7,9,10], [1,2,4,5,7,8,10,11]])
    arrays = dict(cells=cells[::-1], coordinates=coordinates, part=np.array([2,1]))
    if mutate:
        mutate(arrays)
    path = tmp_path/'mesh.npz'
    np.savez(path, **arrays)
    record = dict(schema='agentfem.array-mesh.v1', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cell_type='hexahedron', cell_order='dolfinx_basix', geometry_degree=1, length_unit='m',
        source={'description':'two unit cells'})
    path.with_suffix('.json').write_text(json.dumps(record))
    return path, record


def area(region):
    from dolfinx import fem
    return fem.assemble_scalar(fem.form(1.*region.ds(region.tag)))


@pytest.mark.parametrize('disconnected', [False, True])
def test_input_identity_and_exterior_not_material_interface(tmp_path, disconnected):
    path, _ = archive(tmp_path, disconnected=disconnected)
    data = mesh.read_array_mesh(path)
    regions = mesh.exterior_regions(data.domain, input_cell_tags=data.arrays['part'],
        names={'left':1, 'right':2})
    assert regions.names == ('left','right')
    assert area(regions.left) == pytest.approx(6. if disconnected else 5.)
    assert area(regions.right) == pytest.approx(6. if disconnected else 5.)
    assert area(mesh.exterior_boundary(data.domain)) == pytest.approx(12. if disconnected else 10.)
    # Input cell reorder must not swap left and right (check boundary centroid).
    import ufl
    from dolfinx import fem
    x = ufl.SpatialCoordinate(data.domain)
    for name, expected in [('left', .5 if disconnected else .4), ('right',1.5 if disconnected else 1.6)]:
        region=regions[name]
        centroid=fem.assemble_scalar(fem.form(x[0]*region.ds(region.tag)))/area(region)
        assert centroid == pytest.approx(expected)
    if disconnected:
        space=spaces.vector_space(data.domain)
        # These nodes coincide but are not the same topological node.
        dofs=spaces.locate_input_nodes(space,np.array([1,8]))
        assert dofs[0] != dofs[1]
    with pytest.raises(ValueError):
        data.arrays['cells'].setflags(write=True)
    with pytest.raises(TypeError):
        data.arrays['new'] = np.zeros(1)
    metadata=data.metadata; metadata['source']['description']='changed'
    assert data.metadata['source']['description']=='two unit cells'


@pytest.mark.parametrize('change, message', [
    (lambda a:a.update(cells=a['cells'].astype(float)), 'CELLS'),
    (lambda a:a['cells'].__setitem__((0,0),100), 'INDEX'),
    (lambda a:a['cells'].__setitem__((0,0),a['cells'][0,1]), 'CELLS'),
    (lambda a:a['coordinates'].__setitem__((0,0),np.nan), 'ARRAY'),
])
def test_archive_rejects_bad_arrays(tmp_path, change, message):
    path,_=archive(tmp_path,mutate=change)
    with pytest.raises(ValueError,match=message):
        mesh.read_array_mesh(path)


@pytest.mark.parametrize('key,value,error', [
    ('cell_order','vtk',ValueError), ('geometry_degree',2,NotImplementedError),
    ('cell_type','pyramid',NotImplementedError), ('length_unit','',ValueError),
    ('schema','private-mesh',ValueError), ('sha256','incorrect',ValueError),
])
def test_archive_manifest_guards(tmp_path,key,value,error):
    path,record=archive(tmp_path); record[key]=value
    path.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(error):
        mesh.read_array_mesh(path)


def test_region_admission_and_no_implicit_partition(tmp_path):
    path,_=archive(tmp_path); data=mesh.read_array_mesh(path)
    for values,names in [(np.array([1]),{'a':1}), (np.array([1.,2.]),{'a':1}),
                         (np.array([1,2]),{'a':1,'b':1}), (np.array([1,2]),{'absent':3})]:
        with pytest.raises(ValueError,match='AFM-MESH'):
            mesh.exterior_regions(data.domain,input_cell_tags=values,names=names)


def test_distributed_archive_rejected_before_file_access():
    from types import SimpleNamespace
    with pytest.raises(NotImplementedError, match='ARCHIVE-MPI'):
        mesh.read_array_mesh('not-opened.npz', comm=SimpleNamespace(size=2))


@pytest.mark.parametrize('cell_type,cells,coordinates', [
    ('triangle',[[0,1,2]],[[0.,0.],[1.,0.],[0.,1.]]),
    ('quadrilateral',[[0,1,2,3]],[[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),
    ('tetrahedron',[[0,1,2,3]],[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
    ('interval',[[0,1]],[[0.],[1.]]),
])
def test_other_degree_one_topologies(tmp_path,cell_type,cells,coordinates):
    path,record=archive(tmp_path)
    np.savez(path,cells=np.array(cells),coordinates=np.array(coordinates))
    record.update(cell_type=cell_type,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(record))
    data=mesh.read_array_mesh(path)
    assert data.domain.topology.index_map(data.domain.topology.dim).size_local==1
    if cell_type!='interval':
        assert area(mesh.exterior_boundary(data.domain))>0
