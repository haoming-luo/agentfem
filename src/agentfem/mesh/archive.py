# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Checked serial array meshes; input identities are never coordinate matches."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
from mpi4py import MPI


@dataclass(frozen=True)
class ArrayMesh:
    """Ordinary mesh, immutable input arrays and detached provenance metadata.

    Extra arrays retain application-defined meaning; they are not inferred
    boundary conditions, material assignments or contact definitions.
    """
    domain: object
    arrays: object
    _metadata_json: str

    @property
    def metadata(self):
        return json.loads(self._metadata_json)


def read_array_mesh(path, *, manifest=None, comm=None):
    """Read a checksummed NPZ and ``agentfem.array-mesh.v1`` JSON manifest.

    The manifest declares ``sha256``, ``cell_type``, ``cell_order`` (currently
    ``dolfinx_basix``), ``geometry_degree`` (currently 1), and ``length_unit``.
    Named numeric arrays ``cells`` and ``coordinates`` are required. Other
    numeric arrays are retained without guessing their physical meaning.
    Hash and decode the same byte snapshot, reject invalid topology before
    backend construction, and never merge coincident independent nodes.
    This is serial mesh transport, not mesh-quality or engineering validation.
    """
    import basix.ufl
    from . import from_arrays

    comm = MPI.COMM_WORLD if comm is None else comm
    if comm.size != 1:
        raise NotImplementedError("AFM-MESH-ARCHIVE-MPI: serial input only")
    path = Path(path)
    record = json.loads((path.with_suffix('.json') if manifest is None else Path(manifest)).read_text())
    if not isinstance(record, dict) or record.get('schema') != 'agentfem.array-mesh.v1':
        raise ValueError("AFM-MESH-ARCHIVE-SCHEMA: unsupported mesh manifest")
    if record.get('cell_order') != 'dolfinx_basix':
        raise ValueError("AFM-MESH-ARCHIVE-ORDER: explicit dolfinx_basix order required")
    degree = record.get('geometry_degree')
    if type(degree) is not int or degree != 1:
        raise NotImplementedError("AFM-MESH-ARCHIVE-DEGREE: only degree-one geometry is admitted")
    types = {'interval': (2, 1), 'triangle': (3, 2), 'quadrilateral': (4, 2),
             'tetrahedron': (4, 3), 'hexahedron': (8, 3)}
    cell_type = record.get('cell_type')
    if not isinstance(cell_type, str) or cell_type not in types:
        raise NotImplementedError("AFM-MESH-ARCHIVE-TOPOLOGY: unsupported cell type")
    if not isinstance(record.get('length_unit'), str) or not record['length_unit'].strip():
        raise ValueError("AFM-MESH-ARCHIVE-UNIT: declare coordinate length unit; no implicit conversion")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.get('sha256'):
        raise ValueError("AFM-MESH-ARCHIVE-HASH: mesh archive integrity mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError("AFM-MESH-ARCHIVE-ARRAY: duplicate array names")
        arrays = {}
        for name in saved.files:
            value = saved[name]
            if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
                raise ValueError(f"AFM-MESH-ARCHIVE-ARRAY: nonfinite or nonnumeric array {name}")
            arrays[name] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if not {'cells', 'coordinates'} <= arrays.keys():
        raise ValueError("AFM-MESH-ARCHIVE-ARRAY: cells and coordinates are required")
    cells, points = arrays['cells'], arrays['coordinates']
    width, dimension = types[cell_type]
    if (cells.ndim != 2 or cells.shape[1] != width or not len(cells)
            or cells.dtype.kind not in 'iu'):
        raise ValueError("AFM-MESH-ARCHIVE-CELLS: connectivity shape/type does not match cell_type")
    if (points.ndim != 2 or not dimension <= points.shape[1] <= 3
            or not len(points) or points.dtype.kind not in 'iuf'):
        raise ValueError("AFM-MESH-ARCHIVE-POINTS: invalid coordinate shape/type")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("AFM-MESH-ARCHIVE-INDEX: connectivity index outside input coordinates")
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("AFM-MESH-ARCHIVE-CELLS: repeated node within a cell")
    element = basix.ufl.element('Lagrange', cell_type, 1, shape=(points.shape[1],))
    domain = from_arrays(cells=cells, coordinates=points, coordinate_element=element, comm=comm)
    return ArrayMesh(domain, MappingProxyType(arrays), json.dumps(record, allow_nan=False))


def exterior_regions(domain, *, input_cell_tags, names):
    """Select global exterior facets by the owning cell's input-order tag.

    ``names`` maps region names to distinct integer tag values. The input array
    contains exactly one tag per input cell, before backend cell reordering.
    Only global exterior facets are selected: an interior material interface
    is not turned into a boundary. Coincident topologically independent parts
    remain independent. Serial only; no assumption about distributed input IDs.
    Returns the existing RegionSet/BoundaryRegion objects, not a new surface type.
    """
    from dolfinx import mesh as dxmesh
    from . import tagged_boundary_region
    from .regions import RegionSet

    domain = getattr(domain, 'domain', domain)
    if domain.comm.size != 1:
        raise NotImplementedError("AFM-MESH-EXTERIOR-MPI: input-order cell tags require serial mesh")
    if np.ma.isMaskedArray(input_cell_tags):
        raise ValueError("AFM-MESH-CELL-TAGS: masked tags are not supported")
    values = np.asarray(input_cell_tags)
    count = domain.topology.index_map(domain.topology.dim).size_local
    if values.shape != (count,) or values.dtype.kind not in 'iu':
        raise ValueError("AFM-MESH-CELL-TAGS: one integer tag per input cell required")
    limits = np.iinfo(np.int32)
    if np.any(values < limits.min) or np.any(values > limits.max):
        raise ValueError("AFM-MESH-CELL-TAGS: tags must fit int32")
    if (not isinstance(names, dict) or not names
            or any(not isinstance(n, str) or not n or type(v) is not int for n, v in names.items())
            or len(set(names.values())) != len(names)):
        raise ValueError("AFM-MESH-REGION-NAMES: require names mapped to distinct integer tags")
    original = np.asarray(domain.topology.original_cell_index)
    if original.shape != (count,) or not np.array_equal(np.sort(original), np.arange(count)):
        raise ValueError("AFM-MESH-CELL-IDENTITY: input cell indices must form a serial permutation")
    facets, owners = _exterior_owners(domain)
    boundary_values = values[original[owners]].astype(np.int32)
    missing = set(names.values()) - set(boundary_values.tolist())
    if missing:
        raise ValueError(f"AFM-MESH-EMPTY-EXTERIOR: requested tags have no exterior facets: {sorted(missing)}")
    tags = dxmesh.meshtags(domain, domain.topology.dim-1, facets, boundary_values)
    regions = {name: tagged_boundary_region(domain, tags, tag=tag, name=name)
               for name, tag in names.items()}
    return RegionSet(domain, regions, tags=tags, kind='boundary')


def exterior_boundary(domain, *, name='exterior'):
    """Return all exterior facets as an ordinary named BoundaryRegion (serial)."""
    from dolfinx import mesh as dxmesh
    from . import tagged_boundary_region
    domain = getattr(domain, 'domain', domain)
    if domain.comm.size != 1:
        raise NotImplementedError("AFM-MESH-EXTERIOR-MPI: serial boundary extraction only")
    facets, _ = _exterior_owners(domain)
    tags = dxmesh.meshtags(domain, domain.topology.dim-1, facets, np.ones(len(facets), dtype=np.int32))
    return tagged_boundary_region(domain, tags, tag=1, name=name)


def _exterior_owners(domain):
    from dolfinx import mesh as dxmesh
    dimension = domain.topology.dim
    if dimension < 2:
        raise NotImplementedError("AFM-MESH-EXTERIOR-DIMENSION: require a 2D or 3D mesh")
    domain.topology.create_connectivity(dimension-1, dimension)
    facets = dxmesh.exterior_facet_indices(domain.topology)
    parent = domain.topology.connectivity(dimension-1, dimension)
    owners = np.array([parent.links(int(facet))[0] for facet in facets], dtype=np.int32)
    return facets, owners
