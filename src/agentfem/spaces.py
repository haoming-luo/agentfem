# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Function-space construction helpers."""

from __future__ import annotations

import basix
import basix.ufl
import ufl
from dolfinx import fem


def locate_input_nodes(space, input_node_indices):
    """Map original mesh input indices to scalar/block P1/Q1 DOFs, in order.

    Serial degree-one coordinate and continuous nodal spaces only. Input
    indices are ``mesh.geometry.input_global_indices``, not source-solver node
    labels. The backend's entity layouts determine the map; coordinates are not
    matched, so coincident independent interface nodes remain independent.
    Returned indices address blocks, not flattened vector components.
    """
    import numpy as np
    domain = space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError("Input-node lookup currently requires a serial mesh")
    if np.ma.isMaskedArray(input_node_indices):
        raise ValueError("Input-node indices cannot be masked")
    requested = np.asarray(input_node_indices)
    if requested.ndim != 1 or requested.dtype.kind not in 'iu':
        raise ValueError("Input-node indices must be an integer vector")
    element = space.element.basix_element
    maps = getattr(domain.geometry, 'cmaps', None)
    if maps is not None and len(maps) != 1:
        raise NotImplementedError("Input-node lookup requires one coordinate element")
    coordinate = maps[0] if maps is not None else domain.geometry.cmap
    vertices = basix.cell.topology(element.cell_type)[0]
    layout = coordinate.create_dof_layout()
    geometric = [layout.entity_dofs(0, vertex) for vertex in range(len(vertices))]
    field = element.entity_dofs[0]
    if (element.degree != 1 or element.discontinuous or coordinate.degree != 1
            or any(len(dofs) != 1 for dofs in (*geometric, *field))
            or element.dim != len(vertices)):
        raise NotImplementedError("Input-node lookup requires continuous nodal P1/Q1 and degree-one geometry")
    count = domain.topology.index_map(domain.topology.dim).size_local
    gmap = np.asarray(domain.geometry.dofmaps[0])[:count, np.array(geometric)[:, 0]]
    dofs = np.asarray(space.dofmap.list)[:count, np.array(field)[:, 0]]
    ids = np.asarray(domain.geometry.input_global_indices)[gmap]
    size = space.dofmap.index_map.size_local
    low, high = np.full(size, np.iinfo(np.int64).max), np.full(size, -1, dtype=np.int64)
    np.minimum.at(low, dofs.ravel(), ids.ravel())
    np.maximum.at(high, dofs.ravel(), ids.ravel())
    if np.any(low != high) or len(np.unique(low)) != size:
        raise ValueError("Input-node identity does not define a one-to-one nodal DOF map")
    order = np.argsort(low)
    positions = np.searchsorted(low[order], requested)
    if np.any(positions >= size) or not np.array_equal(low[order[positions]], requested):
        raise ValueError("Requested input node is absent from this function space")
    selected = np.ascontiguousarray(order[positions], dtype=np.int32)
    return np.frombuffer(selected.tobytes(), dtype=selected.dtype)


def lagrange_space(domain, degree: int = 1):
    """Create a scalar Lagrange function space."""

    domain = _domain(domain)
    return fem.functionspace(domain, ("Lagrange", degree))


def scalar_space(domain, degree: int = 1):
    """Create a scalar Lagrange function space."""

    return lagrange_space(domain, degree=degree)


def vector_lagrange_space(domain, degree: int = 1, dim: int | None = None):
    """Create a vector Lagrange function space."""

    domain = _domain(domain)
    value_dim = domain.geometry.dim if dim is None else dim
    return fem.functionspace(domain, ("Lagrange", degree, (value_dim,)))


def vector_space(domain, degree: int = 1, dim: int | None = None):
    """Create a vector Lagrange function space."""

    return vector_lagrange_space(domain, degree=degree, dim=dim)


def velocity_pressure_space(
    domain,
    *,
    velocity_degree: int = 2,
    pressure_degree: int = 1,
):
    """Create a Taylor--Hood velocity/pressure mixed space.

    The default ``P2/P1`` pair is the standard conforming choice for
    incompressible Stokes and Navier--Stokes flow. The interpolation policy is
    explicit because a mesh alone never determines a stable mixed method.
    """

    domain = _domain(domain)
    if int(velocity_degree) < 2:
        raise ValueError("velocity_degree must be at least two for Taylor--Hood.")
    if int(pressure_degree) < 1:
        raise ValueError("pressure_degree must be at least one.")
    if int(velocity_degree) <= int(pressure_degree):
        raise ValueError(
            "Taylor--Hood requires velocity_degree greater than pressure_degree."
        )
    cell = domain.basix_cell()
    velocity_element = basix.ufl.element(
        "Lagrange",
        cell,
        int(velocity_degree),
        shape=(int(domain.geometry.dim),),
    )
    pressure_element = basix.ufl.element(
        "Lagrange",
        cell,
        int(pressure_degree),
    )
    return fem.functionspace(
        domain,
        basix.ufl.mixed_element((velocity_element, pressure_element)),
    )


def independent_subspace(space, component: int):
    """Build an independently numbered space for one mixed-field component.

    Use it for boundary interpolation or standalone output. Its coefficient
    numbering is not the parent numbering: transfer Functions by interpolation,
    and locate boundary dofs with ``(space.sub(component), independent)``.
    The finite element (including degree and variant) is preserved.
    """
    selected = space.sub(component)
    return fem.functionspace(space.mesh, selected.ufl_element())


def displacement_pressure_space(
    domain,
    *,
    displacement_degree: int = 2,
    pressure_degree: int = 0,
    pressure_family: str = "DG",
):
    """Create the mixed ``H1`` displacement / discontinuous-pressure space.

    The default ``P2/DG0`` pair provides one constant pressure unknown per
    cell.  This is AgentFEM's explicit mixed-field analogue for constant-
    pressure hybrid solid formulations such as Abaqus ``C3D10H``; the mesh
    topology alone never selects this formulation implicitly.  ``DPC`` is an
    explicit alternative on quadrilateral and hexahedral cells; in particular,
    ``P2/DPC1`` gives the complete three-mode discontinuous pressure space on a
    quadrilateral without changing the public mixed-field contract.
    """

    domain = _domain(domain)
    if int(displacement_degree) < 1:
        raise ValueError("displacement_degree must be at least one.")
    if int(pressure_degree) < 0:
        raise ValueError("pressure_degree must be non-negative.")
    cell = domain.basix_cell()
    displacement_element = basix.ufl.element(
        "Lagrange",
        cell,
        int(displacement_degree),
        shape=(int(domain.geometry.dim),),
    )
    pressure_element = _discontinuous_pressure_element(
        cell,
        degree=int(pressure_degree),
        family=pressure_family,
    )
    return fem.functionspace(
        domain,
        basix.ufl.mixed_element((displacement_element, pressure_element)),
    )


def _discontinuous_pressure_element(
    cell: basix.CellType,
    *,
    degree: int,
    family: str,
):
    """Construct from the deliberately small public pressure-family set."""

    if not isinstance(family, str):
        raise TypeError("pressure_family must be a string: 'DG' or 'DPC'.")
    selected = family.strip().upper()
    if selected == "DG":
        return basix.ufl.element("DG", cell, degree)
    if selected != "DPC":
        raise ValueError("pressure_family must be either 'DG' or 'DPC'.")
    supported_cells = {
        basix.CellType.quadrilateral,
        basix.CellType.hexahedron,
    }
    if cell not in supported_cells:
        raise ValueError(
            "DPC pressure interpolation requires a quadrilateral or "
            f"hexahedron cell; received {cell.name}."
        )
    return basix.ufl.element(
        basix.ElementFamily.DPC,
        cell,
        degree,
        dpc_variant=basix.DPCVariant.legendre,
        discontinuous=True,
    )


def test_function(V):
    """Create a UFL test function for a function space."""

    return ufl.TestFunction(V)


def trial_function(V):
    """Create a UFL trial function for a function space."""

    return ufl.TrialFunction(V)


def named_function(V, name: str, value=0.0):
    """Create a named finite-element function and optionally initialize it."""

    function = fem.Function(V, name=name)
    function.x.array[:] = value
    function.x.scatter_forward()
    return function


def _domain(mesh_or_domain):
    """Accept a DOLFINx mesh or an AgentFEM imported-mesh facade."""

    return getattr(mesh_or_domain, "domain", mesh_or_domain)
