"""Degree-of-freedom location and vector ownership helpers."""

from __future__ import annotations

from dolfinx import fem


def locate_dofs(V, marker):
    """Locate dofs in a scalar function space by a geometrical marker."""

    return fem.locate_dofs_geometrical(V, marker)


def locate_component_dofs(V, component: int, marker):
    """Locate parent dofs for one component of a vector function space."""

    Vc, _ = V.sub(component).collapse()
    parent_dofs, _ = fem.locate_dofs_geometrical((V.sub(component), Vc), marker)
    return parent_dofs


def owned_size(function: fem.Function) -> int:
    """Number of owned scalar entries in a DOLFINx function vector."""

    dofmap = function.function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def owned_array(function: fem.Function):
    """Writable owned slice of a DOLFINx function vector."""

    return function.x.array[: owned_size(function)]


def assign_owned(function: fem.Function, values) -> None:
    """Assign owned entries and update ghost entries."""

    owned_array(function)[:] = values
    function.x.scatter_forward()


def copy_function(target: fem.Function, source: fem.Function) -> None:
    """Copy one finite-element function into another and update ghosts."""

    target.x.array[:] = source.x.array
    target.x.scatter_forward()
