"""Degree-of-freedom location and vector ownership helpers."""

from __future__ import annotations

from dolfinx import fem


def _tagged_location(location):
    selection = getattr(location, "selection", None)
    if selection not in {"tagged", "hybrid"}:
        return None
    domain = location.domain
    return domain.topology.dim - 1, location.facets


def locate_dofs(V, marker):
    """Locate dofs from a geometric marker or a tagged boundary region."""

    tagged = _tagged_location(marker)
    if tagged is not None:
        entity_dim, entities = tagged
        return fem.locate_dofs_topological(V, entity_dim, entities)
    geometric_marker = getattr(marker, "marker", marker)
    return fem.locate_dofs_geometrical(V, geometric_marker)


def locate_component_dofs(V, component: int, marker):
    """Locate parent dofs for one vector component on a marker or region."""

    tagged = _tagged_location(marker)
    if tagged is not None:
        entity_dim, entities = tagged
        return fem.locate_dofs_topological(V.sub(component), entity_dim, entities)
    geometric_marker = getattr(marker, "marker", marker)
    Vc, _ = V.sub(component).collapse()
    parent_dofs, _ = fem.locate_dofs_geometrical(
        (V.sub(component), Vc), geometric_marker
    )
    return parent_dofs


def owned_size(function: fem.Function) -> int:
    """Number of owned scalar entries in a DOLFINx function vector."""

    function = _unwrap(function)
    dofmap = function.function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def owned_array(function: fem.Function):
    """Writable owned slice of a DOLFINx function vector."""

    function = _unwrap(function)
    return function.x.array[: owned_size(function)]


def assign_owned(function: fem.Function, values) -> None:
    """Assign owned entries and update ghost entries."""

    function = _unwrap(function)
    owned_array(function)[:] = values
    function.x.scatter_forward()


def copy_function(target: fem.Function, source: fem.Function) -> None:
    """Copy one finite-element function into another and update ghosts."""

    target = _unwrap(target)
    source = _unwrap(source)
    target.x.array[:] = source.x.array
    target.x.scatter_forward()


def _unwrap(function):
    return function.function if hasattr(function, "function") and hasattr(function, "x") else function
