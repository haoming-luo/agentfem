"""Shared, side-effect-free helpers for the engineering Model boundary.

The helpers in this module inspect registered assets without assembling forms,
advancing a procedure, or importing :mod:`agentfem.models`.  They are shared by
model validation and model inspection so those concerns do not grow back into
the public facade.
"""

from __future__ import annotations


def describe(item):
    """Return a JSON-friendly description when an asset provides one."""

    if item is None:
        return None
    if hasattr(item, "summary"):
        return item.summary()
    if hasattr(item, "as_dict"):
        return item.as_dict()
    return getattr(item, "name", repr(item))


def short_description(item) -> str:
    """Return a compact human-readable label for a model tree."""

    if item is None:
        return "<none>"
    if hasattr(item, "topology") and hasattr(item, "geometry"):
        return (
            f"mesh: tdim={item.topology.dim}, "
            f"gdim={item.geometry.dim}, "
            f"cells={item.topology.index_map(item.topology.dim).size_global}"
        )
    name = getattr(item, "name", None)
    kind = getattr(item, "kind", None)
    if name is not None and kind is not None:
        return f"{kind}: {name}"
    if name is not None:
        return str(name)
    if hasattr(item, "summary"):
        summary = item.summary()
        if isinstance(summary, dict):
            if "item" in summary and "region" in summary:
                region = summary["region"] or "whole domain"
                return f"material: {summary['item']} on {region}"
            if "dirichlet" in summary:
                count = len(summary.get("dirichlet", ()))
                return f"constraint set: {count} dirichlet"
            summary_name = summary.get("name")
            summary_kind = summary.get("kind") or summary.get("analysis")
            if summary_name is not None and summary_kind is not None:
                return f"{summary_kind}: {summary_name}"
            if summary_name is not None:
                return str(summary_name)
            return repr(summary)
    return repr(item)


def mesh_summary(mesh):
    """Return the stable dimensional part of a registered mesh summary."""

    if mesh is None:
        return None
    mesh = domain(mesh)
    return {
        "topological_dim": mesh.topology.dim,
        "geometric_dim": mesh.geometry.dim,
    }


def domain(mesh):
    """Return the DOLFINx domain stored directly or inside ``FEMMesh``."""

    if mesh is None:
        return None
    return getattr(mesh, "domain", mesh)


def field_domain(field_object):
    """Return a field mesh when it can be determined without assembly."""

    space = getattr(field_object, "space", None)
    if space is None:
        value = getattr(field_object, "value", field_object)
        space = getattr(value, "function_space", None)
    return getattr(space, "mesh", None)


def duplicate_names(items) -> tuple[str, ...]:
    """Return stable duplicate asset names for addressable validation."""

    counts: dict[str, int] = {}
    for item in items:
        name = getattr(item, "name", None)
        if name is None and hasattr(item, "summary"):
            summary = item.summary()
            if isinstance(summary, dict):
                name = summary.get("name")
        if isinstance(name, str) and name:
            counts[name] = counts.get(name, 0) + 1
    return tuple(sorted(name for name, count in counts.items() if count > 1))


__all__ = ()
