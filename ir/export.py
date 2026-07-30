"""Export current AgentFEM objects into versioned AF-IR records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Mapping

from .schema import IRDocument, to_json_safe


WORKFLOW_ORDER = (
    "study",
    "mesh",
    "regions",
    "fields",
    "materials",
    "amplitudes",
    "constraints",
    "loads",
    "boundary_models",
    "steps",
)


def model_document(
    model,
    *,
    agentfem_version: str,
    backend: Mapping[str, object] | None = None,
    include_validation: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> IRDocument:
    """Build an experimental AF-IR model document.

    The exporter records supported public semantics and marks unknown backend
    values as opaque.  It does not claim that this record can yet reconstruct
    every UFL expression or execute on a second backend.
    """

    root: dict[str, object] = {
        "kind": "finite_element_model",
        "name": getattr(model, "name", "model"),
        "workflow_order": WORKFLOW_ORDER,
        "study": describe(getattr(model, "study", None)),
        "mesh": describe_mesh(getattr(model, "mesh", None)),
        "regions": describe_many(getattr(model, "regions", ())),
        "fields": describe_many(getattr(model, "fields", ())),
        "materials": describe_many(getattr(model, "materials", ())),
        "amplitudes": describe_many(getattr(model, "amplitudes", ())),
        "constraints": describe_many(getattr(model, "constraints", ())),
        "loads": describe_many(getattr(model, "loads", ())),
        "boundary_models": describe_many(
            getattr(model, "boundary_models", ())
        ),
        "steps": describe_many(getattr(model, "steps", ())),
    }
    if include_validation and hasattr(model, "validate"):
        root["validation"] = model.validate().as_dict()
    if backend is not None:
        root["execution_backend"] = dict(backend)

    return IRDocument(
        document_type="model",
        root=to_json_safe(root),
        generator={
            "name": "AgentFEM",
            "version": agentfem_version,
            "exporter": "agentfem.ir.model_document",
        },
        metadata={} if metadata is None else dict(metadata),
    )


def describe_many(items: Iterable[object]) -> tuple[object, ...]:
    """Describe a collection without retaining backend memory addresses."""

    return tuple(describe(item) for item in items)


def describe(item):
    """Prefer semantic records over display-only summaries."""

    if item is None:
        return None
    if hasattr(item, "item") and hasattr(item, "region"):
        return {
            "kind": "regional_assignment",
            "item": describe(item.item),
            "region": _reference(item.region),
        }
    for method_name in ("to_ir", "as_dict", "summary"):
        method = getattr(item, method_name, None)
        if callable(method):
            result = method()
            if result is not item:
                return to_json_safe(result)
    return to_json_safe(item)


def describe_mesh(mesh):
    """Record partition-independent mesh facts without claiming reconstruction."""

    if mesh is None:
        return None
    mesh = getattr(mesh, "domain", mesh)
    topology = getattr(mesh, "topology", None)
    geometry = getattr(mesh, "geometry", None)
    result: dict[str, object] = {
        "kind": "mesh",
        "backend_type": _qualified_type(mesh),
        "representation": "runtime_mesh_summary",
        "reconstructable": False,
    }
    if topology is not None and hasattr(topology, "dim"):
        result["topological_dim"] = int(topology.dim)
        index_map = getattr(topology, "index_map", None)
        if callable(index_map):
            try:
                cells = index_map(topology.dim)
                result["global_cells"] = int(cells.size_global)
            except (AttributeError, RuntimeError, TypeError):
                pass
    if geometry is not None and hasattr(geometry, "dim"):
        result["geometric_dim"] = int(geometry.dim)
    return result


def _reference(item):
    if item is None:
        return None
    return {
        "name": getattr(item, "name", None),
        "kind": getattr(item, "kind", type(item).__name__),
        "tag": getattr(item, "tag", None),
    }


def _qualified_type(item) -> str:
    item_type = type(item)
    return f"{item_type.__module__}.{item_type.__qualname__}"


__all__ = ["WORKFLOW_ORDER", "describe", "describe_many", "model_document"]
