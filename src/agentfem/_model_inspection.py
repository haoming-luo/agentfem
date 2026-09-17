# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Inspection and serialization views for the engineering Model facade."""

from __future__ import annotations

from pathlib import Path

from ._model_support import describe, domain, mesh_summary, short_description


def model_summary(model) -> dict[str, object]:
    """Return an agent-readable model summary without mutating the model."""

    return {
        "name": model.name,
        "study": describe(model.study),
        "unit_system": describe(model.unit_system),
        "mesh": mesh_summary(model.mesh),
        "fields": tuple(describe(item) for item in model.fields),
        "amplitudes": tuple(describe(item) for item in model.amplitudes),
        "materials": tuple(describe(item) for item in model.materials),
        "constraints": tuple(describe(item) for item in model.constraints),
        "loads": tuple(describe(item) for item in model.loads),
        "boundary_models": tuple(describe(item) for item in model.boundary_models),
        "regions": tuple(describe(item) for item in model.regions),
        "engineering_steps": tuple(describe(item) for item in model.engineering_steps),
        "steps": tuple(describe(item) for item in model.steps),
    }


def model_manifest(model) -> dict[str, object]:
    """Return the stable machine-readable model manifest."""

    return {
        "kind": "agentfem_model_manifest",
        "version": 1,
        "schema": "agentfem.af-ir",
        "schema_version": "0.1.0",
        "status": "experimental",
        "model": model_summary(model),
        "workflow_order": (
            "study",
            "mesh",
            "regions",
            "fields",
            "materials",
            "constraints",
            "loads",
            "boundary_models",
            "engineering_steps",
            "steps",
        ),
    }


def model_to_ir(
    model,
    *,
    include_validation: bool = True,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Export supported model semantics as an AF-IR document."""

    from . import __version__
    from .backends import get_backend
    from .ir import model_document

    document = model_document(
        model,
        agentfem_version=__version__,
        backend=get_backend().descriptor.as_dict(),
        include_validation=include_validation,
        metadata=metadata,
    )
    return document.as_dict()


def write_model_ir(
    model,
    path,
    *,
    include_validation: bool = True,
    metadata: dict[str, object] | None = None,
):
    """Write a deterministic AF-IR JSON record and return its path."""

    from .ir import write_document

    output = Path(path)
    mesh_domain = domain(model.mesh)
    comm = getattr(mesh_domain, "comm", None)
    rank = getattr(comm, "rank", 0)
    if rank == 0:
        write_document(
            model_to_ir(
                model,
                include_validation=include_validation,
                metadata=metadata,
            ),
            output,
        )
    if comm is not None and hasattr(comm, "barrier"):
        comm.barrier()
    return output


def model_tree(model) -> str:
    """Return a compact text model tree for logs, notebooks, and agents."""

    sections = [
        ("study", (model.study,)),
        ("mesh", (model.mesh,) if model.mesh is not None else ()),
        ("regions", model.regions),
        ("fields", model.fields),
        ("materials", model.materials),
        ("constraints", model.constraints),
        ("loads", model.loads),
        ("boundary_models", model.boundary_models),
        ("steps", model.steps),
    ]
    lines = [f"Model: {model.name}"]
    for title, items in sections:
        lines.append(f"  {title}:")
        if not items:
            lines.append("    - <empty>")
            continue
        for item in items:
            lines.append(f"    - {short_description(item)}")
    return "\n".join(lines)


__all__ = ()
