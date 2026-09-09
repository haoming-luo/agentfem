"""Executable identities for operator-owned finite-element systems.

Scientific summaries explain an operator, but restart safety must bind the
operator that is actually executed.  This module combines UFL's canonical form
signature with live Constant/Function contents, MeshTags membership, mesh
connectivity and concrete homogeneous Dirichlet degrees of freedom.  Any
layout that cannot be made MPI-partition independent fails closed.
"""

from __future__ import annotations

from hashlib import sha256
import json

from dolfinx import mesh as mesh_api
from mpi4py import MPI
import numpy as np

from .. import fields
from ..provenance import content_fingerprint


def harmonic_executable_identity(system, *, solution, bcs=()) -> dict[str, object]:
    """Return a partition-neutral identity for an executable harmonic system."""

    function = fields.unwrap(solution)
    domain = function.function_space.mesh
    missing: list[dict[str, str]] = []
    operators = {}
    for name, operator in (
        ("storage", system.storage),
        ("mass", system.mass),
        ("damping", system.damping),
        ("loss", system.loss),
        ("force", system.force),
    ):
        if operator is None:
            operators[name] = None
            continue
        operators[name] = _form_identity(
            getattr(operator, "expression", operator),
            domain=domain,
            path=f"harmonic_system.{name}",
            missing=missing,
        )
    record = {
        "schema": "agentfem.harmonic-executable-identity.v1",
        "equation": system.equation,
        "phasor_convention": system.phasor_convention,
        "mesh": mesh_executable_identity(domain),
        # The DOLFINx wrapper repr contains a process-local memory address.
        # UFL's element description is deterministic for equivalent spaces and
        # therefore belongs in a portable executable identity.
        "target_element": str(function.ufl_element()),
        "operators": operators,
        "homogeneous_dirichlet": _homogeneous_dirichlet_identity(
            function,
            tuple(bcs),
            missing=missing,
        ),
    }
    return {
        "complete": not missing,
        "missing": tuple(missing),
        "record": record,
        "fingerprint": content_fingerprint(record),
    }


def mesh_executable_identity(domain) -> dict[str, object]:
    """Hash source-node connectivity and geometry independent of partition."""

    topology = domain.topology
    cell_map = topology.index_map(topology.dim)
    geometry_dofmap = np.asarray(domain.geometry.dofmaps[0])
    input_indices = np.asarray(domain.geometry.input_global_indices, dtype=np.int64)
    coordinates = np.asarray(domain.geometry.x, dtype=np.float64)[
        :, : int(domain.geometry.dim)
    ]
    policy = _coordinate_key_policy(domain)
    local = []
    for cell in range(int(cell_map.size_local)):
        geometry_dofs = np.asarray(geometry_dofmap[cell], dtype=np.int64)
        source_ids = input_indices[geometry_dofs]
        coordinate_keys = _coordinate_keys(
            coordinates[geometry_dofs], domain, policy=policy
        )
        nodes = sorted(
            (
                int(source_id),
                *(int(value) for value in coordinate),
            )
            for source_id, coordinate in zip(
                source_ids, coordinate_keys, strict=True
            )
        )
        local.append(nodes)
    cells = [item for rank_items in domain.comm.allgather(local) for item in rank_items]
    cells.sort()
    record = {
        "cell_type": str(topology.cell_name()),
        "topology_dimension": int(topology.dim),
        "geometry_dimension": int(domain.geometry.dim),
        "global_cells": len(cells),
        "cells": cells,
        "coordinate_key": "relative_bounds_scaled_int64_with_input_node_id",
    }
    return {
        key: value for key, value in record.items() if key != "cells"
    } | {"connectivity_sha256": content_fingerprint(record)}


def _form_identity(form, *, domain, path: str, missing) -> dict[str, object]:
    signature = getattr(form, "signature", None)
    if not callable(signature):
        missing.append({"path": path, "reason": "operator_is_not_a_ufl_form"})
        return {
            "python_type": f"{type(form).__module__}.{type(form).__qualname__}",
            "ufl_signature": None,
        }
    try:
        ufl_signature = str(signature())
    except Exception as exc:
        missing.append({"path": path, "reason": "ufl_signature_unavailable"})
        return {"ufl_signature": None, "error": f"{type(exc).__name__}: {exc}"}

    coefficients = []
    for index, coefficient in enumerate(tuple(form.coefficients())):
        try:
            coefficients.append(_function_content_identity(coefficient))
        except (AttributeError, NotImplementedError, RuntimeError, ValueError) as exc:
            missing.append(
                {
                    "path": f"{path}.coefficients[{index}]",
                    "reason": f"portable_coefficient_identity_unavailable:{exc}",
                }
            )
            coefficients.append(
                {
                    "python_type": (
                        f"{type(coefficient).__module__}."
                        f"{type(coefficient).__qualname__}"
                    ),
                    "content_sha256": None,
                }
            )
    constants = []
    for index, constant in enumerate(tuple(form.constants())):
        value = getattr(constant, "value", None)
        if value is None:
            missing.append(
                {
                    "path": f"{path}.constants[{index}]",
                    "reason": "constant_value_unavailable",
                }
            )
            constants.append({"content_sha256": None})
            continue
        constants.append(_array_identity(value))

    tags = []
    for _ufl_domain, by_integral_type in form.subdomain_data().items():
        for integral_type in sorted(by_integral_type):
            for item in by_integral_type[integral_type]:
                if item is None:
                    continue
                try:
                    identity = _meshtags_identity(domain, item)
                except (AttributeError, RuntimeError, ValueError) as exc:
                    missing.append(
                        {
                            "path": f"{path}.subdomain_data.{integral_type}",
                            "reason": f"portable_meshtags_identity_unavailable:{exc}",
                        }
                    )
                    identity = {"content_sha256": None}
                tags.append(
                    {
                        "integral_type": str(integral_type),
                        "identity": identity,
                    }
                )
    tags.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {
        "ufl_signature": ufl_signature,
        "coefficients": coefficients,
        "constants": constants,
        "subdomain_data": tags,
    }


def _function_content_identity(function) -> dict[str, object]:
    value = fields.unwrap(function)
    V = value.function_space
    if int(V.dofmap.bs) != int(V.dofmap.index_map_bs):
        raise NotImplementedError("mixed or subspace coefficient layout")
    index_map = V.dofmap.index_map
    owned = int(index_map.size_local)
    block_size = int(V.dofmap.index_map_bs)
    coordinates = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    if len(coordinates) < owned:
        raise ValueError("coefficient does not expose every owned dof coordinate")
    keys = _coordinate_keys(coordinates[:owned], V.mesh)
    values = np.asarray(value.x.array[: owned * block_size]).reshape(
        (owned, block_size)
    )
    local = [
        (
            tuple(int(item) for item in key),
            _array_identity(row),
        )
        for key, row in zip(keys, values, strict=True)
    ]
    rows = [item for rank_items in V.mesh.comm.allgather(local) for item in rank_items]
    rows.sort(key=lambda item: item[0])
    if any(left[0] == right[0] for left, right in zip(rows[:-1], rows[1:])):
        raise NotImplementedError("coincident coefficient dof coordinates")
    return {
        "element": str(value.ufl_element()),
        "value_shape": list(value.ufl_shape),
        "block_size": block_size,
        "global_block_dofs": len(rows),
        "content_sha256": content_fingerprint(rows),
        "key": "quantized_physical_dof_coordinate",
    }


def _meshtags_identity(domain, tags) -> dict[str, object]:
    dimension = int(tags.dim)
    index_map = domain.topology.index_map(dimension)
    if index_map is None:
        raise ValueError(f"mesh has no entity index map for dimension {dimension}")
    indices = np.asarray(tags.indices, dtype=np.int32)
    values = np.asarray(tags.values)
    owned = indices < int(index_map.size_local)
    selected_indices = indices[owned]
    selected_values = values[owned]
    if len(selected_indices):
        geometry_dofs = mesh_api.entities_to_geometry(
            domain, dimension, selected_indices, permute=False
        )
    else:
        geometry_dofs = np.empty((0, 0), dtype=np.int32)
    input_indices = np.asarray(domain.geometry.input_global_indices, dtype=np.int64)
    coordinates = np.asarray(domain.geometry.x, dtype=np.float64)[
        :, : int(domain.geometry.dim)
    ]
    policy = _coordinate_key_policy(domain)
    local = []
    for dofs, tag_value in zip(geometry_dofs, selected_values, strict=True):
        dofs = np.asarray(dofs, dtype=np.int64)
        source_ids = input_indices[dofs]
        coordinate_keys = _coordinate_keys(coordinates[dofs], domain, policy=policy)
        closure = sorted(
            (
                int(source_id),
                *(int(value) for value in coordinate),
            )
            for source_id, coordinate in zip(
                source_ids, coordinate_keys, strict=True
            )
        )
        local.append((int(tag_value), closure))
    entities = [
        item for rank_items in domain.comm.allgather(local) for item in rank_items
    ]
    entities.sort()
    return {
        "dimension": dimension,
        "global_tagged_entities": len(entities),
        "content_sha256": content_fingerprint(entities),
    }


def _homogeneous_dirichlet_identity(function, bcs, *, missing) -> dict[str, object]:
    V = function.function_space
    block_size = int(V.dofmap.index_map_bs)
    owned_scalar_dofs = int(V.dofmap.index_map.size_local) * block_size
    coordinates = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    keys = _coordinate_keys(
        coordinates[: int(V.dofmap.index_map.size_local)], V.mesh
    )
    local = set()
    for index, bc in enumerate(bcs):
        dof_indices = getattr(bc, "dof_indices", None)
        if not callable(dof_indices):
            missing.append(
                {
                    "path": f"harmonic_system.bcs[{index}]",
                    "reason": "concrete_dirichlet_dofs_unavailable",
                }
            )
            continue
        dofs, owned_count = dof_indices()
        for dof in np.asarray(dofs[: int(owned_count)], dtype=np.int64):
            if dof < 0 or dof >= owned_scalar_dofs:
                missing.append(
                    {
                        "path": f"harmonic_system.bcs[{index}]",
                        "reason": "dirichlet_dof_outside_target_space",
                    }
                )
                continue
            block, component = divmod(int(dof), block_size)
            local.add((*tuple(int(item) for item in keys[block]), component))
    constrained = [
        item
        for rank_items in V.mesh.comm.allgather(sorted(local))
        for item in rank_items
    ]
    constrained = sorted(set(constrained))
    return {
        "value": "homogeneous_zero",
        "global_scalar_dofs": len(constrained),
        "dof_set_sha256": content_fingerprint(constrained),
        "key": "quantized_physical_block_coordinate_and_component",
    }


def _array_identity(value) -> dict[str, object]:
    selected = np.ascontiguousarray(np.asarray(value))
    digest = sha256(selected.tobytes(order="C")).hexdigest()
    return {
        "dtype": selected.dtype.str,
        "shape": list(selected.shape),
        "sha256": digest,
    }


def _coordinate_key_policy(domain) -> tuple[np.ndarray, float]:
    coordinates = np.asarray(domain.geometry.x, dtype=np.float64)[
        :, : int(domain.geometry.dim)
    ]
    dimension = int(domain.geometry.dim)
    local_min = (
        np.min(coordinates, axis=0)
        if len(coordinates)
        else np.full(dimension, np.inf)
    )
    local_max = (
        np.max(coordinates, axis=0)
        if len(coordinates)
        else np.full(dimension, -np.inf)
    )
    global_min = np.empty(dimension, dtype=np.float64)
    global_max = np.empty(dimension, dtype=np.float64)
    domain.comm.Allreduce(local_min, global_min, op=MPI.MIN)
    domain.comm.Allreduce(local_max, global_max, op=MPI.MAX)
    span = float(np.max(global_max - global_min))
    scale = max(
        span,
        float(np.max(np.abs(global_min))),
        float(np.max(np.abs(global_max))),
        np.finfo(np.float64).tiny,
    )
    tolerance = max(
        np.finfo(np.float64).tiny,
        64.0 * np.finfo(np.float64).eps * scale,
    )
    return global_min, tolerance


def _coordinate_keys(coordinates, domain, *, policy=None) -> np.ndarray:
    selected = np.asarray(coordinates, dtype=np.float64)
    global_min, tolerance = policy or _coordinate_key_policy(domain)
    dimension = int(domain.geometry.dim)
    return np.rint(
        (selected[:, :dimension] - global_min) / tolerance
    ).astype(np.int64)


__all__ = ["harmonic_executable_identity", "mesh_executable_identity"]
