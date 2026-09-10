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
        presence = tuple(domain.comm.allgather(operator is not None))
        if any(value != presence[0] for value in presence[1:]):
            missing.append(
                {
                    "path": f"harmonic_system.{name}",
                    "reason": "rank_inconsistent_operator_presence",
                }
            )
            operators[name] = {"rank_presence": presence}
            continue
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
            path="harmonic_system.bcs",
        ),
    }
    return _finalize_collective_identity(
        record,
        missing,
        comm=domain.comm,
        path="harmonic_system",
    )


def modal_executable_identity(
    *,
    stiffness,
    mass,
    solution,
    bcs=(),
) -> dict[str, object]:
    """Return a partition-neutral identity for an executable modal system."""

    function = fields.unwrap(solution)
    domain = function.function_space.mesh
    missing: list[dict[str, str]] = []
    operators = {}
    for name, operator in (("stiffness", stiffness), ("mass", mass)):
        operators[name] = _form_identity(
            getattr(operator, "expression", operator),
            domain=domain,
            path=f"modal_system.{name}",
            missing=missing,
        )
    record = {
        "schema": "agentfem.modal-executable-identity.v1",
        "equation": "K phi = lambda M phi",
        "mesh": mesh_executable_identity(domain),
        "target_element": str(function.ufl_element()),
        "operators": operators,
        "homogeneous_dirichlet": _homogeneous_dirichlet_identity(
            function,
            tuple(bcs),
            missing=missing,
            path="modal_system.bcs",
        ),
    }
    return _finalize_collective_identity(
        record,
        missing,
        comm=domain.comm,
        path="modal_system",
    )


def mesh_executable_identity(domain) -> dict[str, object]:
    """Hash source-node connectivity and geometry independent of partition."""

    comm = domain.comm
    local_error = None
    try:
        topology = domain.topology
        cell_map = topology.index_map(topology.dim)
        geometry_dofmap = np.asarray(domain.geometry.dofmaps[0])
        input_indices = np.asarray(
            domain.geometry.input_global_indices,
            dtype=np.int64,
        )
        coordinates = np.asarray(domain.geometry.x, dtype=np.float64)[
            :, : int(domain.geometry.dim)
        ]
    except Exception as exc:  # pragma: no cover - malformed distributed mesh
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local mesh identity inputs",
    )
    local_error = None
    local = None
    try:
        local = []
        for cell in range(int(cell_map.size_local)):
            geometry_dofs = np.asarray(geometry_dofmap[cell], dtype=np.int64)
            source_ids = input_indices[geometry_dofs]
            coordinate_keys = _coordinate_keys(coordinates[geometry_dofs], domain)
            nodes = tuple(
                (
                    int(source_id),
                    *(str(value) for value in coordinate),
                )
                for source_id, coordinate in zip(
                    source_ids, coordinate_keys, strict=True
                )
            )
            local.append(nodes)
    except Exception as exc:  # pragma: no cover - malformed distributed mesh
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local mesh identity payload",
    )
    cells = [item for rank_items in comm.allgather(local) for item in rank_items]
    cells.sort()
    record = {
        "cell_type": str(topology.cell_name()),
        "topology_dimension": int(topology.dim),
        "geometry_dimension": int(domain.geometry.dim),
        "global_cells": len(cells),
        "cells": cells,
        "coordinate_key": "exact_ieee754_hex_with_input_node_id",
        "cell_node_order": "dolfinx_geometry_dofmap",
    }
    return {
        key: value for key, value in record.items() if key != "cells"
    } | {"connectivity_sha256": content_fingerprint(record)}


def _form_identity(form, *, domain, path: str, missing) -> dict[str, object]:
    local_error = None
    ufl_signature = None
    coefficients_source = ()
    constants_source = ()
    tag_entries = []
    try:
        signature = getattr(form, "signature", None)
        if not callable(signature):
            raise TypeError("operator_is_not_a_ufl_form")
        ufl_signature = str(signature())
        coefficients_source = tuple(form.coefficients())
        constants_source = tuple(form.constants())
        for _ufl_domain, by_integral_type in form.subdomain_data().items():
            for integral_type in sorted(by_integral_type):
                tag_entries.extend(
                    (str(integral_type), item)
                    for item in by_integral_type[integral_type]
                    if item is not None
                )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    errors = domain.comm.allgather(local_error)
    if any(error is not None for error in errors):
        reason = next(error for error in errors if error is not None)
        missing.append(
            {
                "path": path,
                "reason": f"operator_identity_unavailable:{reason}",
            }
        )
        return {
            "python_type": f"{type(form).__module__}.{type(form).__qualname__}",
            "ufl_signature": None,
        }

    collective_layout = (
        len(coefficients_source),
        len(constants_source),
        tuple(integral_type for integral_type, _item in tag_entries),
    )
    layouts = domain.comm.allgather(collective_layout)
    if any(layout != layouts[0] for layout in layouts[1:]):
        missing.append(
            {
                "path": path,
                "reason": "rank_inconsistent_operator_layout",
            }
        )
        return {"ufl_signature": ufl_signature, "layout": collective_layout}

    coefficients = []
    for index, coefficient in enumerate(coefficients_source):
        try:
            coefficients.append(
                _function_content_identity(coefficient, domain=domain)
            )
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
    for index, constant in enumerate(constants_source):
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
        try:
            constants.append(_array_identity(value))
        except (TypeError, ValueError) as exc:
            missing.append(
                {
                    "path": f"{path}.constants[{index}]",
                    "reason": f"constant_identity_unavailable:{exc}",
                }
            )
            constants.append({"content_sha256": None})

    tags = []
    for integral_type, item in tag_entries:
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
                "integral_type": integral_type,
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


def _function_content_identity(function, *, domain) -> dict[str, object]:
    comm = domain.comm
    local_error = None
    local = None
    value = None
    V = None
    block_size = None
    try:
        value = fields.unwrap(function)
        V = value.function_space
        if V.mesh is not domain:
            raise ValueError("coefficient belongs to a different mesh")
        if int(V.dofmap.bs) != int(V.dofmap.index_map_bs):
            raise NotImplementedError("mixed or subspace coefficient layout")
        index_map = V.dofmap.index_map
        owned = int(index_map.size_local)
        block_size = int(V.dofmap.index_map_bs)
        coordinates = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
        if len(coordinates) < owned:
            raise ValueError(
                "coefficient does not expose every owned dof coordinate"
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local coefficient identity inputs",
    )
    # DOLFINx forms may consume ghost coefficients.  All ranks have now
    # validated the field layout, so synchronize before recording the identity;
    # the field that is hashed and the field subsequently assembled cannot then
    # silently diverge through stale ghosts.
    value.x.scatter_forward()
    local_error = None
    try:
        values = np.asarray(value.x.array[: owned * block_size]).reshape(
            (owned, block_size)
        )
        try:
            source_ids = _owned_p1_input_node_ids(value)
        except NotImplementedError:
            source_ids = None
        keys = (
            tuple((int(source_id),) for source_id in source_ids)
            if source_ids is not None
            else _coordinate_keys(coordinates[:owned], V.mesh)
        )
        local = [
            (
                tuple(str(item) for item in key),
                _array_identity(row),
            )
            for key, row in zip(keys, values, strict=True)
        ]
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local coefficient identity payload",
    )
    rows = [item for rank_items in comm.allgather(local) for item in rank_items]
    rows.sort(key=lambda item: item[0])
    if any(left[0] == right[0] for left, right in zip(rows[:-1], rows[1:])):
        raise NotImplementedError("coincident coefficient dof coordinates")
    return {
        "element": str(value.ufl_element()),
        "value_shape": list(value.ufl_shape),
        "block_size": block_size,
        "global_block_dofs": len(rows),
        "content_sha256": content_fingerprint(rows),
        "key": (
            "input_geometry_node_id"
            if source_ids is not None
            else "exact_ieee754_hex_physical_dof_coordinate"
        ),
    }


def _meshtags_identity(domain, tags) -> dict[str, object]:
    comm = domain.comm
    local_error = None
    local = None
    dimension = None
    try:
        dimension = int(tags.dim)
        index_map = domain.topology.index_map(dimension)
        if index_map is None:
            raise ValueError(
                f"mesh has no entity index map for dimension {dimension}"
            )
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
        input_indices = np.asarray(
            domain.geometry.input_global_indices,
            dtype=np.int64,
        )
        coordinates = np.asarray(domain.geometry.x, dtype=np.float64)[
            :, : int(domain.geometry.dim)
        ]
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local MeshTags identity inputs",
    )
    local_error = None
    try:
        local = []
        for dofs, tag_value in zip(
            geometry_dofs,
            selected_values,
            strict=True,
        ):
            dofs = np.asarray(dofs, dtype=np.int64)
            source_ids = input_indices[dofs]
            coordinate_keys = _coordinate_keys(
                coordinates[dofs],
                domain,
            )
            closure = sorted(
                (
                    int(source_id),
                    *(str(value) for value in coordinate),
                )
                for source_id, coordinate in zip(
                    source_ids, coordinate_keys, strict=True
                )
            )
            local.append((int(tag_value), closure))
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local MeshTags identity payload",
    )
    entities = [
        item for rank_items in comm.allgather(local) for item in rank_items
    ]
    entities.sort()
    return {
        "dimension": dimension,
        "global_tagged_entities": len(entities),
        "content_sha256": content_fingerprint(entities),
    }


def _homogeneous_dirichlet_identity(
    function,
    bcs,
    *,
    missing,
    path: str,
) -> dict[str, object]:
    V = function.function_space
    comm = V.mesh.comm
    local_error = None
    local_missing = []
    local = None
    try:
        block_size = int(V.dofmap.index_map_bs)
        owned_scalar_dofs = int(V.dofmap.index_map.size_local) * block_size
        coordinates = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local homogeneous-Dirichlet identity inputs",
    )
    local_error = None
    try:
        try:
            source_ids = _owned_p1_input_node_ids(function)
        except NotImplementedError:
            source_ids = None
        keys = (
            tuple((int(source_id),) for source_id in source_ids)
            if source_ids is not None
            else _coordinate_keys(
                coordinates[: int(V.dofmap.index_map.size_local)],
                V.mesh,
            )
        )
        constrained_local_dofs = set()
        for index, bc in enumerate(bcs):
            dof_indices = getattr(bc, "dof_indices", None)
            if not callable(dof_indices):
                local_missing.append(
                    {
                        "path": f"{path}[{index}]",
                        "reason": "concrete_dirichlet_dofs_unavailable",
                    }
                )
                continue
            dofs, owned_count = dof_indices()
            for dof in np.asarray(dofs[: int(owned_count)], dtype=np.int64):
                if dof < 0 or dof >= owned_scalar_dofs:
                    local_missing.append(
                        {
                            "path": f"{path}[{index}]",
                            "reason": "dirichlet_dof_outside_target_space",
                        }
                    )
                    continue
                block, component = divmod(int(dof), block_size)
                constrained_local_dofs.add(int(dof))
        local = [
            (
                *(str(item) for item in keys[dof // block_size]),
                int(dof % block_size),
            )
            for dof in sorted(constrained_local_dofs)
        ]
        target_local = [
            (*(str(item) for item in key), component)
            for key in keys
            for component in range(block_size)
        ]
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context="build local homogeneous-Dirichlet identity payload",
    )
    for rank_missing in comm.allgather(local_missing):
        for item in rank_missing:
            if item not in missing:
                missing.append(item)
    target_dofs = [
        item
        for rank_items in comm.allgather(target_local)
        for item in rank_items
    ]
    if len(set(target_dofs)) != len(target_dofs):
        missing.append(
            {
                "path": path,
                "reason": "coincident_target_dof_coordinates",
            }
        )
    constrained = [
        item
        for rank_items in comm.allgather(local)
        for item in rank_items
    ]
    constrained.sort()
    return {
        "value": "homogeneous_zero",
        "global_scalar_dofs": len(constrained),
        "dof_set_sha256": content_fingerprint(constrained),
        "key": (
            "input_geometry_node_id_and_block_component"
            if source_ids is not None
            else "exact_ieee754_hex_block_coordinate_and_component"
        ),
    }


def _owned_p1_input_node_ids(function) -> np.ndarray:
    """Map owned blocked P1 dofs to partition-neutral mesh input-node ids.

    ``tabulate_dof_coordinates`` is geometrically correct but may differ by a
    few floating-point ulps when the same mesh is repartitioned.  For a
    first-order nodal field, the mesh input-node identity is both exact and
    invariant under that repartitioning.  Higher-order or mixed layouts retain
    the lossless coordinate fallback and therefore still fail closed if an
    equivalent portable identity cannot be established.
    """

    V = function.function_space
    domain = V.mesh
    owned = int(V.dofmap.index_map.size_local)
    if int(V.dofmap.bs) != int(V.dofmap.index_map_bs):
        raise NotImplementedError("field is not a blocked nodal space")
    geometry_maps = getattr(domain.geometry, "dofmaps", None)
    geometry_dofmap = (
        domain.geometry.dofmap if geometry_maps is None else geometry_maps[0]
    )
    input_indices = np.asarray(domain.geometry.input_global_indices, dtype=np.int64)
    source = np.full(owned, -1, dtype=np.int64)
    cell_map = domain.topology.index_map(domain.topology.dim)
    for cell in range(int(cell_map.size_local + cell_map.num_ghosts)):
        geometry_dofs = np.asarray(geometry_dofmap[cell], dtype=np.int64)
        field_dofs = np.asarray(V.dofmap.cell_dofs(cell), dtype=np.int64)
        if geometry_dofs.size != field_dofs.size:
            raise NotImplementedError(
                "field dofs do not coincide with first-order geometry nodes"
            )
        for geometry_dof, field_dof in zip(
            geometry_dofs,
            field_dofs,
            strict=True,
        ):
            selected = int(field_dof)
            if selected >= owned:
                continue
            node = int(input_indices[int(geometry_dof)])
            previous = int(source[selected])
            if previous not in {-1, node}:
                raise RuntimeError(
                    "one owned field dof maps to inconsistent input-node ids"
                )
            source[selected] = node
    missing = np.flatnonzero(source < 0)
    if missing.size:
        raise NotImplementedError(
            "field lacks input-node identity for owned dofs: "
            f"{missing.tolist()}"
        )
    return source


def _array_identity(value) -> dict[str, object]:
    selected = np.ascontiguousarray(np.asarray(value))
    digest = sha256(selected.tobytes(order="C")).hexdigest()
    return {
        "dtype": selected.dtype.str,
        "shape": list(selected.shape),
        "sha256": digest,
    }


def _raise_collective_identity_error(comm, local_error, *, context: str) -> None:
    """Make a rank-local identity-construction failure fail together."""

    errors = comm.allgather(local_error)
    failures = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError(
            f"Executable identity could not {context}; " + "; ".join(failures)
        )


def _finalize_collective_identity(record, missing, *, comm, path: str):
    """Return one communicator-wide identity or one shared fail-closed record."""

    local_error = None
    local_fingerprint = None
    try:
        local_fingerprint = content_fingerprint(record)
    except Exception as exc:  # pragma: no cover - malformed identity record
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_identity_error(
        comm,
        local_error,
        context=f"finalize {path}",
    )
    local = {
        "fingerprint": local_fingerprint,
        "missing": tuple(dict(item) for item in missing),
    }
    gathered = comm.allgather(local)
    fingerprints = tuple(str(item["fingerprint"]) for item in gathered)
    merged_missing = []
    for item in gathered:
        for problem in item["missing"]:
            if problem not in merged_missing:
                merged_missing.append(problem)
    if len(set(fingerprints)) != 1:
        merged_missing.append(
            {
                "path": path,
                "reason": "rank_inconsistent_executable_identity",
            }
        )
    shared_record = comm.bcast(record if comm.rank == 0 else None, root=0)
    fingerprint = (
        fingerprints[0]
        if len(set(fingerprints)) == 1
        else content_fingerprint({"rank_fingerprints": sorted(fingerprints)})
    )
    return {
        "complete": not merged_missing,
        "missing": tuple(merged_missing),
        "record": shared_record,
        "fingerprint": fingerprint,
    }


def _coordinate_keys(coordinates, domain, *, policy=None) -> tuple[tuple[str, ...], ...]:
    """Return lossless, portable physical-coordinate keys.

    A tolerance-scaled integer key is unsuitable for an executable identity:
    translation can disappear when coordinates are made relative to the lower
    bound, and a large absolute offset can quantize away mechanically material
    changes on a small domain.  Python's hexadecimal float representation is a
    canonical, lossless encoding of the finite IEEE-754 binary64 value and is
    independent of host byte order.

    ``policy`` remains an ignored compatibility keyword for private test and
    downstream callers that used the former helper signature.
    """

    del policy
    selected = np.asarray(coordinates, dtype=np.float64)
    dimension = int(domain.geometry.dim)
    physical = selected[:, :dimension]
    if not np.all(np.isfinite(physical)):
        raise ValueError("mesh or degree-of-freedom coordinates contain NaN or Inf")
    return tuple(
        tuple(float(0.0 if value == 0.0 else value).hex() for value in row)
        for row in physical
    )


__all__ = [
    "harmonic_executable_identity",
    "mesh_executable_identity",
    "modal_executable_identity",
]
