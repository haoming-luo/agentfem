# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-light validation for the engineering Model boundary."""

from __future__ import annotations

import numpy as np

from . import constraints as constraint_api
from ._model_support import domain, duplicate_names, field_domain


def validate_model(model, *, target=None, step_options=None):
    """Return structured, addressable model-validation results."""

    from .validation import ValidationReport, issue

    issues = []
    study = model.study
    mesh_domain = domain(model.mesh)

    if study is None:
        issues.append(
            issue(
                "AFM-MODEL-001",
                "model.study",
                "A finite-element model requires a Study.",
                hint="Create a study with agentfem.studies before the model.",
            )
        )
    elif hasattr(study, "validate"):
        try:
            study.validate()
        except (TypeError, ValueError) as exc:
            issues.append(
                issue(
                    "AFM-STUDY-001",
                    "model.study",
                    str(exc),
                    hint="Revise the analysis, physics, dimension, or assumption.",
                )
            )

    if study is not None and model.fields:
        from .step_providers import step_capability

        capability = step_capability(model, target=target, options=step_options)
        if not capability["supported"]:
            issues.append(
                issue(
                    "AFM-STUDY-002",
                    "model.study",
                    (
                        "No executable step provider supports this Study, "
                        "field, material, and procedure combination."
                    ),
                    hint=(
                        "Choose a supported Study/material combination or "
                        "register a StepProvider before solving."
                    ),
                    capability=capability,
                )
            )

    if mesh_domain is None:
        issues.append(
            issue(
                "AFM-MODEL-002",
                "model.mesh",
                "A finite-element model requires a mesh.",
                hint="Create or import the mesh before defining regions and fields.",
            )
        )
    elif study is not None:
        _validate_mesh_study(
            issues,
            model=model,
            study=study,
            mesh_domain=mesh_domain,
            issue=issue,
        )

    if not model.fields:
        issues.append(
            issue(
                "AFM-MODEL-004",
                "model.fields",
                "A finite-element model requires at least one field.",
                hint="Register an unknown with model.field(...).",
            )
        )
    elif mesh_domain is not None:
        for index, field_object in enumerate(model.fields):
            registered_domain = field_domain(field_object)
            if registered_domain is not None and registered_domain is not mesh_domain:
                issues.append(
                    issue(
                        "AFM-FIELD-001",
                        f"model.fields[{index}]",
                        "The field is defined on a different mesh from the model.",
                        hint="Create the field on model.mesh or register the intended mesh.",
                    )
                )

    selected_material = None if step_options is None else step_options.get("material")
    analysis = None if study is None else getattr(study, "analysis", None)
    required_operators = {
        "modal": ("K", "M"),
        "second_order_dynamics": ("K", "M", "F"),
    }.get(analysis, ("K", "F"))
    complete_operator_system = bool(
        step_options
        and all(step_options.get(name) is not None for name in required_operators)
    )
    if (
        study is not None
        and (
            getattr(study, "is_solid_mechanics", False)
            or getattr(study, "is_heat_transfer", False)
        )
        and not model.materials
        and selected_material is None
        and not complete_operator_system
    ):
        physics = getattr(study, "physics", "finite-element")
        issues.append(
            issue(
                "AFM-MATERIAL-001",
                "model.materials",
                f"{physics.replace('_', ' ').title()} models require at least one material.",
                hint="Register material properties with model.material(...).",
            )
        )

    if len(model.materials) > 1:
        for index, record in enumerate(model.materials):
            if getattr(record, "region", None) is None:
                issues.append(
                    issue(
                        "AFM-MATERIAL-002",
                        f"model.materials[{index}].region",
                        "Every material in a multi-material model needs a region.",
                        hint="Pass region=... when registering each material.",
                    )
                )
        _validate_material_partition(
            issues,
            model=model,
            mesh_domain=mesh_domain,
            issue=issue,
        )

    for index, source in enumerate(getattr(model, "eigenstrains", ())):
        if not callable(getattr(source, "strain", None)) or not callable(
            getattr(source, "equivalent_stress", None)
        ):
            issues.append(
                issue(
                    "AFM-EIGENSTRAIN-001",
                    f"model.eigenstrains[{index}]",
                    "An eigenstrain source must expose strain and equivalent_stress.",
                    hint="Create the source with agentfem.eigenstrains.",
                )
            )

    if study is not None:
        selected_options = {} if step_options is None else dict(step_options)
        communicator = (
            None if mesh_domain is None else getattr(mesh_domain, "comm", None)
        )
        compatibility = constraint_api.validate_solver_compatibility(
            constraints=selected_options.get("constraints", model.constraints),
            analysis=getattr(study, "analysis", ""),
            procedure=(
                selected_options.get("method")
                or getattr(study, "preferred_procedure", None)
            ),
            comm_size=int(getattr(communicator, "size", 1)),
        )
        issues.extend(compatibility.issues)

    for collection_name in (
        "fields",
        "amplitudes",
        "eigenstrains",
        "constraints",
        "loads",
        "boundary_models",
        "regions",
        "steps",
    ):
        for name in duplicate_names(getattr(model, collection_name)):
            issues.append(
                issue(
                    "AFM-NAME-001",
                    f"model.{collection_name}",
                    f"Name {name!r} is used more than once.",
                    severity="warning",
                    hint="Use unique names when objects must be addressed for repair or reuse.",
                    duplicate_name=name,
                )
            )

    if mesh_domain is not None:
        for index, region in enumerate(model.regions):
            region_domain = getattr(region, "domain", None)
            if region_domain is not None and region_domain is not mesh_domain:
                issues.append(
                    issue(
                        "AFM-REGION-001",
                        f"model.regions[{index}]",
                        "The region belongs to a different mesh from the model.",
                        hint="Recreate the region on model.mesh.",
                    )
                )

        # Element/runtime-topology compatibility belongs to the discretization
        # boundary, not to Model or any one Step provider. Keep the common
        # validation path metadata-only; elements.audit(check_quality=True)
        # performs collective cell-quality work before trusted long runs.
        from .elements import _lightweight_issues

        issues.extend(_lightweight_issues(model))

    return ValidationReport.from_issues(issues, scope=f"model:{model.name}")


def _validate_material_partition(issues, *, model, mesh_domain, issue) -> None:
    """Require an exact owned-cell partition for multi-material models."""

    if mesh_domain is None:
        return
    topology = getattr(mesh_domain, "topology", None)
    if topology is None:
        return
    tdim = int(topology.dim)
    cell_map = topology.index_map(tdim)
    local_cells = int(cell_map.size_local)
    membership = np.zeros(local_cells, dtype=np.int16)
    unsupported = []
    for index, record in enumerate(model.materials):
        region = getattr(record, "region", None)
        tags = getattr(region, "cell_tags", None)
        tag = getattr(region, "tag", None)
        if tags is None or tag is None or int(getattr(tags, "dim", -1)) != tdim:
            unsupported.append(index)
            continue
        entities = np.asarray(tags.find(int(tag)), dtype=np.int64)
        owned = entities[(entities >= 0) & (entities < local_cells)]
        membership[owned] += 1
    if unsupported:
        issues.append(
            issue(
                "AFM-MATERIAL-003",
                "model.materials",
                "Multi-material coverage cannot be audited because one or more "
                "regions do not expose cell tags.",
                hint="Use named CellRegion assets for every material assignment.",
                material_indices=tuple(unsupported),
            )
        )
        return
    communicator = getattr(mesh_domain, "comm", None)
    local_missing = int(np.count_nonzero(membership == 0))
    local_overlap = int(np.count_nonzero(membership > 1))
    if communicator is not None:
        from mpi4py import MPI

        missing = int(communicator.allreduce(local_missing, op=MPI.SUM))
        overlap = int(communicator.allreduce(local_overlap, op=MPI.SUM))
    else:
        missing, overlap = local_missing, local_overlap
    if missing:
        issues.append(
            issue(
                "AFM-MATERIAL-004",
                "model.materials",
                f"Material regions leave {missing} owned cells uncovered.",
                hint="Partition every cell exactly once before solving.",
                uncovered_owned_cells=missing,
            )
        )
    if overlap:
        issues.append(
            issue(
                "AFM-MATERIAL-005",
                "model.materials",
                f"Material regions overlap on {overlap} owned cells.",
                hint="Use disjoint CellRegion assets for constitutive assignments.",
                overlapping_owned_cells=overlap,
            )
        )


def _validate_mesh_study(issues, *, model, study, mesh_domain, issue) -> None:
    geometry = getattr(mesh_domain, "geometry", None)
    mesh_dimension = getattr(geometry, "dim", None)
    if mesh_dimension is None:
        issues.append(
            issue(
                "AFM-MESH-001",
                "model.mesh",
                "The registered mesh does not expose a geometric dimension.",
                hint="Register a DOLFINx mesh or an AgentFEM FEMMesh.",
            )
        )
        return
    if int(mesh_dimension) != int(study.dimension):
        issues.append(
            issue(
                "AFM-MODEL-003",
                "model.mesh.geometry.dim",
                (
                    f"Study dimension {study.dimension} does not match "
                    f"mesh geometric dimension {mesh_dimension}."
                ),
                hint="Revise the Study dimension or use the intended mesh.",
                study_dimension=int(study.dimension),
                mesh_dimension=int(mesh_dimension),
            )
        )
        return
    if getattr(study, "assumption", None) != "axisymmetric" or int(mesh_dimension) != 2:
        return

    from mpi4py import MPI

    coordinates = np.asarray(mesh_domain.geometry.x[:, 0], dtype=float)
    local_min = float(np.min(coordinates)) if coordinates.size else np.inf
    local_max = float(np.max(np.abs(coordinates))) if coordinates.size else 0.0
    minimum_radius = float(mesh_domain.comm.allreduce(local_min, op=MPI.MIN))
    radius_scale = float(mesh_domain.comm.allreduce(local_max, op=MPI.MAX))
    tolerance = 1.0e-12 * max(1.0, radius_scale)
    if minimum_radius < -tolerance:
        issues.append(
            issue(
                "AFM-AXISYM-001",
                "model.mesh.geometry.x[:,0]",
                "An axisymmetric meridian cannot contain negative radius.",
                hint="Define the meridian in coordinates (r,z) with r >= 0.",
                minimum_radius=minimum_radius,
            )
        )
        return
    if minimum_radius > tolerance:
        return

    names = {
        str(getattr(item, "name", "")).strip().lower()
        for item in constraint_api.dirichlet_constraints(model.constraints)
    }
    has_axis_regularity = any(
        name == "axisymmetric_axis" or name.startswith("symmetry_x") for name in names
    )
    if not has_axis_regularity:
        issues.append(
            issue(
                "AFM-AXISYM-002",
                "model.constraints",
                "The meridian reaches r=0 without a declared radial regularity constraint.",
                severity="warning",
                hint=(
                    "Register constraints.axisymmetric_axis(u, on=axis) "
                    "or an equivalent named x-normal symmetry constraint."
                ),
            )
        )


__all__ = ()
