# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private result assembly for transient procedures."""

from __future__ import annotations

from .. import fields as field_api
from .core import from_solution
from .execution import add_execution_trace
from .lifecycle import execution_context


def from_transient_step(
    step,
    solution,
    *,
    output_fields,
    metadata=None,
):
    """Build a result after a transient procedure has advanced its state."""

    result = from_solution(
        solution,
        name=step.name,
        metadata={"step": step.summary()},
    )
    if metadata:
        result.metadata.update(dict(metadata))
    context = execution_context(step)
    if context is not None:
        result.metadata.setdefault("execution_context", context.summary())
    add_execution_trace(result, step.execution_events)
    _attach_transient_output(result, step, tuple(output_fields))
    return result


def _attach_transient_output(result, step, output_fields) -> None:
    result.metadata["accepted_times"] = tuple(
        float(item) for item in step.accepted_times
    )
    output_start = step.last_output_start_time
    result.metadata["transient"] = {
        "completed_steps": int(step.completed_steps),
        "total_steps": int(step.steps),
        "output_start_time": output_start,
        "output_scope": (
            None
            if step.last_output is None
            else "complete"
            if output_start == 0.0
            else "continuation_segment"
        ),
    }
    if step.history_requests:
        result.metadata["transient"]["history_requests"] = [
            request.summary() for request in step.history_requests
        ]
    _attach_histories(result, step)
    for checkpoint in step.checkpoints:
        result.add_checkpoint(checkpoint)
    if step.last_output is not None:
        _attach_field_output(result, step, output_fields)


def _attach_histories(result, step) -> None:
    if not step.history_records:
        return
    coordinates = [item["time"] for item in step.history_records]
    names = tuple(name for name in step.history_records[0] if name != "time")
    requests = {
        request.name: request for request in getattr(step, "history_requests", ())
    }
    result.add_histories(
        coordinates,
        {name: [item[name] for item in step.history_records] for name in names},
        abscissa_name="time",
        abscissa_unit="s",
        units={name: getattr(requests.get(name), "unit", None) for name in names},
        descriptions={
            name: (
                getattr(requests.get(name), "description", "")
                or _HISTORY_DESCRIPTIONS.get(name, "")
            )
            for name in names
        },
    )


def _attach_field_output(result, step, output_fields) -> None:
    path = step.last_output
    output_start = step.last_output_start_time
    primary = None if not output_fields else field_api.unwrap(output_fields[0])
    primary_shape = () if primary is None else tuple(getattr(primary, "ufl_shape", ()))
    vector_primary = len(primary_shape) == 1
    domain = None if primary is None else primary.function_space.mesh
    backend = getattr(step, "last_output_backend", None)
    storage_name = (
        None
        if not vector_primary
        else (
            "U"
            if backend == "agentfem_unified_xdmf"
            else str(getattr(primary, "name", "Displacement"))
        )
    )
    semantic_name = "Displacement" if vector_primary else None
    is_paraview = path.suffix.lower() == ".pvd"
    result.metadata["field_output"] = {
        "status": "completed",
        "backend": backend,
        "layout": getattr(step, "last_output_layout", None),
        "geometry": "reference",
        "scientific_artifact": None if is_paraview else str(path),
        "scientific_xdmf_layout": (
            "not_emitted" if is_paraview else "single_uniform_grid"
        ),
        "recommended_visualization_artifact": str(path),
        "visualization_geometry_datasets_per_time": 1,
        "visualization_requires_extract_block": False,
        "warp_field": storage_name,
        "warp_field_semantic": semantic_name,
        "physical_components": (int(primary_shape[0]) if vector_primary else None),
        "stored_components": (
            None
            if not vector_primary or domain is None
            else int(domain.geometry.x.shape[1])
        ),
        "geometry_dimension": (
            None if domain is None else int(domain.geometry.x.shape[1])
        ),
        "physical_model_dimension": (
            None if domain is None else int(domain.geometry.dim)
        ),
        "warp_compatible": bool(vector_primary),
        "field_aliases": (
            {}
            if storage_name is None or semantic_name is None
            else {semantic_name: storage_name}
        ),
    }
    if is_paraview:
        result.add_artifact("fields_paraview", path)
    else:
        result.add_artifact("fields_xdmf", path)
        heavy_data = path.with_suffix(".h5")
        if heavy_data.is_file():
            result.add_artifact("fields_hdf5", heavy_data)
    for item in output_fields:
        function = getattr(item, "value", item)
        name = getattr(function, "name", type(function).__name__)
        result.add_field(
            name,
            function,
            artifact=path,
            description=(
                "Transient field in the shared single-geometry series; "
                f"this output segment starts at time {output_start:g}."
            ),
        )


_HISTORY_DESCRIPTIONS = {
    "kinetic_energy": "Discrete kinetic energy, one half v-transpose M v.",
    "strain_energy": "Recoverable linear strain energy, one half u-transpose K u.",
    "total_mechanical_energy": "Sum of discrete kinetic and recoverable strain energy.",
    "bulk_strain_energy": "Finite-strain constitutive energy integrated in the reference body.",
    "cohesive_stored_energy": "Recoverable energy currently stored by the cohesive interface.",
    "cohesive_fracture_dissipation": "Irreversible cohesive dissipation relative to the initial interface state.",
    "numerical_damping_dissipation": "Accepted nonnegative work dissipated by the declared viscous damping model.",
    "natural_load_work": "Accepted-path trapezoidal work of weak natural loads.",
    "prescribed_motion_work": "Accepted-path trapezoidal work of strong prescribed-motion reactions.",
    "external_work": "Sum of natural-load and prescribed-motion work.",
    "energy_balance_error": "Initial accounted energy plus external work minus current accounted energy.",
    "relative_energy_balance_error": "Absolute energy-balance error normalized by the largest energy scale.",
    "thermal_content": (
        "Discrete thermal content, one-transpose C T, relative to the model's "
        "temperature zero."
    ),
    "applied_heat_rate": "Applied volumetric, flux, and Robin-source heat rate, one-transpose Q.",
    "outward_heat_rate": "Net discrete conduction/Robin rate, one-transpose K T.",
    "heat_balance_residual": (
        "Implicit-Euler energy residual: delta thermal content plus dt times "
        "outward rate minus applied rate. Strong-temperature reactions appear "
        "in this residual until reported separately."
    ),
}


__all__ = ()
