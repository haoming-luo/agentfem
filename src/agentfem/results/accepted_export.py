# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-station output for providers exposing accepted cell responses."""
from pathlib import Path
import json
import numpy as np


def export_accepted_response(step, directory, *, state_fields=None,
                             velocity=False, acceleration=False, units=None):
    """Export accepted response fields with explicit state selectors.

    ``state_fields`` maps output names to ``(schema_variable, component)``.
    Component is a nonnegative integer for a vector variable, or None for the
    entire variable. No source-law SDV numbering is inferred. Raw state and
    cell keys are retained in NPZ; visual fields use the ordinary ResultWriter.
    Currently serial and one response point per cell, as declared by the
    owning provider. This is output, not validation or state advancement.
    """
    from ..io import ResultWriter
    from ..checkpointing import atomic_savez, atomic_write_text
    from .cell_data import cell_field_from_values

    record = step.accepted_material_fields()
    displacement = step.state.u.value
    domain = displacement.function_space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError("AFM-EXPORT-MPI: accepted response export is serial")
    keys = record["cell_keys"]
    selected = {} if state_fields is None else dict(state_fields)
    reserved = {"cauchy_stress", displacement.name, "velocity", "acceleration"}
    fields = [cell_field_from_values(domain, record["cauchy_stress"], name="cauchy_stress", cell_keys=keys)]
    for name, selector in selected.items():
        if not isinstance(name, str) or not name or name in reserved:
            raise ValueError("AFM-EXPORT-NAME: invalid or reserved state field name")
        if not isinstance(selector, (tuple, list)) or len(selector) != 2:
            raise ValueError("AFM-EXPORT-STATE: require (variable, component)")
        variable, component = selector
        if variable not in record["state_variables"]:
            raise ValueError(f"AFM-EXPORT-STATE: unknown variable {variable}")
        values = np.asarray(record["state_variables"][variable])
        if component is not None:
            if type(component) is not int or values.ndim != 2 or not 0 <= component < values.shape[1]:
                raise ValueError("AFM-EXPORT-COMPONENT: require a valid vector component")
            values = values[:, component]
        fields.append(cell_field_from_values(domain, values, name=name, cell_keys=keys))
    nodal = [displacement]
    for enabled, name in ((velocity, "v"), (acceleration, "a")):
        if type(enabled) is not bool:
            raise ValueError("AFM-EXPORT-FLAG: expected boolean")
        if enabled:
            nodal.append(getattr(step.state, name).value)
    metadata = {"schema": "agentfem.accepted-response-export", "schema_version": "0.1.0",
                "time": record["time"], "complete": step.completed_steps == step.steps,
                "location": record["location"], "stress_measure": record["stress_measure"],
                "state_schema": record["state_schema"], "state_fields": selected,
                "units": units or {}, "scientifically_validated": False}
    encoded = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"fields": directory / "fields.xdmf", "field_arrays": directory / "fields.h5",
             "accepted_material": directory / "accepted_material.npz",
             "accepted_material_metadata": directory / "accepted_material.json"}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("AFM-EXPORT-EXISTS: use a fresh output directory")
    with ResultWriter(paths["fields"], domain, (*nodal, *fields)) as writer:
        writer.write(record["time"])
    atomic_savez(paths["accepted_material"], cell_keys=keys, state=record["state"],
                 cauchy_stress=record["cauchy_stress"], deformation_gradient=record["deformation_gradient"],
                 physical_time=np.asarray(record["time"]))
    atomic_write_text(paths["accepted_material_metadata"], encoded)
    return paths
