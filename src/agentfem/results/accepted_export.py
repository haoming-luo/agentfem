# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-station output for providers exposing accepted cell responses."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np


def _response_fields(step, *, state_fields=None, velocity=False, acceleration=False,
                     transport=None, materialize=True):
    from .cell_data import _CellFieldTransport

    record = step.accepted_material_fields()
    displacement = step.state.u.value
    domain = displacement.function_space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError("AFM-EXPORT-MPI: accepted response export is serial")
    keys = record["cell_keys"]
    stress = np.asarray(record.get("cauchy_stress"))
    if (stress.shape != (len(keys), 3, 3) or stress.dtype.kind not in "fiu"
            or not np.isfinite(stress).all()):
        raise ValueError("AFM-EXPORT-ARRAY: expected one finite real 3x3 Cauchy stress per cell")
    if (record.get("stress_measure") != "cauchy"
            or not isinstance(record.get("location"), str) or not record["location"].strip()):
        raise ValueError("AFM-EXPORT-SEMANTICS: require Cauchy stress and an explicit source location")
    selected = {} if state_fields is None else dict(state_fields)
    reserved = {"cauchy_stress", displacement.name, "velocity", "acceleration"}
    if transport is None or not transport.matches(domain, keys):
        transport = _CellFieldTransport(domain, keys)
    fields = [transport.field(record["cauchy_stress"], name="cauchy_stress")] if materialize else []
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
        if materialize:
            fields.append(transport.field(values, name=name))
    nodal = [displacement]
    for enabled, name in ((velocity, "v"), (acceleration, "a")):
        if type(enabled) is not bool:
            raise ValueError("AFM-EXPORT-FLAG: expected boolean")
        if enabled:
            nodal.append(getattr(step.state, name).value)
    return record, nodal, fields, selected, transport


def export_accepted_response(step, directory, *, state_fields=None,
                             velocity=False, acceleration=False, units=None,
                             visualization=True):
    """Export one accepted response, raw state and cell keys.

    State selectors map names to (schema_variable, component); no SDV meaning
    is inferred. Serial, one response point per cell. Does not advance state.
    With a process series, use visualization=False to archive raw endpoint data
    without duplicating its final frame in another XDMF/HDF5 dataset.
    """
    from ..io import ResultWriter
    from ..checkpointing import atomic_savez, atomic_write_text

    if type(visualization) is not bool:
        raise ValueError("AFM-EXPORT-FLAG: visualization must be boolean")
    record, nodal, fields, selected, _ = _response_fields(step, state_fields=state_fields,
        velocity=velocity, acceleration=acceleration, materialize=visualization)
    domain = nodal[0].function_space.mesh
    keys = record["cell_keys"]
    # Raw archival is a public output boundary even without visualization.
    # Do not rely on constructing DG0 fields to detect invalid provider data.
    count = len(keys)
    for name in ("state", "cauchy_stress", "deformation_gradient"):
        values = np.asarray(record[name])
        expected = (count, 3, 3) if name != "state" else None
        if (values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or (expected is not None and values.shape != expected)
                or (name == "state" and (values.ndim != 2 or len(values) != count))):
            raise ValueError(f"AFM-EXPORT-ARRAY: invalid accepted {name}")
    metadata = {"schema": "agentfem.accepted-response-export", "schema_version": "0.1.0",
                "time": record["time"], "complete": step.completed_steps == step.steps,
                "location": record["location"], "stress_measure": record["stress_measure"],
                "state_schema": record["state_schema"], "state_fields": selected,
                "units": units or {}, "scientifically_validated": False}
    encoded = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"accepted_material": directory / "accepted_material.npz",
             "accepted_material_metadata": directory / "accepted_material.json"}
    if visualization:
        paths.update(fields=directory / "fields.xdmf", field_arrays=directory / "fields.h5")
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("AFM-EXPORT-EXISTS: use a fresh output directory")
    if visualization:
        with ResultWriter(paths["fields"], domain, (*nodal, *fields)) as writer:
            writer.write(record["time"])
    atomic_savez(paths["accepted_material"], cell_keys=keys, state=record["state"],
                 cauchy_stress=record["cauchy_stress"], deformation_gradient=record["deformation_gradient"],
                 physical_time=np.asarray(record["time"]))
    atomic_write_text(paths["accepted_material_metadata"], encoded)
    return paths


class AcceptedResponseSeries:
    """Streaming accepted fields; fixed mesh stored once, never a restart archive.

    Use with project.execute_step(frames=..., RunPolicy(field_intervals=...)),
    or write accepted stations explicitly and close. A continuation is a new
    file, never an append to an earlier run. Private state selection stays data.
    """

    def __init__(self, path, *, state_fields=None, velocity=False, acceleration=False, units=None):
        self.path = Path(path)
        if self.path.suffix != ".xdmf":
            raise ValueError("AFM-FRAMES-PATH: expected .xdmf")
        # Output choices belong to this series. In particular, list-valued
        # selectors must not change when a caller edits its original mapping.
        self.options = dict(state_fields=deepcopy(state_fields),
                            velocity=velocity, acceleration=acceleration)
        self.units = json.loads(json.dumps(units or {}, allow_nan=False))
        self._writer = None
        self._closed = False
        self._published = False
        self.stations = []
        self._identity = None
        self._transport = None

    def write(self, step):
        from ..io import ResultWriter

        if self._closed:
            raise RuntimeError("AFM-FRAMES-CLOSED: use a new series")
        record, nodal, fields, selected, self._transport = _response_fields(
            step, transport=self._transport, **self.options)
        identity = (record["state_schema"], selected, record["location"], record["stress_measure"])
        if self._identity is not None and identity != self._identity:
            raise ValueError("AFM-FRAMES-SCHEMA: state schema changed")
        if self._writer is None:
            if any(p.exists() for p in (self.path, self.path.with_suffix('.h5'), self.path.with_suffix('.json'))):
                raise FileExistsError("AFM-FRAMES-EXISTS: use a fresh series path")
            self._writer = ResultWriter(self.path, nodal[0].function_space.mesh)
            self._writer.__enter__()
        self._writer.write(record["time"], *nodal, *fields)
        if self._identity is None:
            # Freeze only a successfully written descriptor, and never retain
            # a provider-owned mutable schema as the previous-frame identity.
            self._identity = deepcopy(identity)
        self.stations.append({"increment": int(step.completed_steps), "time": float(record["time"])})

    def close(self, *, complete=False):
        from ..checkpointing import atomic_write_text

        if self._closed:
            return self.artifacts()
        self._closed = True
        if self._writer is not None:
            if self.stations:
                self._writer.__exit__(None, None, None)
            else:
                self._writer.__exit__(RuntimeError, RuntimeError("No accepted frames"), None)
        if self.stations:
            atomic_write_text(self.path.with_suffix('.json'), json.dumps({
                "schema": "agentfem.accepted-response-series.v1", "complete": bool(complete),
                "restart_archive": False, "scientifically_validated": False,
                "stations": self.stations, "units": self.units,
                "state_schema": self._identity[0], "state_fields": self._identity[1],
                "location": self._identity[2], "stress_measure": self._identity[3],
            }, indent=2, allow_nan=False) + '\n')
            self._published = True
        return self.artifacts()

    def artifacts(self):
        if not self._published:
            return {}
        return {"process_fields": self.path, "process_arrays": self.path.with_suffix('.h5'),
                "process_metadata": self.path.with_suffix('.json')}
