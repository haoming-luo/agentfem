# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Co-located accepted histories, keyed by original cell and quadrature point.

No spatial interpolation or extremum selection is performed. A Step owns the
sampling and commit boundary; this module owns selection and result semantics.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np


@dataclass(frozen=True)
class MaterialPointHistoryRequest:
    """One fixed reference integration point, not a mesh node or a probe."""

    name: str
    cell: int
    point: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("Material point name must be a nonempty identifier without punctuation or whitespace.")
        for key in ("cell", "point"):
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"Material point {key} must be a nonnegative integer.")
            object.__setattr__(self, key, int(value))


def material_point(name, *, cell, point=0):
    """Select an original input-cell ID and reference quadrature-point index."""
    return MaterialPointHistoryRequest(name, cell, point)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


class _MaterialPointRecorder:
    def __init__(self, requests, sampling, *, channels, time_unit, scientific_identity):
        from .integration_points import _rule_key
        from ..checkpointing import mesh_portable_identity

        domain = sampling.function.function_space.mesh
        self._sampling = sampling
        self._reference_geometry = domain.geometry.x.copy()
        self._reference_points = sampling.points.copy()
        self._reference_weights = sampling.weights.copy()
        if domain.comm.size != 1:
            raise NotImplementedError("Accepted material-point histories currently require a serial Step.")
        requests = tuple(requests)
        if not all(isinstance(item, MaterialPointHistoryRequest) for item in requests):
            raise TypeError("material_points must contain results.material_point(...) requests.")
        if len({item.name for item in requests}) != len(requests):
            raise ValueError("Material point history names must be unique.")
        keys = sampling.owned_cell_keys
        coordinates = sampling.owned_physical_points
        count = len(sampling.points)
        self.indices = []
        points = []
        for item in requests:
            matches = np.flatnonzero(keys == item.cell)
            if matches.size != 1 or item.point >= count:
                raise ValueError(f"Unknown material point {item.name!r}: cell={item.cell}, point={item.point}.")
            local = int(matches[0])
            self.indices.append(local * count + item.point)
            points.append({"name": item.name, "cell": item.cell, "point": item.point,
                           "reference_position": coordinates[local, item.point].tolist()})
        self.time_unit = time_unit
        self.channels = json.loads(_canonical(channels))
        self.identity = json.loads(_canonical({
            "schema": "agentfem.material-point-history.v1", "mesh": mesh_portable_identity(domain),
            "quadrature_rule": _rule_key(sampling), "points": points,
            "channels": self.channels, "time_unit": time_unit,
            "scientific_identity": scientific_identity,
            "sampling": "every_accepted_increment_no_unequilibrated_initial_sample",
        }))
        self.frames = []

    def validate_reference(self):
        sampling = self._sampling
        if (not np.array_equal(sampling.function.function_space.mesh.geometry.x, self._reference_geometry)
                or not np.array_equal(sampling.points, self._reference_points)
                or not np.array_equal(sampling.weights, self._reference_weights)):
            raise ValueError("Material point histories require unchanged reference geometry and quadrature.")

    def prepare(self, time, values):
        if not np.isfinite(time) or time <= 0 or (self.frames and time <= self.frames[-1]["time"]):
            raise ValueError("Material point history requires increasing positive accepted physical time.")
        if set(values) != set(self.channels):
            raise ValueError("Material point history channels changed during the Step.")
        selected = {}
        for name, data in values.items():
            array = np.asarray(data)
            shape = tuple(self.channels[name]["shape"])
            if np.iscomplexobj(array) or array.shape[1:] != shape:
                raise ValueError(f"Material point channel {name} has invalid shape or complex values.")
            array = array[self.indices]
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Material point channel {name} must be finite.")
            selected[name] = array.tolist()
        return {"time": float(time), "values": selected}

    def payload(self):
        body = {"identity": self.identity, "frames": self.frames}
        return {**json.loads(_canonical(body)), "sha256": sha256(_canonical(body).encode()).hexdigest()}

    def restore(self, payload, *, accepted_times):
        if not isinstance(payload, dict) or payload.get("identity") != self.identity:
            raise ValueError("Material point checkpoint identity differs or history is missing.")
        body = {"identity": payload["identity"], "frames": payload.get("frames")}
        if payload.get("sha256") != sha256(_canonical(body).encode()).hexdigest():
            raise ValueError("Material point checkpoint history integrity mismatch.")
        frames = body["frames"]
        if not isinstance(frames, list) or [frame["time"] for frame in frames] != list(accepted_times):
            raise ValueError("Material point history must cover exactly the accepted increment ledger.")
        previous = 0.
        for frame in frames:
            if not np.isfinite(frame["time"]) or frame["time"] <= previous:
                raise ValueError("Material point checkpoint time is invalid.")
            previous = frame["time"]
            if set(frame["values"]) != set(self.channels):
                raise ValueError("Material point checkpoint channels differ.")
            for name, channel in self.channels.items():
                array = np.asarray(frame["values"][name])
                if (array.shape != (len(self.indices), *channel["shape"])
                        or np.iscomplexobj(array) or not np.all(np.isfinite(array))):
                    raise ValueError(f"Material point checkpoint channel {name} has invalid values.")
        self.frames = json.loads(_canonical(frames))

    def histories(self):
        from .core import HistoryResult

        time = np.asarray([frame["time"] for frame in self.frames])
        return tuple(
            HistoryResult(
                f"{point['name']}.{name}", time,
                np.asarray([frame["values"][name][i] for frame in self.frames]).reshape((len(time), *channel["shape"])),
                unit=channel["unit"], abscissa_unit=self.time_unit,
                description=f"{channel['measure']}; original cell {point['cell']}, quadrature point {point['point']}.",
            )
            for i, point in enumerate(self.identity["points"])
            for name, channel in self.channels.items()
        )
