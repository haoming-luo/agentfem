"""Time-coordinate histories for scientific fields and scalar inputs.

The history object is deliberately independent of a solver.  A transient
step may record into it, while a later step may sample the same history at
its own accepted or attempted physical times.  This makes the transfer from
thermal analysis to mechanics explicit and inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def _unwrap(value):
    return getattr(value, "value", getattr(value, "function", value))


@dataclass
class FieldHistory:
    """A sampled scalar or finite-element field over physical time.

    ``interpolation`` may be ``"linear"`` or ``"previous"``.  Values outside
    the recorded interval raise by default; ``outside="clamp"`` must be
    selected explicitly when endpoint holding is physically intended.
    """

    source: object
    name: str = "field_history"
    unit: str | None = None
    coordinate_name: str = "time"
    interpolation: str = "linear"
    outside: str = "error"
    every: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)
    _times: list[float] = field(default_factory=list, init=False, repr=False)
    _snapshots: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _active_time: float | None = field(default=None, init=False, repr=False)
    _identity_cache: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )
    _portable_identity_cache: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.interpolation = str(self.interpolation).lower().replace("-", "_")
        self.outside = str(self.outside).lower().replace("-", "_")
        if self.interpolation not in {"linear", "previous"}:
            raise ValueError("FieldHistory interpolation must be linear or previous.")
        if self.outside not in {"error", "clamp"}:
            raise ValueError("FieldHistory outside must be error or clamp.")
        if int(self.every) <= 0:
            raise ValueError("FieldHistory every must be a positive integer.")
        self.every = int(self.every)
        try:
            normalized_metadata = json.loads(
                json.dumps(dict(self.metadata), sort_keys=True)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("FieldHistory metadata must be JSON serializable.") from exc
        self.metadata = normalized_metadata
        selected = _unwrap(self.source)
        if hasattr(selected, "function_space"):
            shape = getattr(selected, "ufl_shape", ())
            if shape not in {(), None}:
                raise ValueError(
                    "FieldHistory currently records scalar finite-element fields."
                )
        else:
            values = np.asarray(selected, dtype=float)
            if values.size != 1:
                raise ValueError(
                    "FieldHistory source must be a scalar or scalar finite-element field."
                )

    @property
    def value(self):
        """Return the live source after the most recent :meth:`apply`."""

        return _unwrap(self.source)

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(self._times)

    @property
    def frame_count(self) -> int:
        return len(self._times)

    @property
    def active_time(self) -> float | None:
        return self._active_time

    def record(self, time: float, source=None) -> "FieldHistory":
        """Record a copy at ``time``; an equal last time replaces that frame."""

        selected_time = float(time)
        if not np.isfinite(selected_time):
            raise ValueError("FieldHistory time must be finite.")
        selected = _unwrap(self.source if source is None else source)
        if hasattr(selected, "function_space"):
            values = np.asarray(selected.x.array, dtype=float).copy()
        else:
            values = np.asarray(selected, dtype=float).reshape(-1).copy()
        if self._times and selected_time < self._times[-1] - 1.0e-14:
            raise ValueError("FieldHistory records must have increasing physical time.")
        if self._times and np.isclose(selected_time, self._times[-1], rtol=0.0, atol=1.0e-14):
            self._times[-1] = selected_time
            self._snapshots[-1] = values
        else:
            self._times.append(selected_time)
            self._snapshots.append(values)
        self._active_time = selected_time
        self._identity_cache = None
        self._portable_identity_cache = None
        return self

    def sample(self, time: float) -> np.ndarray:
        """Return an interpolated numerical copy without changing the source."""

        if not self._times:
            raise RuntimeError(f"FieldHistory {self.name!r} has no recorded frames.")
        selected_time = float(time)
        if not np.isfinite(selected_time):
            raise ValueError("FieldHistory sample time must be finite.")
        lower = self._times[0]
        upper = self._times[-1]
        tolerance = 1.0e-12 * max(1.0, abs(lower), abs(upper))
        if selected_time < lower - tolerance or selected_time > upper + tolerance:
            if self.outside == "error":
                raise ValueError(
                    f"FieldHistory {self.name!r} covers [{lower:g}, {upper:g}] "
                    f"{self.coordinate_name}; requested {selected_time:g}."
                )
            selected_time = min(max(selected_time, lower), upper)
        if selected_time <= lower + tolerance:
            return self._snapshots[0].copy()
        if selected_time >= upper - tolerance:
            return self._snapshots[-1].copy()
        right = int(np.searchsorted(self._times, selected_time, side="right"))
        left = right - 1
        if self.interpolation == "previous":
            return self._snapshots[left].copy()
        span = self._times[right] - self._times[left]
        weight = (selected_time - self._times[left]) / span
        return (
            (1.0 - weight) * self._snapshots[left]
            + weight * self._snapshots[right]
        )

    def apply(self, time: float):
        """Sample ``time`` and assign it to the live scalar or FE source."""

        values = self.sample(time)
        selected = _unwrap(self.source)
        if hasattr(selected, "function_space"):
            if values.size != selected.x.array.size:
                raise ValueError("FieldHistory frame and target field layouts differ.")
            selected.x.array[:] = values
            selected.x.scatter_forward()
        else:
            if values.size != 1:
                raise ValueError("Scalar FieldHistory frame must contain one value.")
            self.source = float(values[0])
        self._active_time = float(time)
        return self.value

    def scientific_identity(self) -> dict[str, object]:
        """Return a deterministic identity for the complete recorded history."""

        if self._identity_cache is not None:
            return dict(self._identity_cache)
        digest = sha256()
        digest.update(np.asarray(self._times, dtype=np.float64).tobytes())
        for snapshot in self._snapshots:
            digest.update(np.ascontiguousarray(snapshot, dtype=np.float64).tobytes())
        identity = {
            "kind": "field_history",
            "name": self.name,
            "coordinate_name": self.coordinate_name,
            "unit": self.unit,
            "interpolation": self.interpolation,
            "outside": self.outside,
            "metadata": dict(self.metadata),
            "frame_count": self.frame_count,
            "times": list(self._times),
            "values_sha256": digest.hexdigest(),
            "layout": (
                "current_partition"
                if hasattr(self.value, "function_space")
                else "scalar"
            ),
        }
        self._identity_cache = identity
        return dict(identity)

    def portable_identity(self) -> dict[str, object]:
        """Return a partition-independent identity for a nodal history.

        Distributed construction is collective.  It is therefore explicit
        rather than hidden inside :meth:`summary`, which may legitimately be
        called only on rank zero by reporting code.
        """

        if not hasattr(self.value, "function_space"):
            return self.scientific_identity()
        if self._portable_identity_cache is None:
            _, identity = self._portable_archive()
            self._portable_identity_cache = identity
        return dict(self._portable_identity_cache)

    def summary(self) -> dict[str, object]:
        identity = self.scientific_identity()
        identity["active_time"] = self.active_time
        return identity

    def save(self, path) -> Path:
        """Save a compact scalar or partition-portable nodal history."""

        if not self._times:
            raise RuntimeError("Cannot save an empty FieldHistory.")
        selected_source = self.value
        selected = Path(path)
        if selected.suffix != ".npz":
            selected = selected.with_suffix(".npz")
        selected.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(selected_source, "function_space"):
            payload, identity = self._portable_archive()
            comm = selected_source.function_space.mesh.comm
            error = None
            if comm.rank == 0:
                try:
                    from .checkpointing import atomic_savez

                    atomic_savez(
                        selected,
                        schema="agentfem.field-history.v2",
                        metadata=json.dumps(
                            {
                                **identity,
                                "active_time": self.active_time,
                                "every": self.every,
                            },
                            sort_keys=True,
                        ),
                        times=np.asarray(self._times, dtype=float),
                        coordinates=payload["coordinates"],
                        snapshots=payload["snapshots"],
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error = comm.bcast(error, root=0)
            if error is not None:
                raise RuntimeError(f"FieldHistory portable write failed: {error}")
            comm.barrier()
            self._portable_identity_cache = identity
            return selected
        metadata = {
            **self.scientific_identity(),
            "active_time": self.active_time,
            "every": self.every,
        }
        np.savez_compressed(
            selected,
            schema="agentfem.field-history.v1",
            metadata=json.dumps(metadata, sort_keys=True),
            times=np.asarray(self._times, dtype=float),
            snapshots=np.stack(self._snapshots),
        )
        return selected

    @classmethod
    def load(cls, path, *, source) -> "FieldHistory":
        """Load a history and bind it to a compatible live scalar or FE source."""

        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"])
            if schema not in {
                "agentfem.field-history.v1",
                "agentfem.field-history.v2",
            }:
                raise ValueError("Unsupported AgentFEM field-history schema.")
            metadata = json.loads(str(data["metadata"]))
            result = cls(
                source,
                name=metadata["name"],
                unit=metadata.get("unit"),
                coordinate_name=metadata["coordinate_name"],
                interpolation=metadata["interpolation"],
                outside=metadata["outside"],
                every=int(metadata.get("every", 1)),
                metadata=metadata.get("metadata", {}),
            )
            times = np.asarray(data["times"], dtype=float)
            snapshots = np.asarray(data["snapshots"], dtype=float)
            coordinates = (
                None
                if schema == "agentfem.field-history.v1"
                else np.asarray(data["coordinates"], dtype=np.int64)
            )
        _validate_loaded_series(times, snapshots, metadata)
        if schema == "agentfem.field-history.v2":
            result._restore_portable_frames(
                times=times,
                coordinates=coordinates,
                snapshots=snapshots,
                metadata=metadata,
            )
            return result
        for selected_time, values in zip(times, snapshots, strict=True):
            result._times.append(float(selected_time))
            result._snapshots.append(np.asarray(values, dtype=float).copy())
        if result.scientific_identity()["values_sha256"] != metadata["values_sha256"]:
            raise ValueError("FieldHistory content hash does not match its metadata.")
        active_time = metadata.get("active_time")
        if active_time is not None:
            result.apply(float(active_time))
        return result

    def _portable_archive(self):
        """Collect a coordinate-keyed nodal history on rank zero.

        This first portable archive deliberately reuses the same physical-dof
        identity as transient checkpoints.  It is compact and independent of
        MPI partitioning, while remaining a root-gathered format rather than
        an extreme-scale parallel archive.
        """

        source = self.value
        if not hasattr(source, "function_space"):
            raise TypeError("A portable field archive requires a finite-element field.")
        from .checkpointing import (
            _coordinate_order,
            _has_duplicate_coordinates,
            _portable_local_field,
            function_portable_identity,
        )

        local = _portable_local_field(source)
        block_size = int(source.function_space.dofmap.index_map_bs)
        owned = int(source.function_space.dofmap.index_map.size_local)
        local_frames = np.stack(
            [
                np.asarray(frame[: owned * block_size], dtype=np.float64).reshape(
                    (owned, block_size)
                )
                for frame in self._snapshots
            ],
            axis=0,
        )
        comm = source.function_space.mesh.comm
        gathered = comm.gather(
            {
                "coordinates": local["coordinates"],
                "snapshots": local_frames,
                "key_mode": local["key_mode"],
            },
            root=0,
        )
        payload = None
        error = None
        digest = None
        key_mode = local["key_mode"]
        if comm.rank == 0:
            try:
                modes = {item["key_mode"] for item in gathered}
                if len(modes) != 1:
                    raise ValueError(
                        "FieldHistory has inconsistent portable key modes."
                    )
                key_mode = modes.pop()
                coordinates = np.concatenate(
                    [item["coordinates"] for item in gathered], axis=0
                )
                snapshots = np.concatenate(
                    [item["snapshots"] for item in gathered], axis=1
                )
                order = _coordinate_order(coordinates)
                coordinates = np.ascontiguousarray(coordinates[order])
                snapshots = np.ascontiguousarray(snapshots[:, order, :])
                if _has_duplicate_coordinates(coordinates):
                    raise ValueError(
                        "FieldHistory portable archive has duplicate owned dof keys."
                    )
                digest = _history_digest(self._times, snapshots, coordinates)
                payload = {
                    "coordinates": coordinates,
                    "snapshots": snapshots,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        response = comm.bcast(
            {"digest": digest, "key_mode": key_mode, "error": error}, root=0
        )
        if response["error"] is not None:
            raise RuntimeError(
                "FieldHistory portable collection failed: " + response["error"]
            )
        identity = {
            "kind": "field_history",
            "name": self.name,
            "coordinate_name": self.coordinate_name,
            "unit": self.unit,
            "interpolation": self.interpolation,
            "outside": self.outside,
            "metadata": dict(self.metadata),
            "frame_count": self.frame_count,
            "times": list(self._times),
            "values_sha256": response["digest"],
            "layout": "portable_nodal",
            "key": response["key_mode"],
            "field": function_portable_identity(source),
        }
        return payload, identity

    def _restore_portable_frames(
        self,
        *,
        times: np.ndarray,
        coordinates: np.ndarray,
        snapshots: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        source = self.value
        if not hasattr(source, "function_space"):
            raise TypeError(
                "A portable nodal FieldHistory must be loaded into a finite-element field."
            )
        from .checkpointing import _portable_local_field, function_portable_identity

        if metadata.get("field") != function_portable_identity(source):
            raise ValueError("FieldHistory mesh/function identity differs.")
        if snapshots.ndim != 3 or snapshots.shape[0] != len(times):
            raise ValueError("FieldHistory portable snapshots have an invalid shape.")
        if snapshots.shape[1] != len(coordinates):
            raise ValueError("FieldHistory coordinate and snapshot sizes differ.")
        if (
            _history_digest(times, snapshots, coordinates)
            != metadata["values_sha256"]
        ):
            raise ValueError("FieldHistory content hash does not match its metadata.")
        local = _portable_local_field(source)
        if metadata.get("key") != local["key_mode"]:
            raise ValueError("FieldHistory portable key mode differs.")
        lookup = {row.tobytes(): index for index, row in enumerate(coordinates)}
        indices = []
        for row in local["coordinates"]:
            key = row.tobytes()
            if key not in lookup:
                raise ValueError("FieldHistory lacks a local physical dof coordinate.")
            indices.append(lookup[key])
        indices = np.asarray(indices, dtype=np.int64)
        V = source.function_space
        owned = int(V.dofmap.index_map.size_local)
        block_size = int(V.dofmap.index_map_bs)
        original = source.x.array.copy()
        try:
            for selected_time, frame in zip(times, snapshots, strict=True):
                source.x.array[: owned * block_size] = frame[indices].reshape(-1)
                source.x.scatter_forward()
                self._times.append(float(selected_time))
                self._snapshots.append(source.x.array.copy())
        finally:
            source.x.array[:] = original
            source.x.scatter_forward()
        active_time = metadata.get("active_time")
        if active_time is not None:
            self.apply(float(active_time))
        self._portable_identity_cache = {
            key: value
            for key, value in metadata.items()
            if key not in {"active_time", "every"}
        }


def _history_digest(times, snapshots, coordinates=None) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
    if coordinates is not None:
        digest.update(np.ascontiguousarray(coordinates, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(snapshots, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _validate_loaded_series(
    times: np.ndarray,
    snapshots: np.ndarray,
    metadata: dict[str, object],
) -> None:
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("FieldHistory times must be a non-empty finite vector.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("FieldHistory times must be strictly increasing.")
    if snapshots.ndim < 2 or snapshots.shape[0] != len(times):
        raise ValueError("FieldHistory snapshots do not match recorded times.")
    if int(metadata.get("frame_count", -1)) != len(times):
        raise ValueError("FieldHistory frame count does not match its metadata.")


def field_history(source, **kwargs) -> FieldHistory:
    """Create a generic field-history recorder."""

    return FieldHistory(source, **kwargs)


def temperature(source, *, name: str = "temperature", unit: str = "K", **kwargs) -> FieldHistory:
    """Create a physical-time temperature history."""

    return FieldHistory(source, name=name, unit=unit, **kwargs)
