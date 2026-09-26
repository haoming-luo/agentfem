# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained evidence packages for dynamic-fracture research."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
from pathlib import Path
import re
from shutil import copy2
from typing import Mapping

import numpy as np

from .provenance import seal_manifest, verify_manifest


_SCHEMA = "agentfem.dynamic-fracture-evidence.v1"


@dataclass(frozen=True)
class CohesiveInterfaceTrace:
    """Portable accepted-frame record on one fixed cohesive interface."""

    time: np.ndarray
    path_coordinate: np.ndarray
    opening: np.ndarray
    traction: np.ndarray
    damage: np.ndarray
    dissipated_energy_density: np.ndarray
    tangential_jump: np.ndarray | None = None
    tangential_traction: np.ndarray | None = None
    mode_mixity: np.ndarray | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        coordinate = np.asarray(self.path_coordinate, dtype=float)
        if time.ndim != 1 or time.size < 1 or np.any(~np.isfinite(time)):
            raise ValueError("Cohesive trace time must be a finite 1D array.")
        if time.size > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("Cohesive trace time must be strictly increasing.")
        if (
            coordinate.ndim != 1
            or coordinate.size < 1
            or np.any(~np.isfinite(coordinate))
        ):
            raise ValueError("Cohesive trace coordinates must be a finite 1D array.")
        shape = (time.size, coordinate.size)
        arrays = {}
        for name in ("opening", "traction", "damage", "dissipated_energy_density"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape or np.any(~np.isfinite(values)):
                raise ValueError(
                    f"Cohesive trace {name} must have shape {shape} and be finite."
                )
            arrays[name] = values.copy()
        vector_arrays = {}
        for name in ("tangential_jump", "tangential_traction"):
            declared = getattr(self, name)
            if declared is None:
                continue
            values = np.asarray(declared, dtype=float)
            if (
                values.ndim != 3
                or values.shape[:2] != shape
                or np.any(~np.isfinite(values))
            ):
                raise ValueError(
                    f"Cohesive trace {name} must have shape "
                    f"{shape}+(components,) and be finite."
                )
            vector_arrays[name] = values.copy()
        scalar_optional = None
        if self.mode_mixity is not None:
            scalar_optional = np.asarray(self.mode_mixity, dtype=float)
            if scalar_optional.shape != shape or np.any(~np.isfinite(scalar_optional)):
                raise ValueError(
                    f"Cohesive trace mode_mixity must have shape {shape} and be finite."
                )
        object.__setattr__(self, "time", time.copy())
        object.__setattr__(self, "path_coordinate", coordinate.copy())
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        for name in ("tangential_jump", "tangential_traction"):
            object.__setattr__(self, name, vector_arrays.get(name))
        object.__setattr__(
            self,
            "mode_mixity",
            None if scalar_optional is None else scalar_optional.copy(),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def write(self, path: str | Path) -> Path:
        """Write a compact, dependency-free NPZ research artifact."""

        location = Path(path)
        if location.suffix.lower() != ".npz":
            location = location.with_suffix(".npz")
        location.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": np.asarray("agentfem.cohesive-interface-trace.v2"),
            "time": self.time,
            "path_coordinate": self.path_coordinate,
            "opening": self.opening,
            "traction": self.traction,
            "damage": self.damage,
            "dissipated_energy_density": self.dissipated_energy_density,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for name in ("tangential_jump", "tangential_traction", "mode_mixity"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        np.savez_compressed(location, **payload)
        return location

    @classmethod
    def read(cls, path: str | Path) -> "CohesiveInterfaceTrace":
        location = Path(path)
        with np.load(location, allow_pickle=False) as archive:
            schema = str(archive["schema"])
            if schema not in {
                "agentfem.cohesive-interface-trace.v1",
                "agentfem.cohesive-interface-trace.v2",
            }:
                raise ValueError(f"Unsupported cohesive trace schema {schema!r}.")
            return cls(
                time=archive["time"],
                path_coordinate=archive["path_coordinate"],
                opening=archive["opening"],
                traction=archive["traction"],
                damage=archive["damage"],
                dissipated_energy_density=archive["dissipated_energy_density"],
                tangential_jump=(
                    archive["tangential_jump"]
                    if "tangential_jump" in archive.files
                    else None
                ),
                tangential_traction=(
                    archive["tangential_traction"]
                    if "tangential_traction" in archive.files
                    else None
                ),
                mode_mixity=(
                    archive["mode_mixity"] if "mode_mixity" in archive.files else None
                ),
                metadata=json.loads(str(archive["metadata_json"])),
            )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "cohesive_interface_trace",
            "frames": int(self.time.size),
            "interface_points": int(self.path_coordinate.size),
            "time_interval": [float(self.time[0]), float(self.time[-1])],
            "maximum_opening": float(np.max(self.opening)),
            "maximum_damage": float(np.max(self.damage)),
            "vector_kinematics": self.tangential_jump is not None,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ScientificComparison:
    """Common scalar evidence for a simulation-to-observation comparison."""

    kind: str
    samples: int
    root_mean_square_error: float
    normalized_root_mean_square_error: float
    correlation: float | None
    metadata: dict[str, object] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "samples": self.samples,
            "root_mean_square_error": self.root_mean_square_error,
            "normalized_root_mean_square_error": self.normalized_root_mean_square_error,
            "correlation": self.correlation,
            "metadata": dict(self.metadata or {}),
        }


def _producer_version() -> str:
    try:
        return version("agentfem")
    except PackageNotFoundError:
        return "source"


def _artifact_name(name: str, source: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    if not stem:
        raise ValueError("Evidence artifact names must contain a portable character.")
    suffixes = "".join(source.suffixes)
    return stem if stem.endswith(suffixes) else stem + suffixes


@dataclass(frozen=True)
class DynamicFractureEvidenceBundle:
    """Trace, fields, energies, comparisons, and provenance for one condition.

    The bundle is an integrity-checked exchange artifact.  It does not promote
    a completed run to scientific validation; comparison and verification
    conclusions remain explicit records supplied by the research workflow.
    """

    benchmark_id: str
    trace: object
    wave_speeds: Mapping[str, float]
    energy_history: Mapping[str, np.ndarray]
    comparisons: tuple[object, ...] = ()
    artifacts: Mapping[str, str | Path] | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        identifier = str(self.benchmark_id).strip()
        if not identifier:
            raise ValueError("DynamicFractureEvidenceBundle.benchmark_id is required.")
        if not hasattr(self.trace, "write") or not hasattr(self.trace, "summary"):
            raise TypeError("DynamicFractureEvidenceBundle.trace needs write() and summary().")
        speeds = {str(name): float(value) for name, value in self.wave_speeds.items()}
        if not speeds or any(not isfinite(value) or value <= 0.0 for value in speeds.values()):
            raise ValueError("Dynamic-fracture wave speeds must be finite and positive.")
        histories = {
            str(name): np.asarray(values, dtype=float).reshape(-1).copy()
            for name, values in self.energy_history.items()
        }
        sizes = {values.size for values in histories.values()}
        if not histories or len(sizes) != 1 or next(iter(sizes)) < 1:
            raise ValueError("Energy-history channels must be nonempty and equally sized.")
        if any(np.any(~np.isfinite(values)) for values in histories.values()):
            raise ValueError("Energy-history channels must be finite.")
        if "time" not in histories:
            raise ValueError("Energy history requires a 'time' channel.")
        if histories["time"].size > 1 and np.any(np.diff(histories["time"]) <= 0.0):
            raise ValueError("Energy-history time must be strictly increasing.")
        comparisons = tuple(self.comparisons)
        if any(not hasattr(item, "summary") for item in comparisons):
            raise TypeError("Every dynamic-fracture comparison needs summary().")
        artifacts = {str(name): Path(path) for name, path in (self.artifacts or {}).items()}
        object.__setattr__(self, "benchmark_id", identifier)
        object.__setattr__(self, "wave_speeds", speeds)
        object.__setattr__(self, "energy_history", histories)
        object.__setattr__(self, "comparisons", comparisons)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def write(self, directory: str | Path) -> Path:
        """Write one self-contained directory and sealed manifest."""

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        trace_path = self.trace.write(root / "cohesive_interface_trace.npz")
        energy_path = root / "energy_history.npz"
        np.savez_compressed(energy_path, **self.energy_history)
        registered: dict[str, str] = {
            "cohesive_interface_trace": trace_path.name,
            "energy_history": energy_path.name,
        }
        copied: dict[str, str] = {}
        artifact_directory = root / "artifacts"
        destinations: set[Path] = set()
        for name, source in sorted((self.artifacts or {}).items()):
            selected = Path(source).expanduser()
            if not selected.is_file():
                raise FileNotFoundError(
                    f"Dynamic-fracture evidence artifact {name!r} is missing: {selected}"
                )
            artifact_directory.mkdir(parents=True, exist_ok=True)
            destination = artifact_directory / _artifact_name(name, selected)
            if destination in destinations:
                raise ValueError(
                    "Dynamic-fracture evidence artifact names collide after "
                    f"portable normalization: {destination.name!r}."
                )
            destinations.add(destination)
            if selected.resolve() != destination.resolve():
                copy2(selected, destination)
            relative = str(destination.relative_to(root))
            registered[f"research:{name}"] = relative
            copied[name] = relative
        record: dict[str, object] = {
            "schema": _SCHEMA,
            "producer": {"name": "AgentFEM", "version": _producer_version()},
            "benchmark_id": self.benchmark_id,
            "scientific_status": "evidence_package_not_automatic_validation",
            "trace": self.trace.summary(),
            "wave_speeds": dict(self.wave_speeds),
            "energy_channels": list(self.energy_history),
            "energy_frames": int(self.energy_history["time"].size),
            "comparisons": [item.summary() for item in self.comparisons],
            "research_artifacts": copied,
            "artifacts": registered,
            "metadata": dict(self.metadata or {}),
        }
        record["provenance_seal"] = seal_manifest(
            record,
            base=root,
            producer_version=_producer_version(),
        )
        manifest = root / "fracture.evidence.json"
        manifest.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def read(cls, manifest: str | Path) -> "DynamicFractureEvidenceBundle":
        """Verify every byte before reconstructing a research package."""

        selected = Path(manifest).expanduser().resolve()
        verification = verify_manifest(selected)
        if not verification.verified:
            raise ValueError(
                "Dynamic-fracture evidence integrity failed: " + verification.format()
            )
        record = json.loads(selected.read_text(encoding="utf-8"))
        if record.get("schema") != _SCHEMA:
            raise ValueError("Unsupported dynamic-fracture evidence schema.")
        root = selected.parent
        trace = CohesiveInterfaceTrace.read(
            root / record["artifacts"]["cohesive_interface_trace"]
        )
        with np.load(
            root / record["artifacts"]["energy_history"],
            allow_pickle=False,
        ) as archive:
            energy = {name: archive[name] for name in archive.files}
        comparisons = tuple(
            ScientificComparison(
                kind=item["kind"],
                samples=int(item["samples"]),
                root_mean_square_error=float(item["root_mean_square_error"]),
                normalized_root_mean_square_error=float(
                    item["normalized_root_mean_square_error"]
                ),
                correlation=(
                    None if item.get("correlation") is None else float(item["correlation"])
                ),
                metadata=item.get("metadata", {}),
            )
            for item in record.get("comparisons", ())
        )
        artifacts = {
            name: root / path
            for name, path in record.get("research_artifacts", {}).items()
        }
        return cls(
            benchmark_id=record["benchmark_id"],
            trace=trace,
            wave_speeds=record["wave_speeds"],
            energy_history=energy,
            comparisons=comparisons,
            artifacts=artifacts,
            metadata=record.get("metadata", {}),
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dynamic_fracture_evidence_bundle",
            "benchmark_id": self.benchmark_id,
            "trace": self.trace.summary(),
            "wave_speeds": dict(self.wave_speeds),
            "energy_channels": tuple(self.energy_history),
            "energy_frames": int(self.energy_history["time"].size),
            "comparisons": [item.summary() for item in self.comparisons],
            "artifacts": tuple(sorted((self.artifacts or {}))),
            "scientific_status": "evidence_package_not_automatic_validation",
            "metadata": dict(self.metadata or {}),
        }


__all__ = [
    "CohesiveInterfaceTrace",
    "DynamicFractureEvidenceBundle",
    "ScientificComparison",
]
