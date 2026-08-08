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
        from .fracture import CohesiveInterfaceTrace, ScientificComparison

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


__all__ = ["DynamicFractureEvidenceBundle"]
