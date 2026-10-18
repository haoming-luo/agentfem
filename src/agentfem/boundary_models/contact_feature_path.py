# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted-path evidence for bounded polyhedral rigid contact.

This is a State observer, not a contact law or a whole-face certificate. It
retains only scalar sampled hazards, so output-frame thinning cannot erase an
earlier accepted feature event.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


_COUNT_KEYS = (
    "accepted_station_count",
    "maximum_sampled_mixed_feature_slave_faces",
    "maximum_active_edge_points",
    "maximum_active_vertex_points",
)
_PAIR_KEYS = {
    "constraint", "accepted_station_count", "last_factor",
    "maximum_sampled_mixed_feature_slave_faces", "maximum_active_edge_points",
    "maximum_active_vertex_points", "all_observed_integrals_certified",
}


@dataclass(eq=False)
class AcceptedContactFeaturePath:
    """Observe each accepted configuration of named polyhedral contact pairs."""

    providers: tuple[object, ...]
    complete: bool = field(default=True, init=False)
    accepted_station_count: int = field(default=0, init=False)
    pairs: dict[str, dict[str, object]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        from .closed_polyhedron import ClosedTriangulatedRigidSolid
        from .convex_polyhedron import ConvexPolyhedralRigidSurface

        selected = tuple(self.providers)
        names = tuple(str(provider.name) for provider in selected)
        if (
            not selected or len(names) != len(set(names))
            or any(not name for name in names)
            or any(not isinstance(
                provider.surface,
                (ClosedTriangulatedRigidSolid, ConvexPolyhedralRigidSurface),
            ) for provider in selected)
        ):
            raise ValueError("Accepted contact feature path requires unique polyhedral pairs.")
        self.providers = selected

    def _observations(self) -> dict[str, dict[str, object]]:
        observed = {}
        for provider in self.providers:
            sample = provider.current_point_diagnostics()
            mixed = sample.get("sampled_mixed_feature_slave_faces")
            certified = sample.get("feature_transition_integral_certified")
            counts = sample.get("active_feature_counts")
            if counts is None:
                counts = {}
            elif (
                not isinstance(counts, Mapping)
                or set(counts) != {"face", "edge", "vertex"}
                or type(sample.get("active_point_count")) is not int
                or any(type(value) is not int or value < 0 for value in counts.values())
                or sum(counts.values()) != sample["active_point_count"]
            ):
                raise ValueError("Polyhedral active-feature counts differ from contact points.")
            if (
                type(mixed) is not int or mixed < 0
                or type(certified) is not bool
                or any(
                    type(counts.get(key, 0)) is not int or counts.get(key, 0) < 0
                    for key in ("edge", "vertex")
                )
            ):
                raise ValueError("Polyhedral feature diagnostics are incomplete or invalid.")
            observed[str(provider.name)] = {
                "mixed": mixed,
                "edge": int(counts.get("edge", 0)),
                "vertex": int(counts.get("vertex", 0)),
                "certified": certified,
            }
        return observed

    def _capture(self, snapshot, *, reset: bool) -> None:
        factor = float(snapshot.load_factor)
        if not np.isfinite(factor) or (
            not reset and self.pairs and factor <= next(iter(self.pairs.values()))["last_factor"] + 1e-12
        ):
            raise ValueError("Accepted contact feature stations must increase strictly.")
        observations = self._observations()
        next_pairs = {}
        for name, sample in observations.items():
            previous = None if reset else self.pairs.get(name)
            next_pairs[name] = {
                "constraint": name,
                "accepted_station_count": (
                    1 if previous is None else previous["accepted_station_count"] + 1
                ),
                "last_factor": factor,
                "maximum_sampled_mixed_feature_slave_faces": max(
                    sample["mixed"], 0 if previous is None else
                    previous["maximum_sampled_mixed_feature_slave_faces"],
                ),
                "maximum_active_edge_points": max(
                    sample["edge"], 0 if previous is None else
                    previous["maximum_active_edge_points"],
                ),
                "maximum_active_vertex_points": max(
                    sample["vertex"], 0 if previous is None else
                    previous["maximum_active_vertex_points"],
                ),
                "all_observed_integrals_certified": (
                    sample["certified"] if previous is None else
                    previous["all_observed_integrals_certified"] and sample["certified"]
                ),
            }
        self.pairs = next_pairs
        self.accepted_station_count = int(snapshot.index) + 1

    def reset(self, snapshot) -> None:
        self.complete = True
        self._capture(snapshot, reset=True)

    def prepare_resume(self, snapshot) -> None:
        if (
            self.accepted_station_count != int(snapshot.index) + 1
            or not self.pairs
            or any(abs(pair["last_factor"] - float(snapshot.load_factor)) > 1e-12
                   for pair in self.pairs.values())
        ):
            raise ValueError("Accepted contact feature path differs from resume station.")

    def accept(self, snapshot) -> None:
        if self.accepted_station_count != int(snapshot.index):
            raise ValueError("Accepted contact feature path skipped a station.")
        self._capture(snapshot, reset=False)

    def snapshot_runtime_state(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "accepted_station_count": self.accepted_station_count,
            "pairs": {name: dict(item) for name, item in self.pairs.items()},
        }

    def restore_runtime_state(self, state) -> None:
        self.complete = bool(state["complete"])
        self.accepted_station_count = int(state["accepted_station_count"])
        self.pairs = {name: dict(item) for name, item in state["pairs"].items()}

    def evidence(self) -> dict[str, object]:
        return {
            "schema": "agentfem.contact-feature-path.v1",
            "complete": self.complete,
            "accepted_station_count": self.accepted_station_count,
            "pairs": tuple(dict(self.pairs[name]) for name in sorted(self.pairs)),
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "accepted_polyhedral_contact_feature_path",
            "pair_names": tuple(sorted(str(provider.name) for provider in self.providers)),
            "scope": "sampled_accepted_stations_not_whole_face_certificate",
            "complete": self.complete,
            "accepted_station_count": self.accepted_station_count,
        }

    def checkpoint_state(self) -> dict[str, object]:
        evidence = self.evidence()
        return evidence | {"pairs": list(evidence["pairs"])}

    def restore_checkpoint_state(self, record, *, current_snapshot) -> None:
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "complete", "accepted_station_count", "pairs"}
            or record["schema"] != "agentfem.contact-feature-path.v1"
            or type(record["complete"]) is not bool
            or type(record["accepted_station_count"]) is not int
            or record["accepted_station_count"] != int(current_snapshot.index) + 1
            or not isinstance(record["pairs"], list)
        ):
            raise ValueError("Accepted contact feature checkpoint schema differs.")
        selected = {}
        for item in record["pairs"]:
            if not isinstance(item, dict) or set(item) != _PAIR_KEYS:
                raise ValueError("Accepted contact feature checkpoint pair schema differs.")
            name = item["constraint"]
            if (
                not isinstance(name, str) or name in selected
                or any(type(item[key]) is not int or item[key] < (
                    1 if key == "accepted_station_count" else 0
                ) for key in _COUNT_KEYS)
                or item["accepted_station_count"] > record["accepted_station_count"]
                or type(item["all_observed_integrals_certified"]) is not bool
                or type(item["last_factor"]) not in (int, float)
                or not np.isfinite(item["last_factor"])
                or abs(item["last_factor"] - float(current_snapshot.load_factor)) > 1e-12
                or (record["complete"] and item["accepted_station_count"] !=
                    record["accepted_station_count"])
            ):
                raise ValueError("Accepted contact feature checkpoint pair differs.")
            selected[name] = dict(item)
        if set(selected) != {str(provider.name) for provider in self.providers}:
            raise ValueError("Accepted contact feature checkpoint pair coverage differs.")
        self.complete = record["complete"]
        self.accepted_station_count = record["accepted_station_count"]
        self.pairs = selected

    def restore_missing_checkpoint_state(self, *, current_snapshot) -> None:
        """Load an older checkpoint without fabricating unobserved history."""

        self.complete = False
        self._capture(current_snapshot, reset=True)
